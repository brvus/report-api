"""
main.py  —  FastAPI server AMC Report  (three-step: analyze → recalc → report)
===============================================================================
POST /analyze   -> DSP automatico + grafici PNG base64 + metadati + t_max
POST /recalc    -> ricalcola DSP con taglio manuale start/stop sincrono
POST /report    -> genera .docx scaricabile
GET  /          -> pagina web
GET  /health    -> healthcheck Render
"""

import io
import json
import uuid
import base64
from typing import Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse

from report_engine import (
    run_pipeline, read_bin, parse_txt,
    detect_active_channels, detect_shared_zone,
    analyze_peaks, trim_initial_from_peaks,
    render_channel_png, generate_report,
    PEAK_FACTOR, REF_WIN_SEC, FS
)
import numpy as np

app = FastAPI(title="AMC Report API", version="3.0")

# Cache sessioni: session_id -> dict con dati grezzi + stato corrente
_sessions: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_dsp(lf, meta, peak_factor, ref_win_sec, t_start=None, t_stop=None):
    """Esegue la pipeline DSP e restituisce (active, zones, peak_result, t)."""
    N = lf.shape[0]
    t = np.linspace(1, N, N) / FS

    try:
        n_funi_exp = int(meta.get("nfuni", "0"))
    except (ValueError, TypeError):
        n_funi_exp = 0

    active = detect_active_channels(lf, n_funi_exp, n_ch=6)

    if t_start is not None and t_stop is not None and t_stop > t_start:
        # Taglio manuale sincrono su tutti i canali attivi
        s_v = int(np.searchsorted(t, t_start))
        e_v = min(int(np.searchsorted(t, t_stop)) - 1, N - 1)
        s_v = max(0, s_v)
        zones = [(s_v, e_v) if (ch < len(active) and active[ch])
                 else (0, N - 1) for ch in range(6)]
        pr = analyze_peaks(lf, active, zones,
                           peak_factor=peak_factor, ref_win_sec=ref_win_sec)
    else:
        # Rilevamento automatico
        zones = detect_shared_zone(lf, active, n_ch=6)
        pr    = analyze_peaks(lf, active, zones,
                              peak_factor=peak_factor, ref_win_sec=ref_win_sec)
        zones, trimmed = trim_initial_from_peaks(pr, active, zones)
        if trimmed:
            pr = analyze_peaks(lf, active, zones,
                               peak_factor=peak_factor, ref_win_sec=ref_win_sec)

    return active, zones, pr, t


def _build_charts(lf, t, active, zones, pr):
    """Genera lista chart con PNG base64."""
    per_ch = pr.get("per_canale", [])
    charts = []
    for ch in range(min(6, lf.shape[1])):
        s_v, e_v = zones[ch] if ch < len(zones) else (0, len(t) - 1)
        esito_ch = per_ch[ch].get("esito", "OK") if ch < len(per_ch) else "OK"
        is_active = bool(active[ch]) if ch < len(active) else False
        buf = render_channel_png(lf, t, ch, s_v, e_v, esito_ch,
                                 width_in=12, height_in=2.8, dpi=100)
        charts.append({
            "ch":       ch,
            "label":    f"Fune {ch+1}",
            "esito":    esito_ch,
            "n_picchi": per_ch[ch].get("n_picchi", 0) if ch < len(per_ch) else 0,
            "active":   is_active,
            "t_start":  float(t[s_v]),
            "t_stop":   float(t[min(e_v, len(t)-1)]),
            "png_b64":  base64.b64encode(buf.read()).decode(),
        })
    return charts


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Step 1 — Analisi automatica
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(
    bin_file:    UploadFile = File(...),
    txt_file:    UploadFile = File(...),
    peak_factor: float      = Form(default=PEAK_FACTOR),
    ref_win_sec: float      = Form(default=REF_WIN_SEC),
):
    bin_bytes = await bin_file.read()
    txt_bytes = await txt_file.read()
    if not bin_bytes:
        raise HTTPException(400, "File .bin vuoto.")
    if not txt_bytes:
        raise HTTPException(400, "File .txt vuoto.")

    try:
        lf   = read_bin(bin_bytes)
        meta = parse_txt(txt_bytes)
        active, zones, pr, t = _run_dsp(lf, meta, peak_factor, ref_win_sec)
        charts = _build_charts(lf, t, active, zones, pr)
    except Exception as e:
        raise HTTPException(500, str(e))

    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "bin_bytes":   bin_bytes,
        "txt_bytes":   txt_bytes,
        "lf":          lf,
        "meta":        meta,
        "t":           t,
        "peak_factor": peak_factor,
        "ref_win_sec": ref_win_sec,
        "active":      active,
        "zones":       zones,
        "pr":          pr,
    }
    if len(_sessions) > 20:
        del _sessions[next(iter(_sessions))]

    # t_start/t_stop della zona automatica (primo canale attivo)
    auto_start, auto_stop = 0.0, float(t[-1])
    for ch in range(6):
        if ch < len(active) and active[ch] and ch < len(zones):
            auto_start = float(t[zones[ch][0]])
            auto_stop  = float(t[min(zones[ch][1], len(t)-1)])
            break

    return {
        "session_id":    sid,
        "t_max":         float(t[-1]),
        "auto_start":    auto_start,
        "auto_stop":     auto_stop,
        "esito_globale": pr.get("esito_globale", "OK"),
        "meta": {
            "matricola": meta.get("matricola", "N.R."),
            "indirizzo": meta.get("indirizzo", "N.R."),
            "civico":    meta.get("civico", ""),
            "citta":     meta.get("citta", "N.R."),
            "provincia": meta.get("provincia", ""),
            "nfuni":     meta.get("nfuni", "N.R."),
        },
        "charts": charts,
    }


# ---------------------------------------------------------------------------
# Step 2 — Ricalcolo con taglio manuale
# ---------------------------------------------------------------------------

@app.post("/recalc")
async def recalc(
    session_id: str   = Form(...),
    t_start:    float = Form(...),
    t_stop:     float = Form(...),
):
    if session_id not in _sessions:
        raise HTTPException(400, "Sessione scaduta. Ricarica i file.")

    sess = _sessions[session_id]
    lf          = sess["lf"]
    meta        = sess["meta"]
    t           = sess["t"]
    peak_factor = sess["peak_factor"]
    ref_win_sec = sess["ref_win_sec"]

    try:
        active, zones, pr, _ = _run_dsp(lf, meta, peak_factor, ref_win_sec,
                                         t_start=t_start, t_stop=t_stop)
        charts = _build_charts(lf, t, active, zones, pr)
    except Exception as e:
        raise HTTPException(500, str(e))

    # Aggiorna sessione con nuove zone
    sess["active"] = active
    sess["zones"]  = zones
    sess["pr"]     = pr

    return {
        "esito_globale": pr.get("esito_globale", "OK"),
        "charts":        charts,
    }


# ---------------------------------------------------------------------------
# Step 3 — Genera .docx
# ---------------------------------------------------------------------------

@app.post("/report")
async def create_report(
    session_id:  str = Form(...),
    doc_number:  str = Form(default="—"),
    test_date:   str = Form(default=""),
    esito:       str = Form(default="AUTO"),
    channels:    str = Form(default="null"),
):
    if session_id not in _sessions:
        raise HTTPException(400, "Sessione scaduta. Ricarica i file.")

    sess = _sessions[session_id]

    try:
        selected = json.loads(channels) if channels and channels != "null" else None
    except Exception:
        selected = None

    pr     = sess["pr"]
    eg     = pr.get("esito_globale", "OK")
    if esito == "AUTO":
        esito_manual = {"OK": "PROVA RIUSCITA",
                        "KO_singolo": "PROVA NON RIUSCITA",
                        "KO_concentrazione": "SOSTITUIRE"}.get(eg, "PROVA NON RIUSCITA")
    else:
        esito_manual = esito

    try:
        lf     = sess["lf"]
        t      = sess["t"]
        active = sess["active"]
        zones  = sess["zones"]
        meta   = sess["meta"]

        docx_buf = generate_report(
            meta                 = meta,
            lf                   = lf,
            t                    = t,
            active_mask          = active,
            valid_zones          = zones,
            peak_result          = pr,
            selected_channels    = selected,
            esito_globale_manual = esito_manual,
            test_date_str        = test_date,
            doc_number           = doc_number,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    return StreamingResponse(
        docx_buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="Report_AMC.docx"'},
    )


# ---------------------------------------------------------------------------
# Pagina web
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AMC Report Generator</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: Arial, sans-serif; max-width: 860px; margin: 28px auto;
         padding: 0 18px; color: #222; background: #f5f6f8; }
  h1  { font-size: 1.35rem; color: #003d6b; margin-bottom: 2px; }
  h2  { font-size: 1.05rem; color: #003d6b; margin: 0 0 12px; }
  .card { background: white; border-radius: 8px; padding: 18px 22px;
          box-shadow: 0 1px 4px rgba(0,0,0,.10); margin-bottom: 18px; }
  label { display: block; margin-top: 10px; font-size: .87rem; font-weight: bold; }
  input[type=text], select {
    width: 100%; padding: 6px 10px; margin-top: 3px;
    border: 1px solid #ccc; border-radius: 5px; font-size: .88rem; }
  input[type=file]  { margin-top: 5px; font-size: .86rem; }
  input[type=range] { width: 100%; margin-top: 4px; accent-color: #003d6b; }
  .row { display: flex; gap: 14px; }
  .row > div { flex: 1; }
  .btn { padding: 9px 22px; color: white; border: none; border-radius: 5px;
         font-size: .92rem; cursor: pointer; margin-top: 12px; }
  .btn:hover    { filter: brightness(1.12); }
  .btn:disabled { background: #aaa !important; cursor: default; }
  .btn-blue  { background: #003d6b; }
  .btn-red   { background: #b22000; }
  .btn-grey  { background: #555; }
  .btn-teal  { background: #006f6f; }
  #status, #status2 { margin-top: 8px; font-size: .86rem; color: #555; min-height: 18px; }
  #step2 { display: none; }

  /* Charts */
  .chart-card { border: 2px solid #ddd; border-radius: 7px; padding: 9px 13px;
                margin-bottom: 10px; background: #fff; }
  .chart-card.active   { border-color: #003d6b; }
  .chart-card.inactive { border-color: #ccc; opacity: .65; }
  .chart-header { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
  .chart-header input[type=checkbox] { width: 17px; height: 17px; cursor: pointer; }
  .chart-header span.name { font-weight: bold; font-size: .93rem; }
  .badge { padding: 2px 8px; border-radius: 11px; font-size: .75rem;
           font-weight: bold; color: white; }
  .badge.ok       { background: #2a9d2a; }
  .badge.ko       { background: #cc2200; }
  .badge.inattivo { background: #888; }
  .chart-card img { width: 100%; border-radius: 4px; display: block; }

  /* Slider zone */
  .slider-box { background: #f0f4fa; border-radius: 6px; padding: 10px 14px;
                margin-bottom: 14px; }
  .slider-box h3 { margin: 0 0 8px; font-size: .92rem; color: #003d6b; }
  .slider-row { display: flex; align-items: center; gap: 10px; }
  .slider-row label { margin: 0; font-size: .82rem; width: 46px; flex-shrink: 0; }
  .slider-val { font-size: .82rem; color: #555; width: 48px; text-align: right; flex-shrink: 0; }
  .note { font-size: .75rem; color: #888; margin-top: 3px; }
  .meta-bar { font-size: .86rem; color: #444; margin-bottom: 12px;
              padding: 8px 12px; background: #eef2f8; border-radius: 5px; }
</style>
</head>
<body>

<h1>AMC Instruments — Report Generator</h1>

<!-- STEP 1 -->
<div class="card" id="step1">
  <h2>Step 1 — Carica i file</h2>
  <div class="row">
    <div><label>File .bin *<input type="file" id="bin_file" accept=".bin"></label></div>
    <div><label>File .txt *<input type="file" id="txt_file" accept=".txt"></label></div>
  </div>
  <div class="row">
    <div><label>Fattore soglia k<input type="text" id="peak_factor" value="2.5"></label></div>
    <div><label>Finestra riferimento [s]<input type="text" id="ref_win" value="5.0"></label></div>
  </div>
  <button class="btn btn-blue" id="btn_analyze" onclick="doAnalyze()">
    Analizza e mostra grafici
  </button>
  <div id="status"></div>
</div>

<!-- STEP 2 -->
<div id="step2">
  <div class="card">
    <h2>Step 2 — Verifica grafici e zona di analisi</h2>
    <div class="meta-bar" id="meta_info"></div>

    <!-- Slider sincrono -->
    <div class="slider-box">
      <h3>Zona di analisi (sincronizzata su tutti i canali)</h3>
      <div class="slider-row">
        <label>Start</label>
        <input type="range" id="sl_start" min="0" max="100" step="0.1" value="0"
               oninput="onSlider()">
        <span class="slider-val" id="val_start">0.0 s</span>
      </div>
      <div class="slider-row">
        <label>Stop</label>
        <input type="range" id="sl_stop" min="0" max="100" step="0.1" value="100"
               oninput="onSlider()">
        <span class="slider-val" id="val_stop">0.0 s</span>
      </div>
      <div style="display:flex; gap:10px; margin-top:8px;">
        <button class="btn btn-teal" style="margin-top:0; padding:7px 16px; font-size:.84rem;"
                onclick="doRecalc()">Applica taglio e ricalcola</button>
        <button class="btn btn-grey"  style="margin-top:0; padding:7px 16px; font-size:.84rem;"
                onclick="resetSliders()">Reset automatico</button>
      </div>
      <div id="status_recalc" class="note" style="margin-top:6px;"></div>
    </div>

    <div id="charts_container"></div>
  </div>

  <!-- Parametri report -->
  <div class="card">
    <h2>Step 3 — Parametri report</h2>
    <div class="row">
      <div><label>Numero documento<input type="text" id="doc_number" placeholder="es. 207004"></label></div>
      <div><label>Data test (gg/mm/aaaa)<input type="text" id="test_date"></label></div>
    </div>
    <label>Esito conclusioni
      <select id="esito_select">
        <option value="AUTO">Automatico (dal DSP)</option>
        <option value="PROVA RIUSCITA">Funi integre — nessun difetto significativo</option>
        <option value="PROVA NON RIUSCITA">Difetti lievi — mantenere con test annuale</option>
        <option value="SOSTITUIRE">Difetti gravi — sostituire le funi</option>
      </select>
    </label>
    <p class="note">Seleziona le funi da includere spuntando i checkbox sui grafici sopra.</p>
    <div style="display:flex; gap:12px; flex-wrap:wrap;">
      <button class="btn btn-red" onclick="doReport()">Genera e scarica Report Word</button>
      <button class="btn btn-grey" onclick="resetAll()">← Carica altri file</button>
    </div>
    <div id="status2"></div>
  </div>
</div>

<script>
  let sessionId  = null;
  let tMax       = 100;
  let autoStart  = 0;
  let autoStop   = 100;

  // Data odierna
  const today = new Date();
  document.getElementById('test_date').value =
    String(today.getDate()).padStart(2,'0') + '/' +
    String(today.getMonth()+1).padStart(2,'0') + '/' +
    today.getFullYear();

  // ---------- STEP 1: analisi ----------
  async function doAnalyze() {
    const binFile = document.getElementById('bin_file').files[0];
    const txtFile = document.getElementById('txt_file').files[0];
    if (!binFile || !txtFile) { alert('Seleziona entrambi i file .bin e .txt'); return; }

    const btn = document.getElementById('btn_analyze');
    btn.disabled = true;
    setStatus('status', 'Analisi in corso... (20-40 secondi)');

    const fd = new FormData();
    fd.append('bin_file',    binFile);
    fd.append('txt_file',    txtFile);
    fd.append('peak_factor', document.getElementById('peak_factor').value);
    fd.append('ref_win_sec', document.getElementById('ref_win').value);

    try {
      const resp = await fetch('/analyze', { method: 'POST', body: fd });
      if (!resp.ok) { const e = await resp.json(); setStatus('status', 'Errore: ' + e.detail); btn.disabled=false; return; }
      const data = await resp.json();
      sessionId = data.session_id;
      tMax      = data.t_max;
      autoStart = data.auto_start;
      autoStop  = data.auto_stop;
      initSliders(data.auto_start, data.auto_stop);
      renderStep2(data);
      setStatus('status', '');
    } catch(e) {
      setStatus('status', 'Errore di rete: ' + e.message);
      btn.disabled = false;
    }
  }

  // ---------- Slider ----------
  function initSliders(start, stop) {
    const sl_s = document.getElementById('sl_start');
    const sl_e = document.getElementById('sl_stop');
    sl_s.min = 0; sl_s.max = tMax; sl_s.step = 0.1; sl_s.value = start;
    sl_e.min = 0; sl_e.max = tMax; sl_e.step = 0.1; sl_e.value = stop;
    updateSliderLabels();
  }

  function onSlider() {
    const s = parseFloat(document.getElementById('sl_start').value);
    const e = parseFloat(document.getElementById('sl_stop').value);
    // Impedisci incrocio
    if (s >= e) {
      document.getElementById('sl_start').value = e - 0.1;
    }
    updateSliderLabels();
  }

  function updateSliderLabels() {
    const s = parseFloat(document.getElementById('sl_start').value);
    const e = parseFloat(document.getElementById('sl_stop').value);
    document.getElementById('val_start').textContent = s.toFixed(1) + ' s';
    document.getElementById('val_stop').textContent  = e.toFixed(1) + ' s';
  }

  function resetSliders() {
    initSliders(autoStart, autoStop);
    doRecalc();
  }

  // ---------- STEP 2a: ricalcolo con taglio ----------
  async function doRecalc() {
    if (!sessionId) return;
    const t_start = parseFloat(document.getElementById('sl_start').value);
    const t_stop  = parseFloat(document.getElementById('sl_stop').value);
    setStatus('status_recalc', 'Ricalcolo in corso...');

    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('t_start', t_start);
    fd.append('t_stop',  t_stop);

    try {
      const resp = await fetch('/recalc', { method: 'POST', body: fd });
      if (!resp.ok) { const e = await resp.json(); setStatus('status_recalc', 'Errore: ' + e.detail); return; }
      const data = await resp.json();
      updateCharts(data.charts, data.esito_globale);
      setStatus('status_recalc', '✅ Grafici aggiornati.');
      syncEsitoSelect(data.esito_globale);
    } catch(e) {
      setStatus('status_recalc', 'Errore di rete: ' + e.message);
    }
  }

  // ---------- Render grafico ----------
  function renderStep2(data) {
    const m = data.meta;
    document.getElementById('meta_info').innerHTML =
      `<b>${m.indirizzo} ${m.civico}, ${m.citta} (${m.provincia})</b> &nbsp;|&nbsp;
       Matricola: <b>${m.matricola}</b> &nbsp;|&nbsp; N. funi: <b>${m.nfuni}</b> &nbsp;|&nbsp;
       Esito DSP: <b style="color:${data.esito_globale==='OK'?'#2a9d2a':'#cc2200'}">${data.esito_globale}</b>`;

    renderCharts(data.charts);
    syncEsitoSelect(data.esito_globale);

    document.getElementById('step1').style.display = 'none';
    document.getElementById('step2').style.display = 'block';
  }

  function renderCharts(charts) {
    const container = document.getElementById('charts_container');
    container.innerHTML = '';
    charts.forEach(ch => buildChartCard(container, ch));
  }

  function updateCharts(charts) {
    const container = document.getElementById('charts_container');
    // Salva stato checkbox prima di aggiornare
    const checked = {};
    container.querySelectorAll('input[type=checkbox]').forEach(cb => {
      checked[cb.value] = cb.checked;
    });
    container.innerHTML = '';
    charts.forEach(ch => {
      buildChartCard(container, ch);
      // Ripristina checkbox
      const cb = container.querySelector(`input[value="${ch.ch}"]`);
      if (cb && checked[String(ch.ch)] !== undefined) cb.checked = checked[String(ch.ch)];
    });
  }

  function buildChartCard(container, ch) {
    const isKO = ch.esito.startsWith('KO');
    const badgeClass = ch.esito === 'inattivo' ? 'inattivo' : (isKO ? 'ko' : 'ok');
    const card = document.createElement('div');
    card.className = 'chart-card ' + (ch.active ? 'active' : 'inactive');
    card.innerHTML = `
      <div class="chart-header">
        <input type="checkbox" value="${ch.ch}" ${ch.active ? 'checked' : ''}>
        <span class="name">${ch.label}</span>
        <span class="badge ${badgeClass}">${ch.esito}</span>
        ${ch.n_picchi > 0 ? `<span style="font-size:.78rem;color:#888;">${ch.n_picchi} picchi</span>` : ''}
        <span style="font-size:.78rem;color:#888;margin-left:auto;">
          ${ch.t_start.toFixed(1)}s – ${ch.t_stop.toFixed(1)}s
        </span>
      </div>
      <img src="data:image/png;base64,${ch.png_b64}" alt="${ch.label}">
    `;
    container.appendChild(card);
  }

  function syncEsitoSelect(eg) {
    const sel = document.getElementById('esito_select');
    if (eg === 'OK')                  sel.value = 'PROVA RIUSCITA';
    else if (eg === 'KO_concentrazione') sel.value = 'SOSTITUIRE';
    else                              sel.value = 'PROVA NON RIUSCITA';
  }

  // ---------- STEP 3: report ----------
  async function doReport() {
    if (!sessionId) return;
    const selected = [...document.querySelectorAll('#charts_container input[type=checkbox]:checked')]
                       .map(c => parseInt(c.value));
    if (selected.length === 0) { alert('Seleziona almeno una fune.'); return; }

    setStatus('status2', 'Generazione report in corso...');
    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('doc_number', document.getElementById('doc_number').value || '—');
    fd.append('test_date',  document.getElementById('test_date').value);
    fd.append('esito',      document.getElementById('esito_select').value);
    fd.append('channels',   JSON.stringify(selected));

    try {
      const resp = await fetch('/report', { method: 'POST', body: fd });
      if (!resp.ok) { const e = await resp.json(); setStatus('status2', 'Errore: ' + e.detail); return; }
      const blob = await resp.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href = url; a.download = 'Report_AMC.docx'; a.click();
      URL.revokeObjectURL(url);
      setStatus('status2', '✅ Report scaricato.');
    } catch(e) {
      setStatus('status2', 'Errore di rete: ' + e.message);
    }
  }

  // ---------- Reset ----------
  function resetAll() {
    sessionId = null;
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step1').style.display = 'block';
    document.getElementById('btn_analyze').disabled = false;
    document.getElementById('bin_file').value = '';
    document.getElementById('txt_file').value = '';
    setStatus('status', '');
  }

  function setStatus(id, msg) {
    document.getElementById(id).textContent = msg;
  }
</script>
</body>
</html>
"""
