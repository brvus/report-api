"""
main.py  —  FastAPI server AMC Report  (two-step: analyze → report)
===================================================================
Step 1  POST /analyze   -> DSP + grafici PNG base64 + metadati
Step 2  POST /report    -> .docx scaricabile
        (i file sono cachati in memoria con un session_id UUID)
GET  /                  -> pagina web
GET  /health            -> healthcheck Render
"""

import io
import json
import uuid
import base64
from typing import Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse

from report_engine import (
    run_pipeline, read_bin, parse_txt, detect_active_channels,
    detect_shared_zone, analyze_peaks, trim_initial_from_peaks,
    render_channel_png, generate_report,
    PEAK_FACTOR, REF_WIN_SEC, FS
)
import numpy as np

app = FastAPI(title="AMC Report API", version="2.0")

# Cache in memoria: session_id -> {bin_bytes, txt_bytes, meta, lf, t, active, zones, peak_result}
_sessions: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Step 1 — Analisi + preview grafici
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
        N    = lf.shape[0]
        t    = np.linspace(1, N, N) / FS

        try:
            n_funi_exp = int(meta.get("nfuni", "0"))
        except (ValueError, TypeError):
            n_funi_exp = 0

        active = detect_active_channels(lf, n_funi_exp, n_ch=6)
        zones  = detect_shared_zone(lf, active, n_ch=6)
        pr     = analyze_peaks(lf, active, zones,
                               peak_factor=peak_factor, ref_win_sec=ref_win_sec)
        zones, trimmed = trim_initial_from_peaks(pr, active, zones)
        if trimmed:
            pr = analyze_peaks(lf, active, zones,
                               peak_factor=peak_factor, ref_win_sec=ref_win_sec)

        # Genera PNG per ogni canale (attivo o no, così l'utente può scegliere)
        charts = []
        per_ch = pr.get("per_canale", [])
        for ch in range(min(6, lf.shape[1])):
            s_v, e_v = zones[ch] if ch < len(zones) else (0, len(t) - 1)
            esito_ch = per_ch[ch].get("esito", "OK") if ch < len(per_ch) else "OK"
            is_active = bool(active[ch]) if ch < len(active) else False
            buf = render_channel_png(lf, t, ch, s_v, e_v, esito_ch,
                                     width_in=12, height_in=2.8, dpi=100)
            charts.append({
                "ch":        ch,
                "label":     f"Fune {ch+1}",
                "esito":     esito_ch,
                "n_picchi":  per_ch[ch].get("n_picchi", 0) if ch < len(per_ch) else 0,
                "active":    is_active,
                "png_b64":   base64.b64encode(buf.read()).decode(),
            })

    except Exception as e:
        raise HTTPException(500, str(e))

    # Salva sessione
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "bin_bytes": bin_bytes,
        "txt_bytes": txt_bytes,
        "peak_factor": peak_factor,
        "ref_win_sec": ref_win_sec,
    }
    # Pulizia sessioni vecchie (tieni max 20)
    if len(_sessions) > 20:
        oldest = next(iter(_sessions))
        del _sessions[oldest]

    return {
        "session_id":   sid,
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
# Step 2 — Genera .docx
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
        raise HTTPException(400, "Sessione scaduta o non trovata. Ricarica i file.")

    sess = _sessions[session_id]
    bin_bytes    = sess["bin_bytes"]
    txt_bytes    = sess["txt_bytes"]
    peak_factor  = sess["peak_factor"]
    ref_win_sec  = sess["ref_win_sec"]

    try:
        selected = json.loads(channels) if channels and channels != "null" else None
    except Exception:
        selected = None

    esito_manual = esito if esito != "AUTO" else "PROVA NON RIUSCITA"

    try:
        docx_buf = run_pipeline(
            bin_bytes         = bin_bytes,
            txt_bytes         = txt_bytes,
            selected_channels = selected,
            esito_manual      = esito_manual,
            test_date_str     = test_date,
            doc_number        = doc_number,
            peak_factor       = peak_factor,
            ref_win_sec       = ref_win_sec,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    filename = f"Report_AMC.docx"
    return StreamingResponse(
        docx_buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Pagina web two-step
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
  *    { box-sizing: border-box; }
  body { font-family: Arial, sans-serif; max-width: 820px; margin: 32px auto;
         padding: 0 20px; color: #222; background: #f7f8fa; }
  h1   { font-size: 1.4rem; color: #003d6b; margin-bottom: 4px; }
  h2   { font-size: 1.1rem; color: #003d6b; margin: 24px 0 8px; }
  .card { background: white; border-radius: 8px; padding: 20px 24px;
          box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 20px; }
  label { display: block; margin-top: 12px; font-size: .88rem; font-weight: bold; }
  input[type=text], select {
    width: 100%; padding: 7px 10px; margin-top: 4px;
    border: 1px solid #ccc; border-radius: 5px; font-size: .9rem; }
  input[type=file] { margin-top: 6px; font-size: .88rem; }
  .row { display: flex; gap: 16px; }
  .row > div { flex: 1; }
  button {
    margin-top: 16px; padding: 10px 26px;
    background: #003d6b; color: white; border: none;
    border-radius: 5px; font-size: .95rem; cursor: pointer; }
  button:hover  { background: #005a9e; }
  button:disabled { background: #aaa; cursor: default; }
  #status { margin-top: 10px; font-size: .88rem; color: #555; min-height: 20px; }

  /* Step 2 */
  #step2 { display: none; }
  .chart-card {
    border: 2px solid #ddd; border-radius: 7px; padding: 10px 14px;
    margin-bottom: 12px; background: #fff; }
  .chart-card.active   { border-color: #003d6b; }
  .chart-card.inactive { border-color: #ccc; opacity: .7; }
  .chart-header { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
  .chart-header input[type=checkbox] { width: 18px; height: 18px; cursor: pointer; }
  .chart-header span { font-weight: bold; font-size: .95rem; }
  .badge {
    padding: 2px 9px; border-radius: 12px; font-size: .78rem;
    font-weight: bold; color: white; }
  .badge.ok   { background: #2a9d2a; }
  .badge.ko   { background: #cc2200; }
  .badge.inattivo { background: #888; }
  .chart-card img { width: 100%; border-radius: 4px; }
  .btn-report { background: #b22000; margin-top: 8px; }
  .btn-report:hover { background: #d42800; }
  .note { font-size: .76rem; color: #888; margin-top: 3px; }
</style>
</head>
<body>

<h1>AMC Instruments — Report Generator</h1>

<!-- STEP 1: Upload -->
<div class="card" id="step1">
  <h2>Step 1 — Carica i file</h2>
  <div class="row">
    <div>
      <label>File .bin *
        <input type="file" id="bin_file" accept=".bin">
      </label>
    </div>
    <div>
      <label>File .txt *
        <input type="file" id="txt_file" accept=".txt">
      </label>
    </div>
  </div>
  <div class="row">
    <div>
      <label>Fattore soglia k
        <input type="text" id="peak_factor" value="2.5">
      </label>
    </div>
    <div>
      <label>Finestra riferimento [s]
        <input type="text" id="ref_win" value="5.0">
      </label>
    </div>
  </div>
  <button id="btn_analyze" onclick="doAnalyze()">Analizza e mostra grafici</button>
  <div id="status"></div>
</div>

<!-- STEP 2: Preview + parametri report -->
<div id="step2">
  <div class="card">
    <h2>Step 2 — Seleziona funi e genera report</h2>
    <div id="meta_info" style="font-size:.88rem; color:#555; margin-bottom:10px;"></div>

    <div id="charts_container"></div>

    <div class="row" style="margin-top:16px;">
      <div>
        <label>Numero documento
          <input type="text" id="doc_number" placeholder="es. 207004">
        </label>
      </div>
      <div>
        <label>Data test (gg/mm/aaaa)
          <input type="text" id="test_date">
        </label>
      </div>
    </div>
    <label>Esito conclusioni
      <select id="esito_select">
        <option value="AUTO">Automatico (dal DSP)</option>
        <option value="PROVA RIUSCITA">Funi integre — nessun difetto significativo</option>
        <option value="PROVA NON RIUSCITA">Difetti lievi — mantenere con test annuale</option>
        <option value="SOSTITUIRE">Difetti gravi — sostituire le funi</option>
      </select>
    </label>
    <p class="note">L'esito "Automatico" usa il risultato DSP: OK→integre, KO singolo→difetti lievi, KO concentrazione→sostituire.</p>

    <div style="display:flex; gap:12px; flex-wrap:wrap;">
      <button class="btn-report" onclick="doReport()">Genera e scarica Report Word</button>
      <button style="background:#555; margin-top:16px;" onclick="resetAll()">← Carica altri file</button>
    </div>
    <div id="status2"></div>
  </div>
</div>

<script>
  let sessionId = null;

  // Pre-compila data odierna
  const today = new Date();
  document.getElementById('test_date').value =
    String(today.getDate()).padStart(2,'0') + '/' +
    String(today.getMonth()+1).padStart(2,'0') + '/' +
    today.getFullYear();

  async function doAnalyze() {
    const binFile = document.getElementById('bin_file').files[0];
    const txtFile = document.getElementById('txt_file').files[0];
    if (!binFile || !txtFile) {
      alert('Seleziona entrambi i file .bin e .txt');
      return;
    }
    const status = document.getElementById('status');
    const btn    = document.getElementById('btn_analyze');
    btn.disabled = true;
    status.textContent = 'Analisi in corso... (può richiedere 20-40 secondi)';

    const fd = new FormData();
    fd.append('bin_file',    binFile);
    fd.append('txt_file',    txtFile);
    fd.append('peak_factor', document.getElementById('peak_factor').value);
    fd.append('ref_win_sec', document.getElementById('ref_win').value);

    try {
      const resp = await fetch('/analyze', { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json();
        status.textContent = 'Errore: ' + (err.detail || resp.statusText);
        btn.disabled = false;
        return;
      }
      const data = await resp.json();
      sessionId = data.session_id;
      renderStep2(data);
      status.textContent = '';
    } catch(e) {
      status.textContent = 'Errore di rete: ' + e.message;
      btn.disabled = false;
    }
  }

  function renderStep2(data) {
    // Metadati
    const m = data.meta;
    document.getElementById('meta_info').innerHTML =
      `<b>${m.indirizzo} ${m.civico}, ${m.citta} (${m.provincia})</b> &nbsp;|&nbsp; 
       Matricola: <b>${m.matricola}</b> &nbsp;|&nbsp; N. funi attese: <b>${m.nfuni}</b> &nbsp;|&nbsp;
       Esito DSP: <b style="color:${data.esito_globale==='OK'?'#2a9d2a':'#cc2200'}">${data.esito_globale}</b>`;

    // Grafici
    const container = document.getElementById('charts_container');
    container.innerHTML = '';
    data.charts.forEach(ch => {
      const isKO = ch.esito.startsWith('KO');
      const badgeClass = ch.esito === 'inattivo' ? 'inattivo' : (isKO ? 'ko' : 'ok');
      const card = document.createElement('div');
      card.className = 'chart-card ' + (ch.active ? 'active' : 'inactive');
      card.innerHTML = `
        <div class="chart-header">
          <input type="checkbox" id="ch_${ch.ch}" value="${ch.ch}" ${ch.active ? 'checked' : ''}>
          <span>${ch.label}</span>
          <span class="badge ${badgeClass}">${ch.esito}</span>
          ${ch.n_picchi > 0 ? `<span style="font-size:.8rem;color:#888;">${ch.n_picchi} picchi</span>` : ''}
        </div>
        <img src="data:image/png;base64,${ch.png_b64}" alt="Segnale ${ch.label}">
      `;
      container.appendChild(card);
    });

    // Preseleziona esito
    const sel = document.getElementById('esito_select');
    if (data.esito_globale === 'OK') sel.value = 'PROVA RIUSCITA';
    else if (data.esito_globale === 'KO_concentrazione') sel.value = 'SOSTITUIRE';
    else sel.value = 'PROVA NON RIUSCITA';

    document.getElementById('step1').style.display = 'none';
    document.getElementById('step2').style.display = 'block';
  }

  async function doReport() {
    if (!sessionId) return;
    const selected = [...document.querySelectorAll('#charts_container input[type=checkbox]:checked')]
                       .map(c => parseInt(c.value));
    if (selected.length === 0) {
      alert('Seleziona almeno una fune.');
      return;
    }
    const status = document.getElementById('status2');
    status.textContent = 'Generazione report in corso...';

    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('doc_number', document.getElementById('doc_number').value || '—');
    fd.append('test_date',  document.getElementById('test_date').value);
    fd.append('esito',      document.getElementById('esito_select').value);
    fd.append('channels',   JSON.stringify(selected));

    try {
      const resp = await fetch('/report', { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json();
        status.textContent = 'Errore: ' + (err.detail || resp.statusText);
        return;
      }
      const blob = await resp.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href = url;
      a.download = 'Report_AMC.docx';
      a.click();
      URL.revokeObjectURL(url);
      status.textContent = '✅ Report scaricato.';
    } catch(e) {
      status.textContent = 'Errore di rete: ' + e.message;
    }
  }

  function resetAll() {
    sessionId = null;
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step1').style.display = 'block';
    document.getElementById('btn_analyze').disabled = false;
    document.getElementById('status').textContent = '';
    document.getElementById('bin_file').value = '';
    document.getElementById('txt_file').value = '';
  }
</script>
</body>
</html>
"""
