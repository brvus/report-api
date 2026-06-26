"""
main.py  —  FastAPI server per generazione report AMC
=====================================================
Endpoint:
  POST /report      -> scarica il .docx generato
  GET  /            -> pagina web di upload (HTML inline)
  GET  /health      -> healthcheck per Render
"""

import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse

from report_engine import run_pipeline, PEAK_FACTOR, REF_WIN_SEC

app = FastAPI(title="AMC Report API", version="1.0")


# ---------------------------------------------------------------------------
# Health check (Render lo chiama per sapere se il container è vivo)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Pagina web di upload
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
  body { font-family: Arial, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1   { font-size: 1.4rem; color: #003d6b; }
  label { display: block; margin-top: 14px; font-size: 0.9rem; font-weight: bold; }
  input[type=text], select { width: 100%; padding: 7px; margin-top: 4px; border: 1px solid #bbb; border-radius: 4px; box-sizing: border-box; }
  input[type=file] { margin-top: 4px; }
  .ch-row { display: flex; gap: 16px; margin-top: 6px; }
  .ch-row label { font-weight: normal; display: flex; align-items: center; gap: 4px; margin-top: 0; }
  button { margin-top: 24px; padding: 10px 28px; background: #003d6b; color: white; border: none; border-radius: 5px; font-size: 1rem; cursor: pointer; }
  button:hover { background: #005a9e; }
  #status { margin-top: 14px; font-size: 0.9rem; color: #555; }
  .note { font-size: 0.78rem; color: #888; margin-top: 4px; }
</style>
</head>
<body>
<h1>AMC Instruments — Report Generator</h1>
<form id="form">
  <label>File .bin *
    <input type="file" name="bin_file" accept=".bin" required>
  </label>

  <label>File .txt *
    <input type="file" name="txt_file" accept=".txt" required>
  </label>

  <label>Numero documento
    <input type="text" name="doc_number" placeholder="es. 207004">
  </label>

  <label>Data test (gg/mm/aaaa)
    <input type="text" name="test_date" id="test_date">
  </label>

  <label>Funi da includere nel report
    <div class="ch-row" id="ch_checks">
      <label><input type="checkbox" name="channels" value="0" checked> Fune 1</label>
      <label><input type="checkbox" name="channels" value="1" checked> Fune 2</label>
      <label><input type="checkbox" name="channels" value="2" checked> Fune 3</label>
      <label><input type="checkbox" name="channels" value="3" checked> Fune 4</label>
      <label><input type="checkbox" name="channels" value="4" checked> Fune 5</label>
      <label><input type="checkbox" name="channels" value="5" checked> Fune 6</label>
    </div>
    <p class="note">Deseleziona le funi non presenti o non da includere.</p>
  </label>

  <label>Esito conclusioni
    <select name="esito">
      <option value="AUTO">Automatico (dal DSP)</option>
      <option value="PROVA RIUSCITA">Prova riuscita — funi integre</option>
      <option value="PROVA NON RIUSCITA">Prova non riuscita — difetti lievi</option>
      <option value="SOSTITUIRE">Sostituire — difetti gravi</option>
    </select>
  </label>

  <button type="button" onclick="submitForm()">Genera e scarica Report</button>
</form>
<div id="status"></div>

<script>
  // Pre-compila data odierna
  const today = new Date();
  document.getElementById('test_date').value =
    String(today.getDate()).padStart(2,'0') + '/' +
    String(today.getMonth()+1).padStart(2,'0') + '/' +
    today.getFullYear();

  async function submitForm() {
    const form = document.getElementById('form');
    const status = document.getElementById('status');
    const fd = new FormData();
    fd.append('bin_file', form.bin_file.files[0]);
    fd.append('txt_file', form.txt_file.files[0]);
    fd.append('doc_number', form.doc_number.value || '—');
    fd.append('test_date',  form.test_date.value);
    fd.append('esito',      form.esito.value);
    const checked = [...form.querySelectorAll('input[name=channels]:checked')].map(c => c.value);
    fd.append('channels', JSON.stringify(checked.map(Number)));

    status.textContent = 'Elaborazione in corso...';
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
      a.download = resp.headers.get('content-disposition')
                    ?.split('filename=')[1]?.replace(/"/g,'') || 'report.docx';
      a.click();
      URL.revokeObjectURL(url);
      status.textContent = 'Report scaricato.';
    } catch(e) {
      status.textContent = 'Errore di rete: ' + e.message;
    }
  }
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Endpoint principale
# ---------------------------------------------------------------------------

@app.post("/report")
async def create_report(
    bin_file:   UploadFile = File(...),
    txt_file:   UploadFile = File(...),
    doc_number: str        = Form(default="—"),
    test_date:  str        = Form(default=""),
    esito:      str        = Form(default="AUTO"),
    channels:   str        = Form(default="null"),   # JSON array, es. "[0,1,2]"
    peak_factor: float     = Form(default=PEAK_FACTOR),
    ref_win_sec: float     = Form(default=REF_WIN_SEC),
):
    # Leggi i file
    bin_bytes = await bin_file.read()
    txt_bytes = await txt_file.read()

    if len(bin_bytes) == 0:
        raise HTTPException(status_code=400, detail="File .bin vuoto.")
    if len(txt_bytes) == 0:
        raise HTTPException(status_code=400, detail="File .txt vuoto.")

    # Canali selezionati
    try:
        selected = json.loads(channels) if channels and channels != "null" else None
    except Exception:
        selected = None

    # Esito: AUTO = lascia decidere al DSP
    esito_manual = esito if esito != "AUTO" else "PROVA NON RIUSCITA"

    try:
        docx_buf = run_pipeline(
            bin_bytes        = bin_bytes,
            txt_bytes        = txt_bytes,
            selected_channels= selected,
            esito_manual     = esito_manual,
            test_date_str    = test_date,
            doc_number       = doc_number,
            peak_factor      = peak_factor,
            ref_win_sec      = ref_win_sec,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = f"Report_{bin_file.filename.replace('.bin','')}.docx"
    return StreamingResponse(
        docx_buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
