# Resume Extraction API Starter

Full project documentation: [../README.md](../README.md) (overview) and [../DOCS/PROJECT_MANUAL.md](../DOCS/PROJECT_MANUAL.md) (manual).

## Run locally

From the project root, activate the venv, then start the API.

**macOS / Linux**

```bash
source env/bin/activate   # or: source .venv/bin/activate
cd code
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Windows (PowerShell)** — use `.\` so the script is found:

```powershell
.\env\Scripts\Activate.ps1
cd code
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If execution policy blocks the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Dependencies (any OS):** from the repo root, install with `-r` (not `pip install requirements.txt`):

```powershell
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

That uses the venv’s Python even if `pip` is not on your PATH. Prefer `Activate.ps1` in PowerShell; `activate.bat` is mainly for **cmd.exe**.

On **Windows**, `uvloop` is skipped automatically (it is not supported on Windows); Uvicorn still runs using the default event loop.

## Using the `env` venv (from repo root)

With `env` activated (`.\env\Scripts\Activate.ps1`):

```powershell
# Unit tests (mock inference, fast)
$env:USE_MOCK_INFERENCE = "1"
cd code
python -m pytest -q

# API (uses NER weights under models/resume-ner/final if present)
$env:USE_MOCK_INFERENCE = "0"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Back on repo root — baseline vs LoRA metrics → `reports/`:

```powershell
cd ..
python code/scripts/audit_ner_dataset.py
python code/scripts/evaluate_baseline_vs_lora.py
```

WP1 manifests (inventory + splits) committed under `manifests/` — regenerate after corpus changes:

```powershell
python code/scripts/prepare_dataset_manifests.py --dataset-root "dataset/Resumes PDF" --output-dir manifests --overrides code/config/label_overrides.yaml
```

Parser benchmark (150-sample + synthetic self-check) → `reports/parser_benchmark.json`:

```powershell
python code/scripts/parser_benchmark.py --sample 150 --dataset-root "dataset/Resumes PDF"
```

**Windows (Command Prompt)**

```cmd
env\Scripts\activate.bat
cd code
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open API docs:
- http://127.0.0.1:8000/docs

## Endpoints
- `GET /health`
- `POST /extract` (multipart form field: `file`, PDF only)

## Parser behavior
- Primary path: embedded text extraction with PyMuPDF.
- Fallback: pdfminer.six when embedded text is very short, then OCR for scanned PDFs (PyMuPDF render + Tesseract).

If scanned PDFs return OCR-related errors, install the **Tesseract** system binary. On Windows the parser also checks the default install path `C:\Program Files\Tesseract-OCR\tesseract.exe` (no PATH change needed). One-line install:

```powershell
winget install UB-Mannheim.TesseractOCR --accept-package-agreements
```

Or set `TESSERACT_CMD` to the full path to `tesseract.exe` (see `.env.example`). Restart **uvicorn** after installing.

```bash
brew install tesseract   # macOS
```

## Run tests

After activating the venv (see above), from `code/`:

```bash
python -m pytest -q
```

## Environment

- `RESUME_MODEL_PATH` — directory with merged LoRA export (default: repo `models/resume-ner/final`).
- `USE_MOCK_INFERENCE=1` — skip torch NER (useful for quick tests without model weights).
- `NER_DOC_PEAK_MIN`, `NER_ABS_SCORE_FLOOR`, `NER_REL_TO_PEAK` — optional NER confidence gating (see `models/resume-ner/MODEL_CARD.md`).

After training, `models/resume-ner/final/merged/` should exist; or run `python code/scripts/export_merged_lora.py` to build it from adapters.
