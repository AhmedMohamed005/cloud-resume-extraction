# Resume Extraction API Starter

## Run locally

From the project root:

```bash
source .env/bin/activate
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
- Fallback path: OCR for scanned/image PDFs using PyMuPDF page rendering + Tesseract.

If scanned PDFs return OCR-related errors, install the system binary:

```bash
brew install tesseract
```

## Run tests

```bash
source .env/bin/activate
cd code
python -m pytest -q
```

## Next integration steps
1. Replace `run_mock_inference` in `app/services/inference.py` with baseline model inference.
2. Add confidence and page-level diagnostics for OCR extraction quality.
3. Add request logging and model timing breakdown in route metadata.
4. Add schema-level validation tests for required JSON keys.
