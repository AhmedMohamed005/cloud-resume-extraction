# API guide

## Base URL

Local: `http://127.0.0.1:8000`  
OpenAPI: `http://127.0.0.1:8000/docs`

## Endpoints

### `GET /health`

Returns `{"status":"ok"}`.

### `POST /extract`

- **Body:** `multipart/form-data` with field **`file`** (PDF).
- **Query:** `debug=true` optional — includes `raw_text`, `cleaned_text`, `detected_sections`, `final_profile` (see router implementation).

**Success (200):**

```json
{
  "profile": {
    "name": "...",
    "email": "...",
    "phone": "...",
    "skills": [],
    "education": [],
    "experience": []
  },
  "metadata": {
    "request_id": "...",
    "parser_used": "pymupdf",
    "text_length": 1234,
    "confidence": 0.72,
    "processing_ms": 850,
    "inference_backend": "ner_lora_hybrid",
    "parser_quality": "ok",
    "parser_confidence": 1.0
  },
  "debug": null
}
```

### Inference backends (`metadata.inference_backend`)

| Value | Meaning |
|-------|---------|
| `ner_lora_hybrid` | Hugging Face BERT + merged LoRA NER ran; profile merged with heuristics for gaps/stability. |
| `mock_heuristic` | `USE_MOCK_INFERENCE=1` or no model weights — rule-based extraction only. |
| `mock_heuristic_fallback` | NER load/inference failed; heuristics used. |

**NLP note:** The Concept note and architecture sketch describe **named-entity extraction** with LoRA; the hybrid path is an implementation detail so JSON stays usable when NER scores are flat or the model is still being improved.

**Errors (structured `detail` object):**

| Code | `error` | Typical cause |
|------|---------|----------------|
| 400 | `invalid_content_type` | Not PDF |
| 400 | `empty_file` | Zero bytes |
| 408 | `upload_timeout` | Slow client (30s read timeout) |
| 413 | `file_too_large` | Over 10 MB |
| 422 | `parse_failed` | Unreadable PDF / OCR failure / missing OCR deps |

`detail` is a JSON object: `{ "error": "<code>", "message": "<human text>" }`.

## Postman

Import `code/postman/Resume_Extraction.postman_collection.json` and set collection variable **`baseUrl`** (e.g. `http://127.0.0.1:8000` locally, or `http://YOUR_EC2_IP:8000` after deploy). See [deployment.md](deployment.md) and [POSTMAN_AND_REMOTE_TESTING.md](POSTMAN_AND_REMOTE_TESTING.md).

## Environment

| Variable | Purpose |
|----------|---------|
| `USE_MOCK_INFERENCE=1` | Skip torch NER (CI / no weights). |
| `RESUME_MODEL_PATH` | Override default `models/resume-ner/final` (repo root–relative or absolute). |
| `TESSERACT_CMD` | Full path to `tesseract.exe` if not on `PATH` (Windows). |
| `OCR_RENDER_ZOOM`, `OCR_TESS_PSMS`, `OCR_TESS_OEM` | OCR quality — see `preprocessing_rules.md`. |
| `NER_DOC_PEAK_MIN`, `NER_ABS_SCORE_FLOOR`, `NER_REL_TO_PEAK` | NER abstention / filtering — see `models/resume-ner/MODEL_CARD.md`. |
