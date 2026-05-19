# Preprocessing rules (WP3)

Implemented in `code/app/services/preprocess.py`.

## Goals

- Deterministic cleaning so the same PDF text always yields the same normalized string before NER.
- Reduce OCR artifacts that confuse the name and section heuristics used in the heuristic fallback path.

## Rules (high level)

1. **Claire OCR artifact repair** — detected when “Claire” appears at abnormal frequency; selective substitution to restore `s` / `S3` patterns.
2. **Whitespace** — normalize line breaks where needed for stable tokenization (see code for exact regex and ordering).

## Section heuristics (fallback path only)

When `USE_MOCK_INFERENCE=1` or NER fails, `code/app/services/inference.py` splits the cleaned text into sections using keyword headers (`experience`, `education`, `skills`, etc.). Lines matching **`Header: body`** keep the body in that section (e.g. `Experience: Software Engineer at …`). This is **deterministic** for a given cleaned string and complements NER; it is not used as ground truth for model training.

## Traceability

- API `debug=true` returns `cleaned_text` alongside `raw_text` from the parser, plus `detected_sections` from the heuristic pass and the merged `final_profile`.

## OCR (PDF path)

Parser env (not FastAPI): `OCR_RENDER_ZOOM` (default `3.5`), `OCR_TESS_PSMS` (default `6,4` — tries multiple page segmentation modes per page and keeps the longest text), `OCR_TESS_OEM` (default `3`).

## Future

- Explicit section tagging as model features (beyond current NER) if instruction-style models are adopted later.
