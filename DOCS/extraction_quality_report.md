# PDF extraction quality (WP2)

## Strategy

1. **PyMuPDF** — fast embedded text.
2. **pdfminer.six** — second attempt when embedded text is shorter than ~40 characters (weak signal).
3. **Tesseract OCR** (via rendered page images) when both fail.

## Metadata (API)

- `parser_used`: `pymupdf` | `pdfminer` | `pymupdf+ocr`
- `parser_quality`: `ok` | `low_text` | `empty` | `noisy_symbols`
- `parser_confidence`: heuristic 0–1 derived from `parser_quality` (not a calibrated probability)

## Latest benchmark run

Artifact: [`reports/parser_benchmark.json`](../reports/parser_benchmark.json) (regenerate after environment or corpus changes).

| Field | Last recorded run |
|-------|-------------------|
| When (UTC) | See `generated_at` in JSON |
| Corpus | `dataset/Resumes PDF` (local; gitignored) |
| Sample size | 150 (of 294 PDFs), seed 42 |
| Synthetic self-check | Embedded-text PDF must parse with PyMuPDF (`synthetic_embedded_text.ok`) |
| Corpus success rate | See `non_empty_text_rate` |

**Interpretation of a 0% corpus rate:** the team’s PDFs are largely **image-only** (no extractable text). PyMuPDF and pdfminer return empty; OCR then requires the **Tesseract binary** plus `pytesseract` / `Pillow`. After installing Tesseract, re-run the benchmark; the ≥95% WP2 target applies to the **intended runtime environment** (EC2/Docker image with OCR), not to a dev laptop without OCR.

## Commands

```bash
# From repository root, with dependencies installed
python code/scripts/parser_benchmark.py --sample 150 --dataset-root "dataset/Resumes PDF"
```

The script always runs a **synthetic embedded-text PDF** so parser wiring can be validated even when the corpus is scanned-only.

## Acceptance target

Execution plan WP2: ≥95% of sampled PDFs yield non-empty meaningful text **when OCR is available** for scanned resumes; tune `_MIN_MEANINGFUL_CHARS` in `parser.py` only with team agreement.
