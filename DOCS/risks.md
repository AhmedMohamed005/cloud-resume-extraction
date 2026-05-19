# Risk register (project)

| Risk | Impact | Mitigation |
|------|--------|------------|
| PDF parsing fails (layout, scan quality) | Empty or noisy text | pdfminer fallback + OCR; `parser_quality` in metadata; manual QA on sample PDFs |
| Model truncation (512 tokens) | Missing tail fields | Document limit; future chunking / long-document models |
| PII in logs | Privacy breach | Log only `request_id`, lengths, and errors — not full resume text |
| Cloud cost / abuse | Bill shock | Rate limiting (future), size limits, private security groups |
| Baseline metrics look random | Confusing report | Document that baseline = untrained head on same backbone; compare delta to LoRA |

See `Resume_Extraction_Full_Execution_Plan.md` §11 for the original mitigation list.
