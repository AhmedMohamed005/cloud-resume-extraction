# Demo script (examiner-friendly)

1. Show OpenAPI docs at `/docs` and health at `GET /health`.
2. Upload an **unseen** PDF via `POST /extract` (Postman or Swagger).
3. Point out `metadata.parser_used`, `parser_quality`, `parser_confidence`, and `processing_ms`.
4. Show structured `profile` JSON (name, skills, education, experience, contact).
5. Repeat with `?debug=true` and briefly show cleaned text snippet (no full PII if policy requires).
6. Open `reports/baseline_metrics.json` vs `reports/fine_tuned_metrics.json` (after running `evaluate_baseline_vs_lora.py`) and compare overall F1.
7. If deployed on EC2, show public URL and same curl/Postman call.
