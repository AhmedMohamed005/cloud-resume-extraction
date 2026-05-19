# Proposal and concept documents — technical alignment

This table maps the **submitted narrative** ([Project_Idea_Proposal_Template_and_Supervisor_Checklist_Final.txt](Project_Idea_Proposal_Template_and_Supervisor_Checklist_Final.txt), [Concept_Note_Resume_Extraction.txt](Concept_Note_Resume_Extraction.txt), [Architecture_Sketch_Resume_Extraction.txt](Architecture_Sketch_Resume_Extraction.txt)) to **what this repository implements**. Use it for supervisor Q&A without rewriting historical PDFs/TXTs.

| Proposal / sketch topic | Document says | Repository reality | One-line for supervisor |
|---------------------------|---------------|--------------------|-------------------------|
| PDF stack (§5) | pdfplumber / PyMuPDF | **PyMuPDF**, **pdfminer.six** fallback, **Tesseract OCR** for image-only PDFs | Same functional role as the proposal; tooling matches Architecture/Concept note (pdfminer), not pdfplumber. |
| AI approach (§7) | “Instruction-based learning (text → structured JSON)” | **Token classification (BERT + LoRA BIO tags)** + span grouping + JSON schema mapping; **`ner_lora_hybrid`** merges NER with **heuristic** section/skill rules when the model is weak or disabled | Course outcome is structured JSON from PDFs; implementation uses **NER + formatter**, which matches the Concept note and architecture sketch (“entity predictions → JSON”). |
| Output example (§7) | `"education": "..."` (string) | **`education`**: `string[]` (and **`experience`**: `string[]`), plus **`email`**, **`phone`** | API schema is **richer** than the MVP example; lists match real multi-line education/experience. |
| Data storage (§5 / §8) | SQLite + optional S3 | **Default: stateless API** (no DB). Optional S3 described as future. See [mvp_storage_decision.md](mvp_storage_decision.md). | **Data Storage Module** in the proposal is **optional**; MVP is extraction-only. Add SQLite only if the rubric explicitly requires a database (see that doc). |
| Evaluation (§9) | P/R/F1, baseline vs fine-tuned | **`code/scripts/evaluate_baseline_vs_lora.py`** → `reports/baseline_metrics.json`, `reports/fine_tuned_metrics.json`; **`code/scripts/train.py`**; [final_evaluation_report.md](final_evaluation_report.md) | Metrics are **offline** (not a separate microservice on each `/extract` call), consistent with the updated architecture sketch note. |
| Cloud (§8) | Docker + AWS EC2 | **Dockerfile**, [deployment.md](deployment.md), EC2 outline | Aligns; IAM/CloudWatch/S3 are **documented** as operational practices, not all wired in application code. |
| Modular pipeline | PDF → preprocess → model → JSON | Same order in [extract.py](../code/app/routers/extract.py) | **Aligned.** |

## Canonical API schema

The live contract is Pydantic in `code/app/schemas/output_schema.py` and OpenAPI at `/docs`. Fields: **`name`**, **`email`**, **`phone`**, **`skills`[]**, **`education`[]**, **`experience`[]**, plus **`metadata`** (`request_id`, `parser_used`, `parser_quality`, `parser_confidence`, `inference_backend`, etc.).

## SQLite decision

- **Not required** for Definition of Done in this repo’s MVP: extraction returns JSON in the HTTP response.
- **Add SQLite** only if course/supervisor requires proof of persistence: then implement a small append-only log (`request_id`, timestamp, optional filename, JSON blob) behind `ENABLE_SQLITE=1` (see [mvp_storage_decision.md](mvp_storage_decision.md)).
