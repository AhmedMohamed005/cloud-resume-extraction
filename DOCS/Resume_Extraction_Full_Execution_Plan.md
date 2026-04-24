# Resume Information Extraction - Full Execution Plan

## 1) Project Goal
Build an end-to-end cloud-based system that accepts a resume PDF and returns structured JSON fields (minimum: name, skills, education; target: add experience and contact details), aligned with your proposal template, concept note, and architecture sketch.

## 2) Current Inputs and Constraints
- Dataset location: dataset/Resumes PDF
- Total class folders: 97
- Total files: 8905
- File format: PDF only
- Constraint found: many labels are duplicated as variants such as "Accountant" and "Accountant resumes"

## 3) Definition of Done
The project is complete when all items below are true:
1. Working API endpoint processes uploaded PDF and returns valid JSON.
2. Pipeline runs: PDF parsing -> preprocessing -> model inference -> JSON formatting.
3. Baseline model and LoRA fine-tuned model are both evaluated.
4. Evaluation report includes Precision, Recall, F1 (overall and per field).
5. Deployed on AWS EC2 with logging and basic monitoring.
6. Demo flow works reliably with unseen resumes.
7. Documentation package is complete (API, architecture, training, deployment, risks).

## 4) Scope Control
### MVP scope (must have)
- Input: PDF resume upload
- Output JSON keys: name, skills, education
- One production API endpoint: POST /extract
- Baseline vs fine-tuned comparison with measurable metrics
- Cloud deployment on EC2 (single instance)

### Stretch scope (if time allows)
- Additional keys: experience, email, phone, projects
- Better handling of complex two-column resumes
- S3 storage for uploads/results
- Auto Scaling policy and stronger observability

### First items to cut if schedule slips
1. Extra fields beyond MVP
2. Advanced parser fallbacks
3. S3 archival and optional analytics

## 5) Target Architecture (Implementation View)
1. Client (Postman or web client) sends PDF to FastAPI.
2. FastAPI validates file type/size and generates request id.
3. Parser module extracts raw text (PyMuPDF first, pdfminer fallback).
4. Preprocessor normalizes text and removes parsing noise.
5. NLP inference module predicts entities from cleaned text.
6. JSON formatter maps entities to required schema.
7. Response is returned; metadata and logs are stored.
8. Optional: PDF and output stored in S3.

## 6) Work Breakdown Structure (WBS)

### WP1 - Data Audit and Label Standardization
Deliverables:
- dataset_inventory.csv
- label_map.yaml
- train/val/test split manifest files

Tasks:
1. Enumerate all class folders and file counts.
2. Merge label variants into canonical labels using mapping rules.
3. Remove corrupt/empty PDFs and duplicate files.
4. Produce stratified train/val/test split (for class-level analyses if needed).

Starter canonical mapping examples:
- Accountant + Accountant resumes -> accountant
- DevOps Engineer + DevOps Engineer resumes -> devops_engineer
- Data Science + data science resumes + DataScience -> data_science
- Information Technology + IT -> information_technology (confirm with team before final merge)

Acceptance criteria:
- Every file has one canonical label.
- Mapping decisions are documented and reproducible.

### WP2 - PDF Parsing and Text Quality Layer
Deliverables:
- parser.py
- extraction_quality_report.md

Tasks:
1. Implement parser strategy:
   - Primary: PyMuPDF
   - Fallback: pdfminer.six
2. Add extraction sanity checks (min chars, invalid symbol ratio, blank pages).
3. Add parser confidence flags in output metadata.
4. Build a small benchmark over 150 random resumes.

Acceptance criteria:
- >= 95% of sampled files produce non-empty meaningful text.
- Fallback path works for parser failures.

### WP3 - Text Preprocessing Pipeline
Deliverables:
- preprocess.py
- preprocessing_rules.md

Tasks:
1. Normalize whitespace and line breaks.
2. Fix encoding artifacts.
3. Segment likely resume sections (skills, education, experience).
4. Keep both raw and cleaned text for traceability.

Acceptance criteria:
- Preprocessing is deterministic.
- Section heuristics improve extraction quality on validation subset.

### WP4 - Baseline Information Extraction
Deliverables:
- baseline_inference.py
- baseline_metrics.json

Tasks:
1. Select baseline transformer (token classification or instruction extraction pattern).
2. Define target schema and entity tags.
3. Run baseline on validation subset.
4. Record precision, recall, F1 per entity and macro average.

Acceptance criteria:
- Reproducible baseline metrics report.
- Output schema validation pass rate >= 98%.

### WP5 - LoRA Fine-Tuning
Deliverables:
- train_lora.py
- model card
- fine_tuned_metrics.json

Tasks:
1. Prepare annotation format for fields (BIO tags or structured generation labels).
2. Configure PEFT LoRA parameters.
3. Train with early stopping and checkpointing.
4. Compare against baseline.

Acceptance criteria:
- Fine-tuned model improves F1 by agreed target (recommend +8% or more macro F1 vs baseline).
- Inference latency remains within API target.

### WP6 - API and Integration
Deliverables:
- FastAPI service with POST /extract
- OpenAPI docs
- Postman collection

Tasks:
1. Implement upload validation (type, size, timeout handling).
2. Integrate parser, preprocessing, model, formatter.
3. Return structured JSON with confidence and processing metadata.
4. Add structured error responses.

Acceptance criteria:
- End-to-end API pass rate >= 99% on test suite.
- Median response time target achieved.

### WP7 - Cloud Deployment (AWS)
Deliverables:
- Dockerfile
- deployment guide
- EC2 deployment scripts

Tasks:
1. Containerize API and model runtime.
2. Deploy to EC2 with secure inbound rules.
3. Configure IAM least-privilege roles.
4. Enable CloudWatch logs/metrics.
5. Optional: add S3 storage for files/results.

Acceptance criteria:
- Public endpoint reachable and stable.
- Logs and health checks available.

### WP8 - Evaluation, Demo, and Documentation
Deliverables:
- final_evaluation_report.md
- demo_script.md
- final_architecture_diagram
- supervisor checklist evidence

Tasks:
1. Functional tests: valid PDF, invalid file, malformed PDF, large file.
2. Model tests: precision, recall, F1 per field.
3. System tests: latency, uptime under small load.
4. Prepare 5-10 step live demo script.

Acceptance criteria:
- Demo completes without manual recovery.
- All required documents submitted.

## 7) Recommended Repository Structure

code/
- app/
  - main.py
  - routers/extract.py
  - services/parser.py
  - services/preprocess.py
  - services/inference.py
  - services/formatter.py
  - schemas/output_schema.py
- training/
  - data_prep/
  - baseline/
  - lora/
- tests/
  - test_api.py
  - test_parser.py
  - test_inference.py
- docker/
- scripts/
- reports/

DOCS/
- Resume_Extraction_Full_Execution_Plan.md
- architecture.md
- api_guide.md
- evaluation_report.md

dataset/
- Resumes PDF/
- processed/
- splits/
- annotations/

## 8) Timeline (10 Weeks)
Week 1:
- Confirm scope, schema, success metrics, and role ownership.
- Complete dataset audit and label mapping draft.

Week 2:
- Finalize canonical labels and splits.
- Implement parser v1 and quality checks.

Week 3:
- Implement preprocessing rules and section heuristics.
- Build baseline extraction pipeline.

Week 4:
- Baseline evaluation and error analysis.
- Freeze training dataset format for LoRA.

Week 5:
- First LoRA training run.
- Hyperparameter tuning iteration 1.

Week 6:
- Hyperparameter tuning iteration 2.
- Final model candidate selection.

Week 7:
- FastAPI full integration and schema validation.
- Postman collection and negative test cases.

Week 8:
- Dockerization and EC2 deployment.
- CloudWatch logging and health endpoint.

Week 9:
- Full system testing and latency optimization.
- Produce final metrics tables and charts.

Week 10:
- Demo rehearsal, final report, and supervisor checklist package.

## 9) Team Role Split (7 Members)
1. AI Lead + System Integration: model strategy, final merge decisions, release gate.
2. Data Engineer: dataset audit, label mapping, split generation.
3. Parser Engineer: PDF extraction pipeline and fallback handling.
4. NLP Preprocessing Engineer: cleaning, sectioning, normalization.
5. Training Engineer: baseline + LoRA training and experiment tracking.
6. Evaluation Engineer: metrics, test harness, error analysis.
7. QA/DevOps Engineer: API tests, Docker, EC2 deployment, monitoring.

## 10) Metrics and Targets
Functional metrics:
- JSON schema validity rate >= 98%
- API success rate >= 99%

Model metrics:
- Precision, Recall, F1 overall and per field
- Improvement target: fine-tuned model > baseline macro F1

System metrics:
- p50 latency <= 2.5s per resume
- p95 latency <= 6.0s per resume
- Error rate <= 1%

## 11) Risk Register and Mitigation
1. Parsing failures on complex PDFs
- Mitigation: parser fallback + quality flags + test corpus expansion.

2. Weak model generalization
- Mitigation: diverse training data, validation by resume style, targeted augmentation.

3. Timeline pressure
- Mitigation: strict MVP gate at Week 4, defer stretch tasks.

4. Cloud cost or instability
- Mitigation: single EC2 for MVP, right-sizing, monitoring alerts.

5. Privacy/ethics concerns
- Mitigation: public/synthetic data only, avoid storing personal data longer than needed.

## 12) Demo Flow (Examiner-Friendly)
1. Upload one unseen PDF resume to POST /extract.
2. Show parser output summary (text length and status).
3. Show cleaned text snippet.
4. Run model inference.
5. Return structured JSON in response.
6. Validate JSON against schema.
7. Show confidence and timing metadata.
8. Compare baseline and fine-tuned result quality on same input.

## 13) Immediate Next Actions (Start This Week)
1. Finalize label_map.yaml for all duplicate folder names.
2. Build parser benchmark script over random 150 PDFs.
3. Lock MVP schema keys and response contract.
4. Implement /extract endpoint skeleton with mocked model output.
5. Set up experiment tracking for baseline and LoRA runs.

## 14) Supervisor Checklist Alignment
This plan directly addresses approval gates:
- Clear problem and user value: covered in Sections 1, 2.
- Realistic scope and MVP fallback: Sections 4, 8.
- Technical depth and course mapping: Sections 5, 6, 7.
- Dataset usability and evaluation measurability: Sections 2, 10.
- Team workload fairness and integration: Section 9.
- Stable end-to-end demo readiness: Sections 12 and 13.
- Cloud contribution is essential, not decorative: Section 7.
