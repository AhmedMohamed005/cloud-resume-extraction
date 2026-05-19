# Supervisor checklist — evidence index

Use this file to map approval items to artifacts in the repository.

| Checklist theme | Evidence location |
|-----------------|-------------------|
| Problem & scope | [Concept_Note_Resume_Extraction.txt](Concept_Note_Resume_Extraction.txt), [Resume_Extraction_Full_Execution_Plan.md](Resume_Extraction_Full_Execution_Plan.md) |
| Proposal vs implementation | [PROPOSAL_TECHNICAL_ALIGNMENT.md](PROPOSAL_TECHNICAL_ALIGNMENT.md) |
| Full project narrative | [PROJECT_MANUAL.md](PROJECT_MANUAL.md), root [README.md](../README.md) |
| Architecture | [architecture.md](architecture.md), [Architecture_Sketch_Resume_Extraction.txt](Architecture_Sketch_Resume_Extraction.txt) |
| API | [api_guide.md](api_guide.md), [POSTMAN_AND_REMOTE_TESTING.md](POSTMAN_AND_REMOTE_TESTING.md), `code/postman/Resume_Extraction.postman_collection.json`, `/docs` when running |
| Parsing / preprocessing | [extraction_quality_report.md](extraction_quality_report.md), `reports/parser_benchmark.json`, [preprocessing_rules.md](preprocessing_rules.md), `code/app/services/parser.py` |
| WP1 data inventory | `manifests/dataset_inventory.csv`, `manifests/label_map.yaml`, `manifests/train.csv` (and val/test), `manifests/README.md` |
| Model & evaluation | `models/resume-ner/MODEL_CARD.md`, `code/scripts/evaluate_baseline_vs_lora.py`, `reports/*.json` (after eval), [final_evaluation_report.md](final_evaluation_report.md) |
| Cloud | `Dockerfile`, [deployment.md](deployment.md), `.env.example` |
| Demo | [demo_script.md](demo_script.md) |
| Risks | [risks.md](risks.md) |
| Storage (MVP) | [mvp_storage_decision.md](mvp_storage_decision.md) |

Fill in dates, team IDs, and screenshots of EC2/CloudWatch as required by the course.
