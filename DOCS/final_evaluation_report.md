# Final evaluation report (template)

## How to generate metrics

From the repository root (with `transformers`, `peft`, `torch`, `datasets` installed):

```bash
py code/scripts/evaluate_baseline_vs_lora.py
```

Artifacts:

- `reports/baseline_metrics.json` — BERT + **untrained** token classification head (same label space as fine-tuned model).
- `reports/fine_tuned_metrics.json` — merged LoRA checkpoint from `models/resume-ner/final/`.

Each file includes **`token_accuracy_excl_padding`** (token ID match rate) alongside **`overall_f1`** (strict entity-level F1 from `seqeval`). If entity F1 is `0` but you see non-trivial token accuracy, spans are still misaligned; if both are very low, re-check `dataset/label_map.json` vs training labels and consider retraining.

## Fill in after running the script

| Metric | Baseline | Fine-tuned (LoRA) |
|--------|----------|-------------------|
| Overall entity F1 | _paste `overall_f1`_ | _paste `overall_f1`_ |
| Macro F1 (if present) | _from `macro_avg`_ | _from `macro_avg`_ |

### Per-entity F1 (fine-tuned)

_Copy rows from `fine_tuned_metrics.json` → `per_entity`._

## Functional / API metrics

- JSON schema validity rate on held-out PDFs: _run test suite + manual sample_
- p50 / p95 latency on EC2: _fill from load test_

## Notes

- Baseline is intentionally weak (random head); the comparison shows the value of LoRA fine-tuning on resume NER.
