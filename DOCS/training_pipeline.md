# Training pipeline

## Data

- Annotated JSON for token classification: `dataset/train.json`, `dataset/val.json`, `dataset/label_map.json`.
- Raw PDFs live under `dataset/Resumes PDF/` (gitignored locally).

## Dataset sanity check (before training)

```bash
python code/scripts/audit_ner_dataset.py
```

Fix any **length mismatches** or **out-of-range label ids** reported there; bad rows dominate poor NER quality.

## Label inventory and splits

```bash
py code/scripts/prepare_dataset_manifests.py \
  --dataset-root "dataset/Resumes PDF" \
  --output-dir manifests
```

Committed copies live under `manifests/` (see `manifests/README.md`). You may also write to `dataset/processed/` locally; those files stay gitignored with the corpus.

## Fine-tune LoRA

From `code/scripts` (paths assume JSON already in `dataset/`):

```bash
cd code/scripts
py train.py
```

Checkpoints and final adapter: `models/resume-ner/`. Training also writes **`models/resume-ner/final/merged/`** for the API.

### Optional environment overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `RESUME_EPOCHS` | 8 | Max epochs |
| `RESUME_EARLY_STOPPING_PATIENCE` | 2 | Stop if val F1 does not improve |
| `RESUME_BATCH_SIZE` | 8 | Per-device batch (lower if OOM) |
| `RESUME_LR` | 2e-4 | Learning rate |
| `RESUME_LORA_R` | 16 | LoRA rank |
| `RESUME_LORA_ALPHA` | 2×r | LoRA scaling |
| `RESUME_LORA_TARGET_MODULES` | `query,value,key` | Comma-separated module suffixes |
| `RESUME_WARMUP_RATIO` | 0.06 | Warmup fraction of steps |
| `RESUME_SAVE_TOTAL_LIMIT` | 4 | On-disk checkpoint cap |

## Baseline vs fine-tuned metrics

```bash
py code/scripts/evaluate_baseline_vs_lora.py
```

See `models/resume-ner/MODEL_CARD.md` and `DOCS/final_evaluation_report.md`.
