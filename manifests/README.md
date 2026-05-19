# Committed dataset manifests (WP1)

These files are generated from the local corpus under `dataset/Resumes PDF` (gitignored) so the repo still carries **reproducible inventory and split lists** for supervisors and CI.

Regenerate after adding or relabeling PDFs:

```bash
# From repository root
python code/scripts/prepare_dataset_manifests.py ^
  --dataset-root "dataset/Resumes PDF" ^
  --output-dir manifests ^
  --overrides code/config/label_overrides.yaml
```

- `dataset_inventory.csv` — one row per PDF with path, raw folder label, canonical label, size.
- `label_map.yaml` — canonical label → variant folder names.
- `train.csv` / `val.csv` / `test.csv` — stratified splits (see script `--val-ratio` / `--test-ratio` / `--seed`).
- `dataset_summary.md` — counts per label.

Training JSON (`dataset/train.json`, `val.json`) may still be produced by separate annotation export scripts; these CSVs describe the **raw PDF corpus** alignment with `label_overrides.yaml`.
