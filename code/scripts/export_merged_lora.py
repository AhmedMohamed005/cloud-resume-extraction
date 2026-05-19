#!/usr/bin/env python3
"""
Merge LoRA adapters + classifier into a single HuggingFace TokenClassification model.

This avoids runtime PEFT merge quirks and loads cleanly via AutoModelForTokenClassification.
Default output: models/resume-ner/final/merged/ (alongside adapter files in final/).

Usage (repo root):
  python code/scripts/export_merged_lora.py
  python code/scripts/export_merged_lora.py --adapter-dir models/resume-ner/final --out-dir models/resume-ner/final/merged
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=REPO_ROOT / "models" / "resume-ner" / "final",
        help="Directory with adapter_model.safetensors + adapter_config.json + label_map.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Merged model output (default: <adapter-dir>/merged)",
    )
    args = ap.parse_args()
    adapter_dir: Path = args.adapter_dir
    out_dir: Path = args.out_dir if args.out_dir else adapter_dir / "merged"

    lm = adapter_dir / "label_map.json"
    if not lm.is_file():
        print(f"Missing {lm}", file=sys.stderr)
        return 1
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        print(f"Missing adapter weights in {adapter_dir}", file=sys.stderr)
        return 1

    with open(lm, encoding="utf-8") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    print("Loading base BERT + adapter, merging...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    peft = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    shutil.copy2(lm, out_dir / "label_map.json")

    print(f"Wrote merged model to {out_dir}")
    print("Inference will prefer this directory when config.json is present (see ner_engine).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
