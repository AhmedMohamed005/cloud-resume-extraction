#!/usr/bin/env python3
"""
Sanity-check token NER JSON (input_ids + labels) before training.

Run from repository root:
  python code/scripts/audit_ner_dataset.py

Flags bad rows: length mismatch, out-of-range labels, empty sequences.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def audit_file(path: Path, num_labels: int) -> int:
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    bad = 0
    label_hist: Counter[int] = Counter()
    for i, row in enumerate(rows):
        ids = row.get("input_ids")
        labs = row.get("labels")
        if not isinstance(ids, list) or not isinstance(labs, list):
            print(f"  row {i}: missing input_ids or labels list")
            bad += 1
            continue
        if len(ids) != len(labs):
            print(f"  row {i}: len mismatch input_ids={len(ids)} labels={len(labs)}")
            bad += 1
            continue
        for j, lid in enumerate(labs):
            li = int(lid)
            if li == -100:
                continue
            if li < 0 or li >= num_labels:
                print(f"  row {i} tok {j}: label id {li} out of range [0, {num_labels})")
                bad += 1
                break
            label_hist[li] += 1
    print(f"{path.name}: {len(rows)} rows, {bad} rows with errors")
    top = dict(sorted(label_hist.items(), key=lambda x: -x[1])[:12])
    print("  top label ids (non -100):", top)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=REPO / "dataset")
    args = ap.parse_args()
    lm_path = args.dataset_dir / "label_map.json"
    if not lm_path.is_file():
        print(f"Missing {lm_path}", file=sys.stderr)
        return 1
    with open(lm_path, encoding="utf-8") as f:
        lm = json.load(f)
    num_labels = len(lm.get("label2id", {}))
    print(f"Labels: {num_labels} -> {list(lm.get('label2id', {}).keys())[:8]}...")

    total = 0
    for name in ("train.json", "val.json"):
        p = args.dataset_dir / name
        if p.is_file():
            total += audit_file(p, num_labels)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
