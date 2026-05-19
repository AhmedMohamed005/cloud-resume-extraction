#!/usr/bin/env python3
"""Build dataset inventory, label map, and deterministic train/val/test splits."""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def normalize_label(raw: str) -> str:
    label = raw.strip().lower().replace("_", " ")
    label = re.sub(r"\bresumes?\b", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def canonical_slug(raw: str) -> str:
    normalized = normalize_label(raw)
    return normalized.replace(" ", "_")


def load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Override YAML must be a mapping: raw_label -> canonical_label")

    clean: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("Override keys and values must be strings")
        clean[k.strip()] = v.strip().lower().replace(" ", "_")
    return clean


def split_counts(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 2:
        return n, 0, 0

    val = int(round(n * val_ratio))
    test = int(round(n * test_ratio))
    train = n - val - test

    # Keep at least one sample in train for any class with >= 3 samples.
    while train < 1 and (val > 0 or test > 0):
        if val >= test and val > 0:
            val -= 1
        elif test > 0:
            test -= 1
        train = n - val - test

    # Try to keep non-empty val/test for medium/large classes.
    if n >= 10:
        if val == 0:
            val = 1
            train -= 1
        if test == 0 and train > 1:
            test = 1
            train -= 1

    return train, val, test


def build_inventory(dataset_root: Path, overrides: dict[str, str]):
    rows = []
    label_variants: dict[str, list[str]] = defaultdict(list)

    for label_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        raw_label = label_dir.name
        canonical = overrides.get(raw_label, canonical_slug(raw_label))
        label_variants[canonical].append(raw_label)

        for file_path in sorted(label_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() != ".pdf":
                continue

            rows.append(
                {
                    "file_path": str(file_path).replace("\\", "/"),
                    "raw_label": raw_label,
                    "canonical_label": canonical,
                    "file_name": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                }
            )

    return rows, label_variants


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def make_splits(rows: list[dict], seed: int, val_ratio: float, test_ratio: float):
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_label"]].append(row)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for label in sorted(grouped.keys()):
        items = grouped[label][:]
        rng.shuffle(items)

        n_train, n_val, n_test = split_counts(len(items), val_ratio, test_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val : n_train + n_val + n_test])

    return train, val, test


def write_label_map(path: Path, variants: dict[str, list[str]]) -> None:
    output = {
        "canonical_label_map": {
            canonical: sorted(set(raw_labels)) for canonical, raw_labels in sorted(variants.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False, allow_unicode=False)


def write_summary(path: Path, rows: list[dict], train: list[dict], val: list[dict], test: list[dict]) -> None:
    counts = Counter(r["canonical_label"] for r in rows)
    lines = [
        "# Dataset Preparation Summary",
        "",
        f"- Total PDFs: {len(rows)}",
        f"- Canonical labels: {len(counts)}",
        f"- Train: {len(train)}",
        f"- Validation: {len(val)}",
        f"- Test: {len(test)}",
        "",
        "## Label Counts",
        "",
        "| Canonical Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(counts.items()):
        lines.append(f"| {label} | {count} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare inventory and splits for resume PDFs")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/Resumes PDF"),
        help="Root directory that contains label folders with PDF files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Directory where generated artifacts will be stored",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("code/config/label_overrides.yaml"),
        help="Optional YAML mapping raw folder name -> canonical label",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_root}")

    overrides = load_overrides(args.overrides)
    rows, variants = build_inventory(args.dataset_root, overrides)

    if not rows:
        raise RuntimeError("No PDF files were found. Check dataset-root path.")

    inventory_path = args.output_dir / "dataset_inventory.csv"
    label_map_path = args.output_dir / "label_map.yaml"
    train_path = args.output_dir / "train.csv"
    val_path = args.output_dir / "val.csv"
    test_path = args.output_dir / "test.csv"
    summary_path = args.output_dir / "dataset_summary.md"

    headers = ["file_path", "raw_label", "canonical_label", "file_name", "size_bytes"]
    write_csv(inventory_path, rows, headers)
    write_label_map(label_map_path, variants)

    train, val, test = make_splits(rows, args.seed, args.val_ratio, args.test_ratio)
    write_csv(train_path, train, headers)
    write_csv(val_path, val, headers)
    write_csv(test_path, test, headers)

    write_summary(summary_path, rows, train, val, test)

    print(f"Inventory written to: {inventory_path}")
    print(f"Label map written to: {label_map_path}")
    print(f"Splits written to: {train_path}, {val_path}, {test_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
