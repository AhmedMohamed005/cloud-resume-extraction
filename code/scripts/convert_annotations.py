"""
Convert Label Studio JSON export → HuggingFace token classification dataset.

Usage:
    python convert_annotations.py

Input:  annotations.json  (Label Studio export, JSON format)
Output: dataset/train.json and dataset/val.json

What this does:
    1. Reads every annotated task from Label Studio
    2. Tokenizes the text with bert-base-cased tokenizer
    3. Aligns your character-level spans to token-level BIO labels
    4. Splits 80% train / 20% validation
    5. Saves in HuggingFace Dataset format
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
from transformers import AutoTokenizer

INPUT_FILE  = "../../annotation/project-2-auto-annotated.json"
OUTPUT_DIR  = "../../dataset"
MODEL_NAME  = "bert-base-cased"
SEED        = 42
TRAIN_SPLIT = 0.8

# ── Label map ─────────────────────────────────────────────────────────────────
# BIO scheme: B- = beginning of entity, I- = inside entity, O = outside
LABEL2ID = {
    "O":           0,
    "B-NAME":      1, "I-NAME":      2,
    "B-SKILL":     3, "I-SKILL":     4,
    "B-EDUCATION": 5, "I-EDUCATION": 6,
    "B-EXPERIENCE":7, "I-EXPERIENCE":8,
    "B-COMPANY":   9, "I-COMPANY":  10,
    "B-DATE":     11, "I-DATE":     12,
    "B-LOCATION": 13, "I-LOCATION": 14,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_label_studio_export(path: str) -> list[dict]:
    """Load and return all tasks from a Label Studio JSON export."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} tasks from {path}")
    return data


def extract_spans(task: dict) -> tuple[str, list[tuple[int, int, str]]]:
    """
    Extract the text and entity spans from one Label Studio task.

    Returns:
        text  — the raw text string
        spans — list of (start, end, label) tuples from annotations
    """
    text = task["data"]["text"]
    spans = []

    annotations = task.get("annotations", [])
    for ann in annotations:
        if ann.get("was_cancelled"):
            continue
        for result in ann.get("result", []):
            if result.get("type") != "labels":
                continue
            value = result["value"]
            label = value["labels"][0] if value["labels"] else None
            if label and label in LABEL2ID or any(
                label == l.split("-", 1)[-1] for l in LABEL2ID if "-" in l
            ):
                start = value["start"]
                end   = value["end"]
                # Clean up: strip punctuation from end of span
                span_text = text[start:end]
                while span_text and not span_text[-1].isalnum():
                    end -= 1
                    span_text = text[start:end]
                if end > start:
                    spans.append((start, end, label))

    # Remove duplicate/overlapping spans — keep the longest span at each position
    spans = _resolve_overlaps(spans)
    return text, spans


def _resolve_overlaps(spans: list[tuple]) -> list[tuple]:
    """Remove overlapping spans, keeping the longest one."""
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    resolved = []
    last_end = -1
    for start, end, label in spans:
        if start >= last_end:
            resolved.append((start, end, label))
            last_end = end
    return resolved


def convert_to_bio(
    text: str,
    spans: list[tuple[int, int, str]],
    tokenizer,
) -> dict | None:
    """
    Tokenize text and align character spans to BIO token labels.

    Returns a dict with:
        input_ids      — token ids
        attention_mask — 1 for real tokens, 0 for padding
        labels         — BIO label id per token (-100 for special tokens)
    """
    # Tokenize with offset mapping so we can align spans to tokens
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    offset_mapping = encoding["offset_mapping"]
    labels = []

    # Build a character → label lookup from spans
    char_label: dict[int, str] = {}
    for start, end, label in spans:
        for i in range(start, end):
            char_label[i] = label

    prev_label = None
    for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        # Special tokens (CLS, SEP) get -100 (ignored in loss)
        if tok_start == tok_end:
            labels.append(-100)
            prev_label = None
            continue

        # Find which entity label (if any) covers this token
        token_label = None
        for char_pos in range(tok_start, tok_end):
            if char_pos in char_label:
                token_label = char_label[char_pos]
                break

        if token_label is None:
            labels.append(LABEL2ID["O"])
            prev_label = None
        else:
            # BIO: B- if this is the start of a new entity, I- if continuation
            if token_label != prev_label:
                bio_label = f"B-{token_label}"
            else:
                bio_label = f"I-{token_label}"

            if bio_label not in LABEL2ID:
                labels.append(LABEL2ID["O"])
                prev_label = None
            else:
                labels.append(LABEL2ID[bio_label])
                prev_label = token_label

    return {
        "input_ids":       encoding["input_ids"],
        "attention_mask":  encoding["attention_mask"],
        "labels":          labels,
        "text":            text,
    }


def main():
    random.seed(SEED)

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load Label Studio export
    tasks = load_label_studio_export(INPUT_FILE)

    # Convert each task
    examples = []
    skipped  = 0

    for task in tasks:
        text, spans = extract_spans(task)

        # Skip empty texts
        if not text or not text.strip():
            skipped += 1
            continue

        # Skip tasks with no annotations and no cancellation
        # (means the annotator never touched it)
        if not task.get("annotations"):
            skipped += 1
            continue

        example = convert_to_bio(text, spans, tokenizer)
        if example:
            examples.append(example)

    print(f"Converted: {len(examples)} examples  |  Skipped: {skipped}")

    # Count label distribution
    label_counts: dict[str, int] = {}
    for ex in examples:
        for label_id in ex["labels"]:
            if label_id == -100:
                continue
            label_name = ID2LABEL[label_id]
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        bar = "█" * min(count // 5, 40)
        print(f"  {label:20s} {count:5d}  {bar}")

    # Warn if any entity type has very few examples
    print()
    for label_type in ["NAME", "SKILL", "EDUCATION", "EXPERIENCE", "COMPANY"]:
        b_count = label_counts.get(f"B-{label_type}", 0)
        if b_count < 10:
            print(f"  ⚠  {label_type}: only {b_count} B- tokens — consider labeling more")
        else:
            print(f"  ✓  {label_type}: {b_count} entity spans")

    # Train / val split
    random.shuffle(examples)
    split_idx  = int(len(examples) * TRAIN_SPLIT)
    train_data = examples[:split_idx]
    val_data   = examples[split_idx:]

    print(f"\nSplit: {len(train_data)} train  |  {len(val_data)} val")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = Path(OUTPUT_DIR) / "train.json"
    val_path   = Path(OUTPUT_DIR) / "val.json"

    train_path.write_text(json.dumps(train_data, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_data, indent=2), encoding="utf-8")

    # Also save label map for training script
    label_map_path = Path(OUTPUT_DIR) / "label_map.json"
    label_map_path.write_text(
        json.dumps({"label2id": LABEL2ID, "id2label": ID2LABEL}, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved:")
    print(f"  {train_path}  ({len(train_data)} examples)")
    print(f"  {val_path}    ({len(val_data)} examples)")
    print(f"  {label_map_path}")
    print("\nNext step: run  python train.py")


if __name__ == "__main__":
    main()