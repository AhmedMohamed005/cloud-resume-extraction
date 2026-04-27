"""
Label Studio NER tasks expect a JSON list where each item has:
    { "text": "the full resume text" }

Run AFTER filter_for_annotation.py:
    python prepare_label_studio.py

This reads annotation/ready.txt (the list of filenames to label),
loads each .txt file from cleaned/, and writes one JSON file per
batch of 50 resumes — manageable chunks for annotation sessions.
"""
from __future__ import annotations
import os
import json
from pathlib import Path

CLEANED_DIR    = "../cleaned"
ANNOTATION_DIR = "../annotation"
READY_LIST     = os.path.join(ANNOTATION_DIR, "ready.txt")
BATCH_SIZE     = 50


def main():
    ready_path = Path(READY_LIST)
    if not ready_path.exists():
        print(f"Run filter_for_annotation.py first — {READY_LIST} not found.")
        return

    ready_files = [
        line.strip()
        for line in ready_path.read_text().splitlines()
        if line.strip()
    ]

    if not ready_files:
        print("No READY files found. Check your cleaned/ directory and re-run the filter.")
        return

    print(f"Preparing {len(ready_files)} files for Label Studio...\n")

    tasks = []
    skipped = []

    for filename in ready_files:
        txt_path = Path(CLEANED_DIR) / filename
        if not txt_path.exists():
            print(f"  [WARN] File not found: {txt_path}")
            skipped.append(filename)
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            skipped.append(filename)
            continue

        # Label Studio NER task format
        tasks.append({
            "data": {
                "text": text,
                "meta": {
                    "filename": filename,
                    "char_count": len(text),
                }
            }
        })

    # Split into batches
    batches = [tasks[i:i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]

    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    for i, batch in enumerate(batches, 1):
        out_path = Path(ANNOTATION_DIR) / f"label_studio_batch_{i:02d}.json"
        out_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Batch {i:02d}: {len(batch)} tasks → {out_path}")

    print(f"\nDone. {len(tasks)} tasks written across {len(batches)} batch file(s).")
    if skipped:
        print(f"Skipped {len(skipped)} files (not found or empty): {skipped[:5]}")

    print("\n── Next steps ────────────────────────────────────────────────")
    print("1. Open Label Studio:  label-studio start")
    print("2. Create a new project → choose 'Named Entity Recognition'")
    print("3. In Label config, set these labels:")
    print("     NAME    SKILL    EDUCATION    EXPERIENCE    COMPANY    DATE")
    print("4. Import →  Upload JSON →  select label_studio_batch_01.json")
    print("5. Start annotating. Aim for 50 tasks minimum before training.")
    print("──────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()