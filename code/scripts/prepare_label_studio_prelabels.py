"""Create Label Studio import batches with rule-based preannotations.

This script reads cleaned resume texts from ../cleaned, uses the existing
heuristic extractor to generate NAME / SKILL / EDUCATION / EXPERIENCE spans,
and writes a Label Studio import JSON that you can upload directly.

Run from the code/ directory:
    python -m scripts.prepare_label_studio_prelabels
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.services.inference import run_mock_inference


CLEANED_DIR = Path("../cleaned")
ANNOTATION_DIR = Path("../annotation")
READY_LIST = ANNOTATION_DIR / "ready.txt"
BATCH_SIZE = 50
MODEL_VERSION = "rule-prelabel-v1"
OUTPUT_PREFIX = "label_studio_preannotated_batch"


def load_ready_files() -> list[str]:
    if READY_LIST.exists():
        return [line.strip() for line in READY_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]

    return [path.name for path in sorted(CLEANED_DIR.glob("*.txt"))]


def _build_pattern(phrase: str) -> re.Pattern[str] | None:
    phrase = phrase.strip()
    if not phrase:
        return None

    tokens = [token for token in re.split(r"\s+", phrase) if token]
    if not tokens:
        return None

    if len(tokens) == 1:
        token = tokens[0]
        if len(re.sub(r"\W", "", token)) <= 1:
            return re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", re.IGNORECASE)
        return re.compile(re.escape(token), re.IGNORECASE)

    parts = [re.escape(token) for token in tokens]
    return re.compile(r"(?<!\w)" + r"[\s\W]+".join(parts) + r"(?!\w)", re.IGNORECASE)


def find_first_span(text: str, phrase: str) -> tuple[int, int, str] | None:
    pattern = _build_pattern(phrase)
    if pattern is None:
        return None

    match = pattern.search(text)
    if not match:
        return None
    start, end = match.span()
    matched_text = text[start:end]
    return start, end, matched_text


def add_span(spans: list[dict], text: str, label: str, phrase: str) -> None:
    span = find_first_span(text, phrase)
    if span is None:
        return

    start, end, matched_text = span
    spans.append(
        {
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": start,
                "end": end,
                "text": matched_text,
                "labels": [label],
            },
        }
    )


def unique_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = item.strip()
        if not value:
            continue
        key = re.sub(r"\s+", " ", value.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def remove_overlaps(results: list[dict]) -> list[dict]:
    if not results:
        return results

    ordered = sorted(
        results,
        key=lambda item: (item["value"]["start"], -(item["value"]["end"] - item["value"]["start"])),
    )
    kept: list[dict] = []
    last_end = -1
    for item in ordered:
        start = item["value"]["start"]
        end = item["value"]["end"]
        if start >= last_end:
            kept.append(item)
            last_end = end
    return kept


def make_prediction(text: str) -> dict:
    profile, confidence, detected = run_mock_inference(text)

    spans: list[dict] = []

    if profile.name:
        add_span(spans, text, "NAME", profile.name)

    for phrase in unique_items(profile.education):
        add_span(spans, text, "EDUCATION", phrase)

    for phrase in unique_items(profile.experience):
        add_span(spans, text, "EXPERIENCE", phrase)

    for phrase in unique_items(profile.skills):
        add_span(spans, text, "SKILL", phrase)

    spans = remove_overlaps(spans)

    return {
        "model_version": MODEL_VERSION,
        "score": confidence,
        "result": spans,
        "detected_sections": detected,
        "final_profile": profile.model_dump(),
    }


def main() -> None:
    ready_files = load_ready_files()
    if not ready_files:
        print("No ready files found. Check annotation/ready.txt or cleaned/.")
        return

    tasks: list[dict] = []
    missing = []

    for filename in ready_files:
        txt_path = CLEANED_DIR / filename
        if not txt_path.exists():
            missing.append(filename)
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            missing.append(filename)
            continue

        prediction = make_prediction(text)
        tasks.append(
            {
                "data": {
                    "text": text,
                    "meta": {
                        "filename": filename,
                        "char_count": len(text),
                    },
                },
                "predictions": [prediction],
            }
        )

    if not tasks:
        print("No tasks were created.")
        return

    batches = [tasks[i : i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for index, batch in enumerate(batches, start=1):
        out_path = ANNOTATION_DIR / f"{OUTPUT_PREFIX}_{index:02d}.json"
        out_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        created_files.append(out_path)
        print(f"Batch {index:02d}: {len(batch)} tasks -> {out_path}")

    print(f"\nCreated {len(tasks)} preannotated tasks across {len(created_files)} file(s).")
    if missing:
        print(f"Skipped {len(missing)} files that were missing or empty.")
        print("Examples:", missing[:5])


if __name__ == "__main__":
    main()