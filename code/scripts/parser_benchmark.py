#!/usr/bin/env python3
"""Random-sample benchmark for PDF text extraction (WP2)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from app.services.parser import extract_text_from_pdf_bytes  # noqa: E402


def _synthetic_embedded_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Senior Engineer\nPython AWS Docker\n" * 5)
    return doc.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=ROOT / "dataset" / "Resumes PDF")
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports" / "parser_benchmark.json",
        help="Write machine-readable summary (use --no-report-json to skip).",
    )
    ap.add_argument("--no-report-json", action="store_true")
    ap.add_argument(
        "--self-check",
        action="store_true",
        default=True,
        help="Run synthetic embedded-text PDF through parser (default: on).",
    )
    ap.add_argument("--no-self-check", action="store_false", dest="self_check")
    args = ap.parse_args()

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
    }

    if args.self_check:
        try:
            out = extract_text_from_pdf_bytes(_synthetic_embedded_pdf_bytes())
            report["synthetic_embedded_text"] = {
                "ok": True,
                "parser_used": out.parser_used,
                "char_count": len(out.text.strip()),
                "quality_flag": out.quality_flag,
            }
        except Exception as exc:  # noqa: BLE001 — benchmark harness
            report["synthetic_embedded_text"] = {"ok": False, "error": str(exc)}

    if not args.dataset_root.is_dir():
        print(f"Dataset root not found: {args.dataset_root}", file=sys.stderr)
        report["dataset_note"] = "dataset_root missing — skipped corpus sample"
        _write_report(args, report)
        return 1

    pdfs = list(args.dataset_root.rglob("*.pdf"))
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        report["dataset_note"] = "no PDFs under dataset_root"
        _write_report(args, report)
        return 1

    rng = random.Random(args.seed)
    take = min(args.sample, len(pdfs))
    chosen = rng.sample(pdfs, take)

    ok = 0
    chars: list[int] = []
    parsers: list[str] = []
    errors: Counter[str] = Counter()
    error_samples: list[dict[str, str]] = []

    for p in chosen:
        try:
            data = p.read_bytes()
            out = extract_text_from_pdf_bytes(data)
            if out.text.strip():
                ok += 1
                chars.append(len(out.text.strip()))
                parsers.append(out.parser_used)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).split("\n")[0][:200]
            errors[msg] += 1
            if len(error_samples) < 8:
                error_samples.append({"file": str(p.relative_to(args.dataset_root)), "error": msg})

    rate = ok / take if take else 0.0
    avg_c = sum(chars) / len(chars) if chars else 0.0

    report.update(
        {
            "corpus_pdf_count": len(pdfs),
            "sampled": take,
            "non_empty_text_rate": round(rate, 4),
            "avg_chars_on_success": round(avg_c, 1),
            "parser_usage_counts": dict(Counter(parsers)),
            "error_counts_top": dict(errors.most_common(5)),
            "error_samples": error_samples,
        }
    )

    if rate < 0.95 and errors:
        top = errors.most_common(1)[0][0] if errors else ""
        if "Tesseract" in top or "tesseract" in top.lower():
            report["interpretation"] = (
                "Sampled PDFs appear to need OCR; install Tesseract and pytesseract to raise success rate."
            )
        elif rate == 0.0:
            report["interpretation"] = (
                "No successes in sample — likely scanned PDFs without OCR, or parse failures."
            )

    print(f"Sampled: {take} / {len(pdfs)} PDFs")
    print(f"Non-empty text rate: {rate:.1%}")
    print(f"Avg chars (successes): {avg_c:.0f}")
    print("Parser usage:", dict(Counter(parsers)))
    if errors:
        print("Top errors:", dict(errors.most_common(3)), file=sys.stderr)

    _write_report(args, report)
    return 0


def _write_report(args: argparse.Namespace, report: dict) -> None:
    if args.no_report_json:
        return
    path: Path = args.report_json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
