"""
Reads all .txt files from the cleaned/ directory and scores each one.
Outputs three files:
  - annotation_ready.txt   : high quality, ready to label in Label Studio
  - review_needed.txt      : borderline, check manually before labeling
  - skip.txt               : too noisy, not worth labeling
"""
from __future__ import annotations
import os
import re
import json
from pathlib import Path

INPUT_DIR  = "../cleaned"    
OUTPUT_DIR = "../annotation" 
os.makedirs(OUTPUT_DIR, exist_ok=True)


def score_text_quality(text: str) -> dict:
    """
    Score a cleaned resume text on several quality dimensions.
    Returns a dict with individual scores and a final verdict.

    Scoring logic:
    - Word error rate proxy: ratio of very short or garbled tokens
    - Has recognisable section headers (EXPERIENCE, EDUCATION, SKILLS)
    - Has an email address (basic sanity check)
    - Text is not too short (< 200 chars = almost certainly failed extraction)
    - Two-column scramble detection: many lines with content on both halves
    """
    lines  = [l.strip() for l in text.splitlines() if l.strip()]
    tokens = text.split()

    def is_garbled(token: str) -> bool:
        t = re.sub(r"[^a-zA-Z]", "", token)
        if len(t) < 4:
            return False
        vowels = sum(1 for c in t.lower() if c in "aeiou")
        if vowels == 0:
            return True
        # consonant cluster of 4+ (not normal English)
        if re.search(r"[^aeiouAEIOU]{5,}", t):
            return True
        return False

    garbled_count = sum(1 for t in tokens if is_garbled(t))
    garbled_ratio = garbled_count / max(len(tokens), 1)

    text_lower = text.lower()
    headers_found = []
    for header in ["experience", "education", "skills", "summary", "work"]:
        if header in text_lower:
            headers_found.append(header)

    has_email = bool(re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text))

    text_length = len(text)

    long_lines = sum(1 for l in lines if len(l) > 120)
    scramble_ratio = long_lines / max(len(lines), 1)

    issues = []

    if garbled_ratio > 0.20:
        issues.append(f"high garble ratio ({garbled_ratio:.0%})")
    if len(headers_found) < 2:
        issues.append(f"few section headers ({headers_found})")
    if not has_email:
        issues.append("no email found")
    if text_length < 300:
        issues.append(f"very short ({text_length} chars)")
    if scramble_ratio > 0.25:
        issues.append(f"possible two-column scramble ({scramble_ratio:.0%} long lines)")

    if len(issues) == 0:
        verdict = "READY"
    elif len(issues) == 1 and "garble" not in issues[0] and "scramble" not in issues[0]:
        verdict = "REVIEW"
    elif len(issues) >= 3 or any("scramble" in i or "short" in i for i in issues):
        verdict = "SKIP"
    else:
        verdict = "REVIEW"

    return {
        "garbled_ratio":   round(garbled_ratio, 3),
        "headers_found":   headers_found,
        "has_email":       has_email,
        "text_length":     text_length,
        "scramble_ratio":  round(scramble_ratio, 3),
        "issues":          issues,
        "verdict":         verdict,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    results = {"READY": [], "REVIEW": [], "SKIP": []}
    scores  = {}

    txt_files = sorted(Path(INPUT_DIR).glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    print(f"Scanning {len(txt_files)} files...\n")

    for path in txt_files:
        text   = path.read_text(encoding="utf-8", errors="ignore")
        result = score_text_quality(text)
        verdict = result["verdict"]

        results[verdict].append(path.name)
        scores[path.name] = result

        # One-line summary per file
        issues_str = ", ".join(result["issues"]) if result["issues"] else "none"
        print(f"[{verdict:6s}] {path.name:40s}  garble={result['garbled_ratio']:.0%}  issues: {issues_str}")

    # Write output files
    for verdict, files in results.items():
        out = Path(OUTPUT_DIR) / f"{verdict.lower()}.txt"
        out.write_text("\n".join(files), encoding="utf-8")

    # Write full scores as JSON for reference
    scores_path = Path(OUTPUT_DIR) / "quality_scores.json"
    scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    # Summary
    print(f"\n{'─'*60}")
    print(f"  READY  (label these first):  {len(results['READY'])}")
    print(f"  REVIEW (check manually):     {len(results['REVIEW'])}")
    print(f"  SKIP   (too noisy):          {len(results['SKIP'])}")
    print(f"{'─'*60}")
    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  annotation_ready.txt  — import these into Label Studio")
    print("  review_needed.txt     — inspect these before labeling")
    print("  skip.txt              — set aside")
    print("  quality_scores.json   — full scores for all files")


if __name__ == "__main__":
    main()