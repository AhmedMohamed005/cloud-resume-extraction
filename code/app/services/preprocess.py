from __future__ import annotations
import re

_CLAIRE_ARTIFACT_RE = re.compile(r"Claire", re.IGNORECASE)


def _detect_claire_artifact(text: str) -> bool:
    """Return True if this text has the 'Claire=s' OCR font corruption."""
    occurrences = len(_CLAIRE_ARTIFACT_RE.findall(text))
    if occurrences < 8:
        return False
    word_count = len(text.split())
    if word_count == 0:
        return False
    ratio = occurrences / word_count
    return ratio > 0.04  # more than 4% of words contain "Claire" → artifact


def _repair_claire_artifact(text: str) -> str:
    """
    Replace the 'Claire' OCR artifact with 's'.

    Special case: 'Claire3' -> 'S3' (Amazon S3). The digit stops the
    letter-only lookahead, so we handle digit-suffix cases first.
    """
    # All replacements use re.IGNORECASE so both "Claire" and "CLAIRE" are caught.
    # "JECLAIRECLAIREICA" has all-caps "CLAIRE" mid-word which case-sensitive
    # patterns miss entirely, producing the "Jeclaireclaireica" name bug.
    import re as _re
    I = _re.IGNORECASE

    # 1. Claire + digit first: "Claire3" / "CLAIRE3" -> "S3" or "s3"
    text = _re.sub(r"(?<=[A-Za-z])Claire(?=[0-9])", "s", text, flags=I)
    text = _re.sub(r"\bClaire(?=[0-9])", "S", text, flags=I)

    # 2. Mid-word Claire (preceded by a letter) -> lowercase 's'
    #    e.g. "MaClaireter"/"MECLAIRICA" -> "Master"/"MEssICA"
    #    extract_name's capitalize() will fix the casing afterward.
    text = _re.sub(r"(?<=[A-Za-z])Claire(?=[A-Za-z])", "s", text, flags=I)
    text = _re.sub(r"(?<=[A-Za-z])Claire(?=[^A-Za-z0-9])", "s", text, flags=I)
    text = _re.sub(r"(?<=[A-Za-z])Claire$", "s", text, flags=I | _re.MULTILINE)

    # 3. Start-of-word Claire -> uppercase 'S'
    #    e.g. "Clairecience" -> "Science"
    text = _re.sub(r"\bClaire(?=[A-Za-z])", "S", text, flags=I)

    # 4. Remaining standalone "Claire" / "CLAIRE" (real surname) -- leave alone
    return text


# ── General OCR normalisations ─────────────────────────────────────────────────

def normalize_ocr(text: str) -> str:
    """
    Fix common OCR misreads that appear consistently across many PDFs.
    Add new entries here as you discover them from the debug output.
    """
    # Run the Claire artifact repair first, before other substitutions
    if _detect_claire_artifact(text):
        text = _repair_claire_artifact(text)

    replacements = {
        # SQL misreads (very common — OCR confuses O and 0, Q and O)
        "SOL":        "SQL",
        "sol ":       "sql ",         # lowercase with trailing space (mid-sentence)
        "ClaireQL":   "SQL",          # residual after Claire repair on some fonts
        # Punctuation / formatting
        "t-sql":      "tsql",
        "I ve":       "I've",
        "i ve":       "i've",
        "Bachelor s": "Bachelor's",
        "bachelor s": "bachelor's",
        "Master s":   "Master's",
        "master s":   "master's",
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    return text


# ── Main cleaning function ────────────────────────────────────────────────────

def clean_resume_text(raw_text: str) -> str:
    """
    Full cleaning pipeline:
    1. Normalise line endings and non-breaking spaces
    2. Repair OCR font artifacts (Claire=s, SOL=SQL, etc.)
    3. Strip non-resume characters while keeping programming symbols
    4. Collapse excessive whitespace
    """
    if not raw_text:
        return ""

    # Step 1: normalise whitespace characters
    text = raw_text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 2: OCR artifact repair (must happen on raw text before symbol stripping)
    text = normalize_ocr(text)

    # Step 3: strip characters that are never meaningful in a resume
    # Keep: word chars, whitespace, email/url chars (@, .), programming symbols
    # (+, #, -, /), parentheses, commas, colons
    text = re.sub(r"[^\w\s@.+#\-/(),:']", " ", text)  # apostrophe kept for contractions and possessives

    # Step 4: collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()