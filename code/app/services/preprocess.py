from __future__ import annotations

import re


def clean_resume_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"(\w)\s+(\w)", r"\1 \2", text)

    text = text.lower()

    text = re.sub(r"\b(summary|objective|resume)\b", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)


    # remove weird OCR symbols
    text = re.sub(r"[^\w\s@.+\-(),:/]", " ", text)

    # fix broken spacing
    text = re.sub(r"\s{2,}", " ", text)

    
    return text.strip()