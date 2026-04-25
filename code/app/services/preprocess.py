from __future__ import annotations

import re


def clean_resume_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Keep meaningful programming symbols like +, #, ., -, /.
    text = re.sub(r"[^\w\s@.+#\-/(),:]", " ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()