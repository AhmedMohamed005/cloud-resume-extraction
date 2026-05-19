from __future__ import annotations

import fitz

from app.services.parser import (
    ParsedText,
    _quality_flag,
    extract_text_from_pdf_bytes,
    parser_confidence_from_quality,
)


def test_quality_flag_short_text_is_low() -> None:
    assert _quality_flag("hi") == "low_text"


def test_quality_flag_substantial_text_ok() -> None:
    t = "Word " * 20
    assert _quality_flag(t) == "ok"


def test_parser_confidence_maps_quality_flags() -> None:
    assert parser_confidence_from_quality("ok") == 1.0
    assert parser_confidence_from_quality("low_text") == 0.62
    assert parser_confidence_from_quality("unknown_flag") == 0.55


def test_extract_embedded_pdf_prefers_pymupdf() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Senior Engineer\nPython AWS Docker\n" * 5)
    buf = doc.tobytes()
    out = extract_text_from_pdf_bytes(buf)
    assert isinstance(out, ParsedText)
    assert out.parser_used == "pymupdf"
    assert out.quality_flag == "ok"
    assert "Python" in out.text
