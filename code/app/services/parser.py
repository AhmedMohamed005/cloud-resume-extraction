"""PDF text extraction with PyMuPDF primary path and OCR fallback."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from shutil import which
from typing import Any

import fitz


PARSER_PYMUPDF = "pymupdf"
PARSER_PYMUPDF_OCR = "pymupdf+ocr"

_OCR_RENDER_ZOOM = 2.0
_ERR_EMPTY_FILE = "Empty file content"
_ERR_NO_TEXT = "No embedded text found in the PDF and OCR also returned empty text."
_ERR_OCR_DEPS = "OCR dependencies are missing. Install with: pip install pytesseract pillow"
_ERR_TESSERACT_MISSING = (
    "Tesseract binary is not installed. On macOS run: brew install tesseract"
)


@dataclass(frozen=True)
class ParsedText:
    text: str
    parser_used: str


def _collapse_text(chunks: list[str]) -> str:
    return "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()


def _extract_embedded_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        parts = [page.get_text("text") for page in doc]
    return _collapse_text(parts)


def _load_ocr_dependencies() -> tuple[Any, Any]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - import availability depends on environment
        raise RuntimeError(_ERR_OCR_DEPS) from exc

    if which("tesseract") is None:
        raise RuntimeError(_ERR_TESSERACT_MISSING)

    return pytesseract, Image


def _extract_text_with_ocr(pdf_bytes: bytes) -> str:
    pytesseract, image_module = _load_ocr_dependencies()

    parts: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            # Render each page as an image and run OCR.
            pix = page.get_pixmap(matrix=fitz.Matrix(_OCR_RENDER_ZOOM, _OCR_RENDER_ZOOM), alpha=False)
            with image_module.open(BytesIO(pix.tobytes("png"))) as image:
                text = pytesseract.image_to_string(image)
            if text and text.strip():
                parts.append(text.strip())

    return _collapse_text(parts)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> ParsedText:
    if not pdf_bytes:
        raise ValueError(_ERR_EMPTY_FILE)

    text = _extract_embedded_text(pdf_bytes)

    if text:
        return ParsedText(text=text, parser_used=PARSER_PYMUPDF)

    ocr_text = _extract_text_with_ocr(pdf_bytes)
    if ocr_text:
        return ParsedText(text=ocr_text, parser_used=PARSER_PYMUPDF_OCR)

    raise ValueError(_ERR_NO_TEXT)
