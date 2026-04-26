"""PDF text extraction with PyMuPDF primary path and OCR fallback."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from shutil import which
from typing import Any
import fitz

PARSER_PYMUPDF     = "pymupdf"
PARSER_PYMUPDF_OCR = "pymupdf+ocr"

_OCR_RENDER_ZOOM = 3.0   
_ERR_EMPTY_FILE        = "Empty file content"
_ERR_NO_TEXT           = "No embedded text found in the PDF and OCR also returned empty text."
_ERR_OCR_DEPS          = "OCR dependencies are missing. Install with: pip install pytesseract pillow"
_ERR_TESSERACT_MISSING = (
    "Tesseract binary is not installed. On macOS run: brew install tesseract"
)

_TESS_CONFIG = "--oem 3 --psm 6"


@dataclass(frozen=True)
class ParsedText:
    text: str
    parser_used: str


def _collapse_text(chunks: list[str]) -> str:
    return "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()


def _extract_embedded_text(pdf_bytes: bytes) -> str:
    """
    Try to extract text directly embedded in the PDF (no OCR needed).
    This is fast and perfectly accurate when the PDF was created digitally
    (e.g. exported from Word, Google Docs). It returns empty string for
    scanned PDFs or PDFs with image-only pages.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        parts = [page.get_text("text") for page in doc]
    return _collapse_text(parts)


def _load_ocr_dependencies() -> tuple[Any, Any]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(_ERR_OCR_DEPS) from exc
    if which("tesseract") is None:
        raise RuntimeError(_ERR_TESSERACT_MISSING)
    return pytesseract, Image


def _extract_text_with_ocr(pdf_bytes: bytes) -> str:
    """
    Render each PDF page as a high-DPI image and run Tesseract OCR on it.

    Key improvements over the original:
    - Render zoom raised from 2.0 → 3.0 (better character separation at 216 DPI)
    - Explicit Tesseract config (LSTM engine + single-block PSM)
    - Convert to greyscale before OCR — colour channels confuse the LSTM engine
      and cause the kind of ligature misreads we see with decorative fonts
    - Per-page confidence check: if a page returns mostly garbage (< 20 chars
      after stripping whitespace) we skip it rather than polluting the output
    """
    pytesseract, image_module = _load_ocr_dependencies()
    parts: list[str] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(
                matrix=fitz.Matrix(_OCR_RENDER_ZOOM, _OCR_RENDER_ZOOM),
                alpha=False,
            )
            raw_png = pix.tobytes("png")

            with image_module.open(BytesIO(raw_png)) as img:
                # Convert to greyscale — single channel is faster and more
                # accurate for text OCR than RGB
                grey = img.convert("L")

                text = pytesseract.image_to_string(grey, config=_TESS_CONFIG)

            if text and len(text.strip()) > 20:
                parts.append(text.strip())
            else:
                print(f"[parser] Page {page_num + 1}: OCR returned very little text, skipping.")

    return _collapse_text(parts)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> ParsedText:
    if not pdf_bytes:
        raise ValueError(_ERR_EMPTY_FILE)

    text = _extract_embedded_text(pdf_bytes)
    if text:
        return ParsedText(text=text, parser_used=PARSER_PYMUPDF)

    ocr_text = _extract_text_with_ocr(pdf_bytes)
    if ocr_text:
        print(f"[parser] No embedded text found, OCR succeeded ({len(ocr_text)} chars).")
        return ParsedText(text=ocr_text, parser_used=PARSER_PYMUPDF_OCR)

    raise ValueError(_ERR_NO_TEXT)