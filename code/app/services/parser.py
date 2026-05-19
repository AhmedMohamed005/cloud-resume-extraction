"""PDF text extraction: PyMuPDF primary, pdfminer.six fallback, then OCR."""
from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from shutil import which
from typing import Any

import fitz

logger = logging.getLogger(__name__)

PARSER_PYMUPDF = "pymupdf"
PARSER_PDFMINER = "pdfminer"
PARSER_PYMUPDF_OCR = "pymupdf+ocr"

_MIN_MEANINGFUL_CHARS = 40


def _ocr_render_zoom() -> float:
    try:
        return float(os.environ.get("OCR_RENDER_ZOOM", "3.5"))
    except ValueError:
        return 3.5


def _ocr_tesseract_configs() -> list[str]:
    """Multiple PSM tries per page; pick longest text (env OCR_TESS_PSMS='6,4,3')."""
    oem = (os.environ.get("OCR_TESS_OEM", "3") or "3").strip()
    raw = (os.environ.get("OCR_TESS_PSMS", "6,4") or "6,4").strip()
    configs: list[str] = []
    for part in raw.split(","):
        p = part.strip()
        if p.isdigit():
            configs.append(f"--oem {oem} --psm {p}")
    return configs or ["--oem 3 --psm 6"]
_ERR_EMPTY_FILE = "Empty file content"
_ERR_NO_TEXT = "No embedded text found in the PDF and OCR also returned empty text."
_ERR_OCR_DEPS = "OCR dependencies are missing. Install with: pip install pytesseract pillow"
_ERR_TESSERACT_MISSING = (
    "Tesseract binary not found. "
    "Windows: install https://github.com/UB-Mannheim/tesseract/wiki (default path is auto-detected) "
    "or `winget install UB-Mannheim.TesseractOCR`, or set TESSERACT_CMD to the full path to tesseract.exe. "
    "macOS: brew install tesseract. "
    "Linux: apt install tesseract-ocr"
)


def _resolve_tesseract_executable() -> str | None:
    """PATH, then TESSERACT_CMD, then common Windows install locations."""
    explicit = os.environ.get("TESSERACT_CMD", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit

    w = which("tesseract")
    if w:
        return w

    if platform.system() == "Windows":
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Tesseract-OCR"
            / "tesseract.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Tesseract-OCR"
            / "tesseract.exe",
        ]
        for p in candidates:
            if p.is_file():
                return str(p)
    return None

@dataclass(frozen=True)
class ParsedText:
    text: str
    parser_used: str
    """Human-readable quality flag for clients and logs."""
    quality_flag: str = "ok"
    char_count: int = 0


def _collapse_text(chunks: list[str]) -> str:
    return "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()


def parser_confidence_from_quality(quality_flag: str) -> float:
    """Rough 0–1 score for clients (WP2); not calibrated probability."""
    return {
        "ok": 1.0,
        "low_text": 0.62,
        "empty": 0.0,
        "noisy_symbols": 0.45,
    }.get(quality_flag, 0.55)


def _quality_flag(text: str) -> str:
    s = text.strip()
    n = len(s)
    if n < _MIN_MEANINGFUL_CHARS:
        return "low_text"
    if n == 0:
        return "empty"
    weird = sum(1 for c in s if c in "\x00\x01\x02\x03" or (ord(c) > 0x10FFFF))
    if weird / max(n, 1) > 0.05:
        return "noisy_symbols"
    return "ok"


def _extract_embedded_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        parts = [page.get_text("text") for page in doc]
    return _collapse_text(parts)


def _extract_pdfminer_text(pdf_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
    except ImportError as exc:
        raise RuntimeError(
            "pdfminer.six is required for the pdfminer fallback. Install: pip install pdfminer.six"
        ) from exc

    with BytesIO(pdf_bytes) as bio:
        raw = extract_text(bio) or ""
    return raw.strip()


def _load_ocr_dependencies() -> tuple[Any, Any]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(_ERR_OCR_DEPS) from exc

    tess = _resolve_tesseract_executable()
    if not tess:
        raise RuntimeError(_ERR_TESSERACT_MISSING)
    pytesseract.pytesseract.tesseract_cmd = tess
    logger.debug("Using Tesseract at %s", tess)
    return pytesseract, Image


def _extract_text_with_ocr(pdf_bytes: bytes) -> str:
    pytesseract, image_module = _load_ocr_dependencies()
    parts: list[str] = []

    zoom = _ocr_render_zoom()
    zoom_m = fitz.Matrix(zoom, zoom)
    tess_cfgs = _ocr_tesseract_configs()

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=zoom_m, alpha=False)
            raw_png = pix.tobytes("png")

            with image_module.open(BytesIO(raw_png)) as img:
                grey = img.convert("L")
                best = ""
                for cfg in tess_cfgs:
                    chunk = pytesseract.image_to_string(grey, config=cfg) or ""
                    chunk = chunk.strip()
                    if len(chunk) > len(best):
                        best = chunk

            if best and len(best) > 20:
                parts.append(best)
            else:
                logger.debug("Page %s: OCR returned very little text, skipping.", page_num + 1)

    return _collapse_text(parts)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> ParsedText:
    if not pdf_bytes:
        raise ValueError(_ERR_EMPTY_FILE)

    text = _extract_embedded_text(pdf_bytes)
    used = PARSER_PYMUPDF

    if len(text.strip()) < _MIN_MEANINGFUL_CHARS:
        try:
            pm = _extract_pdfminer_text(pdf_bytes)
            if len(pm.strip()) > len(text.strip()):
                text = pm
                used = PARSER_PDFMINER
        except RuntimeError:
            pass

    if len(text.strip()) >= _MIN_MEANINGFUL_CHARS:
        q = _quality_flag(text)
        return ParsedText(text=text, parser_used=used, quality_flag=q, char_count=len(text.strip()))

    ocr_text = _extract_text_with_ocr(pdf_bytes)
    if ocr_text:
        logger.info("Embedded/pdfminer text weak; OCR succeeded (%s chars).", len(ocr_text))
        q = _quality_flag(ocr_text)
        return ParsedText(
            text=ocr_text,
            parser_used=PARSER_PYMUPDF_OCR,
            quality_flag=q,
            char_count=len(ocr_text.strip()),
        )

    raise ValueError(_ERR_NO_TEXT)
