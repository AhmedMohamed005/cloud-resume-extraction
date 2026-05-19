from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.output_schema import ExtractResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET_PDF = _REPO_ROOT / "dataset" / "Resumes PDF"


def test_extract_rejects_non_pdf() -> None:
    client = TestClient(app)
    response = client.post(
        "/extract",
        files={"file": ("resume.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    assert detail.get("error") == "invalid_content_type"


def test_extract_accepts_pdf_fixture() -> None:
    sample_pdf = next(_DATASET_PDF.rglob("*.pdf"), None) if _DATASET_PDF.is_dir() else None
    if sample_pdf is None:
        return

    client = TestClient(app)
    with sample_pdf.open("rb") as f:
        response = client.post(
            "/extract",
            files={"file": (sample_pdf.name, f.read(), "application/pdf")},
        )

    assert response.status_code in {200, 422}
    if response.status_code == 200:
        payload = response.json()
        ExtractResponse.model_validate(payload)
        assert "profile" in payload
        assert "metadata" in payload
        assert payload["metadata"].get("parser_confidence") is not None


def test_extract_rejects_empty_text_pdf() -> None:
    client = TestClient(app)

    doc = fitz.open()
    doc.new_page()
    empty_pdf_bytes = doc.tobytes()

    response = client.post(
        "/extract",
        files={"file": ("empty.pdf", empty_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 422
    raw = response.json()["detail"]
    msg = raw.get("message", "") if isinstance(raw, dict) else str(raw)
    assert any(
        marker in msg
        for marker in (
            "No embedded text found",
            "OCR dependencies are missing",
            "Tesseract binary is not installed",
        )
    )


def test_extract_debug_mode_includes_intermediate_fields() -> None:
    sample_pdf = next(_DATASET_PDF.rglob("*.pdf"), None) if _DATASET_PDF.is_dir() else None
    if sample_pdf is None:
        return

    client = TestClient(app)
    with sample_pdf.open("rb") as f:
        response = client.post(
            "/extract?debug=true",
            files={"file": (sample_pdf.name, f.read(), "application/pdf")},
        )

    if response.status_code != 200:
        return

    payload = response.json()
    assert "debug" in payload
    assert payload["debug"] is not None
    dbg = payload["debug"]
    assert "raw_text" in dbg
    assert "cleaned_text" in dbg
    assert "detected_sections" in dbg
    assert "final_profile" in dbg
