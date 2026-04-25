from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from app.main import app


def test_extract_rejects_non_pdf() -> None:
    client = TestClient(app)
    response = client.post(
        "/extract",
        files={"file": ("resume.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400


def test_extract_accepts_pdf_fixture() -> None:
    sample_pdf = next(Path("dataset/Resumes PDF").rglob("*.pdf"), None)
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
        assert "profile" in payload
        assert "metadata" in payload


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
    detail = response.json()["detail"]
    assert any(
        marker in detail
        for marker in (
            "No embedded text found",
            "OCR dependencies are missing",
            "Tesseract binary is not installed",
        )
    )


def test_extract_debug_mode_includes_intermediate_fields() -> None:
    sample_pdf = next(Path("dataset/Resumes PDF").rglob("*.pdf"), None)
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
    assert "raw_sample" in payload["debug"]
    assert "clean_sample" in payload["debug"]
    assert "lines" in payload["debug"]
    assert "name_candidates" in payload["debug"]
