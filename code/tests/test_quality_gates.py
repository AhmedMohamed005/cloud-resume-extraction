"""Execution-plan §10 style smoke gates: valid JSON shape, success path, latency ceiling."""
from __future__ import annotations

import fitz
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.output_schema import ExtractResponse


def _embedded_text_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Jane Candidate\nPython AWS Docker\njane@example.com\n" * 4)
    return doc.tobytes()


def test_extract_returns_valid_schema_under_mock() -> None:
    client = TestClient(app)
    pdf = _embedded_text_pdf_bytes()
    response = client.post(
        "/extract",
        files={"file": ("synthetic.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    ExtractResponse.model_validate(payload)
    meta = payload["metadata"]
    assert meta["inference_backend"] in {
        "mock_heuristic",
        "ner_lora_hybrid",
        "mock_heuristic_fallback",
    }
    assert 0.0 <= meta["confidence"] <= 1.0
    assert 0.0 <= meta["parser_confidence"] <= 1.0
    # Loose latency ceiling (CI / dev machines vary); catches hung handlers
    assert meta["processing_ms"] < 180_000


def test_health_is_json_object() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert body.get("status") == "ok"
