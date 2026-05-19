from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.output_schema import ExtractResponse, ExtractionMetadata
from app.services.formatter import sanitize_profile
from app.services.inference import run_mock_inference
from app.services.ner_engine import _model_ready, resolve_model_dir, run_ner_with_heuristic_merge
from app.services.parser import extract_text_from_pdf_bytes, parser_confidence_from_quality
from app.services.preprocess import clean_resume_text

router = APIRouter(prefix="/extract", tags=["extract"])

logger = logging.getLogger(__name__)

_MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
_READ_TIMEOUT_S = 30.0


@router.post("", response_model=ExtractResponse)
async def extract_resume(
    file: UploadFile = File(...),
    debug: bool = Query(default=False, description="Return intermediate extraction/debug details"),
) -> ExtractResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    logger.info(f"[{request_id}] Incoming file: {file.filename}")

    # Validate content type
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        logger.warning(f"[{request_id}] Invalid content type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_content_type", "message": "Only PDF files are supported"},
        )

    try:
        content = await asyncio.wait_for(file.read(), timeout=_READ_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        logger.warning(f"[{request_id}] Upload read timed out")
        raise HTTPException(
            status_code=408,
            detail={"error": "upload_timeout", "message": f"Reading upload exceeded {_READ_TIMEOUT_S}s"},
        ) from exc

    # Validate empty file
    if not content:
        logger.warning(f"[{request_id}] Empty file uploaded")
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "message": "Uploaded file is empty"},
        )

    # Validate size
    if len(content) > _MAX_PDF_SIZE_BYTES:
        logger.warning(f"[{request_id}] File too large: {len(content)} bytes")
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "message": "PDF exceeds 10 MB limit"},
        )

    # Parsing
    try:
        logger.info(f"[{request_id}] Starting PDF parsing")
        parsed = extract_text_from_pdf_bytes(content)
        logger.info(f"[{request_id}] Parser used: {parsed.parser_used}")
    except Exception as exc:
        logger.error(f"[{request_id}] PDF parsing failed", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail={"error": "parse_failed", "message": str(exc)},
        ) from exc

    # Preprocessing
    clean_text = clean_resume_text(parsed.text)
    logger.info(f"[{request_id}] Text length after cleaning: {len(clean_text)}")

    # Inference: LoRA NER by default; mock for USE_MOCK_INFERENCE or missing model / failure
    use_mock = os.environ.get("USE_MOCK_INFERENCE", "").lower() in ("1", "true", "yes")
    backend = "mock_heuristic"
    if use_mock:
        profile, confidence, sections = run_mock_inference(clean_text)
    elif _model_ready(resolve_model_dir()):
        try:
            profile, confidence, sections = run_ner_with_heuristic_merge(clean_text)
            backend = "ner_lora_hybrid"
        except Exception:
            logger.exception(f"[{request_id}] NER inference failed, using heuristic fallback")
            profile, confidence, sections = run_mock_inference(clean_text)
            backend = "mock_heuristic_fallback"
    else:
        logger.warning(
            f"[{request_id}] No NER model at {resolve_model_dir()}; set RESUME_MODEL_PATH or USE_MOCK_INFERENCE=1"
        )
        profile, confidence, sections = run_mock_inference(clean_text)
        backend = "mock_heuristic"

    profile = sanitize_profile(profile)
    logger.info(f"[{request_id}] Inference done backend={backend} confidence={confidence}")

    # Timing
    processing_ms = int((time.perf_counter() - start) * 1000)

    metadata = ExtractionMetadata(
        request_id=request_id,
        parser_used=parsed.parser_used,
        text_length=len(clean_text),
        confidence=confidence,
        processing_ms=processing_ms,
        inference_backend=backend,
        parser_quality=parsed.quality_flag,
        parser_confidence=parser_confidence_from_quality(parsed.quality_flag),
    )

    logger.info(f"[{request_id}] Completed in {processing_ms} ms")

    debug_payload = None
    if debug:
        debug_payload = {
            "raw_text": parsed.text,
            "cleaned_text": clean_text,
            "detected_sections": sections,
            "final_profile": profile,
        }
    return ExtractResponse(profile=profile, metadata=metadata, debug=debug_payload)