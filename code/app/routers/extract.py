from __future__ import annotations

import time
import uuid
import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.output_schema import ExtractResponse, ExtractionMetadata
from app.services.formatter import sanitize_profile
from app.services.inference import debug_output, run_mock_inference
from app.services.parser import extract_text_from_pdf_bytes
from app.services.preprocess import clean_resume_text

router = APIRouter(prefix="/extract", tags=["extract"])

logger = logging.getLogger(__name__)

_MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024


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
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()

    # Validate empty file
    if not content:
        logger.warning(f"[{request_id}] Empty file uploaded")
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Validate size
    if len(content) > _MAX_PDF_SIZE_BYTES:
        logger.warning(f"[{request_id}] File too large: {len(content)} bytes")
        raise HTTPException(status_code=413, detail="PDF exceeds 10 MB limit")

    # Parsing
    try:
        logger.info(f"[{request_id}] Starting PDF parsing")
        parsed = extract_text_from_pdf_bytes(content)
        logger.info(f"[{request_id}] Parser used: {parsed.parser_used}")
    except Exception as exc:
        logger.error(f"[{request_id}] PDF parsing failed", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Unable to parse PDF: {exc}") from exc

    # Preprocessing
    clean_text = clean_resume_text(parsed.text)
    logger.info(f"[{request_id}] Text length after cleaning: {len(clean_text)}")

    # Inference
    profile, confidence = run_mock_inference(clean_text)
    profile = sanitize_profile(profile)
    logger.info(f"[{request_id}] Inference done with confidence: {confidence}")

    # Timing
    processing_ms = int((time.perf_counter() - start) * 1000)

    metadata = ExtractionMetadata(
        request_id=request_id,
        parser_used=parsed.parser_used,
        text_length=len(clean_text),
        confidence=confidence,
        processing_ms=processing_ms,
    )

    logger.info(f"[{request_id}] Completed in {processing_ms} ms")

    debug_payload = None
    if debug:
        debug_payload = debug_output(parsed.text, clean_text, profile)

    return ExtractResponse(profile=profile, metadata=metadata, debug=debug_payload)