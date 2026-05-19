from __future__ import annotations

import logging
from fastapi import FastAPI

from app.routers.extract import router as extract_router


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


setup_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Resume Extraction API",
    version="0.1.0",
    description="PDF resume to structured JSON extraction service",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    logger.info("Health check endpoint called")
    return {"status": "ok"}


app.include_router(extract_router)