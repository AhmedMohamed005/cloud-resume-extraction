from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    skills: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)


class ExtractionMetadata(BaseModel):
    request_id: str
    parser_used: str
    text_length: int
    confidence: float = Field(ge=0.0, le=1.0)
    processing_ms: int = Field(ge=0)
    inference_backend: Optional[str] = Field(
        default=None,
        description="ner_lora_hybrid (NER + heuristics), mock_heuristic, or mock_heuristic_fallback",
    )
    parser_quality: Optional[str] = Field(
        default=None,
        description="ok, low_text, empty, noisy_symbols",
    )
    parser_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Heuristic 0–1 from parser_quality (WP2 metadata).",
    )


class ExtractResponse(BaseModel):
    profile: CandidateProfile
    metadata: ExtractionMetadata
    debug: Optional[dict] = None
