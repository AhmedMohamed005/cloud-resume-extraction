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


class ExtractResponse(BaseModel):
    profile: CandidateProfile
    metadata: ExtractionMetadata
    debug: Optional[dict[str, Any]] = Field(default=None)
