"""Pydantic schemas for API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    language: str = "en"


class TextResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    entities: List[Dict[str, Any]]
    session_id: str
    audio_url: Optional[str] = None


class Entity(BaseModel):
    type: str
    value: str
    confidence: float


class NLUResponse(BaseModel):
    intent: str
    confidence: float
    entities: List[Entity]
    sentiment: Optional[str] = None
