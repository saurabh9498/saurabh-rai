"""Additional API routes."""

from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()


@router.get("/intents")
async def list_intents() -> List[str]:
    """List all supported intents."""
    from ..nlu.intent_classifier import INTENT_TAXONOMY
    
    all_intents = []
    for category, intents in INTENT_TAXONOMY.items():
        all_intents.extend(intents)
    return all_intents


@router.get("/voices")
async def list_voices() -> List[str]:
    """List available TTS voices."""
    return ["default", "ljspeech", "vctk_p225", "vctk_p226"]


@router.get("/languages")
async def list_languages() -> List[str]:
    """List supported languages."""
    return ["en", "es", "fr", "de", "zh", "ja", "ko", "ar", "hi"]
