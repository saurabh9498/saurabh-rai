"""
Pytest configuration and fixtures for Conversational AI tests.

This module provides shared fixtures for unit, integration, and load tests.
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Global test configuration."""
    return {
        "asr": {
            "model_size": "tiny",
            "language": "en",
            "sample_rate": 16000,
        },
        "nlu": {
            "intent_threshold": 0.5,
            "entity_threshold": 0.7,
            "max_length": 512,
        },
        "dialog": {
            "max_turns": 20,
            "context_window": 5,
            "timeout_seconds": 30,
        },
        "tts": {
            "model": "tts_models/en/ljspeech/tacotron2-DDC",
            "sample_rate": 22050,
        },
        "api": {
            "host": "localhost",
            "port": 8000,
        },
    }


# =============================================================================
# Audio Fixtures
# =============================================================================


@pytest.fixture
def sample_audio_16k() -> np.ndarray:
    """Generate sample 16kHz audio (1 second of sine wave)."""
    duration = 1.0
    sample_rate = 16000
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio


@pytest.fixture
def sample_audio_bytes(sample_audio_16k: np.ndarray) -> bytes:
    """Convert sample audio to bytes."""
    import struct
    
    # Convert float32 to int16
    audio_int16 = (sample_audio_16k * 32767).astype(np.int16)
    return audio_int16.tobytes()


@pytest.fixture
def sample_audio_file(sample_audio_16k: np.ndarray, tmp_path: Path) -> Path:
    """Create a temporary WAV file with sample audio."""
    import wave

    audio_path = tmp_path / "test_audio.wav"
    audio_int16 = (sample_audio_16k * 32767).astype(np.int16)

    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_int16.tobytes())

    return audio_path


# =============================================================================
# NLU Fixtures
# =============================================================================


@pytest.fixture
def sample_intents() -> List[Dict[str, Any]]:
    """Sample intent definitions for testing."""
    return [
        {
            "intent": "greeting",
            "examples": ["hello", "hi", "hey", "good morning"],
            "slots": [],
        },
        {
            "intent": "book_flight",
            "examples": [
                "book a flight to London",
                "I need to fly to Paris",
                "find flights to Tokyo",
            ],
            "slots": ["destination", "date"],
        },
        {
            "intent": "check_weather",
            "examples": [
                "what's the weather like",
                "will it rain tomorrow",
                "weather in New York",
            ],
            "slots": ["city", "date"],
        },
    ]


@pytest.fixture
def sample_entities() -> List[Dict[str, Any]]:
    """Sample entity definitions for testing."""
    return [
        {
            "entity_type": "city",
            "examples": [
                {"text": "New York", "value": "new_york"},
                {"text": "London", "value": "london"},
                {"text": "Paris", "value": "paris"},
            ],
        },
        {
            "entity_type": "date",
            "examples": [
                {"text": "tomorrow", "value": "tomorrow"},
                {"text": "next Monday", "value": "next_monday"},
            ],
        },
    ]


@pytest.fixture
def sample_nlu_result() -> Dict[str, Any]:
    """Sample NLU processing result."""
    return {
        "text": "Book a flight to London tomorrow",
        "intent": "book_flight",
        "intent_confidence": 0.95,
        "entities": [
            {"type": "destination", "value": "london", "text": "London", "start": 17, "end": 23},
            {"type": "date", "value": "tomorrow", "text": "tomorrow", "start": 24, "end": 32},
        ],
        "sentiment": {"label": "neutral", "score": 0.8},
    }


# =============================================================================
# Dialog Fixtures
# =============================================================================


@pytest.fixture
def sample_conversation() -> Dict[str, Any]:
    """Sample multi-turn conversation."""
    return {
        "conversation_id": "conv_001",
        "session_id": "session_abc123",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "user",
                "text": "Hi there",
                "intent": "greeting",
            },
            {
                "turn_id": 2,
                "speaker": "assistant",
                "text": "Hello! How can I help you today?",
            },
            {
                "turn_id": 3,
                "speaker": "user",
                "text": "Book a flight to London",
                "intent": "book_flight",
            },
            {
                "turn_id": 4,
                "speaker": "assistant",
                "text": "I'd be happy to help you book a flight to London. When would you like to travel?",
            },
        ],
    }


@pytest.fixture
def sample_dialog_state() -> Dict[str, Any]:
    """Sample dialog state."""
    return {
        "session_id": "session_abc123",
        "turn_count": 4,
        "current_intent": "book_flight",
        "slots": {
            "destination": "london",
        },
        "missing_slots": ["date", "origin"],
        "intent_history": ["greeting", "book_flight"],
        "context": {
            "user_name": None,
            "preferences": {},
        },
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper ASR model."""
    model = MagicMock()

    # Mock transcription result
    mock_result = {
        "text": "Hello, I need help booking a flight",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "Hello, I need help booking a flight",
            }
        ],
        "language": "en",
    }
    model.transcribe.return_value = mock_result

    return model


@pytest.fixture
def mock_tts_synthesizer():
    """Mock TTS synthesizer."""
    synth = MagicMock()

    # Return sample audio
    synth.synthesize.return_value = np.zeros(22050, dtype=np.float32)  # 1 second of silence
    synth.is_ready.return_value = True

    return synth


@pytest.fixture
def mock_nlu_pipeline():
    """Mock NLU pipeline."""
    pipeline = AsyncMock()

    async def mock_process(text: str):
        result = MagicMock()
        result.intent = "greeting"
        result.confidence = 0.95
        result.entities = []
        result.sentiment = {"label": "neutral", "score": 0.8}
        return result

    pipeline.process = mock_process
    return pipeline


@pytest.fixture
def mock_dialog_manager():
    """Mock dialog manager."""
    manager = MagicMock()

    manager.process_turn.return_value = {
        "response": "Hello! How can I help you?",
        "action": None,
        "slots": {},
    }
    manager.get_state.return_value = {
        "turn_count": 1,
        "intent_history": ["greeting"],
    }

    return manager


# =============================================================================
# API Client Fixtures
# =============================================================================


@pytest.fixture
async def async_client():
    """Async HTTP client for API testing."""
    from httpx import AsyncClient
    from api.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def api_headers() -> Dict[str, str]:
    """Standard API headers."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# =============================================================================
# Temporary Files Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create temporary directory for model checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_files(
    temp_data_dir: Path,
    sample_intents: List[Dict],
    sample_entities: List[Dict],
    sample_conversation: Dict,
) -> Dict[str, Path]:
    """Create temporary data files for testing."""
    intents_path = temp_data_dir / "intents.json"
    entities_path = temp_data_dir / "entities.json"
    conversations_path = temp_data_dir / "conversations.json"

    with open(intents_path, "w") as f:
        json.dump(sample_intents, f)

    with open(entities_path, "w") as f:
        json.dump(sample_entities, f)

    with open(conversations_path, "w") as f:
        json.dump([sample_conversation], f)

    return {
        "intents": intents_path,
        "entities": entities_path,
        "conversations": conversations_path,
    }


# =============================================================================
# Event Loop Fixture
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Markers Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external deps)")
    config.addinivalue_line("markers", "integration: Integration tests (require services)")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "audio: Tests requiring audio devices")


# =============================================================================
# Skip Conditions
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on environment."""
    import torch

    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_integration = pytest.mark.skip(reason="Integration services not available")
    skip_audio = pytest.mark.skip(reason="Audio device not available")

    for item in items:
        # Skip GPU tests if no GPU
        if "gpu" in item.keywords:
            if not torch.cuda.is_available():
                item.add_marker(skip_gpu)

        # Skip integration tests unless explicitly enabled
        if "integration" in item.keywords:
            if not os.environ.get("RUN_INTEGRATION_TESTS"):
                item.add_marker(skip_integration)

        # Skip audio tests unless audio device available
        if "audio" in item.keywords:
            if not os.environ.get("AUDIO_DEVICE_AVAILABLE"):
                item.add_marker(skip_audio)
