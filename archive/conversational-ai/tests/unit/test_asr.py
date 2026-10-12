"""Unit tests for ASR components."""

import pytest
import numpy as np
from src.asr.whisper_asr import WhisperASR, WhisperConfig, TranscriptionResult
from src.asr.vad import VoiceActivityDetector, VADConfig
from src.asr.audio_processor import AudioProcessor, normalize_audio


class TestWhisperASR:
    """Tests for Whisper ASR."""
    
    def test_config_defaults(self):
        config = WhisperConfig()
        assert config.model_size == "large-v3"
        assert config.device == "cuda"
        assert config.beam_size == 5
    
    def test_transcribe_mock(self):
        config = WhisperConfig(model_size="tiny")
        asr = WhisperASR(config)
        
        # Create mock audio (1 second of silence)
        audio = np.zeros(16000, dtype=np.float32)
        result = asr.transcribe(audio)
        
        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert result.duration == 1.0
    
    def test_detect_language(self):
        config = WhisperConfig(model_size="tiny")
        asr = WhisperASR(config)
        
        audio = np.zeros(16000, dtype=np.float32)
        lang, conf = asr.detect_language(audio)
        
        assert lang == "en"
        assert 0 <= conf <= 1


class TestVAD:
    """Tests for Voice Activity Detection."""
    
    def test_config_defaults(self):
        config = VADConfig()
        assert config.threshold == 0.5
        assert config.sample_rate == 16000
    
    def test_is_speech_silence(self):
        vad = VoiceActivityDetector()
        
        # Silence should not be detected as speech
        silence = np.zeros(512, dtype=np.float32)
        assert vad.is_speech(silence) == False
    
    def test_is_speech_noise(self):
        vad = VoiceActivityDetector()
        
        # Random noise might trigger energy-based fallback
        noise = np.random.randn(512).astype(np.float32) * 0.1
        # Result depends on threshold
        result = vad.is_speech(noise)
        assert isinstance(result, bool)


class TestAudioProcessor:
    """Tests for audio preprocessing."""
    
    def test_normalize(self):
        audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        normalized = normalize_audio(audio)
        
        assert normalized.max() <= 1.0
        assert normalized.min() >= -1.0
    
    def test_process(self):
        processor = AudioProcessor()
        audio = np.random.randn(16000).astype(np.float32)
        
        processed = processor.process(audio, sample_rate=16000)
        
        assert len(processed) == len(audio)
        assert processed.max() <= 1.0
