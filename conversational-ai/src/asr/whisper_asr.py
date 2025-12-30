"""
Whisper ASR Model Wrapper

OpenAI Whisper integration for high-accuracy speech recognition.
Supports streaming, batch processing, and multi-language transcription.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

# Try to import whisper
try:
    import whisper
    from whisper.tokenizer import get_tokenizer
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed, using mock implementation")


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    language: Optional[str] = None
    words: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: List[TranscriptionSegment]
    language: str
    language_confidence: float
    duration: float
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "confidence": seg.confidence,
                }
                for seg in self.segments
            ],
            "language": self.language,
            "language_confidence": self.language_confidence,
            "duration": self.duration,
            "processing_time": self.processing_time,
        }


@dataclass
class WhisperConfig:
    """Configuration for Whisper ASR."""
    model_size: str = "large-v3"  # tiny, base, small, medium, large, large-v3
    device: str = "cuda"
    compute_type: str = "float16"  # float32, float16, int8
    language: Optional[str] = None  # None for auto-detect
    task: str = "transcribe"  # transcribe or translate
    
    # Decoding options
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    
    # VAD options
    vad_filter: bool = True
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 500
    
    # Output options
    word_timestamps: bool = True
    highlight_words: bool = False
    max_initial_timestamp: float = 1.0


class WhisperASR:
    """
    Whisper-based Automatic Speech Recognition.
    
    Features:
    - Multiple model sizes (tiny to large-v3)
    - 99 language support with auto-detection
    - Word-level timestamps
    - VAD-based filtering
    - GPU acceleration with FP16/INT8
    """
    
    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available, using mock mode")
            return
        
        logger.info(f"Loading Whisper model: {self.config.model_size}")
        start_time = time.time()
        
        # Load model
        self.model = whisper.load_model(
            self.config.model_size,
            device=self.config.device,
        )
        
        # Apply quantization if requested
        if self.config.compute_type == "float16" and self.config.device == "cuda":
            self.model = self.model.half()
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(
            multilingual=True,
            language=self.config.language,
            task=self.config.task,
        )
        
        load_time = time.time() - start_time
        logger.info(f"Whisper model loaded in {load_time:.2f}s")
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio file path, numpy array, or torch tensor
            language: Language code (None for auto-detect)
            **kwargs: Additional whisper options
            
        Returns:
            TranscriptionResult with full transcription
        """
        start_time = time.time()
        
        # Prepare audio
        if isinstance(audio, str):
            audio_array = whisper.load_audio(audio) if WHISPER_AVAILABLE else np.zeros(16000)
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.numpy()
        else:
            audio_array = audio
        
        # Calculate duration
        duration = len(audio_array) / 16000  # Assuming 16kHz
        
        if not WHISPER_AVAILABLE or self.model is None:
            # Mock result
            return self._mock_transcribe(audio_array, duration, start_time)
        
        # Merge config with kwargs
        options = {
            "language": language or self.config.language,
            "task": self.config.task,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "patience": self.config.patience,
            "temperature": self.config.temperature,
            "word_timestamps": self.config.word_timestamps,
            **kwargs,
        }
        
        # Transcribe
        result = self.model.transcribe(audio_array, **options)
        
        # Build segments
        segments = []
        for seg in result.get("segments", []):
            segment = TranscriptionSegment(
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                confidence=seg.get("avg_logprob", 0.0),
                words=seg.get("words", []),
            )
            segments.append(segment)
        
        processing_time = time.time() - start_time
        
        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", "en"),
            language_confidence=1.0,  # Whisper doesn't expose this directly
            duration=duration,
            processing_time=processing_time,
        )
    
    def _mock_transcribe(
        self,
        audio: np.ndarray,
        duration: float,
        start_time: float,
    ) -> TranscriptionResult:
        """Mock transcription for testing."""
        processing_time = time.time() - start_time
        
        return TranscriptionResult(
            text="This is a mock transcription.",
            segments=[
                TranscriptionSegment(
                    text="This is a mock transcription.",
                    start=0.0,
                    end=duration,
                    confidence=0.95,
                )
            ],
            language="en",
            language_confidence=0.99,
            duration=duration,
            processing_time=processing_time,
        )
    
    def detect_language(
        self,
        audio: Union[str, np.ndarray],
    ) -> Tuple[str, float]:
        """
        Detect the language of audio.
        
        Args:
            audio: Audio file or array
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not WHISPER_AVAILABLE or self.model is None:
            return ("en", 0.99)
        
        # Load and pad audio
        if isinstance(audio, str):
            audio_array = whisper.load_audio(audio)
        else:
            audio_array = audio
        
        # Pad/trim to 30 seconds
        audio_array = whisper.pad_or_trim(audio_array)
        
        # Compute mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_array).to(self.model.device)
        
        if self.config.compute_type == "float16":
            mel = mel.half()
        
        # Detect language
        _, probs = self.model.detect_language(mel)
        
        # Get top language
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]
        
        return (detected_lang, confidence)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        if WHISPER_AVAILABLE:
            return list(whisper.tokenizer.LANGUAGES.keys())
        return ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]


class WhisperQuantized:
    """
    Quantized Whisper for faster inference.
    
    Uses INT8 quantization for ~2x speedup with minimal accuracy loss.
    """
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
    ):
        self.model_size = model_size
        self.device = device
        self.model = None
        
        # Try to load faster-whisper
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type="int8" if device == "cuda" else "int8",
            )
            logger.info(f"Loaded faster-whisper {model_size}")
        except ImportError:
            logger.warning("faster-whisper not available, falling back to standard Whisper")
            self.fallback = WhisperASR(WhisperConfig(model_size=model_size))
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe with quantized model."""
        if self.model is None:
            return self.fallback.transcribe(audio, language, **kwargs)
        
        start_time = time.time()
        
        # Calculate duration
        if isinstance(audio, str):
            import soundfile as sf
            info = sf.info(audio)
            duration = info.duration
        else:
            duration = len(audio) / 16000
        
        # Transcribe with faster-whisper
        segments_gen, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=kwargs.get("beam_size", 5),
            word_timestamps=kwargs.get("word_timestamps", True),
        )
        
        # Collect segments
        segments = []
        full_text = []
        
        for seg in segments_gen:
            segment = TranscriptionSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                confidence=seg.avg_logprob,
                words=[
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (seg.words or [])
                ],
            )
            segments.append(segment)
            full_text.append(seg.text)
        
        processing_time = time.time() - start_time
        
        return TranscriptionResult(
            text=" ".join(full_text).strip(),
            segments=segments,
            language=info.language,
            language_confidence=info.language_probability,
            duration=duration,
            processing_time=processing_time,
        )


def create_asr(
    model_size: str = "large-v3",
    quantized: bool = True,
    device: str = "cuda",
) -> Union[WhisperASR, WhisperQuantized]:
    """
    Factory function to create ASR instance.
    
    Args:
        model_size: Whisper model size
        quantized: Use quantized model for faster inference
        device: Device to use (cuda/cpu)
        
    Returns:
        ASR instance
    """
    if quantized:
        return WhisperQuantized(model_size=model_size, device=device)
    else:
        config = WhisperConfig(model_size=model_size, device=device)
        return WhisperASR(config)
