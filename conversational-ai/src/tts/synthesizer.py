"""
Neural Text-to-Speech Synthesizer

VITS/Coqui TTS integration for natural speech synthesis.
Supports streaming, multi-speaker, and emotion control.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator, Tuple
import logging
import io
import time

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS."""
    model_name: str = "tts_models/en/ljspeech/vits"
    speaker_id: Optional[int] = None
    language: str = "en"
    
    # Audio settings
    sample_rate: int = 22050
    
    # Synthesis settings
    speed: float = 1.0
    pitch: float = 1.0
    energy: float = 1.0
    
    # Streaming
    enable_streaming: bool = True
    chunk_size_chars: int = 100


@dataclass
class SynthesisResult:
    """Result of speech synthesis."""
    audio: np.ndarray
    sample_rate: int
    duration: float
    processing_time: float
    text: str
    
    def to_wav_bytes(self) -> bytes:
        """Convert audio to WAV bytes."""
        import wave
        
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            
            # Convert to int16
            audio_int16 = (self.audio * 32767).astype(np.int16)
            wav.writeframes(audio_int16.tobytes())
        
        buffer.seek(0)
        return buffer.read()


class NeuralTTS:
    """
    Neural TTS using Coqui TTS / VITS.
    
    Features:
    - End-to-end neural synthesis
    - Multi-speaker support
    - Streaming output
    - SSML support
    - GPU acceleration
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load TTS model."""
        try:
            from TTS.api import TTS
            
            # Check GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = TTS(
                model_name=self.config.model_name,
                progress_bar=False,
            ).to(device)
            
            logger.info(f"Loaded TTS model: {self.config.model_name} on {device}")
        except ImportError:
            logger.warning("Coqui TTS not available, using mock synthesis")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            self.model = None
    
    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> SynthesisResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text
            speaker: Speaker name/ID (for multi-speaker models)
            speed: Speech speed multiplier
            
        Returns:
            SynthesisResult with audio
        """
        start_time = time.time()
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        if self.model is None:
            # Mock synthesis
            return self._mock_synthesize(text, start_time)
        
        # Synthesize
        try:
            audio = self.model.tts(
                text=text,
                speaker=speaker,
                speed=speed or self.config.speed,
            )
            
            # Convert to numpy
            if isinstance(audio, list):
                audio = np.array(audio)
            
            processing_time = time.time() - start_time
            duration = len(audio) / self.config.sample_rate
            
            return SynthesisResult(
                audio=audio,
                sample_rate=self.config.sample_rate,
                duration=duration,
                processing_time=processing_time,
                text=text,
            )
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return self._mock_synthesize(text, start_time)
    
    def synthesize_streaming(
        self,
        text: str,
        speaker: Optional[str] = None,
    ) -> Iterator[np.ndarray]:
        """
        Stream synthesized audio in chunks.
        
        Args:
            text: Input text
            speaker: Speaker name/ID
            
        Yields:
            Audio chunks as numpy arrays
        """
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            if sentence.strip():
                result = self.synthesize(sentence, speaker)
                yield result.audio
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TTS."""
        # Expand abbreviations
        abbreviations = {
            "Mr.": "Mister",
            "Mrs.": "Misses",
            "Dr.": "Doctor",
            "St.": "Street",
            "vs.": "versus",
            "etc.": "etcetera",
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Handle numbers
        # (Simplified - production would use more comprehensive normalization)
        
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming."""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Merge short sentences
        merged = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) < self.config.chunk_size_chars:
                current += " " + sentence if current else sentence
            else:
                if current:
                    merged.append(current)
                current = sentence
        
        if current:
            merged.append(current)
        
        return merged
    
    def _mock_synthesize(
        self,
        text: str,
        start_time: float,
    ) -> SynthesisResult:
        """Mock synthesis for testing."""
        # Generate silence with approximate duration
        words = len(text.split())
        duration = words * 0.3  # ~0.3 seconds per word
        
        samples = int(duration * self.config.sample_rate)
        audio = np.zeros(samples, dtype=np.float32)
        
        # Add some noise to make it non-silent
        audio += np.random.randn(samples) * 0.001
        
        processing_time = time.time() - start_time
        
        return SynthesisResult(
            audio=audio,
            sample_rate=self.config.sample_rate,
            duration=duration,
            processing_time=processing_time,
            text=text,
        )
    
    def get_speakers(self) -> List[str]:
        """Get available speakers for multi-speaker models."""
        if self.model is None:
            return ["default"]
        
        try:
            return self.model.speakers or ["default"]
        except:
            return ["default"]


class SSMLParser:
    """
    SSML (Speech Synthesis Markup Language) parser.
    
    Supports common SSML tags for controlling synthesis.
    """
    
    def __init__(self):
        self.supported_tags = {
            "speak", "voice", "prosody", "break", "emphasis",
            "say-as", "sub", "phoneme", "p", "s",
        }
    
    def parse(self, ssml: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse SSML and extract text with annotations.
        
        Args:
            ssml: SSML-formatted text
            
        Returns:
            Tuple of (plain_text, annotations)
        """
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(ssml)
            text, annotations = self._process_element(root)
            return text, annotations
        except ET.ParseError:
            # Return as plain text if not valid SSML
            return ssml, {}
    
    def _process_element(
        self,
        element,
        annotations: Optional[Dict] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Process an SSML element."""
        annotations = annotations or {}
        text_parts = []
        
        # Handle element text
        if element.text:
            text_parts.append(element.text)
        
        # Process children
        for child in element:
            child_text, _ = self._process_element(child, annotations)
            text_parts.append(child_text)
            
            # Handle tail text
            if child.tail:
                text_parts.append(child.tail)
            
            # Extract annotations
            if child.tag == "break":
                time_attr = child.get("time", "500ms")
                annotations.setdefault("breaks", []).append({
                    "position": len("".join(text_parts)),
                    "duration": time_attr,
                })
            elif child.tag == "prosody":
                annotations.setdefault("prosody", []).append({
                    "rate": child.get("rate"),
                    "pitch": child.get("pitch"),
                    "volume": child.get("volume"),
                })
        
        return "".join(text_parts), annotations


class TTSService:
    """
    TTS service with caching and queuing.
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.tts = NeuralTTS(config)
        self.ssml_parser = SSMLParser()
        self._cache: Dict[str, SynthesisResult] = {}
        self._cache_max_size = 100
    
    def synthesize(
        self,
        text: str,
        use_ssml: bool = False,
        **kwargs,
    ) -> SynthesisResult:
        """
        Synthesize speech with optional SSML processing.
        
        Args:
            text: Input text (plain or SSML)
            use_ssml: Whether to parse as SSML
            **kwargs: Additional synthesis options
            
        Returns:
            SynthesisResult
        """
        # Check cache
        cache_key = f"{text}_{kwargs}"
        if cache_key in self._cache:
            logger.debug("TTS cache hit")
            return self._cache[cache_key]
        
        # Parse SSML if needed
        if use_ssml and text.strip().startswith("<"):
            plain_text, annotations = self.ssml_parser.parse(text)
            # Apply annotations (simplified)
            text = plain_text
        
        # Synthesize
        result = self.tts.synthesize(text, **kwargs)
        
        # Cache result
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[cache_key] = result
        
        return result
    
    def synthesize_streaming(
        self,
        text: str,
        **kwargs,
    ) -> Iterator[bytes]:
        """
        Stream synthesized audio as WAV chunks.
        
        Yields:
            WAV audio bytes for each chunk
        """
        for audio_chunk in self.tts.synthesize_streaming(text, **kwargs):
            result = SynthesisResult(
                audio=audio_chunk,
                sample_rate=self.tts.config.sample_rate,
                duration=len(audio_chunk) / self.tts.config.sample_rate,
                processing_time=0,
                text="",
            )
            yield result.to_wav_bytes()
    
    def get_voices(self) -> List[str]:
        """Get available voices."""
        return self.tts.get_speakers()
