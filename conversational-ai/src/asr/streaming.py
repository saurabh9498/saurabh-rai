"""
Streaming ASR Implementation

Real-time speech recognition with partial results and low latency.
Supports WebSocket streaming and chunked audio processing.
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Callable, List, Dict, Any
from collections import deque
import threading
import queue
import time
import logging

from .whisper_asr import WhisperASR, WhisperConfig, TranscriptionResult, TranscriptionSegment
from .vad import VoiceActivityDetector, VADConfig

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming ASR."""
    # Audio settings
    sample_rate: int = 16000
    chunk_duration_ms: int = 30
    
    # Streaming settings
    min_audio_length_ms: int = 500
    max_audio_length_ms: int = 30000
    silence_threshold_ms: int = 500
    
    # VAD settings
    use_vad: bool = True
    vad_threshold: float = 0.5
    
    # Output settings
    emit_partial_results: bool = True
    partial_result_interval_ms: int = 300


@dataclass
class PartialResult:
    """Partial transcription result during streaming."""
    text: str
    is_final: bool
    confidence: float
    timestamp: float
    audio_duration: float


class AudioBuffer:
    """Thread-safe audio buffer for streaming."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration_sec: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)
        self._buffer = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()
        self._last_vad_speech = time.time()
    
    def append(self, audio_chunk: np.ndarray):
        """Add audio samples to buffer."""
        with self._lock:
            self._buffer.extend(audio_chunk.flatten())
    
    def get_audio(self) -> np.ndarray:
        """Get all buffered audio."""
        with self._lock:
            return np.array(list(self._buffer), dtype=np.float32)
    
    def get_duration(self) -> float:
        """Get buffer duration in seconds."""
        with self._lock:
            return len(self._buffer) / self.sample_rate
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
    
    def trim_to_duration(self, duration_sec: float):
        """Keep only the last N seconds of audio."""
        samples_to_keep = int(duration_sec * self.sample_rate)
        with self._lock:
            while len(self._buffer) > samples_to_keep:
                self._buffer.popleft()


class StreamingASR:
    """
    Streaming Automatic Speech Recognition.
    
    Features:
    - Real-time transcription with partial results
    - Voice Activity Detection for endpoint detection
    - Automatic silence detection
    - WebSocket-compatible async interface
    """
    
    def __init__(
        self,
        asr: Optional[WhisperASR] = None,
        config: Optional[StreamingConfig] = None,
    ):
        self.config = config or StreamingConfig()
        self.asr = asr or WhisperASR()
        
        # Initialize VAD
        if self.config.use_vad:
            self.vad = VoiceActivityDetector(VADConfig(
                threshold=self.config.vad_threshold,
            ))
        else:
            self.vad = None
        
        # State
        self._buffer = AudioBuffer(self.config.sample_rate)
        self._is_speaking = False
        self._last_speech_time = 0.0
        self._partial_text = ""
        
        # Callbacks
        self._on_partial: Optional[Callable[[PartialResult], None]] = None
        self._on_final: Optional[Callable[[TranscriptionResult], None]] = None
    
    def on_partial(self, callback: Callable[[PartialResult], None]):
        """Set callback for partial results."""
        self._on_partial = callback
    
    def on_final(self, callback: Callable[[TranscriptionResult], None]):
        """Set callback for final results."""
        self._on_final = callback
    
    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[PartialResult]:
        """
        Process streaming audio and yield transcription results.
        
        Args:
            audio_stream: Async iterator of audio chunks (16-bit PCM)
            
        Yields:
            PartialResult for each transcription update
        """
        last_partial_time = time.time()
        
        async for chunk in audio_stream:
            # Convert bytes to numpy
            audio_chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to buffer
            self._buffer.append(audio_chunk)
            
            # Check VAD
            if self.vad:
                is_speech = self.vad.is_speech(audio_chunk)
                
                if is_speech:
                    self._is_speaking = True
                    self._last_speech_time = time.time()
                elif self._is_speaking:
                    silence_duration = time.time() - self._last_speech_time
                    
                    if silence_duration * 1000 > self.config.silence_threshold_ms:
                        # End of utterance detected
                        self._is_speaking = False
                        
                        # Get final transcription
                        result = await self._transcribe_final()
                        
                        if result and result.text:
                            yield PartialResult(
                                text=result.text,
                                is_final=True,
                                confidence=0.95,
                                timestamp=time.time(),
                                audio_duration=result.duration,
                            )
                            
                            if self._on_final:
                                self._on_final(result)
                        
                        # Clear buffer for next utterance
                        self._buffer.clear()
                        continue
            
            # Emit partial result periodically
            current_time = time.time()
            if (
                self.config.emit_partial_results
                and current_time - last_partial_time > self.config.partial_result_interval_ms / 1000
                and self._buffer.get_duration() > self.config.min_audio_length_ms / 1000
            ):
                partial = await self._transcribe_partial()
                
                if partial and partial.text != self._partial_text:
                    self._partial_text = partial.text
                    last_partial_time = current_time
                    
                    yield partial
                    
                    if self._on_partial:
                        self._on_partial(partial)
    
    async def _transcribe_partial(self) -> Optional[PartialResult]:
        """Transcribe current buffer for partial result."""
        audio = self._buffer.get_audio()
        
        if len(audio) < self.config.sample_rate * 0.3:  # Less than 300ms
            return None
        
        try:
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.asr.transcribe(audio),
            )
            
            return PartialResult(
                text=result.text,
                is_final=False,
                confidence=0.8,
                timestamp=time.time(),
                audio_duration=result.duration,
            )
        except Exception as e:
            logger.error(f"Partial transcription failed: {e}")
            return None
    
    async def _transcribe_final(self) -> Optional[TranscriptionResult]:
        """Transcribe complete utterance."""
        audio = self._buffer.get_audio()
        
        if len(audio) < self.config.sample_rate * 0.1:  # Less than 100ms
            return None
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.asr.transcribe(audio),
            )
            return result
        except Exception as e:
            logger.error(f"Final transcription failed: {e}")
            return None
    
    def reset(self):
        """Reset streaming state."""
        self._buffer.clear()
        self._is_speaking = False
        self._last_speech_time = 0.0
        self._partial_text = ""


class StreamingSession:
    """
    Manages a single streaming ASR session.
    
    Handles connection lifecycle, audio buffering, and result delivery.
    """
    
    def __init__(
        self,
        session_id: str,
        streaming_asr: StreamingASR,
    ):
        self.session_id = session_id
        self.asr = streaming_asr
        self.created_at = time.time()
        self._active = False
        self._results: List[PartialResult] = []
    
    @property
    def is_active(self) -> bool:
        return self._active
    
    @property
    def duration(self) -> float:
        return time.time() - self.created_at
    
    async def start(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[PartialResult]:
        """Start processing audio stream."""
        self._active = True
        
        try:
            async for result in self.asr.process_audio_stream(audio_stream):
                self._results.append(result)
                yield result
        finally:
            self._active = False
    
    def get_transcript(self) -> str:
        """Get full transcript from all final results."""
        final_results = [r for r in self._results if r.is_final]
        return " ".join(r.text for r in final_results)
    
    def stop(self):
        """Stop the session."""
        self._active = False
        self.asr.reset()


class StreamingASRManager:
    """Manages multiple concurrent streaming sessions."""
    
    def __init__(self, asr_config: Optional[WhisperConfig] = None):
        self.asr_config = asr_config
        self._sessions: Dict[str, StreamingSession] = {}
        self._lock = threading.Lock()
    
    def create_session(self, session_id: str) -> StreamingSession:
        """Create a new streaming session."""
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            asr = WhisperASR(self.asr_config)
            streaming_asr = StreamingASR(asr=asr)
            session = StreamingSession(session_id, streaming_asr)
            
            self._sessions[session_id] = session
            logger.info(f"Created streaming session: {session_id}")
            
            return session
    
    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get an existing session."""
        return self._sessions.get(session_id)
    
    def close_session(self, session_id: str):
        """Close and remove a session."""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                session.stop()
                logger.info(f"Closed streaming session: {session_id}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return [
            sid for sid, session in self._sessions.items()
            if session.is_active
        ]
    
    def cleanup_inactive(self, max_age_sec: float = 300):
        """Remove inactive sessions older than max_age_sec."""
        with self._lock:
            to_remove = []
            
            for session_id, session in self._sessions.items():
                if not session.is_active and session.duration > max_age_sec:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                self._sessions.pop(session_id)
                logger.info(f"Cleaned up inactive session: {session_id}")
