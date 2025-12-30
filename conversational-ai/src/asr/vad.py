"""
Voice Activity Detection (VAD)

Detects speech segments in audio for endpoint detection and noise filtering.
Uses Silero VAD for high accuracy with low latency.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    window_size_samples: int = 512  # 32ms at 16kHz
    sample_rate: int = 16000
    
    # Model settings
    model_path: Optional[str] = None
    use_onnx: bool = False


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    start: float  # seconds
    end: float    # seconds
    confidence: float


class VoiceActivityDetector:
    """
    Voice Activity Detection using Silero VAD.
    
    Features:
    - High accuracy (>95% on LibriSpeech)
    - Low latency (<10ms per frame)
    - Works with 8kHz and 16kHz audio
    - ONNX support for faster inference
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.model = None
        self._state = None
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model."""
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=self.config.use_onnx,
            )
            
            self.get_speech_timestamps = utils[0]
            self.save_audio = utils[1]
            self.read_audio = utils[2]
            self.VADIterator = utils[3]
            self.collect_chunks = utils[4]
            
            logger.info("Loaded Silero VAD model")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}, using energy-based fallback")
            self.model = None
    
    def is_speech(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> bool:
        """
        Check if audio chunk contains speech.
        
        Args:
            audio: Audio samples (1D array)
            sample_rate: Sample rate (default: 16000)
            
        Returns:
            True if speech detected
        """
        sample_rate = sample_rate or self.config.sample_rate
        
        if self.model is None:
            return self._energy_based_vad(audio)
        
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio, sample_rate).item()
        
        return speech_prob > self.config.threshold
    
    def get_speech_probability(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> float:
        """Get speech probability for audio chunk."""
        sample_rate = sample_rate or self.config.sample_rate
        
        if self.model is None:
            return self._energy_based_vad_prob(audio)
        
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        with torch.no_grad():
            return self.model(audio, sample_rate).item()
    
    def get_speech_segments(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> List[SpeechSegment]:
        """
        Get all speech segments in audio.
        
        Args:
            audio: Full audio (1D array)
            sample_rate: Sample rate
            
        Returns:
            List of SpeechSegment with start/end times
        """
        sample_rate = sample_rate or self.config.sample_rate
        
        if self.model is None:
            return self._energy_based_segmentation(audio, sample_rate)
        
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Get speech timestamps using Silero utility
        timestamps = self.get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=sample_rate,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
        )
        
        segments = []
        for ts in timestamps:
            segment = SpeechSegment(
                start=ts['start'] / sample_rate,
                end=ts['end'] / sample_rate,
                confidence=0.9,  # Silero doesn't provide per-segment confidence
            )
            segments.append(segment)
        
        return segments
    
    def _energy_based_vad(self, audio: np.ndarray) -> bool:
        """Simple energy-based VAD fallback."""
        energy = np.sqrt(np.mean(audio ** 2))
        threshold = 0.01  # Adjust based on typical noise floor
        return energy > threshold
    
    def _energy_based_vad_prob(self, audio: np.ndarray) -> float:
        """Energy-based speech probability."""
        energy = np.sqrt(np.mean(audio ** 2))
        # Simple sigmoid mapping
        prob = 1 / (1 + np.exp(-100 * (energy - 0.01)))
        return float(prob)
    
    def _energy_based_segmentation(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[SpeechSegment]:
        """Simple energy-based segmentation."""
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_size = int(0.010 * sample_rate)    # 10ms hop
        
        # Compute frame energies
        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        # Threshold
        threshold = np.mean(energies) * 0.5
        is_speech = [e > threshold for e in energies]
        
        # Find segments
        segments = []
        in_speech = False
        start = 0
        
        for i, speech in enumerate(is_speech):
            time = i * hop_size / sample_rate
            
            if speech and not in_speech:
                start = time
                in_speech = True
            elif not speech and in_speech:
                segments.append(SpeechSegment(
                    start=start,
                    end=time,
                    confidence=0.7,
                ))
                in_speech = False
        
        # Handle trailing speech
        if in_speech:
            segments.append(SpeechSegment(
                start=start,
                end=len(audio) / sample_rate,
                confidence=0.7,
            ))
        
        return segments
    
    def reset(self):
        """Reset VAD state for new stream."""
        self._state = None


class VADIterator:
    """
    Streaming VAD iterator for real-time processing.
    
    Processes audio in chunks and yields speech/silence events.
    """
    
    def __init__(
        self,
        vad: VoiceActivityDetector,
        sample_rate: int = 16000,
    ):
        self.vad = vad
        self.sample_rate = sample_rate
        self._is_speaking = False
        self._speech_start = 0.0
        self._current_time = 0.0
        self._min_speech_samples = int(0.25 * sample_rate)  # 250ms
        self._min_silence_samples = int(0.5 * sample_rate)  # 500ms
        self._speech_buffer = []
        self._silence_count = 0
    
    def process(
        self,
        audio_chunk: np.ndarray,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Process audio chunk and return speech events.
        
        Args:
            audio_chunk: Audio samples
            
        Returns:
            Tuple of (event_type, start_time, end_time) or None
            event_type: 'speech_start', 'speech_end'
        """
        is_speech = self.vad.is_speech(audio_chunk)
        chunk_duration = len(audio_chunk) / self.sample_rate
        
        event = None
        
        if is_speech:
            self._silence_count = 0
            
            if not self._is_speaking:
                # Speech started
                self._is_speaking = True
                self._speech_start = self._current_time
                event = ('speech_start', self._current_time, None)
        else:
            if self._is_speaking:
                self._silence_count += len(audio_chunk)
                
                if self._silence_count >= self._min_silence_samples:
                    # Speech ended
                    self._is_speaking = False
                    event = ('speech_end', self._speech_start, self._current_time)
                    self._silence_count = 0
        
        self._current_time += chunk_duration
        return event
    
    def reset(self):
        """Reset iterator state."""
        self._is_speaking = False
        self._speech_start = 0.0
        self._current_time = 0.0
        self._silence_count = 0
        self.vad.reset()
