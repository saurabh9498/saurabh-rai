"""Audio preprocessing utilities."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Simple linear interpolation
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


def apply_noise_reduction(audio: np.ndarray, noise_floor: float = 0.01) -> np.ndarray:
    """Apply simple noise gate."""
    mask = np.abs(audio) > noise_floor
    return audio * mask


class AudioProcessor:
    """Audio preprocessing pipeline."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio through preprocessing pipeline."""
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            audio = resample(audio, sample_rate, self.config.sample_rate)
        
        # Normalize
        audio = normalize_audio(audio)
        
        return audio
