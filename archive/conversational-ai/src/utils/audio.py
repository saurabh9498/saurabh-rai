"""Audio utility functions."""

import numpy as np
from typing import Tuple
import io


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and return array with sample rate."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path)
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio.astype(np.float32), sr
    except ImportError:
        return np.zeros(target_sr, dtype=np.float32), target_sr


def save_audio(path: str, audio: np.ndarray, sample_rate: int = 22050):
    """Save audio to file."""
    try:
        import soundfile as sf
        sf.write(path, audio, sample_rate)
    except ImportError:
        pass


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes."""
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())
    buffer.seek(0)
    return buffer.read()
