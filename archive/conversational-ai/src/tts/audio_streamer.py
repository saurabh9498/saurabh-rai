"""Audio streaming utilities for TTS output."""

import asyncio
from typing import AsyncIterator, Iterator
import numpy as np


class AudioStreamer:
    """Stream audio chunks for real-time playback."""
    
    def __init__(self, chunk_size_ms: int = 100, sample_rate: int = 22050):
        self.chunk_size = int(chunk_size_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
    
    def chunk_audio(self, audio: np.ndarray) -> Iterator[np.ndarray]:
        for i in range(0, len(audio), self.chunk_size):
            yield audio[i:i + self.chunk_size]
    
    async def stream_audio(self, audio: np.ndarray) -> AsyncIterator[bytes]:
        for chunk in self.chunk_audio(audio):
            audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
            yield audio_bytes
            await asyncio.sleep(len(chunk) / self.sample_rate)
