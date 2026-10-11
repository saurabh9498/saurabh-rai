"""
INT8 calibration for the TensorRT build — KL-divergence (entropy) calibration.

Requires an NVIDIA GPU with TensorRT 8.6 and pycuda; NOT run in this repo's CPU checks.
(TensorRT 10 deprecates this implicit-calibration API in favour of explicit Q/DQ
quantization from TensorRT Model Optimizer.)

Why entropy calibration: naive INT8 scaling (max-abs) lets rare activation outliers
stretch the scale and crush the resolution of common values, which costs ranking
accuracy (AUC / normalized entropy). TensorRT's IInt8EntropyCalibrator2 builds a
histogram of each activation over real traffic and picks the clipping threshold that
minimises the KL divergence between the FP32 and quantised distributions.

The calibration set must be real, recent bid traffic (a few thousand requests sampled
across time of day). Synthetic data gives scales that do not match production.
"""
from __future__ import annotations

import os
from typing import Iterator

import numpy as np

try:  # GPU-only imports
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError as e:  # pragma: no cover - documented GPU requirement
    raise SystemExit(f"calibrator.py needs TensorRT + pycuda on a GPU host: {e}")


class BidTrafficCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds batches of (dense, embeddings) tensors captured from real traffic."""

    def __init__(self, batches: Iterator[dict[str, np.ndarray]], batch_size: int, cache_file: str):
        super().__init__()
        self._batches = batches
        self._batch_size = batch_size
        self._cache_file = cache_file
        self._device = {}

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names):
        try:
            host = next(self._batches)
        except StopIteration:
            return None
        ptrs = []
        for name in names:
            arr = np.ascontiguousarray(host[name], dtype=np.float32)
            if name not in self._device:
                self._device[name] = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(self._device[name], arr)
            ptrs.append(int(self._device[name]))
        return ptrs

    def read_calibration_cache(self):
        # Reusing the cache keeps scales stable across the scheduled rebuilds unless the
        # traffic distribution has actually shifted.
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache) -> None:
        with open(self._cache_file, "wb") as f:
            f.write(cache)
