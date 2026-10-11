"""
GPU-Accelerated ML Pipeline

High-performance ML inference pipeline with:
- Custom CUDA preprocessing kernels
- TensorRT optimization (FP16/INT8)
- Triton Inference Server integration
"""

__version__ = "1.0.0"
__author__ = "Saurabh Rai"

from .preprocessing.pipeline import Pipeline, GPUPreprocessor, PreprocessConfig
from .tensorrt.inference import TensorRTInference
from .tensorrt.builder import TensorRTBuilder, BuildConfig
from .triton.client import TritonClient

__all__ = [
    # Pipeline
    "Pipeline",
    "GPUPreprocessor",
    "PreprocessConfig",
    # TensorRT
    "TensorRTInference",
    "TensorRTBuilder",
    "BuildConfig",
    # Triton
    "TritonClient",
]
