"""
TensorRT optimization and inference module.

Provides:
- TensorRTBuilder: Build optimized engines from ONNX
- TensorRTInference: Run inference with TensorRT engines
- Calibrators: INT8 calibration utilities
- Optimization: Graph optimization utilities
"""

from .builder import TensorRTBuilder, BuildConfig, OptimizationProfile
from .inference import TensorRTInference, benchmark_engine
from .calibrator import (
    EntropyCalibrator,
    MinMaxCalibrator,
    CalibrationDataLoader,
    calibrate_int8,
)
from .optimization import (
    NetworkAnalyzer,
    PrecisionOptimizer,
    optimize_for_inference,
)

__all__ = [
    "TensorRTBuilder",
    "BuildConfig",
    "OptimizationProfile",
    "TensorRTInference",
    "benchmark_engine",
    "EntropyCalibrator",
    "MinMaxCalibrator",
    "CalibrationDataLoader",
    "calibrate_int8",
    "NetworkAnalyzer",
    "PrecisionOptimizer",
    "optimize_for_inference",
]
