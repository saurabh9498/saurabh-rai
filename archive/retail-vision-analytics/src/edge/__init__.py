"""
Retail Vision Analytics - Edge Computing Module.

This module provides edge deployment components for NVIDIA DeepStream
and Jetson platforms including:
- TensorRT engine optimization
- DeepStream pipeline management
- Jetson device utilities
- Edge-cloud synchronization

Typical usage:
    from src.edge import TensorRTEngineBuilder, DeepStreamPipeline
    from src.edge.jetson_utils import JetsonDevice
"""

from .tensorrt_engine import (
    TensorRTEngineBuilder,
    TensorRTInference,
    EngineConfig,
    Precision,
)
from .deepstream_app import (
    DeepStreamPipeline,
    PipelineBuilder,
    PipelineConfig,
    StreamConfig,
)
from .jetson_utils import (
    JetsonDevice,
    JetsonMonitor,
    PowerMode,
)
from .sync_manager import (
    SyncManager,
    SyncConfig,
)

__all__ = [
    # TensorRT
    "TensorRTEngineBuilder",
    "TensorRTInference", 
    "EngineConfig",
    "Precision",
    # DeepStream
    "DeepStreamPipeline",
    "PipelineBuilder",
    "PipelineConfig",
    "StreamConfig",
    # Jetson
    "JetsonDevice",
    "JetsonMonitor",
    "PowerMode",
    # Sync
    "SyncManager",
    "SyncConfig",
]
