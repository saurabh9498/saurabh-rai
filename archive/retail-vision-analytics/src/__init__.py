"""
Retail Vision Analytics - Real-Time Video Analytics for Retail.

A production-grade video analytics system for retail environments,
powered by NVIDIA DeepStream and TensorRT.

Features:
- Multi-camera real-time processing (up to 64 streams)
- Customer journey tracking and conversion analytics
- Queue monitoring with wait time estimation
- Traffic heatmap visualization
- Edge deployment on Jetson devices
- Cloud synchronization

Modules:
- vision: Detection and tracking (YOLOv8, ByteTrack)
- analytics: Customer journey, queue, heatmap analysis
- edge: DeepStream pipeline, TensorRT, Jetson utilities
- api: REST API with FastAPI

Example:
    >>> from src.edge import PipelineBuilder, TrackerType
    >>> from src.analytics import CustomerJourneyTracker, QueueMonitor
    >>> 
    >>> # Build pipeline
    >>> pipeline = (PipelineBuilder()
    ...     .with_inference("models/yolov8n_retail_fp16.engine")
    ...     .with_tracker(TrackerType.ByteTrack)
    ...     .add_rtsp_stream("cam1", "rtsp://192.168.1.10/stream")
    ...     .build())
    >>> 
    >>> pipeline.start()

Author: Saurabh
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Saurabh"

from . import vision
from . import analytics
from . import edge
from . import api

__all__ = [
    "vision",
    "analytics", 
    "edge",
    "api",
    "__version__",
]
