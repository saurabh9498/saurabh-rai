"""
Retail Vision Analytics - Vision Module.

Object detection and tracking components using YOLOv8 and ByteTrack.

Components:
- detector: YOLOv8 + TensorRT inference engine
- tracker: ByteTrack multi-object tracker with Kalman filtering
"""

from .detector import (
    RetailDetector,
    DetectorConfig,
    Detection,
    DetectionResult,
)

from .tracker import (
    ByteTracker,
    SORTTracker,
    TrackerConfig,
    Track,
    TrackState,
)

__all__ = [
    # Detector
    "RetailDetector",
    "DetectorConfig", 
    "Detection",
    "DetectionResult",
    # Tracker
    "ByteTracker",
    "SORTTracker",
    "TrackerConfig",
    "Track",
    "TrackState",
]
