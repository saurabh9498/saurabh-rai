"""
Retail Vision Analytics - Analytics Module.

Customer behavior analytics including journey tracking,
queue monitoring, and heatmap generation.

Components:
- customer_journey: Zone-based journey tracking and conversion funnels
- queue_monitor: Real-time queue length and wait time estimation
- heatmap: Traffic visualization and hotspot detection
"""

from .customer_journey import (
    CustomerJourneyTracker,
    JourneyConfig,
    CustomerJourney,
    ZoneConfig,
    ZoneTransition,
    ConversionFunnel,
)

from .queue_monitor import (
    QueueMonitor,
    QueueConfig,
    QueueMetrics,
    QueueAlert,
    LaneConfig,
)

from .heatmap import (
    HeatmapGenerator,
    HeatmapConfig,
    HeatmapData,
    Hotspot,
    FlowField,
)

__all__ = [
    # Customer Journey
    "CustomerJourneyTracker",
    "JourneyConfig",
    "CustomerJourney",
    "ZoneConfig",
    "ZoneTransition",
    "ConversionFunnel",
    # Queue Monitor
    "QueueMonitor",
    "QueueConfig",
    "QueueMetrics",
    "QueueAlert",
    "LaneConfig",
    # Heatmap
    "HeatmapGenerator",
    "HeatmapConfig",
    "HeatmapData",
    "Hotspot",
    "FlowField",
]
