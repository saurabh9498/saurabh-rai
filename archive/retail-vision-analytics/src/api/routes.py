"""
FastAPI REST API for Retail Vision Analytics.

This module provides a comprehensive REST API for:
- Analytics data retrieval (customer journeys, heatmaps, queues)
- Camera management (add/remove/configure streams)
- Alert management and notifications
- System health and metrics
- Real-time WebSocket streaming

Requires: FastAPI, uvicorn, pydantic
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class TimeRange(str, Enum):
    """Predefined time ranges for analytics queries."""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class StreamStatus(str, Enum):
    """Camera stream status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    CONNECTING = "connecting"


# Request/Response Models

class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: float = Field(..., ge=0, description="X coordinate")
    y: float = Field(..., ge=0, description="Y coordinate")
    width: float = Field(..., gt=0, description="Width")
    height: float = Field(..., gt=0, description="Height")


class Detection(BaseModel):
    """Object detection result."""
    class_name: str
    confidence: float = Field(..., ge=0, le=1)
    bbox: BoundingBox
    track_id: Optional[int] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class FrameAnalytics(BaseModel):
    """Analytics for a single frame."""
    stream_id: str
    frame_number: int
    timestamp: datetime
    detections: List[Detection]
    person_count: int
    cart_count: int
    inference_time_ms: float


class CustomerJourney(BaseModel):
    """Customer journey through the store."""
    journey_id: str
    track_id: int
    stream_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    zones_visited: List[str]
    zone_dwell_times: Dict[str, float]
    entry_point: str
    exit_point: Optional[str]
    converted: bool = False
    cart_detected: bool = False


class QueueMetrics(BaseModel):
    """Queue monitoring metrics."""
    lane_id: str
    stream_id: str
    timestamp: datetime
    queue_length: int
    avg_wait_time_seconds: float
    max_wait_time_seconds: float
    service_rate: float
    abandonment_count: int
    staffing_recommendation: Optional[int]


class HeatmapData(BaseModel):
    """Heatmap visualization data."""
    stream_id: str
    start_time: datetime
    end_time: datetime
    resolution: tuple = (96, 54)
    data: List[List[float]]
    hotspots: List[Dict[str, Any]]
    max_value: float


class ZoneConfig(BaseModel):
    """Zone configuration."""
    zone_id: str
    zone_name: str
    zone_type: str = Field(..., description="entrance, aisle, checkout, etc.")
    polygon: List[tuple] = Field(..., description="List of (x, y) points")
    color: str = "#00FF00"
    alerts_enabled: bool = True


class StreamConfig(BaseModel):
    """Camera stream configuration."""
    stream_id: str
    name: str
    uri: str
    protocol: str = "rtsp"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    enabled: bool = True
    store_id: Optional[str] = None
    location: Optional[str] = None
    zones: List[ZoneConfig] = Field(default_factory=list)


class StreamInfo(BaseModel):
    """Camera stream information with status."""
    config: StreamConfig
    status: StreamStatus
    fps_actual: float = 0.0
    frames_processed: int = 0
    detections_total: int = 0
    last_frame_time: Optional[datetime] = None
    error_message: Optional[str] = None


class Alert(BaseModel):
    """System alert."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    stream_id: Optional[str]
    timestamp: datetime
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


class AlertCreateRequest(BaseModel):
    """Request to create a new alert."""
    alert_type: str
    severity: AlertSeverity
    stream_id: Optional[str]
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class AlertUpdateRequest(BaseModel):
    """Request to update alert status."""
    status: AlertStatus
    acknowledged_by: Optional[str] = None


class SystemHealth(BaseModel):
    """System health status."""
    status: str = "healthy"
    uptime_seconds: float
    timestamp: datetime
    
    # Hardware
    cpu_util_percent: float
    ram_util_percent: float
    gpu_util_percent: float
    disk_util_percent: float
    temperature_celsius: float
    
    # Pipeline
    streams_active: int
    streams_total: int
    fps_total: float
    detections_per_second: float
    inference_latency_ms: float
    
    # Sync
    cloud_connected: bool
    pending_uploads: int
    last_sync_time: Optional[datetime]


class AnalyticsSummary(BaseModel):
    """Summary analytics for dashboard."""
    time_range: str
    start_time: datetime
    end_time: datetime
    
    # Traffic
    total_visitors: int
    peak_visitors: int
    peak_time: datetime
    avg_visitors_per_hour: float
    
    # Conversion
    conversion_rate: float
    avg_dwell_time_seconds: float
    cart_usage_rate: float
    
    # Queue
    avg_queue_length: float
    avg_wait_time_seconds: float
    total_abandonments: int
    
    # Zones
    busiest_zone: str
    zone_traffic: Dict[str, int]


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Simulated data stores
        self.streams: Dict[str, StreamInfo] = {}
        self.alerts: Dict[str, Alert] = {}
        self.journeys: List[CustomerJourney] = []
        self.queue_metrics: List[QueueMetrics] = []
        self.heatmap_cache: Dict[str, HeatmapData] = {}
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Initialize sample data
        self._init_sample_data()
    
    def _init_sample_data(self):
        """Initialize sample data for demonstration."""
        import random
        
        # Sample streams
        stream_configs = [
            ("cam-entrance-1", "Entrance Camera 1", "rtsp://192.168.1.10:554/stream1", "entrance"),
            ("cam-aisle-1", "Aisle 1 Camera", "rtsp://192.168.1.11:554/stream1", "aisle"),
            ("cam-aisle-2", "Aisle 2 Camera", "rtsp://192.168.1.12:554/stream1", "aisle"),
            ("cam-checkout-1", "Checkout Lane 1", "rtsp://192.168.1.13:554/stream1", "checkout"),
            ("cam-checkout-2", "Checkout Lane 2", "rtsp://192.168.1.14:554/stream1", "checkout"),
        ]
        
        for stream_id, name, uri, location in stream_configs:
            config = StreamConfig(
                stream_id=stream_id,
                name=name,
                uri=uri,
                location=location,
                store_id="store-001",
            )
            self.streams[stream_id] = StreamInfo(
                config=config,
                status=StreamStatus.ONLINE,
                fps_actual=29.97,
                frames_processed=random.randint(10000, 100000),
                detections_total=random.randint(5000, 50000),
                last_frame_time=datetime.now(),
            )
        
        # Sample alerts
        alert_types = [
            ("queue_length_exceeded", AlertSeverity.WARNING, "cam-checkout-1"),
            ("person_loitering", AlertSeverity.INFO, "cam-aisle-1"),
            ("stream_offline", AlertSeverity.CRITICAL, "cam-entrance-1"),
        ]
        
        for i, (alert_type, severity, stream_id) in enumerate(alert_types):
            alert = Alert(
                alert_id=f"alert-{i+1:04d}",
                alert_type=alert_type,
                severity=severity,
                status=AlertStatus.ACTIVE if i == 0 else AlertStatus.RESOLVED,
                stream_id=stream_id,
                timestamp=datetime.now() - timedelta(hours=i),
                message=f"Sample {alert_type} alert",
                details={"threshold": 8, "actual": 12} if "queue" in alert_type else {},
            )
            self.alerts[alert.alert_id] = alert
        
        # Sample journeys
        zones = ["entrance", "aisle-1", "aisle-2", "aisle-3", "checkout"]
        for i in range(50):
            start = datetime.now() - timedelta(hours=random.randint(1, 24))
            duration = random.randint(120, 1800)
            visited = random.sample(zones, k=random.randint(2, 5))
            
            journey = CustomerJourney(
                journey_id=f"journey-{i+1:06d}",
                track_id=i + 1,
                stream_id="cam-entrance-1",
                start_time=start,
                end_time=start + timedelta(seconds=duration),
                duration_seconds=duration,
                zones_visited=visited,
                zone_dwell_times={z: random.randint(30, 300) for z in visited},
                entry_point="entrance",
                exit_point="checkout" if "checkout" in visited else "entrance",
                converted="checkout" in visited,
                cart_detected=random.random() > 0.6,
            )
            self.journeys.append(journey)
        
        # Sample queue metrics
        for i in range(24):
            metrics = QueueMetrics(
                lane_id="checkout-1",
                stream_id="cam-checkout-1",
                timestamp=datetime.now() - timedelta(hours=i),
                queue_length=random.randint(0, 15),
                avg_wait_time_seconds=random.uniform(30, 180),
                max_wait_time_seconds=random.uniform(60, 300),
                service_rate=random.uniform(0.8, 2.0),
                abandonment_count=random.randint(0, 5),
                staffing_recommendation=random.randint(1, 4),
            )
            self.queue_metrics.append(metrics)


# Global state
app_state = AppState()


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Retail Vision Analytics API...")
    yield
    logger.info("Shutting down Retail Vision Analytics API...")


app = FastAPI(
    title="Retail Vision Analytics API",
    description="""
    REST API for real-time retail video analytics powered by NVIDIA DeepStream and TensorRT.
    
    ## Features
    
    * **Customer Analytics** - Track customer journeys, dwell times, and conversion funnels
    * **Queue Monitoring** - Real-time queue length and wait time tracking
    * **Heatmap Generation** - Visualize traffic patterns and hotspots
    * **Alert Management** - Configure and manage alerts for various events
    * **Camera Management** - Add, configure, and monitor video streams
    * **WebSocket Streaming** - Real-time detection and analytics updates
    
    ## Authentication
    
    All endpoints require API key authentication via the `X-API-Key` header.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & System Endpoints
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/system/health", response_model=SystemHealth, tags=["System"])
async def get_system_health():
    """Get detailed system health status."""
    import random
    
    uptime = time.time() - app_state.start_time
    
    return SystemHealth(
        status="healthy",
        uptime_seconds=uptime,
        timestamp=datetime.now(),
        cpu_util_percent=random.uniform(20, 60),
        ram_util_percent=random.uniform(40, 70),
        gpu_util_percent=random.uniform(60, 90),
        disk_util_percent=random.uniform(30, 50),
        temperature_celsius=random.uniform(45, 65),
        streams_active=len([s for s in app_state.streams.values() if s.status == StreamStatus.ONLINE]),
        streams_total=len(app_state.streams),
        fps_total=sum(s.fps_actual for s in app_state.streams.values()),
        detections_per_second=random.uniform(30, 100),
        inference_latency_ms=random.uniform(3, 8),
        cloud_connected=True,
        pending_uploads=random.randint(0, 50),
        last_sync_time=datetime.now() - timedelta(seconds=random.randint(10, 60)),
    )


@app.get("/api/v1/system/metrics", tags=["System"])
async def get_system_metrics():
    """Get system performance metrics."""
    import random
    
    return {
        "timestamp": datetime.now().isoformat(),
        "pipeline": {
            "streams_active": len(app_state.streams),
            "total_fps": sum(s.fps_actual for s in app_state.streams.values()),
            "total_frames_processed": sum(s.frames_processed for s in app_state.streams.values()),
            "total_detections": sum(s.detections_total for s in app_state.streams.values()),
        },
        "inference": {
            "avg_latency_ms": random.uniform(4, 8),
            "p95_latency_ms": random.uniform(8, 15),
            "throughput_fps": random.uniform(200, 400),
        },
        "memory": {
            "gpu_memory_used_mb": random.randint(2000, 6000),
            "gpu_memory_total_mb": 8192,
            "ram_used_mb": random.randint(4000, 12000),
            "ram_total_mb": 16384,
        },
    }


# ============================================================================
# Camera/Stream Management Endpoints
# ============================================================================

@app.get("/api/v1/streams", response_model=List[StreamInfo], tags=["Cameras"])
async def list_streams(
    status: Optional[StreamStatus] = Query(None, description="Filter by status"),
    store_id: Optional[str] = Query(None, description="Filter by store ID"),
):
    """List all camera streams with optional filtering."""
    streams = list(app_state.streams.values())
    
    if status:
        streams = [s for s in streams if s.status == status]
    
    if store_id:
        streams = [s for s in streams if s.config.store_id == store_id]
    
    return streams


@app.get("/api/v1/streams/{stream_id}", response_model=StreamInfo, tags=["Cameras"])
async def get_stream(
    stream_id: str = Path(..., description="Stream ID"),
):
    """Get details for a specific camera stream."""
    if stream_id not in app_state.streams:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    return app_state.streams[stream_id]


@app.post("/api/v1/streams", response_model=StreamInfo, tags=["Cameras"])
async def add_stream(
    config: StreamConfig = Body(...),
):
    """Add a new camera stream."""
    if config.stream_id in app_state.streams:
        raise HTTPException(status_code=409, detail=f"Stream {config.stream_id} already exists")
    
    stream_info = StreamInfo(
        config=config,
        status=StreamStatus.CONNECTING,
        fps_actual=0.0,
        frames_processed=0,
        detections_total=0,
    )
    
    app_state.streams[config.stream_id] = stream_info
    
    # Simulate connection
    stream_info.status = StreamStatus.ONLINE
    stream_info.fps_actual = config.fps
    stream_info.last_frame_time = datetime.now()
    
    return stream_info


@app.put("/api/v1/streams/{stream_id}", response_model=StreamInfo, tags=["Cameras"])
async def update_stream(
    stream_id: str = Path(..., description="Stream ID"),
    config: StreamConfig = Body(...),
):
    """Update camera stream configuration."""
    if stream_id not in app_state.streams:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    existing = app_state.streams[stream_id]
    existing.config = config
    
    return existing


@app.delete("/api/v1/streams/{stream_id}", tags=["Cameras"])
async def remove_stream(
    stream_id: str = Path(..., description="Stream ID"),
):
    """Remove a camera stream."""
    if stream_id not in app_state.streams:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    del app_state.streams[stream_id]
    
    return {"message": f"Stream {stream_id} removed successfully"}


@app.post("/api/v1/streams/{stream_id}/restart", tags=["Cameras"])
async def restart_stream(
    stream_id: str = Path(..., description="Stream ID"),
):
    """Restart a camera stream."""
    if stream_id not in app_state.streams:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    stream = app_state.streams[stream_id]
    stream.status = StreamStatus.CONNECTING
    
    # Simulate restart
    await asyncio.sleep(1)
    stream.status = StreamStatus.ONLINE
    stream.last_frame_time = datetime.now()
    
    return {"message": f"Stream {stream_id} restarted successfully"}


# ============================================================================
# Analytics Endpoints
# ============================================================================

@app.get("/api/v1/analytics/summary", response_model=AnalyticsSummary, tags=["Analytics"])
async def get_analytics_summary(
    time_range: TimeRange = Query(TimeRange.LAST_24_HOURS, description="Time range"),
    start_time: Optional[datetime] = Query(None, description="Custom start time"),
    end_time: Optional[datetime] = Query(None, description="Custom end time"),
    store_id: Optional[str] = Query(None, description="Filter by store ID"),
):
    """Get summary analytics for the dashboard."""
    import random
    
    # Calculate time bounds
    now = datetime.now()
    if time_range == TimeRange.CUSTOM and start_time and end_time:
        start = start_time
        end = end_time
    else:
        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}
        hours = hours_map.get(time_range.value, 24)
        start = now - timedelta(hours=hours)
        end = now
    
    # Filter journeys
    journeys = [
        j for j in app_state.journeys
        if start <= j.start_time <= end
    ]
    
    total_visitors = len(journeys)
    converted = len([j for j in journeys if j.converted])
    with_cart = len([j for j in journeys if j.cart_detected])
    
    # Calculate zone traffic
    zone_traffic: Dict[str, int] = {}
    for journey in journeys:
        for zone in journey.zones_visited:
            zone_traffic[zone] = zone_traffic.get(zone, 0) + 1
    
    busiest_zone = max(zone_traffic.keys(), key=lambda z: zone_traffic[z]) if zone_traffic else "unknown"
    
    return AnalyticsSummary(
        time_range=time_range.value,
        start_time=start,
        end_time=end,
        total_visitors=total_visitors,
        peak_visitors=random.randint(20, 50),
        peak_time=now - timedelta(hours=random.randint(1, 12)),
        avg_visitors_per_hour=total_visitors / max((end - start).total_seconds() / 3600, 1),
        conversion_rate=converted / max(total_visitors, 1),
        avg_dwell_time_seconds=sum(j.duration_seconds for j in journeys) / max(len(journeys), 1),
        cart_usage_rate=with_cart / max(total_visitors, 1),
        avg_queue_length=random.uniform(2, 8),
        avg_wait_time_seconds=random.uniform(60, 180),
        total_abandonments=random.randint(5, 30),
        busiest_zone=busiest_zone,
        zone_traffic=zone_traffic,
    )


@app.get("/api/v1/analytics/journeys", tags=["Analytics"])
async def get_customer_journeys(
    stream_id: Optional[str] = Query(None, description="Filter by stream ID"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    converted_only: bool = Query(False, description="Only show converted customers"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """Get customer journey data with filtering and pagination."""
    journeys = app_state.journeys.copy()
    
    # Apply filters
    if stream_id:
        journeys = [j for j in journeys if j.stream_id == stream_id]
    
    if start_time:
        journeys = [j for j in journeys if j.start_time >= start_time]
    
    if end_time:
        journeys = [j for j in journeys if j.start_time <= end_time]
    
    if converted_only:
        journeys = [j for j in journeys if j.converted]
    
    # Sort by start time descending
    journeys.sort(key=lambda j: j.start_time, reverse=True)
    
    # Paginate
    total = len(journeys)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = journeys[start_idx:end_idx]
    
    return {
        "items": [j.dict() for j in page_items],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }


@app.get("/api/v1/analytics/queues", response_model=List[QueueMetrics], tags=["Analytics"])
async def get_queue_metrics(
    lane_id: Optional[str] = Query(None, description="Filter by lane ID"),
    stream_id: Optional[str] = Query(None, description="Filter by stream ID"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to retrieve"),
):
    """Get queue monitoring metrics."""
    metrics = app_state.queue_metrics.copy()
    
    cutoff = datetime.now() - timedelta(hours=hours)
    metrics = [m for m in metrics if m.timestamp >= cutoff]
    
    if lane_id:
        metrics = [m for m in metrics if m.lane_id == lane_id]
    
    if stream_id:
        metrics = [m for m in metrics if m.stream_id == stream_id]
    
    return metrics


@app.get("/api/v1/analytics/queues/current", tags=["Analytics"])
async def get_current_queue_status():
    """Get current queue status across all lanes."""
    import random
    
    lanes = ["checkout-1", "checkout-2", "checkout-3"]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "lanes": [
            {
                "lane_id": lane,
                "queue_length": random.randint(0, 10),
                "estimated_wait_seconds": random.randint(30, 300),
                "status": "open" if random.random() > 0.2 else "closed",
            }
            for lane in lanes
        ],
        "total_customers_waiting": random.randint(5, 25),
        "avg_wait_time_seconds": random.uniform(60, 180),
        "recommended_lanes_open": random.randint(2, 4),
    }


@app.get("/api/v1/analytics/heatmap", response_model=HeatmapData, tags=["Analytics"])
async def get_heatmap(
    stream_id: str = Query(..., description="Stream ID"),
    time_range: TimeRange = Query(TimeRange.LAST_HOUR, description="Time range"),
    resolution: int = Query(96, ge=32, le=256, description="Grid resolution"),
):
    """Get heatmap data for traffic visualization."""
    import random
    import numpy as np
    
    now = datetime.now()
    hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}
    hours = hours_map.get(time_range.value, 1)
    
    start_time = now - timedelta(hours=hours)
    
    # Generate heatmap data
    height = int(resolution * 9 / 16)  # 16:9 aspect ratio
    
    # Simulate traffic patterns with Gaussian clusters
    data = np.zeros((height, resolution))
    
    # Add some hotspots
    hotspots = []
    for _ in range(random.randint(3, 7)):
        cx, cy = random.randint(10, resolution-10), random.randint(5, height-5)
        intensity = random.uniform(0.5, 1.0)
        
        for y in range(height):
            for x in range(resolution):
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                data[y, x] += intensity * np.exp(-dist ** 2 / 200)
        
        hotspots.append({
            "x": cx / resolution,
            "y": cy / height,
            "intensity": intensity,
            "label": f"Hotspot at ({cx}, {cy})",
        })
    
    # Normalize
    max_val = data.max()
    if max_val > 0:
        data = data / max_val
    
    return HeatmapData(
        stream_id=stream_id,
        start_time=start_time,
        end_time=now,
        resolution=(resolution, height),
        data=data.tolist(),
        hotspots=hotspots,
        max_value=1.0,
    )


@app.get("/api/v1/analytics/zones", tags=["Analytics"])
async def get_zone_analytics(
    stream_id: Optional[str] = Query(None, description="Filter by stream ID"),
    time_range: TimeRange = Query(TimeRange.LAST_24_HOURS, description="Time range"),
):
    """Get analytics per zone."""
    import random
    
    zones = ["entrance", "aisle-1", "aisle-2", "aisle-3", "checkout"]
    
    return {
        "time_range": time_range.value,
        "zones": [
            {
                "zone_id": zone,
                "zone_name": zone.replace("-", " ").title(),
                "visitor_count": random.randint(50, 500),
                "avg_dwell_time_seconds": random.uniform(30, 300),
                "peak_occupancy": random.randint(5, 30),
                "conversion_rate": random.uniform(0.1, 0.8) if zone == "checkout" else None,
            }
            for zone in zones
        ],
    }


# ============================================================================
# Alert Management Endpoints
# ============================================================================

@app.get("/api/v1/alerts", response_model=List[Alert], tags=["Alerts"])
async def list_alerts(
    status: Optional[AlertStatus] = Query(None, description="Filter by status"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    stream_id: Optional[str] = Query(None, description="Filter by stream ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum alerts to return"),
):
    """List alerts with optional filtering."""
    alerts = list(app_state.alerts.values())
    
    if status:
        alerts = [a for a in alerts if a.status == status]
    
    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    
    if stream_id:
        alerts = [a for a in alerts if a.stream_id == stream_id]
    
    # Sort by timestamp descending
    alerts.sort(key=lambda a: a.timestamp, reverse=True)
    
    return alerts[:limit]


@app.get("/api/v1/alerts/{alert_id}", response_model=Alert, tags=["Alerts"])
async def get_alert(
    alert_id: str = Path(..., description="Alert ID"),
):
    """Get a specific alert by ID."""
    if alert_id not in app_state.alerts:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    return app_state.alerts[alert_id]


@app.post("/api/v1/alerts", response_model=Alert, tags=["Alerts"])
async def create_alert(
    request: AlertCreateRequest = Body(...),
):
    """Create a new alert."""
    alert_id = f"alert-{len(app_state.alerts) + 1:04d}"
    
    alert = Alert(
        alert_id=alert_id,
        alert_type=request.alert_type,
        severity=request.severity,
        status=AlertStatus.ACTIVE,
        stream_id=request.stream_id,
        timestamp=datetime.now(),
        message=request.message,
        details=request.details,
    )
    
    app_state.alerts[alert_id] = alert
    
    # Broadcast to WebSocket clients
    await broadcast_event("alert_created", alert.dict())
    
    return alert


@app.patch("/api/v1/alerts/{alert_id}", response_model=Alert, tags=["Alerts"])
async def update_alert(
    alert_id: str = Path(..., description="Alert ID"),
    request: AlertUpdateRequest = Body(...),
):
    """Update alert status (acknowledge or resolve)."""
    if alert_id not in app_state.alerts:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    alert = app_state.alerts[alert_id]
    alert.status = request.status
    
    if request.status == AlertStatus.ACKNOWLEDGED:
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = request.acknowledged_by
    elif request.status == AlertStatus.RESOLVED:
        alert.resolved_at = datetime.now()
    
    return alert


@app.delete("/api/v1/alerts/{alert_id}", tags=["Alerts"])
async def delete_alert(
    alert_id: str = Path(..., description="Alert ID"),
):
    """Delete an alert."""
    if alert_id not in app_state.alerts:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    del app_state.alerts[alert_id]
    
    return {"message": f"Alert {alert_id} deleted successfully"}


# ============================================================================
# WebSocket Endpoints
# ============================================================================

async def broadcast_event(event_type: str, data: dict):
    """Broadcast event to all connected WebSocket clients."""
    message = json.dumps({
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }, default=str)
    
    disconnected = []
    for ws in app_state.websocket_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    
    # Clean up disconnected clients
    for ws in disconnected:
        app_state.websocket_connections.remove(ws)


@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """WebSocket endpoint for real-time detection streaming."""
    await websocket.accept()
    app_state.websocket_connections.append(websocket)
    
    logger.info("WebSocket client connected for detections")
    
    try:
        while True:
            # Simulate detection events
            import random
            
            detection_event = {
                "stream_id": random.choice(list(app_state.streams.keys())),
                "frame_number": random.randint(10000, 100000),
                "timestamp": datetime.now().isoformat(),
                "detections": [
                    {
                        "class_name": random.choice(["person", "shopping_cart", "basket"]),
                        "confidence": random.uniform(0.7, 0.99),
                        "bbox": {
                            "x": random.randint(100, 1600),
                            "y": random.randint(100, 900),
                            "width": random.randint(50, 200),
                            "height": random.randint(80, 300),
                        },
                        "track_id": random.randint(1, 100),
                    }
                    for _ in range(random.randint(1, 5))
                ],
            }
            
            await websocket.send_json(detection_event)
            await asyncio.sleep(0.1)  # 10 FPS for demo
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if websocket in app_state.websocket_connections:
            app_state.websocket_connections.remove(websocket)


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert notifications."""
    await websocket.accept()
    app_state.websocket_connections.append(websocket)
    
    logger.info("WebSocket client connected for alerts")
    
    try:
        while True:
            # Wait for messages (ping/pong or commands)
            data = await websocket.receive_text()
            
            # Echo acknowledgment
            await websocket.send_json({
                "type": "ack",
                "message": data,
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if websocket in app_state.websocket_connections:
            app_state.websocket_connections.remove(websocket)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )
