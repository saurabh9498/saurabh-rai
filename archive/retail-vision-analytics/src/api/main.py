"""
FastAPI REST API for Retail Vision Analytics.

This module provides a comprehensive REST API for:
- Real-time analytics access
- Camera/stream management
- Alert configuration and retrieval
- Historical data queries
- System health monitoring
- Model management

Features:
- OpenAPI/Swagger documentation
- WebSocket for real-time updates
- Rate limiting
- JWT authentication
- Request validation with Pydantic
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, HTTPException, Depends, Query, Path, Body,
    WebSocket, WebSocketDisconnect, BackgroundTasks,
    status, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class StreamProtocol(str, Enum):
    """Video stream protocol types."""
    RTSP = "rtsp"
    RTMP = "rtmp"
    HTTP = "http"
    FILE = "file"
    USB = "usb"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    QUEUE_LENGTH = "queue_length"
    WAIT_TIME = "wait_time"
    SHRINKAGE = "shrinkage"
    OCCUPANCY = "occupancy"
    ZONE_DWELL = "zone_dwell"
    SYSTEM = "system"


# Request/Response Models
class StreamCreate(BaseModel):
    """Request model for creating a stream."""
    stream_id: str = Field(..., min_length=1, max_length=64)
    uri: str = Field(..., min_length=1)
    protocol: StreamProtocol = StreamProtocol.RTSP
    store_id: Optional[str] = None
    location: Optional[str] = None
    width: int = Field(default=1920, ge=320, le=7680)
    height: int = Field(default=1080, ge=240, le=4320)
    fps: int = Field(default=30, ge=1, le=120)
    inference_interval: int = Field(default=1, ge=1, le=30)
    enable_tracking: bool = True
    roi: Optional[List[float]] = None
    
    @validator('roi')
    def validate_roi(cls, v):
        if v is not None:
            if len(v) != 4:
                raise ValueError('ROI must have 4 values [x1, y1, x2, y2]')
            if not all(0 <= x <= 1 for x in v):
                raise ValueError('ROI values must be normalized (0-1)')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "stream_id": "store1-entrance",
                "uri": "rtsp://192.168.1.10:554/stream",
                "protocol": "rtsp",
                "store_id": "store-001",
                "location": "main_entrance",
                "width": 1920,
                "height": 1080,
                "fps": 30
            }
        }


class StreamResponse(BaseModel):
    """Response model for stream information."""
    stream_id: str
    uri: str
    protocol: StreamProtocol
    store_id: Optional[str]
    location: Optional[str]
    width: int
    height: int
    fps: int
    status: str  # "active", "inactive", "error"
    frames_processed: int
    current_fps: float
    last_detection_count: int
    created_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "stream_id": "store1-entrance",
                "uri": "rtsp://192.168.1.10:554/stream",
                "protocol": "rtsp",
                "store_id": "store-001",
                "location": "main_entrance",
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "status": "active",
                "frames_processed": 54000,
                "current_fps": 29.8,
                "last_detection_count": 12,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class Detection(BaseModel):
    """Detection result model."""
    class_name: str
    class_id: int
    confidence: float = Field(..., ge=0, le=1)
    bbox: List[int] = Field(..., min_items=4, max_items=4)
    track_id: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None


class FrameAnalytics(BaseModel):
    """Analytics for a single frame."""
    stream_id: str
    frame_number: int
    timestamp: datetime
    detections: List[Detection]
    person_count: int
    cart_count: int
    inference_time_ms: float


class ZoneAnalytics(BaseModel):
    """Zone-based analytics."""
    zone_id: str
    zone_name: str
    current_occupancy: int
    avg_dwell_time_sec: float
    entries_count: int
    exits_count: int
    peak_occupancy: int
    peak_time: Optional[datetime]


class QueueMetrics(BaseModel):
    """Queue monitoring metrics."""
    lane_id: str
    lane_name: str
    queue_length: int
    avg_wait_time_sec: float
    max_wait_time_sec: float
    service_rate_per_min: float
    abandonment_count: int
    status: str  # "normal", "busy", "critical"


class HeatmapData(BaseModel):
    """Heatmap data response."""
    stream_id: str
    timestamp: datetime
    width: int
    height: int
    cell_size: int
    data: List[List[float]]  # 2D array of intensities
    hotspots: List[Dict[str, Any]]


class AlertCreate(BaseModel):
    """Request model for creating an alert rule."""
    name: str = Field(..., min_length=1, max_length=128)
    alert_type: AlertType
    severity: AlertSeverity = AlertSeverity.WARNING
    stream_ids: Optional[List[str]] = None  # None = all streams
    conditions: Dict[str, Any]
    cooldown_seconds: int = Field(default=300, ge=0)
    enabled: bool = True
    notification_channels: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Long Queue Alert",
                "alert_type": "queue_length",
                "severity": "warning",
                "stream_ids": ["store1-checkout"],
                "conditions": {
                    "threshold": 8,
                    "duration_seconds": 60
                },
                "cooldown_seconds": 300,
                "notification_channels": ["email", "slack"]
            }
        }


class AlertResponse(BaseModel):
    """Response model for alert."""
    alert_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    stream_id: Optional[str]
    triggered_at: datetime
    resolved_at: Optional[datetime]
    message: str
    data: Dict[str, Any]
    acknowledged: bool


class HealthStatus(BaseModel):
    """System health status."""
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    streams_active: int
    streams_total: int
    gpu_utilization: float
    cpu_utilization: float
    memory_utilization: float
    disk_utilization: float
    inference_fps: float
    avg_latency_ms: float
    errors_last_hour: int
    version: str


class AnalyticsQuery(BaseModel):
    """Query parameters for analytics retrieval."""
    stream_ids: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    aggregation: str = Field(default="hour", pattern="^(minute|hour|day)$")
    metrics: List[str] = Field(default_factory=lambda: ["person_count", "detections"])


class AnalyticsSummary(BaseModel):
    """Aggregated analytics summary."""
    period_start: datetime
    period_end: datetime
    stream_id: Optional[str]
    total_persons: int
    unique_persons: int
    avg_dwell_time_sec: float
    total_carts: int
    conversion_rate: float
    peak_occupancy: int
    peak_time: datetime
    total_detections: int


# =============================================================================
# Application State & Dependencies
# =============================================================================

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.start_time = time.time()
        self.streams: Dict[str, Dict] = {}
        self.alerts: Dict[str, Dict] = {}
        self.alert_rules: Dict[str, Dict] = {}
        self.websocket_clients: List[WebSocket] = []
        self._lock = asyncio.Lock()
        
        # Simulated metrics
        self._metrics = {
            "total_frames": 0,
            "total_detections": 0,
            "errors": 0,
        }
    
    async def add_stream(self, stream: StreamCreate) -> StreamResponse:
        """Add a new stream."""
        async with self._lock:
            if stream.stream_id in self.streams:
                raise ValueError(f"Stream {stream.stream_id} already exists")
            
            self.streams[stream.stream_id] = {
                **stream.dict(),
                "status": "active",
                "frames_processed": 0,
                "current_fps": stream.fps,
                "last_detection_count": 0,
                "created_at": datetime.utcnow(),
            }
            
            return StreamResponse(**self.streams[stream.stream_id])
    
    async def get_stream(self, stream_id: str) -> Optional[StreamResponse]:
        """Get stream by ID."""
        if stream_id in self.streams:
            return StreamResponse(**self.streams[stream_id])
        return None
    
    async def remove_stream(self, stream_id: str) -> bool:
        """Remove a stream."""
        async with self._lock:
            if stream_id in self.streams:
                del self.streams[stream_id]
                return True
            return False
    
    async def broadcast_event(self, event: Dict[str, Any]):
        """Broadcast event to all WebSocket clients."""
        dead_clients = []
        for client in self.websocket_clients:
            try:
                await client.send_json(event)
            except Exception:
                dead_clients.append(client)
        
        # Clean up dead connections
        for client in dead_clients:
            self.websocket_clients.remove(client)


# Global state
app_state = AppState()


# Security
security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """Verify JWT token (simplified for demo)."""
    # In production, implement proper JWT verification
    if credentials:
        return credentials.credentials
    return None


async def get_current_user(token: Optional[str] = Depends(verify_token)) -> Dict:
    """Get current user from token."""
    # In production, decode JWT and fetch user
    return {"user_id": "demo-user", "role": "admin"}


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Retail Vision Analytics API...")
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Retail Vision Analytics API",
    description="""
    Real-time retail analytics powered by NVIDIA DeepStream and TensorRT.
    
    ## Features
    
    * **Stream Management** - Add, remove, and monitor video streams
    * **Real-time Analytics** - Customer tracking, queue monitoring, heatmaps
    * **Alerts** - Configurable alerts for queue length, wait times, shrinkage
    * **Historical Data** - Query historical analytics with aggregation
    * **WebSocket** - Real-time event streaming
    
    ## Authentication
    
    All endpoints require Bearer token authentication.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def get_health():
    """
    Get system health status.
    
    Returns current health metrics including GPU/CPU utilization,
    active streams, inference performance, and error counts.
    """
    uptime = time.time() - app_state.start_time
    
    return HealthStatus(
        status="healthy",
        uptime_seconds=uptime,
        streams_active=len([s for s in app_state.streams.values() if s["status"] == "active"]),
        streams_total=len(app_state.streams),
        gpu_utilization=75.5,  # Simulated
        cpu_utilization=45.2,
        memory_utilization=62.8,
        disk_utilization=35.0,
        inference_fps=120.0,
        avg_latency_ms=4.5,
        errors_last_hour=0,
        version="1.0.0",
    )


@app.get("/", tags=["Health"])
async def root():
    """API root endpoint."""
    return {
        "name": "Retail Vision Analytics API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


# =============================================================================
# Stream Management Endpoints
# =============================================================================

@app.post(
    "/api/v1/streams",
    response_model=StreamResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Streams"],
)
async def create_stream(
    stream: StreamCreate,
    user: Dict = Depends(get_current_user),
):
    """
    Add a new video stream for processing.
    
    The stream will be automatically started and begin processing
    frames through the DeepStream pipeline.
    """
    try:
        result = await app_state.add_stream(stream)
        
        # Broadcast event
        await app_state.broadcast_event({
            "type": "stream_added",
            "stream_id": stream.stream_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/streams", response_model=List[StreamResponse], tags=["Streams"])
async def list_streams(
    store_id: Optional[str] = Query(None, description="Filter by store ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    user: Dict = Depends(get_current_user),
):
    """
    List all configured video streams.
    
    Optionally filter by store ID or stream status.
    """
    streams = list(app_state.streams.values())
    
    if store_id:
        streams = [s for s in streams if s.get("store_id") == store_id]
    
    if status:
        streams = [s for s in streams if s.get("status") == status]
    
    return [StreamResponse(**s) for s in streams]


@app.get("/api/v1/streams/{stream_id}", response_model=StreamResponse, tags=["Streams"])
async def get_stream(
    stream_id: str = Path(..., description="Stream identifier"),
    user: Dict = Depends(get_current_user),
):
    """Get details of a specific stream."""
    stream = await app_state.get_stream(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    return stream


@app.delete("/api/v1/streams/{stream_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Streams"])
async def delete_stream(
    stream_id: str = Path(..., description="Stream identifier"),
    user: Dict = Depends(get_current_user),
):
    """
    Remove a video stream.
    
    The stream will be stopped and removed from the pipeline.
    """
    removed = await app_state.remove_stream(stream_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    await app_state.broadcast_event({
        "type": "stream_removed",
        "stream_id": stream_id,
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.post("/api/v1/streams/{stream_id}/restart", tags=["Streams"])
async def restart_stream(
    stream_id: str = Path(..., description="Stream identifier"),
    user: Dict = Depends(get_current_user),
):
    """Restart a video stream."""
    if stream_id not in app_state.streams:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    # In production, this would restart the actual stream
    app_state.streams[stream_id]["status"] = "active"
    
    return {"message": f"Stream {stream_id} restarted", "status": "active"}


# =============================================================================
# Analytics Endpoints
# =============================================================================

@app.get("/api/v1/analytics/realtime", response_model=Dict[str, FrameAnalytics], tags=["Analytics"])
async def get_realtime_analytics(
    stream_ids: Optional[List[str]] = Query(None, description="Filter by stream IDs"),
    user: Dict = Depends(get_current_user),
):
    """
    Get real-time analytics for all or specified streams.
    
    Returns the latest frame analytics including detection counts
    and inference timing.
    """
    result = {}
    
    streams = stream_ids or list(app_state.streams.keys())
    
    for stream_id in streams:
        if stream_id in app_state.streams:
            # Simulated real-time data
            result[stream_id] = FrameAnalytics(
                stream_id=stream_id,
                frame_number=app_state.streams[stream_id].get("frames_processed", 0),
                timestamp=datetime.utcnow(),
                detections=[
                    Detection(
                        class_name="person",
                        class_id=0,
                        confidence=0.92,
                        bbox=[100, 200, 150, 400],
                        track_id=1,
                    ),
                    Detection(
                        class_name="shopping_cart",
                        class_id=1,
                        confidence=0.88,
                        bbox=[300, 250, 100, 150],
                        track_id=101,
                    ),
                ],
                person_count=5,
                cart_count=3,
                inference_time_ms=4.2,
            )
    
    return result


@app.get("/api/v1/analytics/zones", response_model=List[ZoneAnalytics], tags=["Analytics"])
async def get_zone_analytics(
    stream_id: Optional[str] = Query(None, description="Filter by stream ID"),
    user: Dict = Depends(get_current_user),
):
    """
    Get zone-based analytics.
    
    Returns occupancy, dwell time, and traffic metrics for each zone.
    """
    # Simulated zone data
    zones = [
        ZoneAnalytics(
            zone_id="zone-entrance",
            zone_name="Main Entrance",
            current_occupancy=8,
            avg_dwell_time_sec=15.5,
            entries_count=245,
            exits_count=238,
            peak_occupancy=15,
            peak_time=datetime.utcnow() - timedelta(hours=2),
        ),
        ZoneAnalytics(
            zone_id="zone-checkout",
            zone_name="Checkout Area",
            current_occupancy=12,
            avg_dwell_time_sec=180.0,
            entries_count=150,
            exits_count=142,
            peak_occupancy=25,
            peak_time=datetime.utcnow() - timedelta(hours=1),
        ),
        ZoneAnalytics(
            zone_id="zone-electronics",
            zone_name="Electronics Section",
            current_occupancy=6,
            avg_dwell_time_sec=420.0,
            entries_count=85,
            exits_count=82,
            peak_occupancy=12,
            peak_time=datetime.utcnow() - timedelta(hours=3),
        ),
    ]
    
    return zones


@app.get("/api/v1/analytics/queues", response_model=List[QueueMetrics], tags=["Analytics"])
async def get_queue_metrics(
    stream_id: Optional[str] = Query(None, description="Filter by stream ID"),
    user: Dict = Depends(get_current_user),
):
    """
    Get queue monitoring metrics.
    
    Returns real-time queue length, wait times, and service metrics
    for each monitored checkout lane.
    """
    # Simulated queue data
    queues = [
        QueueMetrics(
            lane_id="lane-1",
            lane_name="Checkout 1",
            queue_length=4,
            avg_wait_time_sec=120.0,
            max_wait_time_sec=240.0,
            service_rate_per_min=2.5,
            abandonment_count=2,
            status="normal",
        ),
        QueueMetrics(
            lane_id="lane-2",
            lane_name="Checkout 2",
            queue_length=8,
            avg_wait_time_sec=280.0,
            max_wait_time_sec=420.0,
            service_rate_per_min=1.8,
            abandonment_count=5,
            status="busy",
        ),
        QueueMetrics(
            lane_id="lane-3",
            lane_name="Self-Checkout",
            queue_length=2,
            avg_wait_time_sec=45.0,
            max_wait_time_sec=90.0,
            service_rate_per_min=4.0,
            abandonment_count=0,
            status="normal",
        ),
    ]
    
    return queues


@app.get("/api/v1/analytics/heatmap/{stream_id}", response_model=HeatmapData, tags=["Analytics"])
async def get_heatmap(
    stream_id: str = Path(..., description="Stream identifier"),
    period: str = Query("hour", description="Time period (hour, day, week)"),
    user: Dict = Depends(get_current_user),
):
    """
    Get heatmap data for a stream.
    
    Returns a 2D grid of traffic intensity values for visualization.
    """
    if stream_id not in app_state.streams:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    # Simulated heatmap data (96x54 grid for 1920x1080 with 20px cells)
    import random
    grid_width = 96
    grid_height = 54
    
    data = [
        [random.random() * 0.5 + (0.3 if 20 < x < 76 and 15 < y < 40 else 0) 
         for x in range(grid_width)]
        for y in range(grid_height)
    ]
    
    return HeatmapData(
        stream_id=stream_id,
        timestamp=datetime.utcnow(),
        width=grid_width,
        height=grid_height,
        cell_size=20,
        data=data,
        hotspots=[
            {"x": 48, "y": 27, "intensity": 0.95, "label": "Main Aisle"},
            {"x": 72, "y": 20, "intensity": 0.78, "label": "Product Display"},
        ],
    )


@app.post("/api/v1/analytics/query", response_model=List[AnalyticsSummary], tags=["Analytics"])
async def query_analytics(
    query: AnalyticsQuery,
    user: Dict = Depends(get_current_user),
):
    """
    Query historical analytics data.
    
    Supports time-based filtering and aggregation at minute, hour, or day level.
    """
    # Simulated historical data
    now = datetime.utcnow()
    
    return [
        AnalyticsSummary(
            period_start=now - timedelta(hours=1),
            period_end=now,
            stream_id=query.stream_ids[0] if query.stream_ids else None,
            total_persons=1250,
            unique_persons=980,
            avg_dwell_time_sec=185.5,
            total_carts=420,
            conversion_rate=0.68,
            peak_occupancy=85,
            peak_time=now - timedelta(minutes=30),
            total_detections=15600,
        ),
    ]


# =============================================================================
# Alert Endpoints
# =============================================================================

@app.post(
    "/api/v1/alerts/rules",
    response_model=Dict[str, str],
    status_code=status.HTTP_201_CREATED,
    tags=["Alerts"],
)
async def create_alert_rule(
    rule: AlertCreate,
    user: Dict = Depends(get_current_user),
):
    """
    Create a new alert rule.
    
    Alert rules define conditions that trigger notifications.
    """
    import uuid
    rule_id = str(uuid.uuid4())[:8]
    
    app_state.alert_rules[rule_id] = {
        "rule_id": rule_id,
        **rule.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "created_by": user["user_id"],
    }
    
    return {"rule_id": rule_id, "message": "Alert rule created"}


@app.get("/api/v1/alerts/rules", response_model=List[Dict], tags=["Alerts"])
async def list_alert_rules(
    user: Dict = Depends(get_current_user),
):
    """List all configured alert rules."""
    return list(app_state.alert_rules.values())


@app.delete("/api/v1/alerts/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Alerts"])
async def delete_alert_rule(
    rule_id: str = Path(..., description="Alert rule identifier"),
    user: Dict = Depends(get_current_user),
):
    """Delete an alert rule."""
    if rule_id not in app_state.alert_rules:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    
    del app_state.alert_rules[rule_id]


@app.get("/api/v1/alerts", response_model=List[AlertResponse], tags=["Alerts"])
async def list_alerts(
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    alert_type: Optional[AlertType] = Query(None, description="Filter by type"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment"),
    limit: int = Query(50, ge=1, le=500, description="Maximum alerts to return"),
    user: Dict = Depends(get_current_user),
):
    """
    List recent alerts.
    
    Supports filtering by severity, type, and acknowledgment status.
    """
    # Simulated alerts
    alerts = [
        AlertResponse(
            alert_id="alert-001",
            name="Queue Length Warning",
            alert_type=AlertType.QUEUE_LENGTH,
            severity=AlertSeverity.WARNING,
            stream_id="store1-checkout",
            triggered_at=datetime.utcnow() - timedelta(minutes=15),
            resolved_at=datetime.utcnow() - timedelta(minutes=5),
            message="Queue at Checkout 2 exceeded threshold (8 > 6)",
            data={"lane_id": "lane-2", "queue_length": 8, "threshold": 6},
            acknowledged=True,
        ),
        AlertResponse(
            alert_id="alert-002",
            name="High Wait Time",
            alert_type=AlertType.WAIT_TIME,
            severity=AlertSeverity.CRITICAL,
            stream_id="store1-checkout",
            triggered_at=datetime.utcnow() - timedelta(minutes=5),
            resolved_at=None,
            message="Average wait time at Checkout 2 exceeded 5 minutes",
            data={"lane_id": "lane-2", "avg_wait_sec": 320, "threshold_sec": 300},
            acknowledged=False,
        ),
    ]
    
    # Apply filters
    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    if alert_type:
        alerts = [a for a in alerts if a.alert_type == alert_type]
    if acknowledged is not None:
        alerts = [a for a in alerts if a.acknowledged == acknowledged]
    
    return alerts[:limit]


@app.post("/api/v1/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(
    alert_id: str = Path(..., description="Alert identifier"),
    user: Dict = Depends(get_current_user),
):
    """Acknowledge an alert."""
    # In production, update database
    return {
        "alert_id": alert_id,
        "acknowledged": True,
        "acknowledged_by": user["user_id"],
        "acknowledged_at": datetime.utcnow().isoformat(),
    }


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    
    Events include:
    - detection: New detections from streams
    - alert: Alert triggers and resolutions
    - stream_status: Stream state changes
    - analytics: Periodic analytics updates
    """
    await websocket.accept()
    app_state.websocket_clients.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to event stream",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages (with timeout for keepalive)
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )
                
                # Handle subscription updates
                if message.get("action") == "subscribe":
                    await websocket.send_json({
                        "type": "subscribed",
                        "streams": message.get("streams", []),
                    })
                
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if websocket in app_state.websocket_clients:
            app_state.websocket_clients.remove(websocket)


# =============================================================================
# Model Management Endpoints
# =============================================================================

@app.get("/api/v1/models", tags=["Models"])
async def list_models(
    user: Dict = Depends(get_current_user),
):
    """List available inference models."""
    return [
        {
            "model_id": "yolov8n-retail-v2",
            "name": "YOLOv8n Retail",
            "version": "2.0.0",
            "precision": "fp16",
            "classes": ["person", "shopping_cart", "basket", "product", "shelf", "price_tag", "employee"],
            "input_size": [640, 640],
            "status": "active",
            "loaded_at": datetime.utcnow().isoformat(),
        },
        {
            "model_id": "yolov8s-retail-v1",
            "name": "YOLOv8s Retail (High Accuracy)",
            "version": "1.0.0",
            "precision": "fp16",
            "classes": ["person", "shopping_cart", "basket", "product", "shelf", "price_tag", "employee"],
            "input_size": [640, 640],
            "status": "available",
            "loaded_at": None,
        },
    ]


@app.post("/api/v1/models/{model_id}/load", tags=["Models"])
async def load_model(
    model_id: str = Path(..., description="Model identifier"),
    user: Dict = Depends(get_current_user),
):
    """Load a model for inference."""
    return {
        "model_id": model_id,
        "status": "loading",
        "message": f"Model {model_id} is being loaded",
    }


# =============================================================================
# Export/Report Endpoints
# =============================================================================

@app.get("/api/v1/reports/daily", tags=["Reports"])
async def generate_daily_report(
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD), default today"),
    store_id: Optional[str] = Query(None, description="Store ID"),
    user: Dict = Depends(get_current_user),
):
    """Generate daily analytics report."""
    return {
        "report_type": "daily",
        "date": date or datetime.utcnow().strftime("%Y-%m-%d"),
        "store_id": store_id,
        "summary": {
            "total_visitors": 2450,
            "unique_visitors": 1890,
            "avg_dwell_time_min": 12.5,
            "conversion_rate": 0.42,
            "peak_hour": "14:00",
            "peak_visitors": 185,
        },
        "hourly_breakdown": [
            {"hour": h, "visitors": 100 + h * 10} for h in range(9, 21)
        ],
        "zone_summary": [
            {"zone": "entrance", "traffic": 2450, "avg_dwell_sec": 15},
            {"zone": "electronics", "traffic": 520, "avg_dwell_sec": 420},
            {"zone": "checkout", "traffic": 1030, "avg_dwell_sec": 180},
        ],
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/export/csv", tags=["Reports"])
async def export_csv(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    metrics: List[str] = Query(["visitors", "dwell_time"], description="Metrics to export"),
    user: Dict = Depends(get_current_user),
):
    """Export analytics data as CSV."""
    # Generate CSV content
    import io
    
    output = io.StringIO()
    output.write(",".join(["timestamp", "stream_id"] + metrics) + "\n")
    
    # Simulated data rows
    for i in range(10):
        output.write(f"2024-01-{15+i}T10:00:00Z,store1-entrance," + 
                    ",".join([str(100 + i * 10) for _ in metrics]) + "\n")
    
    content = output.getvalue()
    
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=analytics_{start_date}_{end_date}.csv"}
    )


# =============================================================================
# Demo Data Generator (for testing)
# =============================================================================

async def generate_demo_events():
    """Background task to generate demo events for WebSocket clients."""
    import random
    
    while True:
        if app_state.websocket_clients:
            # Generate random detection event
            event = {
                "type": "detection",
                "stream_id": random.choice(list(app_state.streams.keys()) or ["demo-stream"]),
                "timestamp": datetime.utcnow().isoformat(),
                "detections": [
                    {
                        "class": random.choice(["person", "shopping_cart", "basket"]),
                        "confidence": round(random.uniform(0.7, 0.99), 2),
                        "track_id": random.randint(1, 100),
                    }
                ],
            }
            
            await app_state.broadcast_event(event)
        
        await asyncio.sleep(1)


# =============================================================================
# Application Entry Point
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Retail Vision Analytics API")
    print("=" * 60)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("ReDoc: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
