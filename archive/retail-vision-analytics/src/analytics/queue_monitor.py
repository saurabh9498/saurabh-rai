"""
Queue Monitoring Module for Retail Vision Analytics.

Real-time checkout queue analysis including:
- Queue length detection
- Wait time estimation
- Staffing recommendations
- Queue abandonment detection
- Predictive analytics

Optimized for retail checkout operations with
configurable alert thresholds and staffing rules.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QueueStatus(Enum):
    """Queue status levels."""
    EMPTY = "empty"
    LOW = "low"           # 1-2 people
    MODERATE = "moderate"  # 3-5 people
    HIGH = "high"         # 6-8 people
    CRITICAL = "critical"  # 9+ people


class AlertType(Enum):
    """Queue alert types."""
    LONG_QUEUE = "long_queue"
    HIGH_WAIT_TIME = "high_wait_time"
    QUEUE_ABANDONED = "queue_abandoned"
    STAFF_NEEDED = "staff_needed"
    REGISTER_IDLE = "register_idle"


@dataclass
class CheckoutLane:
    """Checkout lane definition and state."""
    
    lane_id: str
    name: str
    position: Tuple[int, int, int, int]  # Bounding box x1, y1, x2, y2
    lane_type: str = "standard"  # standard, express, self_checkout
    is_open: bool = True
    max_capacity: int = 10
    
    # Current state
    current_queue_length: int = 0
    customers_in_queue: List[int] = field(default_factory=list)  # Track IDs
    
    # Historical data
    service_times: deque = field(default_factory=lambda: deque(maxlen=100))
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=60))
    
    @property
    def status(self) -> QueueStatus:
        """Get current queue status."""
        n = self.current_queue_length
        if n == 0:
            return QueueStatus.EMPTY
        elif n <= 2:
            return QueueStatus.LOW
        elif n <= 5:
            return QueueStatus.MODERATE
        elif n <= 8:
            return QueueStatus.HIGH
        else:
            return QueueStatus.CRITICAL
    
    @property
    def avg_service_time(self) -> float:
        """Average service time in seconds."""
        if not self.service_times:
            return 60.0  # Default 60 seconds
        return np.mean(list(self.service_times))
    
    @property
    def estimated_wait_time(self) -> float:
        """Estimated wait time for new customer in seconds."""
        return self.current_queue_length * self.avg_service_time
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is in lane area."""
        x1, y1, x2, y2 = self.position
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lane_id": self.lane_id,
            "name": self.name,
            "lane_type": self.lane_type,
            "is_open": self.is_open,
            "current_queue_length": self.current_queue_length,
            "status": self.status.value,
            "avg_service_time_seconds": round(self.avg_service_time, 2),
            "estimated_wait_time_seconds": round(self.estimated_wait_time, 2)
        }


@dataclass
class QueueEvent:
    """Queue-related event."""
    
    event_type: str  # join, leave, served, abandon
    lane_id: str
    track_id: int
    timestamp: datetime
    wait_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueAlert:
    """Queue monitoring alert."""
    
    alert_type: AlertType
    lane_id: str
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    is_acknowledged: bool = False


class QueueMonitor:
    """
    Real-time checkout queue monitoring and analytics.
    
    Features:
    - Multi-lane queue detection
    - Wait time estimation
    - Throughput analytics
    - Staffing recommendations
    - Alert generation
    
    Example:
        >>> monitor = QueueMonitor()
        >>> monitor.add_lane(CheckoutLane(
        ...     lane_id="lane_1",
        ...     name="Register 1",
        ...     position=(100, 400, 200, 600)
        ... ))
        >>> monitor.update_detections(detections, timestamp)
        >>> stats = monitor.get_statistics()
    """
    
    # Alert thresholds
    QUEUE_LENGTH_WARNING = 5
    QUEUE_LENGTH_CRITICAL = 8
    WAIT_TIME_WARNING = 180  # 3 minutes
    WAIT_TIME_CRITICAL = 300  # 5 minutes
    IDLE_TIME_THRESHOLD = 120  # 2 minutes
    
    def __init__(
        self,
        lanes: Optional[List[CheckoutLane]] = None,
        alert_callback: Optional[Callable[[QueueAlert], None]] = None,
        enable_predictions: bool = True,
        prediction_horizon_minutes: int = 30
    ):
        """
        Initialize queue monitor.
        
        Args:
            lanes: List of checkout lanes
            alert_callback: Function to call when alert is generated
            enable_predictions: Enable wait time predictions
            prediction_horizon_minutes: How far ahead to predict
        """
        self.lanes: Dict[str, CheckoutLane] = {}
        self.alert_callback = alert_callback
        self.enable_predictions = enable_predictions
        self.prediction_horizon = prediction_horizon_minutes
        
        # Tracking state
        self._customer_entries: Dict[int, Dict] = {}  # track_id -> entry info
        self._events: List[QueueEvent] = []
        self._alerts: List[QueueAlert] = []
        
        # Time-series data for predictions
        self._queue_history: Dict[str, deque] = {}
        self._throughput_history: Dict[str, deque] = {}
        
        # Add initial lanes
        if lanes:
            for lane in lanes:
                self.add_lane(lane)
        
        logger.info(f"QueueMonitor initialized with {len(self.lanes)} lanes")
    
    def add_lane(self, lane: CheckoutLane) -> None:
        """Add a checkout lane."""
        self.lanes[lane.lane_id] = lane
        self._queue_history[lane.lane_id] = deque(maxlen=1000)
        self._throughput_history[lane.lane_id] = deque(maxlen=1000)
    
    def remove_lane(self, lane_id: str) -> None:
        """Remove a lane."""
        if lane_id in self.lanes:
            del self.lanes[lane_id]
    
    def set_lane_status(self, lane_id: str, is_open: bool) -> None:
        """Open or close a lane."""
        if lane_id in self.lanes:
            self.lanes[lane_id].is_open = is_open
    
    def update_detections(
        self,
        detections: List[Dict[str, Any]],
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Update queue state with new detections.
        
        Args:
            detections: List of person detections with bbox and track_id
            timestamp: Current Unix timestamp
        
        Returns:
            Current queue state summary
        """
        current_time = datetime.fromtimestamp(timestamp)
        
        # Track which customers are in which lanes
        lane_customers: Dict[str, List[int]] = {lid: [] for lid in self.lanes}
        
        for det in detections:
            if det.get("class_name", "").lower() != "person":
                continue
            
            track_id = det.get("track_id")
            if track_id is None:
                continue
            
            # Get person center
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            
            # Check which lane they're in
            for lane_id, lane in self.lanes.items():
                if lane.contains_point(cx, cy):
                    lane_customers[lane_id].append(track_id)
                    
                    # Track new customer entry
                    if track_id not in self._customer_entries:
                        self._customer_entries[track_id] = {
                            "lane_id": lane_id,
                            "entry_time": current_time,
                            "last_seen": current_time
                        }
                        self._record_event(QueueEvent(
                            event_type="join",
                            lane_id=lane_id,
                            track_id=track_id,
                            timestamp=current_time
                        ))
                    else:
                        self._customer_entries[track_id]["last_seen"] = current_time
                    break
        
        # Update lane states
        for lane_id, lane in self.lanes.items():
            previous_customers = set(lane.customers_in_queue)
            current_customers = set(lane_customers[lane_id])
            
            # Detect served customers (left queue but were at front)
            left = previous_customers - current_customers
            for track_id in left:
                if track_id in self._customer_entries:
                    entry = self._customer_entries[track_id]
                    if entry["lane_id"] == lane_id:
                        wait_time = (current_time - entry["entry_time"]).total_seconds()
                        
                        # Record service time
                        lane.service_times.append(wait_time)
                        
                        self._record_event(QueueEvent(
                            event_type="served",
                            lane_id=lane_id,
                            track_id=track_id,
                            timestamp=current_time,
                            wait_time_seconds=wait_time
                        ))
                        
                        del self._customer_entries[track_id]
            
            # Update queue state
            lane.customers_in_queue = lane_customers[lane_id]
            lane.current_queue_length = len(lane_customers[lane_id])
            
            # Record history
            self._queue_history[lane_id].append({
                "timestamp": current_time,
                "length": lane.current_queue_length
            })
        
        # Check for alerts
        self._check_alerts(current_time)
        
        # Check for queue abandonment
        self._check_abandonment(current_time)
        
        return self.get_current_state()
    
    def _record_event(self, event: QueueEvent) -> None:
        """Record a queue event."""
        self._events.append(event)
        
        # Limit event history
        if len(self._events) > 10000:
            self._events = self._events[-5000:]
    
    def _check_alerts(self, current_time: datetime) -> None:
        """Check for alert conditions."""
        for lane_id, lane in self.lanes.items():
            if not lane.is_open:
                continue
            
            # Check queue length
            if lane.current_queue_length >= self.QUEUE_LENGTH_CRITICAL:
                self._generate_alert(
                    AlertType.LONG_QUEUE,
                    lane_id,
                    "critical",
                    f"Critical queue length at {lane.name}: {lane.current_queue_length} customers",
                    current_time,
                    {"queue_length": lane.current_queue_length}
                )
            elif lane.current_queue_length >= self.QUEUE_LENGTH_WARNING:
                self._generate_alert(
                    AlertType.LONG_QUEUE,
                    lane_id,
                    "high",
                    f"Long queue at {lane.name}: {lane.current_queue_length} customers",
                    current_time,
                    {"queue_length": lane.current_queue_length}
                )
            
            # Check wait time
            if lane.estimated_wait_time >= self.WAIT_TIME_CRITICAL:
                self._generate_alert(
                    AlertType.HIGH_WAIT_TIME,
                    lane_id,
                    "critical",
                    f"Critical wait time at {lane.name}: {lane.estimated_wait_time/60:.1f} minutes",
                    current_time,
                    {"wait_time_seconds": lane.estimated_wait_time}
                )
            elif lane.estimated_wait_time >= self.WAIT_TIME_WARNING:
                self._generate_alert(
                    AlertType.HIGH_WAIT_TIME,
                    lane_id,
                    "high",
                    f"High wait time at {lane.name}: {lane.estimated_wait_time/60:.1f} minutes",
                    current_time,
                    {"wait_time_seconds": lane.estimated_wait_time}
                )
    
    def _check_abandonment(self, current_time: datetime) -> None:
        """Check for queue abandonment."""
        stale_threshold = timedelta(seconds=30)
        
        for track_id, entry in list(self._customer_entries.items()):
            if current_time - entry["last_seen"] > stale_threshold:
                wait_time = (entry["last_seen"] - entry["entry_time"]).total_seconds()
                
                self._record_event(QueueEvent(
                    event_type="abandon",
                    lane_id=entry["lane_id"],
                    track_id=track_id,
                    timestamp=current_time,
                    wait_time_seconds=wait_time
                ))
                
                # Generate alert if significant wait time
                if wait_time >= 60:  # Waited at least 1 minute
                    self._generate_alert(
                        AlertType.QUEUE_ABANDONED,
                        entry["lane_id"],
                        "medium",
                        f"Customer abandoned queue after {wait_time:.0f}s wait",
                        current_time,
                        {"wait_time_seconds": wait_time}
                    )
                
                del self._customer_entries[track_id]
    
    def _generate_alert(
        self,
        alert_type: AlertType,
        lane_id: str,
        severity: str,
        message: str,
        timestamp: datetime,
        data: Dict[str, Any]
    ) -> None:
        """Generate and dispatch alert."""
        # Check for duplicate recent alerts
        recent = [a for a in self._alerts[-20:] 
                  if a.alert_type == alert_type 
                  and a.lane_id == lane_id
                  and (timestamp - a.timestamp).total_seconds() < 60]
        
        if recent:
            return  # Don't duplicate
        
        alert = QueueAlert(
            alert_type=alert_type,
            lane_id=lane_id,
            severity=severity,
            message=message,
            timestamp=timestamp,
            data=data
        )
        
        self._alerts.append(alert)
        
        # Limit alert history
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-500:]
        
        logger.warning(f"Queue Alert: {message}")
        
        if self.alert_callback:
            self.alert_callback(alert)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current queue state for all lanes."""
        total_customers = sum(l.current_queue_length for l in self.lanes.values())
        open_lanes = sum(1 for l in self.lanes.values() if l.is_open)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_customers_in_queue": total_customers,
            "open_lanes": open_lanes,
            "total_lanes": len(self.lanes),
            "lanes": {
                lane_id: lane.to_dict()
                for lane_id, lane in self.lanes.items()
            },
            "overall_status": self._get_overall_status().value
        }
    
    def _get_overall_status(self) -> QueueStatus:
        """Get overall queue status across all lanes."""
        if not self.lanes:
            return QueueStatus.EMPTY
        
        statuses = [l.status for l in self.lanes.values() if l.is_open]
        
        if not statuses:
            return QueueStatus.EMPTY
        
        if QueueStatus.CRITICAL in statuses:
            return QueueStatus.CRITICAL
        if QueueStatus.HIGH in statuses:
            return QueueStatus.HIGH
        if QueueStatus.MODERATE in statuses:
            return QueueStatus.MODERATE
        if QueueStatus.LOW in statuses:
            return QueueStatus.LOW
        return QueueStatus.EMPTY
    
    def get_statistics(
        self,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get queue statistics for time window."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        
        # Filter events
        recent_events = [e for e in self._events if e.timestamp >= cutoff]
        
        # Calculate metrics
        served = [e for e in recent_events if e.event_type == "served"]
        abandoned = [e for e in recent_events if e.event_type == "abandon"]
        
        wait_times = [e.wait_time_seconds for e in served if e.wait_time_seconds]
        abandon_waits = [e.wait_time_seconds for e in abandoned if e.wait_time_seconds]
        
        return {
            "time_window_minutes": time_window_minutes,
            "customers_served": len(served),
            "customers_abandoned": len(abandoned),
            "abandonment_rate": round(
                len(abandoned) / (len(served) + len(abandoned)) * 100, 2
            ) if (served or abandoned) else 0,
            "average_wait_time_seconds": round(np.mean(wait_times), 2) if wait_times else 0,
            "median_wait_time_seconds": round(np.median(wait_times), 2) if wait_times else 0,
            "max_wait_time_seconds": round(max(wait_times), 2) if wait_times else 0,
            "throughput_per_hour": len(served) * (60 / time_window_minutes),
            "average_abandon_wait_seconds": round(np.mean(abandon_waits), 2) if abandon_waits else 0
        }
    
    def get_staffing_recommendation(self) -> Dict[str, Any]:
        """Get staffing recommendations based on current state."""
        current_state = self.get_current_state()
        total_customers = current_state["total_customers_in_queue"]
        open_lanes = current_state["open_lanes"]
        closed_lanes = current_state["total_lanes"] - open_lanes
        
        # Calculate ideal lanes needed
        avg_service_time = np.mean([
            l.avg_service_time for l in self.lanes.values()
        ]) if self.lanes else 60
        
        target_wait_time = 120  # 2 minutes target
        ideal_lanes = max(1, int(np.ceil(
            total_customers * avg_service_time / target_wait_time
        )))
        
        lanes_needed = max(0, ideal_lanes - open_lanes)
        
        recommendation = {
            "current_open_lanes": open_lanes,
            "closed_lanes_available": closed_lanes,
            "recommended_total_lanes": ideal_lanes,
            "additional_lanes_needed": lanes_needed,
            "action": "none"
        }
        
        if lanes_needed > 0 and closed_lanes > 0:
            recommendation["action"] = "open_lanes"
            recommendation["message"] = f"Open {min(lanes_needed, closed_lanes)} additional lane(s)"
        elif lanes_needed > closed_lanes:
            recommendation["action"] = "understaffed"
            recommendation["message"] = "All lanes open but still understaffed"
        elif total_customers == 0 and open_lanes > 1:
            recommendation["action"] = "close_lanes"
            recommendation["message"] = "Consider closing unused lanes"
        
        return recommendation
    
    def predict_wait_times(
        self,
        horizon_minutes: int = 30
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Predict future wait times using historical patterns."""
        if not self.enable_predictions:
            return {}
        
        predictions = {}
        
        for lane_id, history in self._queue_history.items():
            if len(history) < 10:
                continue
            
            # Simple moving average prediction
            recent = list(history)[-30:]
            queue_lengths = [h["length"] for h in recent]
            
            avg_length = np.mean(queue_lengths)
            trend = 0
            if len(queue_lengths) >= 2:
                trend = (queue_lengths[-1] - queue_lengths[0]) / len(queue_lengths)
            
            lane = self.lanes[lane_id]
            current_time = datetime.now()
            
            lane_predictions = []
            for minutes in range(0, horizon_minutes + 1, 5):
                predicted_length = max(0, avg_length + trend * minutes)
                predicted_wait = predicted_length * lane.avg_service_time
                
                lane_predictions.append({
                    "time": (current_time + timedelta(minutes=minutes)).isoformat(),
                    "predicted_queue_length": round(predicted_length, 1),
                    "predicted_wait_seconds": round(predicted_wait, 0)
                })
            
            predictions[lane_id] = lane_predictions
        
        return predictions
    
    def get_alerts(
        self,
        since: Optional[datetime] = None,
        unacknowledged_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filters."""
        alerts = self._alerts
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.is_acknowledged]
        
        return [
            {
                "alert_type": a.alert_type.value,
                "lane_id": a.lane_id,
                "severity": a.severity,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
                "data": a.data,
                "is_acknowledged": a.is_acknowledged
            }
            for a in alerts
        ]
    
    def acknowledge_alerts(self, lane_id: Optional[str] = None) -> int:
        """Acknowledge alerts, optionally by lane."""
        count = 0
        for alert in self._alerts:
            if not alert.is_acknowledged:
                if lane_id is None or alert.lane_id == lane_id:
                    alert.is_acknowledged = True
                    count += 1
        return count


class QueueOptimizer:
    """
    Queue optimization recommendations.
    
    Analyzes patterns to suggest:
    - Optimal staffing schedules
    - Lane configurations
    - Peak time predictions
    """
    
    def __init__(self, monitor: QueueMonitor):
        """Initialize optimizer with queue monitor."""
        self.monitor = monitor
        self._hourly_patterns: Dict[int, List[float]] = {h: [] for h in range(24)}
    
    def update_patterns(self) -> None:
        """Update hourly patterns from recent data."""
        for lane_id, history in self.monitor._queue_history.items():
            for record in history:
                hour = record["timestamp"].hour
                self._hourly_patterns[hour].append(record["length"])
    
    def get_peak_hours(self) -> List[Dict[str, Any]]:
        """Identify peak queue hours."""
        hourly_avg = {}
        
        for hour, lengths in self._hourly_patterns.items():
            if lengths:
                hourly_avg[hour] = np.mean(lengths)
        
        if not hourly_avg:
            return []
        
        # Find peaks (above 75th percentile)
        threshold = np.percentile(list(hourly_avg.values()), 75)
        
        peaks = [
            {"hour": h, "average_queue_length": round(avg, 2)}
            for h, avg in hourly_avg.items()
            if avg >= threshold
        ]
        
        return sorted(peaks, key=lambda x: x["average_queue_length"], reverse=True)
    
    def get_optimal_schedule(self) -> Dict[str, Any]:
        """Generate optimal staffing schedule."""
        schedule = {}
        
        for hour in range(24):
            lengths = self._hourly_patterns.get(hour, [])
            avg_length = np.mean(lengths) if lengths else 0
            
            # Calculate recommended lanes
            recommended = max(1, int(np.ceil(avg_length / 3)))  # 3 customers per lane
            
            schedule[f"{hour:02d}:00"] = {
                "expected_queue_length": round(avg_length, 1),
                "recommended_lanes": recommended
            }
        
        return {
            "schedule": schedule,
            "peak_hours": self.get_peak_hours()
        }


if __name__ == "__main__":
    # Demo usage
    import time
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor with sample lanes
    monitor = QueueMonitor()
    
    monitor.add_lane(CheckoutLane(
        lane_id="lane_1",
        name="Register 1",
        position=(100, 400, 200, 600)
    ))
    
    monitor.add_lane(CheckoutLane(
        lane_id="lane_2",
        name="Register 2",
        position=(250, 400, 350, 600)
    ))
    
    # Simulate detections
    base_time = time.time()
    
    for i in range(10):
        detections = [
            {
                "class_name": "person",
                "track_id": j,
                "bbox": (120 + j*10, 450, 160 + j*10, 550)
            }
            for j in range(i % 5 + 1)  # 1-5 people
        ]
        
        state = monitor.update_detections(detections, base_time + i)
        print(f"Step {i}: {state['total_customers_in_queue']} customers")
    
    # Get statistics
    print("\nStatistics:")
    stats = monitor.get_statistics(time_window_minutes=1)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get staffing recommendation
    print("\nStaffing Recommendation:")
    rec = monitor.get_staffing_recommendation()
    for key, value in rec.items():
        print(f"  {key}: {value}")
