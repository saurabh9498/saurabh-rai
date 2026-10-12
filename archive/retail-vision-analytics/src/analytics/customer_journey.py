"""
Customer Journey Analytics Module for Retail Vision Analytics.

Tracks customer movements through the store, analyzing:
- Zone transitions and path patterns
- Dwell time at product displays
- Shopping journey stages
- Cross-camera tracking with ReID

Provides insights for:
- Store layout optimization
- Product placement effectiveness
- Customer engagement metrics
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Store zone categories."""
    ENTRANCE = "entrance"
    EXIT = "exit"
    AISLE = "aisle"
    CHECKOUT = "checkout"
    DISPLAY = "display"
    PROMOTION = "promotion"
    SERVICE = "service"
    FITTING_ROOM = "fitting_room"
    CAFE = "cafe"
    RESTROOM = "restroom"


class JourneyStage(Enum):
    """Customer journey stages."""
    ENTERED = "entered"
    BROWSING = "browsing"
    ENGAGED = "engaged"  # Actively looking at products
    CONSIDERING = "considering"  # High dwell time
    ASSISTED = "assisted"  # With staff
    CHECKOUT = "checkout"
    EXITED = "exited"


@dataclass
class Zone:
    """Store zone definition."""
    
    zone_id: str
    name: str
    zone_type: ZoneType
    polygon: List[Tuple[int, int]]  # Polygon vertices
    camera_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside zone polygon."""
        return self._point_in_polygon(x, y, self.polygon)
    
    @staticmethod
    def _point_in_polygon(
        x: int,
        y: int,
        polygon: List[Tuple[int, int]]
    ) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Calculate zone centroid."""
        if not self.polygon:
            return (0.0, 0.0)
        
        x_coords = [p[0] for p in self.polygon]
        y_coords = [p[1] for p in self.polygon]
        
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Get zone bounding box (x1, y1, x2, y2)."""
        if not self.polygon:
            return (0, 0, 0, 0)
        
        x_coords = [p[0] for p in self.polygon]
        y_coords = [p[1] for p in self.polygon]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


@dataclass
class ZoneVisit:
    """Record of a zone visit."""
    
    zone_id: str
    zone_name: str
    zone_type: ZoneType
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_point: Optional[Tuple[int, int]] = None
    exit_point: Optional[Tuple[int, int]] = None
    
    @property
    def duration_seconds(self) -> float:
        """Calculate visit duration in seconds."""
        if self.exit_time is None:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds()
    
    @property
    def is_active(self) -> bool:
        """Check if visit is ongoing."""
        return self.exit_time is None


@dataclass
class CustomerJourney:
    """Complete customer journey through the store."""
    
    customer_id: str
    track_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Journey data
    zone_visits: List[ZoneVisit] = field(default_factory=list)
    trajectory: List[Tuple[int, int, float]] = field(default_factory=list)  # x, y, timestamp
    
    # Analytics
    current_stage: JourneyStage = JourneyStage.ENTERED
    total_distance: float = 0.0
    
    # Cross-camera tracking
    camera_transitions: List[Tuple[str, str, datetime]] = field(default_factory=list)
    appearance_features: Optional[np.ndarray] = None
    
    @property
    def total_duration_seconds(self) -> float:
        """Total time in store."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def zones_visited(self) -> List[str]:
        """List of unique zones visited."""
        return list(dict.fromkeys([v.zone_id for v in self.zone_visits]))
    
    @property
    def num_zones(self) -> int:
        """Number of unique zones visited."""
        return len(set(v.zone_id for v in self.zone_visits))
    
    @property
    def current_zone(self) -> Optional[str]:
        """Get current zone if any."""
        for visit in reversed(self.zone_visits):
            if visit.is_active:
                return visit.zone_id
        return None
    
    def get_dwell_time(self, zone_id: str) -> float:
        """Get total dwell time in a specific zone."""
        return sum(
            v.duration_seconds
            for v in self.zone_visits
            if v.zone_id == zone_id and v.exit_time is not None
        )
    
    def get_zone_visits_count(self, zone_id: str) -> int:
        """Get number of visits to a zone."""
        return sum(1 for v in self.zone_visits if v.zone_id == zone_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "customer_id": self.customer_id,
            "track_id": self.track_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_stage": self.current_stage.value,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "num_zones_visited": self.num_zones,
            "zones_visited": self.zones_visited,
            "current_zone": self.current_zone,
            "total_distance": round(self.total_distance, 2),
            "zone_visits": [
                {
                    "zone_id": v.zone_id,
                    "zone_name": v.zone_name,
                    "zone_type": v.zone_type.value,
                    "duration_seconds": round(v.duration_seconds, 2)
                }
                for v in self.zone_visits
            ]
        }


class CustomerJourneyAnalyzer:
    """
    Analyzes customer journeys through the store.
    
    Features:
    - Real-time zone tracking
    - Dwell time measurement
    - Path pattern analysis
    - Journey stage classification
    - Cross-camera customer matching
    
    Example:
        >>> analyzer = CustomerJourneyAnalyzer()
        >>> analyzer.add_zone(Zone(
        ...     zone_id="entrance",
        ...     name="Main Entrance",
        ...     zone_type=ZoneType.ENTRANCE,
        ...     polygon=[(0, 0), (100, 0), (100, 100), (0, 100)]
        ... ))
        >>> analyzer.update_position(track_id=1, x=50, y=50, timestamp=time.time())
        >>> journey = analyzer.get_journey(track_id=1)
    """
    
    # Thresholds for journey stage classification
    ENGAGED_DWELL_THRESHOLD = 5.0   # seconds
    CONSIDERING_DWELL_THRESHOLD = 15.0  # seconds
    CHECKOUT_PROXIMITY_THRESHOLD = 10.0  # seconds near checkout
    
    def __init__(
        self,
        zones: Optional[List[Zone]] = None,
        dwell_threshold_seconds: float = 3.0,
        journey_timeout_seconds: float = 300.0,
        enable_reid: bool = True
    ):
        """
        Initialize customer journey analyzer.
        
        Args:
            zones: List of store zones
            dwell_threshold_seconds: Minimum time to count as dwell
            journey_timeout_seconds: Time after which journey is considered ended
            enable_reid: Enable re-identification for cross-camera tracking
        """
        self.zones: Dict[str, Zone] = {}
        self.dwell_threshold = dwell_threshold_seconds
        self.journey_timeout = journey_timeout_seconds
        self.enable_reid = enable_reid
        
        # Active journeys by track ID
        self._journeys: Dict[int, CustomerJourney] = {}
        
        # Completed journeys
        self._completed_journeys: List[CustomerJourney] = []
        
        # Statistics
        self._total_customers = 0
        self._zone_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_visits": 0,
            "total_dwell_time": 0.0,
            "visitors": set()
        })
        
        # Add initial zones
        if zones:
            for zone in zones:
                self.add_zone(zone)
        
        logger.info(f"CustomerJourneyAnalyzer initialized with {len(self.zones)} zones")
    
    def add_zone(self, zone: Zone) -> None:
        """Add a zone to the analyzer."""
        self.zones[zone.zone_id] = zone
        logger.debug(f"Added zone: {zone.zone_id} ({zone.zone_type.value})")
    
    def remove_zone(self, zone_id: str) -> None:
        """Remove a zone."""
        if zone_id in self.zones:
            del self.zones[zone_id]
    
    def get_zone_at_point(self, x: int, y: int) -> Optional[Zone]:
        """Get zone containing the given point."""
        for zone in self.zones.values():
            if zone.contains_point(x, y):
                return zone
        return None
    
    def update_position(
        self,
        track_id: int,
        x: int,
        y: int,
        timestamp: float,
        class_name: str = "person",
        confidence: float = 1.0,
        appearance_features: Optional[np.ndarray] = None
    ) -> Optional[CustomerJourney]:
        """
        Update customer position and journey.
        
        Args:
            track_id: Object tracker ID
            x: X coordinate
            y: Y coordinate
            timestamp: Unix timestamp
            class_name: Object class (should be "person")
            confidence: Detection confidence
            appearance_features: ReID features for cross-camera matching
        
        Returns:
            Updated CustomerJourney or None
        """
        # Only track people
        if class_name.lower() != "person":
            return None
        
        current_time = datetime.fromtimestamp(timestamp)
        
        # Get or create journey
        journey = self._get_or_create_journey(track_id, current_time)
        
        # Update trajectory
        journey.trajectory.append((x, y, timestamp))
        
        # Calculate distance from last point
        if len(journey.trajectory) >= 2:
            prev_x, prev_y, _ = journey.trajectory[-2]
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            journey.total_distance += distance
        
        # Update appearance features for ReID
        if appearance_features is not None:
            journey.appearance_features = appearance_features
        
        # Check zone transitions
        current_zone = self.get_zone_at_point(x, y)
        self._update_zone_visits(journey, current_zone, current_time, (x, y))
        
        # Update journey stage
        self._update_journey_stage(journey, current_zone)
        
        return journey
    
    def _get_or_create_journey(
        self,
        track_id: int,
        timestamp: datetime
    ) -> CustomerJourney:
        """Get existing journey or create new one."""
        if track_id in self._journeys:
            return self._journeys[track_id]
        
        # Create new journey
        self._total_customers += 1
        customer_id = f"C{self._total_customers:06d}"
        
        journey = CustomerJourney(
            customer_id=customer_id,
            track_id=track_id,
            start_time=timestamp
        )
        
        self._journeys[track_id] = journey
        logger.debug(f"Created journey for track {track_id}: {customer_id}")
        
        return journey
    
    def _update_zone_visits(
        self,
        journey: CustomerJourney,
        current_zone: Optional[Zone],
        timestamp: datetime,
        point: Tuple[int, int]
    ) -> None:
        """Update zone visit tracking."""
        # Get previous zone
        prev_zone_id = journey.current_zone
        current_zone_id = current_zone.zone_id if current_zone else None
        
        # Check for zone transition
        if prev_zone_id != current_zone_id:
            # Close previous zone visit
            if prev_zone_id and journey.zone_visits:
                for visit in reversed(journey.zone_visits):
                    if visit.zone_id == prev_zone_id and visit.is_active:
                        visit.exit_time = timestamp
                        visit.exit_point = point
                        
                        # Update zone statistics
                        if visit.duration_seconds >= self.dwell_threshold:
                            stats = self._zone_stats[prev_zone_id]
                            stats["total_visits"] += 1
                            stats["total_dwell_time"] += visit.duration_seconds
                            stats["visitors"].add(journey.customer_id)
                        break
            
            # Open new zone visit
            if current_zone:
                visit = ZoneVisit(
                    zone_id=current_zone.zone_id,
                    zone_name=current_zone.name,
                    zone_type=current_zone.zone_type,
                    entry_time=timestamp,
                    entry_point=point
                )
                journey.zone_visits.append(visit)
                logger.debug(
                    f"Customer {journey.customer_id} entered zone {current_zone.name}"
                )
    
    def _update_journey_stage(
        self,
        journey: CustomerJourney,
        current_zone: Optional[Zone]
    ) -> None:
        """Update journey stage based on behavior."""
        if not current_zone:
            return
        
        # Get current zone dwell time
        current_dwell = 0.0
        for visit in reversed(journey.zone_visits):
            if visit.zone_id == current_zone.zone_id and visit.is_active:
                current_dwell = (datetime.now() - visit.entry_time).total_seconds()
                break
        
        # Determine stage based on zone and dwell
        if current_zone.zone_type == ZoneType.ENTRANCE:
            journey.current_stage = JourneyStage.ENTERED
        
        elif current_zone.zone_type == ZoneType.EXIT:
            journey.current_stage = JourneyStage.EXITED
        
        elif current_zone.zone_type == ZoneType.CHECKOUT:
            if current_dwell >= self.CHECKOUT_PROXIMITY_THRESHOLD:
                journey.current_stage = JourneyStage.CHECKOUT
        
        elif current_zone.zone_type in [ZoneType.DISPLAY, ZoneType.PROMOTION]:
            if current_dwell >= self.CONSIDERING_DWELL_THRESHOLD:
                journey.current_stage = JourneyStage.CONSIDERING
            elif current_dwell >= self.ENGAGED_DWELL_THRESHOLD:
                journey.current_stage = JourneyStage.ENGAGED
            else:
                journey.current_stage = JourneyStage.BROWSING
        
        elif current_zone.zone_type == ZoneType.SERVICE:
            journey.current_stage = JourneyStage.ASSISTED
        
        else:
            if journey.current_stage == JourneyStage.ENTERED:
                journey.current_stage = JourneyStage.BROWSING
    
    def end_journey(
        self,
        track_id: int,
        timestamp: Optional[float] = None
    ) -> Optional[CustomerJourney]:
        """
        Mark journey as ended (customer exited store).
        
        Args:
            track_id: Track ID to end
            timestamp: End timestamp (uses current time if None)
        
        Returns:
            Completed journey or None
        """
        if track_id not in self._journeys:
            return None
        
        journey = self._journeys[track_id]
        end_time = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
        
        # Close any active zone visits
        for visit in journey.zone_visits:
            if visit.is_active:
                visit.exit_time = end_time
        
        journey.end_time = end_time
        journey.current_stage = JourneyStage.EXITED
        
        # Move to completed
        self._completed_journeys.append(journey)
        del self._journeys[track_id]
        
        logger.info(
            f"Journey ended: {journey.customer_id}, "
            f"duration={journey.total_duration_seconds:.1f}s, "
            f"zones={journey.num_zones}"
        )
        
        return journey
    
    def get_journey(self, track_id: int) -> Optional[CustomerJourney]:
        """Get journey by track ID."""
        return self._journeys.get(track_id)
    
    def get_active_journeys(self) -> List[CustomerJourney]:
        """Get all active journeys."""
        return list(self._journeys.values())
    
    def get_completed_journeys(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[CustomerJourney]:
        """Get completed journeys with optional time filter."""
        journeys = self._completed_journeys
        
        if since:
            journeys = [j for j in journeys if j.start_time >= since]
        
        return journeys[-limit:]
    
    def cleanup_stale_journeys(self, current_time: Optional[datetime] = None) -> int:
        """Remove journeys that haven't been updated recently."""
        current = current_time or datetime.now()
        timeout = timedelta(seconds=self.journey_timeout)
        
        stale_tracks = []
        for track_id, journey in self._journeys.items():
            if journey.trajectory:
                last_update = datetime.fromtimestamp(journey.trajectory[-1][2])
                if current - last_update > timeout:
                    stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            self.end_journey(track_id, current.timestamp())
        
        return len(stale_tracks)
    
    def get_zone_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all zones."""
        stats = {}
        
        for zone_id, zone in self.zones.items():
            zone_stats = self._zone_stats[zone_id]
            total_visits = zone_stats["total_visits"]
            
            stats[zone_id] = {
                "zone_name": zone.name,
                "zone_type": zone.zone_type.value,
                "total_visits": total_visits,
                "unique_visitors": len(zone_stats["visitors"]),
                "total_dwell_time_seconds": round(zone_stats["total_dwell_time"], 2),
                "average_dwell_time_seconds": round(
                    zone_stats["total_dwell_time"] / total_visits
                    if total_visits > 0 else 0,
                    2
                )
            }
        
        return stats
    
    def get_popular_paths(self, min_count: int = 5) -> List[Dict[str, Any]]:
        """Analyze common customer paths through zones."""
        path_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        
        all_journeys = list(self._journeys.values()) + self._completed_journeys
        
        for journey in all_journeys:
            if len(journey.zones_visited) >= 2:
                path = tuple(journey.zones_visited)
                path_counts[path] += 1
        
        # Sort by frequency
        popular = sorted(
            path_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                "path": list(path),
                "count": count,
                "percentage": round(count / len(all_journeys) * 100, 2)
                if all_journeys else 0
            }
            for path, count in popular
            if count >= min_count
        ]
    
    def get_conversion_funnel(self) -> Dict[str, Any]:
        """Analyze customer conversion funnel."""
        all_journeys = list(self._journeys.values()) + self._completed_journeys
        
        if not all_journeys:
            return {"stages": {}, "conversion_rates": {}}
        
        stage_counts = defaultdict(int)
        for journey in all_journeys:
            # Count highest stage reached
            stage_reached = self._get_highest_stage(journey)
            stage_counts[stage_reached.value] += 1
        
        total = len(all_journeys)
        
        return {
            "total_customers": total,
            "stages": {
                stage.value: stage_counts.get(stage.value, 0)
                for stage in JourneyStage
            },
            "conversion_rates": {
                "browsing_rate": round(
                    (stage_counts["browsing"] + stage_counts["engaged"] +
                     stage_counts["considering"] + stage_counts["checkout"]) / total * 100, 2
                ),
                "engagement_rate": round(
                    (stage_counts["engaged"] + stage_counts["considering"] +
                     stage_counts["checkout"]) / total * 100, 2
                ),
                "consideration_rate": round(
                    (stage_counts["considering"] + stage_counts["checkout"]) / total * 100, 2
                ),
                "checkout_rate": round(
                    stage_counts["checkout"] / total * 100, 2
                )
            }
        }
    
    def _get_highest_stage(self, journey: CustomerJourney) -> JourneyStage:
        """Determine highest stage reached in journey."""
        stage_order = [
            JourneyStage.ENTERED,
            JourneyStage.BROWSING,
            JourneyStage.ENGAGED,
            JourneyStage.CONSIDERING,
            JourneyStage.ASSISTED,
            JourneyStage.CHECKOUT,
            JourneyStage.EXITED
        ]
        
        # Check zones visited to determine stages
        highest = JourneyStage.ENTERED
        
        for visit in journey.zone_visits:
            if visit.zone_type == ZoneType.CHECKOUT and visit.duration_seconds >= 10:
                return JourneyStage.CHECKOUT
            
            if visit.zone_type in [ZoneType.DISPLAY, ZoneType.PROMOTION]:
                if visit.duration_seconds >= self.CONSIDERING_DWELL_THRESHOLD:
                    if stage_order.index(JourneyStage.CONSIDERING) > stage_order.index(highest):
                        highest = JourneyStage.CONSIDERING
                elif visit.duration_seconds >= self.ENGAGED_DWELL_THRESHOLD:
                    if stage_order.index(JourneyStage.ENGAGED) > stage_order.index(highest):
                        highest = JourneyStage.ENGAGED
                else:
                    if stage_order.index(JourneyStage.BROWSING) > stage_order.index(highest):
                        highest = JourneyStage.BROWSING
            
            elif visit.zone_type == ZoneType.SERVICE:
                if stage_order.index(JourneyStage.ASSISTED) > stage_order.index(highest):
                    highest = JourneyStage.ASSISTED
        
        return highest
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get overall journey statistics."""
        all_journeys = list(self._journeys.values()) + self._completed_journeys
        completed = self._completed_journeys
        
        if not completed:
            return {
                "active_customers": len(self._journeys),
                "total_customers": self._total_customers,
                "completed_journeys": 0
            }
        
        durations = [j.total_duration_seconds for j in completed]
        distances = [j.total_distance for j in completed]
        zones = [j.num_zones for j in completed]
        
        return {
            "active_customers": len(self._journeys),
            "total_customers": self._total_customers,
            "completed_journeys": len(completed),
            "average_duration_seconds": round(np.mean(durations), 2),
            "median_duration_seconds": round(np.median(durations), 2),
            "average_distance": round(np.mean(distances), 2),
            "average_zones_visited": round(np.mean(zones), 2),
            "funnel": self.get_conversion_funnel()
        }


class DwellTimeAnalyzer:
    """
    Specialized dwell time analysis.
    
    Provides detailed dwell metrics for display optimization.
    """
    
    def __init__(
        self,
        min_dwell_seconds: float = 2.0,
        max_dwell_seconds: float = 300.0
    ):
        """Initialize dwell analyzer."""
        self.min_dwell = min_dwell_seconds
        self.max_dwell = max_dwell_seconds
        
        # Dwell records by zone
        self._dwell_records: Dict[str, List[Dict]] = defaultdict(list)
    
    def record_dwell(
        self,
        zone_id: str,
        customer_id: str,
        dwell_seconds: float,
        timestamp: datetime
    ) -> None:
        """Record a dwell event."""
        if self.min_dwell <= dwell_seconds <= self.max_dwell:
            self._dwell_records[zone_id].append({
                "customer_id": customer_id,
                "dwell_seconds": dwell_seconds,
                "timestamp": timestamp
            })
    
    def get_zone_dwell_stats(self, zone_id: str) -> Dict[str, Any]:
        """Get dwell statistics for a zone."""
        records = self._dwell_records.get(zone_id, [])
        
        if not records:
            return {
                "zone_id": zone_id,
                "total_dwells": 0,
                "avg_dwell_seconds": 0,
                "median_dwell_seconds": 0,
                "engagement_score": 0
            }
        
        dwells = [r["dwell_seconds"] for r in records]
        
        # Calculate engagement score (0-100)
        avg_dwell = np.mean(dwells)
        engagement_score = min(100, (avg_dwell / 30) * 100)  # 30s = 100 score
        
        return {
            "zone_id": zone_id,
            "total_dwells": len(records),
            "avg_dwell_seconds": round(np.mean(dwells), 2),
            "median_dwell_seconds": round(np.median(dwells), 2),
            "min_dwell_seconds": round(min(dwells), 2),
            "max_dwell_seconds": round(max(dwells), 2),
            "std_dwell_seconds": round(np.std(dwells), 2),
            "engagement_score": round(engagement_score, 1)
        }
    
    def get_hourly_patterns(self, zone_id: str) -> Dict[int, float]:
        """Get hourly dwell patterns for a zone."""
        records = self._dwell_records.get(zone_id, [])
        
        hourly: Dict[int, List[float]] = defaultdict(list)
        for record in records:
            hour = record["timestamp"].hour
            hourly[hour].append(record["dwell_seconds"])
        
        return {
            hour: round(np.mean(dwells), 2)
            for hour, dwells in hourly.items()
        }


if __name__ == "__main__":
    # Demo usage
    import time
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer with sample zones
    analyzer = CustomerJourneyAnalyzer()
    
    # Add zones
    analyzer.add_zone(Zone(
        zone_id="entrance",
        name="Main Entrance",
        zone_type=ZoneType.ENTRANCE,
        polygon=[(0, 0), (200, 0), (200, 100), (0, 100)]
    ))
    
    analyzer.add_zone(Zone(
        zone_id="aisle_1",
        name="Aisle 1 - Electronics",
        zone_type=ZoneType.AISLE,
        polygon=[(0, 100), (200, 100), (200, 400), (0, 400)]
    ))
    
    analyzer.add_zone(Zone(
        zone_id="display_1",
        name="Featured Products",
        zone_type=ZoneType.DISPLAY,
        polygon=[(200, 200), (400, 200), (400, 300), (200, 300)]
    ))
    
    analyzer.add_zone(Zone(
        zone_id="checkout",
        name="Checkout Area",
        zone_type=ZoneType.CHECKOUT,
        polygon=[(400, 0), (600, 0), (600, 200), (400, 200)]
    ))
    
    # Simulate customer journey
    base_time = time.time()
    track_id = 1
    
    # Customer enters
    analyzer.update_position(track_id, 100, 50, base_time, "person")
    print(f"Step 1: {analyzer.get_journey(track_id).current_stage.value}")
    
    # Customer browses aisle
    analyzer.update_position(track_id, 100, 200, base_time + 5, "person")
    print(f"Step 2: {analyzer.get_journey(track_id).current_stage.value}")
    
    # Customer at display (engaged)
    for i in range(10):
        analyzer.update_position(track_id, 300, 250, base_time + 10 + i, "person")
    print(f"Step 3: {analyzer.get_journey(track_id).current_stage.value}")
    
    # Customer at checkout
    for i in range(15):
        analyzer.update_position(track_id, 500, 100, base_time + 25 + i, "person")
    print(f"Step 4: {analyzer.get_journey(track_id).current_stage.value}")
    
    # Get journey summary
    journey = analyzer.get_journey(track_id)
    print(f"\nJourney Summary:")
    print(f"  Customer ID: {journey.customer_id}")
    print(f"  Zones visited: {journey.zones_visited}")
    print(f"  Total distance: {journey.total_distance:.1f} pixels")
    print(f"  Duration: {journey.total_duration_seconds:.1f} seconds")
    
    # Get zone statistics
    print(f"\nZone Statistics:")
    for zone_id, stats in analyzer.get_zone_statistics().items():
        print(f"  {zone_id}: {stats['total_visits']} visits")
