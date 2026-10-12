#!/usr/bin/env python3
"""
Analytics Demo.

Demonstrate customer journey tracking, queue monitoring, and heatmap generation
using sample data or live detection feeds.

Usage:
    python examples/analytics_demo.py
    python examples/analytics_demo.py --data data/sample/detections.json
    python examples/analytics_demo.py --export report.json
"""

import argparse
import json
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ZoneConfig:
    """Zone configuration."""
    id: str
    name: str
    type: str
    polygon: List[Tuple[float, float]]


@dataclass
class CustomerJourney:
    """Customer journey data."""
    journey_id: str
    track_id: int
    start_time: datetime
    end_time: Optional[datetime]
    zones_visited: List[str]
    zone_dwell_times: Dict[str, float]
    converted: bool
    cart_detected: bool


@dataclass
class QueueMetrics:
    """Queue monitoring metrics."""
    lane_id: str
    timestamp: datetime
    queue_length: int
    avg_wait_time_seconds: float
    max_wait_time_seconds: float
    service_rate: float
    abandonment_count: int


@dataclass
class HeatmapData:
    """Heatmap accumulator."""
    resolution: Tuple[int, int]
    data: np.ndarray
    hotspots: List[Dict[str, float]]


# Sample zone configuration
SAMPLE_ZONES = [
    ZoneConfig("entrance", "Main Entrance", "entrance", [(0.0, 0.8), (0.15, 0.8), (0.15, 1.0), (0.0, 1.0)]),
    ZoneConfig("aisle-1", "Produce Aisle", "aisle", [(0.0, 0.25), (0.30, 0.25), (0.30, 0.60), (0.0, 0.60)]),
    ZoneConfig("aisle-2", "Dairy Aisle", "aisle", [(0.32, 0.25), (0.60, 0.25), (0.60, 0.60), (0.32, 0.60)]),
    ZoneConfig("aisle-3", "Electronics", "aisle", [(0.62, 0.25), (0.90, 0.25), (0.90, 0.60), (0.62, 0.60)]),
    ZoneConfig("checkout-1", "Checkout Lane 1", "checkout", [(0.68, 0.70), (0.78, 0.70), (0.78, 0.95), (0.68, 0.95)]),
    ZoneConfig("checkout-2", "Checkout Lane 2", "checkout", [(0.80, 0.70), (0.90, 0.70), (0.90, 0.95), (0.80, 0.95)]),
]


class JourneyTracker:
    """Track customer journeys through the store."""
    
    def __init__(self, zones: List[ZoneConfig]):
        self.zones = {z.id: z for z in zones}
        self.active_journeys: Dict[int, CustomerJourney] = {}
        self.completed_journeys: List[CustomerJourney] = []
        self.next_journey_id = 1
    
    def update(
        self,
        track_id: int,
        position: Tuple[float, float],
        timestamp: datetime,
        has_cart: bool = False,
    ):
        """Update journey with new position."""
        current_zone = self._get_zone_at_position(position)
        
        if track_id not in self.active_journeys:
            # Start new journey
            journey = CustomerJourney(
                journey_id=f"journey-{self.next_journey_id:06d}",
                track_id=track_id,
                start_time=timestamp,
                end_time=None,
                zones_visited=[current_zone] if current_zone else [],
                zone_dwell_times={},
                converted=False,
                cart_detected=has_cart,
            )
            self.active_journeys[track_id] = journey
            self.next_journey_id += 1
        else:
            journey = self.active_journeys[track_id]
            
            # Update zone visits
            if current_zone and (not journey.zones_visited or 
                                  journey.zones_visited[-1] != current_zone):
                journey.zones_visited.append(current_zone)
            
            # Update cart detection
            if has_cart:
                journey.cart_detected = True
            
            # Update dwell time
            if current_zone:
                journey.zone_dwell_times[current_zone] = \
                    journey.zone_dwell_times.get(current_zone, 0) + 1.0
    
    def complete_journey(self, track_id: int, timestamp: datetime):
        """Complete a journey when track is lost."""
        if track_id in self.active_journeys:
            journey = self.active_journeys.pop(track_id)
            journey.end_time = timestamp
            
            # Check if converted (visited checkout)
            checkout_zones = [z for z in journey.zones_visited 
                            if z.startswith("checkout")]
            journey.converted = len(checkout_zones) > 0
            
            self.completed_journeys.append(journey)
    
    def _get_zone_at_position(self, position: Tuple[float, float]) -> Optional[str]:
        """Get zone ID at position."""
        x, y = position
        
        for zone_id, zone in self.zones.items():
            if self._point_in_polygon((x, y), zone.polygon):
                return zone_id
        
        return None
    
    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]],
    ) -> bool:
        """Ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get journey analytics."""
        all_journeys = self.completed_journeys + list(self.active_journeys.values())
        
        if not all_journeys:
            return {}
        
        # Calculate metrics
        total_journeys = len(all_journeys)
        completed = [j for j in all_journeys if j.end_time]
        converted = [j for j in completed if j.converted]
        
        # Average dwell times per zone
        zone_dwell_times: Dict[str, List[float]] = {}
        for journey in all_journeys:
            for zone, dwell in journey.zone_dwell_times.items():
                zone_dwell_times.setdefault(zone, []).append(dwell)
        
        avg_dwell_times = {
            zone: np.mean(times) for zone, times in zone_dwell_times.items()
        }
        
        # Journey durations
        durations = []
        for j in completed:
            if j.end_time and j.start_time:
                duration = (j.end_time - j.start_time).total_seconds()
                durations.append(duration)
        
        # Most common paths
        path_counts: Dict[str, int] = {}
        for journey in completed:
            path = " → ".join(journey.zones_visited[:5])  # First 5 zones
            path_counts[path] = path_counts.get(path, 0) + 1
        
        top_paths = sorted(path_counts.items(), key=lambda x: -x[1])[:5]
        
        return {
            "total_journeys": total_journeys,
            "completed_journeys": len(completed),
            "active_journeys": len(self.active_journeys),
            "conversion_rate": len(converted) / len(completed) if completed else 0,
            "cart_usage_rate": sum(1 for j in all_journeys if j.cart_detected) / total_journeys,
            "avg_journey_duration_seconds": np.mean(durations) if durations else 0,
            "avg_zones_visited": np.mean([len(j.zones_visited) for j in all_journeys]),
            "avg_dwell_times_by_zone": avg_dwell_times,
            "top_paths": [{"path": p, "count": c} for p, c in top_paths],
        }


class QueueMonitor:
    """Monitor checkout queue metrics."""
    
    def __init__(self, lane_ids: List[str]):
        self.lane_ids = lane_ids
        self.queue_history: Dict[str, List[QueueMetrics]] = {
            lane: [] for lane in lane_ids
        }
        self.current_queues: Dict[str, List[Tuple[int, datetime]]] = {
            lane: [] for lane in lane_ids
        }
    
    def update(
        self,
        lane_id: str,
        track_ids: List[int],
        timestamp: datetime,
    ):
        """Update queue state for a lane."""
        if lane_id not in self.lane_ids:
            return
        
        current = self.current_queues[lane_id]
        
        # Calculate wait times
        wait_times = []
        for track_id, enter_time in current:
            if track_id in track_ids:
                wait_time = (timestamp - enter_time).total_seconds()
                wait_times.append(wait_time)
        
        # Add new arrivals
        existing_ids = {t[0] for t in current}
        for track_id in track_ids:
            if track_id not in existing_ids:
                current.append((track_id, timestamp))
        
        # Remove departed
        self.current_queues[lane_id] = [
            (tid, t) for tid, t in current if tid in track_ids
        ]
        
        # Record metrics
        metrics = QueueMetrics(
            lane_id=lane_id,
            timestamp=timestamp,
            queue_length=len(track_ids),
            avg_wait_time_seconds=np.mean(wait_times) if wait_times else 0,
            max_wait_time_seconds=max(wait_times) if wait_times else 0,
            service_rate=1.0,  # Simplified
            abandonment_count=0,
        )
        
        self.queue_history[lane_id].append(metrics)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get queue analytics."""
        all_metrics: List[QueueMetrics] = []
        for lane_metrics in self.queue_history.values():
            all_metrics.extend(lane_metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate by lane
        lane_stats = {}
        for lane_id in self.lane_ids:
            metrics = self.queue_history[lane_id]
            if metrics:
                lane_stats[lane_id] = {
                    "avg_queue_length": np.mean([m.queue_length for m in metrics]),
                    "max_queue_length": max(m.queue_length for m in metrics),
                    "avg_wait_time_seconds": np.mean([m.avg_wait_time_seconds for m in metrics]),
                    "max_wait_time_seconds": max(m.max_wait_time_seconds for m in metrics),
                }
        
        return {
            "total_observations": len(all_metrics),
            "overall_avg_queue_length": np.mean([m.queue_length for m in all_metrics]),
            "overall_avg_wait_time": np.mean([m.avg_wait_time_seconds for m in all_metrics]),
            "lane_statistics": lane_stats,
        }


class HeatmapGenerator:
    """Generate traffic heatmaps."""
    
    def __init__(self, resolution: Tuple[int, int] = (96, 54)):
        self.resolution = resolution
        self.data = np.zeros(resolution[::-1], dtype=np.float32)  # H x W
        self.total_updates = 0
    
    def update(self, position: Tuple[float, float], weight: float = 1.0):
        """Add position to heatmap."""
        x, y = position
        
        # Convert normalized coords to grid
        grid_x = int(x * (self.resolution[0] - 1))
        grid_y = int(y * (self.resolution[1] - 1))
        
        # Bounds check
        grid_x = max(0, min(grid_x, self.resolution[0] - 1))
        grid_y = max(0, min(grid_y, self.resolution[1] - 1))
        
        # Add Gaussian blob
        sigma = 2
        for dy in range(-sigma * 2, sigma * 2 + 1):
            for dx in range(-sigma * 2, sigma * 2 + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.resolution[0] and 0 <= ny < self.resolution[1]:
                    dist = np.sqrt(dx**2 + dy**2)
                    value = weight * np.exp(-dist**2 / (2 * sigma**2))
                    self.data[ny, nx] += value
        
        self.total_updates += 1
    
    def get_hotspots(self, top_n: int = 5) -> List[Dict[str, float]]:
        """Get top N hotspot locations."""
        # Normalize
        normalized = self.data / (self.data.max() + 1e-6)
        
        hotspots = []
        data_copy = normalized.copy()
        
        for _ in range(top_n):
            max_idx = np.unravel_index(np.argmax(data_copy), data_copy.shape)
            y, x = max_idx
            
            if data_copy[y, x] < 0.1:
                break
            
            hotspots.append({
                "x": x / (self.resolution[0] - 1),
                "y": y / (self.resolution[1] - 1),
                "intensity": float(data_copy[y, x]),
            })
            
            # Suppress area around hotspot
            y1, y2 = max(0, y - 5), min(self.resolution[1], y + 6)
            x1, x2 = max(0, x - 5), min(self.resolution[0], x + 6)
            data_copy[y1:y2, x1:x2] = 0
        
        return hotspots
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get heatmap analytics."""
        normalized = self.data / (self.data.max() + 1e-6)
        
        return {
            "resolution": self.resolution,
            "total_updates": self.total_updates,
            "max_intensity": float(self.data.max()),
            "mean_intensity": float(self.data.mean()),
            "coverage_percent": float((normalized > 0.01).sum() / normalized.size * 100),
            "hotspots": self.get_hotspots(5),
        }


def generate_sample_data(
    duration_minutes: int = 60,
    avg_customers_per_hour: int = 100,
) -> List[Dict[str, Any]]:
    """Generate sample detection data for demonstration."""
    print(f"Generating {duration_minutes} minutes of sample data...")
    
    detections = []
    start_time = datetime.now() - timedelta(minutes=duration_minutes)
    
    # Simulate customers
    num_customers = int(avg_customers_per_hour * duration_minutes / 60)
    
    for customer_id in range(num_customers):
        # Random entry time
        entry_offset = random.uniform(0, duration_minutes * 60)
        entry_time = start_time + timedelta(seconds=entry_offset)
        
        # Random journey duration
        journey_duration = random.uniform(120, 900)  # 2-15 minutes
        
        # Generate path
        has_cart = random.random() > 0.6
        will_convert = random.random() > 0.3
        
        # Simulate movement
        zones = ["entrance"]
        if random.random() > 0.2:
            zones.append("aisle-1")
        if random.random() > 0.3:
            zones.append("aisle-2")
        if random.random() > 0.5:
            zones.append("aisle-3")
        if will_convert:
            zones.append(random.choice(["checkout-1", "checkout-2"]))
        
        # Generate detections along path
        time_per_zone = journey_duration / len(zones)
        current_time = entry_time
        
        for zone_idx, zone in enumerate(zones):
            zone_config = next((z for z in SAMPLE_ZONES if z.id == zone), None)
            if not zone_config:
                continue
            
            # Generate positions within zone
            num_detections = int(time_per_zone / 0.5)  # Every 0.5 seconds
            
            for _ in range(max(1, num_detections)):
                # Random position within zone polygon
                min_x = min(p[0] for p in zone_config.polygon)
                max_x = max(p[0] for p in zone_config.polygon)
                min_y = min(p[1] for p in zone_config.polygon)
                max_y = max(p[1] for p in zone_config.polygon)
                
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                
                detections.append({
                    "track_id": customer_id,
                    "timestamp": current_time.isoformat(),
                    "position": [x, y],
                    "class": "person",
                    "has_cart": has_cart,
                    "confidence": random.uniform(0.7, 0.98),
                })
                
                current_time += timedelta(seconds=0.5)
    
    # Sort by timestamp
    detections.sort(key=lambda d: d["timestamp"])
    
    print(f"Generated {len(detections)} detection events")
    return detections


def run_analytics_demo(
    data_path: Optional[str] = None,
    export_path: Optional[str] = None,
):
    """Run analytics demonstration."""
    print("=" * 60)
    print("Retail Vision Analytics Demo")
    print("=" * 60)
    
    # Load or generate data
    if data_path and Path(data_path).exists():
        print(f"\nLoading data from: {data_path}")
        with open(data_path) as f:
            detections = json.load(f)
    else:
        detections = generate_sample_data(duration_minutes=30)
    
    # Initialize analytics components
    journey_tracker = JourneyTracker(SAMPLE_ZONES)
    queue_monitor = QueueMonitor(["checkout-1", "checkout-2"])
    heatmap_generator = HeatmapGenerator()
    
    print(f"\nProcessing {len(detections)} detection events...")
    
    # Process detections
    for det in detections:
        timestamp = datetime.fromisoformat(det["timestamp"])
        position = tuple(det["position"])
        track_id = det["track_id"]
        has_cart = det.get("has_cart", False)
        
        # Update journey tracker
        journey_tracker.update(track_id, position, timestamp, has_cart)
        
        # Update heatmap
        heatmap_generator.update(position)
        
        # Check if in checkout zone
        for zone in SAMPLE_ZONES:
            if zone.type == "checkout":
                if journey_tracker._point_in_polygon(position, zone.polygon):
                    queue_monitor.update(zone.id, [track_id], timestamp)
    
    # Complete remaining journeys
    final_time = datetime.now()
    for track_id in list(journey_tracker.active_journeys.keys()):
        journey_tracker.complete_journey(track_id, final_time)
    
    # Get analytics
    journey_analytics = journey_tracker.get_analytics()
    queue_analytics = queue_monitor.get_analytics()
    heatmap_analytics = heatmap_generator.get_analytics()
    
    # Print results
    print("\n" + "=" * 60)
    print("ANALYTICS RESULTS")
    print("=" * 60)
    
    print("\n📊 Customer Journey Analytics")
    print("-" * 40)
    print(f"  Total journeys:        {journey_analytics.get('total_journeys', 0)}")
    print(f"  Completed journeys:    {journey_analytics.get('completed_journeys', 0)}")
    print(f"  Conversion rate:       {journey_analytics.get('conversion_rate', 0):.1%}")
    print(f"  Cart usage rate:       {journey_analytics.get('cart_usage_rate', 0):.1%}")
    print(f"  Avg journey duration:  {journey_analytics.get('avg_journey_duration_seconds', 0):.0f}s")
    print(f"  Avg zones visited:     {journey_analytics.get('avg_zones_visited', 0):.1f}")
    
    if journey_analytics.get("top_paths"):
        print("\n  Top Paths:")
        for path_info in journey_analytics["top_paths"][:3]:
            print(f"    {path_info['path']} ({path_info['count']} customers)")
    
    print("\n⏱️ Queue Analytics")
    print("-" * 40)
    print(f"  Avg queue length:      {queue_analytics.get('overall_avg_queue_length', 0):.1f}")
    print(f"  Avg wait time:         {queue_analytics.get('overall_avg_wait_time', 0):.0f}s")
    
    if queue_analytics.get("lane_statistics"):
        print("\n  By Lane:")
        for lane_id, stats in queue_analytics["lane_statistics"].items():
            print(f"    {lane_id}: {stats['avg_queue_length']:.1f} avg, "
                  f"{stats['avg_wait_time_seconds']:.0f}s wait")
    
    print("\n🔥 Heatmap Analytics")
    print("-" * 40)
    print(f"  Total position updates: {heatmap_analytics.get('total_updates', 0)}")
    print(f"  Coverage:               {heatmap_analytics.get('coverage_percent', 0):.1f}%")
    
    if heatmap_analytics.get("hotspots"):
        print("\n  Top Hotspots:")
        for i, hotspot in enumerate(heatmap_analytics["hotspots"][:3], 1):
            print(f"    #{i}: ({hotspot['x']:.2f}, {hotspot['y']:.2f}) "
                  f"- intensity {hotspot['intensity']:.2f}")
    
    # Export results
    if export_path:
        report = {
            "generated_at": datetime.now().isoformat(),
            "detection_count": len(detections),
            "journey_analytics": journey_analytics,
            "queue_analytics": queue_analytics,
            "heatmap_analytics": heatmap_analytics,
        }
        
        with open(export_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📁 Report exported to: {export_path}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Analytics Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to detection data JSON",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export analytics report to JSON",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in minutes for sample data generation",
    )
    
    args = parser.parse_args()
    
    run_analytics_demo(
        data_path=args.data,
        export_path=args.export,
    )


if __name__ == "__main__":
    main()
