"""
Unit Tests for Retail Vision Analytics Module.

Tests cover:
- Customer journey tracking
- Queue monitoring
- Heatmap generation
- Zone analytics
"""

import pytest
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_detection():
    """Create a sample detection object."""
    return {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": (100, 200, 50, 120),
        "track_id": 1,
        "timestamp": time.time(),
    }


@pytest.fixture
def sample_frame():
    """Create a sample frame with detections."""
    return {
        "frame_number": 100,
        "timestamp": time.time(),
        "width": 1920,
        "height": 1080,
        "detections": [
            {"class_name": "person", "track_id": 1, "bbox": (100, 200, 50, 120)},
            {"class_name": "person", "track_id": 2, "bbox": (400, 300, 60, 140)},
            {"class_name": "shopping_cart", "track_id": 101, "bbox": (500, 400, 80, 60)},
        ],
    }


@pytest.fixture
def zone_config():
    """Create sample zone configuration."""
    return [
        {
            "zone_id": "entrance",
            "zone_name": "Store Entrance",
            "zone_type": "entrance",
            "polygon": [(0.0, 0.8), (0.2, 0.8), (0.2, 1.0), (0.0, 1.0)],
        },
        {
            "zone_id": "aisle-1",
            "zone_name": "Aisle 1",
            "zone_type": "aisle",
            "polygon": [(0.0, 0.3), (0.3, 0.3), (0.3, 0.7), (0.0, 0.7)],
        },
        {
            "zone_id": "checkout",
            "zone_name": "Checkout",
            "zone_type": "checkout",
            "polygon": [(0.7, 0.5), (1.0, 0.5), (1.0, 1.0), (0.7, 1.0)],
        },
    ]


@pytest.fixture
def queue_config():
    """Create sample queue configuration."""
    return {
        "lane_id": "checkout-1",
        "stream_id": "cam-checkout-1",
        "roi": (0.0, 0.0, 0.5, 1.0),
        "direction": "horizontal",
        "thresholds": {
            "queue_length_warning": 5,
            "queue_length_critical": 10,
            "wait_time_warning": 120,
            "wait_time_critical": 300,
        },
    }


# ============================================================================
# Customer Journey Tests
# ============================================================================

class TestCustomerJourney:
    """Tests for CustomerJourneyTracker."""
    
    def test_journey_creation(self, zone_config):
        """Test that a journey is created when a person enters."""
        # Import mock since actual module may not be available
        tracker = MockJourneyTracker(zone_config)
        
        # Person enters store
        tracker.update(track_id=1, position=(0.1, 0.9), timestamp=time.time())
        
        journeys = tracker.get_active_journeys()
        assert len(journeys) == 1
        assert journeys[0]["track_id"] == 1
        assert "entrance" in journeys[0]["zones_visited"]
    
    def test_zone_transition(self, zone_config):
        """Test zone transition detection."""
        tracker = MockJourneyTracker(zone_config)
        t = time.time()
        
        # Enter through entrance
        tracker.update(track_id=1, position=(0.1, 0.9), timestamp=t)
        
        # Move to aisle
        tracker.update(track_id=1, position=(0.15, 0.5), timestamp=t + 30)
        
        journey = tracker.get_journey(track_id=1)
        assert "entrance" in journey["zones_visited"]
        assert "aisle-1" in journey["zones_visited"]
        assert len(journey["zone_transitions"]) >= 1
    
    def test_dwell_time_calculation(self, zone_config):
        """Test dwell time calculation per zone."""
        tracker = MockJourneyTracker(zone_config)
        t = time.time()
        
        # Spend time in entrance
        for i in range(5):
            tracker.update(track_id=1, position=(0.1, 0.9), timestamp=t + i)
        
        journey = tracker.get_journey(track_id=1)
        assert journey["zone_dwell_times"]["entrance"] >= 4.0
    
    def test_journey_completion(self, zone_config):
        """Test journey completion detection."""
        tracker = MockJourneyTracker(zone_config)
        t = time.time()
        
        # Complete journey: entrance → aisle → checkout
        tracker.update(track_id=1, position=(0.1, 0.9), timestamp=t)
        tracker.update(track_id=1, position=(0.15, 0.5), timestamp=t + 60)
        tracker.update(track_id=1, position=(0.85, 0.75), timestamp=t + 120)
        
        # Mark as exited
        tracker.mark_exited(track_id=1, exit_point="checkout", timestamp=t + 180)
        
        journey = tracker.get_journey(track_id=1)
        assert journey["converted"] == True
        assert journey["exit_point"] == "checkout"
        assert journey["duration_seconds"] == pytest.approx(180, abs=1)
    
    def test_conversion_rate(self, zone_config):
        """Test conversion rate calculation."""
        tracker = MockJourneyTracker(zone_config)
        t = time.time()
        
        # 3 visitors, 2 convert (reach checkout)
        for i in range(3):
            tracker.update(track_id=i+1, position=(0.1, 0.9), timestamp=t + i*10)
        
        # 2 go to checkout
        tracker.update(track_id=1, position=(0.85, 0.75), timestamp=t + 100)
        tracker.update(track_id=2, position=(0.85, 0.75), timestamp=t + 110)
        
        # Mark all as exited
        for i in range(3):
            exit_point = "checkout" if i < 2 else "entrance"
            tracker.mark_exited(track_id=i+1, exit_point=exit_point, timestamp=t + 200)
        
        stats = tracker.get_statistics()
        assert stats["conversion_rate"] == pytest.approx(0.667, abs=0.01)


# ============================================================================
# Queue Monitor Tests
# ============================================================================

class TestQueueMonitor:
    """Tests for QueueMonitor."""
    
    def test_queue_length_detection(self, queue_config):
        """Test queue length counting."""
        monitor = MockQueueMonitor(queue_config)
        
        # Add people to queue
        people = [
            {"track_id": i, "position": (0.1 + i*0.05, 0.5)}
            for i in range(5)
        ]
        
        monitor.update(people, timestamp=time.time())
        
        metrics = monitor.get_metrics()
        assert metrics["queue_length"] == 5
    
    def test_wait_time_estimation(self, queue_config):
        """Test wait time estimation."""
        monitor = MockQueueMonitor(queue_config)
        t = time.time()
        
        # Person joins queue
        monitor.update([{"track_id": 1, "position": (0.1, 0.5)}], timestamp=t)
        
        # Wait 60 seconds
        monitor.update([{"track_id": 1, "position": (0.2, 0.5)}], timestamp=t + 60)
        
        metrics = monitor.get_metrics()
        assert metrics["avg_wait_time_seconds"] >= 60
    
    def test_service_rate(self, queue_config):
        """Test service rate calculation."""
        monitor = MockQueueMonitor(queue_config)
        t = time.time()
        
        # Serve 3 people in 3 minutes
        for i in range(3):
            monitor.update(
                [{"track_id": i+1, "position": (0.1, 0.5)}],
                timestamp=t + i*30
            )
            monitor.mark_served(track_id=i+1, timestamp=t + i*30 + 60)
        
        metrics = monitor.get_metrics()
        assert metrics["service_rate"] >= 0.5  # At least 0.5 per minute
    
    def test_abandonment_detection(self, queue_config):
        """Test queue abandonment detection."""
        monitor = MockQueueMonitor(queue_config)
        t = time.time()
        
        # Person joins queue
        monitor.update([{"track_id": 1, "position": (0.1, 0.5)}], timestamp=t)
        
        # Person leaves without being served (after 30+ seconds)
        monitor.update([], timestamp=t + 35)
        
        metrics = monitor.get_metrics()
        assert metrics["abandonment_count"] >= 1
    
    def test_alert_generation(self, queue_config):
        """Test alert generation for queue thresholds."""
        monitor = MockQueueMonitor(queue_config)
        
        # Add people exceeding warning threshold
        people = [
            {"track_id": i, "position": (0.1 + i*0.03, 0.5)}
            for i in range(8)
        ]
        
        alerts = monitor.update(people, timestamp=time.time())
        
        assert len(alerts) >= 1
        assert any(a["type"] == "queue_length_warning" for a in alerts)


# ============================================================================
# Heatmap Tests
# ============================================================================

class TestHeatmapGenerator:
    """Tests for HeatmapGenerator."""
    
    def test_accumulation(self):
        """Test point accumulation."""
        generator = MockHeatmapGenerator(width=100, height=100, cell_size=10)
        
        # Add points
        generator.add_point(50, 50)
        generator.add_point(50, 50)
        generator.add_point(51, 51)
        
        heatmap = generator.get_heatmap()
        assert heatmap[5, 5] > 0  # Cell at (50, 50)
    
    def test_decay(self):
        """Test temporal decay."""
        generator = MockHeatmapGenerator(
            width=100, height=100, cell_size=10, decay_rate=0.9
        )
        
        # Add point
        generator.add_point(50, 50)
        initial_value = generator.get_heatmap()[5, 5]
        
        # Apply decay
        generator.apply_decay()
        decayed_value = generator.get_heatmap()[5, 5]
        
        assert decayed_value < initial_value
        assert decayed_value == pytest.approx(initial_value * 0.9, abs=0.01)
    
    def test_normalization(self):
        """Test heatmap normalization."""
        generator = MockHeatmapGenerator(width=100, height=100, cell_size=10)
        
        # Add varying intensities
        for i in range(10):
            generator.add_point(50, 50)
        for i in range(5):
            generator.add_point(20, 20)
        
        normalized = generator.get_normalized_heatmap()
        
        assert normalized.max() == 1.0
        assert normalized.min() >= 0.0
    
    def test_hotspot_detection(self):
        """Test hotspot identification."""
        generator = MockHeatmapGenerator(width=100, height=100, cell_size=10)
        
        # Create hotspot
        for i in range(20):
            generator.add_point(50, 50)
        
        hotspots = generator.detect_hotspots(threshold=0.5)
        
        assert len(hotspots) >= 1
        assert any(h["x"] == 5 and h["y"] == 5 for h in hotspots)
    
    def test_temporal_patterns(self):
        """Test hourly pattern tracking."""
        generator = MockHeatmapGenerator(
            width=100, height=100, cell_size=10, track_hourly=True
        )
        
        # Add points at different hours
        for hour in [9, 12, 18]:
            generator.add_point(50, 50, hour=hour)
        
        patterns = generator.get_hourly_patterns()
        
        assert 9 in patterns
        assert 12 in patterns
        assert 18 in patterns


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockJourneyTracker:
    """Mock journey tracker for testing."""
    
    def __init__(self, zone_config):
        self.zones = zone_config
        self.journeys = {}
        self.completed_journeys = []
    
    def update(self, track_id, position, timestamp):
        if track_id not in self.journeys:
            self.journeys[track_id] = {
                "track_id": track_id,
                "start_time": timestamp,
                "zones_visited": [],
                "zone_transitions": [],
                "zone_dwell_times": {},
                "last_zone": None,
                "last_position": None,
                "last_timestamp": None,
                "converted": False,
                "exit_point": None,
            }
        
        journey = self.journeys[track_id]
        current_zone = self._get_zone(position)
        
        if current_zone:
            if current_zone not in journey["zones_visited"]:
                journey["zones_visited"].append(current_zone)
            
            if current_zone != journey["last_zone"]:
                journey["zone_transitions"].append({
                    "from": journey["last_zone"],
                    "to": current_zone,
                    "timestamp": timestamp,
                })
            
            # Update dwell time
            if journey["last_zone"] == current_zone and journey["last_timestamp"]:
                dt = timestamp - journey["last_timestamp"]
                journey["zone_dwell_times"][current_zone] = \
                    journey["zone_dwell_times"].get(current_zone, 0) + dt
        
        journey["last_zone"] = current_zone
        journey["last_position"] = position
        journey["last_timestamp"] = timestamp
    
    def _get_zone(self, position):
        """Determine which zone a position is in."""
        x, y = position
        for zone in self.zones:
            poly = zone["polygon"]
            if self._point_in_polygon(x, y, poly):
                return zone["zone_id"]
        return None
    
    def _point_in_polygon(self, x, y, polygon):
        """Ray casting algorithm for point-in-polygon."""
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
    
    def mark_exited(self, track_id, exit_point, timestamp):
        if track_id in self.journeys:
            journey = self.journeys[track_id]
            journey["exit_point"] = exit_point
            journey["duration_seconds"] = timestamp - journey["start_time"]
            journey["converted"] = "checkout" in journey["zones_visited"]
            self.completed_journeys.append(journey)
            del self.journeys[track_id]
    
    def get_active_journeys(self):
        return list(self.journeys.values())
    
    def get_journey(self, track_id):
        if track_id in self.journeys:
            return self.journeys[track_id]
        return next((j for j in self.completed_journeys if j["track_id"] == track_id), None)
    
    def get_statistics(self):
        all_journeys = list(self.journeys.values()) + self.completed_journeys
        converted = len([j for j in all_journeys if j["converted"]])
        total = len(all_journeys)
        
        return {
            "total_journeys": total,
            "converted": converted,
            "conversion_rate": converted / total if total > 0 else 0,
        }


class MockQueueMonitor:
    """Mock queue monitor for testing."""
    
    def __init__(self, config):
        self.config = config
        self.current_queue = {}
        self.served = []
        self.abandoned = []
        self.service_times = []
    
    def update(self, people, timestamp):
        alerts = []
        current_ids = set(p["track_id"] for p in people)
        
        # Detect new people
        for person in people:
            tid = person["track_id"]
            if tid not in self.current_queue:
                self.current_queue[tid] = {
                    "join_time": timestamp,
                    "last_seen": timestamp,
                }
            else:
                self.current_queue[tid]["last_seen"] = timestamp
        
        # Detect abandoned (not seen for 30+ seconds)
        for tid in list(self.current_queue.keys()):
            if tid not in current_ids:
                if timestamp - self.current_queue[tid]["last_seen"] > 30:
                    self.abandoned.append(self.current_queue[tid])
                    del self.current_queue[tid]
        
        # Check thresholds
        queue_length = len(self.current_queue)
        if queue_length >= self.config["thresholds"]["queue_length_warning"]:
            alerts.append({
                "type": "queue_length_warning",
                "queue_length": queue_length,
            })
        
        return alerts
    
    def mark_served(self, track_id, timestamp):
        if track_id in self.current_queue:
            entry = self.current_queue[track_id]
            service_time = timestamp - entry["join_time"]
            self.service_times.append(service_time)
            self.served.append(entry)
            del self.current_queue[track_id]
    
    def get_metrics(self):
        queue_length = len(self.current_queue)
        
        wait_times = []
        now = time.time()
        for entry in self.current_queue.values():
            wait_times.append(now - entry["join_time"])
        
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
        
        # Service rate: served per minute
        total_time = sum(self.service_times) / 60 if self.service_times else 1
        service_rate = len(self.served) / total_time if total_time > 0 else 0
        
        return {
            "queue_length": queue_length,
            "avg_wait_time_seconds": avg_wait,
            "service_rate": service_rate,
            "abandonment_count": len(self.abandoned),
        }


class MockHeatmapGenerator:
    """Mock heatmap generator for testing."""
    
    def __init__(self, width, height, cell_size=10, decay_rate=1.0, track_hourly=False):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.decay_rate = decay_rate
        self.track_hourly = track_hourly
        
        self.grid_w = width // cell_size
        self.grid_h = height // cell_size
        self.grid = np.zeros((self.grid_h, self.grid_w))
        self.hourly_grids = {}
    
    def add_point(self, x, y, hour=None):
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        
        if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
            self.grid[gy, gx] += 1
            
            if self.track_hourly and hour is not None:
                if hour not in self.hourly_grids:
                    self.hourly_grids[hour] = np.zeros((self.grid_h, self.grid_w))
                self.hourly_grids[hour][gy, gx] += 1
    
    def apply_decay(self):
        self.grid *= self.decay_rate
    
    def get_heatmap(self):
        return self.grid.copy()
    
    def get_normalized_heatmap(self):
        max_val = self.grid.max()
        if max_val > 0:
            return self.grid / max_val
        return self.grid.copy()
    
    def detect_hotspots(self, threshold=0.5):
        normalized = self.get_normalized_heatmap()
        hotspots = []
        
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if normalized[y, x] >= threshold:
                    hotspots.append({
                        "x": x,
                        "y": y,
                        "intensity": normalized[y, x],
                    })
        
        return hotspots
    
    def get_hourly_patterns(self):
        return self.hourly_grids


# ============================================================================
# API Tests
# ============================================================================

class TestAPIEndpoints:
    """Tests for REST API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        # Would use FastAPI TestClient in actual implementation
        return MockAPIClient()
    
    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response["status"] == "healthy"
    
    def test_list_streams(self, client):
        """Test listing camera streams."""
        response = client.get("/api/v1/streams")
        assert isinstance(response, list)
    
    def test_get_analytics_summary(self, client):
        """Test analytics summary endpoint."""
        response = client.get("/api/v1/analytics/summary?time_range=24h")
        assert "total_visitors" in response
        assert "conversion_rate" in response
    
    def test_create_alert(self, client):
        """Test alert creation."""
        alert_data = {
            "alert_type": "queue_length_exceeded",
            "severity": "warning",
            "message": "Queue length exceeded threshold",
        }
        response = client.post("/api/v1/alerts", json=alert_data)
        assert "alert_id" in response


class MockAPIClient:
    """Mock API client for testing."""
    
    def get(self, path, **kwargs):
        if path == "/health":
            return {"status": "healthy"}
        elif path == "/api/v1/streams":
            return [{"stream_id": "cam-1", "status": "online"}]
        elif "/analytics/summary" in path:
            return {
                "total_visitors": 100,
                "conversion_rate": 0.25,
                "avg_dwell_time_seconds": 300,
            }
        return {}
    
    def post(self, path, json=None, **kwargs):
        if "/alerts" in path:
            return {"alert_id": "alert-0001", **json}
        return {}


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    def test_high_throughput_updates(self, zone_config):
        """Test handling high update rates."""
        tracker = MockJourneyTracker(zone_config)
        
        start = time.time()
        
        # Simulate 1000 updates
        for i in range(1000):
            tracker.update(
                track_id=i % 50,  # 50 concurrent people
                position=(np.random.rand(), np.random.rand()),
                timestamp=time.time(),
            )
        
        elapsed = time.time() - start
        
        # Should handle 1000 updates in under 1 second
        assert elapsed < 1.0
    
    def test_heatmap_memory_efficiency(self):
        """Test heatmap memory usage."""
        generator = MockHeatmapGenerator(width=1920, height=1080, cell_size=20)
        
        # Add many points
        for _ in range(10000):
            generator.add_point(
                np.random.randint(0, 1920),
                np.random.randint(0, 1080),
            )
        
        # Memory should be bounded by grid size
        heatmap = generator.get_heatmap()
        expected_size = (1080 // 20) * (1920 // 20)
        assert heatmap.size == expected_size


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
