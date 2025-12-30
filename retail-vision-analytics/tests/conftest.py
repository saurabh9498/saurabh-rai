"""
Pytest Configuration and Shared Fixtures.

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List


# =============================================================================
# Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# =============================================================================
# Frame Fixtures
# =============================================================================

@pytest.fixture
def sample_frame():
    """Create a sample video frame (1080p)."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_720p():
    """Create a sample video frame (720p)."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_batch():
    """Create a batch of sample frames."""
    return np.random.randint(0, 255, (8, 640, 640, 3), dtype=np.uint8)


# =============================================================================
# Detection Fixtures
# =============================================================================

@pytest.fixture
def sample_detection():
    """Create a single sample detection."""
    return {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.92,
        "bbox": {
            "x": 100,
            "y": 200,
            "width": 80,
            "height": 180,
        },
        "track_id": 1,
        "timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def sample_detections():
    """Create a list of sample detections."""
    return [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.92,
            "bbox": {"x": 100, "y": 200, "width": 80, "height": 180},
            "track_id": 1,
        },
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.88,
            "bbox": {"x": 400, "y": 300, "width": 75, "height": 170},
            "track_id": 2,
        },
        {
            "class_id": 1,
            "class_name": "shopping_cart",
            "confidence": 0.95,
            "bbox": {"x": 500, "y": 400, "width": 100, "height": 80},
            "track_id": 101,
        },
        {
            "class_id": 2,
            "class_name": "basket",
            "confidence": 0.78,
            "bbox": {"x": 250, "y": 350, "width": 60, "height": 50},
            "track_id": 102,
        },
    ]


# =============================================================================
# Zone Configuration Fixtures
# =============================================================================

@pytest.fixture
def zone_config():
    """Create sample zone configuration."""
    return [
        {
            "zone_id": "entrance",
            "zone_name": "Store Entrance",
            "zone_type": "entrance",
            "polygon": [(0.0, 0.8), (0.2, 0.8), (0.2, 1.0), (0.0, 1.0)],
            "color": "#00FF00",
        },
        {
            "zone_id": "aisle-1",
            "zone_name": "Aisle 1 - Produce",
            "zone_type": "aisle",
            "polygon": [(0.0, 0.3), (0.3, 0.3), (0.3, 0.7), (0.0, 0.7)],
            "color": "#0000FF",
        },
        {
            "zone_id": "aisle-2",
            "zone_name": "Aisle 2 - Dairy",
            "zone_type": "aisle",
            "polygon": [(0.35, 0.3), (0.65, 0.3), (0.65, 0.7), (0.35, 0.7)],
            "color": "#FF00FF",
        },
        {
            "zone_id": "checkout",
            "zone_name": "Checkout Area",
            "zone_type": "checkout",
            "polygon": [(0.7, 0.5), (1.0, 0.5), (1.0, 1.0), (0.7, 1.0)],
            "color": "#FFFF00",
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
            "wait_time_warning_seconds": 120,
            "wait_time_critical_seconds": 300,
        },
    }


# =============================================================================
# Stream Configuration Fixtures
# =============================================================================

@pytest.fixture
def stream_config():
    """Create sample stream configuration."""
    return {
        "stream_id": "cam-test-1",
        "name": "Test Camera 1",
        "uri": "rtsp://192.168.1.100:554/stream1",
        "protocol": "rtsp",
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "enabled": True,
        "store_id": "store-001",
        "location": "entrance",
    }


@pytest.fixture
def multi_stream_config():
    """Create configuration for multiple streams."""
    return [
        {
            "stream_id": f"cam-{i}",
            "name": f"Camera {i}",
            "uri": f"rtsp://192.168.1.{10+i}:554/stream1",
            "protocol": "rtsp",
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "enabled": True,
        }
        for i in range(4)
    ]


# =============================================================================
# Journey Fixtures
# =============================================================================

@pytest.fixture
def sample_journey():
    """Create a sample customer journey."""
    start_time = datetime.now() - timedelta(minutes=15)
    return {
        "journey_id": "journey-000001",
        "track_id": 1,
        "stream_id": "cam-entrance-1",
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "duration_seconds": 900,
        "zones_visited": ["entrance", "aisle-1", "aisle-2", "checkout"],
        "zone_dwell_times": {
            "entrance": 30,
            "aisle-1": 300,
            "aisle-2": 240,
            "checkout": 330,
        },
        "entry_point": "entrance",
        "exit_point": "checkout",
        "converted": True,
        "cart_detected": True,
    }


@pytest.fixture
def sample_journeys():
    """Create multiple sample journeys."""
    journeys = []
    for i in range(10):
        start = datetime.now() - timedelta(hours=i)
        duration = np.random.randint(120, 1800)
        converted = np.random.random() > 0.4
        
        journeys.append({
            "journey_id": f"journey-{i+1:06d}",
            "track_id": i + 1,
            "stream_id": "cam-entrance-1",
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(seconds=duration)).isoformat(),
            "duration_seconds": duration,
            "zones_visited": ["entrance", "aisle-1"] + (["checkout"] if converted else []),
            "converted": converted,
        })
    
    return journeys


# =============================================================================
# Temporary File Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary config file."""
    config_path = os.path.join(temp_dir, "test_config.yaml")
    
    config_content = """
app:
  name: "Test App"
  debug: true

pipeline:
  batch_size: 8
  inference_interval: 1
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    yield config_path


# =============================================================================
# Mock Service Fixtures
# =============================================================================

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    class MockRedis:
        def __init__(self):
            self.data = {}
            self.streams = {}
        
        def set(self, key, value):
            self.data[key] = value
        
        def get(self, key):
            return self.data.get(key)
        
        def xadd(self, stream, fields):
            if stream not in self.streams:
                self.streams[stream] = []
            self.streams[stream].append(fields)
            return f"{len(self.streams[stream])}-0"
        
        def xread(self, streams, count=10):
            result = []
            for stream, _ in streams.items():
                if stream in self.streams:
                    result.append((stream, self.streams[stream][:count]))
            return result
    
    return MockRedis()


@pytest.fixture
def mock_db():
    """Create a mock database connection."""
    class MockDB:
        def __init__(self):
            self.tables = {}
        
        def insert(self, table, data):
            if table not in self.tables:
                self.tables[table] = []
            self.tables[table].append(data)
        
        def query(self, table, **filters):
            if table not in self.tables:
                return []
            return self.tables[table]
    
    return MockDB()
