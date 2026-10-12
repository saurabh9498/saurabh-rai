#!/usr/bin/env python3
"""
Sample Data Generator for Retail Vision Analytics.

Generates realistic sample data for testing and demonstration:
- Detection events
- Customer journeys
- Queue metrics
- Heatmap data

Usage:
    python scripts/generate_sample_data.py --output data/sample/ --hours 24
"""

import os
import json
import argparse
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import csv


def generate_detections(
    num_frames: int = 10000,
    streams: List[str] = None,
    fps: int = 30,
) -> List[Dict[str, Any]]:
    """Generate sample detection data."""
    
    if streams is None:
        streams = [
            "cam-entrance-1",
            "cam-aisle-1", 
            "cam-aisle-2",
            "cam-checkout-1",
        ]
    
    classes = [
        ("person", 0.7),
        ("shopping_cart", 0.1),
        ("basket", 0.1),
        ("product", 0.05),
        ("employee", 0.05),
    ]
    
    detections = []
    start_time = datetime.now() - timedelta(hours=24)
    
    track_id_counter = 1
    active_tracks = {}
    
    for frame_num in range(num_frames):
        timestamp = start_time + timedelta(seconds=frame_num / fps)
        stream_id = random.choice(streams)
        
        # Generate detections for this frame
        num_detections = random.randint(0, 15)
        frame_detections = []
        
        for _ in range(num_detections):
            # Pick class
            class_name = random.choices(
                [c[0] for c in classes],
                weights=[c[1] for c in classes]
            )[0]
            
            # Assign or create track
            if random.random() > 0.3 and active_tracks:
                track_id = random.choice(list(active_tracks.keys()))
            else:
                track_id = track_id_counter
                track_id_counter += 1
                active_tracks[track_id] = {
                    "class": class_name,
                    "created": timestamp,
                }
            
            # Random bbox
            x = random.randint(50, 1800)
            y = random.randint(50, 1000)
            w = random.randint(40, 150)
            h = random.randint(80, 250)
            
            detection = {
                "stream_id": stream_id,
                "frame_number": frame_num,
                "timestamp": timestamp.isoformat(),
                "class_name": class_name,
                "confidence": round(random.uniform(0.5, 0.99), 3),
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "track_id": track_id,
            }
            frame_detections.append(detection)
        
        # Randomly remove old tracks
        for tid in list(active_tracks.keys()):
            if random.random() > 0.95:
                del active_tracks[tid]
        
        detections.extend(frame_detections)
    
    return detections


def generate_journeys(
    num_journeys: int = 500,
    hours: int = 24,
) -> List[Dict[str, Any]]:
    """Generate sample customer journey data."""
    
    zones = ["entrance", "aisle-1", "aisle-2", "aisle-3", "produce", "dairy", "checkout"]
    
    journeys = []
    start_time = datetime.now() - timedelta(hours=hours)
    
    for i in range(num_journeys):
        # Random start time within the period
        journey_start = start_time + timedelta(
            seconds=random.randint(0, hours * 3600)
        )
        
        # Duration: 2-30 minutes
        duration = random.randint(120, 1800)
        journey_end = journey_start + timedelta(seconds=duration)
        
        # Path through store
        num_zones = random.randint(2, 6)
        path = ["entrance"]
        
        for _ in range(num_zones - 1):
            next_zone = random.choice([z for z in zones if z not in path[-2:]])
            path.append(next_zone)
        
        # Add checkout for converted customers
        converted = random.random() > 0.4
        if converted and "checkout" not in path:
            path.append("checkout")
        
        # Calculate dwell times
        zone_dwell = {}
        remaining = duration
        for zone in path[:-1]:
            dwell = random.randint(30, min(300, remaining - 30))
            zone_dwell[zone] = dwell
            remaining -= dwell
        zone_dwell[path[-1]] = remaining
        
        journey = {
            "journey_id": f"journey-{i+1:06d}",
            "track_id": i + 1,
            "stream_id": "cam-entrance-1",
            "start_time": journey_start.isoformat(),
            "end_time": journey_end.isoformat(),
            "duration_seconds": duration,
            "zones_visited": path,
            "zone_dwell_times": zone_dwell,
            "entry_point": "entrance",
            "exit_point": path[-1],
            "converted": converted,
            "cart_detected": random.random() > 0.6,
        }
        journeys.append(journey)
    
    return journeys


def generate_queue_metrics(
    hours: int = 24,
    interval_minutes: int = 5,
) -> List[Dict[str, Any]]:
    """Generate sample queue metrics data."""
    
    lanes = ["checkout-1", "checkout-2", "checkout-3"]
    metrics = []
    
    start_time = datetime.now() - timedelta(hours=hours)
    num_intervals = (hours * 60) // interval_minutes
    
    for i in range(num_intervals):
        timestamp = start_time + timedelta(minutes=i * interval_minutes)
        hour = timestamp.hour
        
        # Traffic pattern (busier at certain hours)
        base_traffic = 1.0
        if 11 <= hour <= 13:  # Lunch rush
            base_traffic = 2.0
        elif 17 <= hour <= 19:  # Evening rush
            base_traffic = 2.5
        elif hour < 8 or hour > 21:  # Low traffic
            base_traffic = 0.3
        
        for lane in lanes:
            queue_length = max(0, int(random.gauss(4 * base_traffic, 2)))
            
            # More abandonment when queues are long
            abandon_prob = 0.1 if queue_length > 5 else 0.02
            abandonments = sum(1 for _ in range(queue_length) if random.random() < abandon_prob)
            
            metric = {
                "lane_id": lane,
                "stream_id": f"cam-{lane}",
                "timestamp": timestamp.isoformat(),
                "queue_length": queue_length,
                "avg_wait_time_seconds": round(queue_length * 45 + random.gauss(0, 15), 1),
                "max_wait_time_seconds": round(queue_length * 60 + random.gauss(0, 30), 1),
                "service_rate": round(random.uniform(0.8, 1.5), 2),
                "abandonment_count": abandonments,
                "staffing_recommendation": min(4, max(1, queue_length // 3 + 1)),
            }
            metrics.append(metric)
    
    return metrics


def generate_heatmap(
    width: int = 1920,
    height: int = 1080,
    cell_size: int = 20,
) -> Dict[str, Any]:
    """Generate sample heatmap data."""
    
    grid_w = width // cell_size
    grid_h = height // cell_size
    
    # Create base heatmap
    heatmap = np.zeros((grid_h, grid_w))
    
    # Add hotspots
    hotspots = [
        (grid_w * 0.1, grid_h * 0.9, 0.8),   # Entrance
        (grid_w * 0.2, grid_h * 0.5, 0.6),   # Aisle entrance
        (grid_w * 0.5, grid_h * 0.3, 0.7),   # Popular product
        (grid_w * 0.8, grid_h * 0.7, 0.9),   # Checkout
    ]
    
    for cx, cy, intensity in hotspots:
        for y in range(grid_h):
            for x in range(grid_w):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                heatmap[y, x] += intensity * np.exp(-dist ** 2 / 100)
    
    # Add noise
    heatmap += np.random.rand(grid_h, grid_w) * 0.1
    
    # Normalize
    heatmap = heatmap / heatmap.max()
    
    return {
        "stream_id": "cam-entrance-1",
        "resolution": [grid_w, grid_h],
        "cell_size": cell_size,
        "data": heatmap.tolist(),
        "hotspots": [
            {"x": cx / grid_w, "y": cy / grid_h, "intensity": intensity}
            for cx, cy, intensity in hotspots
        ],
    }


def generate_alerts(
    hours: int = 24,
    num_alerts: int = 50,
) -> List[Dict[str, Any]]:
    """Generate sample alert data."""
    
    alert_types = [
        ("queue_length_exceeded", "warning"),
        ("wait_time_exceeded", "critical"),
        ("person_loitering", "info"),
        ("occupancy_exceeded", "warning"),
        ("stream_offline", "critical"),
    ]
    
    alerts = []
    start_time = datetime.now() - timedelta(hours=hours)
    
    for i in range(num_alerts):
        alert_type, severity = random.choice(alert_types)
        timestamp = start_time + timedelta(
            seconds=random.randint(0, hours * 3600)
        )
        
        # Determine status based on age
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        if age_hours < 1:
            status = "active"
        elif age_hours < 4:
            status = random.choice(["active", "acknowledged"])
        else:
            status = "resolved"
        
        alert = {
            "alert_id": f"alert-{i+1:04d}",
            "alert_type": alert_type,
            "severity": severity,
            "status": status,
            "stream_id": f"cam-{random.choice(['entrance', 'aisle', 'checkout'])}-1",
            "timestamp": timestamp.isoformat(),
            "message": f"Sample {alert_type.replace('_', ' ')} alert",
            "details": {},
        }
        
        if alert_type == "queue_length_exceeded":
            alert["details"] = {"threshold": 8, "actual": random.randint(9, 15)}
        elif alert_type == "wait_time_exceeded":
            alert["details"] = {"threshold_seconds": 300, "actual_seconds": random.randint(301, 600)}
        
        alerts.append(alert)
    
    # Sort by timestamp
    alerts.sort(key=lambda a: a["timestamp"], reverse=True)
    
    return alerts


def save_data(data: Any, filepath: str, format: str = "json"):
    """Save data to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == "json":
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "csv":
        if not data:
            return
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    print(f"Saved {len(data) if isinstance(data, list) else 1} records to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample data")
    parser.add_argument("--output", "-o", default="data/sample/", help="Output directory")
    parser.add_argument("--hours", "-t", type=int, default=24, help="Hours of data")
    parser.add_argument("--format", "-f", default="json", choices=["json", "csv"])
    args = parser.parse_args()
    
    print(f"Generating {args.hours} hours of sample data...")
    print(f"Output directory: {args.output}")
    
    # Generate all data types
    print("\n1. Generating detections...")
    detections = generate_detections(num_frames=args.hours * 30 * 60)
    save_data(detections, f"{args.output}/detections.{args.format}", args.format)
    
    print("\n2. Generating customer journeys...")
    journeys = generate_journeys(num_journeys=args.hours * 20, hours=args.hours)
    save_data(journeys, f"{args.output}/journeys.{args.format}", args.format)
    
    print("\n3. Generating queue metrics...")
    queue_metrics = generate_queue_metrics(hours=args.hours)
    save_data(queue_metrics, f"{args.output}/queue_metrics.{args.format}", args.format)
    
    print("\n4. Generating heatmap...")
    heatmap = generate_heatmap()
    save_data(heatmap, f"{args.output}/heatmap.json", "json")
    
    print("\n5. Generating alerts...")
    alerts = generate_alerts(hours=args.hours)
    save_data(alerts, f"{args.output}/alerts.{args.format}", args.format)
    
    print("\n✓ Sample data generation complete!")
    print(f"\nGenerated files in {args.output}:")
    for f in os.listdir(args.output):
        size = os.path.getsize(os.path.join(args.output, f))
        print(f"  - {f} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
