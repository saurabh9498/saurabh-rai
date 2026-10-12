#!/usr/bin/env python3
"""
Real-Time Stream Processing Demo.

Process video streams with detection, tracking, and real-time visualization.
Supports RTSP cameras, video files, and webcams.

Usage:
    python examples/stream_demo.py --source rtsp://192.168.1.100:554/stream
    python examples/stream_demo.py --source store_footage.mp4
    python examples/stream_demo.py --source 0  # Webcam
    python examples/stream_demo.py --sources cam1.mp4 cam2.mp4  # Multi-stream
"""

import argparse
import sys
import time
import threading
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Detection:
    """Detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    """Processing result for a frame."""
    frame_id: int
    timestamp: float
    detections: List[Detection]
    inference_time_ms: float
    tracking_time_ms: float


# Class configuration
CLASS_NAMES = ["person", "shopping_cart", "basket", "product", "shelf", "price_tag", "employee"]
CLASS_COLORS = {
    "person": (0, 255, 0),
    "shopping_cart": (255, 165, 0),
    "basket": (255, 0, 255),
    "product": (0, 255, 255),
    "shelf": (128, 128, 128),
    "price_tag": (0, 0, 255),
    "employee": (255, 0, 0),
}


class SimpleTracker:
    """
    Simple centroid-based tracker for demonstration.
    
    In production, use ByteTrack or DeepSORT from src/vision/tracker.py
    """
    
    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: Dict[int, np.ndarray] = {}  # track_id -> centroid
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracker with new detections."""
        if not detections:
            # Mark all existing objects as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)
            return detections
        
        # Get centroids of new detections
        centroids = np.array([
            [d.bbox[0] + d.bbox[2] // 2, d.bbox[1] + d.bbox[3] // 2]
            for d in detections
        ])
        
        if not self.objects:
            # Register all new detections
            for i, det in enumerate(detections):
                track_id = self._register(centroids[i])
                det.track_id = track_id
            return detections
        
        # Match existing objects to new detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        
        # Compute distance matrix
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis] - centroids[np.newaxis, :],
            axis=2
        )
        
        # Greedy matching
        used_rows = set()
        used_cols = set()
        
        for _ in range(min(len(object_ids), len(detections))):
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            row, col = min_idx
            
            if distances[row, col] > 100:  # Max distance threshold
                break
            
            if row in used_rows or col in used_cols:
                distances[row, col] = float("inf")
                continue
            
            track_id = object_ids[row]
            self.objects[track_id] = centroids[col]
            self.disappeared[track_id] = 0
            detections[col].track_id = track_id
            
            used_rows.add(row)
            used_cols.add(col)
            distances[row, :] = float("inf")
            distances[:, col] = float("inf")
        
        # Handle unmatched existing objects
        for row in range(len(object_ids)):
            if row not in used_rows:
                track_id = object_ids[row]
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)
        
        # Register new detections
        for col in range(len(detections)):
            if col not in used_cols:
                track_id = self._register(centroids[col])
                detections[col].track_id = track_id
        
        return detections
    
    def _register(self, centroid: np.ndarray) -> int:
        """Register new object."""
        track_id = self.next_id
        self.objects[track_id] = centroid
        self.disappeared[track_id] = 0
        self.next_id += 1
        return track_id
    
    def _deregister(self, track_id: int):
        """Deregister object."""
        del self.objects[track_id]
        del self.disappeared[track_id]


class StreamProcessor:
    """Process video stream with detection and tracking."""
    
    def __init__(
        self,
        source: str,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        display: bool = True,
    ):
        self.source = source
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.display = display
        
        self.tracker = SimpleTracker()
        self.frame_count = 0
        self.fps = 0.0
        self.running = False
        
        # Statistics
        self.total_detections = 0
        self.total_tracks = 0
        self.avg_inference_time = 0.0
    
    def start(self):
        """Start processing stream."""
        import cv2
        
        # Open video source
        if self.source.isdigit():
            source = int(self.source)
        else:
            source = self.source
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        print(f"Stream opened: {width}x{height} @ {fps:.1f} FPS")
        print("Press 'q' to quit, 's' to save screenshot")
        
        self.running = True
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                if isinstance(source, str):
                    # Video file ended, loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Process frame
            result = self._process_frame(frame)
            
            # Update statistics
            self.frame_count += 1
            fps_frame_count += 1
            self.total_detections += len(result.detections)
            
            # Calculate FPS
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                self.fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Draw results
            if self.display:
                output_frame = self._draw_results(frame, result)
                cv2.imshow(f"Stream: {self.source}", output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                elif key == ord("s"):
                    filename = f"screenshot_{self.frame_count}.jpg"
                    cv2.imwrite(filename, output_frame)
                    print(f"Saved: {filename}")
        
        cap.release()
        if self.display:
            cv2.destroyAllWindows()
        
        self._print_summary()
    
    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process single frame."""
        timestamp = time.time()
        
        # Detection
        det_start = time.time()
        detections = self._run_detection(frame)
        inference_time = (time.time() - det_start) * 1000
        
        # Tracking
        track_start = time.time()
        detections = self.tracker.update(detections)
        tracking_time = (time.time() - track_start) * 1000
        
        # Update average inference time
        alpha = 0.1
        self.avg_inference_time = (
            alpha * inference_time + (1 - alpha) * self.avg_inference_time
        )
        
        return FrameResult(
            frame_id=self.frame_count,
            timestamp=timestamp,
            detections=detections,
            inference_time_ms=inference_time,
            tracking_time_ms=tracking_time,
        )
    
    def _run_detection(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on frame (mock for demo)."""
        h, w = frame.shape[:2]
        detections = []
        
        # Mock detection - in production use TensorRT
        # Simulate 1-4 person detections
        num_people = np.random.randint(1, 5)
        for _ in range(num_people):
            x = np.random.randint(50, w - 100)
            y = np.random.randint(50, h - 200)
            det_w = np.random.randint(60, 120)
            det_h = np.random.randint(150, 250)
            
            detections.append(Detection(
                class_id=0,
                class_name="person",
                confidence=np.random.uniform(0.7, 0.98),
                bbox=(x, y, det_w, det_h),
            ))
        
        # Simulate 0-1 cart detections
        if np.random.random() > 0.5:
            x = np.random.randint(50, w - 80)
            y = np.random.randint(h // 2, h - 100)
            det_w = np.random.randint(70, 120)
            det_h = np.random.randint(80, 130)
            
            detections.append(Detection(
                class_id=1,
                class_name="shopping_cart",
                confidence=np.random.uniform(0.6, 0.95),
                bbox=(x, y, det_w, det_h),
            ))
        
        return [d for d in detections if d.confidence >= self.conf_threshold]
    
    def _draw_results(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw detection results on frame."""
        import cv2
        
        output = frame.copy()
        
        # Draw detections
        for det in result.detections:
            x, y, w, h = det.bbox
            color = CLASS_COLORS.get(det.class_name, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with track ID
            if det.track_id is not None:
                label = f"#{det.track_id} {det.class_name}: {det.confidence:.2f}"
            else:
                label = f"{det.class_name}: {det.confidence:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                output,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1,
            )
            cv2.putText(
                output, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
        
        # Draw stats overlay
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Detections: {len(result.detections)}",
            f"Inference: {result.inference_time_ms:.1f}ms",
            f"Active Tracks: {len(self.tracker.objects)}",
        ]
        
        y_offset = 30
        for stat in stats:
            cv2.putText(
                output, stat, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            y_offset += 25
        
        return output
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 50)
        print("Processing Summary")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {self.total_detections}")
        print(f"Average FPS: {self.fps:.1f}")
        print(f"Average inference time: {self.avg_inference_time:.1f}ms")
        print(f"Total unique tracks: {self.tracker.next_id}")


class MultiStreamProcessor:
    """Process multiple video streams concurrently."""
    
    def __init__(self, sources: List[str], **kwargs):
        self.sources = sources
        self.kwargs = kwargs
        self.processors: List[StreamProcessor] = []
        self.threads: List[threading.Thread] = []
    
    def start(self):
        """Start all stream processors."""
        print(f"Starting {len(self.sources)} stream processors...")
        
        for source in self.sources:
            processor = StreamProcessor(source, **self.kwargs)
            self.processors.append(processor)
            
            thread = threading.Thread(target=processor.start)
            thread.daemon = True
            self.threads.append(thread)
            thread.start()
        
        # Wait for all threads
        try:
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nStopping all streams...")
            for processor in self.processors:
                processor.running = False


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Stream Processing Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="Video source (RTSP URL, file path, or camera index)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        help="Multiple video sources for multi-stream processing",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="data/models/yolov8n_retail_fp16.engine",
        help="Model path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window",
    )
    
    args = parser.parse_args()
    
    if args.sources:
        # Multi-stream mode
        processor = MultiStreamProcessor(
            args.sources,
            model_path=args.model,
            conf_threshold=args.conf,
            display=not args.no_display,
        )
        processor.start()
    elif args.source:
        # Single stream mode
        processor = StreamProcessor(
            args.source,
            model_path=args.model,
            conf_threshold=args.conf,
            display=not args.no_display,
        )
        processor.start()
    else:
        print("Error: Please provide --source or --sources")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
