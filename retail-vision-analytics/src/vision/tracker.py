"""
Multi-Object Tracking Module for Retail Vision Analytics.

Provides robust object tracking across video frames using multiple algorithms:
- ByteTrack: High-performance association-based tracker
- NvDCF: NVIDIA Deep Correlation Filter (DeepStream)
- SORT: Simple Online Realtime Tracking
- Deep SORT: SORT with deep appearance features

Optimized for retail scenarios with:
- Occlusion handling for crowded stores
- Re-identification for cross-camera tracking
- Track lifecycle management
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = 0   # New track, not yet confirmed
    CONFIRMED = 1   # Active confirmed track
    LOST = 2        # Temporarily lost (occlusion)
    DELETED = 3     # Track removed


@dataclass
class TrackInfo:
    """Metadata for a tracked object."""
    
    track_id: int
    class_id: int
    class_name: str
    state: TrackState
    age: int  # Frames since creation
    hits: int  # Successful detection associations
    time_since_update: int  # Frames since last detection match
    
    # Motion state
    bbox: Tuple[int, int, int, int]  # Current x1, y1, x2, y2
    velocity: Tuple[float, float] = (0.0, 0.0)  # vx, vy in pixels/frame
    
    # History
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    
    # Re-identification features
    appearance_features: Optional[np.ndarray] = None
    
    # Retail-specific attributes
    zone_history: List[str] = field(default_factory=list)
    dwell_times: Dict[str, float] = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get current bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def is_active(self) -> bool:
        """Check if track is active."""
        return self.state in [TrackState.TENTATIVE, TrackState.CONFIRMED]
    
    @property
    def total_distance(self) -> float:
        """Calculate total distance traveled."""
        if len(self.trajectory) < 2:
            return 0.0
        
        distance = 0.0
        for i in range(1, len(self.trajectory)):
            x1, y1 = self.trajectory[i - 1]
            x2, y2 = self.trajectory[i]
            distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        return distance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "state": self.state.name,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "bbox": self.bbox,
            "center": self.center,
            "velocity": self.velocity,
            "trajectory_length": len(self.trajectory),
            "total_distance": round(self.total_distance, 2),
            "zone_history": self.zone_history[-10:],  # Last 10 zones
            "dwell_times": self.dwell_times
        }


class KalmanFilter:
    """
    Kalman Filter for bounding box tracking.
    
    State vector: [x, y, s, r, vx, vy, vs]
    - (x, y): Bounding box center
    - s: Scale (area)
    - r: Aspect ratio (constant)
    - (vx, vy, vs): Velocities
    """
    
    def __init__(self, bbox: Tuple[int, int, int, int]):
        """Initialize filter with initial bounding box."""
        # Convert bbox to [cx, cy, s, r]
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s = w * h  # Scale (area)
        r = w / h if h > 0 else 1.0  # Aspect ratio
        
        # State: [cx, cy, s, r, vx, vy, vs]
        self.x = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrices
        self.P = np.eye(7, dtype=np.float32) * 10.0  # State covariance
        self.Q = np.eye(7, dtype=np.float32) * 0.01  # Process noise
        self.R = np.eye(4, dtype=np.float32) * 1.0   # Measurement noise
        
        # Increase uncertainty for unobserved velocities
        self.P[4:, 4:] *= 1000.0
    
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next state and return predicted bbox."""
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self._state_to_bbox()
    
    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Update state with new measurement."""
        # Convert measurement
        z = self._bbox_to_measurement(bbox)
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        
        # Covariance update
        I = np.eye(7, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        return self._state_to_bbox()
    
    def _bbox_to_measurement(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Convert bbox to measurement vector."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s = w * h
        r = w / h if h > 0 else 1.0
        return np.array([cx, cy, s, r], dtype=np.float32)
    
    def _state_to_bbox(self) -> Tuple[int, int, int, int]:
        """Convert state vector to bbox."""
        cx, cy, s, r = self.x[:4]
        
        # Ensure positive values
        s = max(s, 1.0)
        r = max(r, 0.1)
        
        w = np.sqrt(s * r)
        h = s / w if w > 0 else np.sqrt(s)
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        return (x1, y1, x2, y2)
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return (float(self.x[4]), float(self.x[5]))


class BaseTracker(ABC):
    """Abstract base class for object trackers."""
    
    @abstractmethod
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> List[TrackInfo]:
        """Update tracker with new detections."""
        pass
    
    @abstractmethod
    def get_active_tracks(self) -> List[TrackInfo]:
        """Get all active tracks."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass


class ByteTracker(BaseTracker):
    """
    ByteTrack: High-performance multi-object tracker.
    
    Key features:
    - Associates both high and low confidence detections
    - Robust to occlusions
    - Linear assignment-based matching
    
    Reference: https://arxiv.org/abs/2110.06864
    """
    
    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_hits: int = 3,
        frame_rate: int = 30
    ):
        """
        Initialize ByteTracker.
        
        Args:
            track_high_thresh: High confidence threshold for first association
            track_low_thresh: Low confidence threshold for second association
            new_track_thresh: Minimum confidence for new tracks
            match_thresh: IoU threshold for matching
            track_buffer: Frames to keep lost tracks
            min_hits: Minimum hits to confirm track
            frame_rate: Video frame rate (for time calculations)
        """
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.frame_rate = frame_rate
        
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1
        self._frame_count = 0
        
        logger.info(
            f"ByteTracker initialized: high={track_high_thresh}, "
            f"low={track_low_thresh}, buffer={track_buffer}"
        )
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> List[TrackInfo]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'class_id'
            frame: Optional frame for appearance features
        
        Returns:
            List of active TrackInfo objects
        """
        self._frame_count += 1
        
        if not detections:
            # Age all tracks
            self._age_tracks()
            return self.get_active_tracks()
        
        # Split detections by confidence
        high_dets = [d for d in detections if d["confidence"] >= self.track_high_thresh]
        low_dets = [d for d in detections if self.track_low_thresh <= d["confidence"] < self.track_high_thresh]
        
        # Get currently tracked objects
        confirmed_tracks = [
            t for t in self._tracks.values()
            if t["state"] == TrackState.CONFIRMED
        ]
        unconfirmed_tracks = [
            t for t in self._tracks.values()
            if t["state"] == TrackState.TENTATIVE
        ]
        
        # First association: high confidence detections with confirmed tracks
        matched_c, unmatched_tracks_c, unmatched_dets_high = self._associate(
            confirmed_tracks, high_dets, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_c:
            track = confirmed_tracks[track_idx]
            det = high_dets[det_idx]
            self._update_track(track["id"], det)
        
        # Second association: low confidence with remaining tracks
        remaining_tracks = [confirmed_tracks[i] for i in unmatched_tracks_c]
        matched_l, unmatched_tracks_l, _ = self._associate(
            remaining_tracks, low_dets, self.match_thresh
        )
        
        # Update matched tracks from second association
        for track_idx, det_idx in matched_l:
            track = remaining_tracks[track_idx]
            det = low_dets[det_idx]
            self._update_track(track["id"], det)
        
        # Third association: unconfirmed tracks with unmatched high detections
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets_high]
        matched_u, unmatched_tracks_u, unmatched_dets_final = self._associate(
            unconfirmed_tracks, unmatched_high_dets, self.match_thresh
        )
        
        # Update matched unconfirmed tracks
        for track_idx, det_idx in matched_u:
            track = unconfirmed_tracks[track_idx]
            det = unmatched_high_dets[det_idx]
            self._update_track(track["id"], det)
        
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks_l:
            track = remaining_tracks[track_idx]
            self._mark_lost(track["id"])
        
        for track_idx in unmatched_tracks_u:
            track = unconfirmed_tracks[track_idx]
            self._mark_lost(track["id"])
        
        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets_final:
            det = unmatched_high_dets[det_idx]
            if det["confidence"] >= self.new_track_thresh:
                self._create_track(det)
        
        # Clean up deleted tracks
        self._cleanup_tracks()
        
        return self.get_active_tracks()
    
    def _associate(
        self,
        tracks: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU and linear assignment.
        
        Returns:
            - matched pairs (track_idx, det_idx)
            - unmatched track indices
            - unmatched detection indices
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IoU cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                iou = self._compute_iou(track["bbox"], det["bbox"])
                cost_matrix[t_idx, d_idx] = 1.0 - iou  # Convert to cost
        
        # Hungarian algorithm
        if not SCIPY_AVAILABLE:
            # Fallback to greedy matching
            return self._greedy_match(cost_matrix, thresh)
        
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter by threshold
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] < (1.0 - thresh):  # IoU > thresh
                matched.append((t_idx, d_idx))
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _greedy_match(
        self,
        cost_matrix: np.ndarray,
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy matching fallback when scipy unavailable."""
        matched = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_dets = list(range(cost_matrix.shape[1]))
        
        while unmatched_tracks and unmatched_dets:
            # Find minimum cost
            min_cost = float("inf")
            min_t, min_d = -1, -1
            
            for t_idx in unmatched_tracks:
                for d_idx in unmatched_dets:
                    if cost_matrix[t_idx, d_idx] < min_cost:
                        min_cost = cost_matrix[t_idx, d_idx]
                        min_t, min_d = t_idx, d_idx
            
            if min_cost < (1.0 - thresh):
                matched.append((min_t, min_d))
                unmatched_tracks.remove(min_t)
                unmatched_dets.remove(min_d)
            else:
                break
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Compute Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _create_track(self, detection: Dict[str, Any]) -> int:
        """Create new track from detection."""
        track_id = self._next_id
        self._next_id += 1
        
        bbox = detection["bbox"]
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        self._tracks[track_id] = {
            "id": track_id,
            "class_id": detection.get("class_id", 0),
            "class_name": detection.get("class_name", "unknown"),
            "state": TrackState.TENTATIVE,
            "bbox": bbox,
            "kalman": KalmanFilter(bbox),
            "age": 1,
            "hits": 1,
            "time_since_update": 0,
            "trajectory": [center],
            "confidence_history": [detection["confidence"]],
            "zone_history": [],
            "dwell_times": {}
        }
        
        logger.debug(f"Created track {track_id}")
        return track_id
    
    def _update_track(self, track_id: int, detection: Dict[str, Any]) -> None:
        """Update existing track with new detection."""
        if track_id not in self._tracks:
            return
        
        track = self._tracks[track_id]
        
        # Update Kalman filter
        new_bbox = track["kalman"].update(detection["bbox"])
        
        # Update track state
        track["bbox"] = new_bbox
        track["hits"] += 1
        track["time_since_update"] = 0
        track["age"] += 1
        
        # Update trajectory
        center = ((new_bbox[0] + new_bbox[2]) // 2, (new_bbox[1] + new_bbox[3]) // 2)
        track["trajectory"].append(center)
        
        # Limit trajectory length
        if len(track["trajectory"]) > 1000:
            track["trajectory"] = track["trajectory"][-500:]
        
        track["confidence_history"].append(detection["confidence"])
        if len(track["confidence_history"]) > 100:
            track["confidence_history"] = track["confidence_history"][-50:]
        
        # Confirm track if enough hits
        if track["state"] == TrackState.TENTATIVE and track["hits"] >= self.min_hits:
            track["state"] = TrackState.CONFIRMED
            logger.debug(f"Confirmed track {track_id}")
    
    def _mark_lost(self, track_id: int) -> None:
        """Mark track as lost."""
        if track_id not in self._tracks:
            return
        
        track = self._tracks[track_id]
        track["time_since_update"] += 1
        track["age"] += 1
        
        # Predict position with Kalman
        predicted_bbox = track["kalman"].predict()
        track["bbox"] = predicted_bbox
        
        if track["time_since_update"] > self.track_buffer:
            track["state"] = TrackState.DELETED
    
    def _age_tracks(self) -> None:
        """Age all tracks without detections."""
        for track_id in list(self._tracks.keys()):
            self._mark_lost(track_id)
    
    def _cleanup_tracks(self) -> None:
        """Remove deleted tracks."""
        to_remove = [
            tid for tid, t in self._tracks.items()
            if t["state"] == TrackState.DELETED
        ]
        for tid in to_remove:
            del self._tracks[tid]
            logger.debug(f"Deleted track {tid}")
    
    def get_active_tracks(self) -> List[TrackInfo]:
        """Get all active tracks as TrackInfo objects."""
        active = []
        
        for track in self._tracks.values():
            if track["state"] in [TrackState.TENTATIVE, TrackState.CONFIRMED]:
                info = TrackInfo(
                    track_id=track["id"],
                    class_id=track["class_id"],
                    class_name=track["class_name"],
                    state=track["state"],
                    age=track["age"],
                    hits=track["hits"],
                    time_since_update=track["time_since_update"],
                    bbox=track["bbox"],
                    velocity=track["kalman"].velocity,
                    trajectory=track["trajectory"].copy(),
                    confidence_history=track["confidence_history"].copy(),
                    zone_history=track["zone_history"].copy(),
                    dwell_times=track["dwell_times"].copy()
                )
                active.append(info)
        
        return active
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackInfo]:
        """Get specific track by ID."""
        if track_id not in self._tracks:
            return None
        
        track = self._tracks[track_id]
        return TrackInfo(
            track_id=track["id"],
            class_id=track["class_id"],
            class_name=track["class_name"],
            state=track["state"],
            age=track["age"],
            hits=track["hits"],
            time_since_update=track["time_since_update"],
            bbox=track["bbox"],
            velocity=track["kalman"].velocity,
            trajectory=track["trajectory"].copy(),
            confidence_history=track["confidence_history"].copy(),
            zone_history=track["zone_history"].copy(),
            dwell_times=track["dwell_times"].copy()
        )
    
    def update_zone(self, track_id: int, zone_name: str) -> None:
        """Update track's current zone."""
        if track_id in self._tracks:
            self._tracks[track_id]["zone_history"].append(zone_name)
    
    def add_dwell_time(self, track_id: int, zone_name: str, time_seconds: float) -> None:
        """Add dwell time for a zone."""
        if track_id in self._tracks:
            dwell = self._tracks[track_id]["dwell_times"]
            dwell[zone_name] = dwell.get(zone_name, 0.0) + time_seconds
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.info("Tracker reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        active_tracks = [t for t in self._tracks.values() if t["state"] != TrackState.DELETED]
        confirmed = [t for t in active_tracks if t["state"] == TrackState.CONFIRMED]
        
        return {
            "total_tracks_created": self._next_id - 1,
            "active_tracks": len(active_tracks),
            "confirmed_tracks": len(confirmed),
            "tentative_tracks": len(active_tracks) - len(confirmed),
            "frames_processed": self._frame_count
        }


class SORTTracker(BaseTracker):
    """
    Simple Online Realtime Tracking (SORT).
    
    A simpler tracker using only Kalman filter and IoU matching.
    Faster but less robust to occlusions than ByteTrack.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """Initialize SORT tracker."""
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self._tracks: Dict[int, Dict] = {}
        self._next_id = 1
        self._frame_count = 0
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> List[TrackInfo]:
        """Update with new detections."""
        self._frame_count += 1
        
        # Predict all tracks
        for track in self._tracks.values():
            track["bbox"] = track["kalman"].predict()
            track["time_since_update"] += 1
        
        if detections:
            # Match detections to tracks
            matched, unmatched_tracks, unmatched_dets = self._match(detections)
            
            # Update matched tracks
            for t_idx, d_idx in matched:
                track_id = list(self._tracks.keys())[t_idx]
                self._update_track(track_id, detections[d_idx])
            
            # Create new tracks
            for d_idx in unmatched_dets:
                self._create_track(detections[d_idx])
        
        # Remove old tracks
        self._remove_old_tracks()
        
        return self.get_active_tracks()
    
    def _match(
        self,
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to existing tracks."""
        if not self._tracks:
            return [], [], list(range(len(detections)))
        
        tracks_list = list(self._tracks.values())
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(tracks_list), len(detections)))
        for t, track in enumerate(tracks_list):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._iou(track["bbox"], det["bbox"])
        
        # Hungarian matching
        if SCIPY_AVAILABLE and iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            matched = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched.append((r, c))
            
            matched_tracks = {m[0] for m in matched}
            matched_dets = {m[1] for m in matched}
            
            unmatched_tracks = [i for i in range(len(tracks_list)) if i not in matched_tracks]
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
            
            return matched, unmatched_tracks, unmatched_dets
        
        return [], list(range(len(tracks_list))), list(range(len(detections)))
    
    def _iou(self, bbox1, bbox2) -> float:
        """Compute IoU between two boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    def _create_track(self, det: Dict) -> None:
        """Create new track."""
        track_id = self._next_id
        self._next_id += 1
        
        self._tracks[track_id] = {
            "id": track_id,
            "bbox": det["bbox"],
            "kalman": KalmanFilter(det["bbox"]),
            "class_id": det.get("class_id", 0),
            "class_name": det.get("class_name", "unknown"),
            "hits": 1,
            "age": 1,
            "time_since_update": 0,
            "trajectory": []
        }
    
    def _update_track(self, track_id: int, det: Dict) -> None:
        """Update track with detection."""
        track = self._tracks[track_id]
        track["bbox"] = track["kalman"].update(det["bbox"])
        track["hits"] += 1
        track["time_since_update"] = 0
    
    def _remove_old_tracks(self) -> None:
        """Remove tracks exceeding max age."""
        to_remove = [
            tid for tid, t in self._tracks.items()
            if t["time_since_update"] > self.max_age
        ]
        for tid in to_remove:
            del self._tracks[tid]
    
    def get_active_tracks(self) -> List[TrackInfo]:
        """Get active tracks."""
        result = []
        for track in self._tracks.values():
            if track["hits"] >= self.min_hits or track["time_since_update"] == 0:
                state = TrackState.CONFIRMED if track["hits"] >= self.min_hits else TrackState.TENTATIVE
                result.append(TrackInfo(
                    track_id=track["id"],
                    class_id=track["class_id"],
                    class_name=track["class_name"],
                    state=state,
                    age=track["age"],
                    hits=track["hits"],
                    time_since_update=track["time_since_update"],
                    bbox=track["bbox"]
                ))
        return result
    
    def reset(self) -> None:
        """Reset tracker."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0


def create_tracker(
    algorithm: str = "bytetrack",
    **kwargs
) -> BaseTracker:
    """
    Factory function to create a tracker.
    
    Args:
        algorithm: Tracker algorithm (bytetrack, sort)
        **kwargs: Tracker-specific arguments
    
    Returns:
        Configured tracker instance
    """
    algorithm = algorithm.lower()
    
    if algorithm == "bytetrack":
        return ByteTracker(**kwargs)
    elif algorithm == "sort":
        return SORTTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker algorithm: {algorithm}")


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Create tracker
    tracker = ByteTracker(
        track_high_thresh=0.5,
        track_buffer=30
    )
    
    # Simulate detections
    for frame_id in range(10):
        detections = [
            {
                "bbox": (100 + frame_id * 5, 100, 200 + frame_id * 5, 250),
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "person"
            }
        ]
        
        tracks = tracker.update(detections)
        
        print(f"Frame {frame_id}: {len(tracks)} active tracks")
        for t in tracks:
            print(f"  Track {t.track_id}: {t.state.name}, hits={t.hits}")
    
    print(f"\nStatistics: {tracker.get_statistics()}")
