"""
YOLO-based Object Detection Module for Retail Vision Analytics.

This module provides a robust object detection pipeline using YOLOv8,
optimized for retail environments with support for TensorRT acceleration.

Features:
- Multi-class detection (people, products, carts, baskets)
- TensorRT INT8/FP16 optimization
- Batch inference support
- Confidence filtering and NMS
- Retail-specific class mappings
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

logger = logging.getLogger(__name__)


class RetailClass(Enum):
    """Retail-specific object classes."""
    PERSON = 0
    SHOPPING_CART = 1
    SHOPPING_BASKET = 2
    PRODUCT = 3
    SHELF = 4
    PRICE_TAG = 5
    EMPLOYEE = 6
    SECURITY_PERSONNEL = 7
    CHILD = 8
    WHEELCHAIR = 9


@dataclass
class Detection:
    """Single object detection result."""
    
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get bounding box center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Get bounding box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def aspect_ratio(self) -> float:
        """Get bounding box aspect ratio (width/height)."""
        x1, y1, x2, y2 = self.bbox
        height = y2 - y1
        width = x2 - x1
        return width / height if height > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "center": self.center,
            "area": self.area,
            "track_id": self.track_id,
            "attributes": self.attributes
        }


@dataclass
class DetectionResult:
    """Result container for frame detections."""
    
    frame_id: int
    timestamp: float
    detections: List[Detection]
    inference_time_ms: float
    frame_shape: Tuple[int, int, int]  # height, width, channels
    
    @property
    def num_detections(self) -> int:
        """Get total number of detections."""
        return len(self.detections)
    
    @property
    def persons(self) -> List[Detection]:
        """Get all person detections."""
        return [d for d in self.detections if d.class_name.lower() == "person"]
    
    @property
    def products(self) -> List[Detection]:
        """Get all product detections."""
        return [d for d in self.detections if d.class_name.lower() == "product"]
    
    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Filter detections by minimum confidence threshold."""
        return [d for d in self.detections if d.confidence >= min_confidence]
    
    def filter_by_class(self, class_names: List[str]) -> List[Detection]:
        """Filter detections by class names."""
        names_lower = [n.lower() for n in class_names]
        return [d for d in self.detections if d.class_name.lower() in names_lower]
    
    def filter_by_region(
        self,
        region: Tuple[int, int, int, int]  # x1, y1, x2, y2
    ) -> List[Detection]:
        """Filter detections by spatial region (ROI)."""
        rx1, ry1, rx2, ry2 = region
        filtered = []
        for det in self.detections:
            cx, cy = det.center
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                filtered.append(det)
        return filtered
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "num_detections": self.num_detections,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "frame_shape": self.frame_shape,
            "detections": [d.to_dict() for d in self.detections]
        }


class RetailDetector:
    """
    YOLO-based object detector optimized for retail environments.
    
    Supports multiple backends:
    - PyTorch (default)
    - TensorRT (optimized)
    - ONNX Runtime
    
    Example:
        >>> detector = RetailDetector(
        ...     model_path="yolov8m_retail.pt",
        ...     confidence_threshold=0.5,
        ...     device="cuda:0"
        ... )
        >>> result = detector.detect(frame)
        >>> print(f"Found {result.num_detections} objects")
    """
    
    # Default retail class mapping (COCO + custom)
    DEFAULT_CLASS_NAMES = {
        0: "person",
        1: "shopping_cart",
        2: "shopping_basket",
        3: "product",
        4: "shelf",
        5: "price_tag",
        6: "employee",
        7: "handbag",
        8: "backpack",
        9: "bottle",
        10: "box",
    }
    
    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cuda:0",
        class_names: Optional[Dict[int, str]] = None,
        input_size: Tuple[int, int] = (640, 640),
        half_precision: bool = True,
        max_detections: int = 300,
        use_tensorrt: bool = False,
        tensorrt_engine_path: Optional[str] = None
    ):
        """
        Initialize the retail object detector.
        
        Args:
            model_path: Path to YOLO model weights (.pt, .onnx, or .engine)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression IoU threshold
            device: Inference device (cuda:0, cuda:1, cpu)
            class_names: Custom class ID to name mapping
            input_size: Model input resolution (width, height)
            half_precision: Use FP16 inference (GPU only)
            max_detections: Maximum detections per frame
            use_tensorrt: Use TensorRT engine if available
            tensorrt_engine_path: Path to pre-built TensorRT engine
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.input_size = input_size
        self.half_precision = half_precision
        self.max_detections = max_detections
        self.use_tensorrt = use_tensorrt
        self.tensorrt_engine_path = tensorrt_engine_path
        
        self._model = None
        self._frame_count = 0
        self._total_inference_time = 0.0
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the detection model."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics package not found. "
                "Install with: pip install ultralytics"
            )
        
        logger.info(f"Loading model from {self.model_path}")
        
        # Determine model format
        suffix = self.model_path.suffix.lower()
        
        if self.use_tensorrt and self.tensorrt_engine_path:
            # Load TensorRT engine
            logger.info("Loading TensorRT engine...")
            self._model = YOLO(self.tensorrt_engine_path)
        elif suffix == ".engine":
            # Direct TensorRT engine
            self._model = YOLO(str(self.model_path))
        elif suffix in [".pt", ".pth"]:
            # PyTorch model
            self._model = YOLO(str(self.model_path))
            
            # Export to TensorRT if requested
            if self.use_tensorrt and TENSORRT_AVAILABLE:
                logger.info("Exporting to TensorRT...")
                self._model.export(
                    format="engine",
                    half=self.half_precision,
                    imgsz=self.input_size[0]
                )
        elif suffix == ".onnx":
            # ONNX model
            self._model = YOLO(str(self.model_path))
        else:
            # Assume pretrained model name
            self._model = YOLO(str(self.model_path))
        
        # Set device
        if "cuda" in self.device:
            self._model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> DetectionResult:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input frame (BGR format, HWC)
            frame_id: Frame identifier
            timestamp: Frame timestamp in seconds
            roi: Optional region of interest (x1, y1, x2, y2)
        
        Returns:
            DetectionResult containing all detections
        """
        import time
        
        # Validate input
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")
        
        original_shape = frame.shape
        
        # Apply ROI if specified
        if roi is not None:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
            roi_offset = (x1, y1)
        else:
            roi_offset = (0, 0)
        
        # Run inference
        start_time = time.perf_counter()
        
        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            max_det=self.max_detections,
            verbose=False,
            device=self.device,
            half=self.half_precision
        )
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Parse detections
        detections = self._parse_results(results[0], roi_offset)
        
        # Update statistics
        self._frame_count += 1
        self._total_inference_time += inference_time
        
        return DetectionResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            inference_time_ms=inference_time,
            frame_shape=original_shape
        )
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        frame_ids: Optional[List[int]] = None,
        timestamps: Optional[List[float]] = None
    ) -> List[DetectionResult]:
        """
        Perform batch detection on multiple frames.
        
        Args:
            frames: List of input frames
            frame_ids: Optional list of frame identifiers
            timestamps: Optional list of timestamps
        
        Returns:
            List of DetectionResult for each frame
        """
        import time
        
        if not frames:
            return []
        
        frame_ids = frame_ids or list(range(len(frames)))
        timestamps = timestamps or [0.0] * len(frames)
        
        # Batch inference
        start_time = time.perf_counter()
        
        results = self._model.predict(
            source=frames,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            max_det=self.max_detections,
            verbose=False,
            device=self.device,
            half=self.half_precision,
            stream=True  # Enable streaming for memory efficiency
        )
        
        # Process results
        detection_results = []
        for idx, result in enumerate(results):
            inference_time = (time.perf_counter() - start_time) * 1000 / len(frames)
            
            detections = self._parse_results(result, (0, 0))
            
            detection_results.append(DetectionResult(
                frame_id=frame_ids[idx],
                timestamp=timestamps[idx],
                detections=detections,
                inference_time_ms=inference_time,
                frame_shape=frames[idx].shape
            ))
        
        return detection_results
    
    def _parse_results(
        self,
        result: Any,
        offset: Tuple[int, int]
    ) -> List[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.cpu().numpy()
        
        for i in range(len(boxes)):
            # Get box coordinates
            x1, y1, x2, y2 = boxes.xyxy[i].astype(int)
            
            # Apply ROI offset
            x1 += offset[0]
            y1 += offset[1]
            x2 += offset[0]
            y2 += offset[1]
            
            # Get class and confidence
            class_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            
            # Get class name
            if hasattr(result, 'names') and class_id in result.names:
                class_name = result.names[class_id]
            elif class_id in self.class_names:
                class_name = self.class_names[class_id]
            else:
                class_name = f"class_{class_id}"
            
            # Create detection object
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2)
            )
            
            detections.append(detection)
        
        return detections
    
    def warmup(self, iterations: int = 10) -> float:
        """
        Warm up the model with dummy inference.
        
        Args:
            iterations: Number of warmup iterations
        
        Returns:
            Average warmup inference time in ms
        """
        import time
        
        logger.info(f"Warming up model with {iterations} iterations...")
        
        dummy_frame = np.random.randint(
            0, 255,
            (self.input_size[1], self.input_size[0], 3),
            dtype=np.uint8
        )
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self._model.predict(
                source=dummy_frame,
                verbose=False,
                device=self.device
            )
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        logger.info(f"Warmup complete. Average inference time: {avg_time:.2f}ms")
        
        return avg_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        avg_time = (
            self._total_inference_time / self._frame_count
            if self._frame_count > 0 else 0.0
        )
        
        return {
            "frames_processed": self._frame_count,
            "total_inference_time_ms": round(self._total_inference_time, 2),
            "average_inference_time_ms": round(avg_time, 2),
            "average_fps": round(1000 / avg_time, 2) if avg_time > 0 else 0.0,
            "model_path": str(self.model_path),
            "device": self.device,
            "input_size": self.input_size,
            "confidence_threshold": self.confidence_threshold
        }
    
    def reset_statistics(self) -> None:
        """Reset detection statistics."""
        self._frame_count = 0
        self._total_inference_time = 0.0
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_labels: bool = True,
        line_thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Draw detection boxes on frame.
        
        Args:
            frame: Input frame (will be modified)
            detections: List of detections to draw
            show_confidence: Display confidence scores
            show_labels: Display class labels
            line_thickness: Bounding box line thickness
            font_scale: Label font scale
        
        Returns:
            Frame with drawn detections
        """
        # Color palette for different classes
        colors = {
            "person": (0, 255, 0),       # Green
            "shopping_cart": (255, 0, 0), # Blue
            "shopping_basket": (255, 128, 0),  # Orange
            "product": (0, 255, 255),     # Yellow
            "employee": (255, 0, 255),    # Magenta
            "default": (128, 128, 128)    # Gray
        }
        
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Get color
            color = colors.get(det.class_name.lower(), colors["default"])
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, line_thickness)
            
            # Draw label
            if show_labels:
                label_parts = [det.class_name]
                
                if show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")
                
                if det.track_id is not None:
                    label_parts.append(f"ID:{det.track_id}")
                
                label = " ".join(label_parts)
                
                # Get label size
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    output,
                    (x1, y1 - label_h - baseline - 5),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    output,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        return output
    
    def export_tensorrt(
        self,
        output_path: Optional[str] = None,
        precision: str = "fp16",
        workspace_size: int = 4,
        batch_size: int = 1
    ) -> str:
        """
        Export model to TensorRT engine.
        
        Args:
            output_path: Output engine file path
            precision: Precision mode (fp32, fp16, int8)
            workspace_size: GPU workspace size in GB
            batch_size: Maximum batch size
        
        Returns:
            Path to exported engine
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        
        half = precision.lower() in ["fp16", "half"]
        int8 = precision.lower() == "int8"
        
        engine_path = self._model.export(
            format="engine",
            half=half,
            int8=int8,
            imgsz=self.input_size[0],
            workspace=workspace_size,
            batch=batch_size
        )
        
        logger.info(f"TensorRT engine exported to: {engine_path}")
        
        return engine_path


class DeepStreamDetector:
    """
    DeepStream-based detector for high-throughput video analytics.
    
    Optimized for multi-stream processing with NVIDIA hardware.
    """
    
    def __init__(
        self,
        config_path: str,
        num_streams: int = 4,
        batch_size: int = 4,
        gpu_id: int = 0
    ):
        """
        Initialize DeepStream detector.
        
        Args:
            config_path: Path to DeepStream config file
            num_streams: Number of parallel streams
            batch_size: Inference batch size
            gpu_id: GPU device ID
        """
        self.config_path = config_path
        self.num_streams = num_streams
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        
        self._pipeline = None
        self._is_running = False
        
        logger.info(
            f"DeepStream detector initialized: "
            f"{num_streams} streams, batch_size={batch_size}"
        )
    
    def start(self) -> None:
        """Start the DeepStream pipeline."""
        # Note: Full DeepStream implementation requires GStreamer bindings
        # This is a skeleton for the pipeline setup
        logger.info("Starting DeepStream pipeline...")
        self._is_running = True
    
    def stop(self) -> None:
        """Stop the DeepStream pipeline."""
        logger.info("Stopping DeepStream pipeline...")
        self._is_running = False
    
    def add_stream(self, uri: str, stream_id: int) -> None:
        """Add a video stream to the pipeline."""
        logger.info(f"Adding stream {stream_id}: {uri}")
    
    def remove_stream(self, stream_id: int) -> None:
        """Remove a video stream from the pipeline."""
        logger.info(f"Removing stream {stream_id}")
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running


# Convenience functions
def create_detector(
    model_type: str = "yolov8m",
    use_tensorrt: bool = False,
    **kwargs
) -> RetailDetector:
    """
    Factory function to create a retail detector.
    
    Args:
        model_type: Model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        use_tensorrt: Use TensorRT optimization
        **kwargs: Additional detector arguments
    
    Returns:
        Configured RetailDetector instance
    """
    model_path = f"{model_type}.pt"
    
    return RetailDetector(
        model_path=model_path,
        use_tensorrt=use_tensorrt,
        **kwargs
    )


if __name__ == "__main__":
    # Demo usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = RetailDetector(
        model_path="yolov8m.pt",
        confidence_threshold=0.5,
        device="cuda:0" if len(sys.argv) > 1 else "cpu"
    )
    
    # Warmup
    detector.warmup(iterations=5)
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Run detection
    result = detector.detect(test_frame, frame_id=0, timestamp=0.0)
    
    print(f"\nDetection Result:")
    print(f"  Frame: {result.frame_id}")
    print(f"  Detections: {result.num_detections}")
    print(f"  Inference time: {result.inference_time_ms:.2f}ms")
    print(f"\nStatistics: {detector.get_statistics()}")
