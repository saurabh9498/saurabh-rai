#!/usr/bin/env python3
"""
Basic Detection Example.

Simple object detection on a single image demonstrating the core detection pipeline.
This is the simplest way to get started with Retail Vision Analytics.

Usage:
    python examples/basic_detection.py --image store_photo.jpg
    python examples/basic_detection.py --image store_photo.jpg --output result.jpg
    python examples/basic_detection.py --image store_photo.jpg --model data/models/yolov8n_fp16.engine
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Class names for retail detection
CLASS_NAMES = [
    "person",
    "shopping_cart", 
    "basket",
    "product",
    "shelf",
    "price_tag",
    "employee",
]

# Colors for visualization (BGR format)
CLASS_COLORS = {
    "person": (0, 255, 0),        # Green
    "shopping_cart": (255, 165, 0),  # Orange
    "basket": (255, 0, 255),      # Magenta
    "product": (0, 255, 255),     # Yellow
    "shelf": (128, 128, 128),     # Gray
    "price_tag": (0, 0, 255),     # Red
    "employee": (255, 0, 0),      # Blue
}


def load_image(image_path: str) -> np.ndarray:
    """Load image from file."""
    try:
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        sys.exit(1)


def preprocess_image(
    image: np.ndarray,
    input_size: tuple = (640, 640),
) -> np.ndarray:
    """
    Preprocess image for inference.
    
    Args:
        image: BGR image (H, W, C)
        input_size: Model input size (H, W)
    
    Returns:
        Preprocessed image (1, C, H, W) float32
    """
    import cv2
    
    # Resize maintaining aspect ratio
    h, w = image.shape[:2]
    target_h, target_w = input_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    # Convert to float and normalize
    blob = padded.astype(np.float32) / 255.0
    
    # HWC -> CHW
    blob = blob.transpose(2, 0, 1)
    
    # Add batch dimension
    blob = np.expand_dims(blob, 0)
    
    return blob, scale, (pad_x, pad_y)


def run_inference_mock(
    image: np.ndarray,
    conf_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Mock inference for demonstration without TensorRT.
    
    In production, replace with actual TensorRT or ONNX inference.
    """
    h, w = image.shape[:2]
    
    # Generate realistic mock detections
    detections = []
    
    # Simulate detecting 2-5 people
    num_people = np.random.randint(2, 6)
    for i in range(num_people):
        x = np.random.randint(50, w - 100)
        y = np.random.randint(50, h - 200)
        det_w = np.random.randint(60, 120)
        det_h = np.random.randint(150, 250)
        
        detections.append({
            "class_id": 0,
            "class_name": "person",
            "confidence": np.random.uniform(0.7, 0.98),
            "bbox": [x, y, det_w, det_h],  # x, y, w, h
        })
    
    # Simulate detecting 0-2 carts
    num_carts = np.random.randint(0, 3)
    for i in range(num_carts):
        x = np.random.randint(50, w - 80)
        y = np.random.randint(h // 2, h - 100)
        det_w = np.random.randint(70, 120)
        det_h = np.random.randint(80, 130)
        
        detections.append({
            "class_id": 1,
            "class_name": "shopping_cart",
            "confidence": np.random.uniform(0.6, 0.95),
            "bbox": [x, y, det_w, det_h],
        })
    
    # Filter by confidence
    detections = [d for d in detections if d["confidence"] >= conf_threshold]
    
    return detections


def run_inference_tensorrt(
    engine_path: str,
    image: np.ndarray,
    conf_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Run inference using TensorRT engine.
    
    Args:
        engine_path: Path to TensorRT engine
        image: Input image (BGR)
        conf_threshold: Confidence threshold
    
    Returns:
        List of detections
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("TensorRT not available, using mock inference")
        return run_inference_mock(image, conf_threshold)
    
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Preprocess
    blob, scale, padding = preprocess_image(image)
    
    # Allocate buffers
    d_input = cuda.mem_alloc(blob.nbytes)
    output_shape = (1, 84, 8400)  # YOLOv8 output shape
    h_output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    # Inference
    cuda.memcpy_htod_async(d_input, blob, stream)
    context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    # Post-process
    detections = postprocess_yolov8(h_output, scale, padding, conf_threshold)
    
    return detections


def postprocess_yolov8(
    output: np.ndarray,
    scale: float,
    padding: tuple,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Post-process YOLOv8 output.
    
    Args:
        output: Raw model output (1, 84, 8400)
        scale: Image scale factor
        padding: Padding applied (pad_x, pad_y)
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
    
    Returns:
        List of detections
    """
    # Transpose to (8400, 84)
    predictions = output[0].T
    
    # Extract boxes and scores
    boxes = predictions[:, :4]  # x_center, y_center, width, height
    scores = predictions[:, 4:]  # Class scores
    
    # Get best class for each detection
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # Filter by confidence
    mask = confidences >= conf_threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    
    # Convert to x, y, w, h format
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x = x_center - w / 2
    y = y_center - h / 2
    
    # Remove padding and scale
    pad_x, pad_y = padding
    x = (x - pad_x) / scale
    y = (y - pad_y) / scale
    w = w / scale
    h = h / scale
    
    # NMS (simplified)
    detections = []
    for i in range(len(boxes)):
        if class_ids[i] < len(CLASS_NAMES):
            detections.append({
                "class_id": int(class_ids[i]),
                "class_name": CLASS_NAMES[class_ids[i]],
                "confidence": float(confidences[i]),
                "bbox": [float(x[i]), float(y[i]), float(w[i]), float(h[i])],
            })
    
    return detections


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw detection boxes on image."""
    import cv2
    
    output = image.copy()
    
    for det in detections:
        x, y, w, h = [int(v) for v in det["bbox"]]
        class_name = det["class_name"]
        confidence = det["confidence"]
        
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        
        # Draw box
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            output,
            (x, y - label_size[1] - 10),
            (x + label_size[0], y),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Basic Detection Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basic_detection.py --image store_photo.jpg
  python basic_detection.py --image store_photo.jpg --output result.jpg
  python basic_detection.py --image store_photo.jpg --model model.engine --conf 0.5
        """,
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output image path (optional)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="data/models/yolov8n_retail_fp16.engine",
        help="Model path (TensorRT engine or ONNX)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output detections as JSON",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display result window",
    )
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run inference
    print(f"Running inference...")
    start_time = time.time()
    
    if Path(args.model).exists():
        detections = run_inference_tensorrt(args.model, image, args.conf)
    else:
        print(f"Model not found: {args.model}")
        print("Using mock inference for demonstration")
        detections = run_inference_mock(image, args.conf)
    
    inference_time = (time.time() - start_time) * 1000
    
    # Output results
    print(f"\nInference time: {inference_time:.2f} ms")
    print(f"Detections: {len(detections)}")
    
    if args.json:
        result = {
            "image": args.image,
            "inference_time_ms": round(inference_time, 2),
            "detections": detections,
        }
        print(json.dumps(result, indent=2))
    else:
        for i, det in enumerate(detections):
            print(f"  [{i+1}] {det['class_name']}: {det['confidence']:.2f} "
                  f"@ ({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                  f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})")
    
    # Draw and save/display
    output_image = draw_detections(image, detections)
    
    if args.output:
        import cv2
        cv2.imwrite(args.output, output_image)
        print(f"\nSaved result to: {args.output}")
    
    if not args.no_display:
        try:
            import cv2
            cv2.imshow("Detections", output_image)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            print("Display not available (headless mode)")


if __name__ == "__main__":
    main()
