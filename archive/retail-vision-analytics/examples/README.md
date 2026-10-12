# Examples

Standalone examples demonstrating key features of Retail Vision Analytics.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic detection
python examples/basic_detection.py --image path/to/image.jpg

# Run stream demo
python examples/stream_demo.py --source rtsp://camera_ip/stream
```

## Available Examples

| Example | Description | Difficulty |
|---------|-------------|------------|
| `basic_detection.py` | Single image detection | Beginner |
| `stream_demo.py` | Real-time video stream processing | Intermediate |
| `analytics_demo.py` | Customer journey & queue analytics | Intermediate |
| `edge_inference.py` | TensorRT inference on Jetson | Advanced |

## Example Details

### basic_detection.py

Simple object detection on a single image. Great for understanding the detection pipeline.

```bash
# Detect objects in an image
python examples/basic_detection.py --image store_photo.jpg --output result.jpg

# Use specific model
python examples/basic_detection.py --image store_photo.jpg --model data/models/yolov8n_fp16.engine

# Adjust confidence threshold
python examples/basic_detection.py --image store_photo.jpg --conf 0.5
```

### stream_demo.py

Process video streams in real-time with visualization.

```bash
# Process RTSP stream
python examples/stream_demo.py --source rtsp://192.168.1.100:554/stream

# Process video file
python examples/stream_demo.py --source store_footage.mp4

# Process webcam
python examples/stream_demo.py --source 0

# Multi-stream processing
python examples/stream_demo.py --sources rtsp://cam1/stream rtsp://cam2/stream
```

### analytics_demo.py

Demonstrate analytics capabilities with sample data.

```bash
# Run with generated sample data
python examples/analytics_demo.py

# Specify custom data
python examples/analytics_demo.py --data data/sample/detections.json

# Export analytics report
python examples/analytics_demo.py --export report.json
```

### edge_inference.py

TensorRT inference optimized for Jetson devices.

```bash
# Run inference benchmark
python examples/edge_inference.py --model data/models/yolov8n_fp16.engine

# Profile memory usage
python examples/edge_inference.py --model data/models/yolov8n_fp16.engine --profile

# Test INT8 model
python examples/edge_inference.py --model data/models/yolov8n_int8.engine --precision int8
```

## Output Examples

### Detection Output

```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": [100, 200, 180, 380],
      "track_id": 1
    },
    {
      "class": "shopping_cart",
      "confidence": 0.87,
      "bbox": [150, 300, 100, 120],
      "track_id": null
    }
  ],
  "inference_time_ms": 4.2,
  "frame_id": 1
}
```

### Analytics Output

```json
{
  "journeys": 45,
  "avg_dwell_time_seconds": 342,
  "conversion_rate": 0.73,
  "queue_wait_time_seconds": 124,
  "hotspots": ["aisle-1", "checkout-2"]
}
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- For edge examples: TensorRT, PyCUDA

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or image resolution:
```bash
python examples/stream_demo.py --source video.mp4 --batch-size 1 --resolution 640
```

### Slow Inference

Ensure TensorRT engine matches your GPU:
```bash
# Rebuild engine for your GPU
python scripts/convert_to_tensorrt.py --input model.onnx --output model.engine
```

### RTSP Connection Failed

Check camera connectivity:
```bash
ffprobe rtsp://camera_ip:554/stream
```
