# Data Directory

This directory contains data files for the Retail Vision Analytics system.

## Directory Structure

```
data/
├── README.md           # This file
├── models/             # Pre-trained and converted models
│   ├── .gitkeep
│   ├── yolov8n_retail.onnx       # ONNX model (after download)
│   └── yolov8n_retail_fp16.engine # TensorRT engine (after conversion)
└── sample/             # Sample data for testing
    ├── .gitkeep
    ├── detections.json           # Sample detection events
    ├── journeys.json             # Sample customer journeys
    ├── queue_metrics.json        # Sample queue metrics
    ├── heatmap.json              # Sample heatmap data
    └── alerts.json               # Sample alerts
```

## Setup Options

### Option 1: Generate Sample Data (Recommended)

```bash
# Generate 24 hours of synthetic data
python scripts/generate_sample_data.py --output data/sample/ --hours 24

# Generate smaller dataset for quick testing
python scripts/generate_sample_data.py --output data/sample/ --hours 1
```

### Option 2: Use Makefile

```bash
make sample-data
```

### Option 3: Docker

```bash
docker-compose run --rm retail-vision python scripts/generate_sample_data.py
```

## Models

### Downloading Pre-trained Models

```bash
# Download and convert models
bash scripts/download_models.sh

# Or use Makefile
make download-models
```

### Model Files

| File | Size | Description |
|------|------|-------------|
| `yolov8n.pt` | ~6 MB | Original PyTorch weights |
| `yolov8n_retail.onnx` | ~12 MB | ONNX export for portability |
| `yolov8n_retail_fp16.engine` | ~8 MB | TensorRT FP16 optimized |
| `yolov8n_retail_int8.engine` | ~4 MB | TensorRT INT8 quantized |
| `osnet_x0_25.onnx` | ~3 MB | Person re-identification |

### Converting Models

```python
# Convert ONNX to TensorRT
python -m src.edge.tensorrt_engine convert \
    --input data/models/yolov8n_retail.onnx \
    --output data/models/yolov8n_retail_fp16.engine \
    --precision fp16

# INT8 quantization (requires calibration data)
python -m src.edge.tensorrt_engine convert \
    --input data/models/yolov8n_retail.onnx \
    --output data/models/yolov8n_retail_int8.engine \
    --precision int8 \
    --calibration-dir data/calibration/
```

## Sample Data Schema

### detections.json

```json
{
  "stream_id": "cam-entrance-1",
  "frame_number": 12345,
  "timestamp": "2024-01-15T10:30:00Z",
  "class_name": "person",
  "confidence": 0.92,
  "bbox": {"x": 100, "y": 200, "width": 80, "height": 180},
  "track_id": 42
}
```

### journeys.json

```json
{
  "journey_id": "journey-000001",
  "track_id": 1,
  "stream_id": "cam-entrance-1",
  "start_time": "2024-01-15T10:15:00Z",
  "end_time": "2024-01-15T10:28:00Z",
  "duration_seconds": 780,
  "zones_visited": ["entrance", "aisle-1", "checkout"],
  "zone_dwell_times": {"entrance": 30, "aisle-1": 450, "checkout": 300},
  "converted": true,
  "cart_detected": true
}
```

### queue_metrics.json

```json
{
  "lane_id": "checkout-1",
  "stream_id": "cam-checkout-1",
  "timestamp": "2024-01-15T10:30:00Z",
  "queue_length": 5,
  "avg_wait_time_seconds": 145,
  "max_wait_time_seconds": 280,
  "service_rate": 1.2,
  "abandonment_count": 1
}
```

### heatmap.json

```json
{
  "stream_id": "cam-entrance-1",
  "resolution": [96, 54],
  "cell_size": 20,
  "data": [[0.1, 0.2, ...], ...],
  "hotspots": [
    {"x": 0.15, "y": 0.85, "intensity": 0.95}
  ]
}
```

## Data Privacy

⚠️ **Important**: This system processes video of real people.

- **No PII Storage**: Raw video is not stored by default
- **Anonymization**: All sample data is synthetic
- **Compliance**: Ensure GDPR/CCPA compliance in production
- **Retention**: Configure data retention policies in `configs/app_config.yaml`

## Storage Requirements

| Data Type | Retention | Size/Day | Monthly |
|-----------|-----------|----------|---------|
| Detections | 7 days | ~500 MB | N/A |
| Journeys | 90 days | ~50 MB | 1.5 GB |
| Queue Metrics | 90 days | ~10 MB | 300 MB |
| Heatmaps | 30 days | ~20 MB | 600 MB |
| Alerts | 365 days | ~5 MB | 60 MB |
| Video Clips | 7 days | ~2 GB | N/A |

## Cleaning Up

```bash
# Remove sample data
make clean-data

# Remove all generated files (models + data)
make clean-all
```
