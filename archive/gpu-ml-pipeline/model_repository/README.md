# Triton Inference Server Model Repository

This directory contains the model repository structure for NVIDIA Triton Inference Server.

## Directory Structure

```
model_repository/
├── resnet50/
│   ├── config.pbtxt          # Model configuration
│   └── 1/
│       └── model.plan        # TensorRT engine (generated)
├── yolov8/
│   ├── config.pbtxt          # Model configuration
│   └── 1/
│       └── model.plan        # TensorRT engine (generated)
└── ensemble/
    ├── config.pbtxt          # Ensemble pipeline configuration
    └── 1/
        └── (empty - ensemble has no model file)
```

## Model Configurations

### ResNet50 (Image Classification)
- **Input**: 3x224x224 RGB image (NCHW format)
- **Output**: 1000-class probability distribution
- **Precision**: FP16
- **Max Batch Size**: 32
- **Dynamic Batching**: Enabled (preferred: 4, 8, 16, 32)

### YOLOv8 (Object Detection)
- **Input**: 3x640x640 RGB image (NCHW format)
- **Output**: 84x8400 detection tensor
- **Precision**: FP16
- **Max Batch Size**: 16
- **Dynamic Batching**: Enabled (preferred: 1, 4, 8, 16)

### Ensemble (Detection Pipeline)
- **Input**: Variable-size raw image (uint8)
- **Output**: Filtered detections with NMS applied
- **Pipeline**: Preprocessing → YOLOv8 → Postprocessing

## Building TensorRT Engines

Before deploying to Triton, you need to build TensorRT engines from ONNX models:

```bash
# ResNet50
python scripts/build_engine.py \
    --onnx data/models/resnet50.onnx \
    --output model_repository/resnet50/1/model.plan \
    --precision fp16

# YOLOv8
python scripts/build_engine.py \
    --onnx data/models/yolov8n.onnx \
    --output model_repository/yolov8/1/model.plan \
    --precision fp16 \
    --input-shape images:1x3x640x640
```

## Starting Triton Server

```bash
# Using Docker
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models

# Or using Docker Compose
docker-compose -f docker/docker-compose.yml up triton
```

## Verifying Deployment

```bash
# Check server health
curl -v localhost:8000/v2/health/ready

# Check model status
curl localhost:8000/v2/models/resnet50

# Run inference
python -c "
from src.triton.client import TritonClient
client = TritonClient('localhost:8001')
result = client.infer_resnet50('data/sample/images/cat.jpg')
print(f'Top prediction: {result[0]}')
"
```

## Performance Tuning

### Dynamic Batching
Adjust `preferred_batch_size` and `max_queue_delay_microseconds` based on your latency/throughput requirements:
- Lower queue delay = Lower latency, potentially lower throughput
- Higher queue delay = Higher throughput via larger batches

### Instance Groups
Increase `count` in `instance_group` to run multiple model instances:
- More instances = Higher throughput
- More instances = More GPU memory usage

### Model Warmup
The warmup configuration ensures consistent latency by pre-loading CUDA graphs:
- Add warmup configurations for your expected batch sizes
- This eliminates cold-start latency on first inference

## Metrics

Triton exposes Prometheus metrics at `localhost:8002/metrics`:
- `nv_inference_request_success`: Successful inference count
- `nv_inference_request_failure`: Failed inference count
- `nv_inference_compute_infer_duration_us`: Inference compute time
- `nv_inference_queue_duration_us`: Time spent in queue

## Troubleshooting

### Model fails to load
1. Check TensorRT engine compatibility with GPU compute capability
2. Verify input/output dimensions match config
3. Check GPU memory availability

### High latency
1. Enable CUDA graphs in optimization settings
2. Tune dynamic batching parameters
3. Consider instance group scaling

### Low throughput
1. Increase instance count
2. Increase max_queue_delay_microseconds
3. Use larger preferred_batch_size values
