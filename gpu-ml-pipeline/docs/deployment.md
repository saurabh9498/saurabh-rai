# Deployment Guide

## Prerequisites

- NVIDIA GPU (Compute Capability 7.0+)
- CUDA 12.x
- TensorRT 8.6+
- Docker with NVIDIA Container Toolkit
- Kubernetes (optional, for production)

## Local Development

### 1. Environment Setup

```bash
# Create conda environment
conda create -n gpu-pipeline python=3.10
conda activate gpu-pipeline

# Install dependencies
pip install -r requirements.txt

# Build CUDA extensions
cd src/cuda && python setup.py install && cd ../..
```

### 2. Build TensorRT Engine

```bash
# From ONNX model
python scripts/build_engine.py \
    --onnx models/resnet50.onnx \
    --output engines/resnet50.engine \
    --precision fp16

# With INT8 calibration
python scripts/build_engine.py \
    --onnx models/resnet50.onnx \
    --output engines/resnet50_int8.engine \
    --precision int8 \
    --calib-data data/calibration/
```

### 3. Run Inference

```python
from src.preprocessing.pipeline import Pipeline

pipeline = Pipeline(
    preprocessing="gpu",
    model_path="engines/resnet50.engine"
)

results = pipeline.run(images)
```

## Docker Deployment

### Build Image

```bash
# Production image
docker build -t gpu-ml-pipeline:latest -f docker/Dockerfile --target production .

# Development image
docker build -t gpu-ml-pipeline:dev -f docker/Dockerfile --target development .
```

### Run Container

```bash
# Single container with GPU
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/engines:/app/engines:ro \
    gpu-ml-pipeline:latest
```

### Docker Compose

```bash
# Start full stack (API + Triton)
docker-compose -f docker/docker-compose.yml up -d

# With monitoring (Prometheus + Grafana)
docker-compose -f docker/docker-compose.yml --profile monitoring up -d

# Scale Triton instances
docker-compose -f docker/docker-compose.yml up -d --scale triton=3
```

## Kubernetes Deployment

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-pipeline-config
data:
  MODEL_PATH: /models/resnet50.engine
  LOG_LEVEL: INFO
  MAX_BATCH_SIZE: "32"
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gpu-pipeline
  template:
    metadata:
      labels:
        app: gpu-pipeline
    spec:
      containers:
      - name: inference
        image: gpu-ml-pipeline:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
        envFrom:
        - configMapRef:
            name: gpu-pipeline-config
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-pvc
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: gpu-pipeline-service
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
  selector:
    app: gpu-pipeline
```

### HPA (Horizontal Pod Autoscaler)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpu-pipeline-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gpu-pipeline
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Triton Inference Server

### Model Repository Structure

```
model_repository/
├── preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── resnet50_trt/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
└── ensemble/
    └── config.pbtxt
```

### Deploy Triton

```bash
# Docker
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models

# Check health
curl localhost:8000/v2/health/ready
```

## Cloud Deployments

### AWS ECS

```bash
# Create cluster with GPU instances
aws ecs create-cluster --cluster-name gpu-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task.json

# Run service
aws ecs create-service \
    --cluster gpu-cluster \
    --service-name gpu-pipeline \
    --task-definition gpu-pipeline:1 \
    --desired-count 2
```

### GCP Cloud Run (GPU Preview)

```bash
gcloud run deploy gpu-pipeline \
    --image gcr.io/project/gpu-ml-pipeline \
    --platform managed \
    --region us-central1 \
    --gpu 1 \
    --gpu-type nvidia-l4
```

## Monitoring

### Prometheus Metrics

Available at `/metrics`:

- `inference_requests_total`
- `inference_latency_seconds`
- `gpu_memory_used_bytes`
- `batch_size_histogram`

### Grafana Dashboard

Import dashboard from `docker/grafana/dashboards/inference.json`

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### OOM Errors

- Reduce batch size
- Use FP16/INT8 precision
- Enable memory pool limits

### Slow Performance

- Run benchmark: `python scripts/benchmark.py`
- Check GPU utilization: `nvidia-smi dmon`
- Profile with Nsight Systems
