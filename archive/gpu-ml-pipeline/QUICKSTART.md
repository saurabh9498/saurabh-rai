# 🚀 Quick Start Guide

Get the GPU-Accelerated ML Pipeline running in minutes.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU | Compute Capability 7.0+ (Volta, Turing, Ampere, Hopper) |
| CUDA 11.8+ | Or CUDA 12.x |
| TensorRT 8.6+ | For optimized inference |
| Python 3.10+ | Required |
| Docker (optional) | Easiest setup with NGC containers |

### Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check TensorRT (if installed)
python -c "import tensorrt; print(tensorrt.__version__)"
```

---

## Option 1: Docker with NGC Container (Recommended)

The fastest way to get everything running with all dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/gpu-ml-pipeline.git
cd gpu-ml-pipeline

# 2. Set up environment variables
cp .env.example .env

# 3. Build and run with Docker Compose
cd docker
docker-compose up --build

# 4. Access the applications:
#    - API:       http://localhost:8000
#    - Triton:    http://localhost:8001 (gRPC), 8002 (HTTP)
#    - Metrics:   http://localhost:8000/metrics
```

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI inference server |
| `triton` | 8001/8002 | NVIDIA Triton Inference Server |
| `prometheus` | 9090 | Metrics collection |

---

## Option 2: Local Development (with CUDA)

For development and debugging.

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/gpu-ml-pipeline.git
cd gpu-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# OR
venv\Scripts\activate           # Windows
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build CUDA extensions (optional - for custom kernels)
cd src/cuda
python setup.py build_ext --inplace
cd ../..
```

### Step 3: Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates:
- Sample test images
- Batch data (numpy arrays)
- Model placeholders with download instructions
- Calibration data for INT8

### Step 4: Download/Prepare Models

```bash
# Option A: Download ResNet50 ONNX
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx \
    -O data/models/resnet50.onnx

# Option B: Export YOLOv8 (requires ultralytics)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
mv yolov8n.onnx data/models/
```

### Step 5: Build TensorRT Engine

```bash
# Build FP16 engine (recommended for most GPUs)
python scripts/build_engine.py \
    --onnx data/models/resnet50.onnx \
    --output data/models/resnet50_fp16.engine \
    --precision fp16

# Build INT8 engine (highest performance, requires calibration)
python scripts/build_engine.py \
    --onnx data/models/resnet50.onnx \
    --output data/models/resnet50_int8.engine \
    --precision int8 \
    --calibration-data data/calibration/calibration_images/
```

### Step 6: Run Inference

```bash
# Run benchmark
python scripts/benchmark.py \
    --engine data/models/resnet50_fp16.engine \
    --batch-size 32

# Start API server
uvicorn src.api.main:app --reload --port 8000
```

---

## Option 3: Python Script Demo

Minimal code to test the pipeline.

```python
# demo.py
from src.tensorrt.inference import TensorRTInference
from src.preprocessing.pipeline import PreprocessingPipeline
import numpy as np

# Initialize preprocessing
preprocess = PreprocessingPipeline(
    input_size=(224, 224),
    normalize=True,
    use_gpu=True  # GPU-accelerated preprocessing
)

# Load TensorRT engine
engine = TensorRTInference("data/models/resnet50_fp16.engine")

# Load and preprocess image
image = preprocess.load_image("data/sample/images/cat.jpg")
batch = preprocess.preprocess(image)

# Run inference
outputs = engine.infer(batch)

# Get top-5 predictions
top5_idx = np.argsort(outputs[0])[-5:][::-1]
print(f"Top-5 predictions: {top5_idx}")
```

Run with:
```bash
python demo.py
```

---

## Option 4: Jupyter Notebook

Interactive exploration of the pipeline.

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/demo.ipynb
```

The demo notebook covers:
1. ✅ CUDA kernel benchmarks
2. ✅ Preprocessing pipeline
3. ✅ TensorRT engine building
4. ✅ Inference with different precisions
5. ✅ Performance comparisons

---

## Verify Installation

### Check GPU Inference

```bash
# Run quick benchmark
python scripts/benchmark.py --quick

# Expected output:
# GPU: NVIDIA GeForce RTX 4090
# Model: resnet50_fp16.engine
# Batch Size: 1
# Throughput: 1,250 FPS
# Latency (P99): 1.2 ms
```

### Test Preprocessing Kernels

```python
python -c "
from src.preprocessing.pipeline import PreprocessingPipeline
p = PreprocessingPipeline(use_gpu=True)
print('GPU preprocessing initialized successfully!')
"
```

### Check API Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu": "NVIDIA GeForce RTX 4090",
  "cuda_version": "12.1",
  "tensorrt_version": "8.6.1"
}
```

---

## Project Structure

```
gpu-ml-pipeline/
│
├── src/                          # Source code
│   ├── cuda/                     # 🔥 Custom CUDA kernels
│   │   ├── kernels/
│   │   │   ├── resize_kernel.cu      # GPU image resize
│   │   │   ├── normalize_kernel.cu   # GPU normalization
│   │   │   ├── preprocess_kernel.cu  # Fused preprocessing
│   │   │   └── nms_kernel.cu         # Non-max suppression
│   │   ├── bindings.cpp              # Python bindings
│   │   └── setup.py                  # Build script
│   │
│   ├── tensorrt/                 # ⚡ TensorRT integration
│   │   ├── builder.py               # Engine builder
│   │   ├── inference.py             # Inference runtime
│   │   ├── calibrator.py            # INT8 calibration
│   │   └── optimization.py          # Optimization configs
│   │
│   ├── triton/                   # 🚀 Triton Inference Server
│   │   └── client.py                # Triton client
│   │
│   ├── preprocessing/            # 📸 Image preprocessing
│   │   └── pipeline.py              # Preprocessing pipeline
│   │
│   └── utils/                    # 🔧 Utilities
│       └── benchmark.py             # Benchmarking tools
│
├── data/                         # 📁 Data files
│   ├── models/                      # ONNX and TensorRT models
│   ├── sample/                      # Test images and batches
│   └── calibration/                 # INT8 calibration data
│
├── configs/                      # ⚙️ Configuration
│   └── tensorrt_config.yaml         # TensorRT settings
│
├── docker/                       # 🐳 Docker setup
│   ├── Dockerfile                   # NGC-based container
│   └── docker-compose.yml
│
├── scripts/                      # 🔨 Utility scripts
│   ├── benchmark.py                 # Performance benchmarks
│   ├── build_engine.py              # TensorRT engine builder
│   └── generate_sample_data.py      # Sample data generator
│
├── tests/                        # ✅ Test suites
│   ├── unit/
│   └── integration/
│
├── notebooks/                    # 📓 Jupyter notebooks
│   └── demo.ipynb
│
└── docs/                         # 📖 Documentation
    ├── architecture.md
    ├── optimization_guide.md
    └── deployment.md
```

---

## Performance Expectations

### ResNet50 (224x224) on RTX 4090

| Precision | Batch 1 | Batch 32 | Memory |
|-----------|---------|----------|--------|
| FP32 | 450 FPS | 2,100 FPS | 180 MB |
| FP16 | 890 FPS | 4,200 FPS | 95 MB |
| INT8 | 1,250 FPS | 5,800 FPS | 52 MB |

### Latency (Batch Size 1)

| Precision | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| FP32 | 2.1 ms | 2.4 ms | 2.6 ms |
| FP16 | 1.1 ms | 1.2 ms | 1.4 ms |
| INT8 | 0.7 ms | 0.9 ms | 1.0 ms |

---

## Common Issues

### "CUDA out of memory"

```bash
# Check GPU memory
nvidia-smi

# Use smaller batch size
python scripts/benchmark.py --batch-size 8

# Or use FP16/INT8 for lower memory
```

### "TensorRT engine build failed"

```bash
# Check TensorRT version compatibility
python -c "import tensorrt; print(tensorrt.__version__)"

# Rebuild with verbose logging
python scripts/build_engine.py --onnx model.onnx --verbose
```

### "CUDA kernel launch failed"

```bash
# Check CUDA installation
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Verify compute capability matches
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Docker GPU not accessible

```bash
# Install NVIDIA Container Toolkit
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

---

## Environment Variables

Key variables in `.env`:

```bash
# GPU Settings
CUDA_VISIBLE_DEVICES=0           # GPU device ID
TRT_LOGGER_LEVEL=WARNING         # TensorRT log level

# Model Settings
MODEL_DIR=./data/models
DEFAULT_ENGINE=resnet50_fp16.engine
DEFAULT_PRECISION=fp16

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
MAX_BATCH_SIZE=32
```

See `.env.example` for all available options.

---

## Next Steps

1. **Run benchmarks**: Compare FP32, FP16, INT8 performance
2. **Try custom models**: Export your own ONNX models
3. **Explore kernels**: Check out custom CUDA kernels in `src/cuda/`
4. **Deploy with Triton**: Set up production serving
5. **Read optimization guide**: See `docs/optimization_guide.md`

---

## Getting Help

- 📖 [Architecture Guide](docs/architecture.md)
- ⚡ [Optimization Guide](docs/optimization_guide.md)
- 🚀 [Deployment Guide](docs/deployment.md)
- 🔌 [API Reference](docs/api_reference.md)
- 🐛 [Open an Issue](https://github.com/your-username/gpu-ml-pipeline/issues)

---

Happy accelerating! 🚀
