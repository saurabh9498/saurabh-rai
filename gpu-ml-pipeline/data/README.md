# Data Directory

This directory contains models, sample data, and calibration data for the GPU-Accelerated ML Pipeline.

## Directory Structure

```
data/
├── README.md                    # This file
├── models/                      # Pre-trained and optimized models
│   ├── resnet50.onnx           # Original ONNX model
│   ├── resnet50_fp16.engine    # TensorRT FP16 engine
│   ├── resnet50_int8.engine    # TensorRT INT8 engine
│   └── yolov8n.onnx            # YOLOv8 nano model
├── sample/                      # Sample data for testing
│   ├── images/                 # Test images
│   │   ├── cat.jpg
│   │   ├── dog.jpg
│   │   └── car.jpg
│   ├── batches/                # Pre-batched test data
│   │   └── batch_32.npy
│   └── expected_outputs/       # Expected inference results
│       └── resnet50_outputs.json
└── calibration/                 # INT8 calibration data
    ├── calibration_images/     # Representative images
    └── calibration_cache.bin   # Cached calibration data
```

## Setting Up Data

### Option 1: Generate Sample Data (Recommended)

Use the provided script to generate synthetic test data:

```bash
# From project root
python scripts/generate_sample_data.py

# Verify files were created
ls -la data/sample/images/
ls -la data/models/
```

This generates:
- Synthetic test images (random noise patterns)
- Sample ONNX model (dummy ResNet50)
- Calibration dataset
- Expected output templates

### Option 2: Download Pre-trained Models

```bash
# Download ResNet50 ONNX
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx \
    -O data/models/resnet50.onnx

# Download YOLOv8 (requires ultralytics)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
mv yolov8n.onnx data/models/
```

### Option 3: Use Your Own Models

1. **Export to ONNX** (from PyTorch):
   ```python
   import torch
   
   model = YourModel()
   model.load_state_dict(torch.load('model.pth'))
   model.eval()
   
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy_input, 'data/models/your_model.onnx',
                     opset_version=17,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                   'output': {0: 'batch_size'}})
   ```

2. **Place in models directory**:
   ```bash
   cp your_model.onnx data/models/
   ```

## Building TensorRT Engines

Convert ONNX models to optimized TensorRT engines:

```bash
# Build FP16 engine
python scripts/build_engine.py \
    --onnx data/models/resnet50.onnx \
    --output data/models/resnet50_fp16.engine \
    --precision fp16

# Build INT8 engine (requires calibration data)
python scripts/build_engine.py \
    --onnx data/models/resnet50.onnx \
    --output data/models/resnet50_int8.engine \
    --precision int8 \
    --calibration-data data/calibration/calibration_images/
```

## Calibration Data (for INT8)

INT8 quantization requires representative calibration data:

### Requirements
- 500-1000 representative images
- Same preprocessing as inference
- Covers typical input distribution

### Setup Calibration Data

```bash
# Copy representative images
cp /path/to/your/images/*.jpg data/calibration/calibration_images/

# Or use the sample generator
python scripts/generate_sample_data.py --calibration-only --num-images 1000
```

### Calibration Process

```python
from src.tensorrt.calibrator import Int8Calibrator

calibrator = Int8Calibrator(
    calibration_dir="data/calibration/calibration_images/",
    cache_file="data/calibration/calibration_cache.bin",
    batch_size=32,
    input_shape=(3, 224, 224)
)
```

## Sample Images

For testing and demos, place sample images in `data/sample/images/`:

| Image | Purpose |
|-------|---------|
| `cat.jpg` | Classification testing |
| `dog.jpg` | Classification testing |
| `car.jpg` | Object detection testing |
| `street.jpg` | Multi-object detection |

### Image Requirements
- Format: JPEG, PNG
- Size: Any (will be resized by preprocessing)
- Color: RGB (3 channels)

## Batch Testing Data

Pre-batched numpy arrays for performance testing:

```python
import numpy as np

# Create batch of 32 images (preprocessed)
batch = np.random.rand(32, 3, 224, 224).astype(np.float32)
np.save('data/sample/batches/batch_32.npy', batch)
```

## File Size Guidelines

| File Type | Typical Size |
|-----------|--------------|
| ONNX Model (ResNet50) | ~100 MB |
| TensorRT FP32 Engine | ~100 MB |
| TensorRT FP16 Engine | ~50 MB |
| TensorRT INT8 Engine | ~25 MB |
| Calibration Cache | ~1 MB |

## Git LFS (Recommended)

For large model files, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.onnx"
git lfs track "*.engine"
git lfs track "*.bin"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model files"
```

## Environment Variables

Configure data paths in `.env`:

```bash
# Model paths
MODEL_DIR=./data/models
DEFAULT_MODEL=resnet50_fp16.engine

# Calibration
CALIBRATION_DIR=./data/calibration/calibration_images
CALIBRATION_CACHE=./data/calibration/calibration_cache.bin

# Sample data
SAMPLE_IMAGES_DIR=./data/sample/images
```

## Troubleshooting

### "Model file not found"

```bash
# Check model exists
ls -la data/models/

# Regenerate if missing
python scripts/generate_sample_data.py
```

### "Calibration data empty"

```bash
# Check calibration images
ls data/calibration/calibration_images/ | wc -l

# Should have at least 100 images for good INT8 accuracy
```

### "Engine build failed"

```bash
# Verify CUDA and TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"

# Check GPU memory (engines need significant GPU memory to build)
nvidia-smi
```

## Data Privacy

⚠️ **Important:**
- Do NOT commit proprietary models to public repos
- Use `.gitignore` for large files
- Model files in this directory are for demonstration only
- For production, use secure model storage (S3, GCS, etc.)
