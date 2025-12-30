# Models Directory

This directory stores trained model artifacts, checkpoints, and exported models.

## Directory Structure

```
models/
├── checkpoints/          # Training checkpoints
│   ├── two_tower/
│   ├── dlrm/
│   └── dcn/
├── exported/             # Production-ready exported models
│   ├── onnx/            # ONNX format for Triton
│   ├── torchscript/     # TorchScript format
│   └── tensorrt/        # TensorRT optimized
├── embeddings/          # Pre-trained embeddings
│   ├── user_embeddings.npy
│   └── item_embeddings.npy
└── README.md
```

## Model Artifacts

### Training Checkpoints

Saved during training with:
```bash
python scripts/train.py --save-dir models/checkpoints/
```

Checkpoint format:
```
checkpoint_epoch_10.pt
├── model_state_dict
├── optimizer_state_dict
├── epoch
├── loss
└── metrics
```

### Exported Models

Export trained models for serving:
```bash
python scripts/export_model.py \
    --checkpoint models/checkpoints/two_tower/best.pt \
    --format onnx \
    --output models/exported/onnx/two_tower.onnx
```

Supported formats:
- **ONNX**: For Triton Inference Server
- **TorchScript**: For PyTorch serving
- **TensorRT**: For GPU-optimized inference

### Embeddings

Pre-computed embeddings for fast retrieval:
```python
# Generate embeddings
python scripts/export_model.py --export-embeddings \
    --output models/embeddings/
```

## Model Registry

| Model | Version | Format | Size | Latency | Notes |
|-------|---------|--------|------|---------|-------|
| Two-Tower | v1.0 | ONNX | 45MB | 8ms | Production |
| DLRM | v1.0 | ONNX | 120MB | 15ms | A/B test |
| DCN-v2 | v0.9 | TorchScript | 85MB | 12ms | Staging |

## Usage

### Loading a Checkpoint

```python
import torch
from src.models.two_tower import TwoTowerModel

# Load checkpoint
checkpoint = torch.load('models/checkpoints/two_tower/best.pt')
model = TwoTowerModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Loading ONNX Model

```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession('models/exported/onnx/two_tower.onnx')

# Run inference
outputs = session.run(None, {
    'user_features': user_features,
    'item_features': item_features
})
```

### Triton Model Repository

For Triton serving, copy models to the model repository:
```bash
cp -r models/exported/onnx/two_tower models/triton_repo/two_tower/1/model.onnx
```

## Storage Notes

- **Git LFS**: Large model files should use Git LFS
- **Cloud Storage**: Production models stored in S3/GCS
- **Versioning**: Use MLflow or DVC for model versioning

## .gitignore

Model files are excluded from git by default. To track:
```bash
git lfs track "*.pt"
git lfs track "*.onnx"
```
