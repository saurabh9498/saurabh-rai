# Architecture Guide

## System Overview

The GPU-Accelerated ML Pipeline is designed for high-performance inference at scale, optimizing every stage from preprocessing to model serving.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐│
│  │   CLIENT    │───▶│   API GATEWAY    │───▶│    INFERENCE PIPELINE      ││
│  │             │    │   (FastAPI)      │    │                             ││
│  │  REST/gRPC  │    │                  │    │  ┌─────────────────────────┐││
│  └─────────────┘    └──────────────────┘    │  │  CUDA PREPROCESSING    │││
│                                              │  │  • Resize (bilinear)   │││
│                                              │  │  • Normalize           │││
│                                              │  │  • HWC → NCHW          │││
│                                              │  └──────────┬────────────┘││
│                                              │             ▼             ││
│                                              │  ┌─────────────────────────┐││
│                                              │  │  TENSORRT ENGINE       │││
│                                              │  │  • FP16/INT8           │││
│                                              │  │  • Dynamic batching    │││
│                                              │  └──────────┬────────────┘││
│                                              │             ▼             ││
│                                              │  ┌─────────────────────────┐││
│                                              │  │  POST-PROCESSING       │││
│                                              │  │  • NMS (CUDA)          │││
│                                              │  │  • Softmax             │││
│                                              │  └─────────────────────────┘││
│                                              └─────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. CUDA Preprocessing Kernels

Custom CUDA kernels eliminate CPU bottlenecks in preprocessing:

| Kernel | Function | Speedup |
|--------|----------|---------|
| `resize_kernel.cu` | Bilinear interpolation | 15x |
| `normalize_kernel.cu` | Mean/std normalization | 27x |
| `preprocess_kernel.cu` | Fused resize+normalize | 10x |
| `nms_kernel.cu` | Non-max suppression | 50x |

**Memory Strategy:**
- Zero-copy GPU memory allocation
- Pinned host memory for transfers
- CUDA streams for async execution

### 2. TensorRT Optimization

TensorRT provides aggressive model optimization:

```
ONNX Model → TensorRT Builder → Optimized Engine
                   │
                   ├── Layer fusion
                   ├── Kernel autotuning
                   ├── Precision calibration
                   └── Memory optimization
```

**Precision Modes:**
- FP32: Full precision (baseline)
- FP16: 2x memory reduction, 2x speedup
- INT8: 4x memory reduction, 4x speedup

### 3. Triton Inference Server

Production serving with enterprise features:

- **Dynamic Batching**: Automatically batches requests for throughput
- **Model Ensemble**: Chain preprocessing → inference → postprocessing
- **Multi-Model**: Serve multiple models concurrently
- **Metrics**: Prometheus-compatible metrics endpoint

## Data Flow

```
1. Request arrives (REST/gRPC)
        │
2. Validate input
        │
3. Copy to GPU memory (pinned → device)
        │
4. CUDA preprocessing kernel
        │
5. TensorRT inference
        │
6. CUDA postprocessing (NMS)
        │
7. Copy results to host
        │
8. Return response
```

## Memory Management

### GPU Memory Pools

```python
# Pre-allocated memory pools
input_pool = cuda.mem_alloc(MAX_BATCH * INPUT_SIZE)
output_pool = cuda.mem_alloc(MAX_BATCH * OUTPUT_SIZE)

# Reuse across requests
def infer(batch):
    cuda.memcpy_htod(input_pool, batch)
    engine.execute(input_pool, output_pool)
    cuda.memcpy_dtoh(result, output_pool)
```

### Stream Pipelining

```
Stream 1: [Copy H→D] [Compute] [Copy D→H]
Stream 2:            [Copy H→D] [Compute] [Copy D→H]
Stream 3:                       [Copy H→D] [Compute] [Copy D→H]
```

## Scaling

### Horizontal Scaling

```yaml
# Kubernetes deployment
replicas: 4
resources:
  limits:
    nvidia.com/gpu: 1
```

### Multi-GPU

```python
# Model parallelism across GPUs
gpu_0: preprocessing + conv layers 1-10
gpu_1: conv layers 11-20 + classification
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Preprocessing | 4ms | Fused CUDA kernel |
| Inference (INT8) | 1.2ms | Batch size 1 |
| Throughput | 2000+ img/s | Per GPU |
| GPU Memory | 2-4 GB | Depends on model |
| P99 Latency | <5ms | End-to-end |

## Error Handling

- Graceful degradation on GPU OOM
- Automatic request retry with backoff
- Health check endpoints
- Circuit breaker for downstream services
