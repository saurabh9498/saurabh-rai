# Optimization Guide

## Overview

This guide covers optimization strategies for maximum inference performance.

## Preprocessing Optimization

### 1. Use Fused CUDA Kernels

Fused kernels combine multiple operations into a single pass:

```python
# Instead of separate operations
resized = resize(image)
normalized = normalize(resized)
transposed = hwc_to_nchw(normalized)

# Use fused kernel (10x faster)
output = fused_preprocess(image)
```

### 2. Memory Optimization

**Pinned Memory:**
```python
# Use pinned memory for faster H2D transfers
host_buffer = cuda.pagelocked_empty(shape, dtype)
```

**Memory Pools:**
```python
# Pre-allocate to avoid allocation overhead
input_pool = cuda.mem_alloc(max_size)
```

### 3. Stream Pipelining

Overlap compute and memory transfers:

```python
for i, batch in enumerate(batches):
    stream = streams[i % num_streams]
    cuda.memcpy_htod_async(d_input, batch, stream)
    engine.execute_async(stream)
    cuda.memcpy_dtoh_async(output, d_output, stream)
```

## TensorRT Optimization

### 1. Precision Selection

| Precision | Memory | Speed | Accuracy |
|-----------|--------|-------|----------|
| FP32 | 1x | 1x | Baseline |
| FP16 | 0.5x | 2x | ~Same |
| INT8 | 0.25x | 4x | -0.5% typical |

**Recommendation:**
- Start with FP16 for most models
- Use INT8 for throughput-critical applications
- Keep sensitive layers in higher precision

### 2. Dynamic Shapes

Configure optimization profiles for variable batch sizes:

```python
profile.set_shape(
    "input",
    min=(1, 3, 224, 224),   # Minimum shape
    opt=(16, 3, 224, 224),  # Optimal (most common)
    max=(64, 3, 224, 224)   # Maximum shape
)
```

### 3. Layer Precision Override

Keep sensitive layers in higher precision:

```yaml
layer_precision:
  - pattern: "Conv_0"    # First conv
    precision: "fp16"
  - pattern: "Softmax*"  # Output layer
    precision: "fp32"
```

### 4. Timing Cache

Reuse kernel autotuning results:

```python
config = BuildConfig(
    enable_timing_cache=True,
    timing_cache_path="cache/timing.cache"
)
```

Reduces rebuild time from 10+ minutes to <1 minute.

### 5. INT8 Calibration Tips

- Use representative dataset (500+ batches)
- Include edge cases in calibration data
- Try different algorithms (entropy vs minmax)
- Use calibration cache for reproducibility

```python
# Compare calibration algorithms
for algorithm in ["entropy", "entropy_2", "minmax"]:
    calibrator = Calibrator(algorithm=algorithm)
    # Evaluate accuracy
```

## Triton Server Optimization

### 1. Dynamic Batching

Automatically batch requests for throughput:

```protobuf
dynamic_batching {
    preferred_batch_size: [ 8, 16, 32, 64 ]
    max_queue_delay_microseconds: 100
}
```

**Tuning:**
- Lower delay = lower latency, lower throughput
- Higher delay = higher latency, higher throughput

### 2. Instance Groups

Run multiple model instances:

```protobuf
instance_group [
    {
        count: 2           # 2 instances
        kind: KIND_GPU
        gpus: [ 0 ]        # On GPU 0
    }
]
```

### 3. Model Ensemble

Chain preprocessing → inference → postprocessing:

```protobuf
ensemble_scheduling {
    step [
        { model_name: "preprocessing" }
        { model_name: "inference" }
        { model_name: "postprocessing" }
    ]
}
```

### 4. Response Cache

Cache repeated requests:

```protobuf
response_cache {
    enable: true
}
```

## Benchmarking

### Measure Baseline

```bash
python scripts/benchmark.py \
    --mode all \
    --batch-sizes 1,8,16,32 \
    --iterations 1000
```

### Profile with Nsight

```bash
nsys profile -o report python your_script.py
nsys-ui report.nsys-rep
```

### Key Metrics

| Metric | Target |
|--------|--------|
| Preprocessing | <5ms |
| Inference (P99) | <10ms |
| GPU Utilization | >80% |
| Memory Usage | <80% capacity |

## Common Issues

### Low GPU Utilization

**Cause:** CPU bottleneck in preprocessing
**Solution:** Use CUDA preprocessing kernels

### High Memory Usage

**Cause:** Large batch sizes or FP32 precision
**Solution:** Use INT8, reduce batch size

### Inconsistent Latency

**Cause:** Dynamic memory allocation
**Solution:** Pre-allocate memory pools

### Slow First Inference

**Cause:** CUDA/TensorRT initialization
**Solution:** Warmup with dummy data

```python
engine.warmup(iterations=50)
```

## Optimization Checklist

- [ ] Using fused CUDA preprocessing
- [ ] TensorRT engine built with optimal precision
- [ ] Dynamic batching configured
- [ ] Memory pools pre-allocated
- [ ] Stream pipelining enabled
- [ ] Timing cache enabled
- [ ] Warmup performed
- [ ] Benchmarks validated
