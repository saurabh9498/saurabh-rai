# API Reference

## Pipeline

### `Pipeline`

Main inference pipeline class.

```python
from src.preprocessing.pipeline import Pipeline

pipeline = Pipeline(
    preprocessing="gpu",      # "gpu", "cpu", or "cuda_kernels"
    model_path="model.engine",
    postprocessing=softmax,   # Optional callable
    device="cuda:0",
    config=PreprocessConfig()
)
```

#### Methods

**`run(images, return_preprocessed=False)`**

Run inference on images.

| Parameter | Type | Description |
|-----------|------|-------------|
| images | np.ndarray \| List | Input images (HWC, uint8) |
| return_preprocessed | bool | Also return preprocessed tensor |

Returns: `np.ndarray` or `Tuple[np.ndarray, np.ndarray]`

**`get_timing()`**

Get timing breakdown.

Returns: `Dict[str, float]` with keys: `preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms`

---

### `GPUPreprocessor`

GPU-accelerated image preprocessor.

```python
from src.preprocessing.pipeline import GPUPreprocessor, PreprocessConfig

config = PreprocessConfig(
    target_size=(224, 224),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    normalize=True
)

preprocessor = GPUPreprocessor(config=config, mode="gpu")
```

#### Methods

**`process(images, return_numpy=False)`**

Preprocess images.

| Parameter | Type | Description |
|-----------|------|-------------|
| images | np.ndarray \| List | Input images |
| return_numpy | bool | Return numpy instead of tensor |

Returns: `np.ndarray` or `torch.Tensor` (NCHW format)

**`benchmark(num_images=100, batch_size=1)`**

Benchmark preprocessing.

Returns: `Dict` with timing statistics

---

## TensorRT

### `TensorRTBuilder`

Builds optimized TensorRT engines from ONNX.

```python
from src.tensorrt.builder import TensorRTBuilder, BuildConfig

config = BuildConfig(
    precision="int8",
    max_batch_size=64,
    workspace_size_gb=4.0,
    optimization_level=5
)

builder = TensorRTBuilder(
    onnx_path="model.onnx",
    config=config,
    calibration_data=calib_loader  # Required for INT8
)
```

#### Methods

**`build()`**

Build the TensorRT engine.

Returns: `TensorRTEngine`

**`get_network_info()`**

Get parsed network information.

Returns: `Dict` with inputs, outputs, layer count

---

### `TensorRTInference`

Wrapper for TensorRT engine inference.

```python
from src.tensorrt.inference import TensorRTInference

engine = TensorRTInference(
    engine_path="model.engine",
    device_id=0,
    enable_profiling=True
)
```

#### Methods

**`infer(inputs, output_names=None)`**

Run inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| inputs | np.ndarray \| Dict | Input data |
| output_names | List[str] | Outputs to return (None for all) |

Returns: `np.ndarray` or `Dict[str, np.ndarray]`

**`warmup(iterations=10)`**

Warmup the engine with dummy data.

**`get_binding_info()`**

Get input/output binding information.

Returns: `Dict` with inputs and outputs

#### Properties

**`last_metrics`**: `InferenceMetrics` with timing data

---

## Triton Client

### `TritonClient`

Client for Triton Inference Server.

```python
from src.triton.client import TritonClient

client = TritonClient(
    url="localhost:8001",
    protocol="grpc",  # or "http"
    verbose=False
)
```

#### Methods

**`is_server_ready()`**

Check if server is ready.

Returns: `bool`

**`is_model_ready(model_name, model_version="")`**

Check if model is loaded.

Returns: `bool`

**`infer(model_name, inputs, model_version="", outputs=None, timeout=30.0)`**

Run inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| model_name | str | Model name |
| inputs | Dict[str, np.ndarray] | Input tensors |
| model_version | str | Version (empty for latest) |
| outputs | List[str] | Output names to retrieve |
| timeout | float | Request timeout |

Returns: `InferenceResult`

**`get_model_metadata(model_name)`**

Get model metadata.

Returns: `Dict` with inputs, outputs, versions

---

### `ModelConfig`

Generate Triton model configuration.

```python
from src.triton.client import ModelConfig

config = ModelConfig(
    name="resnet50",
    platform="tensorrt_plan",
    max_batch_size=64,
    dynamic_batching={
        "preferred_batch_size": [8, 16, 32],
        "max_queue_delay_microseconds": 100
    }
)

config.save("model_repository/resnet50/config.pbtxt")
```

---

## Calibration

### `EntropyCalibrator`

INT8 calibration using entropy algorithm.

```python
from src.tensorrt.calibrator import EntropyCalibrator, CalibrationDataLoader

# Create data loader
data = CalibrationDataLoader.from_directory(
    "calibration_images/",
    batch_size=8,
    image_size=(224, 224)
)

calibrator = EntropyCalibrator(
    data_loader=data,
    cache_file="calibration.cache",
    num_batches=500
)
```

---

## Benchmarking

### `benchmark_function`

Benchmark any function.

```python
from src.utils.benchmark import benchmark_function

result = benchmark_function(
    func=my_function,
    args=(arg1, arg2),
    warmup=10,
    iterations=100,
    batch_size=32,
    name="my_benchmark"
)

print(f"Mean: {result.mean_ms:.2f}ms")
print(f"P99: {result.p99_ms:.2f}ms")
```

### `PipelineBenchmark`

Comprehensive pipeline benchmarking.

```python
from src.utils.benchmark import PipelineBenchmark

benchmark = PipelineBenchmark(
    pipeline=pipeline,
    input_shape=(1080, 1920, 3),
    batch_sizes=[1, 8, 16, 32]
)

results = benchmark.run()
benchmark.print_report()
benchmark.save_results("benchmark_results.json")
```

---

## Data Classes

### `PreprocessConfig`

```python
@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    interpolation: str = "bilinear"
    output_format: str = "NCHW"
    output_dtype: str = "float32"
    normalize: bool = True
    to_rgb: bool = True
```

### `BuildConfig`

```python
@dataclass
class BuildConfig:
    precision: str = "fp16"
    max_batch_size: int = 32
    workspace_size_gb: float = 4.0
    optimization_level: int = 5
    enable_sparse: bool = False
    enable_timing_cache: bool = True
```

### `InferenceMetrics`

```python
@dataclass
class InferenceMetrics:
    inference_time_ms: float
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    total_time_ms: float = 0.0
    throughput: float = 0.0
    batch_size: int = 1
```
