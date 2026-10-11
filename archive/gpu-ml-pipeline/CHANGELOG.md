# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Support for TensorRT 10.x
- Dynamic batching optimization
- Multi-GPU inference support
- ONNX Runtime backend option

## [1.0.0] - 2024-01-15

### Added
- **Custom CUDA Kernels**
  - GPU-accelerated image resize kernel
  - Fused normalization kernel
  - Combined preprocessing kernel
  - Non-Maximum Suppression (NMS) kernel
  - Python bindings via pybind11
  
- **TensorRT Integration**
  - ONNX to TensorRT engine builder
  - FP32, FP16, and INT8 precision support
  - INT8 calibration with entropy calibrator
  - Dynamic shape support
  - Engine serialization and caching
  
- **Triton Inference Server**
  - Model repository configuration
  - gRPC and HTTP client implementations
  - Ensemble model support
  - Dynamic batching configuration
  
- **Preprocessing Pipeline**
  - GPU-accelerated image preprocessing
  - Batch processing support
  - Multiple input format handling
  - Configurable augmentation
  
- **Benchmarking Suite**
  - Throughput measurement
  - Latency percentile analysis (P50, P95, P99)
  - Memory usage tracking
  - Comparison across precisions
  
- **Infrastructure**
  - NGC-based Docker containers
  - Docker Compose for full stack
  - Prometheus metrics integration
  - Health check endpoints

### Performance
- ResNet50 @ 224x224: 1,250 FPS (INT8, RTX 4090)
- YOLOv8n @ 640x640: 890 FPS (INT8, RTX 4090)
- Preprocessing: 10x faster than CPU baseline

### Security
- Input validation for all inference requests
- Memory bounds checking in CUDA kernels
- Secure model loading

## [0.2.0] - 2024-01-01

### Added
- TensorRT engine builder
- Basic preprocessing pipeline
- Initial benchmarking tools

### Changed
- Improved CUDA kernel performance
- Better error handling in inference

## [0.1.0] - 2023-12-15

### Added
- Initial project structure
- Basic CUDA kernel implementations
- ONNX model loading
- Unit test framework

---

[Unreleased]: https://github.com/saurabh-rai/gpu-ml-pipeline/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/saurabh-rai/gpu-ml-pipeline/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/saurabh-rai/gpu-ml-pipeline/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/saurabh-rai/gpu-ml-pipeline/releases/tag/v0.1.0
