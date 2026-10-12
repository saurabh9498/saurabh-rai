# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Add person re-identification across cameras
- Implement shelf gap detection for inventory monitoring
- Add OCR for price tag reading
- Support for PTZ camera control
- Mobile app for real-time alerts

## [1.0.0] - 2024-12-30

### Added
- **Vision Module**
  - YOLOv8-based object detection with TensorRT optimization
  - ByteTrack multi-object tracker with Kalman filtering
  - Support for person, shopping_cart, basket, product, shelf, price_tag, employee classes
  - FP16 and INT8 precision inference
  - Dynamic batching for multi-stream processing

- **Analytics Module**
  - Customer journey tracking with zone-based path reconstruction
  - Conversion funnel analysis with entry/exit point detection
  - Dwell time calculation per zone
  - Queue monitoring with wait time estimation
  - Queue abandonment detection
  - Staffing recommendations based on queue metrics
  - Traffic heatmap generation with hotspot detection
  - Temporal pattern analysis

- **Edge Deployment**
  - TensorRT engine builder with INT8 calibration support
  - DeepStream 6.3+ pipeline for multi-stream processing
  - Support for up to 64 concurrent streams (RTX 4090) / 16 streams (Jetson Orin)
  - Jetson device management (Orin, Xavier, Nano)
  - Power mode and thermal management
  - Edge-cloud synchronization with offline buffering
  - Priority-based sync queue

- **REST API**
  - FastAPI-based REST API with OpenAPI documentation
  - Camera stream management endpoints
  - Analytics retrieval endpoints (journeys, queues, heatmaps)
  - Alert management with severity levels
  - WebSocket streaming for real-time detections
  - Health and metrics endpoints

- **Infrastructure**
  - Docker support with multi-stage builds
  - Jetson-optimized Docker image
  - Docker Compose for full stack deployment
  - Redis integration for real-time events
  - Kafka support for high-throughput streaming
  - TimescaleDB for time-series analytics
  - MinIO for video clip storage

- **Documentation**
  - Comprehensive README with architecture diagrams
  - Quick start guide for 10-minute setup
  - API reference documentation
  - System architecture documentation
  - Deployment guide

### Performance
- Detection latency: <5ms per frame (FP16)
- End-to-end latency: <100ms
- Throughput: 400+ FPS per node
- Multi-stream: 32 cameras @ 30 FPS

---

## [0.3.0] - 2024-12-20

### Added
- Edge-cloud synchronization manager
- SQLite buffer for offline operation
- Automatic retry with exponential backoff
- Health reporting heartbeat

### Changed
- Improved tracker performance with appearance features
- Optimized heatmap memory usage

### Fixed
- Memory leak in multi-stream pipeline
- Track ID persistence across frame drops

---

## [0.2.0] - 2024-12-10

### Added
- Queue monitoring with wait time prediction
- Alert engine with webhook notifications
- Heatmap visualization with flow fields
- Jetson power mode management

### Changed
- Upgraded to DeepStream 6.3
- Improved zone detection accuracy

### Fixed
- RTSP reconnection handling
- GPU memory fragmentation on long runs

---

## [0.1.0] - 2024-12-01

### Added
- Initial project structure
- YOLOv8 detection integration
- ByteTrack tracker implementation
- Basic FastAPI endpoints
- Docker development environment
- Unit test framework

---

[Unreleased]: https://github.com/yourusername/retail-vision-analytics/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/retail-vision-analytics/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/yourusername/retail-vision-analytics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yourusername/retail-vision-analytics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/retail-vision-analytics/releases/tag/v0.1.0
