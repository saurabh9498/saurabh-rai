# System Architecture

## Overview

Retail Vision Analytics is a distributed video analytics system designed for real-time customer behavior analysis in retail environments. The architecture prioritizes low-latency processing, horizontal scalability, and edge-cloud hybrid deployment.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETAIL STORE ENVIRONMENT                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│   │ Camera 1 │  │ Camera 2 │  │ Camera 3 │  │ Camera 4 │  │ Camera N │     │
│   │  (RTSP)  │  │  (RTSP)  │  │  (RTSP)  │  │  (RTSP)  │  │  (RTSP)  │     │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│        │             │             │             │             │            │
│        └─────────────┴──────┬──────┴─────────────┴─────────────┘            │
│                             │                                                │
│                             ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     EDGE COMPUTE NODE                                │   │
│   │                  (Jetson Orin / RTX GPU)                            │   │
│   │  ┌───────────────────────────────────────────────────────────────┐  │   │
│   │  │                    VIDEO INGESTION LAYER                       │  │   │
│   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │  │   │
│   │  │  │ NVDEC   │  │ Stream  │  │ Frame   │  │ Protocol Handler│   │  │   │
│   │  │  │ Decoder │→ │ Demux   │→ │ Buffer  │→ │ (RTSP/RTMP/USB) │   │  │   │
│   │  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │  │   │
│   │  └───────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  ┌───────────────────────────────────────────────────────────────┐  │   │
│   │  │                    DEEPSTREAM PIPELINE                         │  │   │
│   │  │                                                                │  │   │
│   │  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐  │  │   │
│   │  │  │ Stream   │    │ TensorRT │    │ Object   │    │ Analytics│  │  │   │
│   │  │  │  Muxer   │ →  │ Inference│ →  │ Tracker  │ →  │  Probe   │  │  │   │
│   │  │  │ (32 src) │    │ (YOLOv8) │    │(ByteTrack)│   │          │  │  │   │
│   │  │  └──────────┘    └──────────┘    └──────────┘    └─────────┘  │  │   │
│   │  │       │               │               │               │        │  │   │
│   │  │       │          GPU Memory      Kalman Filter    Metadata     │  │   │
│   │  │       │          ~2GB VRAM       Track States     Extraction   │  │   │
│   │  └───────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  ┌───────────────────────────────────────────────────────────────┐  │   │
│   │  │                   ANALYTICS ENGINE                             │  │   │
│   │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │   │
│   │  │  │   Customer   │  │    Queue     │  │      Heatmap         │ │  │   │
│   │  │  │   Journey    │  │   Monitor    │  │     Generator        │ │  │   │
│   │  │  │   Tracker    │  │              │  │                      │ │  │   │
│   │  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │  │   │
│   │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │   │
│   │  │  │    Zone      │  │   Dwell      │  │    Conversion        │ │  │   │
│   │  │  │  Tracking    │  │   Time       │  │     Funnel           │ │  │   │
│   │  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │  │   │
│   │  └───────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  ┌───────────────────────────────────────────────────────────────┐  │   │
│   │  │                    EVENT SYSTEM                                │  │   │
│   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │  │   │
│   │  │  │  Redis  │  │  Kafka  │  │  Alert  │  │  Sync Manager   │   │  │   │
│   │  │  │ Streams │  │ Producer│  │ Engine  │  │  (Edge↔Cloud)   │   │  │   │
│   │  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │  │   │
│   │  └───────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
                               │ HTTPS/WSS
                               │ (Compressed, Batched)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                         API GATEWAY                                │     │
│   │              (Load Balancer, Auth, Rate Limiting)                  │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│          ┌───────────────────┼───────────────────┐                          │
│          ▼                   ▼                   ▼                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │  REST API   │    │  WebSocket  │    │   GraphQL   │                     │
│   │   Server    │    │   Server    │    │   Server    │                     │
│   └─────────────┘    └─────────────┘    └─────────────┘                     │
│          │                   │                   │                          │
│          └───────────────────┴───────────────────┘                          │
│                              │                                               │
│                              ▼                                               │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                      DATA LAYER                                    │     │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │     │
│   │  │ TimescaleDB │  │   Redis     │  │ ClickHouse  │  │  MinIO   │  │     │
│   │  │ (Time-Series)│ │  (Cache)    │  │ (Analytics) │  │ (Objects)│  │     │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                   VISUALIZATION LAYER                              │     │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │     │
│   │  │   Grafana   │  │  Dashboard  │  │    Mobile App           │   │     │
│   │  │ (Monitoring)│  │   (React)   │  │    (React Native)       │   │     │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Video Ingestion Layer

**Purpose**: Capture and decode video streams from multiple cameras.

| Component | Technology | Function |
|-----------|------------|----------|
| Protocol Handler | GStreamer | RTSP/RTMP/USB/CSI support |
| NVDEC | Hardware Decoder | GPU-accelerated H.264/H.265 decode |
| Stream Demux | DeepStream | Multi-stream batching |
| Frame Buffer | CUDA Memory | Zero-copy frame sharing |

**Performance**:
- Jetson Orin: 16 streams @ 1080p30
- RTX 4090: 64 streams @ 1080p30

### 2. DeepStream Pipeline

**Purpose**: Real-time AI inference and object tracking.

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│nvstreammux│→│ nvinfer │→│nvtracker│→│nvdsanalyt│→│ nvdsosd │
│(Batching)│  │(YOLOv8) │  │(ByteTrack)│ │(Metadata)│  │(Display)│
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

**Inference Engine**:
- Model: YOLOv8n (retail fine-tuned)
- Precision: FP16 (TensorRT)
- Batch Size: 16
- Latency: <5ms per frame

**Object Tracker**:
- Algorithm: ByteTrack
- Re-ID: Appearance features + IoU
- Track Management: Kalman filter prediction

### 3. Analytics Engine

**Customer Journey Tracker**:
- Zone-based path reconstruction
- Entry/exit point detection
- Conversion funnel analysis
- Dwell time calculation

**Queue Monitor**:
- Line detection via person positions
- Wait time estimation
- Service rate tracking
- Abandonment detection

**Heatmap Generator**:
- Grid-based accumulation
- Temporal decay
- Hotspot identification
- Flow field calculation

### 4. Event System

**Redis Streams**:
- Real-time detection events
- Analytics aggregations
- Alert notifications

**Kafka (Optional)**:
- High-throughput event streaming
- Cross-datacenter replication
- Event sourcing

### 5. Edge-Cloud Sync

**Upload Pipeline**:
```
Local Buffer → Compression → Batch → HTTPS → Cloud API
   (SQLite)     (gzip)      (100 items)
```

**Download Pipeline**:
```
Cloud API → Model Updates → Local Cache → Hot Reload
            Config Sync      (versioned)
```

## Data Flow

### Real-Time Path (< 100ms)

```
Camera → NVDEC → Inference → Tracker → Analytics → Redis → WebSocket → Dashboard
                   5ms        2ms        1ms        1ms      10ms
```

### Analytics Path (30s batch)

```
Detection Events → Aggregation → TimescaleDB → Query API → Dashboard
                    (30s window)   (compressed)
```

### Alert Path (< 1s)

```
Threshold Breach → Alert Engine → Deduplication → Webhook → Slack/Email
                                    (cooldown)
```

## Scalability

### Horizontal Scaling

| Layer | Scaling Strategy |
|-------|------------------|
| Edge Nodes | Add more Jetson devices per store |
| API Servers | Kubernetes HPA based on CPU/memory |
| Databases | TimescaleDB chunking, ClickHouse sharding |
| Event Streams | Kafka partitioning by store_id |

### Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Detection Latency | <10ms | 5ms |
| End-to-End Latency | <100ms | 80ms |
| Throughput | 500 FPS/node | 420 FPS |
| Availability | 99.9% | 99.95% |

## Security

### Network Security
- TLS 1.3 for all external connections
- mTLS between edge and cloud
- VPN for camera network isolation

### Authentication
- API Key for edge devices
- JWT for user authentication
- RBAC for multi-tenant access

### Data Protection
- Encryption at rest (AES-256)
- PII anonymization options
- Configurable retention policies

## Deployment Models

### Single Store (Standalone)
- 1 Jetson Orin NX
- Local Redis + SQLite
- No cloud dependency

### Multi-Store (Centralized)
- N edge devices
- Central cloud platform
- Cross-store analytics

### Hybrid (Recommended)
- Edge processing + cloud analytics
- Local buffering for offline resilience
- Centralized model management
