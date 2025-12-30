# Quick Start Guide

Get the Retail Vision Analytics system running in under 10 minutes.

## Prerequisites

- **Hardware**: NVIDIA GPU with 8GB+ VRAM (RTX 3060+) or Jetson Orin
- **Software**: Docker 24+, NVIDIA Container Toolkit, CUDA 12.0+
- **Network**: Access to camera streams (RTSP/RTMP)

## Option 1: Docker Compose (Recommended)

### 1. Clone and Configure

```bash
# Clone repository
git clone https://github.com/yourusername/retail-vision-analytics.git
cd retail-vision-analytics

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 2. Configure Camera Sources

Edit `configs/app_config.yaml`:

```yaml
sources:
  - id: "cam-entrance-1"
    name: "Entrance Camera"
    uri: "rtsp://YOUR_CAMERA_IP:554/stream1"
    enabled: true
```

### 3. Start Services

```bash
# Pull and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f retail-vision
```

### 4. Access Dashboard

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Option 2: Local Development

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Install DeepStream Python bindings
pip install pyds-ext
```

### 2. Download Models

```bash
# Download pre-trained YOLOv8 model
./scripts/download_models.sh

# Or convert your own
python -m src.edge.tensorrt_engine convert \
    --input models/yolov8n.onnx \
    --output models/yolov8n_retail_fp16.engine \
    --precision fp16
```

### 3. Run Application

```bash
# Start API server
uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start pipeline
python -m src.edge.deepstream_app --config configs/app_config.yaml
```

## Option 3: Jetson Deployment

### 1. Flash JetPack

Ensure JetPack 5.1.2+ is installed on your Jetson device.

### 2. Deploy with Docker

```bash
# On Jetson device
docker pull yourusername/retail-vision:jetson-latest

# Run with GPU access
docker run -d \
    --runtime nvidia \
    --network host \
    -v /path/to/config:/opt/configs \
    -v /path/to/models:/opt/models \
    yourusername/retail-vision:jetson-latest
```

## Verify Installation

### Check API Health

```bash
curl http://localhost:8000/health
# {"status": "healthy", "timestamp": "..."}
```

### List Camera Streams

```bash
curl http://localhost:8000/api/v1/streams
```

### Get Analytics Summary

```bash
curl "http://localhost:8000/api/v1/analytics/summary?time_range=24h"
```

### WebSocket Test

```python
import asyncio
import websockets

async def test_ws():
    async with websockets.connect('ws://localhost:8000/ws/detections') as ws:
        for _ in range(5):
            msg = await ws.recv()
            print(msg)

asyncio.run(test_ws())
```

## Configuration Quick Reference

### Essential Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `pipeline.inference.confidence_threshold` | Detection confidence | 0.5 |
| `pipeline.tracker.type` | Tracker algorithm | nvdcf |
| `analytics.queue.thresholds.queue_length_warning` | Queue alert threshold | 5 |
| `storage.redis.host` | Redis server | localhost |

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEVICE_ID` | Unique device identifier | Yes |
| `STORE_ID` | Store identifier | Yes |
| `CLOUD_API_KEY` | Cloud sync API key | For sync |
| `REDIS_PASSWORD` | Redis password | If secured |

## Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check container runtime
docker info | grep -i runtime
```

**Camera connection failed:**
```bash
# Test RTSP stream
ffprobe rtsp://camera_ip:554/stream

# Check network
ping camera_ip
```

**High latency:**
```bash
# Check GPU utilization
nvidia-smi -l 1

# Reduce batch size in config
# pipeline.inference.batch_size: 8
```

### Logs Location

- **Container**: `docker-compose logs retail-vision`
- **Local**: `/var/log/retail-vision/app.log`
- **Jetson**: `journalctl -u retail-vision`

## Next Steps

1. **Configure Zones**: Define store zones in `app_config.yaml`
2. **Set Up Alerts**: Configure alert thresholds and webhooks
3. **Connect Dashboard**: Import Grafana dashboards from `dashboards/`
4. **Enable Cloud Sync**: Configure edge-cloud synchronization

## Support

- **Documentation**: [docs/](./docs/)
- **API Reference**: [API_REFERENCE.md](./docs/API_REFERENCE.md)
- **Issues**: GitHub Issues
