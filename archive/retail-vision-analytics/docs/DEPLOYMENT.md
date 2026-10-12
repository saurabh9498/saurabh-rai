# Deployment Guide

Complete guide for deploying Retail Vision Analytics in production environments.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Cloud Deployment](#cloud-deployment)
- [Edge Deployment (Jetson)](#edge-deployment-jetson)
- [Hybrid Deployment](#hybrid-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Deployment Options

| Option | Best For | Hardware | Latency |
|--------|----------|----------|---------|
| **Cloud** | Centralized processing, multiple stores | GPU VMs | 100-500ms |
| **Edge** | Single store, low latency | Jetson Orin | <50ms |
| **Hybrid** | Multi-store with local processing | Both | <50ms local |

## Cloud Deployment

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with 8GB+ VRAM
- NVIDIA Container Toolkit
- 16GB+ RAM
- 100GB+ storage

### AWS Deployment

#### 1. Launch EC2 Instance

```bash
# Recommended: g4dn.xlarge (T4 GPU) or g5.xlarge (A10G GPU)
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \  # Deep Learning AMI
    --instance-type g4dn.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxx \
    --subnet-id subnet-xxx
```

#### 2. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 3. Deploy Application

```bash
# Clone repository
git clone https://github.com/yourusername/retail-vision-analytics.git
cd retail-vision-analytics

# Configure environment
cp .env.example .env
nano .env  # Edit with your settings

# Start services
docker-compose -f docker/docker-compose.yml up -d
```

### GCP Deployment

```bash
# Create GPU VM
gcloud compute instances create retail-vision \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=common-cu121 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retail-vision
spec:
  replicas: 1
  selector:
    matchLabels:
      app: retail-vision
  template:
    metadata:
      labels:
        app: retail-vision
    spec:
      containers:
      - name: retail-vision
        image: retail-vision:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        ports:
        - containerPort: 8000
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        volumeMounts:
        - name: config
          mountPath: /opt/configs
      volumes:
      - name: config
        configMap:
          name: retail-vision-config
```

## Edge Deployment (Jetson)

### Supported Devices

| Device | Streams | FPS | Power | Price |
|--------|---------|-----|-------|-------|
| Jetson Orin Nano | 4-8 | 120 | 7-15W | $499 |
| Jetson Orin NX | 8-16 | 250 | 10-25W | $699 |
| Jetson AGX Orin | 16-32 | 500 | 15-60W | $1999 |

### Jetson Setup

#### 1. Flash JetPack

```bash
# Use NVIDIA SDK Manager to flash JetPack 5.1.2+
# Or download pre-flashed SD card image
```

#### 2. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io nvidia-container-toolkit
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER
```

#### 3. Deploy

```bash
# Pull Jetson image
docker pull yourusername/retail-vision:jetson-latest

# Run container
docker run -d \
    --name retail-vision \
    --runtime nvidia \
    --network host \
    -v /path/to/config:/opt/configs \
    -v /path/to/models:/opt/models \
    -v /var/lib/retail-vision:/var/lib/retail-vision \
    --restart unless-stopped \
    yourusername/retail-vision:jetson-latest
```

#### 4. Optimize for Production

```bash
# Set power mode (MAXN for maximum performance)
sudo nvpmodel -m 0

# Enable jetson_clocks
sudo jetson_clocks

# Set fan to maximum
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

### Jetson as systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/retail-vision.service << EOF
[Unit]
Description=Retail Vision Analytics
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker stop retail-vision
ExecStartPre=-/usr/bin/docker rm retail-vision
ExecStart=/usr/bin/docker run --rm --name retail-vision \
    --runtime nvidia --network host \
    -v /opt/retail-vision/configs:/opt/configs \
    -v /opt/retail-vision/models:/opt/models \
    retail-vision:jetson-latest
ExecStop=/usr/bin/docker stop retail-vision
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable retail-vision
sudo systemctl start retail-vision
```

## Hybrid Deployment

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Store 1       │     │   Store 2       │     │   Store N       │
│  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────┐  │
│  │  Jetson   │  │     │  │  Jetson   │  │     │  │  Jetson   │  │
│  │  (Edge)   │  │     │  │  (Edge)   │  │     │  │  (Edge)   │  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │     │  └─────┬─────┘  │
└────────┼────────┘     └────────┼────────┘     └────────┼────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │      Cloud Platform      │
                    │  ┌────────────────────┐ │
                    │  │  Analytics Server  │ │
                    │  │  (Aggregation)     │ │
                    │  └────────────────────┘ │
                    │  ┌────────────────────┐ │
                    │  │  Dashboard         │ │
                    │  │  (Visualization)   │ │
                    │  └────────────────────┘ │
                    └─────────────────────────┘
```

### Edge Configuration

```yaml
# configs/app_config.yaml (Edge)
edge:
  device_id: "store-001-edge-01"
  store_id: "store-001"
  
  sync:
    enabled: true
    cloud_host: "api.retailvision.cloud"
    upload_interval_seconds: 30
    compress_uploads: true
```

### Cloud Configuration

```yaml
# configs/app_config.yaml (Cloud)
cloud:
  mode: "aggregator"
  
  stores:
    - id: "store-001"
      name: "Downtown Store"
    - id: "store-002"
      name: "Mall Store"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVICE_ID` | Unique device identifier | Required |
| `STORE_ID` | Store identifier | Required |
| `API_PORT` | API server port | 8000 |
| `REDIS_HOST` | Redis server host | localhost |
| `CLOUD_API_KEY` | Cloud API key | Required for sync |

### Camera Configuration

```yaml
sources:
  - id: "cam-entrance-1"
    name: "Entrance Camera"
    uri: "rtsp://admin:password@192.168.1.10:554/stream1"
    protocol: "rtsp"
    width: 1920
    height: 1080
    fps: 30
    enabled: true
```

## Monitoring

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check detailed status
curl http://localhost:8000/api/v1/system/health
```

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'retail-vision'
    static_configs:
      - targets: ['localhost:9090']
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `inference_latency_ms` | Detection latency | >20ms |
| `fps_total` | Total throughput | <100 FPS |
| `gpu_util_percent` | GPU utilization | >95% |
| `queue_length` | Pending uploads | >1000 |

## Scaling

### Horizontal Scaling

```bash
# Scale API servers
docker-compose up -d --scale api=4
```

### Load Balancing

```nginx
# nginx.conf
upstream retail_vision {
    server api1:8000;
    server api2:8000;
    server api3:8000;
    server api4:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://retail_vision;
    }
}
```

## Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check container runtime
docker info | grep -i runtime

# Restart Docker
sudo systemctl restart docker
```

#### High Latency

```bash
# Check GPU utilization
nvidia-smi -l 1

# Reduce batch size in config
# pipeline.inference.batch_size: 8
```

#### Camera Connection Failed

```bash
# Test RTSP stream
ffprobe rtsp://camera_ip:554/stream

# Check network
ping camera_ip
```

### Logs

```bash
# Container logs
docker logs retail-vision -f

# System logs
journalctl -u retail-vision -f

# Jetson thermal
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

### Support

- **Issues**: GitHub Issues
- **Documentation**: [docs/](../docs/)
- **Community**: Discord/Slack channel
