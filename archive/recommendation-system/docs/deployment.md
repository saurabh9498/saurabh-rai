# Deployment Guide

## Overview

This guide covers deploying the Real-Time Personalization Engine in various environments, from local development to production Kubernetes clusters.

## Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- NVIDIA Container Toolkit (for GPU support)
- Kubernetes 1.28+ (for production)
- Helm 3.12+ (for Kubernetes deployment)

## Quick Start (Local Development)

### 1. Clone and Configure

```bash
git clone https://github.com/yourusername/recommendation-engine.git
cd recommendation-engine

# Copy environment template
cp .env.example .env

# Edit configuration
vim .env
```

### 2. Start Services

```bash
# Start all services (CPU mode)
docker-compose up -d

# Start with GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Check status
docker-compose ps
```

### 3. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test recommendation
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "num_recommendations": 5}'
```

---

## Docker Deployment

### Building Images

```bash
# Build API image
docker build -t recommendation-engine:latest -f docker/Dockerfile .

# Build with specific CUDA version
docker build \
  --build-arg CUDA_VERSION=12.2 \
  -t recommendation-engine:cuda12.2 \
  -f docker/Dockerfile .
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment (development/production) | production |
| `APP_PORT` | API server port | 8000 |
| `APP_WORKERS` | Number of uvicorn workers | 4 |
| `REDIS_HOST` | Redis host | localhost |
| `REDIS_PORT` | Redis port | 6379 |
| `TRITON_URL` | Triton server URL | localhost:8001 |
| `MODEL_DIR` | Model directory path | /models |
| `LOG_LEVEL` | Logging level | INFO |

### Docker Compose Services

```yaml
services:
  api:
    image: recommendation-engine:latest
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - TRITON_URL=triton:8001
    depends_on:
      - redis
      - triton

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  triton:
    image: nvcr.io/nvidia/tritonserver:23.10-py3
    ports:
      - "8001:8001"
    volumes:
      - ./models:/models
    command: tritonserver --model-repository=/models
```

---

## Kubernetes Deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Ingress   │────▶│   Service   │────▶│   API Pods  │       │
│  │  (NGINX)    │     │  (LoadBal)  │     │  (HPA: 3-10)│       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                 │                │
│                      ┌──────────────────────────┘                │
│                      │                                           │
│         ┌────────────┴────────────┐                             │
│         │                         │                              │
│         ▼                         ▼                              │
│  ┌─────────────┐          ┌─────────────┐                       │
│  │   Redis     │          │   Triton    │                       │
│  │  Cluster    │          │   Server    │                       │
│  │ (StatefulSet)│         │ (Deployment)│                       │
│  └─────────────┘          └─────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Helm Installation

```bash
# Add Helm repository
helm repo add recommendation-engine https://charts.example.com/recommendation-engine
helm repo update

# Install with default values
helm install reco recommendation-engine/recommendation-engine \
  --namespace recommendation \
  --create-namespace

# Install with custom values
helm install reco recommendation-engine/recommendation-engine \
  --namespace recommendation \
  --create-namespace \
  -f custom-values.yaml
```

### Helm Values Example

```yaml
# custom-values.yaml
replicaCount: 3

image:
  repository: your-registry/recommendation-engine
  tag: v1.0.0
  pullPolicy: IfNotPresent

resources:
  requests:
    cpu: 1000m
    memory: 2Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 4000m
    memory: 8Gi
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

redis:
  enabled: true
  cluster:
    enabled: true
    nodes: 6
  persistence:
    enabled: true
    size: 10Gi

triton:
  enabled: true
  replicaCount: 2
  resources:
    requests:
      nvidia.com/gpu: 1

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: recommendations.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: recommendations-tls
      hosts:
        - recommendations.example.com
```

### Manual Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace recommendation

# Apply configurations
kubectl apply -f k8s/configmap.yaml -n recommendation
kubectl apply -f k8s/secrets.yaml -n recommendation
kubectl apply -f k8s/redis.yaml -n recommendation
kubectl apply -f k8s/triton.yaml -n recommendation
kubectl apply -f k8s/api-deployment.yaml -n recommendation
kubectl apply -f k8s/api-service.yaml -n recommendation
kubectl apply -f k8s/ingress.yaml -n recommendation
kubectl apply -f k8s/hpa.yaml -n recommendation
```

---

## Production Checklist

### Security

- [ ] Enable TLS/HTTPS
- [ ] Configure API authentication
- [ ] Set up network policies
- [ ] Enable Pod Security Standards
- [ ] Rotate secrets regularly
- [ ] Enable audit logging

### High Availability

- [ ] Deploy minimum 3 replicas
- [ ] Configure pod anti-affinity
- [ ] Set up Redis cluster (6 nodes)
- [ ] Enable health checks
- [ ] Configure PodDisruptionBudget

### Monitoring

- [ ] Deploy Prometheus ServiceMonitor
- [ ] Configure Grafana dashboards
- [ ] Set up alerting rules
- [ ] Enable distributed tracing
- [ ] Configure log aggregation

### Performance

- [ ] Enable GPU scheduling
- [ ] Configure resource limits
- [ ] Set up HPA
- [ ] Enable request batching
- [ ] Configure connection pooling

---

## Monitoring Setup

### Prometheus Metrics

Key metrics to monitor:

```yaml
# ServiceMonitor configuration
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: recommendation-engine
spec:
  selector:
    matchLabels:
      app: recommendation-engine
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
```

### Grafana Dashboard

Import the provided dashboard:

```bash
# Dashboard ID: 12345
# Or import from file
kubectl create configmap grafana-dashboard \
  --from-file=dashboards/recommendation-dashboard.json \
  -n monitoring
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `recommendation_latency_p99` | P99 latency | > 50ms |
| `recommendation_error_rate` | Error percentage | > 1% |
| `recommendation_qps` | Requests per second | < 100 (low), > 10000 (high) |
| `feature_store_cache_hit_rate` | Cache hit ratio | < 90% |
| `triton_queue_time_ms` | Model queue time | > 10ms |

### Alerting Rules

```yaml
groups:
  - name: recommendation-engine
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, recommendation_latency_seconds_bucket) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High recommendation latency"

      - alert: HighErrorRate
        expr: rate(recommendation_errors_total[5m]) / rate(recommendation_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in recommendations"
```

---

## Scaling Guidelines

### Horizontal Scaling

```yaml
# HPA Configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: recommendation-engine
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: recommendation-engine
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: recommendation_qps
        target:
          type: AverageValue
          averageValue: "5000"
```

### Capacity Planning

| Traffic Level | API Pods | Redis Nodes | Triton Instances |
|---------------|----------|-------------|------------------|
| Low (<1K QPS) | 3 | 3 | 1 |
| Medium (1-10K QPS) | 5-10 | 6 | 2 |
| High (10-50K QPS) | 10-20 | 6 | 4-8 |
| Very High (>50K QPS) | 20+ | 9+ | 8+ |

---

## Troubleshooting

### Common Issues

#### 1. High Latency

```bash
# Check Triton queue
curl localhost:8001/v2/models/dlrm/stats

# Check Redis latency
redis-cli --latency

# Profile endpoint
python scripts/benchmark.py endpoint --url http://localhost:8000/recommend
```

#### 2. OOM Errors

```bash
# Check memory usage
kubectl top pods -n recommendation

# Increase limits
kubectl set resources deployment/recommendation-engine \
  --limits=memory=8Gi -n recommendation
```

#### 3. Connection Errors

```bash
# Check Redis connectivity
kubectl exec -it deployment/recommendation-engine \
  -- redis-cli -h redis ping

# Check Triton health
kubectl exec -it deployment/recommendation-engine \
  -- curl triton:8000/v2/health/ready
```

### Logs

```bash
# API logs
kubectl logs -f deployment/recommendation-engine -n recommendation

# Triton logs
kubectl logs -f deployment/triton -n recommendation

# All logs with stern
stern recommendation -n recommendation
```

---

## Rollback Procedures

### Application Rollback

```bash
# View history
kubectl rollout history deployment/recommendation-engine -n recommendation

# Rollback to previous
kubectl rollout undo deployment/recommendation-engine -n recommendation

# Rollback to specific revision
kubectl rollout undo deployment/recommendation-engine \
  --to-revision=2 -n recommendation
```

### Model Rollback

```bash
# Update Triton model repository
kubectl exec -it deployment/triton -- \
  curl -X POST localhost:8000/v2/repository/models/dlrm/unload

# Load previous version
kubectl exec -it deployment/triton -- \
  curl -X POST localhost:8000/v2/repository/models/dlrm/load
```

---

## Backup & Recovery

### Redis Backup

```bash
# Create RDB snapshot
kubectl exec -it redis-0 -n recommendation -- redis-cli BGSAVE

# Copy backup
kubectl cp recommendation/redis-0:/data/dump.rdb ./backups/redis-dump.rdb
```

### Model Backup

```bash
# Backup to S3
aws s3 sync ./models s3://your-bucket/models/$(date +%Y%m%d)/

# Restore from S3
aws s3 sync s3://your-bucket/models/20240115/ ./models/
```
