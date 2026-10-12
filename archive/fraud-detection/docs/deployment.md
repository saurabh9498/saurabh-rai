# Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 20 GB | 50+ GB SSD |
| Python | 3.10+ | 3.11+ |

### Required Services

- **Redis 7.0+** - Feature store
- **Apache Kafka 3.5+** - Stream processing
- **PostgreSQL 15+** (optional) - Persistent storage

---

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/saurabh-rai/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start Redis (using Docker)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

```bash
# .env file
REDIS_URL=redis://localhost:6379
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
LOG_LEVEL=INFO
MODEL_PATH=./models
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires Docker)
docker-compose -f docker/docker-compose.yml up -d redis kafka
pytest tests/integration/ -v

# Coverage report
pytest --cov=src --cov-report=html
```

---

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t fraud-detection:latest -f docker/Dockerfile .

# Run container
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  fraud-detection:latest
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# With monitoring
docker-compose -f docker/docker-compose.yml --profile monitoring up -d

# With stream workers
docker-compose -f docker/docker-compose.yml --profile workers up -d

# View logs
docker-compose logs -f api

# Scale API instances
docker-compose up -d --scale api=3
```

### Service Endpoints

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| Redis | 6379 | redis://localhost:6379 |
| Kafka | 9092 | localhost:9092 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.25+
- kubectl configured
- Helm 3.0+ (optional)

### Manifests

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection
  labels:
    app: fraud-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detection
  template:
    metadata:
      labels:
        app: fraud-detection
    spec:
      containers:
      - name: api
        image: fraud-detection:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: fraud-config
              key: redis_url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

#### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection
spec:
  selector:
    app: fraud-detection
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace fraud-detection

# Apply manifests
kubectl apply -f k8s/ -n fraud-detection

# Check status
kubectl get pods -n fraud-detection
kubectl get svc -n fraud-detection

# View logs
kubectl logs -f deployment/fraud-detection -n fraud-detection
```

---

## Cloud Deployments

### AWS ECS

```bash
# Create ECR repository
aws ecr create-repository --repository-name fraud-detection

# Build and push image
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker build -t fraud-detection .
docker tag fraud-detection:latest $ECR_URI/fraud-detection:latest
docker push $ECR_URI/fraud-detection:latest

# Deploy with ECS (using Fargate)
aws ecs create-cluster --cluster-name fraud-detection
aws ecs create-service \
  --cluster fraud-detection \
  --service-name fraud-api \
  --task-definition fraud-detection:1 \
  --desired-count 3 \
  --launch-type FARGATE
```

### GCP Cloud Run

```bash
# Build with Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/fraud-detection

# Deploy to Cloud Run
gcloud run deploy fraud-detection \
  --image gcr.io/$PROJECT_ID/fraud-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10
```

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fraud-detection'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `fraud_request_latency_seconds` | P99 latency | > 20ms |
| `fraud_requests_total` | Request rate | < 100/min |
| `fraud_detection_score` | Score distribution | - |
| `fraud_detection_decisions_total` | Decision breakdown | - |
| `fraud_detection_feature_drift` | Feature drift | > 0.1 |

### Grafana Dashboard

Import the dashboard from `configs/grafana-dashboard.json` or create panels for:

1. Request rate and latency
2. Fraud score distribution
3. Decision breakdown (approve/review/decline)
4. Model drift scores
5. Feature store latency

---

## Troubleshooting

### Common Issues

#### API Not Starting

```bash
# Check logs
docker logs fraud-api

# Verify Redis connection
redis-cli ping

# Check port availability
lsof -i :8000
```

#### High Latency

1. Check Redis connection pooling
2. Verify model is loaded in memory
3. Check for memory pressure
4. Review batch sizes

#### Model Drift Alerts

1. Check feature distributions
2. Review recent data quality
3. Consider model retraining
4. Verify feature store freshness

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Readiness
curl http://localhost:8000/ready

# Metrics
curl http://localhost:8000/metrics
```

---

## Performance Tuning

### API Workers

```bash
# Increase uvicorn workers
uvicorn src.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Redis Optimization

```bash
# Increase connection pool
REDIS_POOL_SIZE=20
REDIS_POOL_TIMEOUT=10
```

### Kafka Tuning

```bash
# Increase consumer parallelism
KAFKA_CONSUMER_THREADS=8
KAFKA_MAX_POLL_RECORDS=1000
```
