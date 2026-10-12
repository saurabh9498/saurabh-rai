# Deployment Guide

This guide covers deploying the Conversational AI Assistant to production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Configuration](#configuration)
- [Scaling](#scaling)
- [Monitoring](#monitoring)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| GPU | - | NVIDIA T4 or better |
| Storage | 20 GB SSD | 50 GB SSD |
| Network | 100 Mbps | 1 Gbps |

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.25+ (for K8s deployment)
- NVIDIA Container Toolkit (for GPU support)

---

## Docker Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/conversational-ai.git
cd conversational-ai

# Configure environment
cp .env.example .env
# Edit .env with production settings

# Build and start
docker compose -f docker/docker-compose.yml up -d

# Verify
curl http://localhost:8000/health
```

### Production Docker Compose

```yaml
# docker/docker-compose.prod.yml
version: "3.8"

services:
  api:
    image: conversational-ai:${VERSION:-latest}
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - LOG_LEVEL=info
      - REDIS_URL=redis://redis:6379
      - ASR_MODEL_SIZE=base
      - USE_GPU=${USE_GPU:-false}
    volumes:
      - model-cache:/app/models
    depends_on:
      redis:
        condition: service_healthy
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "2"
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - api
    restart: unless-stopped

volumes:
  model-cache:
  redis-data:

networks:
  default:
    driver: bridge
```

### GPU Support

```yaml
# docker/docker-compose.gpu.yml
services:
  api:
    build:
      dockerfile: docker/Dockerfile.gpu
    environment:
      - USE_GPU=true
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with GPU:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d
```

---

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: conversational-ai

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: conversational-ai-config
  namespace: conversational-ai
data:
  ENV: "production"
  LOG_LEVEL: "info"
  ASR_MODEL_SIZE: "base"
  REDIS_URL: "redis://redis-service:6379"
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: conversational-ai
  namespace: conversational-ai
  labels:
    app: conversational-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: conversational-ai
  template:
    metadata:
      labels:
        app: conversational-ai
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: api
          image: conversational-ai:1.0.0
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: conversational-ai-config
            - secretRef:
                name: conversational-ai-secrets
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
          volumeMounts:
            - name: model-cache
              mountPath: /app/models
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: conversational-ai
                topologyKey: kubernetes.io/hostname
```

### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: conversational-ai-service
  namespace: conversational-ai
spec:
  selector:
    app: conversational-ai
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: conversational-ai-ingress
  namespace: conversational-ai
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/websocket-services: "conversational-ai-service"
spec:
  tls:
    - hosts:
        - api.example.com
      secretName: conversational-ai-tls
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: conversational-ai-service
                port:
                  number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: conversational-ai-hpa
  namespace: conversational-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: conversational-ai
  minReplicas: 3
  maxReplicas: 10
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
```

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Verify deployment
kubectl get pods -n conversational-ai
kubectl get svc -n conversational-ai
kubectl get ingress -n conversational-ai
```

---

## Cloud Deployments

### AWS ECS

```bash
# Create ECR repository
aws ecr create-repository --repository-name conversational-ai

# Push image
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
docker tag conversational-ai:latest $ECR_REGISTRY/conversational-ai:latest
docker push $ECR_REGISTRY/conversational-ai:latest

# Deploy with Copilot
copilot init --app conversational-ai --type "Load Balanced Web Service"
copilot deploy
```

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/$PROJECT_ID/conversational-ai

# Deploy
gcloud run deploy conversational-ai \
  --image gcr.io/$PROJECT_ID/conversational-ai \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --min-instances 1 \
  --max-instances 10 \
  --allow-unauthenticated
```

### Azure Container Apps

```bash
# Create container app environment
az containerapp env create \
  --name conversational-ai-env \
  --resource-group $RESOURCE_GROUP \
  --location eastus

# Deploy
az containerapp create \
  --name conversational-ai \
  --resource-group $RESOURCE_GROUP \
  --environment conversational-ai-env \
  --image conversational-ai:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 4 \
  --memory 8Gi \
  --min-replicas 1 \
  --max-replicas 10
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (development/production) | development |
| `LOG_LEVEL` | Logging level | info |
| `API_HOST` | API host | 0.0.0.0 |
| `API_PORT` | API port | 8000 |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379 |
| `ASR_MODEL_SIZE` | Whisper model size | base |
| `USE_GPU` | Enable GPU acceleration | false |
| `TTS_MODEL` | TTS model name | tacotron2-DDC |
| `MAX_AUDIO_LENGTH` | Max audio length (seconds) | 30 |
| `SESSION_TTL` | Session timeout (seconds) | 3600 |

### Secrets

Store sensitive values in secrets:

```bash
# Kubernetes
kubectl create secret generic conversational-ai-secrets \
  --from-literal=API_KEY=your-api-key \
  --from-literal=JWT_SECRET=your-jwt-secret \
  -n conversational-ai
```

---

## Scaling

### Scaling Guidelines

| Concurrent Users | Pods | CPU/Pod | Memory/Pod |
|-----------------|------|---------|------------|
| 1-50 | 2 | 2 | 4Gi |
| 50-200 | 4 | 4 | 8Gi |
| 200-500 | 8 | 4 | 8Gi |
| 500+ | 10+ | 4 | 8Gi |

### Redis Scaling

For high load, use Redis Cluster:

```yaml
# k8s/redis-cluster.yaml
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: redis-cluster
spec:
  clusterSize: 3
  clusterVersion: v7
  persistenceEnabled: true
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
```

---

## Monitoring

### Prometheus Metrics

Available at `/metrics`:

```
# Request metrics
conversational_ai_requests_total{method, endpoint, status}
conversational_ai_request_duration_seconds{endpoint}

# Component metrics
conversational_ai_asr_duration_seconds
conversational_ai_nlu_duration_seconds
conversational_ai_tts_duration_seconds

# System metrics
conversational_ai_active_sessions
conversational_ai_websocket_connections
```

### Grafana Dashboard

Import the provided dashboard:

```bash
kubectl apply -f k8s/grafana-dashboard.yaml
```

### Alerting Rules

```yaml
# prometheus-rules.yaml
groups:
  - name: conversational-ai
    rules:
      - alert: HighErrorRate
        expr: rate(conversational_ai_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          
      - alert: HighLatency
        expr: histogram_quantile(0.99, conversational_ai_request_duration_seconds) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High latency detected
```

---

## Security

### TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    location / {
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: conversational-ai-policy
  namespace: conversational-ai
spec:
  podSelector:
    matchLabels:
      app: conversational-ai
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - port: 6379
```

---

## Troubleshooting

### Common Issues

**Pod CrashLoopBackOff:**
```bash
kubectl logs -f deployment/conversational-ai -n conversational-ai
kubectl describe pod <pod-name> -n conversational-ai
```

**High Memory Usage:**
- Reduce ASR model size: `ASR_MODEL_SIZE=tiny`
- Enable model offloading
- Increase memory limits

**WebSocket Connection Drops:**
- Check ingress timeout settings
- Verify load balancer sticky sessions
- Check for network policies blocking connections

**Slow Response Times:**
- Enable GPU acceleration
- Scale horizontally
- Check Redis latency
- Review model caching

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Component status
curl http://localhost:8000/health/detailed

# Metrics
curl http://localhost:8000/metrics
```

### Logs

```bash
# Kubernetes
kubectl logs -f deployment/conversational-ai -n conversational-ai

# Docker
docker compose logs -f api

# Structured query
kubectl logs -l app=conversational-ai --since=1h | grep ERROR
```
