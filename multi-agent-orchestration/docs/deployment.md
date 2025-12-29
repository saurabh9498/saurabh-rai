# Deployment Guide

This guide covers deploying the Multi-Agent AI System in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- Python 3.10+
- Docker 20.10+ (for containerized deployment)
- At least one LLM API key (OpenAI or Anthropic)

### Optional
- Kubernetes cluster (for K8s deployment)
- Redis (for caching)
- PostgreSQL (for persistent storage)

---

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/saurabh-rai/multi-agent-ai-system.git
cd multi-agent-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the API server
uvicorn src.api.main:app --reload --port 8000

# In another terminal, run Streamlit UI
streamlit run ui/streamlit_app.py
```

### Access Points
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

---

## Docker Deployment

### Development

```bash
cd docker

# Start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### Production

```bash
# Build production image
docker build -t multi-agent-system:latest -f docker/Dockerfile --target production .

# Run with production settings
docker run -d \
  --name multi-agent-api \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e ENVIRONMENT=production \
  -v /path/to/data:/app/data \
  multi-agent-system:latest
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...          # OpenAI API key
LLM_PROVIDER=openai            # openai or anthropic

# Optional
ANTHROPIC_API_KEY=sk-ant-...   # If using Anthropic
ENVIRONMENT=production         # development, staging, production
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
```

---

## Kubernetes Deployment

### Prerequisites

- kubectl configured
- Helm 3.x (optional, for Helm chart)
- Kubernetes cluster (1.24+)

### Manifests

Create namespace:
```bash
kubectl create namespace multi-agent-system
```

#### ConfigMap (k8s/configmap.yaml)
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: multi-agent-config
  namespace: multi-agent-system
data:
  LLM_PROVIDER: "openai"
  VECTOR_STORE: "chromadb"
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
```

#### Secret (k8s/secret.yaml)
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: multi-agent-secrets
  namespace: multi-agent-system
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-your-key-here"
```

#### Deployment (k8s/deployment.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-agent-api
  namespace: multi-agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: multi-agent-api
  template:
    metadata:
      labels:
        app: multi-agent-api
    spec:
      containers:
      - name: api
        image: multi-agent-system:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: multi-agent-config
        - secretRef:
            name: multi-agent-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service (k8s/service.yaml)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: multi-agent-api
  namespace: multi-agent-system
spec:
  selector:
    app: multi-agent-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress (k8s/ingress.yaml)
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multi-agent-ingress
  namespace: multi-agent-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: multi-agent-api
            port:
              number: 80
```

### Deploy

```bash
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -n multi-agent-system
kubectl get services -n multi-agent-system
```

---

## Cloud Deployments

### AWS ECS

1. Push image to ECR:
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag multi-agent-system:latest <account>.dkr.ecr.<region>.amazonaws.com/multi-agent-system:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/multi-agent-system:latest
```

2. Create ECS task definition and service via AWS Console or Terraform

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/multi-agent-system

# Deploy
gcloud run deploy multi-agent-api \
  --image gcr.io/PROJECT_ID/multi-agent-system \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=your-key
```

### Azure Container Apps

```bash
az containerapp create \
  --name multi-agent-api \
  --resource-group myResourceGroup \
  --image multi-agent-system:latest \
  --target-port 8000 \
  --env-vars OPENAI_API_KEY=your-key
```

---

## Configuration

### Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Configure proper API keys
- [ ] Enable HTTPS/TLS
- [ ] Set up rate limiting
- [ ] Configure logging to external service
- [ ] Set up monitoring and alerting
- [ ] Configure backup for vector store
- [ ] Set resource limits

### Performance Tuning

```bash
# API workers (default: 1)
uvicorn src.api.main:app --workers 4

# Or with Gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## Monitoring

### Health Endpoints

- `/health` - Basic health check
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe

### Prometheus Metrics

Enable metrics endpoint:
```bash
ENABLE_METRICS=true
```

Metrics available at `/metrics`:
- `http_requests_total`
- `http_request_duration_seconds`
- `agent_execution_duration_seconds`
- `rag_query_duration_seconds`

### Logging

Structured JSON logging in production:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Query executed",
  "query_id": "abc123",
  "duration_ms": 1234
}
```

---

## Troubleshooting

### Common Issues

**API not starting:**
```bash
# Check logs
docker logs multi-agent-api

# Verify environment variables
docker exec multi-agent-api env | grep -E "(OPENAI|LLM)"
```

**Vector store errors:**
```bash
# Check ChromaDB directory permissions
ls -la data/chroma/

# Reset ChromaDB (development only)
rm -rf data/chroma/*
```

**Out of memory:**
- Reduce `max_tokens` in config
- Limit concurrent requests
- Increase container memory limits

### Support

For issues, please:
1. Check existing GitHub issues
2. Review logs for error messages
3. Open a new issue with reproduction steps
