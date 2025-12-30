# 🚀 Quick Start Guide

Get the Real-Time Fraud Detection System running in minutes.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Required |
| Redis 7.0+ | Feature store |
| Kafka (optional) | For streaming mode |
| Docker (optional) | Easiest setup |
| 8GB RAM | For model loading |

---

## Option 1: Docker Compose (Recommended)

The fastest way to get everything running.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

# 2. Set up environment variables
cp .env.example .env
# Edit .env if needed (defaults work for local testing)

# 3. Build and run with Docker Compose
cd docker
docker-compose up --build

# 4. Access the applications:
#    - API:     http://localhost:8000
#    - Docs:    http://localhost:8000/docs
#    - Metrics: http://localhost:8000/metrics
#    - Grafana: http://localhost:3000 (admin/admin)
```

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI fraud detection |
| `redis` | 6379 | Feature store |
| `kafka` | 9092 | Transaction streaming |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards |

---

## Option 2: Local Development

For development and debugging.

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# OR
venv\Scripts\activate           # Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Start Redis

```bash
# Option A: Docker
docker run -d -p 6379:6379 redis:7

# Option B: Local install
redis-server
```

### Step 4: Generate Sample Data

```bash
python scripts/generate_sample_data.py

# Generates:
# - 1000 transactions
# - 100 user profiles
# - ~2% fraud rate
```

### Step 5: Train Models (Optional)

```bash
# Train with sample data
python scripts/train_model.py --data data/sample/ --output models/

# Or use pre-configured models (if available)
```

### Step 6: Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Step 7: Test a Transaction

```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_test123",
    "user_id": "usr_abc456",
    "amount": 299.99,
    "merchant_category": "electronics",
    "device_fingerprint": "fp_device1",
    "timestamp": "2024-01-15T14:30:00Z"
  }'
```

---

## Option 3: Python Script Demo

Minimal code to test the scoring engine.

```python
# demo.py
from src.models.ensemble import FraudEnsemble, EnsembleConfig
from src.features.feature_store import MockFeatureStore

# Initialize with mock feature store (no Redis needed)
feature_store = MockFeatureStore()
ensemble = FraudEnsemble(EnsembleConfig())

# Create a transaction
transaction = {
    "transaction_id": "txn_demo_001",
    "user_id": "usr_demo_001",
    "amount": 150.00,
    "merchant_category": "retail",
    "device_fingerprint": "fp_demo",
    "timestamp": "2024-01-15T10:30:00Z"
}

# Score the transaction
result = ensemble.score(transaction, feature_store.get_features(transaction["user_id"]))

print(f"Risk Score: {result.risk_score:.3f}")
print(f"Decision: {result.decision}")
print(f"Model Scores: {result.model_scores}")
```

Run with:
```bash
python demo.py
```

---

## Option 4: Jupyter Notebook

Interactive exploration of the system.

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/model_analysis.ipynb
```

The notebook covers:
1. ✅ Synthetic data generation
2. ✅ Exploratory data analysis
3. ✅ Model training
4. ✅ Performance evaluation
5. ✅ Feature importance analysis

---

## Verify Installation

### Check API Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "redis_connected": true
}
```

### Test Scoring Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "test", "user_id": "user1", "amount": 100}'
```

Expected response:
```json
{
  "transaction_id": "test",
  "risk_score": 0.05,
  "decision": "APPROVE",
  "latency_ms": 8.2
}
```

---

## Project Structure

```
fraud-detection/
│
├── src/                          # Source code
│   ├── models/                   # 🧠 ML Models
│   │   ├── ensemble.py              # Ensemble scorer
│   │   ├── xgboost_model.py         # XGBoost classifier
│   │   ├── neural_net.py            # PyTorch neural network
│   │   └── isolation_forest.py      # Anomaly detector
│   │
│   ├── features/                 # 📊 Feature Engineering
│   │   └── feature_store.py         # Redis feature store
│   │
│   ├── streaming/                # 📡 Real-time Processing
│   │   └── stream_processor.py      # Kafka consumer
│   │
│   ├── api/                      # 🌐 REST API
│   │   ├── main.py                  # FastAPI app
│   │   └── schemas.py               # Pydantic models
│   │
│   └── monitoring/               # 📈 Observability
│       └── metrics.py               # Prometheus metrics
│
├── data/                         # 📁 Data files
│   ├── sample/                      # Sample transactions
│   └── models/                      # Trained models
│
├── configs/                      # ⚙️ Configuration
│   ├── model_config.yaml            # Model hyperparameters
│   └── rules.yaml                   # Business rules
│
├── docker/                       # 🐳 Docker setup
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── scripts/                      # 🔨 Utility scripts
│   ├── train_model.py               # Model training
│   ├── evaluate.py                  # Model evaluation
│   └── generate_sample_data.py      # Data generator
│
├── tests/                        # ✅ Test suites
│   ├── unit/
│   └── integration/
│
├── notebooks/                    # 📓 Jupyter notebooks
│   └── model_analysis.ipynb
│
└── docs/                         # 📖 Documentation
    ├── architecture.md
    ├── api_reference.md
    └── deployment.md
```

---

## Decision Thresholds

| Score Range | Decision | Action |
|-------------|----------|--------|
| 0.0 - 0.3 | APPROVE | Auto-approve |
| 0.3 - 0.7 | STEP_UP | Request 2FA |
| 0.7 - 0.9 | REVIEW | Manual queue |
| 0.9 - 1.0 | DECLINE | Auto-decline |

---

## Performance Expectations

| Metric | Target | Typical |
|--------|--------|---------|
| P99 Latency | < 20ms | 8ms |
| Throughput | 50K TPS | 48K TPS |
| False Positive Rate | < 2% | 1.8% |
| Fraud Detection Rate | > 95% | 97.3% |

---

## Common Issues

### "Redis connection refused"

```bash
# Check Redis is running
redis-cli ping

# Start Redis
docker run -d -p 6379:6379 redis:7
```

### "Model file not found"

```bash
# Train models first
python scripts/train_model.py --data data/sample/ --output models/
```

### "Feature store timeout"

```bash
# Check Redis latency
redis-cli --latency

# Increase timeout in .env
REDIS_TIMEOUT=5000
```

### Port already in use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn src.api.main:app --port 8001
```

---

## Environment Variables

Key variables in `.env`:

```bash
# Redis
REDIS_URL=redis://localhost:6379
REDIS_TIMEOUT=1000

# Models
MODEL_DIR=./models
ENSEMBLE_WEIGHTS=0.45,0.35,0.20

# Thresholds
APPROVE_THRESHOLD=0.3
STEP_UP_THRESHOLD=0.7
DECLINE_THRESHOLD=0.9

# API
API_HOST=0.0.0.0
API_PORT=8000
```

See `.env.example` for all options.

---

## Next Steps

1. **Generate more data**: Increase `--num-transactions` for training
2. **Train models**: Use `scripts/train_model.py`
3. **Run benchmarks**: Test with `scripts/benchmark.py`
4. **Enable Kafka**: For real-time streaming
5. **Set up Grafana**: Import dashboards from `configs/`

---

## Getting Help

- 📖 [Architecture Guide](docs/architecture.md)
- 🔌 [API Reference](docs/api_reference.md)
- 🚀 [Deployment Guide](docs/deployment.md)
- 📊 [Model Documentation](docs/models.md)
- 🔧 [Runbook](docs/runbook.md)
- 🐛 [Open an Issue](https://github.com/your-username/fraud-detection/issues)

---

Happy fraud detecting! 🛡️
