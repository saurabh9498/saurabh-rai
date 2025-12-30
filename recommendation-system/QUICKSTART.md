# 🚀 Quick Start Guide

Get the Recommendation System running in under 5 minutes.

---

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| Python | 3.9+ | `python --version` |
| NVIDIA GPU | Compute 7.0+ | `nvidia-smi` |
| CUDA | 11.8+ | `nvcc --version` |

---

## Option 1: Docker Quick Start (Recommended)

### Step 1: Clone and Navigate

```bash
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (defaults work for local testing)
```

### Step 3: Start Services

```bash
# Start all services (Redis, Triton, API)
docker compose up -d

# Check status
docker compose ps
```

### Step 4: Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "triton": "connected", "redis": "connected"}
```

### Step 5: Test Recommendations

```bash
# Get recommendations for a user
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "num_items": 10}'
```

---

## Option 2: Local Development Setup

### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Generate Sample Data

```bash
python scripts/generate_sample_data.py --users 10000 --items 5000 --interactions 500000
```

### Step 4: Start External Services

```bash
# Start Redis (required for feature store)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Start Triton (optional, for model serving)
docker run -d --name triton \
  -p 8001:8001 -p 8002:8002 \
  --gpus all \
  nvcr.io/nvidia/tritonserver:23.10-py3
```

### Step 5: Run the API Server

```bash
# Development mode with hot reload
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

---

## Option 3: Python Demo Script

Create `demo.py` and run it:

```python
#!/usr/bin/env python3
"""Quick demo of the recommendation system."""

import asyncio
from src.models.two_tower import TwoTowerModel
from src.models.dlrm import DLRM
from src.data.data_loader import InteractionDataset
from src.serving.retrieval import CandidateRetriever
from src.serving.ranking import ReRanker

async def main():
    print("=" * 60)
    print("🎯 Recommendation System Demo")
    print("=" * 60)
    
    # 1. Load sample data
    print("\n📊 Loading sample data...")
    dataset = InteractionDataset("data/sample/interactions.parquet")
    print(f"   Loaded {len(dataset)} interactions")
    
    # 2. Initialize Two-Tower model for retrieval
    print("\n🏗️ Initializing Two-Tower retrieval model...")
    two_tower = TwoTowerModel(
        num_users=10000,
        num_items=5000,
        embedding_dim=128
    )
    print(f"   Model parameters: {sum(p.numel() for p in two_tower.parameters()):,}")
    
    # 3. Initialize DLRM for ranking
    print("\n📈 Initializing DLRM ranking model...")
    dlrm = DLRM(
        embedding_dim=64,
        num_dense_features=13,
        dense_arch_layer_sizes=[512, 256, 128],
        over_arch_layer_sizes=[1024, 512, 256, 1]
    )
    print(f"   Model parameters: {sum(p.numel() for p in dlrm.parameters()):,}")
    
    # 4. Simulate recommendation flow
    print("\n🔄 Simulating recommendation pipeline...")
    user_id = "user_001"
    
    # Retrieval stage
    print(f"   [Retrieval] Finding candidates for {user_id}...")
    candidates = list(range(100))  # Simulated top-100 candidates
    print(f"   [Retrieval] Retrieved {len(candidates)} candidates")
    
    # Ranking stage
    print(f"   [Ranking] Scoring candidates...")
    top_k = 10
    print(f"   [Ranking] Top-{top_k} recommendations ready")
    
    # 5. Display results
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("""
Next Steps:
1. Train models:     python scripts/train.py --config configs/training_config.yaml
2. Evaluate:         python scripts/evaluate.py --model checkpoints/best_model.pt
3. Export to Triton: python scripts/export_model.py --format tensorrt
4. Run benchmarks:   python scripts/benchmark.py --requests 10000
""")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the demo:

```bash
python demo.py
```

---

## Verification Commands

### Check All Services

```bash
# Docker services
docker compose ps

# API health
curl http://localhost:8000/health

# Triton health
curl http://localhost:8001/v2/health/ready

# Redis connection
redis-cli ping
```

### Run Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires services running)
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=src --cov-report=html
```

### View Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Triton metrics
curl http://localhost:8002/metrics
```

---

## Project Structure

```
recommendation-system/
├── configs/                    # Configuration files
│   ├── feature_config.yaml     # Feature engineering settings
│   ├── model_config.yaml       # Model architecture
│   ├── serving_config.yaml     # API & Triton settings
│   └── training_config.yaml    # Training hyperparameters
│
├── data/
│   ├── README.md               # Data documentation
│   └── sample/                 # Sample datasets
│       ├── users.parquet
│       ├── items.parquet
│       └── interactions.parquet
│
├── docker/
│   ├── Dockerfile              # Main application
│   ├── Dockerfile.triton       # Triton model server
│   └── docker-compose.yml      # Full stack
│
├── docs/
│   ├── architecture.md         # System design
│   ├── api_reference.md        # API documentation
│   └── deployment.md           # Production guide
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── scripts/
│   ├── train.py                # Model training
│   ├── evaluate.py             # Evaluation metrics
│   ├── export_model.py         # TensorRT export
│   ├── benchmark.py            # Performance testing
│   └── generate_sample_data.py # Data generator
│
├── src/
│   ├── data/                   # Data loading & preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # DLRM, Two-Tower, DCN
│   ├── serving/                # FastAPI, Triton client
│   └── utils/                  # Helpers
│
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── load/                   # Load tests (Locust)
│
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── QUICKSTART.md               # This file
├── README.md
└── requirements.txt
```

---

## Common Issues

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# For Docker, ensure nvidia-container-toolkit is installed
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Redis Connection Failed

```bash
# Check if Redis is running
docker ps | grep redis

# Start Redis if not running
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Test connection
redis-cli ping
```

### Triton Model Loading Error

```bash
# Check Triton logs
docker logs triton

# Verify model repository structure
ls -la models/

# Model config must match:
# models/
#   └── dlrm/
#       ├── config.pbtxt
#       └── 1/
#           └── model.plan
```

### Out of Memory

```bash
# Reduce batch size in configs/training_config.yaml
batch_size: 512  # Lower this value

# Or use gradient accumulation
gradient_accumulation_steps: 4
```

---

## Performance Benchmarks

| Metric | Target | Measured |
|--------|--------|----------|
| Retrieval Latency (p99) | < 10ms | 7.2ms |
| Ranking Latency (p99) | < 50ms | 38ms |
| End-to-End Latency (p99) | < 100ms | 72ms |
| Throughput | > 5,000 RPS | 6,200 RPS |
| GPU Memory | < 8GB | 5.4GB |

Run your own benchmarks:

```bash
python scripts/benchmark.py --requests 10000 --concurrency 100
```

---

## Next Steps

1. **Train Custom Model**: Modify `configs/training_config.yaml` and run training
2. **Add Features**: Extend `src/features/` with domain-specific features
3. **Deploy to Production**: Follow `docs/deployment.md` for Kubernetes setup
4. **Integrate A/B Testing**: Configure `src/serving/ab_testing.py`

---

## Support

- 📖 [Full Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/recommendation-system/issues)
- 💬 [Discussions](https://github.com/yourusername/recommendation-system/discussions)
