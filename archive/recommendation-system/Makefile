# =============================================================================
# Real-Time Personalization Engine - Makefile
# =============================================================================

.PHONY: help install dev-install test lint format type-check clean docker-build docker-up docker-down train benchmark docs

# Default target
help:
	@echo "Real-Time Personalization Engine - Available Commands"
	@echo "======================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev-install    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make lint           Run linters"
	@echo "  make format         Format code"
	@echo "  make type-check     Run type checker"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    View service logs"
	@echo ""
	@echo "Training:"
	@echo "  make train-two-tower  Train Two-Tower model"
	@echo "  make train-dlrm       Train DLRM model"
	@echo "  make export-models    Export models to ONNX"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make benchmark-model     Benchmark model inference"
	@echo "  make benchmark-endpoint  Benchmark API endpoint"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo "  make docs-serve     Serve documentation locally"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Clean build artifacts"
	@echo "  make clean-all      Clean everything including models"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

# =============================================================================
# Development
# =============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

lint:
	flake8 src/ tests/
	ruff check src/ tests/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

format-check:
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/

type-check:
	mypy src/

quality: format lint type-check test
	@echo "All quality checks passed!"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec api /bin/bash

docker-clean:
	docker-compose down -v --rmi local

# GPU support
docker-up-gpu:
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# =============================================================================
# Training
# =============================================================================

train-two-tower:
	python scripts/train.py --model two_tower --config configs/model_config.yaml

train-dlrm:
	python scripts/train.py --model dlrm --config configs/model_config.yaml

export-models:
	python scripts/export_model.py --model two_tower --checkpoint models/two_tower.pt --format onnx
	python scripts/export_model.py --model dlrm --checkpoint models/dlrm.pt --format onnx

build-index:
	python scripts/build_index.py --embeddings models/item_embeddings.npy --output models/faiss_index

# =============================================================================
# Benchmarking
# =============================================================================

benchmark-model:
	python scripts/benchmark.py model --model dlrm --batch-sizes 1,8,32,64,128 --iterations 100

benchmark-endpoint:
	python scripts/benchmark.py endpoint --url http://localhost:8000/recommend --requests 1000 --concurrent 10

benchmark-all: benchmark-model benchmark-endpoint

# =============================================================================
# Documentation
# =============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

clean-all: clean
	rm -rf models/
	rm -rf mlruns/
	rm -rf wandb/
	rm -rf logs/

# =============================================================================
# Development Server
# =============================================================================

run-dev:
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4

# =============================================================================
# CI/CD Helpers
# =============================================================================

ci-test:
	pytest tests/ --junitxml=test-results.xml --cov=src --cov-report=xml

ci-lint:
	flake8 src/ tests/ --format=junit-xml > lint-results.xml || true
	black --check src/ tests/ scripts/

ci-build:
	docker build -t recommendation-engine:$${CI_COMMIT_SHA:-latest} -f docker/Dockerfile .
