.PHONY: help install install-dev test test-unit test-integration lint format clean run train evaluate docker-build docker-up docker-down data

# Default target
help:
	@echo "Real-Time Fraud Detection System - Available Commands"
	@echo "======================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run            Run API server"
	@echo "  make run-stream     Run streaming processor"
	@echo "  make data           Generate sample data"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make train          Train all models"
	@echo "  make evaluate       Evaluate model performance"
	@echo "  make feature-store  Start feature store (Redis)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make coverage       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters"
	@echo "  make format         Format code"
	@echo "  make type-check     Run type checking"
	@echo "  make check          Run all code quality checks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Remove cache and build artifacts"
	@echo "  make redis-start    Start local Redis"
	@echo "  make kafka-start    Start local Kafka"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# =============================================================================
# Development
# =============================================================================

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-stream:
	python -m src.streaming.stream_processor

data:
	python scripts/generate_sample_data.py

data-large:
	python scripts/generate_sample_data.py --num-transactions 100000 --num-users 10000

# =============================================================================
# ML Pipeline
# =============================================================================

train:
	python scripts/train_model.py --data data/sample/ --output models/

train-xgboost:
	python scripts/train_model.py --data data/sample/ --model xgboost --output models/

train-neural:
	python scripts/train_model.py --data data/sample/ --model neural_net --output models/

evaluate:
	python scripts/evaluate.py --data data/sample/ --models models/

feature-store:
	docker run -d -p 6379:6379 --name fraud-redis redis:7
	@echo "Redis started on localhost:6379"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-load:
	locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 60s

coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check src/ tests/
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile=black

type-check:
	mypy src/ --ignore-missing-imports

check: lint type-check
	@echo "All checks passed!"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

docker-ps:
	docker-compose -f docker/docker-compose.yml ps

# =============================================================================
# Local Services
# =============================================================================

redis-start:
	docker run -d -p 6379:6379 --name fraud-redis redis:7 || docker start fraud-redis
	@echo "Redis available at localhost:6379"

redis-stop:
	docker stop fraud-redis || true

kafka-start:
	docker-compose -f docker/docker-compose.yml up -d kafka zookeeper
	@echo "Kafka available at localhost:9092"

kafka-stop:
	docker-compose -f docker/docker-compose.yml stop kafka zookeeper

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned up cache and build artifacts"

clean-models:
	rm -rf models/*.pkl models/*.pt
	@echo "Removed trained model files"

# =============================================================================
# CI/CD Helpers
# =============================================================================

ci-install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff black isort mypy

ci-test:
	pytest tests/ -v --tb=short --junitxml=test-results.xml

ci-lint:
	ruff check src/ tests/ --output-format=github
