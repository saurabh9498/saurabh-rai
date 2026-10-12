# =============================================================================
# Retail Vision Analytics - Makefile
# =============================================================================
# Common development tasks and commands
# =============================================================================

.PHONY: help install install-dev test lint format clean docker-build docker-up docker-down docs

# Default target
help:
	@echo "Retail Vision Analytics - Available Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make setup          Complete development setup"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-int       Run integration tests only"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make lint           Run linters (flake8, mypy)"
	@echo "  make format         Format code (black, isort)"
	@echo "  make check          Run all checks (lint + test)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    View service logs"
	@echo ""
	@echo "Data:"
	@echo "  make sample-data    Generate sample data"
	@echo "  make download-models Download pre-trained models"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Remove build artifacts"
	@echo "  make docs           Build documentation"
	@echo "  make run            Run API server locally"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	pre-commit install

setup: install-dev
	cp -n .env.example .env || true
	mkdir -p data/models data/sample
	@echo "Setup complete! Edit .env with your settings."

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-int:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

check: lint test

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

docker-shell:
	docker-compose -f docker/docker-compose.yml exec retail-vision bash

# Build specific images
docker-build-app:
	docker build -f docker/Dockerfile -t retail-vision:latest .

docker-build-jetson:
	docker build -f docker/Dockerfile.jetson -t retail-vision:jetson .

# =============================================================================
# Data
# =============================================================================

sample-data:
	python scripts/generate_sample_data.py --output data/sample/ --hours 24
	@echo "Sample data generated in data/sample/"

download-models:
	bash scripts/download_models.sh
	@echo "Models downloaded to data/models/"

# =============================================================================
# Run
# =============================================================================

run:
	uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --workers 4

# =============================================================================
# Documentation
# =============================================================================

docs:
	@echo "Documentation available in docs/"
	@echo "  - docs/ARCHITECTURE.md"
	@echo "  - docs/API_REFERENCE.md"
	@echo "  - docs/DEPLOYMENT.md"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleaned build artifacts"

clean-data:
	rm -rf data/sample/*.json
	rm -rf data/sample/*.csv
	@echo "Cleaned sample data"

clean-all: clean clean-data
	rm -rf data/models/*.engine
	rm -rf data/models/*.onnx
	@echo "Cleaned all generated files"

# =============================================================================
# Development Shortcuts
# =============================================================================

# Quick test and format
dev: format test

# Full CI check
ci: format lint test-cov

# Rebuild everything
rebuild: clean docker-build docker-up
