.PHONY: help install install-dev test test-unit test-integration lint format clean build-cuda run benchmark docker-build docker-up docker-down data

# Default target
help:
	@echo "GPU-Accelerated ML Pipeline - Available Commands"
	@echo "================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make build-cuda     Build CUDA extensions"
	@echo ""
	@echo "Development:"
	@echo "  make run            Run inference server"
	@echo "  make benchmark      Run performance benchmarks"
	@echo "  make build-engine   Build TensorRT engine"
	@echo "  make data           Generate sample data"
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
	@echo "  make gpu-info       Display GPU information"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

build-cuda:
	cd src/cuda && python setup.py build_ext --inplace
	@echo "CUDA extensions built successfully"

# =============================================================================
# Development
# =============================================================================

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

benchmark:
	python scripts/benchmark.py --engine data/models/resnet50_fp16.engine --batch-size 32

benchmark-all:
	@echo "Running benchmarks for all precisions..."
	python scripts/benchmark.py --engine data/models/resnet50_fp32.engine --batch-size 1
	python scripts/benchmark.py --engine data/models/resnet50_fp16.engine --batch-size 1
	python scripts/benchmark.py --engine data/models/resnet50_int8.engine --batch-size 1

build-engine:
	python scripts/build_engine.py \
		--onnx data/models/resnet50.onnx \
		--output data/models/resnet50_fp16.engine \
		--precision fp16

build-engine-int8:
	python scripts/build_engine.py \
		--onnx data/models/resnet50.onnx \
		--output data/models/resnet50_int8.engine \
		--precision int8 \
		--calibration-data data/calibration/calibration_images/

data:
	python scripts/generate_sample_data.py

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

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

docker-shell:
	docker-compose -f docker/docker-compose.yml exec api bash

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.so" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned up cache and build artifacts"

clean-engines:
	rm -f data/models/*.engine
	@echo "Removed TensorRT engine files"

gpu-info:
	@echo "GPU Information:"
	@nvidia-smi
	@echo ""
	@echo "CUDA Version:"
	@nvcc --version 2>/dev/null || echo "nvcc not found"
	@echo ""
	@echo "TensorRT Version:"
	@python -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "TensorRT not found"

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
