.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check clean run run-api run-ui docker-build docker-up docker-down docs data

# Default target
help:
	@echo "Multi-Agent AI System - Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run            Run API server (development mode)"
	@echo "  make run-api        Run FastAPI server"
	@echo "  make run-ui         Run Streamlit UI"
	@echo "  make data           Generate sample data"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-e2e       Run end-to-end tests"
	@echo "  make coverage       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters (ruff, flake8)"
	@echo "  make format         Format code (black, isort)"
	@echo "  make type-check     Run type checking (mypy)"
	@echo "  make check          Run all code quality checks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    View container logs"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Remove cache and build artifacts"
	@echo "  make docs           Generate documentation"

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

run: run-api

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	streamlit run ui/streamlit_app.py --server.port 8501

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

test-e2e:
	pytest tests/e2e/ -v --tb=short

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

docker-clean:
	docker-compose -f docker/docker-compose.yml down -v --rmi local

# =============================================================================
# Documentation
# =============================================================================

docs:
	@echo "Documentation available in docs/ directory"
	@echo "  - docs/architecture.md"
	@echo "  - docs/api_reference.md"
	@echo "  - docs/deployment.md"

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned up cache and build artifacts"

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
