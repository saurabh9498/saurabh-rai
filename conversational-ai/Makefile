# =============================================================================
# Conversational AI Assistant - Makefile
# =============================================================================

.PHONY: help install install-dev clean lint format test test-unit test-integration \
        coverage docker-build docker-up docker-down docker-logs \
        run dev docs models data benchmark

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Variables
# =============================================================================

PYTHON := python3
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
RUFF := ruff
MYPY := mypy

SRC_DIR := src
TEST_DIR := tests
SCRIPTS_DIR := scripts
DOCS_DIR := docs

DOCKER_COMPOSE := docker compose
DOCKER_FILE := docker/Dockerfile
DOCKER_GPU_FILE := docker/Dockerfile.gpu

IMAGE_NAME := conversational-ai
IMAGE_TAG := latest

# Model settings
WHISPER_MODEL := base
TTS_MODEL := tts_models/en/ljspeech/tacotron2-DDC

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "Conversational AI Assistant - Available Commands"
	@echo "================================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install ## Install development dependencies
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

install-system: ## Install system dependencies (Ubuntu/Debian)
	sudo apt-get update
	sudo apt-get install -y ffmpeg libsndfile1 portaudio19-dev

install-models: ## Download required ML models
	@echo "Downloading Whisper model ($(WHISPER_MODEL))..."
	$(PYTHON) -c "import whisper; whisper.load_model('$(WHISPER_MODEL)')"
	@echo "Downloading TTS model..."
	$(PYTHON) -c "from TTS.api import TTS; TTS('$(TTS_MODEL)')" || true
	@echo "Models downloaded successfully!"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run all linters
	@echo "Running Ruff..."
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "Running mypy..."
	$(MYPY) $(SRC_DIR) --ignore-missing-imports
	@echo "Linting complete!"

format: ## Format code with Black and isort
	@echo "Running isort..."
	$(ISORT) $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "Running Black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "Formatting complete!"

format-check: ## Check code formatting without changes
	$(BLACK) --check --diff $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	$(ISORT) --check-only --diff $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)

type-check: ## Run type checking with mypy
	$(MYPY) $(SRC_DIR) --ignore-missing-imports --show-error-codes

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	$(PYTEST) $(TEST_DIR) -v --tb=short

test-unit: ## Run unit tests only
	$(PYTEST) $(TEST_DIR)/unit -v --tb=short

test-integration: ## Run integration tests (requires services)
	RUN_INTEGRATION_TESTS=true $(PYTEST) $(TEST_DIR)/integration -v --tb=short

test-e2e: ## Run end-to-end tests
	$(PYTEST) $(TEST_DIR)/e2e -v --tb=short

coverage: ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR) \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		--cov-fail-under=70
	@echo "Coverage report generated in htmlcov/"

test-watch: ## Run tests in watch mode
	$(PYTEST) $(TEST_DIR) -v --tb=short -f

# =============================================================================
# Running the Application
# =============================================================================

run: ## Run the API server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

dev: ## Run the API server in development mode with hot reload
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-worker: ## Run background worker
	$(PYTHON) -m src.worker.main

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f $(DOCKER_FILE) .

docker-build-gpu: ## Build GPU-enabled Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG)-gpu -f $(DOCKER_GPU_FILE) .

docker-up: ## Start all services with Docker Compose
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all services
	$(DOCKER_COMPOSE) down

docker-logs: ## View Docker Compose logs
	$(DOCKER_COMPOSE) logs -f

docker-logs-api: ## View API service logs
	$(DOCKER_COMPOSE) logs -f api

docker-restart: ## Restart all services
	$(DOCKER_COMPOSE) restart

docker-clean: ## Remove containers, volumes, and images
	$(DOCKER_COMPOSE) down -v --rmi local
	docker system prune -f

docker-shell: ## Open shell in API container
	$(DOCKER_COMPOSE) exec api /bin/bash

docker-test: ## Run tests in Docker
	$(DOCKER_COMPOSE) run --rm api pytest tests/ -v

# =============================================================================
# Data & Models
# =============================================================================

data: ## Generate sample data
	$(PYTHON) $(SCRIPTS_DIR)/generate_sample_data.py \
		--conversations 100 \
		--audio-samples 50 \
		--output data/sample/

data-large: ## Generate large dataset for training
	$(PYTHON) $(SCRIPTS_DIR)/generate_sample_data.py \
		--conversations 1000 \
		--audio-samples 500 \
		--output data/large/

validate-data: ## Validate data files
	$(PYTHON) $(SCRIPTS_DIR)/validate_data.py --path data/sample/

models: install-models ## Alias for install-models

train-nlu: ## Train NLU models
	$(PYTHON) $(SCRIPTS_DIR)/train_nlu.py \
		--data data/sample/intents.json \
		--output models/nlu/

evaluate: ## Evaluate model performance
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py \
		--model models/nlu/ \
		--data data/sample/

# =============================================================================
# Benchmarking
# =============================================================================

benchmark: ## Run performance benchmarks
	$(PYTHON) $(SCRIPTS_DIR)/benchmark.py \
		--requests 1000 \
		--concurrency 50

benchmark-asr: ## Benchmark ASR performance
	$(PYTHON) $(SCRIPTS_DIR)/benchmark.py \
		--component asr \
		--requests 100

benchmark-nlu: ## Benchmark NLU performance
	$(PYTHON) $(SCRIPTS_DIR)/benchmark.py \
		--component nlu \
		--requests 1000

benchmark-tts: ## Benchmark TTS performance
	$(PYTHON) $(SCRIPTS_DIR)/benchmark.py \
		--component tts \
		--requests 100

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	cd $(DOCS_DIR) && mkdocs build

docs-serve: ## Serve documentation locally
	cd $(DOCS_DIR) && mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	cd $(DOCS_DIR) && mkdocs gh-deploy

# =============================================================================
# Utilities
# =============================================================================

clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleaned!"

clean-docker: ## Clean Docker resources
	docker system prune -af
	docker volume prune -f

clean-all: clean clean-docker ## Clean everything

check: lint test ## Run all checks (lint + test)

ci: format-check lint coverage ## Run CI pipeline locally

health: ## Check service health
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Service not running"

version: ## Show version information
	@$(PYTHON) -c "import src; print(f'Version: {src.__version__}')" 2>/dev/null || echo "Version: development"

env: ## Show environment information
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Working Directory: $$(pwd)"
	@echo "Virtual Env: $${VIRTUAL_ENV:-none}"

# =============================================================================
# Client Tools
# =============================================================================

client: ## Run interactive CLI client
	$(PYTHON) $(SCRIPTS_DIR)/client.py

client-audio: ## Run audio CLI client
	$(PYTHON) $(SCRIPTS_DIR)/client.py --audio

client-websocket: ## Run WebSocket CLI client
	$(PYTHON) $(SCRIPTS_DIR)/client.py --websocket

# =============================================================================
# Development Shortcuts
# =============================================================================

setup: install-dev install-system install-models data ## Complete development setup
	@echo "Development environment ready!"

reset: clean docker-down ## Reset development environment
	@echo "Environment reset!"

quick-test: ## Run quick smoke tests
	$(PYTEST) $(TEST_DIR)/unit -v -x --tb=short -q

# =============================================================================
# Production
# =============================================================================

build: ## Build production package
	$(PYTHON) -m build

publish-test: ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	$(PYTHON) -m twine upload dist/*
