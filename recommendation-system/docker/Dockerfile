# Real-Time Personalization Engine
# GPU-accelerated recommendation system with NVIDIA Merlin and Triton
#
# Build: docker build -t recommendation-system:latest -f docker/Dockerfile .
# Run: docker run --gpus all -p 8000:8000 recommendation-system:latest

# =============================================================================
# Stage 1: Base CUDA image with Python dependencies
# =============================================================================
FROM nvcr.io/nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04 AS base

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set up working directory
WORKDIR /app

# =============================================================================
# Stage 2: Python dependencies installation
# =============================================================================
FROM base AS dependencies

# Copy requirements files
COPY requirements.txt requirements-gpu.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    # Install GPU-specific packages if requirements-gpu.txt exists
    if [ -f requirements-gpu.txt ]; then pip install -r requirements-gpu.txt; fi

# Install FAISS GPU
RUN pip install faiss-gpu

# Install NVIDIA Merlin libraries
RUN pip install \
    nvtabular \
    merlin-core \
    merlin-models \
    merlin-systems

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM dependencies AS production

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
# 8000: FastAPI application
# 8001: Triton HTTP
# 8002: Triton gRPC
# 9090: Prometheus metrics
EXPOSE 8000 8001 8002 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - start FastAPI server
CMD ["python", "-m", "uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =============================================================================
# Stage 4: Development image with additional tools
# =============================================================================
FROM production AS development

USER root

# Install development dependencies
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-benchmark \
    black \
    isort \
    flake8 \
    mypy \
    ipython \
    jupyter \
    notebook

# Install debugging tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    nvtop \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Override command for development
CMD ["python", "-m", "uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Stage 5: Training image with full ML stack
# =============================================================================
FROM dependencies AS training

# Install additional training dependencies
RUN pip install \
    mlflow \
    wandb \
    optuna \
    ray[tune] \
    deepspeed \
    horovod

# Copy training scripts and data utilities
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser configs/ ./configs/

# Create directories for training artifacts
RUN mkdir -p /app/checkpoints /app/experiments /app/data && \
    chown -R appuser:appuser /app

USER appuser

# Default command for training
CMD ["python", "scripts/train.py", "--config", "configs/model_config.yaml"]
