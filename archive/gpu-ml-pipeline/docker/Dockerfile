# GPU ML Pipeline Dockerfile
# Multi-stage build with CUDA, TensorRT, and Python

# =============================================================================
# Stage 1: CUDA/TensorRT Base
# =============================================================================
FROM nvcr.io/nvidia/tensorrt:23.10-py3 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# =============================================================================
# Stage 2: Python Dependencies
# =============================================================================
FROM base AS python-deps

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: CUDA Kernels Build
# =============================================================================
FROM python-deps AS cuda-build

# Copy CUDA kernel source
COPY src/cuda /app/src/cuda

# Build CUDA extension
WORKDIR /app/src/cuda
RUN python setup.py build_ext --inplace && \
    python setup.py install

WORKDIR /app

# =============================================================================
# Stage 4: Production Image
# =============================================================================
FROM python-deps AS production

# Copy built CUDA kernels
COPY --from=cuda-build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Create model and data directories
RUN mkdir -p /app/models /app/engines /app/data /app/cache

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

# Set Python path
ENV PYTHONPATH=/app

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Stage 5: Development Image
# =============================================================================
FROM production AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    mypy \
    ipython \
    jupyter

USER mluser

# Development command
CMD ["python", "-m", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

# =============================================================================
# Stage 6: Triton Server Image
# =============================================================================
FROM nvcr.io/nvidia/tritonserver:23.10-py3 AS triton

# Install Python dependencies for preprocessing
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy model repository
COPY --chown=1000:1000 model_repository /models

# Expose Triton ports
# HTTP: 8000, gRPC: 8001, Metrics: 8002
EXPOSE 8000 8001 8002

# Triton command
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]
