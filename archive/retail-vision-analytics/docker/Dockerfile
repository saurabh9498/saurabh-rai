# =============================================================================
# Retail Vision Analytics - Production Dockerfile
# =============================================================================
# Multi-stage build for optimized production deployment
#
# Build: docker build -t retail-vision:latest .
# Run:   docker run --gpus all -p 8000:8000 retail-vision:latest
#
# Targets:
# - builder: Build dependencies and compile models
# - runtime: Minimal production image
# - dev: Development image with tools
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    libglib2.0-dev \
    libjson-glib-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch and TensorRT Python bindings
RUN pip3 install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    tensorrt>=8.6.0 \
    onnx>=1.14.0 \
    onnxruntime-gpu>=1.15.0

# Install Ultralytics (YOLOv8)
RUN pip3 install --no-cache-dir ultralytics>=8.0.0

# Copy source code
WORKDIR /app
COPY . /app/

# Build any native extensions if needed
# RUN python3 setup.py build_ext --inplace

# Download and optimize models (if not pre-built)
# ARG MODEL_PRECISION=fp16
# RUN python3 scripts/download_models.py && \
#     python3 scripts/optimize_models.py --precision ${MODEL_PRECISION}

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch AS runtime

# Labels
LABEL maintainer="Saurabh <saurabh@example.com>"
LABEL version="1.0.0"
LABEL description="Real-Time Retail Vision Analytics with NVIDIA DeepStream"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Application configuration
ENV APP_HOME=/app
ENV CONFIG_PATH=/app/configs/app_config.yaml
ENV LOG_LEVEL=INFO

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    librdkafka1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r retail && useradd -r -g retail retail

# Create directories
RUN mkdir -p /app /var/lib/retail-vision /var/log/retail-vision \
    && chown -R retail:retail /app /var/lib/retail-vision /var/log/retail-vision

# Copy application
WORKDIR /app
COPY --from=builder /app /app
COPY --chown=retail:retail . /app/

# Copy pre-built models (if available)
# COPY --from=builder /app/models/*.engine /app/models/

# Set permissions
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000  # API
EXPOSE 8554  # RTSP output
EXPOSE 9090  # Prometheus metrics

# Switch to non-root user
USER retail

# Default command
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 3: Development
# -----------------------------------------------------------------------------
FROM runtime AS dev

# Switch to root for installations
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    nvtop \
    gdb \
    strace \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter \
    debugpy

# Enable debugging
ENV PYTHONDEBUG=1
ENV DEBUG=1

# Switch back to retail user
USER retail

# Development command with auto-reload
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
