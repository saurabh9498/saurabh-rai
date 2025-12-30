# =============================================================================
# Conversational AI Assistant - Production Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
# Supports both CPU and GPU inference

# -----------------------------------------------------------------------------
# Stage 1: Base image with system dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # Utilities
    curl \
    wget \
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# -----------------------------------------------------------------------------
# Stage 2: Builder - Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools \
    && pip install -r requirements.txt

# Pre-download models (optional, can be done at runtime)
# This increases image size but speeds up cold starts
ARG PRELOAD_MODELS=false
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
        python -c "import whisper; whisper.load_model('base')" \
        && python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')" \
    ; fi

# -----------------------------------------------------------------------------
# Stage 3: Production image
# -----------------------------------------------------------------------------
FROM base AS production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup scripts/ ./scripts/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs \
    && chown -R appuser:appgroup /app

# Copy model cache from builder if preloaded
COPY --from=builder --chown=appuser:appgroup /root/.cache/whisper /app/models/whisper/ 2>/dev/null || true
COPY --from=builder --chown=appuser:appgroup /root/.cache/huggingface /app/models/huggingface/ 2>/dev/null || true

# Environment variables
ENV ENV=production \
    LOG_LEVEL=info \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    WORKERS=4 \
    # Model paths
    WHISPER_CACHE_DIR=/app/models/whisper \
    HF_HOME=/app/models/huggingface \
    # Performance tuning
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose port
EXPOSE ${API_PORT}

# Switch to non-root user
USER appuser

# Default command
CMD ["sh", "-c", "uvicorn src.api.main:app --host ${API_HOST} --port ${API_PORT} --workers ${WORKERS}"]

# -----------------------------------------------------------------------------
# Stage 4: Development image (optional)
# -----------------------------------------------------------------------------
FROM production AS development

USER root

# Install development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    pytest-asyncio \
    httpx \
    black \
    isort \
    ruff \
    mypy \
    ipython \
    jupyter

# Install additional dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

USER appuser

ENV ENV=development

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
LABEL maintainer="your-email@example.com" \
      version="1.0.0" \
      description="Conversational AI Assistant with ASR, NLU, Dialog, and TTS" \
      org.opencontainers.image.source="https://github.com/yourusername/conversational-ai"
