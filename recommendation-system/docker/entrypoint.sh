#!/bin/bash
# =============================================================================
# Real-Time Personalization Engine - Docker Entrypoint
# Supports multiple run modes: serve, train, evaluate, worker
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for dependent services
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=${4:-30}
    local attempt=1

    log_info "Waiting for ${service_name} at ${host}:${port}..."
    
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            log_error "${service_name} not available after ${max_attempts} attempts"
            return 1
        fi
        log_info "Attempt ${attempt}/${max_attempts}: ${service_name} not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_info "${service_name} is available!"
    return 0
}

# Check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        return 0
    else
        log_warn "No NVIDIA GPU detected, running in CPU mode"
        return 1
    fi
}

# Initialize application
init_app() {
    log_info "Initializing Real-Time Personalization Engine..."
    
    # Check GPU
    check_gpu || true
    
    # Wait for Redis if configured
    if [ -n "$REDIS_HOST" ] && [ -n "$REDIS_PORT" ]; then
        wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis" 30 || log_warn "Continuing without Redis..."
    fi
    
    # Wait for Triton if configured
    if [ -n "$TRITON_URL" ]; then
        TRITON_HOST=$(echo "$TRITON_URL" | cut -d: -f1)
        TRITON_PORT=$(echo "$TRITON_URL" | cut -d: -f2)
        wait_for_service "$TRITON_HOST" "$TRITON_PORT" "Triton" 60 || log_warn "Continuing without Triton..."
    fi
    
    log_info "Initialization complete!"
}

# Start API server
start_server() {
    log_info "Starting recommendation API server..."
    
    # Default settings
    HOST=${APP_HOST:-0.0.0.0}
    PORT=${APP_PORT:-8000}
    WORKERS=${APP_WORKERS:-4}
    
    # Production vs development mode
    if [ "$APP_ENV" = "development" ]; then
        log_info "Running in development mode with auto-reload"
        exec uvicorn src.serving.api:app \
            --host "$HOST" \
            --port "$PORT" \
            --reload \
            --log-level debug
    else
        log_info "Running in production mode with ${WORKERS} workers"
        exec uvicorn src.serving.api:app \
            --host "$HOST" \
            --port "$PORT" \
            --workers "$WORKERS" \
            --log-level info \
            --access-log \
            --proxy-headers \
            --forwarded-allow-ips='*'
    fi
}

# Start training job
start_training() {
    log_info "Starting model training..."
    
    MODEL_TYPE=${1:-two_tower}
    CONFIG_PATH=${CONFIG_DIR:-/config}/model_config.yaml
    
    exec python -m scripts.train \
        --model "$MODEL_TYPE" \
        --config "$CONFIG_PATH" \
        --output-dir "${MODEL_DIR:-/models}" \
        "${@:2}"
}

# Start evaluation
start_evaluation() {
    log_info "Starting model evaluation..."
    
    MODEL_PATH=${1:-${MODEL_DIR}/latest}
    
    exec python -m scripts.evaluate \
        --model-path "$MODEL_PATH" \
        --config "${CONFIG_DIR:-/config}/model_config.yaml" \
        "${@:2}"
}

# Start background worker (for async tasks)
start_worker() {
    log_info "Starting background worker..."
    
    # This could be a Celery worker, custom async processor, etc.
    exec python -m src.workers.background_processor \
        --config "${CONFIG_DIR:-/config}/model_config.yaml"
}

# Build FAISS index
build_index() {
    log_info "Building FAISS index..."
    
    exec python -m scripts.build_index \
        --embeddings-path "${1:-${MODEL_DIR}/item_embeddings.npy}" \
        --output-path "${MODEL_DIR}/faiss_index" \
        --index-type "${INDEX_TYPE:-IVF4096,Flat}" \
        --use-gpu
}

# Health check
health_check() {
    curl -sf "http://localhost:${APP_PORT:-8000}/health" > /dev/null 2>&1
    exit $?
}

# Main entrypoint logic
main() {
    local command=${1:-serve}
    
    case "$command" in
        serve)
            init_app
            start_server
            ;;
        train)
            init_app
            start_training "${@:2}"
            ;;
        evaluate)
            init_app
            start_evaluation "${@:2}"
            ;;
        worker)
            init_app
            start_worker
            ;;
        build-index)
            start_build_index "${@:2}"
            ;;
        health)
            health_check
            ;;
        shell)
            exec /bin/bash
            ;;
        *)
            # Pass through any other command
            exec "$@"
            ;;
    esac
}

# Run main with all arguments
main "$@"
