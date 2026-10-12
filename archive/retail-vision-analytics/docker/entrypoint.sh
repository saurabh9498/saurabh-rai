#!/bin/bash
# =============================================================================
# Retail Vision Analytics - Docker Entrypoint
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  Retail Vision Analytics v1.0.0${NC}"
echo -e "${GREEN}=======================================${NC}"

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Set default values
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

# Application paths
export APP_HOME=${APP_HOME:-/app}
export CONFIG_DIR=${CONFIG_DIR:-/opt/configs}
export MODEL_DIR=${MODEL_DIR:-/opt/models}
export DATA_DIR=${DATA_DIR:-/var/lib/retail-vision}
export LOG_DIR=${LOG_DIR:-/var/log/retail-vision}

# Create directories
mkdir -p ${DATA_DIR} ${LOG_DIR}

# -----------------------------------------------------------------------------
# GPU Check
# -----------------------------------------------------------------------------

echo -e "\n${YELLOW}Checking GPU availability...${NC}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    echo -e "${RED}⚠ No GPU detected - running in CPU mode${NC}"
fi

# -----------------------------------------------------------------------------
# DeepStream Check
# -----------------------------------------------------------------------------

echo -e "\n${YELLOW}Checking DeepStream...${NC}"

if [ -d "/opt/nvidia/deepstream" ]; then
    DS_VERSION=$(cat /opt/nvidia/deepstream/deepstream*/version 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}✓ DeepStream installed: ${DS_VERSION}${NC}"
else
    echo -e "${YELLOW}⚠ DeepStream not found${NC}"
fi

# -----------------------------------------------------------------------------
# Configuration Check
# -----------------------------------------------------------------------------

echo -e "\n${YELLOW}Checking configuration...${NC}"

CONFIG_FILE="${CONFIG_DIR}/app_config.yaml"
if [ -f "${CONFIG_FILE}" ]; then
    echo -e "${GREEN}✓ Configuration found: ${CONFIG_FILE}${NC}"
else
    echo -e "${YELLOW}⚠ No configuration file found, using defaults${NC}"
fi

# -----------------------------------------------------------------------------
# Model Check
# -----------------------------------------------------------------------------

echo -e "\n${YELLOW}Checking models...${NC}"

MODEL_COUNT=$(find ${MODEL_DIR} -name "*.engine" 2>/dev/null | wc -l)
if [ ${MODEL_COUNT} -gt 0 ]; then
    echo -e "${GREEN}✓ Found ${MODEL_COUNT} TensorRT engine(s)${NC}"
    find ${MODEL_DIR} -name "*.engine" -exec basename {} \;
else
    echo -e "${YELLOW}⚠ No TensorRT engines found in ${MODEL_DIR}${NC}"
    echo -e "${YELLOW}  Models will be downloaded/converted on first run${NC}"
fi

# -----------------------------------------------------------------------------
# Service Dependencies
# -----------------------------------------------------------------------------

echo -e "\n${YELLOW}Checking service dependencies...${NC}"

# Check Redis
if [ -n "${REDIS_HOST}" ]; then
    if redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT:-6379} ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis connection OK${NC}"
    else
        echo -e "${YELLOW}⚠ Cannot connect to Redis at ${REDIS_HOST}:${REDIS_PORT:-6379}${NC}"
    fi
fi

# -----------------------------------------------------------------------------
# Start Application
# -----------------------------------------------------------------------------

echo -e "\n${GREEN}Starting application...${NC}"
echo -e "Command: $@"
echo ""

# Execute the command
exec "$@"
