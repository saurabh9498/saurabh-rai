#!/bin/bash
# =============================================================================
# Retail Vision Analytics - Jetson Deployment Script
# =============================================================================
# Automated deployment to NVIDIA Jetson devices (Orin, Xavier, Nano)
#
# Usage:
#   ./scripts/deploy_jetson.sh --host jetson-device --user nvidia
#   ./scripts/deploy_jetson.sh --host 192.168.1.100 --user nvidia --setup
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default values
REMOTE_USER="nvidia"
REMOTE_HOST=""
REMOTE_DIR="/opt/retail-vision"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Docker settings
DOCKER_IMAGE="retail-vision:jetson-latest"
CONTAINER_NAME="retail-vision"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
DO_SETUP=false
DO_BUILD=false
DO_DEPLOY=true
DO_START=true
POWER_MODE="MAXN"
VERBOSE=false

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy Retail Vision Analytics to NVIDIA Jetson devices.

Required:
  --host HOST           Jetson device hostname or IP address

Options:
  --user USER           Remote username (default: nvidia)
  --dir DIR             Remote installation directory (default: /opt/retail-vision)
  --setup               Run initial device setup (install dependencies)
  --build               Build Docker image on device
  --no-start            Deploy without starting the service
  --power-mode MODE     Set Jetson power mode (default: MAXN)
                        Options: MAXN, 15W, 30W, 50W (varies by device)
  --verbose             Enable verbose output
  --help                Show this help message

Examples:
  # Basic deployment
  $(basename "$0") --host jetson-orin.local --user nvidia

  # Full setup on new device
  $(basename "$0") --host 192.168.1.100 --setup --build

  # Deploy with specific power mode
  $(basename "$0") --host jetson-device --power-mode 30W

EOF
}

check_dependencies() {
    log_info "Checking local dependencies..."
    
    local missing=()
    
    for cmd in ssh scp rsync; do
        if ! command -v $cmd &> /dev/null; then
            missing+=($cmd)
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required commands: ${missing[*]}"
        exit 1
    fi
    
    log_info "All local dependencies available"
}

check_connection() {
    log_info "Checking connection to ${REMOTE_USER}@${REMOTE_HOST}..."
    
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "echo 'Connected'" &> /dev/null; then
        log_error "Cannot connect to ${REMOTE_HOST}"
        log_info "Make sure:"
        log_info "  1. Device is powered on and connected"
        log_info "  2. SSH is enabled on the device"
        log_info "  3. SSH key is set up (ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST})"
        exit 1
    fi
    
    log_info "Connection successful"
}

get_device_info() {
    print_header "Device Information"
    
    ssh "${REMOTE_USER}@${REMOTE_HOST}" << 'EOF'
echo "Hostname:     $(hostname)"
echo "IP Address:   $(hostname -I | awk '{print $1}')"

# Jetson model
if [ -f /etc/nv_tegra_release ]; then
    echo "L4T Version:  $(head -n 1 /etc/nv_tegra_release)"
fi

# JetPack version
if [ -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list ]; then
    echo "JetPack:      $(cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list | grep -oP 'r\d+\.\d+' | head -1)"
fi

# CUDA version
if command -v nvcc &> /dev/null; then
    echo "CUDA:         $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
fi

# TensorRT version
if dpkg -l | grep -q tensorrt; then
    echo "TensorRT:     $(dpkg -l | grep tensorrt | head -1 | awk '{print $3}')"
fi

# Memory
echo "Memory:       $(free -h | awk '/Mem:/ {print $2}')"

# Disk
echo "Disk:         $(df -h / | awk 'NR==2 {print $4 " available"}')"

# Power mode
if command -v nvpmodel &> /dev/null; then
    echo "Power Mode:   $(nvpmodel -q 2>/dev/null | head -1 || echo 'Unknown')"
fi

# GPU info
if command -v tegrastats &> /dev/null; then
    echo "GPU:          $(cat /proc/device-tree/model 2>/dev/null || echo 'NVIDIA Jetson')"
fi
EOF
}

setup_device() {
    print_header "Setting Up Device"
    
    log_info "Installing dependencies on ${REMOTE_HOST}..."
    
    ssh "${REMOTE_USER}@${REMOTE_HOST}" << 'EOF'
set -e

echo "[1/5] Updating system packages..."
sudo apt update

echo "[2/5] Installing Docker..."
if ! command -v docker &> /dev/null; then
    sudo apt install -y docker.io
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in."
fi

echo "[3/5] Installing NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi

echo "[4/5] Installing Python dependencies..."
sudo apt install -y python3-pip python3-dev

echo "[5/5] Creating directories..."
sudo mkdir -p /opt/retail-vision/{configs,models,data,logs}
sudo chown -R $USER:$USER /opt/retail-vision

echo "Setup complete!"
EOF
    
    log_info "Device setup complete"
}

sync_files() {
    print_header "Syncing Files"
    
    log_info "Creating remote directories..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}/{configs,models,scripts,data}"
    
    log_info "Syncing configuration files..."
    rsync -avz --progress \
        "${LOCAL_DIR}/configs/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/configs/"
    
    log_info "Syncing scripts..."
    rsync -avz --progress \
        "${LOCAL_DIR}/scripts/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/scripts/"
    
    log_info "Syncing Docker files..."
    rsync -avz --progress \
        "${LOCAL_DIR}/docker/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/docker/"
    
    log_info "Syncing environment file..."
    if [ -f "${LOCAL_DIR}/.env" ]; then
        scp "${LOCAL_DIR}/.env" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/.env"
    else
        scp "${LOCAL_DIR}/.env.example" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/.env"
        log_warn "Copied .env.example - please configure ${REMOTE_DIR}/.env"
    fi
    
    # Sync models if they exist
    if [ -d "${LOCAL_DIR}/data/models" ] && [ "$(ls -A ${LOCAL_DIR}/data/models/*.engine 2>/dev/null)" ]; then
        log_info "Syncing TensorRT models..."
        rsync -avz --progress \
            "${LOCAL_DIR}/data/models/"*.engine \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/models/"
    else
        log_warn "No TensorRT engines found - you'll need to convert models on device"
    fi
    
    log_info "File sync complete"
}

build_docker_image() {
    print_header "Building Docker Image"
    
    log_info "Building ${DOCKER_IMAGE} on ${REMOTE_HOST}..."
    
    ssh "${REMOTE_USER}@${REMOTE_HOST}" << EOF
cd ${REMOTE_DIR}
docker build -f docker/Dockerfile.jetson -t ${DOCKER_IMAGE} .
EOF
    
    log_info "Docker image built successfully"
}

set_power_mode() {
    print_header "Configuring Power Mode"
    
    log_info "Setting power mode to ${POWER_MODE}..."
    
    ssh "${REMOTE_USER}@${REMOTE_HOST}" << EOF
# Map power mode name to ID
case "${POWER_MODE}" in
    MAXN|maxn)
        MODE_ID=0
        ;;
    15W|15w)
        MODE_ID=2
        ;;
    30W|30w)
        MODE_ID=3
        ;;
    50W|50w)
        MODE_ID=4
        ;;
    *)
        echo "Unknown power mode: ${POWER_MODE}"
        echo "Available modes:"
        sudo nvpmodel -q --verbose
        exit 1
        ;;
esac

echo "Setting power mode: ${POWER_MODE} (ID: \$MODE_ID)"
sudo nvpmodel -m \$MODE_ID

# Enable jetson_clocks for maximum performance
if [ "${POWER_MODE}" = "MAXN" ]; then
    echo "Enabling jetson_clocks..."
    sudo jetson_clocks
fi

# Set fan to maximum
if [ -f /sys/devices/pwm-fan/target_pwm ]; then
    echo "Setting fan to maximum..."
    sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
fi

echo "Current power mode:"
sudo nvpmodel -q
EOF
    
    log_info "Power mode configured"
}

create_systemd_service() {
    print_header "Creating Systemd Service"
    
    log_info "Installing systemd service..."
    
    ssh "${REMOTE_USER}@${REMOTE_HOST}" << EOF
sudo tee /etc/systemd/system/retail-vision.service > /dev/null << 'SERVICE'
[Unit]
Description=Retail Vision Analytics
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=${REMOTE_USER}
WorkingDirectory=${REMOTE_DIR}
ExecStartPre=-/usr/bin/docker stop ${CONTAINER_NAME}
ExecStartPre=-/usr/bin/docker rm ${CONTAINER_NAME}
ExecStart=/usr/bin/docker run --rm --name ${CONTAINER_NAME} \\
    --runtime nvidia \\
    --network host \\
    -v ${REMOTE_DIR}/configs:/opt/configs:ro \\
    -v ${REMOTE_DIR}/models:/opt/models:ro \\
    -v ${REMOTE_DIR}/data:/var/lib/retail-vision \\
    -v ${REMOTE_DIR}/logs:/var/log/retail-vision \\
    --env-file ${REMOTE_DIR}/.env \\
    ${DOCKER_IMAGE}
ExecStop=/usr/bin/docker stop ${CONTAINER_NAME}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable retail-vision
EOF
    
    log_info "Systemd service created"
}

start_service() {
    print_header "Starting Service"
    
    log_info "Starting retail-vision service..."
    
    ssh "${REMOTE_USER}@${REMOTE_HOST}" << EOF
sudo systemctl start retail-vision
sleep 3
sudo systemctl status retail-vision --no-pager || true
EOF
    
    log_info "Service started"
}

verify_deployment() {
    print_header "Verifying Deployment"
    
    log_info "Checking service health..."
    
    ssh "${REMOTE_USER}@${REMOTE_HOST}" << EOF
# Check container status
echo "Container status:"
docker ps --filter name=${CONTAINER_NAME} --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check API health
echo ""
echo "API health check:"
sleep 5
curl -s http://localhost:8000/health || echo "API not responding yet (may still be starting)"

# Check GPU
echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader || echo "nvidia-smi not available"
EOF
}

show_next_steps() {
    print_header "Deployment Complete"
    
    cat << EOF

${GREEN}✓ Retail Vision Analytics deployed successfully!${NC}

${BLUE}Access Points:${NC}
  API:       http://${REMOTE_HOST}:8000
  Health:    http://${REMOTE_HOST}:8000/health
  Docs:      http://${REMOTE_HOST}:8000/docs

${BLUE}Useful Commands (on Jetson):${NC}
  # Check service status
  sudo systemctl status retail-vision

  # View logs
  sudo journalctl -u retail-vision -f

  # Restart service
  sudo systemctl restart retail-vision

  # Check GPU
  tegrastats

  # Check container
  docker logs ${CONTAINER_NAME} -f

${BLUE}Next Steps:${NC}
  1. Configure cameras in ${REMOTE_DIR}/configs/app_config.yaml
  2. Add TensorRT models to ${REMOTE_DIR}/models/
  3. Set up cloud sync in ${REMOTE_DIR}/.env (optional)
  4. Monitor via Grafana dashboard (optional)

EOF
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)
                REMOTE_HOST="$2"
                shift 2
                ;;
            --user)
                REMOTE_USER="$2"
                shift 2
                ;;
            --dir)
                REMOTE_DIR="$2"
                shift 2
                ;;
            --setup)
                DO_SETUP=true
                shift
                ;;
            --build)
                DO_BUILD=true
                shift
                ;;
            --no-start)
                DO_START=false
                shift
                ;;
            --power-mode)
                POWER_MODE="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                set -x
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "${REMOTE_HOST}" ]; then
        log_error "Missing required argument: --host"
        show_usage
        exit 1
    fi
    
    print_header "Retail Vision Analytics - Jetson Deployment"
    echo -e "Host:      ${REMOTE_USER}@${REMOTE_HOST}"
    echo -e "Directory: ${REMOTE_DIR}"
    echo -e "Setup:     ${DO_SETUP}"
    echo -e "Build:     ${DO_BUILD}"
    echo -e "Start:     ${DO_START}"
    
    # Run deployment steps
    check_dependencies
    check_connection
    get_device_info
    
    if [ "$DO_SETUP" = true ]; then
        setup_device
    fi
    
    sync_files
    
    if [ "$DO_BUILD" = true ]; then
        build_docker_image
    fi
    
    set_power_mode
    create_systemd_service
    
    if [ "$DO_START" = true ]; then
        start_service
        verify_deployment
    fi
    
    show_next_steps
}

main "$@"
