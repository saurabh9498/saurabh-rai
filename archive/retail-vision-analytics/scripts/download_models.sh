#!/bin/bash
# =============================================================================
# Retail Vision Analytics - Model Download Script
# =============================================================================
# Downloads pre-trained models for retail object detection
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_DIR}/data/models"

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  Retail Vision Analytics - Model Setup${NC}"
echo -e "${GREEN}=======================================${NC}"

# Create model directory
mkdir -p "$MODEL_DIR"

# -----------------------------------------------------------------------------
# Download YOLOv8 Base Model
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/4] Downloading YOLOv8 base model...${NC}"

YOLO_MODEL="yolov8n.pt"
YOLO_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

if [ -f "$MODEL_DIR/$YOLO_MODEL" ]; then
    echo -e "${GREEN}✓ YOLOv8 model already exists${NC}"
else
    echo "Downloading from $YOLO_URL..."
    curl -L -o "$MODEL_DIR/$YOLO_MODEL" "$YOLO_URL"
    echo -e "${GREEN}✓ Downloaded $YOLO_MODEL${NC}"
fi

# -----------------------------------------------------------------------------
# Export to ONNX
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/4] Exporting to ONNX format...${NC}"

ONNX_MODEL="yolov8n_retail.onnx"

if [ -f "$MODEL_DIR/$ONNX_MODEL" ]; then
    echo -e "${GREEN}✓ ONNX model already exists${NC}"
else
    echo "Exporting YOLOv8 to ONNX..."
    python3 << EOF
from ultralytics import YOLO
model = YOLO("$MODEL_DIR/$YOLO_MODEL")
model.export(format="onnx", imgsz=640, opset=17, simplify=True)
import shutil
shutil.move("$MODEL_DIR/yolov8n.onnx", "$MODEL_DIR/$ONNX_MODEL")
print("Export complete!")
EOF
    echo -e "${GREEN}✓ Exported to $ONNX_MODEL${NC}"
fi

# -----------------------------------------------------------------------------
# Download ReID Model (OSNet)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/4] Downloading ReID model (OSNet)...${NC}"

REID_MODEL="osnet_x0_25.onnx"
REID_URL="https://github.com/opencv/opencv_zoo/raw/main/models/person_reid_youtu/person_reid_youtu_2021sep.onnx"

if [ -f "$MODEL_DIR/$REID_MODEL" ]; then
    echo -e "${GREEN}✓ ReID model already exists${NC}"
else
    echo "Downloading ReID model..."
    # Note: Using placeholder URL - replace with actual model
    echo -e "${YELLOW}⚠ ReID model download skipped (use your trained model)${NC}"
    touch "$MODEL_DIR/$REID_MODEL.placeholder"
fi

# -----------------------------------------------------------------------------
# Convert to TensorRT (if TensorRT available)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/4] Checking TensorRT conversion...${NC}"

if command -v trtexec &> /dev/null; then
    ENGINE_FILE="$MODEL_DIR/yolov8n_retail_fp16.engine"
    
    if [ -f "$ENGINE_FILE" ]; then
        echo -e "${GREEN}✓ TensorRT engine already exists${NC}"
    else
        echo "Converting to TensorRT FP16..."
        trtexec --onnx="$MODEL_DIR/$ONNX_MODEL" \
                --saveEngine="$ENGINE_FILE" \
                --fp16 \
                --workspace=4096 \
                --minShapes=images:1x3x640x640 \
                --optShapes=images:8x3x640x640 \
                --maxShapes=images:16x3x640x640
        echo -e "${GREEN}✓ Created TensorRT engine${NC}"
    fi
else
    echo -e "${YELLOW}⚠ TensorRT not found - skipping engine conversion${NC}"
    echo "  Run this script on a system with TensorRT installed"
    echo "  Or use: python -m src.edge.tensorrt_engine convert ..."
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}=======================================${NC}"
echo -e "${GREEN}  Model Setup Complete!${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""
echo "Models directory: $MODEL_DIR"
echo ""
ls -lh "$MODEL_DIR"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Fine-tune YOLOv8 on your retail dataset"
echo "  2. Convert to TensorRT for production deployment"
echo "  3. Configure model paths in configs/app_config.yaml"
