#!/usr/bin/env bash
# Compile the exported ONNX dense scorer into a TensorRT INT8 engine.
#
# Designed for the CI/CD pipeline; NOT run in this repo's CPU checks.
# Run it ON the target GPU type (an A10G builder node): TensorRT picks kernels for the
# exact silicon it builds on, so an engine built on an H100 will not load on an A10G.
#
# Inputs:  artifacts/dense_scorer.onnx       (from model/export_onnx.py)
#          artifacts/calibration.cache       (from build/calibrator.py, KL/entropy calibration)
# Output:  serving/model_repository/dlrm_dense/1/model.plan
#
# Version note: this uses TensorRT 8.6-style implicit INT8 calibration (--int8 --calib).
# TensorRT 10 deprecates implicit quantization; on TRT 10 the equivalent is explicit Q/DQ
# nodes inserted by NVIDIA TensorRT Model Optimizer (modelopt) before export.
set -euo pipefail

ONNX=${ONNX:-artifacts/dense_scorer.onnx}
CALIB=${CALIB:-artifacts/calibration.cache}
OUT=${OUT:-serving/model_repository/dlrm_dense/1/model.plan}
NUM_SPARSE=${NUM_SPARSE:-26}
EMB_DIM=${EMB_DIM:-64}
NUM_DENSE=${NUM_DENSE:-13}

# Workspace for tactic selection at build time. The 12 GB *runtime* workspace in the
# VRAM budget is a separate reservation (activations for 128-request batches + CUDA
# context); see sizing/vram_budget.py.
WORKSPACE_MB=${WORKSPACE_MB:-8192}

mkdir -p "$(dirname "$OUT")"

trtexec \
  --onnx="$ONNX" \
  --saveEngine="$OUT" \
  --int8 --fp16 \
  --calib="$CALIB" \
  --memPoolSize=workspace:"$WORKSPACE_MB" \
  --minShapes=dense:1x${NUM_DENSE},embeddings:1x${NUM_SPARSE}x${EMB_DIM} \
  --optShapes=dense:64x${NUM_DENSE},embeddings:64x${NUM_SPARSE}x${EMB_DIM} \
  --maxShapes=dense:128x${NUM_DENSE},embeddings:128x${NUM_SPARSE}x${EMB_DIM} \
  --timingCacheFile=artifacts/timing.cache

# --int8 with --fp16 lets TensorRT keep any layer in FP16 where INT8 would lose accuracy
# or is not faster. max batch 128 matches Triton's dynamic batcher (serving/.../config.pbtxt).
echo "engine written to $OUT"
