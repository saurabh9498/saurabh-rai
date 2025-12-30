#!/usr/bin/env python3
"""
Build TensorRT Engine Script

Builds optimized TensorRT engines from ONNX models.

Usage:
    python scripts/build_engine.py --onnx models/resnet50.onnx --output engines/resnet50.engine
    python scripts/build_engine.py --config configs/tensorrt_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tensorrt.builder import TensorRTBuilder, BuildConfig, OptimizationProfile


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    
    # Input/Output
    parser.add_argument("--onnx", type=str, help="Path to ONNX model")
    parser.add_argument("--output", type=str, help="Output engine path")
    parser.add_argument("--config", type=str, help="YAML configuration file")
    
    # Build options
    parser.add_argument("--precision", type=str, default="fp16",
                       choices=["fp32", "fp16", "int8"],
                       help="Precision mode")
    parser.add_argument("--max-batch-size", type=int, default=32,
                       help="Maximum batch size")
    parser.add_argument("--workspace", type=float, default=4.0,
                       help="Workspace size in GB")
    parser.add_argument("--opt-level", type=int, default=5,
                       help="Optimization level (0-5)")
    
    # Dynamic shapes
    parser.add_argument("--min-shape", type=str,
                       help="Min input shape (e.g., '1,3,224,224')")
    parser.add_argument("--opt-shape", type=str,
                       help="Optimal input shape")
    parser.add_argument("--max-shape", type=str,
                       help="Max input shape")
    
    # INT8 calibration
    parser.add_argument("--calib-data", type=str,
                       help="Calibration data directory")
    parser.add_argument("--calib-cache", type=str,
                       help="Calibration cache file")
    parser.add_argument("--calib-batches", type=int, default=500,
                       help="Number of calibration batches")
    
    # Other
    parser.add_argument("--timing-cache", type=str,
                       help="Timing cache file")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--verify", action="store_true",
                       help="Verify engine after build")
    
    return parser.parse_args()


def parse_shape(shape_str: str) -> tuple:
    """Parse shape string like '1,3,224,224' to tuple."""
    return tuple(int(x) for x in shape_str.split(","))


def build_from_config(config_path: str):
    """Build engine from YAML config file."""
    import yaml
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Create build config
    build_config = BuildConfig(
        precision=cfg.get("builder", {}).get("precision", "fp16"),
        max_batch_size=cfg.get("builder", {}).get("max_batch_size", 32),
        workspace_size_gb=cfg.get("builder", {}).get("workspace_size_gb", 4.0),
        optimization_level=cfg.get("builder", {}).get("optimization_level", 5),
        enable_timing_cache=cfg.get("builder", {}).get("enable_timing_cache", True),
        timing_cache_path=cfg.get("builder", {}).get("timing_cache_path"),
        calibration_cache_path=cfg.get("calibration", {}).get("cache_file"),
    )
    
    # Parse dynamic shapes
    if "dynamic_shapes" in cfg:
        for name, shapes in cfg["dynamic_shapes"].items():
            build_config.dynamic_shapes[name] = OptimizationProfile(
                name=name,
                min_shape=tuple(shapes["min"]),
                opt_shape=tuple(shapes["opt"]),
                max_shape=tuple(shapes["max"]),
            )
    
    # Get calibration data if INT8
    calib_data = None
    if build_config.precision == "int8" and "calibration" in cfg:
        calib_cfg = cfg["calibration"]
        if "data_dir" in calib_cfg:
            from src.tensorrt.calibrator import CalibrationDataLoader
            calib_data = CalibrationDataLoader.from_directory(
                calib_cfg["data_dir"],
                batch_size=calib_cfg.get("batch_size", 8),
            )
    
    # Build engine
    onnx_path = cfg["model"]["onnx_path"]
    output_path = cfg["model"]["output_path"]
    
    builder = TensorRTBuilder(
        onnx_path=onnx_path,
        config=build_config,
        calibration_data=calib_data,
    )
    
    engine = builder.build()
    engine.save(output_path)
    
    logger.info(f"Engine saved to: {output_path}")
    return output_path


def build_from_args(args):
    """Build engine from command line arguments."""
    if not args.onnx:
        raise ValueError("--onnx is required when not using --config")
    
    # Create output path if not specified
    output_path = args.output
    if not output_path:
        onnx_path = Path(args.onnx)
        output_path = str(onnx_path.with_suffix(f".{args.precision}.engine"))
    
    # Create build config
    build_config = BuildConfig(
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        workspace_size_gb=args.workspace,
        optimization_level=args.opt_level,
        enable_timing_cache=args.timing_cache is not None,
        timing_cache_path=args.timing_cache,
        calibration_cache_path=args.calib_cache,
    )
    
    # Parse dynamic shapes
    if args.min_shape and args.opt_shape and args.max_shape:
        build_config.dynamic_shapes["input"] = OptimizationProfile(
            name="input",
            min_shape=parse_shape(args.min_shape),
            opt_shape=parse_shape(args.opt_shape),
            max_shape=parse_shape(args.max_shape),
        )
    
    # Get calibration data if INT8
    calib_data = None
    if args.precision == "int8" and args.calib_data:
        from src.tensorrt.calibrator import CalibrationDataLoader
        calib_data = CalibrationDataLoader.from_directory(
            args.calib_data,
            batch_size=8,
        )
    
    # Build engine
    builder = TensorRTBuilder(
        onnx_path=args.onnx,
        config=build_config,
        calibration_data=calib_data,
        verbose=args.verbose,
    )
    
    logger.info(f"Building TensorRT engine...")
    logger.info(f"  ONNX: {args.onnx}")
    logger.info(f"  Precision: {args.precision}")
    logger.info(f"  Max batch size: {args.max_batch_size}")
    
    engine = builder.build()
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    engine.save(output_path)
    
    logger.info(f"Engine saved to: {output_path}")
    
    # Verify if requested
    if args.verify:
        verify_engine(output_path, args.onnx)
    
    return output_path


def verify_engine(engine_path: str, onnx_path: str):
    """Verify engine produces similar outputs to ONNX."""
    logger.info("Verifying engine...")
    
    import numpy as np
    import onnxruntime as ort
    from src.tensorrt.inference import TensorRTInference
    
    # Load models
    trt_engine = TensorRTInference(engine_path)
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input shape
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    
    # Handle dynamic dimensions
    shape = [s if isinstance(s, int) else 1 for s in input_shape]
    
    # Generate test input
    test_input = np.random.randn(*shape).astype(np.float32)
    
    # Run inference
    trt_output = trt_engine.infer(test_input)
    ort_output = ort_session.run(None, {input_name: test_input})[0]
    
    # Compare outputs
    max_diff = np.abs(trt_output - ort_output).max()
    mean_diff = np.abs(trt_output - ort_output).mean()
    
    logger.info(f"  Max difference: {max_diff:.6f}")
    logger.info(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 0.01:
        logger.info("  ✓ Verification passed!")
    else:
        logger.warning("  ⚠ Large difference detected (may be expected for INT8)")


def main():
    args = parse_args()
    
    try:
        if args.config:
            build_from_config(args.config)
        else:
            build_from_args(args)
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
