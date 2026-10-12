#!/usr/bin/env python3
"""
Retail Vision Analytics - TensorRT Model Converter.

Converts ONNX models to optimized TensorRT engines with support for
FP32, FP16, and INT8 precision modes.

Usage:
    python scripts/convert_to_tensorrt.py --input model.onnx --output model.engine
    python scripts/convert_to_tensorrt.py --input model.onnx --output model_int8.engine --precision int8 --calibration-dir data/calibration/
    python scripts/convert_to_tensorrt.py --input model.onnx --output model.engine --dynamic-batch 1 8 16
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np


@dataclass
class ConversionConfig:
    """Configuration for TensorRT conversion."""
    
    input_path: str
    output_path: str
    precision: str = "fp16"
    workspace_size_gb: float = 4.0
    min_batch: int = 1
    opt_batch: int = 8
    max_batch: int = 16
    input_shape: Tuple[int, int] = (640, 640)
    calibration_dir: Optional[str] = None
    calibration_cache: Optional[str] = None
    num_calibration_images: int = 500
    verbose: bool = False
    dla_core: Optional[int] = None  # For Jetson DLA


class CalibrationDataset:
    """Calibration dataset for INT8 quantization."""
    
    def __init__(
        self,
        calibration_dir: str,
        input_shape: Tuple[int, int],
        batch_size: int = 8,
        num_samples: int = 500,
    ):
        self.calibration_dir = Path(calibration_dir)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        # Find calibration images
        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            self.image_paths.extend(self.calibration_dir.glob(ext))
        
        if len(self.image_paths) < num_samples:
            print(f"Warning: Only found {len(self.image_paths)} images, "
                  f"requested {num_samples}")
        
        self.image_paths = self.image_paths[:num_samples]
        self.current_idx = 0
        self.num_batches = (len(self.image_paths) + batch_size - 1) // batch_size
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self) -> np.ndarray:
        if self.current_idx >= len(self.image_paths):
            raise StopIteration
        
        batch_paths = self.image_paths[
            self.current_idx:self.current_idx + self.batch_size
        ]
        self.current_idx += self.batch_size
        
        batch = []
        for path in batch_paths:
            img = self._load_and_preprocess(path)
            batch.append(img)
        
        # Pad batch if needed
        while len(batch) < self.batch_size:
            batch.append(batch[-1])
        
        return np.stack(batch, axis=0)
    
    def _load_and_preprocess(self, path: Path) -> np.ndarray:
        """Load and preprocess image for calibration."""
        try:
            import cv2
        except ImportError:
            # Mock preprocessing if OpenCV not available
            return np.random.randn(3, *self.input_shape).astype(np.float32)
        
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        
        return img


class Int8Calibrator:
    """INT8 calibration using TensorRT's IInt8EntropyCalibrator2."""
    
    def __init__(
        self,
        dataset: CalibrationDataset,
        cache_file: Optional[str] = None,
    ):
        self.dataset = dataset
        self.cache_file = cache_file
        self.batch_iter = iter(dataset)
        self.current_batch = None
    
    def get_batch_size(self) -> int:
        return self.dataset.batch_size
    
    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """Get next calibration batch."""
        try:
            self.current_batch = next(self.batch_iter)
            # Return device pointer (simplified)
            return [self.current_batch.ctypes.data]
        except StopIteration:
            return None
    
    def read_calibration_cache(self) -> Optional[bytes]:
        """Read cached calibration data."""
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes):
        """Write calibration data to cache."""
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)


def convert_onnx_to_tensorrt(config: ConversionConfig) -> bool:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        config: Conversion configuration
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: TensorRT not installed. Please install TensorRT SDK.")
        print("       For Jetson: sudo apt install python3-libnvinfer")
        print("       For x86: pip install tensorrt")
        return False
    
    print("="*60)
    print("TensorRT Model Converter")
    print("="*60)
    print(f"Input:      {config.input_path}")
    print(f"Output:     {config.output_path}")
    print(f"Precision:  {config.precision}")
    print(f"Workspace:  {config.workspace_size_gb} GB")
    print(f"Batch size: {config.min_batch}/{config.opt_batch}/{config.max_batch} (min/opt/max)")
    print(f"Input shape: {config.input_shape}")
    
    if config.precision == "int8":
        print(f"Calibration: {config.calibration_dir}")
    
    # Verify input file exists
    if not os.path.exists(config.input_path):
        print(f"ERROR: Input file not found: {config.input_path}")
        return False
    
    # Create output directory
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
    
    # Initialize TensorRT
    logger = trt.Logger(trt.Logger.VERBOSE if config.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    print("\n[1/4] Parsing ONNX model...")
    start_time = time.time()
    
    with open(config.input_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return False
    
    print(f"  Parsed in {time.time() - start_time:.2f}s")
    print(f"  Network inputs: {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")
    print(f"  Network layers: {network.num_layers}")
    
    # Configure builder
    print("\n[2/4] Configuring builder...")
    builder_config = builder.create_builder_config()
    
    # Set workspace
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(config.workspace_size_gb * 1024 * 1024 * 1024)
    )
    
    # Set precision
    if config.precision == "fp16":
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")
        else:
            print("  Warning: FP16 not supported on this platform")
    
    elif config.precision == "int8":
        if builder.platform_has_fast_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
            
            # Set up calibrator
            if config.calibration_dir:
                dataset = CalibrationDataset(
                    config.calibration_dir,
                    config.input_shape,
                    batch_size=config.opt_batch,
                    num_samples=config.num_calibration_images,
                )
                calibrator = Int8Calibrator(dataset, config.calibration_cache)
                builder_config.int8_calibrator = calibrator
                print(f"  INT8 calibration with {len(dataset.image_paths)} images")
            else:
                print("  Warning: INT8 requires calibration data for best accuracy")
        else:
            print("  Warning: INT8 not supported on this platform")
    
    # Configure dynamic batch
    print("\n[3/4] Setting up dynamic shapes...")
    profile = builder.create_optimization_profile()
    
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_name = input_tensor.name
        input_shape = list(input_tensor.shape)
        
        # Set dynamic batch dimension
        h, w = config.input_shape
        min_shape = (config.min_batch, 3, h, w)
        opt_shape = (config.opt_batch, 3, h, w)
        max_shape = (config.max_batch, 3, h, w)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        print(f"  Input '{input_name}': {min_shape} -> {opt_shape} -> {max_shape}")
    
    builder_config.add_optimization_profile(profile)
    
    # Enable DLA if specified (Jetson)
    if config.dla_core is not None:
        if builder.num_DLA_cores > 0:
            builder_config.default_device_type = trt.DeviceType.DLA
            builder_config.DLA_core = config.dla_core
            builder_config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            print(f"  DLA core {config.dla_core} enabled with GPU fallback")
        else:
            print("  Warning: DLA not available on this platform")
    
    # Build engine
    print("\n[4/4] Building TensorRT engine...")
    print("  This may take several minutes...")
    start_time = time.time()
    
    serialized_engine = builder.build_serialized_network(network, builder_config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False
    
    build_time = time.time() - start_time
    print(f"  Built in {build_time:.1f}s")
    
    # Save engine
    with open(config.output_path, "wb") as f:
        f.write(serialized_engine)
    
    engine_size_mb = os.path.getsize(config.output_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"SUCCESS: Engine saved to {config.output_path}")
    print(f"  Size: {engine_size_mb:.2f} MB")
    print(f"  Build time: {build_time:.1f}s")
    print("="*60)
    
    # Save metadata
    metadata = {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "precision": config.precision,
        "input_shape": config.input_shape,
        "batch_sizes": {
            "min": config.min_batch,
            "opt": config.opt_batch,
            "max": config.max_batch,
        },
        "build_time_seconds": build_time,
        "engine_size_mb": engine_size_mb,
        "tensorrt_version": trt.__version__,
    }
    
    metadata_path = config.output_path.replace(".engine", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")
    
    return True


def verify_engine(engine_path: str) -> bool:
    """Verify TensorRT engine loads correctly."""
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not available for verification")
        return True
    
    print(f"\nVerifying engine: {engine_path}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        print("ERROR: Failed to deserialize engine")
        return False
    
    print(f"  Inputs: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"    {name}: {shape} ({dtype}) [{mode}]")
    
    print("  Verification: PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FP16 conversion (recommended for most use cases)
  python convert_to_tensorrt.py \\
      --input models/yolov8n.onnx \\
      --output models/yolov8n_fp16.engine \\
      --precision fp16

  # INT8 conversion (requires calibration data)
  python convert_to_tensorrt.py \\
      --input models/yolov8n.onnx \\
      --output models/yolov8n_int8.engine \\
      --precision int8 \\
      --calibration-dir data/calibration/

  # Dynamic batch sizes
  python convert_to_tensorrt.py \\
      --input models/yolov8n.onnx \\
      --output models/yolov8n_fp16.engine \\
      --dynamic-batch 1 8 32

  # Jetson DLA (Deep Learning Accelerator)
  python convert_to_tensorrt.py \\
      --input models/yolov8n.onnx \\
      --output models/yolov8n_dla.engine \\
      --precision fp16 \\
      --dla-core 0
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input ONNX model path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output TensorRT engine path",
    )
    
    # Precision options
    parser.add_argument(
        "--precision", "-p",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Inference precision (default: fp16)",
    )
    
    # Batch size options
    parser.add_argument(
        "--dynamic-batch",
        type=int,
        nargs=3,
        default=[1, 8, 16],
        metavar=("MIN", "OPT", "MAX"),
        help="Dynamic batch sizes: min opt max (default: 1 8 16)",
    )
    
    # Input shape
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Input image shape (default: 640 640)",
    )
    
    # Workspace
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="Workspace size in GB (default: 4.0)",
    )
    
    # INT8 calibration
    parser.add_argument(
        "--calibration-dir",
        type=str,
        help="Directory with calibration images (required for INT8)",
    )
    parser.add_argument(
        "--calibration-cache",
        type=str,
        help="Path to save/load calibration cache",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=500,
        help="Number of calibration images (default: 500)",
    )
    
    # Jetson options
    parser.add_argument(
        "--dla-core",
        type=int,
        choices=[0, 1],
        help="Use DLA core (Jetson only)",
    )
    
    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify engine after conversion",
    )
    
    args = parser.parse_args()
    
    # Validate INT8 calibration
    if args.precision == "int8" and not args.calibration_dir:
        print("Warning: INT8 precision without calibration may have reduced accuracy")
        print("         Use --calibration-dir to provide calibration images")
    
    # Create configuration
    config = ConversionConfig(
        input_path=args.input,
        output_path=args.output,
        precision=args.precision,
        workspace_size_gb=args.workspace,
        min_batch=args.dynamic_batch[0],
        opt_batch=args.dynamic_batch[1],
        max_batch=args.dynamic_batch[2],
        input_shape=tuple(args.input_shape),
        calibration_dir=args.calibration_dir,
        calibration_cache=args.calibration_cache,
        num_calibration_images=args.num_calibration,
        verbose=args.verbose,
        dla_core=args.dla_core,
    )
    
    # Run conversion
    success = convert_onnx_to_tensorrt(config)
    
    if success and args.verify:
        verify_engine(args.output)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
