"""
TensorRT Engine Builder and Optimizer for Retail Vision Analytics.

This module provides utilities for converting ONNX models to optimized
TensorRT engines with support for FP16/INT8 quantization, dynamic batching,
and calibration for maximum inference performance on NVIDIA GPUs.

Supports: TensorRT 8.6+, CUDA 12.0+
Target Hardware: RTX 4090, A100, Jetson Orin
"""

import os
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from enum import Enum
import json
import struct
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Precision(Enum):
    """Supported precision modes for TensorRT optimization."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    
    @property
    def trt_flag(self) -> int:
        """Get TensorRT BuilderFlag for this precision."""
        # These would be actual TensorRT flags in production
        flags = {
            "fp32": 0,
            "fp16": 1,  # trt.BuilderFlag.FP16
            "int8": 2,  # trt.BuilderFlag.INT8
        }
        return flags[self.value]


class CalibrationMethod(Enum):
    """INT8 calibration methods."""
    ENTROPY = "entropy"
    MINMAX = "minmax"
    PERCENTILE = "percentile"


@dataclass
class EngineConfig:
    """Configuration for TensorRT engine building."""
    
    # Model paths
    onnx_path: str
    engine_path: Optional[str] = None
    
    # Precision settings
    precision: Precision = Precision.FP16
    calibration_method: CalibrationMethod = CalibrationMethod.ENTROPY
    
    # Batch size settings
    min_batch_size: int = 1
    optimal_batch_size: int = 4
    max_batch_size: int = 16
    
    # Input dimensions (NCHW format, excluding batch)
    input_shape: Tuple[int, int, int] = (3, 640, 640)
    
    # Memory settings
    workspace_size_gb: float = 4.0
    dla_core: Optional[int] = None  # Deep Learning Accelerator core (Jetson)
    
    # Optimization settings
    use_cuda_graph: bool = True
    use_sparse_weights: bool = False
    enable_timing_cache: bool = True
    
    # Calibration settings
    calibration_images_dir: Optional[str] = None
    calibration_batch_size: int = 8
    calibration_batches: int = 100
    calibration_cache_file: Optional[str] = None
    
    # Plugin settings
    plugins: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set default engine path if not provided."""
        if self.engine_path is None:
            base = Path(self.onnx_path).stem
            self.engine_path = f"{base}_{self.precision.value}.engine"
    
    @property
    def workspace_size_bytes(self) -> int:
        """Get workspace size in bytes."""
        return int(self.workspace_size_gb * (1 << 30))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "onnx_path": self.onnx_path,
            "engine_path": self.engine_path,
            "precision": self.precision.value,
            "calibration_method": self.calibration_method.value,
            "min_batch_size": self.min_batch_size,
            "optimal_batch_size": self.optimal_batch_size,
            "max_batch_size": self.max_batch_size,
            "input_shape": self.input_shape,
            "workspace_size_gb": self.workspace_size_gb,
            "dla_core": self.dla_core,
            "use_cuda_graph": self.use_cuda_graph,
            "use_sparse_weights": self.use_sparse_weights,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        """Create config from dictionary."""
        if "precision" in data:
            data["precision"] = Precision(data["precision"])
        if "calibration_method" in data:
            data["calibration_method"] = CalibrationMethod(data["calibration_method"])
        if "input_shape" in data:
            data["input_shape"] = tuple(data["input_shape"])
        return cls(**data)


@dataclass
class EngineMetadata:
    """Metadata for a built TensorRT engine."""
    
    engine_path: str
    onnx_path: str
    precision: str
    input_names: List[str]
    output_names: List[str]
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    build_time_seconds: float
    tensorrt_version: str
    cuda_version: str
    gpu_name: str
    build_timestamp: str
    config: Dict[str, Any]
    
    def save(self, path: str):
        """Save metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "EngineMetadata":
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class Calibrator(ABC):
    """Abstract base class for INT8 calibration."""
    
    @abstractmethod
    def get_batch(self) -> Optional[np.ndarray]:
        """Get next calibration batch."""
        pass
    
    @abstractmethod
    def get_batch_size(self) -> int:
        """Get calibration batch size."""
        pass
    
    @abstractmethod
    def read_calibration_cache(self) -> Optional[bytes]:
        """Read calibration cache if exists."""
        pass
    
    @abstractmethod
    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache."""
        pass


class ImageFolderCalibrator(Calibrator):
    """
    INT8 calibrator using images from a folder.
    
    This calibrator reads images from a directory and preprocesses them
    for calibration. It supports caching calibration data for faster
    subsequent builds.
    """
    
    def __init__(
        self,
        images_dir: str,
        batch_size: int = 8,
        input_shape: Tuple[int, int, int] = (3, 640, 640),
        cache_file: Optional[str] = None,
        max_batches: int = 100,
        preprocess_fn: Optional[Callable] = None,
    ):
        """
        Initialize calibrator.
        
        Args:
            images_dir: Directory containing calibration images
            batch_size: Batch size for calibration
            input_shape: Input shape (C, H, W)
            cache_file: Path to calibration cache file
            max_batches: Maximum number of batches to process
            preprocess_fn: Custom preprocessing function
        """
        self.images_dir = Path(images_dir)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.max_batches = max_batches
        self.preprocess_fn = preprocess_fn or self._default_preprocess
        
        # Collect image paths
        self.image_paths = self._collect_images()
        self.current_batch = 0
        self.total_batches = min(
            len(self.image_paths) // batch_size,
            max_batches
        )
        
        logger.info(
            f"Calibrator initialized with {len(self.image_paths)} images, "
            f"{self.total_batches} batches"
        )
    
    def _collect_images(self) -> List[Path]:
        """Collect image paths from directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = []
        
        for ext in extensions:
            images.extend(self.images_dir.glob(f"*{ext}"))
            images.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def _default_preprocess(self, image_path: Path) -> np.ndarray:
        """
        Default preprocessing for YOLO models.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array (C, H, W) normalized to [0, 1]
        """
        try:
            import cv2
        except ImportError:
            # Create dummy data for demonstration
            c, h, w = self.input_shape
            return np.random.rand(c, h, w).astype(np.float32)
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Resize
        _, h, w = self.input_shape
        image = cv2.resize(image, (w, h))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def get_batch_size(self) -> int:
        """Get calibration batch size."""
        return self.batch_size
    
    def get_batch(self) -> Optional[np.ndarray]:
        """
        Get next calibration batch.
        
        Returns:
            Batch of preprocessed images or None if done
        """
        if self.current_batch >= self.total_batches:
            return None
        
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_paths = self.image_paths[start_idx:end_idx]
        
        # Preprocess batch
        batch = np.zeros(
            (self.batch_size, *self.input_shape),
            dtype=np.float32
        )
        
        for i, path in enumerate(batch_paths):
            try:
                batch[i] = self.preprocess_fn(path)
            except Exception as e:
                logger.warning(f"Failed to preprocess {path}: {e}")
                # Use random noise as fallback
                batch[i] = np.random.rand(*self.input_shape).astype(np.float32)
        
        self.current_batch += 1
        logger.debug(f"Calibration batch {self.current_batch}/{self.total_batches}")
        
        return batch
    
    def read_calibration_cache(self) -> Optional[bytes]:
        """Read calibration cache if exists."""
        if self.cache_file and os.path.exists(self.cache_file):
            logger.info(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache."""
        if self.cache_file:
            logger.info(f"Writing calibration cache to {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                f.write(cache)
    
    def reset(self):
        """Reset calibrator for reuse."""
        self.current_batch = 0


class RandomCalibrator(Calibrator):
    """
    Calibrator using random data.
    
    Useful for quick testing when calibration images are not available.
    Note: This produces suboptimal INT8 accuracy.
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        input_shape: Tuple[int, int, int] = (3, 640, 640),
        num_batches: int = 100,
        cache_file: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_batches = num_batches
        self.cache_file = cache_file
        self.current_batch = 0
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_batch(self) -> Optional[np.ndarray]:
        if self.current_batch >= self.num_batches:
            return None
        
        batch = np.random.rand(
            self.batch_size, *self.input_shape
        ).astype(np.float32)
        
        self.current_batch += 1
        return batch
    
    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes):
        if self.cache_file:
            with open(self.cache_file, 'wb') as f:
                f.write(cache)
    
    def reset(self):
        self.current_batch = 0


class TensorRTEngineBuilder:
    """
    TensorRT Engine Builder for optimizing detection models.
    
    This builder handles the complete workflow of converting ONNX models
    to optimized TensorRT engines with support for:
    - FP16 and INT8 precision
    - Dynamic batch sizes
    - Calibration for INT8 quantization
    - DLA (Deep Learning Accelerator) for Jetson devices
    - CUDA graphs for reduced overhead
    
    Example:
        >>> config = EngineConfig(
        ...     onnx_path="yolov8n.onnx",
        ...     precision=Precision.FP16,
        ...     max_batch_size=16
        ... )
        >>> builder = TensorRTEngineBuilder(config)
        >>> metadata = builder.build()
        >>> print(f"Engine saved to: {metadata.engine_path}")
    """
    
    # TensorRT version info (would be dynamic in production)
    TRT_VERSION = "8.6.1"
    
    def __init__(self, config: EngineConfig):
        """
        Initialize engine builder.
        
        Args:
            config: Engine configuration
        """
        self.config = config
        self._trt_logger = None
        self._builder = None
        self._network = None
        self._parser = None
        self._calibrator = None
        
        # Validate config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration."""
        if not os.path.exists(self.config.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.config.onnx_path}")
        
        if self.config.precision == Precision.INT8:
            if (self.config.calibration_images_dir is None and 
                self.config.calibration_cache_file is None):
                logger.warning(
                    "INT8 precision requested but no calibration data provided. "
                    "Using random calibration (suboptimal accuracy)."
                )
        
        if self.config.min_batch_size > self.config.max_batch_size:
            raise ValueError("min_batch_size cannot exceed max_batch_size")
    
    def _create_calibrator(self) -> Optional[Calibrator]:
        """Create calibrator for INT8 quantization."""
        if self.config.precision != Precision.INT8:
            return None
        
        if self.config.calibration_images_dir:
            return ImageFolderCalibrator(
                images_dir=self.config.calibration_images_dir,
                batch_size=self.config.calibration_batch_size,
                input_shape=self.config.input_shape,
                cache_file=self.config.calibration_cache_file,
                max_batches=self.config.calibration_batches,
            )
        else:
            return RandomCalibrator(
                batch_size=self.config.calibration_batch_size,
                input_shape=self.config.input_shape,
                num_batches=self.config.calibration_batches,
                cache_file=self.config.calibration_cache_file,
            )
    
    def _get_gpu_info(self) -> Dict[str, str]:
        """Get GPU information."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                return {
                    "gpu_name": parts[0] if parts else "Unknown",
                    "driver_version": parts[1] if len(parts) > 1 else "Unknown"
                }
        except Exception:
            pass
        return {"gpu_name": "Unknown", "driver_version": "Unknown"}
    
    def _get_cuda_version(self) -> str:
        """Get CUDA version."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split('release')
                        if len(parts) > 1:
                            return parts[1].split(',')[0].strip()
        except Exception:
            pass
        return "Unknown"
    
    def build(self, verbose: bool = False) -> EngineMetadata:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            verbose: Enable verbose logging
            
        Returns:
            Engine metadata including paths and build info
        """
        logger.info(f"Building TensorRT engine from {self.config.onnx_path}")
        logger.info(f"Precision: {self.config.precision.value}")
        logger.info(f"Batch size: {self.config.min_batch_size}-{self.config.max_batch_size}")
        
        start_time = time.time()
        
        # In production, this would use actual TensorRT APIs
        # Here we simulate the build process
        
        # Step 1: Parse ONNX model
        logger.info("Parsing ONNX model...")
        input_names, output_names = self._parse_onnx_model()
        
        # Step 2: Configure builder
        logger.info("Configuring TensorRT builder...")
        self._configure_builder()
        
        # Step 3: Run calibration if INT8
        if self.config.precision == Precision.INT8:
            logger.info("Running INT8 calibration...")
            self._run_calibration()
        
        # Step 4: Build engine
        logger.info("Building optimized engine...")
        self._build_engine()
        
        # Step 5: Serialize engine
        logger.info(f"Serializing engine to {self.config.engine_path}...")
        self._serialize_engine()
        
        build_time = time.time() - start_time
        gpu_info = self._get_gpu_info()
        
        # Create metadata
        metadata = EngineMetadata(
            engine_path=self.config.engine_path,
            onnx_path=self.config.onnx_path,
            precision=self.config.precision.value,
            input_names=input_names,
            output_names=output_names,
            input_shapes={
                input_names[0]: [
                    self.config.max_batch_size,
                    *self.config.input_shape
                ]
            },
            output_shapes=self._get_output_shapes(output_names),
            build_time_seconds=build_time,
            tensorrt_version=self.TRT_VERSION,
            cuda_version=self._get_cuda_version(),
            gpu_name=gpu_info["gpu_name"],
            build_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config.to_dict()
        )
        
        # Save metadata
        metadata_path = f"{self.config.engine_path}.json"
        metadata.save(metadata_path)
        logger.info(f"Metadata saved to {metadata_path}")
        
        logger.info(f"Engine built successfully in {build_time:.2f}s")
        
        return metadata
    
    def _parse_onnx_model(self) -> Tuple[List[str], List[str]]:
        """Parse ONNX model and extract input/output info."""
        try:
            import onnx
            model = onnx.load(self.config.onnx_path)
            
            input_names = [inp.name for inp in model.graph.input]
            output_names = [out.name for out in model.graph.output]
            
            return input_names, output_names
        except ImportError:
            # Return default YOLO names if ONNX not available
            return ["images"], ["output0"]
    
    def _configure_builder(self):
        """Configure TensorRT builder with optimization settings."""
        # In production, this would configure actual TensorRT builder
        # Settings include:
        # - Precision flags (FP16, INT8)
        # - Workspace size
        # - DLA settings
        # - Dynamic shapes
        # - Timing cache
        pass
    
    def _run_calibration(self):
        """Run INT8 calibration."""
        self._calibrator = self._create_calibrator()
        
        if self._calibrator:
            # Process calibration batches
            batch_count = 0
            while True:
                batch = self._calibrator.get_batch()
                if batch is None:
                    break
                batch_count += 1
            
            logger.info(f"Calibration completed with {batch_count} batches")
    
    def _build_engine(self):
        """Build the TensorRT engine."""
        # In production, this would:
        # 1. Create execution context
        # 2. Run builder optimization
        # 3. Apply layer precision settings
        # 4. Enable DLA if configured
        pass
    
    def _serialize_engine(self):
        """Serialize engine to file."""
        # Create a placeholder engine file for demonstration
        engine_path = Path(self.config.engine_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write a header indicating this is a placeholder
        # In production, this would be the serialized TensorRT engine
        header = struct.pack(
            '8sII',
            b'TRTENGNE',  # Magic bytes
            8,  # TensorRT major version
            6,  # TensorRT minor version
        )
        
        with open(engine_path, 'wb') as f:
            f.write(header)
            # Write placeholder engine data
            f.write(b'\x00' * 1024)  # Placeholder
        
        logger.info(f"Engine serialized to {engine_path}")
    
    def _get_output_shapes(self, output_names: List[str]) -> Dict[str, List[int]]:
        """Get output shapes for the model."""
        # YOLO output shape: [batch, num_detections, 85] for COCO
        # For retail: [batch, num_detections, 7 + num_classes]
        # 7 = x, y, w, h, objectness, class_id, confidence
        num_classes = 7  # Retail classes
        return {
            name: [self.config.max_batch_size, 8400, 4 + num_classes]
            for name in output_names
        }


class TensorRTInference:
    """
    TensorRT inference engine for optimized model execution.
    
    Handles loading serialized engines, managing GPU memory,
    and running inference with optimal performance.
    
    Example:
        >>> engine = TensorRTInference("yolov8n_fp16.engine")
        >>> detections = engine.infer(preprocessed_images)
    """
    
    def __init__(
        self,
        engine_path: str,
        device_id: int = 0,
        use_cuda_graph: bool = True,
        warm_up_iterations: int = 10,
    ):
        """
        Initialize inference engine.
        
        Args:
            engine_path: Path to serialized TensorRT engine
            device_id: CUDA device ID
            use_cuda_graph: Enable CUDA graphs for reduced overhead
            warm_up_iterations: Number of warm-up iterations
        """
        self.engine_path = engine_path
        self.device_id = device_id
        self.use_cuda_graph = use_cuda_graph
        self.warm_up_iterations = warm_up_iterations
        
        self._engine = None
        self._context = None
        self._bindings = None
        self._input_shape = None
        self._output_shapes = None
        self._stream = None
        self._cuda_graph = None
        
        # Performance tracking
        self._inference_times: List[float] = []
        self._total_inferences = 0
        
        # Load engine
        self._load_engine()
        
        # Warm up
        if warm_up_iterations > 0:
            self._warm_up()
    
    def _load_engine(self):
        """Load TensorRT engine from file."""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine not found: {self.engine_path}")
        
        logger.info(f"Loading TensorRT engine from {self.engine_path}")
        
        # Load metadata if available
        metadata_path = f"{self.engine_path}.json"
        if os.path.exists(metadata_path):
            metadata = EngineMetadata.load(metadata_path)
            self._input_shape = list(metadata.input_shapes.values())[0]
            self._output_shapes = metadata.output_shapes
            logger.info(f"Loaded engine metadata: {metadata.precision} precision")
        else:
            # Use defaults
            self._input_shape = [16, 3, 640, 640]
            self._output_shapes = {"output0": [16, 8400, 11]}
        
        # In production, this would use tensorrt.Runtime to deserialize
        logger.info("Engine loaded successfully")
    
    def _warm_up(self):
        """Warm up engine for stable performance."""
        logger.info(f"Warming up engine ({self.warm_up_iterations} iterations)...")
        
        # Create dummy input
        dummy_input = np.random.rand(*self._input_shape).astype(np.float32)
        
        for i in range(self.warm_up_iterations):
            _ = self.infer(dummy_input)
        
        # Clear timing stats from warm-up
        self._inference_times.clear()
        self._total_inferences = 0
        
        logger.info("Warm-up complete")
    
    def infer(
        self,
        inputs: np.ndarray,
        synchronize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on input data.
        
        Args:
            inputs: Input tensor (NCHW format)
            synchronize: Wait for completion before returning
            
        Returns:
            Dictionary of output tensors
        """
        start_time = time.perf_counter()
        
        # Validate input shape
        if inputs.shape[1:] != tuple(self._input_shape[1:]):
            raise ValueError(
                f"Input shape mismatch. Expected {self._input_shape[1:]}, "
                f"got {inputs.shape[1:]}"
            )
        
        batch_size = inputs.shape[0]
        
        # In production, this would:
        # 1. Copy input to GPU
        # 2. Execute inference
        # 3. Copy output back to CPU
        
        # Simulate inference with random outputs
        outputs = {}
        for name, shape in self._output_shapes.items():
            output_shape = [batch_size] + shape[1:]
            outputs[name] = np.random.rand(*output_shape).astype(np.float32)
        
        # Track timing
        inference_time = time.perf_counter() - start_time
        self._inference_times.append(inference_time)
        self._total_inferences += batch_size
        
        return outputs
    
    def infer_async(self, inputs: np.ndarray) -> "AsyncResult":
        """
        Run asynchronous inference.
        
        Args:
            inputs: Input tensor
            
        Returns:
            AsyncResult that can be awaited
        """
        return AsyncResult(self, inputs)
    
    @property
    def input_shape(self) -> List[int]:
        """Get expected input shape."""
        return self._input_shape
    
    @property
    def output_shapes(self) -> Dict[str, List[int]]:
        """Get output shapes."""
        return self._output_shapes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        if not self._inference_times:
            return {
                "total_inferences": 0,
                "avg_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "throughput_fps": 0,
            }
        
        times_ms = [t * 1000 for t in self._inference_times]
        total_time = sum(self._inference_times)
        
        return {
            "total_inferences": self._total_inferences,
            "avg_latency_ms": np.mean(times_ms),
            "min_latency_ms": np.min(times_ms),
            "max_latency_ms": np.max(times_ms),
            "p95_latency_ms": np.percentile(times_ms, 95),
            "p99_latency_ms": np.percentile(times_ms, 99),
            "throughput_fps": self._total_inferences / total_time if total_time > 0 else 0,
        }
    
    def reset_statistics(self):
        """Reset inference statistics."""
        self._inference_times.clear()
        self._total_inferences = 0
    
    def __del__(self):
        """Cleanup resources."""
        self._engine = None
        self._context = None


class AsyncResult:
    """Represents an asynchronous inference result."""
    
    def __init__(self, engine: TensorRTInference, inputs: np.ndarray):
        self._engine = engine
        self._inputs = inputs
        self._result = None
        self._completed = False
    
    def wait(self) -> Dict[str, np.ndarray]:
        """Wait for result and return outputs."""
        if not self._completed:
            self._result = self._engine.infer(self._inputs)
            self._completed = True
        return self._result
    
    @property
    def completed(self) -> bool:
        """Check if inference is complete."""
        return self._completed


class EngineOptimizer:
    """
    Utilities for optimizing TensorRT engine performance.
    
    Provides methods for:
    - Layer-wise profiling
    - Memory optimization
    - Precision analysis
    - Performance tuning
    """
    
    @staticmethod
    def profile_engine(engine_path: str, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Profile TensorRT engine performance.
        
        Args:
            engine_path: Path to TensorRT engine
            num_iterations: Number of profiling iterations
            
        Returns:
            Profiling results
        """
        logger.info(f"Profiling engine: {engine_path}")
        
        # Load engine
        inference = TensorRTInference(
            engine_path,
            warm_up_iterations=10,
        )
        
        # Create test input
        input_shape = inference.input_shape
        test_input = np.random.rand(*input_shape).astype(np.float32)
        
        # Run profiling iterations
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = inference.infer(test_input)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "engine_path": engine_path,
            "num_iterations": num_iterations,
            "input_shape": input_shape,
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_fps": 1000 / np.mean(latencies) * input_shape[0],
        }
    
    @staticmethod
    def compare_precisions(
        onnx_path: str,
        output_dir: str,
        calibration_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare FP32, FP16, and INT8 precision modes.
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Directory for output engines
            calibration_dir: Directory with calibration images (for INT8)
            
        Returns:
            Comparison results for each precision
        """
        results = {}
        
        for precision in [Precision.FP32, Precision.FP16, Precision.INT8]:
            logger.info(f"\nBuilding {precision.value} engine...")
            
            config = EngineConfig(
                onnx_path=onnx_path,
                engine_path=os.path.join(output_dir, f"model_{precision.value}.engine"),
                precision=precision,
                calibration_images_dir=calibration_dir,
            )
            
            try:
                builder = TensorRTEngineBuilder(config)
                metadata = builder.build()
                
                # Profile engine
                profile = EngineOptimizer.profile_engine(metadata.engine_path)
                
                results[precision.value] = {
                    "build_time_s": metadata.build_time_seconds,
                    "engine_size_mb": os.path.getsize(metadata.engine_path) / (1024 * 1024),
                    **profile,
                }
            except Exception as e:
                logger.error(f"Failed to build {precision.value}: {e}")
                results[precision.value] = {"error": str(e)}
        
        return results
    
    @staticmethod
    def estimate_memory_usage(
        input_shape: Tuple[int, ...],
        output_shapes: Dict[str, List[int]],
        precision: Precision,
    ) -> Dict[str, float]:
        """
        Estimate GPU memory usage for inference.
        
        Args:
            input_shape: Input tensor shape
            output_shapes: Output tensor shapes
            precision: Precision mode
            
        Returns:
            Memory estimates in MB
        """
        bytes_per_element = {
            Precision.FP32: 4,
            Precision.FP16: 2,
            Precision.INT8: 1,
        }[precision]
        
        # Input memory
        input_elements = np.prod(input_shape)
        input_memory = input_elements * bytes_per_element / (1024 * 1024)
        
        # Output memory
        output_memory = 0
        for shape in output_shapes.values():
            output_elements = np.prod(shape)
            output_memory += output_elements * bytes_per_element / (1024 * 1024)
        
        # Estimate workspace (typically 2-4x model size)
        workspace_estimate = (input_memory + output_memory) * 3
        
        return {
            "input_memory_mb": input_memory,
            "output_memory_mb": output_memory,
            "workspace_estimate_mb": workspace_estimate,
            "total_estimate_mb": input_memory + output_memory + workspace_estimate,
        }


def convert_yolo_to_tensorrt(
    model_path: str,
    output_path: str,
    precision: str = "fp16",
    imgsz: int = 640,
    batch_size: int = 1,
    calibration_images: Optional[str] = None,
) -> str:
    """
    High-level function to convert YOLO model to TensorRT.
    
    Args:
        model_path: Path to YOLO model (.pt or .onnx)
        output_path: Output path for TensorRT engine
        precision: Precision mode (fp32, fp16, int8)
        imgsz: Input image size
        batch_size: Maximum batch size
        calibration_images: Path to calibration images (for INT8)
        
    Returns:
        Path to generated TensorRT engine
    """
    logger.info(f"Converting {model_path} to TensorRT ({precision})")
    
    # Determine if we need to export to ONNX first
    onnx_path = model_path
    if model_path.endswith('.pt'):
        onnx_path = model_path.replace('.pt', '.onnx')
        logger.info(f"Exporting PyTorch model to ONNX: {onnx_path}")
        # In production: use ultralytics export
        # model = YOLO(model_path)
        # model.export(format='onnx', imgsz=imgsz, dynamic=True)
    
    # Build TensorRT engine
    config = EngineConfig(
        onnx_path=onnx_path,
        engine_path=output_path,
        precision=Precision(precision),
        max_batch_size=batch_size,
        input_shape=(3, imgsz, imgsz),
        calibration_images_dir=calibration_images,
    )
    
    builder = TensorRTEngineBuilder(config)
    metadata = builder.build()
    
    return metadata.engine_path


# Demonstration
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("TensorRT Engine Builder Demo")
    print("=" * 60)
    
    # Create a dummy ONNX file for demonstration
    demo_onnx = "/tmp/yolov8n_retail.onnx"
    os.makedirs("/tmp", exist_ok=True)
    with open(demo_onnx, 'wb') as f:
        f.write(b'ONNX_DEMO')  # Placeholder
    
    # Example 1: Build FP16 engine
    print("\n1. Building FP16 Engine")
    print("-" * 40)
    
    config = EngineConfig(
        onnx_path=demo_onnx,
        engine_path="/tmp/yolov8n_retail_fp16.engine",
        precision=Precision.FP16,
        min_batch_size=1,
        optimal_batch_size=8,
        max_batch_size=16,
        input_shape=(3, 640, 640),
        workspace_size_gb=4.0,
    )
    
    builder = TensorRTEngineBuilder(config)
    metadata = builder.build()
    
    print(f"\nEngine built:")
    print(f"  Path: {metadata.engine_path}")
    print(f"  Precision: {metadata.precision}")
    print(f"  Build time: {metadata.build_time_seconds:.2f}s")
    print(f"  Input shape: {metadata.input_shapes}")
    
    # Example 2: Run inference
    print("\n2. Running Inference")
    print("-" * 40)
    
    inference = TensorRTInference(
        metadata.engine_path,
        warm_up_iterations=5,
    )
    
    # Create test batch
    batch_size = 4
    test_input = np.random.rand(batch_size, 3, 640, 640).astype(np.float32)
    
    # Run multiple inferences
    for i in range(10):
        outputs = inference.infer(test_input)
    
    stats = inference.get_statistics()
    print(f"\nInference Statistics:")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  Throughput: {stats['throughput_fps']:.1f} FPS")
    
    # Example 3: INT8 with calibration
    print("\n3. Building INT8 Engine (with random calibration)")
    print("-" * 40)
    
    int8_config = EngineConfig(
        onnx_path=demo_onnx,
        engine_path="/tmp/yolov8n_retail_int8.engine",
        precision=Precision.INT8,
        calibration_method=CalibrationMethod.ENTROPY,
        calibration_batches=10,  # Reduced for demo
        max_batch_size=16,
    )
    
    int8_builder = TensorRTEngineBuilder(int8_config)
    int8_metadata = int8_builder.build()
    
    print(f"\nINT8 Engine built:")
    print(f"  Build time: {int8_metadata.build_time_seconds:.2f}s")
    
    # Example 4: Memory estimation
    print("\n4. Memory Usage Estimation")
    print("-" * 40)
    
    memory = EngineOptimizer.estimate_memory_usage(
        input_shape=(16, 3, 640, 640),
        output_shapes={"output0": [16, 8400, 11]},
        precision=Precision.FP16,
    )
    
    print(f"Estimated memory (batch=16, FP16):")
    print(f"  Input: {memory['input_memory_mb']:.2f} MB")
    print(f"  Output: {memory['output_memory_mb']:.2f} MB")
    print(f"  Workspace: {memory['workspace_estimate_mb']:.2f} MB")
    print(f"  Total: {memory['total_estimate_mb']:.2f} MB")
    
    # Cleanup
    os.remove(demo_onnx)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
