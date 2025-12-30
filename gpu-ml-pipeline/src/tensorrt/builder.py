"""
TensorRT Engine Builder

Builds optimized TensorRT engines from ONNX models with:
- FP32/FP16/INT8 precision support
- Dynamic shape optimization
- Layer-level precision control
- Timing cache for faster rebuilds

Performance: 2-15x inference speedup over PyTorch
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available. Install with: pip install tensorrt")


logger = logging.getLogger(__name__)


@dataclass
class OptimizationProfile:
    """Defines min/opt/max shapes for dynamic inputs."""
    name: str
    min_shape: Tuple[int, ...]
    opt_shape: Tuple[int, ...]
    max_shape: Tuple[int, ...]


@dataclass
class BuildConfig:
    """TensorRT engine build configuration."""
    precision: str = "fp16"  # fp32, fp16, int8
    max_batch_size: int = 32
    workspace_size_gb: float = 4.0
    optimization_level: int = 5  # 0-5, higher = slower build, faster inference
    enable_sparse: bool = False
    enable_timing_cache: bool = True
    timing_cache_path: Optional[str] = None
    calibration_cache_path: Optional[str] = None
    
    # Dynamic shapes
    dynamic_shapes: Dict[str, OptimizationProfile] = field(default_factory=dict)
    
    # Layer precision overrides
    layer_precisions: Dict[str, str] = field(default_factory=dict)


class TensorRTLogger(trt.ILogger):
    """Custom TensorRT logger."""
    
    def __init__(self, min_severity=trt.Logger.WARNING):
        super().__init__()
        self.min_severity = min_severity
    
    def log(self, severity, msg):
        if severity <= self.min_severity:
            if severity == trt.Logger.ERROR:
                logger.error(f"[TensorRT] {msg}")
            elif severity == trt.Logger.WARNING:
                logger.warning(f"[TensorRT] {msg}")
            elif severity == trt.Logger.INFO:
                logger.info(f"[TensorRT] {msg}")
            else:
                logger.debug(f"[TensorRT] {msg}")


class TensorRTBuilder:
    """
    Builds optimized TensorRT engines from ONNX models.
    
    Example:
        builder = TensorRTBuilder(
            onnx_path="model.onnx",
            precision="int8",
            calibration_data=calib_loader
        )
        engine = builder.build()
        engine.save("model.engine")
    """
    
    def __init__(
        self,
        onnx_path: str,
        config: Optional[BuildConfig] = None,
        calibration_data=None,
        verbose: bool = False
    ):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not installed")
        
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        self.config = config or BuildConfig()
        self.calibration_data = calibration_data
        
        # Initialize TensorRT components
        severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        self.trt_logger = TensorRTLogger(severity)
        self.builder = trt.Builder(self.trt_logger)
        self.network = None
        self.parser = None
        
        # Timing cache
        self.timing_cache = None
        if self.config.enable_timing_cache and self.config.timing_cache_path:
            self._load_timing_cache()
    
    def _load_timing_cache(self):
        """Load timing cache from disk."""
        cache_path = Path(self.config.timing_cache_path)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache_data = f.read()
                self.timing_cache = self.builder.create_builder_config().create_timing_cache(
                    cache_data, ignore_mismatch=True
                )
            logger.info(f"Loaded timing cache from {cache_path}")
    
    def _save_timing_cache(self, config):
        """Save timing cache to disk."""
        if self.config.timing_cache_path:
            cache = config.get_timing_cache()
            cache_data = cache.serialize()
            with open(self.config.timing_cache_path, "wb") as f:
                f.write(cache_data)
            logger.info(f"Saved timing cache to {self.config.timing_cache_path}")
    
    def _parse_onnx(self) -> bool:
        """Parse ONNX model into TensorRT network."""
        # Create network with explicit batch
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        
        # Parse ONNX file
        with open(self.onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                for i in range(self.parser.num_errors):
                    logger.error(f"ONNX parse error: {self.parser.get_error(i)}")
                return False
        
        logger.info(f"Parsed ONNX model: {self.onnx_path}")
        logger.info(f"  Inputs: {self.network.num_inputs}")
        logger.info(f"  Outputs: {self.network.num_outputs}")
        logger.info(f"  Layers: {self.network.num_layers}")
        
        return True
    
    def _configure_builder(self) -> trt.IBuilderConfig:
        """Configure TensorRT builder settings."""
        config = self.builder.create_builder_config()
        
        # Workspace size
        workspace_bytes = int(self.config.workspace_size_gb * (1 << 30))
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        
        # Precision flags
        if self.config.precision == "fp16":
            if self.builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision")
            else:
                logger.warning("FP16 not supported on this platform")
        
        elif self.config.precision == "int8":
            if self.builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                
                if self.calibration_data is not None:
                    from .calibrator import EntropyCalibrator
                    calibrator = EntropyCalibrator(
                        self.calibration_data,
                        cache_file=self.config.calibration_cache_path
                    )
                    config.int8_calibrator = calibrator
                    logger.info("Enabled INT8 with calibration")
                else:
                    logger.warning("INT8 enabled but no calibration data provided")
            else:
                logger.warning("INT8 not supported on this platform")
        
        # Optimization level
        config.builder_optimization_level = self.config.optimization_level
        
        # Sparse weights (Ampere+)
        if self.config.enable_sparse:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            logger.info("Enabled sparse weights")
        
        # Timing cache
        if self.timing_cache:
            config.set_timing_cache(self.timing_cache, ignore_mismatch=True)
        
        return config
    
    def _add_optimization_profiles(self, config: trt.IBuilderConfig):
        """Add optimization profiles for dynamic shapes."""
        if not self.config.dynamic_shapes:
            return
        
        profile = self.builder.create_optimization_profile()
        
        for input_name, opt_profile in self.config.dynamic_shapes.items():
            profile.set_shape(
                input_name,
                opt_profile.min_shape,
                opt_profile.opt_shape,
                opt_profile.max_shape
            )
            logger.info(f"Added optimization profile for {input_name}:")
            logger.info(f"  min: {opt_profile.min_shape}")
            logger.info(f"  opt: {opt_profile.opt_shape}")
            logger.info(f"  max: {opt_profile.max_shape}")
        
        config.add_optimization_profile(profile)
    
    def _apply_layer_precisions(self):
        """Apply per-layer precision settings."""
        if not self.config.layer_precisions:
            return
        
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            layer_name = layer.name
            
            for pattern, precision in self.config.layer_precisions.items():
                if pattern in layer_name or pattern == "*":
                    if precision == "fp32":
                        layer.precision = trt.float32
                    elif precision == "fp16":
                        layer.precision = trt.float16
                    elif precision == "int8":
                        layer.precision = trt.int8
                    
                    logger.debug(f"Set {layer_name} precision to {precision}")
    
    def build(self) -> "TensorRTEngine":
        """
        Build the TensorRT engine.
        
        Returns:
            TensorRTEngine wrapper for inference
        """
        logger.info("Starting TensorRT engine build...")
        
        # Parse ONNX
        if not self._parse_onnx():
            raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = self._configure_builder()
        
        # Add optimization profiles
        self._add_optimization_profiles(config)
        
        # Apply layer precisions
        self._apply_layer_precisions()
        
        # Build engine
        logger.info("Building engine (this may take several minutes)...")
        serialized_engine = self.builder.build_serialized_network(self.network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save timing cache
        if self.config.enable_timing_cache:
            self._save_timing_cache(config)
        
        logger.info("Engine build complete!")
        
        # Create engine wrapper
        runtime = trt.Runtime(self.trt_logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        return TensorRTEngine(engine, self.trt_logger)
    
    def get_network_info(self) -> Dict:
        """Get information about the parsed network."""
        if self.network is None:
            self._parse_onnx()
        
        info = {
            "num_inputs": self.network.num_inputs,
            "num_outputs": self.network.num_outputs,
            "num_layers": self.network.num_layers,
            "inputs": [],
            "outputs": [],
        }
        
        for i in range(self.network.num_inputs):
            inp = self.network.get_input(i)
            info["inputs"].append({
                "name": inp.name,
                "shape": tuple(inp.shape),
                "dtype": str(inp.dtype),
            })
        
        for i in range(self.network.num_outputs):
            out = self.network.get_output(i)
            info["outputs"].append({
                "name": out.name,
                "shape": tuple(out.shape),
                "dtype": str(out.dtype),
            })
        
        return info


class TensorRTEngine:
    """Wrapper for serialized TensorRT engine."""
    
    def __init__(self, engine, logger):
        self.engine = engine
        self.logger = logger
        self.context = engine.create_execution_context()
    
    def save(self, path: str):
        """Serialize and save engine to disk."""
        serialized = self.engine.serialize()
        with open(path, "wb") as f:
            f.write(serialized)
        logger.info(f"Saved engine to {path}")
    
    @classmethod
    def load(cls, path: str) -> "TensorRTEngine":
        """Load engine from disk."""
        trt_logger = TensorRTLogger()
        runtime = trt.Runtime(trt_logger)
        
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        return cls(engine, trt_logger)
    
    def get_binding_info(self) -> Dict:
        """Get input/output binding information."""
        info = {"inputs": [], "outputs": []}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            
            binding = {
                "name": name,
                "shape": tuple(shape),
                "dtype": str(dtype),
            }
            
            if mode == trt.TensorIOMode.INPUT:
                info["inputs"].append(binding)
            else:
                info["outputs"].append(binding)
        
        return info


def build_engine_from_config(config_path: str) -> TensorRTEngine:
    """Build engine from YAML configuration file."""
    import yaml
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Parse config
    build_config = BuildConfig(
        precision=cfg.get("precision", "fp16"),
        max_batch_size=cfg.get("max_batch_size", 32),
        workspace_size_gb=cfg.get("workspace_size_gb", 4.0),
        optimization_level=cfg.get("optimization_level", 5),
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
    
    builder = TensorRTBuilder(
        onnx_path=cfg["onnx_path"],
        config=build_config,
    )
    
    return builder.build()
