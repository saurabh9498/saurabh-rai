"""
TensorRT Graph Optimizations

Advanced optimizations for TensorRT engines:
- Layer fusion analysis
- Precision calibration strategies
- Memory optimization
- Kernel selection

Maximizes inference performance while maintaining accuracy.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    MIXED = "mixed"


@dataclass
class LayerInfo:
    """Information about a network layer."""
    name: str
    type: str
    precision: str
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    num_parameters: int = 0


@dataclass
class OptimizationResult:
    """Results from optimization analysis."""
    original_layers: int
    optimized_layers: int
    fused_layers: int
    precision_distribution: Dict[str, int]
    estimated_speedup: float
    memory_reduction: float


class NetworkAnalyzer:
    """
    Analyzes TensorRT network for optimization opportunities.
    """
    
    def __init__(self, network: "trt.INetworkDefinition"):
        self.network = network
        self.layers: List[LayerInfo] = []
        self._analyze_network()
    
    def _analyze_network(self):
        """Extract layer information from network."""
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            
            # Get input shapes
            input_shapes = []
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp:
                    input_shapes.append(tuple(inp.shape))
            
            # Get output shapes
            output_shapes = []
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                if out:
                    output_shapes.append(tuple(out.shape))
            
            info = LayerInfo(
                name=layer.name,
                type=str(layer.type),
                precision=str(layer.precision),
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )
            self.layers.append(info)
    
    def get_layer_summary(self) -> Dict:
        """Get summary of layer types."""
        type_counts = {}
        for layer in self.layers:
            layer_type = layer.type
            type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
        return type_counts
    
    def get_precision_summary(self) -> Dict:
        """Get summary of layer precisions."""
        precision_counts = {}
        for layer in self.layers:
            prec = layer.precision
            precision_counts[prec] = precision_counts.get(prec, 0) + 1
        return precision_counts
    
    def find_fusion_candidates(self) -> List[List[str]]:
        """
        Find layers that could be fused.
        
        Common fusion patterns:
        - Conv + BatchNorm + ReLU
        - Conv + Add + ReLU
        - MatMul + Add
        """
        candidates = []
        
        # Look for Conv + BN + ReLU patterns
        i = 0
        while i < len(self.layers) - 2:
            l1 = self.layers[i]
            l2 = self.layers[i + 1]
            l3 = self.layers[i + 2]
            
            if ("CONVOLUTION" in l1.type and 
                "SCALE" in l2.type and 
                "ACTIVATION" in l3.type):
                candidates.append([l1.name, l2.name, l3.name])
                i += 3
            else:
                i += 1
        
        return candidates
    
    def estimate_memory_usage(self) -> Dict:
        """Estimate memory usage for different precisions."""
        total_params = 0
        
        for layer in self.layers:
            # Estimate parameters from shapes
            for shape in layer.output_shapes:
                if len(shape) >= 2:
                    total_params += np.prod(shape)
        
        return {
            "fp32_mb": total_params * 4 / (1024 * 1024),
            "fp16_mb": total_params * 2 / (1024 * 1024),
            "int8_mb": total_params * 1 / (1024 * 1024),
        }


class PrecisionOptimizer:
    """
    Optimizes layer precision for accuracy/speed tradeoff.
    """
    
    # Layers sensitive to quantization (keep higher precision)
    SENSITIVE_LAYERS = [
        "Softmax",
        "LayerNorm",
        "BatchNorm",
        "ReduceMean",
        "Resize",
    ]
    
    # Layers safe for aggressive quantization
    QUANTIZATION_FRIENDLY = [
        "Conv",
        "MatMul",
        "FullyConnected",
    ]
    
    def __init__(self, network: "trt.INetworkDefinition"):
        self.network = network
    
    def get_precision_config(
        self,
        target_precision: PrecisionMode = PrecisionMode.MIXED
    ) -> Dict[str, str]:
        """
        Get recommended precision for each layer.
        
        Args:
            target_precision: Target precision mode
            
        Returns:
            Dict mapping layer names to precision
        """
        config = {}
        
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            layer_name = layer.name
            layer_type = str(layer.type)
            
            if target_precision == PrecisionMode.FP32:
                config[layer_name] = "fp32"
            
            elif target_precision == PrecisionMode.FP16:
                config[layer_name] = "fp16"
            
            elif target_precision == PrecisionMode.INT8:
                # Keep sensitive layers in FP16
                if any(s in layer_type for s in self.SENSITIVE_LAYERS):
                    config[layer_name] = "fp16"
                else:
                    config[layer_name] = "int8"
            
            else:  # MIXED
                if any(s in layer_type for s in self.SENSITIVE_LAYERS):
                    config[layer_name] = "fp32"
                elif any(s in layer_type for s in self.QUANTIZATION_FRIENDLY):
                    config[layer_name] = "int8"
                else:
                    config[layer_name] = "fp16"
        
        return config
    
    def apply_precision_config(
        self,
        config: Dict[str, str],
        builder_config: "trt.IBuilderConfig"
    ):
        """Apply precision configuration to network."""
        precision_map = {
            "fp32": trt.float32,
            "fp16": trt.float16,
            "int8": trt.int8,
        }
        
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            if layer.name in config:
                prec = config[layer.name]
                if prec in precision_map:
                    layer.precision = precision_map[prec]
                    logger.debug(f"Set {layer.name} precision to {prec}")


class SparsityOptimizer:
    """
    Optimizes networks using structured sparsity (Ampere+ GPUs).
    
    Structured sparsity (2:4 pattern) provides:
    - 2x compute throughput
    - 50% weight reduction
    - Minimal accuracy loss
    """
    
    def __init__(self, network: "trt.INetworkDefinition"):
        self.network = network
    
    def find_sparse_candidates(self) -> List[str]:
        """Find layers suitable for sparsity."""
        candidates = []
        
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            layer_type = str(layer.type)
            
            # Conv and FC layers benefit from sparsity
            if "CONVOLUTION" in layer_type or "FULLY_CONNECTED" in layer_type:
                candidates.append(layer.name)
        
        return candidates
    
    def estimate_sparsity_benefit(self) -> Dict:
        """Estimate benefit from structured sparsity."""
        candidates = self.find_sparse_candidates()
        
        return {
            "eligible_layers": len(candidates),
            "estimated_speedup": 1.5 if candidates else 1.0,
            "memory_reduction": 0.5 if candidates else 0.0,
        }


def optimize_for_inference(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    enable_sparsity: bool = False,
    workspace_gb: float = 4.0
) -> OptimizationResult:
    """
    High-level optimization function.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save optimized engine
        precision: Target precision (fp32, fp16, int8, mixed)
        enable_sparsity: Enable structured sparsity
        workspace_gb: GPU workspace size
        
    Returns:
        OptimizationResult with statistics
    """
    from .builder import TensorRTBuilder, BuildConfig
    
    config = BuildConfig(
        precision=precision,
        workspace_size_gb=workspace_gb,
        enable_sparse=enable_sparsity,
        optimization_level=5,
    )
    
    builder = TensorRTBuilder(onnx_path, config)
    
    # Analyze network
    network_info = builder.get_network_info()
    original_layers = network_info["num_layers"]
    
    # Build engine
    engine = builder.build()
    engine.save(output_path)
    
    # Estimate optimizations
    result = OptimizationResult(
        original_layers=original_layers,
        optimized_layers=original_layers,  # TRT doesn't expose this
        fused_layers=0,
        precision_distribution={precision: original_layers},
        estimated_speedup=2.0 if precision == "fp16" else 4.0 if precision == "int8" else 1.0,
        memory_reduction=0.5 if precision == "fp16" else 0.75 if precision == "int8" else 0.0,
    )
    
    logger.info(f"Optimization complete: {result}")
    return result
