"""
Triton Inference Server Client

High-performance client for Triton Inference Server:
- gRPC and HTTP support
- Async inference
- Streaming inference
- Health checks and metrics

Enables production-scale model serving with dynamic batching.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio

import numpy as np

try:
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient
    from tritonclient.utils import InferenceServerException, triton_to_np_dtype
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton client not available. Install with: pip install tritonclient[all]")


logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for inference results."""
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    model_name: str
    model_version: str = ""


class TritonClient:
    """
    Client for Triton Inference Server.
    
    Supports both gRPC (recommended for performance) and HTTP protocols.
    
    Example:
        client = TritonClient("localhost:8001")  # gRPC
        
        # Check model status
        if client.is_model_ready("resnet50"):
            # Run inference
            result = client.infer("resnet50", {"input": image_batch})
            predictions = result.outputs["output"]
    """
    
    def __init__(
        self,
        url: str,
        protocol: str = "grpc",
        verbose: bool = False,
        ssl: bool = False,
        ssl_options: Optional[Dict] = None
    ):
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton client not installed")
        
        self.url = url
        self.protocol = protocol.lower()
        self.verbose = verbose
        
        # Create client
        if self.protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(
                url=url,
                verbose=verbose,
                ssl=ssl,
                root_certificates=ssl_options.get("root_certificates") if ssl_options else None,
            )
        else:  # http
            self.client = httpclient.InferenceServerClient(
                url=url,
                verbose=verbose,
                ssl=ssl,
            )
        
        logger.info(f"Connected to Triton server at {url} ({protocol})")
    
    def is_server_live(self) -> bool:
        """Check if server is live."""
        try:
            return self.client.is_server_live()
        except Exception as e:
            logger.error(f"Server live check failed: {e}")
            return False
    
    def is_server_ready(self) -> bool:
        """Check if server is ready."""
        try:
            return self.client.is_server_ready()
        except Exception as e:
            logger.error(f"Server ready check failed: {e}")
            return False
    
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """Check if model is loaded and ready."""
        try:
            return self.client.is_model_ready(model_name, model_version)
        except Exception as e:
            logger.error(f"Model ready check failed: {e}")
            return False
    
    def get_model_metadata(self, model_name: str, model_version: str = "") -> Dict:
        """Get model metadata including inputs/outputs."""
        metadata = self.client.get_model_metadata(model_name, model_version)
        
        if self.protocol == "grpc":
            return {
                "name": metadata.name,
                "versions": list(metadata.versions),
                "platform": metadata.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": list(inp.shape),
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": list(out.shape),
                    }
                    for out in metadata.outputs
                ],
            }
        else:
            return metadata
    
    def get_model_config(self, model_name: str, model_version: str = "") -> Dict:
        """Get model configuration."""
        config = self.client.get_model_config(model_name, model_version)
        return config
    
    def get_server_metadata(self) -> Dict:
        """Get server metadata."""
        metadata = self.client.get_server_metadata()
        
        if self.protocol == "grpc":
            return {
                "name": metadata.name,
                "version": metadata.version,
                "extensions": list(metadata.extensions),
            }
        return metadata
    
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        model_version: str = "",
        outputs: Optional[List[str]] = None,
        timeout: float = 30.0,
        request_id: str = "",
    ) -> InferenceResult:
        """
        Run synchronous inference.
        
        Args:
            model_name: Name of the model
            inputs: Dict mapping input names to numpy arrays
            model_version: Model version (empty for latest)
            outputs: List of output names to retrieve (None for all)
            timeout: Request timeout in seconds
            request_id: Optional request ID for tracking
            
        Returns:
            InferenceResult with outputs and latency
        """
        start_time = time.perf_counter()
        
        # Create input objects
        if self.protocol == "grpc":
            triton_inputs = []
            for name, data in inputs.items():
                inp = grpcclient.InferInput(
                    name,
                    list(data.shape),
                    self._numpy_to_triton_dtype(data.dtype)
                )
                inp.set_data_from_numpy(data)
                triton_inputs.append(inp)
            
            # Create output requests
            triton_outputs = None
            if outputs:
                triton_outputs = [
                    grpcclient.InferRequestedOutput(name)
                    for name in outputs
                ]
            
            # Run inference
            result = self.client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                model_version=model_version,
                outputs=triton_outputs,
                request_id=request_id,
                client_timeout=timeout,
            )
        else:  # http
            triton_inputs = []
            for name, data in inputs.items():
                inp = httpclient.InferInput(
                    name,
                    list(data.shape),
                    self._numpy_to_triton_dtype(data.dtype)
                )
                inp.set_data_from_numpy(data)
                triton_inputs.append(inp)
            
            triton_outputs = None
            if outputs:
                triton_outputs = [
                    httpclient.InferRequestedOutput(name)
                    for name in outputs
                ]
            
            result = self.client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                model_version=model_version,
                outputs=triton_outputs,
                request_id=request_id,
            )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Parse outputs
        output_dict = {}
        if outputs:
            for name in outputs:
                output_dict[name] = result.as_numpy(name)
        else:
            # Get all outputs from model metadata
            metadata = self.get_model_metadata(model_name, model_version)
            for out in metadata["outputs"]:
                output_dict[out["name"]] = result.as_numpy(out["name"])
        
        return InferenceResult(
            outputs=output_dict,
            latency_ms=latency,
            model_name=model_name,
            model_version=model_version,
        )
    
    async def infer_async(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        model_version: str = "",
        outputs: Optional[List[str]] = None,
    ) -> InferenceResult:
        """
        Run async inference.
        
        Uses ThreadPoolExecutor to run sync client in async context.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.infer(model_name, inputs, model_version, outputs)
        )
    
    def infer_batch(
        self,
        model_name: str,
        batches: List[Dict[str, np.ndarray]],
        model_version: str = "",
        max_workers: int = 4,
    ) -> List[InferenceResult]:
        """
        Run parallel inference on multiple batches.
        
        Uses ThreadPoolExecutor for concurrent requests.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.infer, model_name, batch, model_version)
                for batch in batches
            ]
            return [f.result() for f in futures]
    
    def _numpy_to_triton_dtype(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to Triton dtype string."""
        dtype_map = {
            np.float32: "FP32",
            np.float16: "FP16",
            np.float64: "FP64",
            np.int32: "INT32",
            np.int64: "INT64",
            np.int16: "INT16",
            np.int8: "INT8",
            np.uint8: "UINT8",
            np.bool_: "BOOL",
        }
        return dtype_map.get(dtype.type, "FP32")
    
    def close(self):
        """Close client connection."""
        if hasattr(self.client, 'close'):
            self.client.close()


class ModelConfig:
    """
    Helper class to generate Triton model configuration.
    
    Example:
        config = ModelConfig(
            name="resnet50",
            platform="tensorrt_plan",
            max_batch_size=64,
            dynamic_batching={
                "preferred_batch_size": [8, 16, 32],
                "max_queue_delay_microseconds": 100
            }
        )
        config.save("model_repository/resnet50/config.pbtxt")
    """
    
    def __init__(
        self,
        name: str,
        platform: str,
        max_batch_size: int = 0,
        inputs: Optional[List[Dict]] = None,
        outputs: Optional[List[Dict]] = None,
        dynamic_batching: Optional[Dict] = None,
        instance_group: Optional[List[Dict]] = None,
        optimization: Optional[Dict] = None,
    ):
        self.name = name
        self.platform = platform
        self.max_batch_size = max_batch_size
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.dynamic_batching = dynamic_batching
        self.instance_group = instance_group
        self.optimization = optimization
    
    def to_pbtxt(self) -> str:
        """Generate protobuf text format configuration."""
        lines = []
        
        lines.append(f'name: "{self.name}"')
        lines.append(f'platform: "{self.platform}"')
        lines.append(f'max_batch_size: {self.max_batch_size}')
        lines.append('')
        
        # Inputs
        for inp in self.inputs:
            lines.append('input [')
            lines.append('  {')
            lines.append(f'    name: "{inp["name"]}"')
            lines.append(f'    data_type: {inp["data_type"]}')
            dims = ", ".join(str(d) for d in inp["dims"])
            lines.append(f'    dims: [ {dims} ]')
            lines.append('  }')
            lines.append(']')
        
        # Outputs
        for out in self.outputs:
            lines.append('output [')
            lines.append('  {')
            lines.append(f'    name: "{out["name"]}"')
            lines.append(f'    data_type: {out["data_type"]}')
            dims = ", ".join(str(d) for d in out["dims"])
            lines.append(f'    dims: [ {dims} ]')
            lines.append('  }')
            lines.append(']')
        
        # Dynamic batching
        if self.dynamic_batching:
            lines.append('')
            lines.append('dynamic_batching {')
            if "preferred_batch_size" in self.dynamic_batching:
                sizes = ", ".join(str(s) for s in self.dynamic_batching["preferred_batch_size"])
                lines.append(f'  preferred_batch_size: [ {sizes} ]')
            if "max_queue_delay_microseconds" in self.dynamic_batching:
                lines.append(f'  max_queue_delay_microseconds: {self.dynamic_batching["max_queue_delay_microseconds"]}')
            lines.append('}')
        
        # Instance group
        if self.instance_group:
            lines.append('')
            for group in self.instance_group:
                lines.append('instance_group [')
                lines.append('  {')
                if "count" in group:
                    lines.append(f'    count: {group["count"]}')
                if "kind" in group:
                    lines.append(f'    kind: {group["kind"]}')
                if "gpus" in group:
                    gpus = ", ".join(str(g) for g in group["gpus"])
                    lines.append(f'    gpus: [ {gpus} ]')
                lines.append('  }')
                lines.append(']')
        
        # Optimization
        if self.optimization:
            lines.append('')
            lines.append('optimization {')
            if "cuda" in self.optimization:
                lines.append('  cuda {')
                cuda = self.optimization["cuda"]
                if "graphs" in cuda:
                    lines.append(f'    graphs: {str(cuda["graphs"]).lower()}')
                lines.append('  }')
            lines.append('}')
        
        return '\n'.join(lines)
    
    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            f.write(self.to_pbtxt())
        logger.info(f"Saved model config to {path}")


def create_ensemble_config(
    name: str,
    steps: List[Dict],
    max_batch_size: int = 64
) -> str:
    """
    Create ensemble model configuration.
    
    Args:
        name: Ensemble model name
        steps: List of pipeline steps with model_name, input_map, output_map
        max_batch_size: Maximum batch size
        
    Returns:
        Protobuf text configuration string
    """
    lines = [
        f'name: "{name}"',
        'platform: "ensemble"',
        f'max_batch_size: {max_batch_size}',
        '',
        'ensemble_scheduling {',
    ]
    
    for i, step in enumerate(steps):
        lines.append('  step [')
        lines.append('    {')
        lines.append(f'      model_name: "{step["model_name"]}"')
        if "model_version" in step:
            lines.append(f'      model_version: {step["model_version"]}')
        
        for key, value in step.get("input_map", {}).items():
            lines.append(f'      input_map {{ key: "{key}" value: "{value}" }}')
        
        for key, value in step.get("output_map", {}).items():
            lines.append(f'      output_map {{ key: "{key}" value: "{value}" }}')
        
        lines.append('    }')
        lines.append('  ]')
    
    lines.append('}')
    
    return '\n'.join(lines)
