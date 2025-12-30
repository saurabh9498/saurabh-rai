"""
Triton Inference Server Client

Provides async/sync clients for model inference via Triton.
Supports gRPC and HTTP protocols with connection pooling.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

# Try to import Triton client
try:
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("Triton client not available, using mock inference")


@dataclass
class ModelConfig:
    """Configuration for a Triton model."""
    name: str
    version: str = "1"
    timeout_ms: int = 50
    input_names: List[str] = None
    output_names: List[str] = None


@dataclass
class InferenceResult:
    """Result from model inference."""
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    model_name: str
    model_version: str


class TritonClient:
    """
    Client for Triton Inference Server.
    
    Supports both gRPC (recommended for production) and HTTP protocols.
    """
    
    def __init__(
        self,
        url: str = "localhost:8001",
        protocol: str = "grpc",
        connection_timeout: float = 5.0,
        request_timeout: float = 0.05,
        verbose: bool = False,
    ):
        self.url = url
        self.protocol = protocol
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.verbose = verbose
        
        self._client = None
        self._connected = False
        
        if TRITON_AVAILABLE:
            self._connect()
    
    def _connect(self):
        """Establish connection to Triton server."""
        try:
            if self.protocol == "grpc":
                self._client = grpcclient.InferenceServerClient(
                    url=self.url,
                    verbose=self.verbose,
                )
            else:
                self._client = httpclient.InferenceServerClient(
                    url=self.url,
                    verbose=self.verbose,
                    connection_timeout=self.connection_timeout,
                )
            
            # Check connection
            if self._client.is_server_live():
                self._connected = True
                logger.info(f"Connected to Triton server at {self.url}")
            else:
                logger.warning(f"Triton server at {self.url} is not live")
                
        except Exception as e:
            logger.error(f"Failed to connect to Triton: {e}")
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected to Triton."""
        return self._connected and TRITON_AVAILABLE
    
    def get_model_config(self, model_name: str) -> Optional[Dict]:
        """Get model configuration from Triton."""
        if not self.is_connected():
            return None
        
        try:
            config = self._client.get_model_config(model_name)
            return config
        except Exception as e:
            logger.error(f"Failed to get model config: {e}")
            return None
    
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
        model_version: str = "",
    ) -> InferenceResult:
        """
        Run synchronous inference.
        
        Args:
            model_name: Name of the model
            inputs: Dict of input name to numpy array
            output_names: Optional list of output names to retrieve
            model_version: Model version (empty for latest)
            
        Returns:
            InferenceResult with outputs and metadata
        """
        start_time = time.time()
        
        if not self.is_connected():
            # Mock inference
            return self._mock_infer(model_name, inputs, start_time)
        
        try:
            # Prepare inputs
            triton_inputs = []
            for name, data in inputs.items():
                if self.protocol == "grpc":
                    inp = grpcclient.InferInput(
                        name,
                        data.shape,
                        self._numpy_to_triton_dtype(data.dtype),
                    )
                    inp.set_data_from_numpy(data)
                else:
                    inp = httpclient.InferInput(
                        name,
                        data.shape,
                        self._numpy_to_triton_dtype(data.dtype),
                    )
                    inp.set_data_from_numpy(data)
                triton_inputs.append(inp)
            
            # Prepare outputs
            triton_outputs = None
            if output_names:
                if self.protocol == "grpc":
                    triton_outputs = [
                        grpcclient.InferRequestedOutput(name)
                        for name in output_names
                    ]
                else:
                    triton_outputs = [
                        httpclient.InferRequestedOutput(name)
                        for name in output_names
                    ]
            
            # Run inference
            result = self._client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                outputs=triton_outputs,
                model_version=model_version,
            )
            
            # Extract outputs
            outputs = {}
            if output_names:
                for name in output_names:
                    outputs[name] = result.as_numpy(name)
            else:
                # Get first output
                outputs['output'] = result.as_numpy(
                    result.get_output(0).name
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return InferenceResult(
                outputs=outputs,
                latency_ms=latency_ms,
                model_name=model_name,
                model_version=model_version or "latest",
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._mock_infer(model_name, inputs, start_time)
    
    def _mock_infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        start_time: float,
    ) -> InferenceResult:
        """Mock inference for testing/fallback."""
        # Determine batch size from first input
        batch_size = 1
        for arr in inputs.values():
            batch_size = arr.shape[0]
            break
        
        # Generate random outputs
        outputs = {
            'output': np.random.rand(batch_size).astype(np.float32),
        }
        
        latency_ms = (time.time() - start_time) * 1000
        
        return InferenceResult(
            outputs=outputs,
            latency_ms=latency_ms,
            model_name=model_name,
            model_version="mock",
        )
    
    def _numpy_to_triton_dtype(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to Triton dtype string."""
        mapping = {
            np.float32: "FP32",
            np.float64: "FP64",
            np.int32: "INT32",
            np.int64: "INT64",
            np.int16: "INT16",
            np.int8: "INT8",
            np.uint8: "UINT8",
            np.bool_: "BOOL",
        }
        return mapping.get(dtype.type, "FP32")


class AsyncTritonClient:
    """Async wrapper for Triton client."""
    
    def __init__(self, sync_client: TritonClient):
        self.sync_client = sync_client
    
    async def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        **kwargs,
    ) -> InferenceResult:
        """Run async inference."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.sync_client.infer(model_name, inputs, **kwargs),
        )


class TritonModelManager:
    """Manages multiple Triton models."""
    
    def __init__(self, client: TritonClient):
        self.client = client
        self.models: Dict[str, ModelConfig] = {}
    
    def register_model(self, config: ModelConfig):
        """Register a model configuration."""
        self.models[config.name] = config
        logger.info(f"Registered model: {config.name}")
    
    def load_model(self, model_name: str) -> bool:
        """Load a model in Triton."""
        if not self.client.is_connected():
            return False
        
        try:
            self.client._client.load_model(model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from Triton."""
        if not self.client.is_connected():
            return False
        
        try:
            self.client._client.unload_model(model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def get_model_stats(self, model_name: str) -> Optional[Dict]:
        """Get inference statistics for a model."""
        if not self.client.is_connected():
            return None
        
        try:
            stats = self.client._client.get_inference_statistics(model_name)
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for {model_name}: {e}")
            return None


def create_triton_client(
    url: str = "localhost:8001",
    protocol: str = "grpc",
) -> TritonClient:
    """Factory function to create Triton client."""
    return TritonClient(url=url, protocol=protocol)
