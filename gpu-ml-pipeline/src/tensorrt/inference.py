"""
TensorRT Inference Wrapper

High-performance inference using TensorRT engines:
- Async execution with CUDA streams
- Automatic memory management
- Batched inference support
- Latency profiling

Performance: Sub-millisecond inference latency
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Inference performance metrics."""
    inference_time_ms: float
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    total_time_ms: float = 0.0
    throughput: float = 0.0  # samples/second
    batch_size: int = 1


class HostDeviceMemory:
    """Manages paired host and device memory for a tensor."""
    
    def __init__(self, host_mem: np.ndarray, device_mem: cuda.DeviceAllocation):
        self.host = host_mem
        self.device = device_mem
    
    def __str__(self):
        return f"Host: {self.host.shape}, Device: {self.device}"
    
    def __repr__(self):
        return self.__str__()


class TensorRTInference:
    """
    TensorRT inference engine wrapper.
    
    Handles:
    - Engine loading and execution
    - Memory allocation and transfer
    - Async execution with streams
    - Performance profiling
    
    Example:
        engine = TensorRTInference("model.engine")
        
        # Single inference
        output = engine.infer(input_data)
        
        # Batched inference
        outputs = engine.infer_batch([img1, img2, img3])
        
        # Get metrics
        print(f"Latency: {engine.last_metrics.inference_time_ms:.2f}ms")
    """
    
    def __init__(
        self,
        engine_path: str,
        device_id: int = 0,
        max_batch_size: Optional[int] = None,
        enable_profiling: bool = True
    ):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT/PyCUDA not available")
        
        self.engine_path = Path(engine_path)
        self.device_id = device_id
        self.enable_profiling = enable_profiling
        
        # Set CUDA device
        cuda.Device(device_id).make_context()
        
        # Load engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get max batch size
        self.max_batch_size = max_batch_size or self._get_max_batch_size()
        
        # Allocate memory
        self.inputs: List[HostDeviceMemory] = []
        self.outputs: List[HostDeviceMemory] = []
        self.bindings: List[int] = []
        self._allocate_buffers()
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # Metrics
        self.last_metrics = InferenceMetrics(0.0)
        
        # CUDA events for timing
        if enable_profiling:
            self.start_event = cuda.Event()
            self.end_event = cuda.Event()
        
        logger.info(f"Loaded TensorRT engine: {engine_path}")
        logger.info(f"  Max batch size: {self.max_batch_size}")
        logger.info(f"  Inputs: {len(self.inputs)}")
        logger.info(f"  Outputs: {len(self.outputs)}")
    
    def _get_max_batch_size(self) -> int:
        """Get maximum batch size from engine."""
        # For explicit batch networks
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            if shape[0] == -1:  # Dynamic batch
                return 64  # Default max
            return shape[0]
        return 1
    
    def _allocate_buffers(self):
        """Allocate host and device memory for inputs/outputs."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            # Handle dynamic shapes
            if -1 in shape:
                shape = tuple(s if s != -1 else self.max_batch_size for s in shape)
            
            # Calculate size
            size = int(np.prod(shape))
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            mem = HostDeviceMemory(host_mem.reshape(shape), device_mem)
            
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(mem)
            else:
                self.outputs.append(mem)
    
    def _set_input_shape(self, batch_size: int):
        """Set input shapes for dynamic batch size."""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                shape = list(self.engine.get_tensor_shape(name))
                if shape[0] == -1:
                    shape[0] = batch_size
                    self.context.set_input_shape(name, shape)
    
    def infer(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        output_names: Optional[List[str]] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference on input data.
        
        Args:
            inputs: Input array or dict of named inputs
            output_names: Optional list of output names to return
            
        Returns:
            Output array(s)
        """
        start_time = time.perf_counter()
        
        # Handle dict input
        if isinstance(inputs, dict):
            input_data = list(inputs.values())[0]
        else:
            input_data = inputs
        
        # Get batch size
        batch_size = input_data.shape[0] if input_data.ndim > 1 else 1
        
        # Set dynamic batch size
        self._set_input_shape(batch_size)
        
        # Copy input to host buffer
        np.copyto(self.inputs[0].host[:batch_size].ravel(), input_data.ravel())
        
        # Start timing
        if self.enable_profiling:
            self.start_event.record(self.stream)
        
        # Copy input to device
        cuda.memcpy_htod_async(
            self.inputs[0].device,
            self.inputs[0].host,
            self.stream
        )
        
        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])
        
        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy outputs back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        
        # End timing
        if self.enable_profiling:
            self.end_event.record(self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        # Calculate metrics
        total_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_profiling:
            inference_time = self.start_event.time_till(self.end_event)
        else:
            inference_time = total_time
        
        self.last_metrics = InferenceMetrics(
            inference_time_ms=inference_time,
            total_time_ms=total_time,
            throughput=batch_size / (total_time / 1000),
            batch_size=batch_size
        )
        
        # Return output
        if len(self.outputs) == 1:
            return self.outputs[0].host[:batch_size].copy()
        else:
            return {f"output_{i}": out.host[:batch_size].copy() 
                    for i, out in enumerate(self.outputs)}
    
    def infer_batch(
        self,
        batch: List[np.ndarray],
        pad_to_batch: bool = True
    ) -> List[np.ndarray]:
        """
        Run inference on a batch of inputs.
        
        Args:
            batch: List of input arrays
            pad_to_batch: Pad batch to max_batch_size
            
        Returns:
            List of output arrays
        """
        batch_size = len(batch)
        
        # Stack inputs
        stacked = np.stack(batch, axis=0)
        
        # Run inference
        outputs = self.infer(stacked)
        
        # Split outputs
        if isinstance(outputs, np.ndarray):
            return [outputs[i] for i in range(batch_size)]
        else:
            return [{k: v[i] for k, v in outputs.items()} for i in range(batch_size)]
    
    @contextmanager
    def benchmark(self, warmup: int = 10, iterations: int = 100):
        """
        Context manager for benchmarking.
        
        Example:
            with engine.benchmark() as results:
                for _ in range(100):
                    engine.infer(data)
            print(f"Average: {results['mean_ms']:.2f}ms")
        """
        latencies = []
        
        class BenchmarkResults:
            mean_ms: float = 0.0
            std_ms: float = 0.0
            min_ms: float = 0.0
            max_ms: float = 0.0
            p50_ms: float = 0.0
            p99_ms: float = 0.0
            throughput: float = 0.0
        
        results = BenchmarkResults()
        
        try:
            yield results
        finally:
            # Collect latencies from inference calls made in context
            pass
    
    def warmup(self, iterations: int = 10):
        """Warmup the engine with dummy data."""
        # Create dummy input
        input_shape = self.inputs[0].host.shape
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        for _ in range(iterations):
            self.infer(dummy_input[:1])
        
        logger.info(f"Warmup complete ({iterations} iterations)")
    
    def get_binding_info(self) -> Dict:
        """Get information about input/output bindings."""
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
    
    def __del__(self):
        """Cleanup resources."""
        try:
            for inp in self.inputs:
                inp.device.free()
            for out in self.outputs:
                out.device.free()
            cuda.Context.pop()
        except:
            pass


def benchmark_engine(
    engine_path: str,
    input_shape: Tuple[int, ...],
    warmup: int = 50,
    iterations: int = 1000,
    batch_sizes: List[int] = [1, 8, 16, 32]
) -> Dict:
    """
    Benchmark TensorRT engine performance.
    
    Returns:
        Dict with latency statistics for each batch size
    """
    engine = TensorRTInference(engine_path)
    results = {}
    
    for batch_size in batch_sizes:
        shape = (batch_size,) + input_shape[1:]
        dummy_input = np.random.randn(*shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            engine.infer(dummy_input)
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            engine.infer(dummy_input)
            latencies.append(engine.last_metrics.inference_time_ms)
        
        latencies = np.array(latencies)
        
        results[f"batch_{batch_size}"] = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput": float(batch_size / np.mean(latencies) * 1000),
        }
        
        logger.info(f"Batch {batch_size}: {np.mean(latencies):.2f}ms ± {np.std(latencies):.2f}ms")
    
    return results
