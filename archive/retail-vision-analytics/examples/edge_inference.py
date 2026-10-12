#!/usr/bin/env python3
"""
Edge Inference Example.

Demonstrates TensorRT inference optimized for NVIDIA Jetson devices.
Includes memory profiling, latency measurement, and batch processing.

Usage:
    python examples/edge_inference.py --model data/models/yolov8n_fp16.engine
    python examples/edge_inference.py --model data/models/yolov8n_int8.engine --profile
    python examples/edge_inference.py --model model.engine --benchmark --iterations 1000
"""

import argparse
import sys
import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class InferenceResult:
    """Inference result container."""
    detections: List[Dict[str, Any]]
    preprocess_time_ms: float
    inference_time_ms: float
    postprocess_time_ms: float
    total_time_ms: float
    memory_mb: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    iterations: int
    total_time_seconds: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    gpu_memory_mb: Optional[float] = None


class TensorRTInference:
    """TensorRT inference engine wrapper."""
    
    def __init__(
        self,
        engine_path: str,
        input_shape: Tuple[int, int] = (640, 640),
        precision: str = "fp16",
    ):
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.precision = precision
        
        self.engine = None
        self.context = None
        self.stream = None
        self.d_input = None
        self.d_output = None
        self.h_output = None
        
        self._init_engine()
    
    def _init_engine(self):
        """Initialize TensorRT engine."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self.trt = trt
            self.cuda = cuda
        except ImportError:
            print("TensorRT/PyCUDA not available")
            print("Install with: pip install tensorrt pycuda")
            print("For Jetson: These should be pre-installed with JetPack")
            self.trt = None
            self.cuda = None
            return
        
        # Check engine file
        if not os.path.exists(self.engine_path):
            print(f"Engine file not found: {self.engine_path}")
            return
        
        print(f"Loading TensorRT engine: {self.engine_path}")
        
        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        h, w = self.input_shape
        batch_size = 1
        
        input_size = batch_size * 3 * h * w * 4  # float32
        output_size = batch_size * 84 * 8400 * 4  # YOLOv8 output
        
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.h_output = np.empty((batch_size, 84, 8400), dtype=np.float32)
        
        self.stream = cuda.Stream()
        
        print(f"Engine loaded successfully")
        print(f"  Input shape: (1, 3, {h}, {w})")
        print(f"  Output shape: {self.h_output.shape}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        import cv2
        
        h, w = self.input_shape
        
        # Resize
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize and transpose
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # Add batch
        blob = np.ascontiguousarray(blob)
        
        return blob
    
    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Run inference."""
        if self.cuda is None:
            return self._mock_inference(blob)
        
        # Copy to device
        self.cuda.memcpy_htod_async(self.d_input, blob, self.stream)
        
        # Execute
        self.context.execute_async_v2(
            [int(self.d_input), int(self.d_output)],
            self.stream.handle
        )
        
        # Copy back
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return self.h_output.copy()
    
    def postprocess(
        self,
        output: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """Post-process YOLOv8 output."""
        class_names = ["person", "shopping_cart", "basket", "product", 
                       "shelf", "price_tag", "employee"]
        
        # Transpose to (8400, 84)
        predictions = output[0].T
        
        # Extract boxes and scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get best class
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences >= conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        # Convert to x, y, w, h
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x = x_center - w / 2
        y = y_center - h / 2
        
        detections = []
        for i in range(len(boxes)):
            if class_ids[i] < len(class_names):
                detections.append({
                    "class_id": int(class_ids[i]),
                    "class_name": class_names[class_ids[i]],
                    "confidence": float(confidences[i]),
                    "bbox": [float(x[i]), float(y[i]), float(w[i]), float(h[i])],
                })
        
        return detections
    
    def _mock_inference(self, blob: np.ndarray) -> np.ndarray:
        """Mock inference when TensorRT not available."""
        time.sleep(0.005)  # Simulate 5ms inference
        return np.random.randn(1, 84, 8400).astype(np.float32)
    
    def run(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> InferenceResult:
        """Run complete inference pipeline."""
        # Preprocess
        t0 = time.perf_counter()
        blob = self.preprocess(image)
        preprocess_time = (time.perf_counter() - t0) * 1000
        
        # Inference
        t1 = time.perf_counter()
        output = self.infer(blob)
        inference_time = (time.perf_counter() - t1) * 1000
        
        # Postprocess
        t2 = time.perf_counter()
        detections = self.postprocess(output, conf_threshold)
        postprocess_time = (time.perf_counter() - t2) * 1000
        
        total_time = (time.perf_counter() - t0) * 1000
        
        return InferenceResult(
            detections=detections,
            preprocess_time_ms=preprocess_time,
            inference_time_ms=inference_time,
            postprocess_time_ms=postprocess_time,
            total_time_ms=total_time,
        )
    
    def benchmark(
        self,
        iterations: int = 500,
        warmup: int = 50,
    ) -> BenchmarkResult:
        """Run inference benchmark."""
        print(f"\nRunning benchmark ({iterations} iterations, {warmup} warmup)...")
        
        # Create dummy input
        h, w = self.input_shape
        dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        blob = self.preprocess(dummy_image)
        
        # Warmup
        print("Warming up...")
        for _ in range(warmup):
            self.infer(blob)
        
        # Benchmark
        print("Benchmarking...")
        latencies = []
        
        start_time = time.perf_counter()
        for i in range(iterations):
            t0 = time.perf_counter()
            self.infer(blob)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{iterations}")
        
        total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        latencies = sorted(latencies)
        n = len(latencies)
        
        # Get GPU memory
        gpu_memory = self._get_gpu_memory()
        
        return BenchmarkResult(
            iterations=iterations,
            total_time_seconds=total_time,
            avg_latency_ms=np.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=latencies[int(n * 0.50)],
            p95_latency_ms=latencies[int(n * 0.95)],
            p99_latency_ms=latencies[int(n * 0.99)],
            throughput_fps=iterations / total_time,
            gpu_memory_mb=gpu_memory,
        )
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return None


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform
    
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    
    # Check for Jetson
    try:
        with open("/etc/nv_tegra_release") as f:
            info["jetson_l4t"] = f.read().strip()
    except FileNotFoundError:
        pass
    
    # GPU info
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            info["gpu_name"] = parts[0]
            info["gpu_memory"] = parts[1]
            info["driver_version"] = parts[2]
    except Exception:
        pass
    
    # TensorRT version
    try:
        import tensorrt as trt
        info["tensorrt_version"] = trt.__version__
    except ImportError:
        pass
    
    return info


def profile_memory(engine: TensorRTInference, iterations: int = 100):
    """Profile memory usage during inference."""
    print("\n" + "=" * 50)
    print("Memory Profiling")
    print("=" * 50)
    
    h, w = engine.input_shape
    dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    memory_samples = []
    
    for i in range(iterations):
        engine.run(dummy_image)
        
        gpu_mem = engine._get_gpu_memory()
        if gpu_mem:
            memory_samples.append(gpu_mem)
        
        if (i + 1) % 20 == 0:
            if memory_samples:
                print(f"  Iteration {i+1}: GPU Memory = {memory_samples[-1]:.0f} MB")
    
    if memory_samples:
        print(f"\nMemory Statistics:")
        print(f"  Min:  {min(memory_samples):.0f} MB")
        print(f"  Max:  {max(memory_samples):.0f} MB")
        print(f"  Avg:  {np.mean(memory_samples):.0f} MB")
        print(f"  Std:  {np.std(memory_samples):.1f} MB")
    else:
        print("Could not measure GPU memory")


def main():
    parser = argparse.ArgumentParser(
        description="Edge Inference Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python edge_inference.py --model model.engine --image store.jpg
  
  # Run benchmark
  python edge_inference.py --model model.engine --benchmark --iterations 1000
  
  # Profile memory
  python edge_inference.py --model model.engine --profile
        """,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="data/models/yolov8n_retail_fp16.engine",
        help="TensorRT engine path",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Input image for single inference",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Model precision",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Benchmark iterations",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile memory usage",
    )
    parser.add_argument(
        "--system-info",
        action="store_true",
        help="Show system information",
    )
    
    args = parser.parse_args()
    
    # System info
    if args.system_info or args.benchmark:
        print("=" * 50)
        print("System Information")
        print("=" * 50)
        info = get_system_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\nModel not found: {args.model}")
        print("Running in mock mode for demonstration")
    
    # Initialize engine
    print(f"\nInitializing inference engine...")
    engine = TensorRTInference(
        args.model,
        precision=args.precision,
    )
    
    # Run based on mode
    if args.benchmark:
        result = engine.benchmark(
            iterations=args.iterations,
            warmup=50,
        )
        
        print("\n" + "=" * 50)
        print("Benchmark Results")
        print("=" * 50)
        print(f"  Iterations:    {result.iterations}")
        print(f"  Total time:    {result.total_time_seconds:.2f}s")
        print(f"  Throughput:    {result.throughput_fps:.1f} FPS")
        print(f"  Latency (avg): {result.avg_latency_ms:.2f} ms")
        print(f"  Latency (min): {result.min_latency_ms:.2f} ms")
        print(f"  Latency (max): {result.max_latency_ms:.2f} ms")
        print(f"  Latency (p50): {result.p50_latency_ms:.2f} ms")
        print(f"  Latency (p95): {result.p95_latency_ms:.2f} ms")
        print(f"  Latency (p99): {result.p99_latency_ms:.2f} ms")
        if result.gpu_memory_mb:
            print(f"  GPU Memory:    {result.gpu_memory_mb:.0f} MB")
    
    elif args.profile:
        profile_memory(engine, iterations=100)
    
    elif args.image:
        import cv2
        
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image: {args.image}")
            sys.exit(1)
        
        print(f"\nRunning inference on: {args.image}")
        result = engine.run(image)
        
        print(f"\nResults:")
        print(f"  Preprocess:  {result.preprocess_time_ms:.2f} ms")
        print(f"  Inference:   {result.inference_time_ms:.2f} ms")
        print(f"  Postprocess: {result.postprocess_time_ms:.2f} ms")
        print(f"  Total:       {result.total_time_ms:.2f} ms")
        print(f"  Detections:  {len(result.detections)}")
        
        for i, det in enumerate(result.detections):
            print(f"    [{i+1}] {det['class_name']}: {det['confidence']:.2f}")
    
    else:
        # Default: quick benchmark
        print("\nRunning quick benchmark (100 iterations)...")
        result = engine.benchmark(iterations=100, warmup=20)
        print(f"\nThroughput: {result.throughput_fps:.1f} FPS")
        print(f"Avg latency: {result.avg_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
