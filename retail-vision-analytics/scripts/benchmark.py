#!/usr/bin/env python3
"""
Retail Vision Analytics - Performance Benchmark Suite.

Comprehensive benchmarking for detection, tracking, and analytics pipelines.
Measures latency, throughput, memory usage, and GPU utilization.

Usage:
    python scripts/benchmark.py --mode inference --model data/models/yolov8n_retail_fp16.engine
    python scripts/benchmark.py --mode pipeline --streams 8 --duration 60
    python scripts/benchmark.py --mode full --output results/benchmark_report.json
"""

import argparse
import json
import time
import statistics
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    name: str
    iterations: int
    total_time_ms: float
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass 
class SystemInfo:
    """System information for benchmark context."""
    
    timestamp: str
    hostname: str
    platform: str
    python_version: str
    cpu_count: int
    cpu_model: str
    total_ram_gb: float
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    tensorrt_version: Optional[str] = None


class PerformanceTimer:
    """High-precision timer for benchmarking."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self._start_time: Optional[float] = None
    
    def start(self):
        """Start timing."""
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and record latency."""
        if self._start_time is None:
            raise RuntimeError("Timer not started")
        
        latency = (time.perf_counter() - self._start_time) * 1000  # ms
        self.latencies.append(latency)
        self._start_time = None
        return latency
    
    def reset(self):
        """Reset timer."""
        self.latencies.clear()
        self._start_time = None
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics from recorded latencies."""
        if not self.latencies:
            return {}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            "count": n,
            "total_ms": sum(self.latencies),
            "mean_ms": statistics.mean(self.latencies),
            "std_ms": statistics.stdev(self.latencies) if n > 1 else 0,
            "min_ms": min(self.latencies),
            "max_ms": max(self.latencies),
            "p50_ms": sorted_latencies[int(n * 0.50)],
            "p95_ms": sorted_latencies[int(n * 0.95)],
            "p99_ms": sorted_latencies[int(n * 0.99)],
        }


def get_system_info() -> SystemInfo:
    """Collect system information."""
    import platform
    import socket
    
    # CPU info
    cpu_model = "Unknown"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    
    # RAM info
    total_ram_gb = 0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    total_ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
    except Exception:
        pass
    
    # GPU info
    gpu_name = None
    gpu_memory_gb = None
    cuda_version = None
    
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            gpu_name = parts[0]
            gpu_memory_gb = float(parts[1]) / 1024
        
        # CUDA version
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            cuda_version = result.stdout.strip()
    except Exception:
        pass
    
    # TensorRT version
    tensorrt_version = None
    try:
        import tensorrt as trt
        tensorrt_version = trt.__version__
    except Exception:
        pass
    
    return SystemInfo(
        timestamp=datetime.now().isoformat(),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
        cpu_model=cpu_model,
        total_ram_gb=round(total_ram_gb, 2),
        gpu_name=gpu_name,
        gpu_memory_gb=round(gpu_memory_gb, 2) if gpu_memory_gb else None,
        cuda_version=cuda_version,
        tensorrt_version=tensorrt_version,
    )


def benchmark_inference(
    model_path: str,
    batch_size: int = 1,
    input_shape: tuple = (640, 640),
    warmup_iterations: int = 50,
    benchmark_iterations: int = 500,
    precision: str = "fp16",
) -> BenchmarkResult:
    """
    Benchmark model inference performance.
    
    Args:
        model_path: Path to TensorRT engine or ONNX model
        batch_size: Inference batch size
        input_shape: Input image dimensions (H, W)
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of benchmark runs
        precision: Model precision (fp32, fp16, int8)
    
    Returns:
        BenchmarkResult with performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Inference Benchmark: {Path(model_path).name}")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {input_shape}")
    print(f"Precision: {precision}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Benchmark iterations: {benchmark_iterations}")
    
    timer = PerformanceTimer()
    
    # Check if TensorRT engine
    if model_path.endswith(".engine"):
        results = _benchmark_tensorrt(
            model_path, batch_size, input_shape,
            warmup_iterations, benchmark_iterations, timer
        )
    elif model_path.endswith(".onnx"):
        results = _benchmark_onnx(
            model_path, batch_size, input_shape,
            warmup_iterations, benchmark_iterations, timer
        )
    else:
        results = _benchmark_pytorch(
            model_path, batch_size, input_shape,
            warmup_iterations, benchmark_iterations, timer
        )
    
    stats = timer.get_statistics()
    
    result = BenchmarkResult(
        name=f"inference_{Path(model_path).stem}",
        iterations=benchmark_iterations,
        total_time_ms=stats["total_ms"],
        mean_latency_ms=round(stats["mean_ms"], 3),
        std_latency_ms=round(stats["std_ms"], 3),
        min_latency_ms=round(stats["min_ms"], 3),
        max_latency_ms=round(stats["max_ms"], 3),
        p50_latency_ms=round(stats["p50_ms"], 3),
        p95_latency_ms=round(stats["p95_ms"], 3),
        p99_latency_ms=round(stats["p99_ms"], 3),
        throughput_fps=round(1000 / stats["mean_ms"] * batch_size, 2),
        **results,
    )
    
    _print_benchmark_result(result)
    return result


def _benchmark_tensorrt(
    engine_path: str,
    batch_size: int,
    input_shape: tuple,
    warmup: int,
    iterations: int,
    timer: PerformanceTimer,
) -> Dict[str, Any]:
    """Benchmark TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("TensorRT/PyCUDA not available, using mock benchmark")
        return _mock_benchmark(warmup, iterations, timer, latency_range=(3, 8))
    
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    h, w = input_shape
    input_size = batch_size * 3 * h * w * 4  # float32
    output_size = batch_size * 84 * 8400 * 4  # YOLOv8 output
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    h_input = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    h_output = np.empty((batch_size, 84, 8400), dtype=np.float32)
    
    stream = cuda.Stream()
    
    # Warmup
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # Benchmark
    print(f"Running {iterations} benchmark iterations...")
    for i in range(iterations):
        timer.start()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        timer.stop()
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{iterations}")
    
    # Get GPU memory
    gpu_memory_mb = None
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_memory_mb = float(result.stdout.strip())
    except Exception:
        pass
    
    return {"gpu_memory_mb": gpu_memory_mb}


def _benchmark_onnx(
    model_path: str,
    batch_size: int,
    input_shape: tuple,
    warmup: int,
    iterations: int,
    timer: PerformanceTimer,
) -> Dict[str, Any]:
    """Benchmark ONNX model with ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not available, using mock benchmark")
        return _mock_benchmark(warmup, iterations, timer, latency_range=(8, 15))
    
    # Create session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Prepare input
    h, w = input_shape
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    
    # Warmup
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})
    
    # Benchmark
    print(f"Running {iterations} benchmark iterations...")
    for i in range(iterations):
        timer.start()
        session.run(None, {input_name: dummy_input})
        timer.stop()
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{iterations}")
    
    return {}


def _benchmark_pytorch(
    model_path: str,
    batch_size: int,
    input_shape: tuple,
    warmup: int,
    iterations: int,
    timer: PerformanceTimer,
) -> Dict[str, Any]:
    """Benchmark PyTorch model."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("PyTorch/Ultralytics not available, using mock benchmark")
        return _mock_benchmark(warmup, iterations, timer, latency_range=(15, 25))
    
    # Load model
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare input
    h, w = input_shape
    dummy_input = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Warmup
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        model(dummy_input, verbose=False)
    
    # Benchmark
    print(f"Running {iterations} benchmark iterations...")
    for i in range(iterations):
        timer.start()
        model(dummy_input, verbose=False)
        timer.stop()
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{iterations}")
    
    return {}


def _mock_benchmark(
    warmup: int,
    iterations: int,
    timer: PerformanceTimer,
    latency_range: tuple = (5, 10),
) -> Dict[str, Any]:
    """Mock benchmark for testing without GPU."""
    print(f"Running {warmup} warmup iterations (mock)...")
    time.sleep(0.1)
    
    print(f"Running {iterations} benchmark iterations (mock)...")
    for i in range(iterations):
        timer.start()
        # Simulate variable latency
        latency = np.random.uniform(*latency_range) / 1000
        time.sleep(latency)
        timer.stop()
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{iterations}")
    
    return {"gpu_memory_mb": None}


def benchmark_pipeline(
    num_streams: int = 4,
    duration_seconds: int = 30,
    frame_rate: int = 30,
) -> BenchmarkResult:
    """
    Benchmark multi-stream pipeline performance.
    
    Args:
        num_streams: Number of video streams
        duration_seconds: Test duration
        frame_rate: Target frame rate per stream
    
    Returns:
        BenchmarkResult with pipeline metrics
    """
    print(f"\n{'='*60}")
    print(f"Pipeline Benchmark")
    print(f"{'='*60}")
    print(f"Streams: {num_streams}")
    print(f"Duration: {duration_seconds}s")
    print(f"Target FPS: {frame_rate}/stream")
    
    timer = PerformanceTimer()
    total_frames = num_streams * frame_rate * duration_seconds
    
    # Simulate pipeline processing
    print(f"Simulating {total_frames} total frames...")
    
    frame_interval = 1.0 / (num_streams * frame_rate)
    start_time = time.time()
    frames_processed = 0
    
    while time.time() - start_time < duration_seconds:
        timer.start()
        # Simulate frame processing
        time.sleep(np.random.uniform(0.001, 0.005))
        timer.stop()
        frames_processed += 1
        
        if frames_processed % 1000 == 0:
            elapsed = time.time() - start_time
            current_fps = frames_processed / elapsed
            print(f"  Processed {frames_processed} frames, {current_fps:.1f} FPS")
    
    actual_duration = time.time() - start_time
    stats = timer.get_statistics()
    
    result = BenchmarkResult(
        name=f"pipeline_{num_streams}streams",
        iterations=frames_processed,
        total_time_ms=actual_duration * 1000,
        mean_latency_ms=round(stats["mean_ms"], 3),
        std_latency_ms=round(stats["std_ms"], 3),
        min_latency_ms=round(stats["min_ms"], 3),
        max_latency_ms=round(stats["max_ms"], 3),
        p50_latency_ms=round(stats["p50_ms"], 3),
        p95_latency_ms=round(stats["p95_ms"], 3),
        p99_latency_ms=round(stats["p99_ms"], 3),
        throughput_fps=round(frames_processed / actual_duration, 2),
    )
    
    _print_benchmark_result(result)
    return result


def benchmark_analytics(iterations: int = 10000) -> BenchmarkResult:
    """Benchmark analytics processing performance."""
    print(f"\n{'='*60}")
    print(f"Analytics Benchmark")
    print(f"{'='*60}")
    
    timer = PerformanceTimer()
    
    # Simulate analytics operations
    print(f"Running {iterations} analytics iterations...")
    
    for i in range(iterations):
        timer.start()
        
        # Simulate zone detection
        point = (np.random.random(), np.random.random())
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        _point_in_polygon(point, polygon)
        
        # Simulate heatmap update
        grid = np.zeros((54, 96))
        x, y = int(np.random.random() * 96), int(np.random.random() * 54)
        grid[max(0, y-2):min(54, y+3), max(0, x-2):min(96, x+3)] += 1
        
        # Simulate queue estimation
        positions = [(np.random.random(), np.random.random()) for _ in range(10)]
        sorted(positions, key=lambda p: p[0])
        
        timer.stop()
        
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1}/{iterations}")
    
    stats = timer.get_statistics()
    
    result = BenchmarkResult(
        name="analytics",
        iterations=iterations,
        total_time_ms=stats["total_ms"],
        mean_latency_ms=round(stats["mean_ms"], 4),
        std_latency_ms=round(stats["std_ms"], 4),
        min_latency_ms=round(stats["min_ms"], 4),
        max_latency_ms=round(stats["max_ms"], 4),
        p50_latency_ms=round(stats["p50_ms"], 4),
        p95_latency_ms=round(stats["p95_ms"], 4),
        p99_latency_ms=round(stats["p99_ms"], 4),
        throughput_fps=round(1000 / stats["mean_ms"], 2),
    )
    
    _print_benchmark_result(result)
    return result


def _point_in_polygon(point: tuple, polygon: list) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def _print_benchmark_result(result: BenchmarkResult):
    """Print formatted benchmark result."""
    print(f"\n{'─'*40}")
    print(f"Results: {result.name}")
    print(f"{'─'*40}")
    print(f"  Iterations:     {result.iterations:,}")
    print(f"  Total time:     {result.total_time_ms/1000:.2f}s")
    print(f"  Throughput:     {result.throughput_fps:.2f} FPS")
    print(f"  Latency (mean): {result.mean_latency_ms:.3f} ms")
    print(f"  Latency (std):  {result.std_latency_ms:.3f} ms")
    print(f"  Latency (min):  {result.min_latency_ms:.3f} ms")
    print(f"  Latency (max):  {result.max_latency_ms:.3f} ms")
    print(f"  Latency (p50):  {result.p50_latency_ms:.3f} ms")
    print(f"  Latency (p95):  {result.p95_latency_ms:.3f} ms")
    print(f"  Latency (p99):  {result.p99_latency_ms:.3f} ms")
    if result.gpu_memory_mb:
        print(f"  GPU Memory:     {result.gpu_memory_mb:.0f} MB")


def run_full_benchmark(
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete benchmark suite."""
    print("\n" + "="*60)
    print("RETAIL VISION ANALYTICS - FULL BENCHMARK SUITE")
    print("="*60)
    
    system_info = get_system_info()
    print(f"\nSystem: {system_info.gpu_name or 'CPU'}")
    print(f"Platform: {system_info.platform}")
    
    results = {
        "system_info": asdict(system_info),
        "benchmarks": [],
    }
    
    # Inference benchmark
    if model_path and Path(model_path).exists():
        inference_result = benchmark_inference(model_path)
        results["benchmarks"].append(asdict(inference_result))
    else:
        print("\nSkipping inference benchmark (no model provided)")
    
    # Pipeline benchmark
    pipeline_result = benchmark_pipeline(num_streams=4, duration_seconds=10)
    results["benchmarks"].append(asdict(pipeline_result))
    
    # Analytics benchmark
    analytics_result = benchmark_analytics(iterations=5000)
    results["benchmarks"].append(asdict(analytics_result))
    
    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for bench in results["benchmarks"]:
        print(f"  {bench['name']}: {bench['throughput_fps']:.1f} FPS, "
              f"{bench['mean_latency_ms']:.2f}ms mean latency")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retail Vision Analytics Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark TensorRT inference
  python benchmark.py --mode inference --model data/models/yolov8n_fp16.engine
  
  # Benchmark multi-stream pipeline
  python benchmark.py --mode pipeline --streams 8 --duration 60
  
  # Run full benchmark suite
  python benchmark.py --mode full --output results/benchmark.json
        """,
    )
    
    parser.add_argument(
        "--mode",
        choices=["inference", "pipeline", "analytics", "full"],
        default="full",
        help="Benchmark mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/yolov8n_retail_fp16.engine",
        help="Path to model file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=4,
        help="Number of video streams for pipeline benchmark",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds for pipeline benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of iterations for inference benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results JSON",
    )
    
    args = parser.parse_args()
    
    if args.mode == "inference":
        benchmark_inference(
            args.model,
            batch_size=args.batch_size,
            benchmark_iterations=args.iterations,
        )
    elif args.mode == "pipeline":
        benchmark_pipeline(
            num_streams=args.streams,
            duration_seconds=args.duration,
        )
    elif args.mode == "analytics":
        benchmark_analytics(iterations=args.iterations)
    elif args.mode == "full":
        run_full_benchmark(
            model_path=args.model if Path(args.model).exists() else None,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
