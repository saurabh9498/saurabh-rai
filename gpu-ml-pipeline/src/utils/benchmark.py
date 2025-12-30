"""
Benchmarking Utilities for GPU ML Pipeline

Comprehensive benchmarking for:
- Preprocessing throughput
- Model inference latency
- End-to-end pipeline performance
- Multi-GPU scaling

Outputs detailed metrics and comparison plots.
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import statistics

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    batch_size: int
    iterations: int
    
    # Latency stats (milliseconds)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    
    # Throughput
    throughput: float  # samples/second
    
    # Memory
    gpu_memory_mb: float = 0.0
    
    # Additional info
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"{self.name} (batch={self.batch_size}):\n"
            f"  Latency: {self.mean_ms:.2f} ± {self.std_ms:.2f} ms\n"
            f"  P50/P90/P99: {self.p50_ms:.2f}/{self.p90_ms:.2f}/{self.p99_ms:.2f} ms\n"
            f"  Throughput: {self.throughput:.1f} samples/s"
        )


class Timer:
    """High-precision timer for benchmarking."""
    
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and TORCH_AVAILABLE and torch.cuda.is_available()
        
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Returns elapsed time in milliseconds."""
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (time.perf_counter() - self._start_time) * 1000


@contextmanager
def benchmark_context(name: str = "operation"):
    """Context manager for quick benchmarking."""
    timer = Timer()
    timer.start()
    yield
    elapsed = timer.stop()
    logger.info(f"{name}: {elapsed:.2f}ms")


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    warmup: int = 10,
    iterations: int = 100,
    batch_size: int = 1,
    name: str = "function"
) -> BenchmarkResult:
    """
    Benchmark a function.
    
    Args:
        func: Function to benchmark
        args: Positional arguments
        kwargs: Keyword arguments
        warmup: Warmup iterations
        iterations: Benchmark iterations
        batch_size: Batch size (for throughput calculation)
        name: Name for result
        
    Returns:
        BenchmarkResult with statistics
    """
    kwargs = kwargs or {}
    timer = Timer()
    
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        timer.start()
        func(*args, **kwargs)
        latencies.append(timer.stop())
    
    # Compute statistics
    latencies = np.array(latencies)
    
    return BenchmarkResult(
        name=name,
        batch_size=batch_size,
        iterations=iterations,
        mean_ms=float(np.mean(latencies)),
        std_ms=float(np.std(latencies)),
        min_ms=float(np.min(latencies)),
        max_ms=float(np.max(latencies)),
        p50_ms=float(np.percentile(latencies, 50)),
        p90_ms=float(np.percentile(latencies, 90)),
        p99_ms=float(np.percentile(latencies, 99)),
        throughput=float(batch_size / np.mean(latencies) * 1000),
    )


class PipelineBenchmark:
    """
    Comprehensive pipeline benchmarking.
    
    Benchmarks each stage and full pipeline across batch sizes.
    """
    
    def __init__(
        self,
        pipeline,
        input_shape: Tuple[int, ...] = (1080, 1920, 3),
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        warmup: int = 50,
        iterations: int = 100
    ):
        self.pipeline = pipeline
        self.input_shape = input_shape
        self.batch_sizes = batch_sizes
        self.warmup = warmup
        self.iterations = iterations
        
        self.results: Dict[str, List[BenchmarkResult]] = {
            "preprocess": [],
            "inference": [],
            "postprocess": [],
            "e2e": [],
        }
    
    def run(self) -> Dict:
        """Run full benchmark suite."""
        logger.info("Starting pipeline benchmark...")
        
        for batch_size in self.batch_sizes:
            logger.info(f"Benchmarking batch size: {batch_size}")
            
            # Generate test data
            images = self._generate_test_images(batch_size)
            
            # Warmup
            for _ in range(self.warmup):
                self.pipeline.run(images)
            
            # Benchmark
            latencies = {
                "preprocess": [],
                "inference": [],
                "postprocess": [],
                "e2e": [],
            }
            
            for _ in range(self.iterations):
                self.pipeline.run(images)
                timing = self.pipeline.get_timing()
                
                latencies["preprocess"].append(timing["preprocess_ms"])
                latencies["inference"].append(timing["inference_ms"])
                latencies["postprocess"].append(timing["postprocess_ms"])
                latencies["e2e"].append(timing["total_ms"])
            
            # Store results
            for stage, times in latencies.items():
                times = np.array(times)
                result = BenchmarkResult(
                    name=stage,
                    batch_size=batch_size,
                    iterations=self.iterations,
                    mean_ms=float(np.mean(times)),
                    std_ms=float(np.std(times)),
                    min_ms=float(np.min(times)),
                    max_ms=float(np.max(times)),
                    p50_ms=float(np.percentile(times, 50)),
                    p90_ms=float(np.percentile(times, 90)),
                    p99_ms=float(np.percentile(times, 99)),
                    throughput=float(batch_size / np.mean(times) * 1000),
                )
                self.results[stage].append(result)
        
        return self.get_summary()
    
    def _generate_test_images(self, batch_size: int) -> np.ndarray:
        """Generate random test images."""
        return np.random.randint(
            0, 256,
            size=(batch_size,) + self.input_shape,
            dtype=np.uint8
        )
    
    def get_summary(self) -> Dict:
        """Get benchmark summary."""
        summary = {}
        
        for stage, results in self.results.items():
            summary[stage] = {
                r.batch_size: {
                    "mean_ms": r.mean_ms,
                    "p99_ms": r.p99_ms,
                    "throughput": r.throughput,
                }
                for r in results
            }
        
        return summary
    
    def save_results(self, path: str):
        """Save results to JSON."""
        data = {
            "input_shape": self.input_shape,
            "batch_sizes": self.batch_sizes,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "results": {
                stage: [r.to_dict() for r in results]
                for stage, results in self.results.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {path}")
    
    def print_report(self):
        """Print formatted benchmark report."""
        print("\n" + "="*60)
        print("PIPELINE BENCHMARK REPORT")
        print("="*60)
        
        for stage in ["preprocess", "inference", "postprocess", "e2e"]:
            print(f"\n{stage.upper()}")
            print("-"*40)
            print(f"{'Batch':<8} {'Mean (ms)':<12} {'P99 (ms)':<12} {'Throughput':<12}")
            print("-"*40)
            
            for result in self.results[stage]:
                print(
                    f"{result.batch_size:<8} "
                    f"{result.mean_ms:<12.2f} "
                    f"{result.p99_ms:<12.2f} "
                    f"{result.throughput:<12.1f}"
                )
        
        print("\n" + "="*60)


def compare_implementations(
    implementations: Dict[str, Callable],
    test_input,
    warmup: int = 10,
    iterations: int = 100,
    batch_size: int = 1
) -> Dict[str, BenchmarkResult]:
    """
    Compare multiple implementations.
    
    Args:
        implementations: Dict mapping names to functions
        test_input: Input data for benchmarking
        
    Returns:
        Dict of benchmark results
    """
    results = {}
    
    for name, func in implementations.items():
        logger.info(f"Benchmarking: {name}")
        result = benchmark_function(
            func,
            args=(test_input,),
            warmup=warmup,
            iterations=iterations,
            batch_size=batch_size,
            name=name
        )
        results[name] = result
        print(result)
    
    # Print comparison
    print("\nCOMPARISON")
    print("-"*50)
    
    baseline = list(results.values())[0]
    for name, result in results.items():
        speedup = baseline.mean_ms / result.mean_ms
        print(f"{name}: {result.mean_ms:.2f}ms ({speedup:.2f}x vs baseline)")
    
    return results


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    except:
        return 0.0
