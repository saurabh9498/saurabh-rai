#!/usr/bin/env python3
"""
Latency Benchmarking Script

Benchmarks recommendation system performance:
- Model inference latency (P50, P95, P99)
- Throughput (QPS)
- Memory usage
- GPU utilization

Usage:
    python scripts/benchmark.py --model two_tower --batch-sizes 1,8,32,64,128
    python scripts/benchmark.py --endpoint http://localhost:8000/recommend --concurrent 10
"""

import argparse
import asyncio
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    batch_size: int
    num_iterations: int
    
    # Latency stats (ms)
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    
    # Throughput
    throughput_qps: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # Memory (MB)
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    
    # GPU stats
    gpu_utilization: float = 0.0
    gpu_memory_mb: float = 0.0
    
    raw_latencies: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'batch_size': self.batch_size,
            'num_iterations': self.num_iterations,
            'latency': {
                'mean_ms': round(self.latency_mean, 3),
                'std_ms': round(self.latency_std, 3),
                'p50_ms': round(self.latency_p50, 3),
                'p95_ms': round(self.latency_p95, 3),
                'p99_ms': round(self.latency_p99, 3),
                'min_ms': round(self.latency_min, 3),
                'max_ms': round(self.latency_max, 3),
            },
            'throughput': {
                'qps': round(self.throughput_qps, 2),
                'samples_per_sec': round(self.throughput_samples_per_sec, 2),
            },
            'memory': {
                'peak_mb': round(self.memory_peak_mb, 2),
                'allocated_mb': round(self.memory_allocated_mb, 2),
            },
            'gpu': {
                'utilization_percent': round(self.gpu_utilization, 2),
                'memory_mb': round(self.gpu_memory_mb, 2),
            },
        }


class ModelBenchmark:
    """Benchmarks PyTorch model inference."""
    
    def __init__(
        self,
        model: 'torch.nn.Module',
        device: str = 'cuda',
        warmup_iterations: int = 10,
    ):
        self.model = model
        self.device = device
        self.warmup_iterations = warmup_iterations
        
        self.model.to(device)
        self.model.eval()
    
    def generate_inputs(
        self,
        batch_size: int,
        model_type: str = 'dlrm',
    ) -> Dict[str, 'torch.Tensor']:
        """Generate random inputs for benchmarking."""
        if model_type == 'dlrm':
            return {
                'sparse_features': torch.randint(0, 100, (batch_size, 26)).to(self.device),
                'dense_features': torch.randn(batch_size, 13).to(self.device),
            }
        elif model_type == 'two_tower':
            return {
                'user_categorical': torch.randint(0, 100, (batch_size, 10)).to(self.device),
                'user_dense': torch.randn(batch_size, 8).to(self.device),
                'user_history': torch.randint(0, 1000, (batch_size, 50)).to(self.device),
                'history_mask': torch.ones(batch_size, 50).to(self.device),
                'item_categorical': torch.randint(0, 100, (batch_size, 10)).to(self.device),
                'item_dense': torch.randn(batch_size, 8).to(self.device),
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def benchmark(
        self,
        batch_size: int,
        num_iterations: int = 100,
        model_type: str = 'dlrm',
    ) -> BenchmarkResult:
        """Run benchmark for given batch size."""
        logger.info(f"Benchmarking batch_size={batch_size}, iterations={num_iterations}")
        
        inputs = self.generate_inputs(batch_size, model_type)
        
        # Warmup
        logger.info(f"Running {self.warmup_iterations} warmup iterations...")
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = self.model(**inputs)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        latencies = []
        
        logger.info(f"Running {num_iterations} benchmark iterations...")
        with torch.no_grad():
            for i in range(num_iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model(**inputs)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
        
        # Calculate stats
        result = BenchmarkResult(
            name=model_type,
            batch_size=batch_size,
            num_iterations=num_iterations,
            raw_latencies=latencies,
        )
        
        result.latency_mean = statistics.mean(latencies)
        result.latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0
        result.latency_min = min(latencies)
        result.latency_max = max(latencies)
        
        sorted_latencies = sorted(latencies)
        result.latency_p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        result.latency_p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        result.latency_p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        total_time = sum(latencies) / 1000  # seconds
        result.throughput_qps = num_iterations / total_time
        result.throughput_samples_per_sec = (num_iterations * batch_size) / total_time
        
        # Memory stats
        if self.device == 'cuda':
            result.memory_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            result.memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return result


class EndpointBenchmark:
    """Benchmarks API endpoint performance."""
    
    def __init__(
        self,
        endpoint_url: str,
        warmup_requests: int = 10,
    ):
        self.endpoint_url = endpoint_url
        self.warmup_requests = warmup_requests
    
    def generate_request(self) -> Dict[str, Any]:
        """Generate sample recommendation request."""
        return {
            "user_id": f"user_{np.random.randint(1, 10000)}",
            "num_recommendations": 10,
            "context": {
                "device": "mobile",
                "page": "home",
            },
        }
    
    async def benchmark_async(
        self,
        num_requests: int = 100,
        concurrent: int = 10,
    ) -> BenchmarkResult:
        """Run async benchmark with concurrent requests."""
        logger.info(f"Benchmarking endpoint: {self.endpoint_url}")
        logger.info(f"Requests: {num_requests}, Concurrency: {concurrent}")
        
        latencies = []
        errors = 0
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Warmup
            logger.info(f"Running {self.warmup_requests} warmup requests...")
            warmup_tasks = [
                client.post(self.endpoint_url, json=self.generate_request())
                for _ in range(self.warmup_requests)
            ]
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
            
            # Benchmark
            semaphore = asyncio.Semaphore(concurrent)
            
            async def make_request():
                async with semaphore:
                    request = self.generate_request()
                    start = time.perf_counter()
                    try:
                        response = await client.post(self.endpoint_url, json=request)
                        response.raise_for_status()
                        end = time.perf_counter()
                        return (end - start) * 1000  # ms
                    except Exception as e:
                        logger.error(f"Request failed: {e}")
                        return None
            
            logger.info(f"Running {num_requests} benchmark requests...")
            tasks = [make_request() for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if result is not None:
                    latencies.append(result)
                else:
                    errors += 1
        
        if not latencies:
            raise RuntimeError("All requests failed")
        
        # Calculate stats
        result = BenchmarkResult(
            name="endpoint",
            batch_size=1,
            num_iterations=num_requests,
            raw_latencies=latencies,
        )
        
        result.latency_mean = statistics.mean(latencies)
        result.latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0
        result.latency_min = min(latencies)
        result.latency_max = max(latencies)
        
        sorted_latencies = sorted(latencies)
        result.latency_p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        result.latency_p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        result.latency_p99 = sorted_latencies[min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)]
        
        total_time = sum(latencies) / 1000
        result.throughput_qps = len(latencies) / total_time
        
        logger.info(f"Completed. Errors: {errors}/{num_requests}")
        
        return result


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Header
    print(f"{'Batch':>8} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'QPS':>12} {'Samples/s':>12}")
    print("-" * 80)
    
    for result in results:
        print(
            f"{result.batch_size:>8} "
            f"{result.latency_mean:>9.2f}ms "
            f"{result.latency_p50:>9.2f}ms "
            f"{result.latency_p95:>9.2f}ms "
            f"{result.latency_p99:>9.2f}ms "
            f"{result.throughput_qps:>12.1f} "
            f"{result.throughput_samples_per_sec:>12.1f}"
        )
    
    print("=" * 80)
    
    # Memory stats (if available)
    if results[0].memory_peak_mb > 0:
        print(f"\nGPU Memory: Peak={results[-1].memory_peak_mb:.1f}MB, "
              f"Allocated={results[-1].memory_allocated_mb:.1f}MB")


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file."""
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [r.to_dict() for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark recommendation system')
    
    subparsers = parser.add_subparsers(dest='mode', help='Benchmark mode')
    
    # Model benchmark
    model_parser = subparsers.add_parser('model', help='Benchmark model inference')
    model_parser.add_argument('--model', type=str, default='dlrm', choices=['dlrm', 'two_tower'])
    model_parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    model_parser.add_argument('--batch-sizes', type=str, default='1,8,32,64,128')
    model_parser.add_argument('--iterations', type=int, default=100)
    model_parser.add_argument('--device', type=str, default='cuda')
    model_parser.add_argument('--warmup', type=int, default=10)
    
    # Endpoint benchmark
    endpoint_parser = subparsers.add_parser('endpoint', help='Benchmark API endpoint')
    endpoint_parser.add_argument('--url', type=str, default='http://localhost:8000/recommend')
    endpoint_parser.add_argument('--requests', type=int, default=100)
    endpoint_parser.add_argument('--concurrent', type=int, default=10)
    endpoint_parser.add_argument('--warmup', type=int, default=10)
    
    # Common args
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    results = []
    
    if args.mode == 'model':
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not installed")
            sys.exit(1)
        
        # Create dummy model for demo (in real use, load from checkpoint)
        from src.models.dlrm import DLRM, DLRMConfig
        from src.models.two_tower import TwoTowerModel, TwoTowerConfig
        
        if args.model == 'dlrm':
            config = DLRMConfig()
            model = DLRM(config)
        else:
            config = TwoTowerConfig()
            model = TwoTowerModel(config)
        
        benchmark = ModelBenchmark(
            model=model,
            device=args.device,
            warmup_iterations=args.warmup,
        )
        
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
        
        for batch_size in batch_sizes:
            result = benchmark.benchmark(
                batch_size=batch_size,
                num_iterations=args.iterations,
                model_type=args.model,
            )
            results.append(result)
        
    elif args.mode == 'endpoint':
        if not HTTPX_AVAILABLE:
            logger.error("httpx not installed")
            sys.exit(1)
        
        benchmark = EndpointBenchmark(
            endpoint_url=args.url,
            warmup_requests=args.warmup,
        )
        
        result = asyncio.run(benchmark.benchmark_async(
            num_requests=args.requests,
            concurrent=args.concurrent,
        ))
        results.append(result)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # Print and save results
    print_results(results)
    
    if args.output:
        save_results(results, args.output)


if __name__ == '__main__':
    main()
