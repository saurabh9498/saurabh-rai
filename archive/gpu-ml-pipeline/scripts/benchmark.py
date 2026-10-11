#!/usr/bin/env python3
"""
Benchmark Script for GPU ML Pipeline

Runs comprehensive benchmarks on preprocessing, inference, and full pipeline.

Usage:
    python scripts/benchmark.py --engine engines/model.engine
    python scripts/benchmark.py --mode preprocessing --batch-sizes 1,8,16,32
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GPU ML Pipeline")
    
    parser.add_argument("--mode", type=str, default="all",
                       choices=["preprocessing", "inference", "pipeline", "all"],
                       help="Benchmark mode")
    parser.add_argument("--engine", type=str,
                       help="TensorRT engine path")
    parser.add_argument("--batch-sizes", type=str, default="1,8,16,32",
                       help="Batch sizes to test (comma-separated)")
    parser.add_argument("--input-size", type=str, default="1080,1920,3",
                       help="Input image size (H,W,C)")
    parser.add_argument("--target-size", type=str, default="224,224",
                       help="Target preprocessing size (H,W)")
    parser.add_argument("--warmup", type=int, default=50,
                       help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Benchmark iterations")
    parser.add_argument("--output", type=str,
                       help="Output JSON file for results")
    parser.add_argument("--compare-cpu", action="store_true",
                       help="Compare GPU vs CPU performance")
    
    return parser.parse_args()


def benchmark_preprocessing(args):
    """Benchmark preprocessing performance."""
    from src.preprocessing.pipeline import GPUPreprocessor, PreprocessConfig
    from src.utils.benchmark import benchmark_function
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    input_shape = tuple(int(x) for x in args.input_size.split(","))
    target_size = tuple(int(x) for x in args.target_size.split(","))
    
    config = PreprocessConfig(target_size=target_size)
    
    results = {"preprocessing": {}}
    
    # GPU preprocessing
    logger.info("Benchmarking GPU preprocessing...")
    try:
        gpu_preprocessor = GPUPreprocessor(config=config, mode="gpu")
        
        for batch_size in batch_sizes:
            images = np.random.randint(0, 256, 
                size=(batch_size,) + input_shape, dtype=np.uint8)
            
            result = benchmark_function(
                gpu_preprocessor.process,
                args=(images,),
                kwargs={"return_numpy": True},
                warmup=args.warmup,
                iterations=args.iterations,
                batch_size=batch_size,
                name=f"gpu_batch_{batch_size}"
            )
            
            results["preprocessing"][f"gpu_batch_{batch_size}"] = {
                "mean_ms": result.mean_ms,
                "p99_ms": result.p99_ms,
                "throughput": result.throughput,
            }
            print(result)
    except Exception as e:
        logger.warning(f"GPU preprocessing not available: {e}")
    
    # CPU preprocessing for comparison
    if args.compare_cpu:
        logger.info("Benchmarking CPU preprocessing...")
        cpu_preprocessor = GPUPreprocessor(config=config, mode="cpu")
        
        for batch_size in batch_sizes:
            images = np.random.randint(0, 256,
                size=(batch_size,) + input_shape, dtype=np.uint8)
            
            result = benchmark_function(
                cpu_preprocessor.process,
                args=(images,),
                kwargs={"return_numpy": True},
                warmup=args.warmup // 5,
                iterations=args.iterations // 5,
                batch_size=batch_size,
                name=f"cpu_batch_{batch_size}"
            )
            
            results["preprocessing"][f"cpu_batch_{batch_size}"] = {
                "mean_ms": result.mean_ms,
                "p99_ms": result.p99_ms,
                "throughput": result.throughput,
            }
            print(result)
    
    return results


def benchmark_inference(args):
    """Benchmark TensorRT inference."""
    if not args.engine:
        logger.error("--engine required for inference benchmark")
        return {}
    
    from src.tensorrt.inference import TensorRTInference, benchmark_engine
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    logger.info(f"Benchmarking inference: {args.engine}")
    
    # Get input shape from engine
    engine = TensorRTInference(args.engine)
    binding_info = engine.get_binding_info()
    input_shape = binding_info["inputs"][0]["shape"]
    
    results = benchmark_engine(
        args.engine,
        input_shape=input_shape,
        warmup=args.warmup,
        iterations=args.iterations,
        batch_sizes=batch_sizes
    )
    
    return {"inference": results}


def benchmark_pipeline(args):
    """Benchmark full pipeline."""
    from src.preprocessing.pipeline import Pipeline, PreprocessConfig
    from src.utils.benchmark import PipelineBenchmark
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    input_shape = tuple(int(x) for x in args.input_size.split(","))
    target_size = tuple(int(x) for x in args.target_size.split(","))
    
    config = PreprocessConfig(target_size=target_size)
    
    pipeline = Pipeline(
        preprocessing="cpu",  # Use cpu for portability
        model_path=args.engine if args.engine else None,
        config=config
    )
    
    benchmark = PipelineBenchmark(
        pipeline=pipeline,
        input_shape=input_shape,
        batch_sizes=batch_sizes,
        warmup=args.warmup,
        iterations=args.iterations
    )
    
    logger.info("Running pipeline benchmark...")
    results = benchmark.run()
    benchmark.print_report()
    
    return {"pipeline": results}


def print_summary(results: dict):
    """Print benchmark summary."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for category, data in results.items():
        print(f"\n{category.upper()}")
        print("-"*40)
        
        for name, metrics in data.items():
            if isinstance(metrics, dict):
                mean = metrics.get("mean_ms", 0)
                throughput = metrics.get("throughput", 0)
                print(f"  {name}: {mean:.2f}ms ({throughput:.0f} samples/s)")


def main():
    args = parse_args()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "batch_sizes": args.batch_sizes,
            "input_size": args.input_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
        }
    }
    
    if args.mode in ["preprocessing", "all"]:
        results.update(benchmark_preprocessing(args))
    
    if args.mode in ["inference", "all"] and args.engine:
        results.update(benchmark_inference(args))
    
    if args.mode in ["pipeline", "all"]:
        results.update(benchmark_pipeline(args))
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
