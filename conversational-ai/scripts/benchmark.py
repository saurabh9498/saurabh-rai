"""Benchmark system performance."""

import argparse
import asyncio
import time
import statistics
import httpx
import numpy as np
from concurrent.futures import ThreadPoolExecutor


async def benchmark_chat(url: str, num_requests: int, concurrency: int):
    """Benchmark chat endpoint."""
    latencies = []
    errors = 0
    
    test_queries = [
        "What's the weather like?",
        "Set a timer for 5 minutes",
        "Play some music",
        "What time is it?",
        "Remind me to call mom tomorrow",
    ]
    
    async def make_request(client, query):
        start = time.time()
        try:
            response = await client.post(
                f"{url}/chat",
                json={"text": query},
                timeout=10.0,
            )
            latency = (time.time() - start) * 1000
            return latency, response.status_code == 200
        except Exception:
            return None, False
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(num_requests):
            query = test_queries[i % len(test_queries)]
            tasks.append(make_request(client, query))
        
        results = await asyncio.gather(*tasks)
        
        for latency, success in results:
            if success and latency:
                latencies.append(latency)
            else:
                errors += 1
    
    return {
        "total_requests": num_requests,
        "successful": len(latencies),
        "errors": errors,
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0,
        "latency_p50_ms": statistics.median(latencies) if latencies else 0,
        "latency_p95_ms": np.percentile(latencies, 95) if latencies else 0,
        "latency_p99_ms": np.percentile(latencies, 99) if latencies else 0,
        "throughput_rps": len(latencies) / (sum(latencies) / 1000) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Conversational AI")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    
    args = parser.parse_args()
    
    print(f"Benchmarking {args.url} with {args.requests} requests...")
    
    results = asyncio.run(benchmark_chat(args.url, args.requests, args.concurrency))
    
    print("\n=== Benchmark Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
