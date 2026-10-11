"""
Basic Completion Example

Demonstrates how to use the multi-agent system for simple completions.
Shows basic LLM interaction with rate limiting, caching, and token tracking.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import system components
from src.core.llm import LLMClient
from src.utils import (
    setup_logging,
    get_logger,
    token_tracker,
    rate_limiter,
    llm_cache,
)

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


async def basic_completion():
    """
    Simple completion with a single prompt.
    
    This example shows the most basic usage of the LLM client.
    """
    logger.info("Starting basic completion example")
    
    # Initialize LLM client
    client = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
    )
    
    # Simple prompt
    prompt = "Explain quantum computing in one paragraph for a high school student."
    
    # Get completion
    response = await client.complete(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
    )
    
    print("\n" + "="*60)
    print("BASIC COMPLETION EXAMPLE")
    print("="*60)
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse:\n{response}")
    print("="*60 + "\n")
    
    return response


async def completion_with_tracking():
    """
    Completion with token tracking and cost estimation.
    
    Demonstrates usage monitoring capabilities.
    """
    logger.info("Starting completion with tracking example")
    
    client = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
    )
    
    # Estimate cost before making the call
    prompt = "Write a haiku about artificial intelligence."
    prompt_tokens = token_tracker.count_tokens(text=prompt, model="gpt-4o-mini")
    estimated_cost = token_tracker.estimate_cost(
        prompt_tokens=prompt_tokens,
        max_completion_tokens=100,
        model="gpt-4o-mini",
    )
    
    print(f"\nEstimated cost: ${estimated_cost:.6f}")
    
    # Make the completion
    response = await client.complete(
        prompt=prompt,
        max_tokens=100,
        temperature=0.9,
    )
    
    # Get actual usage stats
    stats = token_tracker.get_stats()
    
    print("\n" + "="*60)
    print("COMPLETION WITH TRACKING")
    print("="*60)
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse:\n{response}")
    print(f"\n--- Usage Statistics ---")
    print(f"Total requests: {stats.total_requests}")
    print(f"Total tokens: {stats.total_tokens:,}")
    print(f"Total cost: ${stats.total_cost:.6f}")
    print("="*60 + "\n")
    
    return response


async def cached_completion():
    """
    Completion with caching enabled.
    
    Shows how caching can improve response times and reduce costs.
    """
    logger.info("Starting cached completion example")
    
    client = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
    )
    
    prompt = "What is the capital of France?"
    
    # First call - will hit the API
    print("\nFirst call (cache miss)...")
    import time
    start = time.time()
    response1 = await client.complete(prompt=prompt, max_tokens=50)
    time1 = time.time() - start
    
    # Second call - should hit cache
    print("Second call (cache hit)...")
    start = time.time()
    response2 = await client.complete(prompt=prompt, max_tokens=50)
    time2 = time.time() - start
    
    print("\n" + "="*60)
    print("CACHED COMPLETION")
    print("="*60)
    print(f"\nPrompt: {prompt}")
    print(f"\nFirst call time: {time1:.3f}s")
    print(f"Second call time: {time2:.3f}s")
    print(f"Speedup: {time1/time2:.1f}x")
    print(f"\nCache stats: {llm_cache.stats()}")
    print("="*60 + "\n")
    
    return response2


async def rate_limited_completions():
    """
    Multiple completions with rate limiting.
    
    Demonstrates how rate limiting protects against API quota exhaustion.
    """
    logger.info("Starting rate limited completions example")
    
    client = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
    )
    
    prompts = [
        "Name a famous scientist.",
        "Name a famous artist.",
        "Name a famous musician.",
        "Name a famous author.",
        "Name a famous inventor.",
    ]
    
    print("\n" + "="*60)
    print("RATE LIMITED COMPLETIONS")
    print("="*60)
    
    responses = []
    for i, prompt in enumerate(prompts):
        # Rate limiter automatically handles throttling
        await rate_limiter.acquire("openai", tokens=100)
        
        response = await client.complete(prompt=prompt, max_tokens=50)
        responses.append(response)
        
        print(f"\n{i+1}. {prompt}")
        print(f"   {response.strip()}")
    
    print("\n" + "="*60 + "\n")
    
    return responses


async def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# MULTI-AGENT SYSTEM - BASIC COMPLETION EXAMPLES")
    print("#"*60)
    
    try:
        # Run examples
        await basic_completion()
        await completion_with_tracking()
        await cached_completion()
        await rate_limited_completions()
        
        # Print final statistics
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        
        stats = token_tracker.get_stats()
        print(f"\nSession Summary:")
        print(f"  Total requests: {stats.total_requests}")
        print(f"  Total tokens: {stats.total_tokens:,}")
        print(f"  Total cost: ${stats.total_cost:.6f}")
        print(f"  Cache hit rate: {llm_cache.hit_rate:.1%}")
        
        if stats.by_model:
            print(f"\nBy Model:")
            for model, model_stats in stats.by_model.items():
                print(f"  {model}:")
                print(f"    Requests: {model_stats['requests']}")
                print(f"    Tokens: {model_stats['total_tokens']:,}")
                print(f"    Cost: ${model_stats['cost']:.6f}")
        
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
