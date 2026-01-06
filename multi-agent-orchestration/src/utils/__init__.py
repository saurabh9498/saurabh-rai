"""
Utilities Module

Common utilities for the multi-agent system including:
- Rate limiting for API calls
- Token counting and cost tracking
- Response caching
- Structured logging
"""

from .rate_limiter import (
    RateLimitConfig,
    RateLimitExceeded,
    RateLimiter,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    MultiTierRateLimiter,
    APIRateLimiter,
    rate_limiter,
)

from .token_counter import (
    TokenUsage,
    UsageStats,
    TokenCounter,
    TiktokenCounter,
    ApproximateCounter,
    TokenTracker,
    token_tracker,
    MODEL_PRICING,
)

from .cache import (
    CacheEntry,
    CacheBackend,
    MemoryCache,
    DiskCache,
    RedisCache,
    LLMCache,
    cached_llm_call,
    llm_cache,
)

from .logger import (
    JSONFormatter,
    ColoredFormatter,
    AgentLogger,
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    log_execution_time,
)

__all__ = [
    # Rate Limiter
    "RateLimitConfig",
    "RateLimitExceeded",
    "RateLimiter",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "MultiTierRateLimiter",
    "APIRateLimiter",
    "rate_limiter",
    # Token Counter
    "TokenUsage",
    "UsageStats",
    "TokenCounter",
    "TiktokenCounter",
    "ApproximateCounter",
    "TokenTracker",
    "token_tracker",
    "MODEL_PRICING",
    # Cache
    "CacheEntry",
    "CacheBackend",
    "MemoryCache",
    "DiskCache",
    "RedisCache",
    "LLMCache",
    "cached_llm_call",
    "llm_cache",
    # Logger
    "JSONFormatter",
    "ColoredFormatter",
    "AgentLogger",
    "setup_logging",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "log_execution_time",
]
