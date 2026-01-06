"""
Rate Limiter Utility

Implements rate limiting for API calls to prevent exceeding provider limits.
Supports multiple strategies: token bucket, sliding window, and fixed window.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_day: int = 10000
    tokens_per_minute: int = 90000
    tokens_per_day: int = 1000000
    retry_after_seconds: float = 60.0
    max_retries: int = 3
    backoff_multiplier: float = 2.0


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire permission for a request."""
        pass
    
    @abstractmethod
    def wait_time(self) -> float:
        """Return time to wait before next request is allowed."""
        pass
    
    @abstractmethod
    async def acquire_async(self, tokens: int = 1) -> bool:
        """Async version of acquire."""
        pass


class TokenBucketRateLimiter(RateLimiter):
    """
    Token bucket rate limiter.
    
    Tokens are added at a fixed rate. Requests consume tokens.
    Allows for burst traffic up to bucket capacity.
    """
    
    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int,  # maximum bucket size
    ):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_time(self) -> float:
        """Calculate time to wait for one token."""
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                return 0.0
            return (1 - self.tokens) / self.rate
    
    async def acquire_async(self, tokens: int = 1) -> bool:
        """Async acquire with automatic waiting."""
        while not self.acquire(tokens):
            wait = self.wait_time()
            if wait > 0:
                await asyncio.sleep(wait)
        return True


class SlidingWindowRateLimiter(RateLimiter):
    """
    Sliding window rate limiter.
    
    Tracks requests in a sliding time window.
    More accurate than fixed window but uses more memory.
    """
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = Lock()
    
    def _cleanup(self) -> None:
        """Remove expired requests from the window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire a request slot."""
        with self._lock:
            self._cleanup()
            if len(self.requests) < self.max_requests:
                self.requests.append(time.monotonic())
                return True
            return False
    
    def wait_time(self) -> float:
        """Calculate time until oldest request expires."""
        with self._lock:
            self._cleanup()
            if len(self.requests) < self.max_requests:
                return 0.0
            if not self.requests:
                return 0.0
            oldest = self.requests[0]
            return max(0.0, (oldest + self.window_seconds) - time.monotonic())
    
    async def acquire_async(self, tokens: int = 1) -> bool:
        """Async acquire with automatic waiting."""
        while not self.acquire(tokens):
            wait = self.wait_time()
            if wait > 0:
                await asyncio.sleep(wait)
        return True


class MultiTierRateLimiter:
    """
    Combines multiple rate limiters for different time windows.
    
    Useful for APIs with multiple rate limits (per-minute and per-day).
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        
        # Per-minute request limiter
        self.rpm_limiter = SlidingWindowRateLimiter(
            max_requests=config.requests_per_minute,
            window_seconds=60.0
        )
        
        # Per-day request limiter
        self.rpd_limiter = SlidingWindowRateLimiter(
            max_requests=config.requests_per_day,
            window_seconds=86400.0
        )
        
        # Token-based limiters
        self.tpm_limiter = TokenBucketRateLimiter(
            rate=config.tokens_per_minute / 60.0,
            capacity=config.tokens_per_minute
        )
        
        self._lock = Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire from all limiters."""
        with self._lock:
            # Check all limiters
            if not self.rpm_limiter.acquire():
                return False
            if not self.rpd_limiter.acquire():
                return False
            if not self.tpm_limiter.acquire(tokens):
                return False
            return True
    
    def wait_time(self) -> float:
        """Return maximum wait time across all limiters."""
        return max(
            self.rpm_limiter.wait_time(),
            self.rpd_limiter.wait_time(),
            self.tpm_limiter.wait_time()
        )
    
    async def acquire_async(self, tokens: int = 1) -> bool:
        """Async acquire with waiting."""
        max_retries = self.config.max_retries
        backoff = self.config.retry_after_seconds
        
        for attempt in range(max_retries):
            if self.acquire(tokens):
                return True
            
            wait = self.wait_time()
            if wait > 0:
                logger.info(f"Rate limited. Waiting {wait:.2f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            
            backoff *= self.config.backoff_multiplier
        
        raise RateLimitExceeded(
            f"Rate limit exceeded after {max_retries} retries",
            retry_after=self.wait_time()
        )


class APIRateLimiter:
    """
    High-level rate limiter for LLM API calls.
    
    Manages rate limits per provider/model with automatic retries.
    """
    
    def __init__(self):
        self._limiters: Dict[str, MultiTierRateLimiter] = {}
        self._lock = Lock()
        
        # Default configs per provider
        self._default_configs = {
            "openai": RateLimitConfig(
                requests_per_minute=60,
                tokens_per_minute=90000,
                requests_per_day=10000,
            ),
            "anthropic": RateLimitConfig(
                requests_per_minute=60,
                tokens_per_minute=100000,
                requests_per_day=10000,
            ),
            "default": RateLimitConfig(),
        }
    
    def get_limiter(self, provider: str) -> MultiTierRateLimiter:
        """Get or create a rate limiter for a provider."""
        with self._lock:
            if provider not in self._limiters:
                config = self._default_configs.get(
                    provider.lower(),
                    self._default_configs["default"]
                )
                self._limiters[provider] = MultiTierRateLimiter(config)
            return self._limiters[provider]
    
    def configure(self, provider: str, config: RateLimitConfig) -> None:
        """Configure rate limits for a provider."""
        with self._lock:
            self._limiters[provider] = MultiTierRateLimiter(config)
    
    async def acquire(self, provider: str, tokens: int = 1) -> bool:
        """Acquire permission to make an API call."""
        limiter = self.get_limiter(provider)
        return await limiter.acquire_async(tokens)
    
    def __call__(self, provider: str):
        """Decorator for rate-limited functions."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Estimate tokens (can be improved with actual counting)
                estimated_tokens = kwargs.get("max_tokens", 1000)
                await self.acquire(provider, estimated_tokens)
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# Global rate limiter instance
rate_limiter = APIRateLimiter()
