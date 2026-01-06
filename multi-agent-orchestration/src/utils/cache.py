"""
Cache Utility

Provides caching mechanisms for LLM responses and embeddings.
Supports multiple backends: memory, disk, and Redis.
"""

import hashlib
import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from threading import Lock
from functools import wraps
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """Update hit count and access time."""
        self.hits += 1


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None, **metadata) -> None:
        """Store an entry in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an entry from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class MemoryCache(CacheBackend):
    """
    In-memory cache with LRU eviction.
    
    Fast but not persistent across restarts.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self.max_size:
            # Find oldest entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at
            )
            del self._cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key[:32]}...")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve from memory cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._cache[key]
                return None
            entry.touch()
            return entry
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, **metadata) -> None:
        """Store in memory cache."""
        with self._lock:
            self._evict_if_needed()
            
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                metadata=metadata,
            )
    
    def delete(self, key: str) -> bool:
        """Delete from memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear memory cache."""
        with self._lock:
            self._cache.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        entry = self.get(key)
        return entry is not None
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_hits = sum(e.hits for e in self._cache.values())
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_hits": total_hits,
            }


class DiskCache(CacheBackend):
    """
    Disk-based cache for persistence.
    
    Slower but survives restarts and handles large data.
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "_index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()
        self._lock = Lock()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)
    
    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Use hash to avoid filesystem issues with long/special keys
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve from disk cache."""
        with self._lock:
            if key not in self._index:
                return None
            
            meta = self._index[key]
            if meta.get("expires_at") and time.time() > meta["expires_at"]:
                self.delete(key)
                return None
            
            filepath = self._key_to_path(key)
            if not filepath.exists():
                del self._index[key]
                self._save_index()
                return None
            
            try:
                with open(filepath, "rb") as f:
                    entry = pickle.load(f)
                entry.touch()
                return entry
            except Exception as e:
                logger.warning(f"Failed to load cached entry: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, **metadata) -> None:
        """Store in disk cache."""
        with self._lock:
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                metadata=metadata,
            )
            
            filepath = self._key_to_path(key)
            with open(filepath, "wb") as f:
                pickle.dump(entry, f)
            
            self._index[key] = {
                "filepath": str(filepath),
                "created_at": entry.created_at,
                "expires_at": expires_at,
            }
            self._save_index()
    
    def delete(self, key: str) -> bool:
        """Delete from disk cache."""
        with self._lock:
            if key not in self._index:
                return False
            
            filepath = self._key_to_path(key)
            if filepath.exists():
                filepath.unlink()
            
            del self._index[key]
            self._save_index()
            return True
    
    def clear(self) -> None:
        """Clear disk cache."""
        with self._lock:
            for key in list(self._index.keys()):
                filepath = self._key_to_path(key)
                if filepath.exists():
                    filepath.unlink()
            self._index.clear()
            self._save_index()
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._index


class RedisCache(CacheBackend):
    """
    Redis-based cache for distributed caching.
    
    Best for multi-instance deployments.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "llm_cache:",
    ):
        self.prefix = prefix
        self._redis = None
        self._config = {"host": host, "port": port, "db": db}
    
    @property
    def redis(self):
        """Lazy Redis connection."""
        if self._redis is None:
            try:
                import redis
                self._redis = redis.Redis(**self._config)
                self._redis.ping()
            except ImportError:
                raise ImportError("redis package required for RedisCache")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Redis: {e}")
        return self._redis
    
    def _prefixed_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve from Redis."""
        try:
            data = self.redis.get(self._prefixed_key(key))
            if data is None:
                return None
            entry = pickle.loads(data)
            entry.touch()
            return entry
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, **metadata) -> None:
        """Store in Redis."""
        try:
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                metadata=metadata,
            )
            
            prefixed = self._prefixed_key(key)
            data = pickle.dumps(entry)
            
            if ttl:
                self.redis.setex(prefixed, ttl, data)
            else:
                self.redis.set(prefixed, data)
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete from Redis."""
        try:
            return self.redis.delete(self._prefixed_key(key)) > 0
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all keys with prefix."""
        try:
            pattern = f"{self.prefix}*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return self.redis.exists(self._prefixed_key(key)) > 0
        except Exception:
            return False


class LLMCache:
    """
    High-level caching for LLM responses.
    
    Provides semantic caching with configurable backends.
    """
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: int = 3600,  # 1 hour
        enabled: bool = True,
    ):
        self.backend = backend or MemoryCache()
        self.default_ttl = default_ttl
        self.enabled = enabled
        self._stats = {"hits": 0, "misses": 0}
    
    def _generate_key(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate cache key from request parameters."""
        key_data = {
            "model": model,
            "messages": messages,
            "prompt": prompt,
            **{k: v for k, v in kwargs.items() if k in ["temperature", "max_tokens"]},
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[Any]:
        """Get cached response if available."""
        if not self.enabled:
            return None
        
        key = self._generate_key(model, messages, prompt, **kwargs)
        entry = self.backend.get(key)
        
        if entry is not None:
            self._stats["hits"] += 1
            logger.debug(f"Cache hit for {model}")
            return entry.value
        
        self._stats["misses"] += 1
        return None
    
    def set(
        self,
        response: Any,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        ttl: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Cache a response."""
        if not self.enabled:
            return
        
        key = self._generate_key(model, messages, prompt, **kwargs)
        self.backend.set(
            key=key,
            value=response,
            ttl=ttl or self.default_ttl,
            model=model,
        )
        logger.debug(f"Cached response for {model}")
    
    def invalidate(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Invalidate a cached response."""
        key = self._generate_key(model, messages, prompt, **kwargs)
        return self.backend.delete(key)
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.backend.clear()
        self._stats = {"hits": 0, "misses": 0}
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        return self._stats["hits"] / total
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self._stats,
            "hit_rate": self.hit_rate,
        }


def cached_llm_call(
    cache: LLMCache,
    ttl: Optional[int] = None,
) -> Callable:
    """
    Decorator for caching LLM function calls.
    
    Usage:
        @cached_llm_call(cache=my_cache, ttl=3600)
        async def get_completion(model, messages):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get from cache
            model = kwargs.get("model", args[0] if args else "unknown")
            messages = kwargs.get("messages")
            prompt = kwargs.get("prompt")
            
            cached = cache.get(model=model, messages=messages, prompt=prompt, **kwargs)
            if cached is not None:
                return cached
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.set(
                response=result,
                model=model,
                messages=messages,
                prompt=prompt,
                ttl=ttl,
                **kwargs,
            )
            return result
        
        return wrapper
    return decorator


# Global cache instance
llm_cache = LLMCache()
