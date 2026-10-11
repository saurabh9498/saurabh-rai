"""
Tiered context lookup for the synchronous bid path.

The context engine (an LLM on its own GPU) reads the page and writes a semantic vector to
the hub's local Redis. It needs ~100-150 ms. The first ad slot on a page can fire its bid
~15 ms after the page loads, so the bid often arrives before the page vector exists.

Bidding blind loses the value of context; waiting breaks the 100 ms timeout. So the bid
path never waits. It takes the best vector available right now:

    tier 1  page vector     ctx:page:<url hash>   written by the LLM for this exact page
    tier 2  domain vector   ctx:domain:<domain>   a cached summary of the site, 24 h TTL
    tier 3  default vector  no context at all

and records which tier it used, so coverage can be monitored.

`store` is anything with get(key) -> bytes | None (redis.Redis works unchanged).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np

DIM = 128
DOMAIN_TTL_S = 24 * 3600


def page_key(url: str) -> str:
    return "ctx:page:" + hashlib.sha1(url.encode()).hexdigest()


def domain_key(url: str) -> str:
    host = urlparse(url).hostname or ""
    return "ctx:domain:" + host.removeprefix("www.")


def encode(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float16).tobytes()


def decode(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


@dataclass
class ContextResult:
    vector: np.ndarray
    tier: str               # "page" | "domain" | "default"


class ContextResolver:
    def __init__(self, store, default: np.ndarray | None = None):
        self.store = store
        self.default = np.zeros(DIM, dtype=np.float32) if default is None else default
        self.counts = {"page": 0, "domain": 0, "default": 0}

    def resolve(self, url: str) -> ContextResult:
        for tier, key in (("page", page_key(url)), ("domain", domain_key(url))):
            raw = self.store.get(key)
            if raw is not None:
                self.counts[tier] += 1
                return ContextResult(decode(raw), tier)
        self.counts["default"] += 1
        return ContextResult(self.default, "default")

    def coverage(self) -> dict[str, float]:
        total = sum(self.counts.values()) or 1
        return {k: v / total for k, v in self.counts.items()}


def write_domain_vector(store, url: str, vec: np.ndarray) -> None:
    """Cache a site-level vector with a 24 h TTL (redis.Redis.set accepts ex=)."""
    store.set(domain_key(url), encode(vec), ex=DOMAIN_TTL_S)


class DictStore(dict):
    """In-memory stand-in for Redis in tests and demos."""

    def set(self, key: str, value: bytes, ex: int | None = None) -> None:  # noqa: ARG002
        self[key] = value
