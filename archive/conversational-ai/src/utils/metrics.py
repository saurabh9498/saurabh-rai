"""Performance metrics collection."""

import time
from typing import Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading


@dataclass
class MetricsCollector:
    """Collect and aggregate performance metrics."""
    
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_latency(self, name: str, latency_ms: float):
        with self._lock:
            self.latencies[name].append(latency_ms)
            self.counts[f"{name}_count"] += 1
    
    def increment(self, name: str, value: int = 1):
        with self._lock:
            self.counts[name] += value
    
    def get_stats(self, name: str) -> Dict[str, float]:
        with self._lock:
            lats = self.latencies.get(name, [])
            if not lats:
                return {}
            import numpy as np
            return {
                "count": len(lats),
                "mean": np.mean(lats),
                "p50": np.percentile(lats, 50),
                "p95": np.percentile(lats, 95),
                "p99": np.percentile(lats, 99),
            }
    
    def reset(self):
        with self._lock:
            self.counts.clear()
            self.latencies.clear()


metrics = MetricsCollector()
