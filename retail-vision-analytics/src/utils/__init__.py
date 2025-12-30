"""
Retail Vision Analytics - Utilities Module.

Common utilities and helper functions.

Components:
- config: Configuration loading and validation
- logging: Structured logging setup
- metrics: Performance metrics collection
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class AppConfig:
    """Application configuration container."""
    
    app: Dict[str, Any]
    api: Dict[str, Any]
    pipeline: Dict[str, Any]
    analytics: Dict[str, Any]
    alerts: Dict[str, Any]
    storage: Dict[str, Any]
    edge: Dict[str, Any]
    sources: list
    performance: Dict[str, Any]
    monitoring: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create config from dictionary."""
        return cls(
            app=data.get("app", {}),
            api=data.get("api", {}),
            pipeline=data.get("pipeline", {}),
            analytics=data.get("analytics", {}),
            alerts=data.get("alerts", {}),
            storage=data.get("storage", {}),
            edge=data.get("edge", {}),
            sources=data.get("sources", []),
            performance=data.get("performance", {}),
            monitoring=data.get("monitoring", {}),
        )


def load_config(config_path: str) -> AppConfig:
    """
    Load application configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        AppConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Substitute environment variables
    data = _substitute_env_vars(data)
    
    logger.info(f"Loaded configuration from {config_path}")
    return AppConfig.from_dict(data)


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, obj)
        return obj
    return obj


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
):
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
    )
    
    logger.info(f"Logging configured: level={level}")


class MetricsCollector:
    """Simple metrics collector for performance monitoring."""
    
    def __init__(self):
        self._metrics: Dict[str, list] = {}
        self._counters: Dict[str, int] = {}
    
    def record(self, name: str, value: float):
        """Record a metric value."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)
        
        # Keep only last 1000 values
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-1000:]
    
    def increment(self, name: str, value: int = 1):
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + value
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        import numpy as np
        
        values = self._metrics.get(name, [])
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0}
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "min": np.min(values),
            "max": np.max(values),
            "p95": np.percentile(values, 95),
        }
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self._counters.get(name, 0)
    
    def reset(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._counters.clear()


# Global metrics collector
metrics = MetricsCollector()


__all__ = [
    "AppConfig",
    "load_config",
    "setup_logging",
    "MetricsCollector",
    "metrics",
]
