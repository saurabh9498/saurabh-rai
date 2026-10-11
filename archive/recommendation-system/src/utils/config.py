"""
Configuration management for the recommendation system.

Supports loading from YAML files, environment variables, and runtime overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
from dataclasses import dataclass, field, fields, asdict
import yaml
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge multiple configuration dictionaries."""
    result = {}
    
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result


def get_env_override(key: str, prefix: str = 'RECO') -> Optional[str]:
    """Get environment variable override."""
    env_key = f"{prefix}_{key}".upper().replace('.', '_')
    return os.environ.get(env_key)


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = 'localhost'
    port: int = 6379
    password: str = ''
    db: int = 0
    max_connections: int = 100
    socket_timeout: float = 5.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RedisConfig':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class TritonConfig:
    """Triton Inference Server configuration."""
    url: str = 'localhost:8001'
    model_two_tower: str = 'two_tower'
    model_dlrm: str = 'dlrm'
    timeout_ms: int = 50
    max_batch_size: int = 64
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TritonConfig':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class FAISSConfig:
    """FAISS index configuration."""
    index_path: str = '/models/faiss_index'
    index_type: str = 'IVF4096,Flat'
    nprobe: int = 64
    use_gpu: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FAISSConfig':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class RetrievalConfig:
    """Retrieval stage configuration."""
    num_candidates: int = 1000
    cache_ttl: int = 300
    fallback_popular: bool = True
    diversity_factor: float = 0.3
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalConfig':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class RankingConfig:
    """Ranking stage configuration."""
    batch_size: int = 64
    top_k: int = 50
    use_gpu: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RankingConfig':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass 
class APIConfig:
    """API configuration."""
    host: str = '0.0.0.0'
    port: int = 8000
    workers: int = 4
    timeout: int = 10
    max_recommendations: int = 100
    default_recommendations: int = 10
    rate_limit: int = 1000
    cors_origins: str = '*'
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIConfig':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class Config:
    """Main configuration container."""
    
    # Environment
    env: str = 'development'
    debug: bool = False
    
    # Components
    redis: RedisConfig = field(default_factory=RedisConfig)
    triton: TritonConfig = field(default_factory=TritonConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Paths
    model_dir: str = '/models'
    config_dir: str = '/config'
    log_dir: str = '/logs'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        data = load_yaml(path)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        
        # Simple fields
        for key in ['env', 'debug', 'model_dir', 'config_dir', 'log_dir']:
            if key in data:
                setattr(config, key, data[key])
        
        # Nested configs
        if 'redis' in data:
            config.redis = RedisConfig.from_dict(data['redis'])
        if 'triton' in data:
            config.triton = TritonConfig.from_dict(data['triton'])
        if 'faiss' in data:
            config.faiss = FAISSConfig.from_dict(data['faiss'])
        if 'retrieval' in data:
            config.retrieval = RetrievalConfig.from_dict(data['retrieval'])
        if 'ranking' in data:
            config.ranking = RankingConfig.from_dict(data['ranking'])
        if 'api' in data:
            config.api = APIConfig.from_dict(data['api'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'env': self.env,
            'debug': self.debug,
            'model_dir': self.model_dir,
            'config_dir': self.config_dir,
            'log_dir': self.log_dir,
            'redis': asdict(self.redis),
            'triton': asdict(self.triton),
            'faiss': asdict(self.faiss),
            'retrieval': asdict(self.retrieval),
            'ranking': asdict(self.ranking),
            'api': asdict(self.api),
        }
    
    def apply_env_overrides(self, prefix: str = 'RECO'):
        """Apply environment variable overrides."""
        # API overrides
        if val := os.environ.get(f'{prefix}_API_HOST'):
            self.api.host = val
        if val := os.environ.get(f'{prefix}_API_PORT'):
            self.api.port = int(val)
        if val := os.environ.get(f'{prefix}_API_WORKERS'):
            self.api.workers = int(val)
        
        # Redis overrides
        if val := os.environ.get(f'{prefix}_REDIS_HOST'):
            self.redis.host = val
        if val := os.environ.get(f'{prefix}_REDIS_PORT'):
            self.redis.port = int(val)
        if val := os.environ.get(f'{prefix}_REDIS_PASSWORD'):
            self.redis.password = val
        
        # Triton overrides
        if val := os.environ.get(f'{prefix}_TRITON_URL'):
            self.triton.url = val
        
        # General overrides
        if val := os.environ.get(f'{prefix}_ENV'):
            self.env = val
        if val := os.environ.get(f'{prefix}_DEBUG'):
            self.debug = val.lower() in ('true', '1', 'yes')
        if val := os.environ.get(f'{prefix}_MODEL_DIR'):
            self.model_dir = val


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_config(
    config_path: Optional[str] = None,
    apply_env: bool = True,
) -> Config:
    """
    Load configuration from file and environment.
    
    Args:
        config_path: Path to YAML config file
        apply_env: Whether to apply environment variable overrides
        
    Returns:
        Config instance
    """
    global _config
    
    # Default config path
    if config_path is None:
        config_path = os.environ.get('RECO_CONFIG_PATH', 'configs/model_config.yaml')
    
    # Load from file if exists
    if Path(config_path).exists():
        logger.info(f"Loading config from {config_path}")
        config = Config.from_yaml(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = Config()
    
    # Apply environment overrides
    if apply_env:
        config.apply_env_overrides()
    
    _config = config
    return config


def reload_config():
    """Reload configuration from file."""
    global _config
    _config = None
    return get_config()
