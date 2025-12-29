"""
Configuration management for Multi-Agent AI System.

This module handles all configuration loading from environment variables
and config files, providing a centralized configuration interface.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from functools import lru_cache
import yaml


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    
    provider: str = Field(default="openai", env="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    model_name: str = Field(default="gpt-4-turbo", env="LLM_MODEL")
    temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, env="LLM_MAX_TOKENS")
    
    class Config:
        env_file = ".env"


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""
    
    provider: str = Field(default="chromadb", env="VECTOR_STORE")
    persist_directory: str = Field(default="./data/chroma", env="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="knowledge_base", env="COLLECTION_NAME")
    
    # Pinecone settings (if used)
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENV")
    pinecone_index: Optional[str] = Field(default=None, env="PINECONE_INDEX")
    
    class Config:
        env_file = ".env"


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    model_name: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")
    
    class Config:
        env_file = ".env"


class RAGConfig(BaseSettings):
    """RAG pipeline configuration."""
    
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=10, env="TOP_K_RETRIEVAL")
    rerank_top_k: int = Field(default=5, env="RERANK_TOP_K")
    use_hybrid_search: bool = Field(default=True, env="USE_HYBRID_SEARCH")
    semantic_weight: float = Field(default=0.7, env="SEMANTIC_WEIGHT")
    keyword_weight: float = Field(default=0.3, env="KEYWORD_WEIGHT")
    
    class Config:
        env_file = ".env"


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""
    
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=60, env="AGENT_TIMEOUT")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    parallel_execution: bool = Field(default=False, env="PARALLEL_EXECUTION")
    
    class Config:
        env_file = ".env"


class APIConfig(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    rate_limit: int = Field(default=100, env="RATE_LIMIT")
    
    class Config:
        env_file = ".env"


class Settings(BaseSettings):
    """Main application settings."""
    
    app_name: str = "Multi-Agent AI System"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Sub-configurations
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    rag: RAGConfig = RAGConfig()
    agent: AgentConfig = AgentConfig()
    api: APIConfig = APIConfig()
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def load_agent_config(config_path: str = "config/agents.yaml") -> Dict[str, Any]:
    """Load agent-specific configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


# Convenience function for accessing settings
settings = get_settings()
