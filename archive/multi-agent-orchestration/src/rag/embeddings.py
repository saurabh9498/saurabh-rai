"""
Embedding Model Abstraction.

This module provides a unified interface for different embedding models,
supporting both OpenAI and local/open-source models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import asyncio
import logging
from functools import lru_cache

from ..core.config import settings


logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        pass
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        pass
    
    def embed_query_sync(self, text: str) -> List[float]:
        """Synchronous version of embed_query."""
        return asyncio.run(self.embed_query(text))
    
    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous version of embed_documents."""
        return asyncio.run(self.embed_documents(texts))


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model implementation."""
    
    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        self._model = model
        self._dimension = self.MODELS.get(model, 1536)
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key or settings.llm.openai_api_key)
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = await self.client.embeddings.create(
            model=self._model,
            input=text
        )
        return response.data[0].embedding
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with batching."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = await self.client.embeddings.create(
                model=self._model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Embedded batch {i//self.batch_size + 1}")
        
        return all_embeddings


class HuggingFaceEmbedding(EmbeddingModel):
    """Hugging Face embedding model for local inference."""
    
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        self._model_name = model
        self.device = device
        self.batch_size = batch_size
        
        # Load model
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Loaded embedding model: {model} on {device}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        # Run in thread pool for async compatibility
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100
            )
        )
        return embeddings.tolist()


class CohereEmbedding(EmbeddingModel):
    """Cohere embedding model implementation."""
    
    MODELS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
    }
    
    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        batch_size: int = 96
    ):
        self._model = model
        self._dimension = self.MODELS.get(model, 1024)
        self.batch_size = batch_size
        
        import cohere
        self.client = cohere.Client(api_key)
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.embed(
                texts=[text],
                model=self._model,
                input_type="search_query"
            )
        )
        return response.embeddings[0]
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        all_embeddings = []
        loop = asyncio.get_event_loop()
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = await loop.run_in_executor(
                None,
                lambda b=batch: self.client.embed(
                    texts=b,
                    model=self._model,
                    input_type="search_document"
                )
            )
            
            all_embeddings.extend(response.embeddings)
        
        return all_embeddings


def get_embedding_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> EmbeddingModel:
    """
    Factory function to get an embedding model.
    
    Args:
        provider: Provider name (openai, huggingface, cohere)
        model: Model name
        **kwargs: Provider-specific arguments
        
    Returns:
        EmbeddingModel instance
    """
    provider = provider or "openai"
    
    if provider == "openai":
        model = model or settings.embedding.model_name
        return OpenAIEmbedding(model=model, **kwargs)
    
    elif provider == "huggingface":
        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbedding(model=model, **kwargs)
    
    elif provider == "cohere":
        model = model or "embed-english-v3.0"
        return CohereEmbedding(model=model, **kwargs)
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
