"""
RAG (Retrieval-Augmented Generation) Module.

Provides document ingestion, chunking, embedding, retrieval,
and reranking capabilities for knowledge-grounded generation.
"""

from .pipeline import RAGPipeline, Document, QueryResult
from .chunker import DocumentChunker, ChunkingStrategy, ChunkConfig
from .embeddings import (
    EmbeddingModel,
    OpenAIEmbedding,
    HuggingFaceEmbedding,
    get_embedding_model
)
from .retriever import (
    HybridRetriever,
    BM25Retriever,
    RetrievalResult
)
from .reranker import Reranker, get_reranker

__all__ = [
    # Pipeline
    "RAGPipeline",
    "Document",
    "QueryResult",
    # Chunking
    "DocumentChunker",
    "ChunkingStrategy",
    "ChunkConfig",
    # Embeddings
    "EmbeddingModel",
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    "get_embedding_model",
    # Retrieval
    "HybridRetriever",
    "BM25Retriever",
    "RetrievalResult",
    # Reranking
    "Reranker",
    "get_reranker",
]
