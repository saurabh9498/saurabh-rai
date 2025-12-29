"""
Cross-Encoder Reranker.

This module provides reranking capabilities using cross-encoder models
to improve retrieval precision after initial retrieval.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import asyncio

from .retriever import RetrievalResult


logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker for improving retrieval quality.
    
    Uses a cross-encoder model to score query-document pairs
    and rerank the initial retrieval results.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        use_gpu: bool = False
    ):
        self.model_name = model_name
        self.device = "cuda" if use_gpu else device
        self.batch_size = batch_size
        self._model = None
        self._initialized = False
    
    def _initialize_model(self) -> None:
        """Lazy initialization of the cross-encoder model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            self._initialized = True
            logger.info(f"Loaded reranker model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using score-based fallback")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self._initialized = True
    
    async def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Rerank documents using cross-encoder scoring.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        # Initialize model if needed
        self._initialize_model()
        
        if self._model is None:
            # Fallback: return documents sorted by original score
            sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
            return sorted_docs[:top_k]
        
        # Prepare query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Score pairs using cross-encoder
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, batch_size=self.batch_size)
        )
        
        # Combine documents with new scores
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Build reranked results
        reranked = []
        for doc, score in scored_docs[:top_k]:
            reranked.append(RetrievalResult(
                content=doc.content,
                score=float(score),
                doc_id=doc.doc_id,
                metadata={**doc.metadata, "original_score": doc.score},
                source="reranked"
            ))
        
        logger.debug(f"Reranked {len(documents)} documents, returning top {top_k}")
        return reranked


class CohereReranker:
    """Cohere API-based reranker."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v2.0"
    ):
        self.model = model
        
        import cohere
        self.client = cohere.Client(api_key)
    
    async def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank using Cohere's rerank API."""
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        # Prepare documents
        doc_texts = [doc.content for doc in documents]
        
        # Call Cohere rerank API
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_k
            )
        )
        
        # Build reranked results
        reranked = []
        for result in response.results:
            original_doc = documents[result.index]
            reranked.append(RetrievalResult(
                content=original_doc.content,
                score=result.relevance_score,
                doc_id=original_doc.doc_id,
                metadata={**original_doc.metadata, "original_score": original_doc.score},
                source="reranked"
            ))
        
        return reranked


class LLMReranker:
    """
    LLM-based reranker using prompting.
    
    Useful when cross-encoder models aren't available or
    when more nuanced relevance judgments are needed.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    async def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank using LLM-based relevance scoring."""
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        # Score each document
        scores = []
        for doc in documents:
            score = await self._score_document(query, doc.content)
            scores.append(score)
        
        # Combine and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        reranked = []
        for doc, score in scored_docs[:top_k]:
            reranked.append(RetrievalResult(
                content=doc.content,
                score=score,
                doc_id=doc.doc_id,
                metadata={**doc.metadata, "original_score": doc.score},
                source="llm_reranked"
            ))
        
        return reranked
    
    async def _score_document(self, query: str, document: str) -> float:
        """Score a single document's relevance to the query."""
        prompt = f"""Rate the relevance of the following document to the query on a scale of 0-10.

Query: {query}

Document: {document[:500]}...

Respond with only a number from 0 to 10."""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm.generate(messages, max_tokens=10)
        
        try:
            score = float(response.content.strip()) / 10.0
            return min(max(score, 0.0), 1.0)
        except ValueError:
            return 0.5


def get_reranker(
    provider: str = "cross-encoder",
    **kwargs
) -> Reranker:
    """
    Factory function to get a reranker instance.
    
    Args:
        provider: Reranker provider (cross-encoder, cohere, llm)
        **kwargs: Provider-specific arguments
        
    Returns:
        Reranker instance
    """
    if provider == "cross-encoder":
        return Reranker(**kwargs)
    elif provider == "cohere":
        return CohereReranker(**kwargs)
    elif provider == "llm":
        return LLMReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
