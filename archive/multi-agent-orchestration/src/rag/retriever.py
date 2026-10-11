"""
Hybrid Retrieval System.

Combines semantic search (vector) with keyword search (BM25)
for improved retrieval quality.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import asyncio
from collections import defaultdict

from .embeddings import EmbeddingModel
from ..core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    content: str
    score: float
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "hybrid"  # "semantic", "keyword", or "hybrid"
    
    def __hash__(self):
        return hash(self.doc_id)
    
    def __eq__(self, other):
        return self.doc_id == other.doc_id


class BM25Retriever:
    """
    BM25 keyword-based retriever.
    
    Implements the Okapi BM25 ranking function for keyword matching.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.k1 = k1
        self.b = b
        
        self.documents: List[Dict[str, Any]] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.vocab: set = set()
    
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents for BM25 retrieval."""
        self.documents = documents
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        
        # Build vocabulary and document frequencies
        for doc in documents:
            tokens = self._tokenize(doc.get("content", ""))
            self.doc_lengths.append(len(tokens))
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
            
            self.vocab.update(unique_tokens)
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        logger.info(f"Indexed {len(documents)} documents, vocab size: {len(self.vocab)}")
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search using BM25 scoring."""
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores = []
        
        n_docs = len(self.documents)
        
        for i, doc in enumerate(self.documents):
            doc_tokens = self._tokenize(doc.get("content", ""))
            doc_length = self.doc_lengths[i]
            
            score = 0
            for token in query_tokens:
                if token not in self.vocab:
                    continue
                
                # Term frequency in document
                tf = doc_tokens.count(token)
                
                # Inverse document frequency
                df = self.doc_freqs[token]
                idf = self._idf(n_docs, df)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * numerator / denominator
            
            scores.append((i, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                doc = self.documents[idx]
                results.append(RetrievalResult(
                    content=doc.get("content", ""),
                    score=score,
                    doc_id=doc.get("id", f"doc_{idx}"),
                    metadata=doc.get("metadata", {}),
                    source="keyword"
                ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _idf(self, n_docs: int, df: int) -> float:
        """Calculate inverse document frequency."""
        import math
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1)


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from
    different retrieval methods.
    """
    
    def __init__(
        self,
        vector_store,
        embedding_model: EmbeddingModel,
        bm25_retriever: Optional[BM25Retriever] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        
        self._indexed = False
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional metadata filters
            use_hybrid: Whether to use hybrid search
            
        Returns:
            List of retrieval results
        """
        # Get semantic results
        semantic_results = await self._semantic_search(query, top_k * 2, filters)
        
        if not use_hybrid or not self._indexed:
            # Return only semantic results
            return semantic_results[:top_k]
        
        # Get keyword results
        keyword_results = self.bm25_retriever.search(query, top_k * 2)
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            top_k
        )
        
        return combined
    
    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform semantic search using vector store."""
        # Generate query embedding
        query_embedding = await self.embedding_model.embed_query(query)
        
        # Query vector store
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # Convert to RetrievalResult
        retrieval_results = []
        
        if results and results.get("documents"):
            documents = results["documents"][0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]
            
            for i, doc in enumerate(documents):
                # Convert distance to similarity score
                score = 1 - (distances[i] if i < len(distances) else 0)
                
                retrieval_results.append(RetrievalResult(
                    content=doc,
                    score=score,
                    doc_id=ids[i] if i < len(ids) else f"doc_{i}",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    source="semantic"
                ))
        
        return retrieval_results
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank))
        """
        # Build score maps
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_objects: Dict[str, RetrievalResult] = {}
        
        # Add semantic scores
        for rank, result in enumerate(semantic_results):
            rrf_score = self.semantic_weight / (self.rrf_k + rank + 1)
            doc_scores[result.doc_id] += rrf_score
            doc_objects[result.doc_id] = result
        
        # Add keyword scores
        for rank, result in enumerate(keyword_results):
            rrf_score = self.keyword_weight / (self.rrf_k + rank + 1)
            doc_scores[result.doc_id] += rrf_score
            if result.doc_id not in doc_objects:
                doc_objects[result.doc_id] = result
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            result = doc_objects[doc_id]
            results.append(RetrievalResult(
                content=result.content,
                score=score,
                doc_id=doc_id,
                metadata=result.metadata,
                source="hybrid"
            ))
        
        return results
    
    def index_for_bm25(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents for BM25 search."""
        self.bm25_retriever.index(documents)
        self._indexed = True
        logger.info("Indexed documents for BM25 search")
