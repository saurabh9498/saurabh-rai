"""
RAG (Retrieval-Augmented Generation) Pipeline.

This module provides the main RAG pipeline that combines
document retrieval with LLM generation for knowledge-grounded responses.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import asyncio

from .chunker import DocumentChunker, ChunkingStrategy
from .embeddings import EmbeddingModel, get_embedding_model
from .retriever import HybridRetriever, RetrievalResult
from .reranker import Reranker
from ..core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document in the knowledge base."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            import hashlib
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class QueryResult:
    """Result from a RAG query."""
    query: str
    documents: List[Document]
    scores: List[float]
    context: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates document ingestion,
    retrieval, and context assembly for LLM generation.
    
    Features:
    - Multiple chunking strategies
    - Hybrid search (semantic + keyword)
    - Cross-encoder reranking
    - Context window management
    """
    
    def __init__(
        self,
        vector_store=None,
        embedding_model: Optional[EmbeddingModel] = None,
        chunker: Optional[DocumentChunker] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        collection_name: str = "knowledge_base"
    ):
        self.collection_name = collection_name
        
        # Initialize components
        self.embedding_model = embedding_model or get_embedding_model()
        self.chunker = chunker or DocumentChunker()
        
        # Initialize vector store
        if vector_store is None:
            self.vector_store = self._initialize_vector_store()
        else:
            self.vector_store = vector_store
        
        # Initialize retriever and reranker
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model
        )
        self.reranker = reranker or Reranker()
        
        # Configuration
        self.top_k = settings.rag.top_k_retrieval
        self.rerank_top_k = settings.rag.rerank_top_k
        self.max_context_tokens = 4000
    
    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration."""
        provider = settings.vector_store.provider
        
        if provider == "chromadb":
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            client = chromadb.Client(ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=settings.vector_store.persist_directory,
                anonymized_telemetry=False
            ))
            
            collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            return collection
        
        elif provider == "pinecone":
            import pinecone
            
            pinecone.init(
                api_key=settings.vector_store.pinecone_api_key,
                environment=settings.vector_store.pinecone_environment
            )
            
            return pinecone.Index(settings.vector_store.pinecone_index)
        
        else:
            raise ValueError(f"Unsupported vector store: {provider}")
    
    async def ingest_documents(
        self,
        documents: Union[List[str], List[Document], str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ) -> Dict[str, Any]:
        """
        Ingest documents into the knowledge base.
        
        Args:
            documents: Documents to ingest (strings, Document objects, or path)
            metadata: Optional metadata to attach to all documents
            chunking_strategy: How to chunk the documents
            
        Returns:
            Ingestion statistics
        """
        # Handle different input types
        if isinstance(documents, (str, Path)):
            documents = self._load_documents_from_path(documents)
        elif isinstance(documents, list) and documents and isinstance(documents[0], str):
            documents = [Document(content=doc, metadata=metadata or {}) for doc in documents]
        
        # Chunk documents
        chunks = []
        for doc in documents:
            doc_chunks = self.chunker.chunk(
                doc.content,
                strategy=chunking_strategy,
                metadata={**doc.metadata, "doc_id": doc.doc_id}
            )
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Generate embeddings
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await self.embedding_model.embed_documents(chunk_texts)
        
        # Store in vector database
        ids = [f"{chunk['metadata'].get('doc_id', 'doc')}_{i}" for i, chunk in enumerate(chunks)]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        self.vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas
        )
        
        logger.info(f"Ingested {len(chunks)} chunks into vector store")
        
        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "embeddings_generated": len(embeddings)
        }
    
    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_reranking: bool = True
    ) -> QueryResult:
        """
        Query the knowledge base.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional metadata filters
            use_reranking: Whether to apply reranking
            
        Returns:
            QueryResult with retrieved documents and context
        """
        top_k = top_k or self.top_k
        
        # Step 1: Retrieve candidates
        retrieval_results = await self.retriever.retrieve(
            query=query,
            top_k=top_k * 2 if use_reranking else top_k,  # Get more for reranking
            filters=filters
        )
        
        # Step 2: Rerank if enabled
        if use_reranking and len(retrieval_results) > 0:
            retrieval_results = await self.reranker.rerank(
                query=query,
                documents=retrieval_results,
                top_k=self.rerank_top_k
            )
        
        # Step 3: Build documents
        documents = []
        scores = []
        for result in retrieval_results:
            documents.append(Document(
                content=result.content,
                metadata=result.metadata,
                doc_id=result.doc_id
            ))
            scores.append(result.score)
        
        # Step 4: Assemble context
        context = self._assemble_context(documents)
        
        return QueryResult(
            query=query,
            documents=documents,
            scores=scores,
            context=context,
            metadata={
                "retrieval_count": len(retrieval_results),
                "final_count": len(documents)
            }
        )
    
    async def query_with_generation(
        self,
        query: str,
        llm,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query and generate a response using the LLM.
        
        Args:
            query: The user query
            llm: LLM instance to use for generation
            system_prompt: Optional system prompt
            top_k: Number of documents to retrieve
            
        Returns:
            Generated response with sources
        """
        # Retrieve relevant documents
        query_result = await self.query(query, top_k=top_k)
        
        # Build prompt with context
        system_prompt = system_prompt or """You are a helpful assistant. Answer questions based on the provided context.
If the context doesn't contain relevant information, say so clearly.
Always cite your sources when possible."""
        
        context_prompt = f"""Context:
{query_result.context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_prompt}
        ]
        
        response = await llm.generate(messages)
        
        return {
            "answer": response.content,
            "sources": [
                {
                    "content": doc.content[:200] + "...",
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in zip(query_result.documents, query_result.scores)
            ],
            "query": query
        }
    
    def _load_documents_from_path(self, path: Union[str, Path]) -> List[Document]:
        """Load documents from a file or directory."""
        path = Path(path)
        documents = []
        
        if path.is_file():
            documents.append(self._load_single_file(path))
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix in [".txt", ".md", ".pdf", ".docx"]:
                    try:
                        documents.append(self._load_single_file(file_path))
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
        
        return documents
    
    def _load_single_file(self, path: Path) -> Document:
        """Load a single file into a Document."""
        suffix = path.suffix.lower()
        
        if suffix in [".txt", ".md"]:
            content = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            # Placeholder - in production, use a PDF library
            content = f"PDF content from {path.name}"
        elif suffix == ".docx":
            # Placeholder - in production, use python-docx
            content = f"DOCX content from {path.name}"
        else:
            content = path.read_text(encoding="utf-8", errors="ignore")
        
        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": suffix
            }
        )
    
    def _assemble_context(
        self,
        documents: List[Document],
        max_tokens: Optional[int] = None
    ) -> str:
        """Assemble retrieved documents into a context string."""
        max_tokens = max_tokens or self.max_context_tokens
        
        context_parts = []
        estimated_tokens = 0
        
        for i, doc in enumerate(documents):
            # Rough token estimation (4 chars ≈ 1 token)
            doc_tokens = len(doc.content) // 4
            
            if estimated_tokens + doc_tokens > max_tokens:
                # Truncate this document
                remaining_tokens = max_tokens - estimated_tokens
                truncated_content = doc.content[:remaining_tokens * 4]
                context_parts.append(f"[Source {i+1}]\n{truncated_content}...")
                break
            
            context_parts.append(f"[Source {i+1}]\n{doc.content}")
            estimated_tokens += doc_tokens
        
        return "\n\n".join(context_parts)
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        if hasattr(self.vector_store, '_client'):
            self.vector_store._client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        count = self.vector_store.count() if hasattr(self.vector_store, 'count') else 0
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedding_model.model_name,
            "vector_store": settings.vector_store.provider
        }
