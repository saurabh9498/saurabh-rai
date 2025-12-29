"""
Unit tests for RAG module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.rag.chunker import DocumentChunker, ChunkingStrategy, ChunkConfig
from src.rag.retriever import BM25Retriever, RetrievalResult


class TestDocumentChunker:
    """Tests for DocumentChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a document chunker with default config."""
        return DocumentChunker(ChunkConfig(chunk_size=100, chunk_overlap=10))
    
    def test_fixed_chunking(self, chunker):
        """Test fixed-size chunking."""
        text = "a" * 250  # 250 characters
        chunks = chunker.chunk(text, strategy=ChunkingStrategy.FIXED)
        
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["chunk_strategy"] == "fixed"
    
    def test_recursive_chunking(self, chunker):
        """Test recursive chunking."""
        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph to ensure we have enough text."""
        
        chunks = chunker.chunk(text, strategy=ChunkingStrategy.RECURSIVE)
        
        assert len(chunks) >= 1
        assert all("content" in c for c in chunks)
    
    def test_sentence_chunking(self, chunker):
        """Test sentence-based chunking."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunker.chunk(text, strategy=ChunkingStrategy.SENTENCE)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["metadata"]["chunk_strategy"] == "sentence"
    
    def test_paragraph_chunking(self, chunker):
        """Test paragraph-based chunking."""
        text = """First paragraph.

Second paragraph.

Third paragraph."""
        
        chunks = chunker.chunk(text, strategy=ChunkingStrategy.PARAGRAPH)
        
        assert len(chunks) >= 1
    
    def test_markdown_chunking(self):
        """Test markdown-aware chunking."""
        chunker = DocumentChunker(ChunkConfig(chunk_size=200))
        
        text = """# Header 1

Some content under header 1.

## Header 2

More content here.

```python
def hello():
    print("Hello")
```
"""
        
        chunks = chunker.chunk(text, strategy=ChunkingStrategy.MARKDOWN)
        
        assert len(chunks) >= 1
        # Code blocks should be preserved
        code_found = any("def hello" in c["content"] for c in chunks)
        assert code_found
    
    def test_empty_text(self, chunker):
        """Test handling of empty text."""
        chunks = chunker.chunk("")
        assert chunks == []
        
        chunks = chunker.chunk("   ")
        assert chunks == []
    
    def test_metadata_preservation(self, chunker):
        """Test that metadata is preserved in chunks."""
        text = "Some test content that should be chunked."
        metadata = {"source": "test.txt", "author": "tester"}
        
        chunks = chunker.chunk(text, metadata=metadata)
        
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.txt"
            assert chunk["metadata"]["author"] == "tester"
            assert "chunk_index" in chunk["metadata"]


class TestBM25Retriever:
    """Tests for BM25Retriever class."""
    
    @pytest.fixture
    def bm25(self):
        """Create a BM25 retriever."""
        return BM25Retriever()
    
    def test_indexing(self, bm25):
        """Test document indexing."""
        documents = [
            {"content": "The quick brown fox", "id": "doc1"},
            {"content": "The lazy dog sleeps", "id": "doc2"},
            {"content": "Quick brown dogs run", "id": "doc3"},
        ]
        
        bm25.index(documents)
        
        assert len(bm25.documents) == 3
        assert len(bm25.doc_lengths) == 3
        assert "quick" in bm25.vocab
        assert "brown" in bm25.vocab
    
    def test_search(self, bm25):
        """Test BM25 search."""
        documents = [
            {"content": "Python programming language", "id": "doc1"},
            {"content": "Java programming language", "id": "doc2"},
            {"content": "Python is great for AI", "id": "doc3"},
        ]
        
        bm25.index(documents)
        results = bm25.search("Python programming", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Python should be in top results
        assert any("Python" in r.content for r in results)
    
    def test_search_empty_index(self, bm25):
        """Test search on empty index."""
        results = bm25.search("test query")
        assert results == []
    
    def test_search_no_match(self, bm25):
        """Test search with no matching terms."""
        documents = [{"content": "apple orange banana", "id": "doc1"}]
        bm25.index(documents)
        
        results = bm25.search("xyz123")
        assert len(results) == 0


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_creation(self):
        """Test RetrievalResult creation."""
        result = RetrievalResult(
            content="Test content",
            score=0.95,
            doc_id="doc1",
            metadata={"source": "test"},
            source="semantic"
        )
        
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.doc_id == "doc1"
        assert result.source == "semantic"
    
    def test_equality(self):
        """Test RetrievalResult equality based on doc_id."""
        result1 = RetrievalResult(content="A", score=0.9, doc_id="doc1")
        result2 = RetrievalResult(content="B", score=0.8, doc_id="doc1")
        result3 = RetrievalResult(content="A", score=0.9, doc_id="doc2")
        
        assert result1 == result2  # Same doc_id
        assert result1 != result3  # Different doc_id
    
    def test_hashable(self):
        """Test that RetrievalResult is hashable."""
        result = RetrievalResult(content="Test", score=0.9, doc_id="doc1")
        
        # Should be usable in sets
        result_set = {result}
        assert len(result_set) == 1


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""
    
    def test_defaults(self):
        """Test default configuration values."""
        config = ChunkConfig()
        
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.min_chunk_size == 100
        assert config.separators is not None
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = ChunkConfig(
            chunk_size=256,
            chunk_overlap=25,
            separators=["\n", " "]
        )
        
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25
        assert config.separators == ["\n", " "]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
