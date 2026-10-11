"""
Pytest configuration and fixtures for Multi-Agent AI System tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Generator


# ============================================
# Async Configuration
# ============================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================
# Mock LLM Fixtures
# ============================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=Mock(
        content='{"result": "test response"}',
        usage={"total_tokens": 100}
    ))
    return llm


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    def _create_response(content: str):
        return Mock(
            content=content,
            usage={"total_tokens": len(content) // 4}
        )
    return _create_response


# ============================================
# Agent Fixtures
# ============================================

@pytest.fixture
def mock_research_agent(mock_llm):
    """Create a mock research agent."""
    from src.agents import ResearchAgent
    
    agent = ResearchAgent(llm=mock_llm, rag_pipeline=None)
    return agent


@pytest.fixture
def mock_analyst_agent(mock_llm):
    """Create a mock analyst agent."""
    from src.agents import AnalystAgent
    
    agent = AnalystAgent(llm=mock_llm)
    return agent


@pytest.fixture
def mock_code_agent(mock_llm):
    """Create a mock code agent."""
    from src.agents import CodeAgent
    
    agent = CodeAgent(llm=mock_llm, enable_execution=False)
    return agent


# ============================================
# Context Fixtures
# ============================================

@pytest.fixture
def conversation_context():
    """Create a fresh conversation context."""
    from src.core.context import ConversationContext
    return ConversationContext()


# ============================================
# RAG Fixtures
# ============================================

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    store.add = Mock()
    store.query = Mock(return_value={
        "documents": [["Test document content"]],
        "distances": [[0.1]],
        "metadatas": [[{"source": "test"}]],
        "ids": [["doc_1"]]
    })
    store.count = Mock(return_value=10)
    return store


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock()
    model.model_name = "test-embedding-model"
    model.dimension = 1536
    model.embed_query = AsyncMock(return_value=[0.1] * 1536)
    model.embed_documents = AsyncMock(return_value=[[0.1] * 1536])
    return model


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    from src.rag import Document
    
    return [
        Document(
            content="This is a test document about AI.",
            metadata={"source": "test1.txt"}
        ),
        Document(
            content="Another document about machine learning.",
            metadata={"source": "test2.txt"}
        ),
    ]


# ============================================
# API Fixtures
# ============================================

@pytest.fixture
def test_client():
    """Create a test client for API testing."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Create an async test client for API testing."""
    from httpx import AsyncClient
    from src.api.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ============================================
# Pytest Markers
# ============================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, multi-component)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (slowest, full system)"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
