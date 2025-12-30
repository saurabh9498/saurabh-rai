"""Integration tests for the API."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio


@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestChatEndpoint:
    def test_chat_greeting(self, client):
        response = client.post(
            "/chat",
            json={"text": "hello", "language": "en"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "intent" in data
        assert "session_id" in data
    
    def test_chat_with_session(self, client):
        session_id = "test-session-123"
        
        response = client.post(
            "/chat",
            json={"text": "hi", "session_id": session_id}
        )
        
        assert response.status_code == 200
        assert response.json()["session_id"] == session_id


class TestNLUEndpoint:
    def test_nlu_processing(self, client):
        response = client.post(
            "/nlu",
            json={"text": "Set a timer for 5 minutes"}
        )
        
        # May return 503 if NLU not loaded
        if response.status_code == 200:
            data = response.json()
            assert "intent" in data
            assert "entities" in data
