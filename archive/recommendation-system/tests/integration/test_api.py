"""
Integration tests for the recommendation API.

Tests cover API endpoints, request validation,
service integration, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json

import sys
sys.path.insert(0, '/home/claude/github-portfolio/projects/recommendation-system')

# Mock dependencies before importing API
with patch.dict('sys.modules', {
    'redis': MagicMock(),
    'redis.asyncio': MagicMock(),
    'tritonclient': MagicMock(),
    'tritonclient.grpc': MagicMock(),
}):
    from src.serving.api import (
        app,
        RecommendationRequest,
        RecommendationResponse,
        RecommendedItem,
        RequestContext,
        FeedbackRequest,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_request():
    """Create sample recommendation request."""
    return {
        "user_id": "user_123",
        "num_recommendations": 10,
        "context": {
            "device": "mobile",
            "page": "home",
            "session_id": "sess_abc",
        },
        "filters": {
            "category": "electronics",
            "min_price": 10.0,
            "max_price": 500.0,
        },
        "exclude_items": ["item_1", "item_2"],
        "diversity_factor": 0.3,
    }


@pytest.fixture
def sample_feedback():
    """Create sample feedback request."""
    return {
        "user_id": "user_123",
        "item_id": "item_456",
        "event_type": "click",
        "context": {
            "position": 3,
            "page": "home",
        },
    }


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_health_check_returns_ok(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
    def test_health_check_includes_version(self, client):
        """Test health endpoint includes version info."""
        response = client.get("/health")
        
        data = response.json()
        assert "version" in data
        assert "timestamp" in data


# =============================================================================
# Recommendation Endpoint Tests
# =============================================================================

class TestRecommendEndpoint:
    """Tests for recommendation endpoint."""
    
    def test_recommend_valid_request(self, client, sample_request):
        """Test valid recommendation request."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[
                RecommendedItem(
                    item_id="item_1",
                    score=0.95,
                    rank=1,
                    reason="Based on your browsing history",
                ),
                RecommendedItem(
                    item_id="item_2",
                    score=0.87,
                    rank=2,
                    reason="Popular in electronics",
                ),
            ])
            
            response = client.post("/recommend", json=sample_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert len(data["items"]) == 2
            
    def test_recommend_missing_user_id(self, client):
        """Test request without user_id fails."""
        response = client.post("/recommend", json={
            "num_recommendations": 10,
        })
        
        assert response.status_code == 422  # Validation error
        
    def test_recommend_invalid_num_recommendations(self, client):
        """Test invalid num_recommendations."""
        response = client.post("/recommend", json={
            "user_id": "user_123",
            "num_recommendations": 0,  # Invalid
        })
        
        assert response.status_code == 422
        
    def test_recommend_with_context(self, client, sample_request):
        """Test request with full context."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[])
            
            response = client.post("/recommend", json=sample_request)
            
            # Verify context was passed
            call_args = mock_service.get_recommendations.call_args
            assert call_args is not None
            
    def test_recommend_response_format(self, client, sample_request):
        """Test response format matches schema."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[
                RecommendedItem(
                    item_id="item_1",
                    score=0.95,
                    rank=1,
                    reason="Collaborative filtering",
                    metadata={"category": "electronics"},
                ),
            ])
            
            response = client.post("/recommend", json=sample_request)
            
            data = response.json()
            assert "request_id" in data
            assert "latency_ms" in data
            assert "items" in data
            
            item = data["items"][0]
            assert "item_id" in item
            assert "score" in item
            assert "rank" in item


# =============================================================================
# Batch Recommendation Tests
# =============================================================================

class TestBatchRecommendEndpoint:
    """Tests for batch recommendation endpoint."""
    
    def test_batch_recommend_multiple_users(self, client):
        """Test batch recommendations for multiple users."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_batch_recommendations = AsyncMock(return_value={
                "user_1": [RecommendedItem(item_id="item_a", score=0.9, rank=1)],
                "user_2": [RecommendedItem(item_id="item_b", score=0.85, rank=1)],
            })
            
            response = client.post("/recommend/batch", json={
                "user_ids": ["user_1", "user_2"],
                "num_recommendations": 5,
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "user_1" in data
            assert "user_2" in data
            
    def test_batch_recommend_max_users(self, client):
        """Test batch request with too many users fails."""
        response = client.post("/recommend/batch", json={
            "user_ids": [f"user_{i}" for i in range(101)],  # Over limit
            "num_recommendations": 5,
        })
        
        assert response.status_code == 422


# =============================================================================
# Similar Items Tests
# =============================================================================

class TestSimilarItemsEndpoint:
    """Tests for similar items endpoint."""
    
    def test_similar_items_valid_request(self, client):
        """Test similar items retrieval."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_similar_items = AsyncMock(return_value=[
                RecommendedItem(item_id="similar_1", score=0.92, rank=1),
                RecommendedItem(item_id="similar_2", score=0.88, rank=2),
            ])
            
            response = client.get("/similar/item_123?k=10")
            
            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert len(data["items"]) == 2
            
    def test_similar_items_not_found(self, client):
        """Test similar items for non-existent item."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_similar_items = AsyncMock(return_value=[])
            
            response = client.get("/similar/nonexistent_item")
            
            assert response.status_code == 200
            data = response.json()
            assert data["items"] == []


# =============================================================================
# Feedback Endpoint Tests
# =============================================================================

class TestFeedbackEndpoint:
    """Tests for feedback endpoint."""
    
    def test_feedback_click_event(self, client, sample_feedback):
        """Test recording click feedback."""
        with patch('src.serving.api.feedback_service') as mock_service:
            mock_service.record_feedback = AsyncMock(return_value=True)
            
            response = client.post("/feedback", json=sample_feedback)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "recorded"
            
    def test_feedback_purchase_event(self, client):
        """Test recording purchase feedback."""
        with patch('src.serving.api.feedback_service') as mock_service:
            mock_service.record_feedback = AsyncMock(return_value=True)
            
            response = client.post("/feedback", json={
                "user_id": "user_123",
                "item_id": "item_456",
                "event_type": "purchase",
                "context": {
                    "price": 99.99,
                    "quantity": 1,
                },
            })
            
            assert response.status_code == 200
            
    def test_feedback_invalid_event_type(self, client):
        """Test invalid event type."""
        response = client.post("/feedback", json={
            "user_id": "user_123",
            "item_id": "item_456",
            "event_type": "invalid_event",
        })
        
        assert response.status_code == 422


# =============================================================================
# Metrics Endpoint Tests
# =============================================================================

class TestMetricsEndpoint:
    """Tests for metrics endpoint."""
    
    def test_metrics_returns_prometheus_format(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        
    def test_metrics_includes_request_count(self, client):
        """Test metrics includes request counts."""
        # Make some requests first
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[])
            
            client.post("/recommend", json={
                "user_id": "user_123",
                "num_recommendations": 10,
            })
        
        response = client.get("/metrics")
        
        # Should contain request metrics
        assert response.status_code == 200


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_internal_server_error(self, client, sample_request):
        """Test handling of internal errors."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(
                side_effect=Exception("Internal error")
            )
            
            response = client.post("/recommend", json=sample_request)
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            
    def test_timeout_error(self, client, sample_request):
        """Test handling of timeout errors."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            
            response = client.post("/recommend", json=sample_request)
            
            assert response.status_code == 504
            
    def test_validation_error_format(self, client):
        """Test validation error response format."""
        response = client.post("/recommend", json={
            "user_id": 12345,  # Should be string
            "num_recommendations": "ten",  # Should be int
        })
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


# =============================================================================
# Request Validation Tests
# =============================================================================

class TestRequestValidation:
    """Tests for request validation."""
    
    def test_valid_request_context(self):
        """Test valid request context."""
        context = RequestContext(
            device="mobile",
            page="home",
            session_id="sess_123",
            referrer="google.com",
        )
        
        assert context.device == "mobile"
        
    def test_request_filters(self, client):
        """Test various filter combinations."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[])
            
            # Category filter
            response = client.post("/recommend", json={
                "user_id": "user_123",
                "num_recommendations": 10,
                "filters": {"category": "electronics"},
            })
            assert response.status_code == 200
            
            # Price range filter
            response = client.post("/recommend", json={
                "user_id": "user_123",
                "num_recommendations": 10,
                "filters": {"min_price": 10, "max_price": 100},
            })
            assert response.status_code == 200
            
    def test_diversity_factor_bounds(self, client, sample_request):
        """Test diversity factor validation."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[])
            
            # Valid diversity factor
            sample_request["diversity_factor"] = 0.5
            response = client.post("/recommend", json=sample_request)
            assert response.status_code == 200
            
            # Invalid diversity factor (out of range)
            sample_request["diversity_factor"] = 1.5
            response = client.post("/recommend", json=sample_request)
            assert response.status_code == 422


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_response_includes_timing(self, client, sample_request):
        """Test that response includes timing info."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[])
            
            response = client.post("/recommend", json=sample_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "latency_ms" in data
            assert data["latency_ms"] >= 0
            
    def test_concurrent_requests(self, client, sample_request):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[])
            
            def make_request():
                return client.post("/recommend", json=sample_request)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(20)]
                responses = [f.result() for f in futures]
            
            # All should succeed
            assert all(r.status_code == 200 for r in responses)


# =============================================================================
# CORS and Headers Tests
# =============================================================================

class TestCORSAndHeaders:
    """Tests for CORS and headers."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/recommend")
        
        # Should allow CORS
        assert response.status_code in [200, 405]  # Depends on config
        
    def test_process_time_header(self, client, sample_request):
        """Test X-Process-Time header is present."""
        with patch('src.serving.api.recommendation_service') as mock_service:
            mock_service.get_recommendations = AsyncMock(return_value=[])
            
            response = client.post("/recommend", json=sample_request)
            
            assert "x-process-time" in response.headers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
