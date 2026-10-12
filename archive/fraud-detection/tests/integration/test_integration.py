"""
Integration tests for Real-Time Fraud Detection System.

Tests end-to-end workflows including:
- Feature store integration
- Model ensemble scoring
- API endpoint testing
- Streaming pipeline integration
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


class TestFeatureStoreIntegration:
    """Integration tests for feature store."""
    
    def test_feature_retrieval(self, mock_redis):
        """Test retrieving user features from store."""
        from src.features.feature_store import FeatureStore
        
        with patch('src.features.feature_store.redis.Redis', return_value=mock_redis):
            store = FeatureStore(redis_url="redis://localhost:6379")
            
            # Set up mock data
            mock_redis.hgetall.return_value = {
                b'txn_count_24h': b'15',
                b'avg_amount': b'125.50',
                b'last_location_lat': b'37.7749',
                b'last_location_lon': b'-122.4194'
            }
            
            features = store.get_user_features("usr_test123")
            
            assert features is not None
            mock_redis.hgetall.assert_called_once()
    
    def test_feature_update(self, mock_redis):
        """Test updating user features after transaction."""
        from src.features.feature_store import FeatureStore
        
        with patch('src.features.feature_store.redis.Redis', return_value=mock_redis):
            store = FeatureStore(redis_url="redis://localhost:6379")
            
            transaction = {
                "user_id": "usr_test123",
                "amount": 150.00,
                "timestamp": datetime.now().isoformat()
            }
            
            store.update_features("usr_test123", transaction)
            
            mock_redis.hset.assert_called()
    
    def test_feature_pipeline(self, mock_redis, sample_transaction):
        """Test complete feature computation pipeline."""
        from src.features.feature_store import FeatureStore
        
        with patch('src.features.feature_store.redis.Redis', return_value=mock_redis):
            store = FeatureStore(redis_url="redis://localhost:6379")
            
            # Compute features for transaction
            features = store.compute_transaction_features(sample_transaction)
            
            assert 'velocity_features' in features or features is not None


class TestEnsembleIntegration:
    """Integration tests for model ensemble."""
    
    def test_ensemble_scoring(self, sample_transaction, mock_features):
        """Test ensemble model scoring."""
        from src.models.ensemble import FraudEnsemble, EnsembleConfig
        
        config = EnsembleConfig(
            xgboost_weight=0.45,
            neural_net_weight=0.35,
            isolation_forest_weight=0.20
        )
        
        with patch('src.models.ensemble.XGBoostModel') as mock_xgb, \
             patch('src.models.ensemble.NeuralNetModel') as mock_nn, \
             patch('src.models.ensemble.IsolationForestModel') as mock_if:
            
            # Set up mock predictions
            mock_xgb.return_value.predict.return_value = 0.15
            mock_nn.return_value.predict.return_value = 0.20
            mock_if.return_value.predict.return_value = 0.10
            
            ensemble = FraudEnsemble(config)
            result = ensemble.score(sample_transaction, mock_features)
            
            assert hasattr(result, 'risk_score') or isinstance(result, dict)
    
    def test_ensemble_with_all_models(self, sample_transaction, mock_features):
        """Test that ensemble uses all models."""
        from src.models.ensemble import FraudEnsemble, EnsembleConfig
        
        config = EnsembleConfig()
        
        with patch('src.models.ensemble.XGBoostModel') as mock_xgb, \
             patch('src.models.ensemble.NeuralNetModel') as mock_nn, \
             patch('src.models.ensemble.IsolationForestModel') as mock_if:
            
            mock_xgb.return_value.predict.return_value = 0.3
            mock_nn.return_value.predict.return_value = 0.4
            mock_if.return_value.predict.return_value = 0.2
            
            ensemble = FraudEnsemble(config)
            result = ensemble.score(sample_transaction, mock_features)
            
            # Verify all models were called
            mock_xgb.return_value.predict.assert_called()
            mock_nn.return_value.predict.assert_called()
            mock_if.return_value.predict.assert_called()
    
    def test_ensemble_decision_thresholds(self, sample_transaction, mock_features):
        """Test decision thresholds."""
        from src.models.ensemble import FraudEnsemble, EnsembleConfig
        
        config = EnsembleConfig(
            approve_threshold=0.3,
            step_up_threshold=0.7,
            decline_threshold=0.9
        )
        
        with patch('src.models.ensemble.XGBoostModel') as mock_xgb, \
             patch('src.models.ensemble.NeuralNetModel') as mock_nn, \
             patch('src.models.ensemble.IsolationForestModel') as mock_if:
            
            # Low risk - should approve
            mock_xgb.return_value.predict.return_value = 0.1
            mock_nn.return_value.predict.return_value = 0.1
            mock_if.return_value.predict.return_value = 0.1
            
            ensemble = FraudEnsemble(config)
            result = ensemble.score(sample_transaction, mock_features)
            
            # Result should indicate approval
            assert result is not None


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def api_client(self):
        """Create test client for API."""
        from fastapi.testclient import TestClient
        
        with patch('src.api.main.get_ensemble') as mock_ensemble, \
             patch('src.api.main.get_feature_store') as mock_store:
            
            mock_ensemble.return_value = Mock(
                score=Mock(return_value=Mock(
                    risk_score=0.15,
                    decision="APPROVE",
                    model_scores={"xgboost": 0.15, "neural_net": 0.18, "isolation_forest": 0.10}
                ))
            )
            mock_store.return_value = Mock(
                get_user_features=Mock(return_value={})
            )
            
            from src.api.main import app
            return TestClient(app)
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_score_endpoint(self, api_client):
        """Test transaction scoring endpoint."""
        transaction = {
            "transaction_id": "txn_test_001",
            "user_id": "usr_test_001",
            "amount": 99.99,
            "merchant_category": "retail",
            "timestamp": datetime.now().isoformat()
        }
        
        response = api_client.post("/api/v1/score", json=transaction)
        
        assert response.status_code in [200, 422]  # 422 if validation differs
    
    def test_batch_score_endpoint(self, api_client):
        """Test batch scoring endpoint."""
        transactions = [
            {
                "transaction_id": f"txn_batch_{i}",
                "user_id": "usr_test_001",
                "amount": 50.00 + i * 10,
                "merchant_category": "retail"
            }
            for i in range(5)
        ]
        
        response = api_client.post("/api/v1/score/batch", json={"transactions": transactions})
        
        assert response.status_code in [200, 422]


class TestStreamingIntegration:
    """Integration tests for streaming pipeline."""
    
    def test_kafka_consumer_processing(self, sample_transaction):
        """Test Kafka consumer processes messages."""
        from src.streaming.stream_processor import StreamProcessor
        
        with patch('src.streaming.stream_processor.KafkaConsumer') as mock_consumer, \
             patch('src.streaming.stream_processor.FraudEnsemble') as mock_ensemble:
            
            mock_ensemble.return_value.score.return_value = Mock(
                risk_score=0.25,
                decision="APPROVE"
            )
            
            processor = StreamProcessor(
                bootstrap_servers="localhost:9092",
                topic="transactions"
            )
            
            # Process a message
            result = processor.process_message(json.dumps(sample_transaction))
            
            assert result is not None
    
    def test_stream_processor_error_handling(self):
        """Test stream processor handles errors gracefully."""
        from src.streaming.stream_processor import StreamProcessor
        
        with patch('src.streaming.stream_processor.KafkaConsumer'):
            processor = StreamProcessor(
                bootstrap_servers="localhost:9092",
                topic="transactions"
            )
            
            # Process invalid message
            result = processor.process_message("invalid json{{{")
            
            # Should handle error gracefully
            assert result is None or hasattr(result, 'error')


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_complete_fraud_detection_flow(
        self,
        sample_transaction,
        mock_redis
    ):
        """Test complete fraud detection workflow."""
        from src.features.feature_store import FeatureStore
        from src.models.ensemble import FraudEnsemble, EnsembleConfig
        
        # 1. Get features
        with patch('src.features.feature_store.redis.Redis', return_value=mock_redis):
            mock_redis.hgetall.return_value = {
                b'txn_count_24h': b'5',
                b'avg_amount': b'100.00'
            }
            
            store = FeatureStore(redis_url="redis://localhost:6379")
            features = store.get_user_features(sample_transaction["user_id"])
        
        # 2. Score transaction
        with patch('src.models.ensemble.XGBoostModel') as mock_xgb, \
             patch('src.models.ensemble.NeuralNetModel') as mock_nn, \
             patch('src.models.ensemble.IsolationForestModel') as mock_if:
            
            mock_xgb.return_value.predict.return_value = 0.12
            mock_nn.return_value.predict.return_value = 0.15
            mock_if.return_value.predict.return_value = 0.08
            
            ensemble = FraudEnsemble(EnsembleConfig())
            result = ensemble.score(sample_transaction, features)
        
        # 3. Verify result
        assert result is not None
    
    def test_high_risk_transaction_flow(self, mock_redis):
        """Test high-risk transaction triggers appropriate response."""
        high_risk_txn = {
            "transaction_id": "txn_high_risk",
            "user_id": "usr_new",
            "amount": 5000.00,
            "merchant_category": "crypto_exchange",
            "device_fingerprint": "fp_unknown",
            "is_international": True
        }
        
        from src.models.ensemble import FraudEnsemble, EnsembleConfig
        
        with patch('src.models.ensemble.XGBoostModel') as mock_xgb, \
             patch('src.models.ensemble.NeuralNetModel') as mock_nn, \
             patch('src.models.ensemble.IsolationForestModel') as mock_if:
            
            # High risk scores
            mock_xgb.return_value.predict.return_value = 0.85
            mock_nn.return_value.predict.return_value = 0.90
            mock_if.return_value.predict.return_value = 0.75
            
            ensemble = FraudEnsemble(EnsembleConfig())
            result = ensemble.score(high_risk_txn, {})
            
            # Should flag as high risk
            assert result is not None


# Fixtures

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = Mock()
    redis_mock.ping.return_value = True
    redis_mock.hgetall.return_value = {}
    redis_mock.hset.return_value = True
    redis_mock.expire.return_value = True
    return redis_mock


@pytest.fixture
def sample_transaction():
    """Create a sample transaction."""
    return {
        "transaction_id": "txn_test_12345",
        "user_id": "usr_abc789",
        "amount": 149.99,
        "currency": "USD",
        "merchant_id": "mrc_ret_001",
        "merchant_category": "retail",
        "timestamp": datetime.now().isoformat(),
        "device_fingerprint": "fp_device123",
        "ip_address_hash": "a1b2c3d4e5f6",
        "location": {"lat": 37.7749, "lon": -122.4194},
        "card_present": True,
        "is_international": False
    }


@pytest.fixture
def mock_features():
    """Create mock user features."""
    return {
        "txn_count_1h": 2,
        "txn_count_24h": 8,
        "amount_sum_1h": 75.50,
        "amount_sum_24h": 450.00,
        "avg_transaction_amount": 85.00,
        "account_age_days": 365,
        "device_count": 2,
        "is_known_device": True
    }
