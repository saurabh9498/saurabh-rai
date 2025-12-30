"""
Unit Tests for Fraud Detection Models

Tests for XGBoost, Neural Network, Isolation Forest, and Ensemble.
"""

import numpy as np
import pytest
from datetime import datetime


class TestXGBoostModel:
    """Tests for XGBoost fraud model."""
    
    def test_mock_model_predict(self):
        """Test mock model predictions."""
        from src.models.xgboost_model import MockXGBoostModel
        
        model = MockXGBoostModel()
        X = np.random.randn(10, 17).astype(np.float32)
        
        predictions = model.predict_proba(X)
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        
    def test_config_defaults(self):
        """Test default configuration."""
        from src.models.xgboost_model import XGBoostConfig
        
        config = XGBoostConfig()
        
        assert config.n_estimators == 500
        assert config.max_depth == 8
        assert config.scale_pos_weight == 10.0


class TestNeuralNetModel:
    """Tests for Neural Network fraud model."""
    
    def test_mock_model_predict(self):
        """Test mock model predictions."""
        from src.models.neural_net import MockNeuralNetModel
        
        model = MockNeuralNetModel()
        X = np.random.randn(10, 17).astype(np.float32)
        
        predictions = model.predict_proba(X)
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        
    def test_config_defaults(self):
        """Test default configuration."""
        from src.models.neural_net import NeuralNetConfig
        
        config = NeuralNetConfig()
        
        assert config.hidden_dims == [256, 128, 64]
        assert config.dropout == 0.3


class TestIsolationForestModel:
    """Tests for Isolation Forest model."""
    
    def test_mock_model_predict(self):
        """Test mock model predictions."""
        from src.models.isolation_forest import MockIsolationForestModel
        
        model = MockIsolationForestModel()
        X = np.random.randn(10, 17).astype(np.float32)
        
        predictions = model.predict_proba(X)
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)


class TestEnsemble:
    """Tests for Fraud Ensemble."""
    
    @pytest.fixture
    def mock_ensemble(self):
        """Create mock ensemble."""
        from src.models.ensemble import FraudEnsemble, EnsembleConfig
        from src.models.xgboost_model import MockXGBoostModel
        from src.models.neural_net import MockNeuralNetModel
        from src.models.isolation_forest import MockIsolationForestModel
        
        ensemble = FraudEnsemble()
        ensemble.xgboost = MockXGBoostModel()
        ensemble.neural_net = MockNeuralNetModel()
        ensemble.isolation_forest = MockIsolationForestModel()
        ensemble._is_trained = True
        
        return ensemble
    
    def test_ensemble_score(self, mock_ensemble):
        """Test ensemble scoring."""
        X = np.random.randn(17).astype(np.float32)
        
        result = mock_ensemble.score(X, transaction_id="test_001")
        
        assert result.transaction_id == "test_001"
        assert 0 <= result.risk_score <= 1
        assert result.decision is not None
        
    def test_ensemble_batch_score(self, mock_ensemble):
        """Test batch scoring."""
        X = np.random.randn(5, 17).astype(np.float32)
        
        results = mock_ensemble.score_batch(X)
        
        assert len(results) == 5
        for result in results:
            assert 0 <= result.risk_score <= 1
            
    def test_decision_thresholds(self, mock_ensemble):
        """Test decision thresholds."""
        from src.models.ensemble import Decision
        
        # Test approve threshold
        mock_ensemble.config.approve_threshold = 0.3
        mock_ensemble.config.review_threshold = 0.7
        mock_ensemble.config.decline_threshold = 0.9
        
        assert mock_ensemble._make_decision(0.1) == Decision.APPROVE
        assert mock_ensemble._make_decision(0.5) == Decision.STEP_UP
        assert mock_ensemble._make_decision(0.8) == Decision.REVIEW
        assert mock_ensemble._make_decision(0.95) == Decision.DECLINE


class TestFeatureStore:
    """Tests for Feature Store."""
    
    @pytest.fixture
    def mock_store(self):
        """Create mock feature store."""
        from src.features.feature_store import MockFeatureStore
        return MockFeatureStore()
    
    @pytest.mark.asyncio
    async def test_get_features(self, mock_store):
        """Test feature retrieval."""
        await mock_store.connect()
        
        features = await mock_store.get_features(
            card_id="card_001",
            merchant_id="merch_001",
            device_id="device_001",
            amount=100.0,
            timestamp=datetime.utcnow(),
        )
        
        assert features.card_id == "card_001"
        assert features.txn_count_1h >= 0
        
    def test_features_to_array(self):
        """Test feature conversion to array."""
        from src.features.feature_store import TransactionFeatures
        
        features = TransactionFeatures(
            card_id="test",
            timestamp=datetime.utcnow(),
            txn_count_1h=5,
            amount_sum_1h=500.0,
        )
        
        arr = features.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32


class TestScoringResult:
    """Tests for ScoringResult."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.models.ensemble import ScoringResult, Decision
        
        result = ScoringResult(
            transaction_id="txn_001",
            risk_score=0.45,
            decision=Decision.REVIEW,
        )
        
        d = result.to_dict()
        
        assert d["transaction_id"] == "txn_001"
        assert d["risk_score"] == 0.45
        assert d["decision"] == "review"
