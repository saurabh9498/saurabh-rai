"""
Pytest Configuration and Fixtures

Shared fixtures for fraud detection tests.
"""

import asyncio
import numpy as np
import pytest
from datetime import datetime, timedelta
from typing import Dict, List


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


# =============================================================================
# Event Loop Fixture
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample feature vector."""
    return np.array([
        3,      # txn_count_1h
        8,      # txn_count_6h
        15,     # txn_count_24h
        45,     # txn_count_7d
        250.0,  # amount_sum_1h
        1200.0, # amount_sum_24h
        85.0,   # amount_avg_30d
        45.0,   # amount_std_30d
        3600,   # time_since_last_txn
        5,      # unique_merchants_24h
        2,      # unique_channels_24h
        0,      # is_first_transaction
        0,      # is_new_merchant
        0,      # is_new_device
        0.5,    # deviation_from_avg
        0.1,    # merchant_risk_score
        0.05,   # device_risk_score
    ], dtype=np.float32)


@pytest.fixture
def sample_batch() -> np.ndarray:
    """Generate batch of feature vectors."""
    np.random.seed(42)
    return np.random.randn(100, 17).astype(np.float32)


@pytest.fixture
def sample_transaction() -> Dict:
    """Generate sample transaction."""
    return {
        "transaction_id": "txn_test_001",
        "card_id": "card_abc123",
        "amount": 150.00,
        "merchant_id": "merch_xyz",
        "merchant_category": "retail",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "channel": "online",
        "ip_address": "192.168.1.100",
        "device_id": "device_123",
    }


@pytest.fixture
def fraud_transaction() -> Dict:
    """Generate likely fraudulent transaction."""
    return {
        "transaction_id": "txn_fraud_001",
        "card_id": "card_suspicious",
        "amount": 9999.00,
        "merchant_id": "merch_new",
        "merchant_category": "jewelry",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "channel": "online",
        "ip_address": "10.0.0.1",
        "device_id": "device_new",
    }


@pytest.fixture
def transaction_batch() -> List[Dict]:
    """Generate batch of transactions."""
    base_time = datetime.utcnow()
    
    return [
        {
            "transaction_id": f"txn_{i:03d}",
            "card_id": f"card_{i % 10:03d}",
            "amount": float(np.random.uniform(10, 500)),
            "merchant_id": f"merch_{i % 20:03d}",
            "merchant_category": np.random.choice(["retail", "food", "travel", "entertainment"]),
            "timestamp": (base_time - timedelta(minutes=i)).isoformat() + "Z",
            "channel": np.random.choice(["online", "pos", "mobile"]),
            "ip_address": f"192.168.1.{i % 256}",
            "device_id": f"device_{i % 5:03d}",
        }
        for i in range(20)
    ]


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def mock_ensemble():
    """Create mock ensemble for testing."""
    from src.models.ensemble import FraudEnsemble
    from src.models.xgboost_model import MockXGBoostModel
    from src.models.neural_net import MockNeuralNetModel
    from src.models.isolation_forest import MockIsolationForestModel
    
    ensemble = FraudEnsemble()
    ensemble.xgboost = MockXGBoostModel()
    ensemble.neural_net = MockNeuralNetModel()
    ensemble.isolation_forest = MockIsolationForestModel()
    ensemble._is_trained = True
    
    return ensemble


@pytest.fixture
def ensemble_config():
    """Create ensemble configuration."""
    from src.models.ensemble import EnsembleConfig
    
    return EnsembleConfig(
        xgboost_weight=0.45,
        neural_net_weight=0.35,
        isolation_forest_weight=0.20,
        approve_threshold=0.3,
        review_threshold=0.7,
        decline_threshold=0.9,
    )


# =============================================================================
# Feature Store Fixtures
# =============================================================================

@pytest.fixture
def mock_feature_store():
    """Create mock feature store."""
    from src.features.feature_store import MockFeatureStore
    return MockFeatureStore()


@pytest.fixture
def feature_config():
    """Create feature configuration."""
    from src.features.feature_store import FeatureConfig
    
    return FeatureConfig(
        velocity_windows=["1h", "6h", "24h", "7d"],
        aggregation_windows=["1h", "24h", "7d", "30d"],
        ttl_days=90,
    )


# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture
def api_client():
    """Create test API client."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    return TestClient(app)


# =============================================================================
# Streaming Fixtures
# =============================================================================

@pytest.fixture
def kafka_config():
    """Create Kafka configuration."""
    from src.streaming.stream_processor import KafkaConfig
    
    return KafkaConfig(
        bootstrap_servers="localhost:9092",
        input_topic="test-transactions",
        output_topic="test-fraud-scores",
        consumer_group="test-fraud-detector",
    )


# =============================================================================
# Monitoring Fixtures
# =============================================================================

@pytest.fixture
def drift_detector():
    """Create drift detector."""
    from src.monitoring.metrics import DriftDetector, DriftConfig
    
    config = DriftConfig(
        window_size=100,
        reference_size=1000,
        threshold=0.1,
    )
    
    return DriftDetector(config)


@pytest.fixture
def performance_monitor():
    """Create performance monitor."""
    from src.monitoring.metrics import PerformanceMonitor
    return PerformanceMonitor(window_size=100)


# =============================================================================
# Training Data Fixtures
# =============================================================================

@pytest.fixture
def training_data():
    """Generate synthetic training data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 17
    
    # Generate features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate labels (1% fraud rate)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    y[fraud_indices] = 1
    
    return X, y


@pytest.fixture
def validation_data():
    """Generate synthetic validation data."""
    np.random.seed(123)
    n_samples = 200
    n_features = 17
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    y[fraud_indices] = 1
    
    return X, y
