"""
Pytest configuration and fixtures for Recommendation System tests.

This module provides shared fixtures for unit, integration, and load tests.
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Global test configuration."""
    return {
        "model": {
            "embedding_dim": 32,
            "num_users": 100,
            "num_items": 50,
            "hidden_dims": [64, 32],
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 2,
        },
        "serving": {
            "host": "localhost",
            "port": 8000,
            "top_k": 10,
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 15,  # Use separate DB for tests
        },
    }


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_users() -> pd.DataFrame:
    """Generate sample user data."""
    np.random.seed(42)
    n_users = 100

    return pd.DataFrame({
        "user_id": [f"user_{i:04d}" for i in range(n_users)],
        "age_bucket": np.random.randint(0, 6, n_users),
        "gender": np.random.randint(0, 3, n_users),
        "country": np.random.choice(["US", "GB", "DE", "FR", "JP"], n_users),
        "signup_date": [
            datetime(2024, 1, 1) + timedelta(days=int(d))
            for d in np.random.randint(0, 365, n_users)
        ],
        "lifetime_value": np.random.lognormal(4, 1, n_users),
        "activity_level": np.random.randint(1, 6, n_users),
    })


@pytest.fixture(scope="session")
def sample_items() -> pd.DataFrame:
    """Generate sample item data."""
    np.random.seed(42)
    n_items = 50

    categories = ["Electronics", "Fashion", "Home", "Sports", "Beauty"]

    return pd.DataFrame({
        "item_id": [f"item_{i:04d}" for i in range(n_items)],
        "category_l1": np.random.choice(categories, n_items),
        "brand": np.random.choice(["BrandA", "BrandB", "BrandC"], n_items),
        "price": np.random.lognormal(3, 1, n_items),
        "avg_rating": np.random.beta(5, 2, n_items) * 4 + 1,
        "review_count": np.random.randint(0, 1000, n_items),
        "popularity_score": np.random.uniform(0, 10, n_items),
    })


@pytest.fixture(scope="session")
def sample_interactions(sample_users: pd.DataFrame, sample_items: pd.DataFrame) -> pd.DataFrame:
    """Generate sample interaction data."""
    np.random.seed(42)
    n_interactions = 1000

    user_ids = sample_users["user_id"].tolist()
    item_ids = sample_items["item_id"].tolist()

    return pd.DataFrame({
        "user_id": np.random.choice(user_ids, n_interactions),
        "item_id": np.random.choice(item_ids, n_interactions),
        "event_type": np.random.choice(
            ["view", "click", "add_to_cart", "purchase"],
            n_interactions,
            p=[0.7, 0.2, 0.07, 0.03],
        ),
        "timestamp": [
            datetime(2024, 1, 1) + timedelta(hours=int(h))
            for h in np.random.randint(0, 8760, n_interactions)
        ],
        "label": np.random.choice([0, 1], n_interactions, p=[0.9, 0.1]),
    })


@pytest.fixture
def sample_batch(sample_interactions: pd.DataFrame, test_config: Dict) -> Dict[str, torch.Tensor]:
    """Create a sample training batch."""
    batch_size = test_config["training"]["batch_size"]
    num_users = test_config["model"]["num_users"]
    num_items = test_config["model"]["num_items"]

    return {
        "user_ids": torch.randint(0, num_users, (batch_size,)),
        "item_ids": torch.randint(0, num_items, (batch_size,)),
        "labels": torch.randint(0, 2, (batch_size,)).float(),
        "user_features": torch.randn(batch_size, 10),
        "item_features": torch.randn(batch_size, 15),
    }


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def two_tower_model(test_config: Dict, device: torch.device):
    """Create Two-Tower model for testing."""
    from models.two_tower import TwoTowerModel

    config = test_config["model"]
    model = TwoTowerModel(
        num_users=config["num_users"],
        num_items=config["num_items"],
        embedding_dim=config["embedding_dim"],
    )
    return model.to(device)


@pytest.fixture
def dlrm_model(test_config: Dict, device: torch.device):
    """Create DLRM model for testing."""
    from models.dlrm import DLRM

    model = DLRM(
        embedding_dim=test_config["model"]["embedding_dim"],
        num_dense_features=13,
        dense_arch_layer_sizes=[64, 32],
        over_arch_layer_sizes=[64, 32, 1],
    )
    return model.to(device)


@pytest.fixture
def mock_triton_client():
    """Mock Triton Inference Server client."""
    client = MagicMock()
    client.is_server_ready.return_value = True
    client.is_model_ready.return_value = True

    # Mock inference response
    mock_response = MagicMock()
    mock_response.as_numpy.return_value = np.random.rand(10, 1).astype(np.float32)
    client.infer.return_value = mock_response

    return client


# =============================================================================
# Service Fixtures
# =============================================================================


@pytest.fixture
def mock_redis():
    """Mock Redis client for feature store."""
    redis_mock = MagicMock()

    # Storage for mock data
    storage = {}

    def mock_get(key):
        return storage.get(key)

    def mock_set(key, value, ex=None):
        storage[key] = value
        return True

    def mock_mget(keys):
        return [storage.get(k) for k in keys]

    redis_mock.get = mock_get
    redis_mock.set = mock_set
    redis_mock.mget = mock_mget
    redis_mock.ping.return_value = True

    return redis_mock


@pytest.fixture
def mock_feature_store(mock_redis):
    """Mock feature store with Redis backend."""
    from features.feature_store import FeatureStore

    store = FeatureStore.__new__(FeatureStore)
    store.redis = mock_redis
    store.ttl = 3600

    return store


@pytest.fixture
async def async_client():
    """Async HTTP client for API testing."""
    from httpx import AsyncClient
    from serving.api import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# =============================================================================
# Temporary Files Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create temporary directory for model checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_parquet_files(
    temp_data_dir: Path,
    sample_users: pd.DataFrame,
    sample_items: pd.DataFrame,
    sample_interactions: pd.DataFrame,
) -> Dict[str, Path]:
    """Create temporary parquet files for testing."""
    users_path = temp_data_dir / "users.parquet"
    items_path = temp_data_dir / "items.parquet"
    interactions_path = temp_data_dir / "interactions.parquet"

    sample_users.to_parquet(users_path)
    sample_items.to_parquet(items_path)
    sample_interactions.to_parquet(interactions_path)

    return {
        "users": users_path,
        "items": items_path,
        "interactions": interactions_path,
    }


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
# Markers Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external deps)")
    config.addinivalue_line("markers", "integration: Integration tests (require services)")
    config.addinivalue_line("markers", "load: Load/performance tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "slow: Slow running tests")


# =============================================================================
# Skip Conditions
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on environment."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_integration = pytest.mark.skip(reason="Integration services not available")

    for item in items:
        # Skip GPU tests if no GPU
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)

        # Skip integration tests unless explicitly enabled
        if "integration" in item.keywords:
            if not os.environ.get("RUN_INTEGRATION_TESTS"):
                item.add_marker(skip_integration)
