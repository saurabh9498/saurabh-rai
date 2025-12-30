"""
Pytest Configuration and Fixtures

Provides shared fixtures for GPU ML Pipeline tests.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks integration tests")


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Generate a single random test image."""
    return np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch():
    """Generate a batch of random test images."""
    return np.random.randint(0, 256, size=(4, 480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch_large():
    """Generate a large batch of high-resolution images."""
    return np.random.randint(0, 256, size=(16, 1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def normalized_tensor():
    """Generate normalized NCHW tensor."""
    return np.random.randn(4, 3, 224, 224).astype(np.float32)


# =============================================================================
# Preprocessing Fixtures
# =============================================================================

@pytest.fixture
def preprocess_config():
    """Create default preprocessing config."""
    from src.preprocessing.pipeline import PreprocessConfig
    return PreprocessConfig(
        target_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        normalize=True
    )


@pytest.fixture
def cpu_preprocessor(preprocess_config):
    """Create CPU preprocessor."""
    from src.preprocessing.pipeline import GPUPreprocessor
    return GPUPreprocessor(config=preprocess_config, mode="cpu")


@pytest.fixture
def gpu_preprocessor(preprocess_config):
    """Create GPU preprocessor (if available)."""
    from src.preprocessing.pipeline import GPUPreprocessor
    try:
        import torch
        if torch.cuda.is_available():
            return GPUPreprocessor(config=preprocess_config, mode="gpu")
    except ImportError:
        pass
    pytest.skip("GPU not available")


# =============================================================================
# TensorRT Fixtures
# =============================================================================

@pytest.fixture
def mock_tensorrt_engine():
    """Mock TensorRT engine for testing."""
    engine = MagicMock()
    engine.num_io_tensors = 2
    engine.get_tensor_name = Mock(side_effect=["input", "output"])
    engine.get_tensor_shape = Mock(return_value=(1, 3, 224, 224))
    engine.get_tensor_dtype = Mock()
    engine.get_tensor_mode = Mock()
    engine.device_memory_size = 1024 * 1024  # 1MB
    return engine


@pytest.fixture
def build_config():
    """Create default TensorRT build config."""
    from src.tensorrt.builder import BuildConfig
    return BuildConfig(
        precision="fp16",
        max_batch_size=32,
        workspace_size_gb=2.0,
        optimization_level=3
    )


# =============================================================================
# Triton Fixtures
# =============================================================================

@pytest.fixture
def mock_triton_client():
    """Mock Triton client for testing."""
    client = MagicMock()
    client.is_server_ready.return_value = True
    client.is_model_ready.return_value = True
    client.get_model_metadata.return_value = {
        "name": "test_model",
        "inputs": [{"name": "input", "shape": [-1, 3, 224, 224]}],
        "outputs": [{"name": "output", "shape": [-1, 1000]}],
    }
    return client


# =============================================================================
# Pipeline Fixtures
# =============================================================================

@pytest.fixture
def pipeline_cpu(preprocess_config):
    """Create pipeline with CPU preprocessing."""
    from src.preprocessing.pipeline import Pipeline
    return Pipeline(
        preprocessing="cpu",
        model_path=None,
        config=preprocess_config
    )


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def model_dir(temp_dir):
    """Create temporary model directory structure."""
    models = temp_dir / "models"
    models.mkdir()
    engines = temp_dir / "engines"
    engines.mkdir()
    return {"models": models, "engines": engines}


# =============================================================================
# Benchmark Fixtures
# =============================================================================

@pytest.fixture
def benchmark_config():
    """Default benchmark configuration."""
    return {
        "warmup": 5,
        "iterations": 20,
        "batch_sizes": [1, 4, 8],
    }


# =============================================================================
# Skip Conditions
# =============================================================================

@pytest.fixture
def requires_gpu():
    """Skip if GPU not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.fixture
def requires_tensorrt():
    """Skip if TensorRT not available."""
    try:
        import tensorrt
    except ImportError:
        pytest.skip("TensorRT not installed")


@pytest.fixture
def requires_triton():
    """Skip if Triton client not available."""
    try:
        import tritonclient.grpc
    except ImportError:
        pytest.skip("Triton client not installed")
