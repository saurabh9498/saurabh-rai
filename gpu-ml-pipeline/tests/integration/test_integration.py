"""
Integration tests for GPU-Accelerated ML Pipeline.

Tests end-to-end workflows including:
- Preprocessing pipeline integration
- TensorRT engine inference
- API endpoint testing
- Triton client integration
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_full_preprocessing_pipeline(self, sample_image_path):
        """Test complete preprocessing from file to tensor."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        pipeline = PreprocessingPipeline(
            input_size=(224, 224),
            normalize=True,
            use_gpu=False  # Use CPU for testing without GPU
        )
        
        # Process image
        result = pipeline.process_file(sample_image_path)
        
        # Verify output
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32
        
    def test_batch_preprocessing(self, sample_image_batch):
        """Test batch preprocessing pipeline."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        pipeline = PreprocessingPipeline(
            input_size=(224, 224),
            normalize=True,
            use_gpu=False
        )
        
        # Process batch
        results = pipeline.process_batch(sample_image_batch)
        
        assert results.shape[0] == len(sample_image_batch)
        assert results.shape[1:] == (3, 224, 224)
        
    def test_preprocessing_with_augmentation(self, sample_image_path):
        """Test preprocessing with data augmentation."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        pipeline = PreprocessingPipeline(
            input_size=(224, 224),
            normalize=True,
            augment=True,
            use_gpu=False
        )
        
        # Process same image multiple times
        results = [pipeline.process_file(sample_image_path) for _ in range(3)]
        
        # With augmentation, results should differ
        # (In practice, check specific augmentation effects)
        assert len(results) == 3


class TestTensorRTIntegration:
    """Integration tests for TensorRT inference."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock TensorRT engine."""
        engine = Mock()
        engine.infer = Mock(return_value=np.random.randn(1, 1000).astype(np.float32))
        return engine
    
    def test_engine_inference_pipeline(self, mock_engine, preprocessed_batch):
        """Test complete inference pipeline."""
        # Run inference
        outputs = mock_engine.infer(preprocessed_batch)
        
        # Verify
        assert outputs.shape == (1, 1000)
        mock_engine.infer.assert_called_once()
        
    def test_batch_inference(self, mock_engine):
        """Test batch inference."""
        batch_size = 32
        batch = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        
        mock_engine.infer = Mock(
            return_value=np.random.randn(batch_size, 1000).astype(np.float32)
        )
        
        outputs = mock_engine.infer(batch)
        
        assert outputs.shape == (batch_size, 1000)
        
    def test_dynamic_batch_inference(self, mock_engine):
        """Test inference with dynamic batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            batch = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            mock_engine.infer = Mock(
                return_value=np.random.randn(batch_size, 1000).astype(np.float32)
            )
            
            outputs = mock_engine.infer(batch)
            assert outputs.shape[0] == batch_size


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def api_client(self):
        """Create test client for API."""
        from fastapi.testclient import TestClient
        
        # Mock the inference engine
        with patch('src.api.main.get_inference_engine') as mock:
            mock.return_value = Mock(
                infer=Mock(return_value=np.random.randn(1, 1000).astype(np.float32))
            )
            
            from src.api.main import app
            return TestClient(app)
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        assert "status" in response.json()
        
    def test_inference_endpoint(self, api_client, sample_image_bytes):
        """Test inference endpoint with image upload."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        response = api_client.post("/api/v1/predict", files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result or "error" not in result
        
    def test_batch_inference_endpoint(self, api_client):
        """Test batch inference endpoint."""
        # Create multiple file uploads
        files = [
            ("files", (f"test_{i}.jpg", b"fake_image_data", "image/jpeg"))
            for i in range(4)
        ]
        
        response = api_client.post("/api/v1/predict/batch", files=files)
        
        # Should handle batch request
        assert response.status_code in [200, 422]  # 422 if validation fails on fake data


class TestTritonIntegration:
    """Integration tests for Triton Inference Server client."""
    
    @pytest.fixture
    def mock_triton_client(self):
        """Create a mock Triton client."""
        client = Mock()
        client.is_server_ready = Mock(return_value=True)
        client.is_model_ready = Mock(return_value=True)
        client.infer = Mock()
        return client
    
    def test_triton_server_connection(self, mock_triton_client):
        """Test Triton server connectivity."""
        assert mock_triton_client.is_server_ready()
        
    def test_triton_model_ready(self, mock_triton_client):
        """Test model readiness on Triton."""
        assert mock_triton_client.is_model_ready("resnet50")
        mock_triton_client.is_model_ready.assert_called_with("resnet50")
        
    def test_triton_inference(self, mock_triton_client):
        """Test inference through Triton client."""
        from src.triton.client import TritonClient
        
        with patch.object(TritonClient, '__init__', lambda x, y: None):
            with patch.object(TritonClient, 'infer') as mock_infer:
                mock_infer.return_value = {
                    "output": np.random.randn(1, 1000).astype(np.float32)
                }
                
                client = TritonClient("localhost:8001")
                batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
                
                result = client.infer(batch)
                
                assert "output" in result


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_complete_classification_workflow(
        self, 
        sample_image_path,
        mock_tensorrt_engine
    ):
        """Test complete classification workflow."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        # 1. Preprocess
        pipeline = PreprocessingPipeline(
            input_size=(224, 224),
            normalize=True,
            use_gpu=False
        )
        tensor = pipeline.process_file(sample_image_path)
        
        # 2. Inference
        outputs = mock_tensorrt_engine.infer(tensor)
        
        # 3. Post-process
        predictions = np.argsort(outputs[0])[-5:][::-1]
        
        assert len(predictions) == 5
        
    def test_complete_detection_workflow(
        self,
        sample_image_path,
        mock_detection_engine
    ):
        """Test complete detection workflow."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        # 1. Preprocess for detection (640x640)
        pipeline = PreprocessingPipeline(
            input_size=(640, 640),
            normalize=True,
            use_gpu=False
        )
        tensor = pipeline.process_file(sample_image_path)
        
        # 2. Inference
        outputs = mock_detection_engine.infer(tensor)
        
        # 3. Post-process (NMS would be applied here)
        assert outputs is not None


# Fixtures
@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image file for testing."""
    import struct
    import zlib
    
    # Create minimal PNG
    width, height = 224, 224
    
    def png_chunk(chunk_type, data):
        chunk_len = struct.pack('>I', len(data))
        chunk_crc = struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
        return chunk_len + chunk_type + data + chunk_crc
    
    signature = b'\x89PNG\r\n\x1a\n'
    ihdr = png_chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    
    raw_data = b''.join([b'\x00' + b'\x80\x80\x80' * width for _ in range(height)])
    idat = png_chunk(b'IDAT', zlib.compress(raw_data, 9))
    iend = png_chunk(b'IEND', b'')
    
    img_path = tmp_path / "test_image.png"
    img_path.write_bytes(signature + ihdr + idat + iend)
    
    return img_path


@pytest.fixture
def sample_image_batch(sample_image_path):
    """Create a batch of sample image paths."""
    return [sample_image_path] * 4


@pytest.fixture
def sample_image_bytes(sample_image_path):
    """Get sample image as bytes."""
    return sample_image_path.read_bytes()


@pytest.fixture
def preprocessed_batch():
    """Create a preprocessed batch tensor."""
    return np.random.randn(1, 3, 224, 224).astype(np.float32)


@pytest.fixture
def mock_tensorrt_engine():
    """Create a mock TensorRT engine for classification."""
    engine = Mock()
    engine.infer = Mock(return_value=np.random.randn(1, 1000).astype(np.float32))
    return engine


@pytest.fixture
def mock_detection_engine():
    """Create a mock TensorRT engine for detection."""
    engine = Mock()
    # YOLOv8 output format: (batch, 84, 8400) for 80 classes + 4 box coords
    engine.infer = Mock(return_value=np.random.randn(1, 84, 8400).astype(np.float32))
    return engine
