"""
Unit tests for GPU preprocessing pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestPreprocessConfig:
    """Tests for PreprocessConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.preprocessing.pipeline import PreprocessConfig
        
        config = PreprocessConfig()
        
        assert config.target_size == (224, 224)
        assert config.mean == (0.485, 0.456, 0.406)
        assert config.std == (0.229, 0.224, 0.225)
        assert config.normalize is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        from src.preprocessing.pipeline import PreprocessConfig
        
        config = PreprocessConfig(
            target_size=(256, 256),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            output_format="NHWC"
        )
        
        assert config.target_size == (256, 256)
        assert config.output_format == "NHWC"


class TestGPUPreprocessor:
    """Tests for GPUPreprocessor."""
    
    @pytest.fixture
    def sample_images(self):
        """Generate sample test images."""
        return np.random.randint(0, 256, size=(4, 480, 640, 3), dtype=np.uint8)
    
    def test_cpu_preprocessing(self, sample_images):
        """Test CPU preprocessing fallback."""
        from src.preprocessing.pipeline import GPUPreprocessor
        
        preprocessor = GPUPreprocessor(mode="cpu")
        output = preprocessor.process(sample_images, return_numpy=True)
        
        # Check output shape (NCHW format)
        assert output.shape == (4, 3, 224, 224)
        
        # Check output dtype
        assert output.dtype == np.float32
    
    def test_output_normalization(self, sample_images):
        """Test that output is properly normalized."""
        from src.preprocessing.pipeline import GPUPreprocessor, PreprocessConfig
        
        config = PreprocessConfig(normalize=True)
        preprocessor = GPUPreprocessor(config=config, mode="cpu")
        
        output = preprocessor.process(sample_images, return_numpy=True)
        
        # Normalized ImageNet data typically ranges from -2 to 3
        assert output.min() > -5
        assert output.max() < 5
    
    def test_single_image(self):
        """Test preprocessing single image."""
        from src.preprocessing.pipeline import GPUPreprocessor
        
        image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        
        preprocessor = GPUPreprocessor(mode="cpu")
        output = preprocessor.process(image, return_numpy=True)
        
        # Should add batch dimension
        assert output.shape == (1, 3, 224, 224)
    
    def test_list_of_images(self):
        """Test preprocessing list of images."""
        from src.preprocessing.pipeline import GPUPreprocessor
        
        images = [
            np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        preprocessor = GPUPreprocessor(mode="cpu")
        output = preprocessor.process(images, return_numpy=True)
        
        assert output.shape == (3, 3, 224, 224)
    
    def test_benchmark(self, sample_images):
        """Test benchmarking functionality."""
        from src.preprocessing.pipeline import GPUPreprocessor
        
        preprocessor = GPUPreprocessor(mode="cpu")
        results = preprocessor.benchmark(num_images=10, batch_size=2)
        
        assert "mean_ms" in results
        assert "throughput" in results
        assert results["mean_ms"] > 0


class TestPostprocessing:
    """Tests for postprocessing functions."""
    
    def test_softmax(self):
        """Test softmax function."""
        from src.preprocessing.pipeline import softmax
        
        logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        probs = softmax(logits)
        
        # Check probabilities sum to 1
        np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0], rtol=1e-5)
        
        # Check max probability is for highest logit
        assert probs[0].argmax() == 2
    
    def test_top_k(self):
        """Test top-k function."""
        from src.preprocessing.pipeline import top_k
        
        predictions = np.array([
            [0.1, 0.3, 0.05, 0.5, 0.05]
        ])
        
        results = top_k(predictions, k=3)
        
        assert len(results) == 1
        assert len(results[0]) == 3
        assert results[0][0][0] == 3  # Index of highest value


class TestPipeline:
    """Tests for complete Pipeline."""
    
    @pytest.fixture
    def sample_images(self):
        return np.random.randint(0, 256, size=(2, 480, 640, 3), dtype=np.uint8)
    
    def test_pipeline_without_model(self, sample_images):
        """Test pipeline runs preprocessing only when no model."""
        from src.preprocessing.pipeline import Pipeline
        
        pipeline = Pipeline(preprocessing="cpu")
        output = pipeline.run(sample_images)
        
        # Without model, returns preprocessed tensor
        assert output.shape == (2, 3, 224, 224)
    
    def test_pipeline_timing(self, sample_images):
        """Test pipeline timing metrics."""
        from src.preprocessing.pipeline import Pipeline
        
        pipeline = Pipeline(preprocessing="cpu")
        pipeline.run(sample_images)
        
        timing = pipeline.get_timing()
        
        assert "preprocess_ms" in timing
        assert "total_ms" in timing
        assert timing["preprocess_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
