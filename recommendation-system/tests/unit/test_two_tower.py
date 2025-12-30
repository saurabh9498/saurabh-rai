"""
Unit tests for Two-Tower retrieval model.

Tests cover model architecture, forward pass, training loop,
embedding generation, and FAISS index integration.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import model components
import sys
sys.path.insert(0, '/home/claude/github-portfolio/projects/recommendation-system')

from src.models.two_tower import (
    TwoTowerConfig,
    MLP,
    HistoryEncoder,
    UserTower,
    ItemTower,
    TwoTowerModel,
    TwoTowerTrainer,
    build_faiss_index,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create a test configuration with smaller dimensions."""
    return TwoTowerConfig(
        # Embedding dimensions
        user_embedding_dim=32,
        item_embedding_dim=32,
        output_embedding_dim=64,
        
        # Feature counts (smaller for testing)
        num_user_categorical_features=5,
        num_item_categorical_features=5,
        user_categorical_cardinalities=[100, 50, 20, 10, 5],
        item_categorical_cardinalities=[1000, 100, 50, 20, 10],
        num_user_dense_features=8,
        num_item_dense_features=8,
        
        # History settings
        max_history_length=10,
        history_embedding_dim=32,
        num_attention_heads=2,
        
        # Architecture
        user_mlp_dims=[64, 64],
        item_mlp_dims=[64, 64],
        dropout_rate=0.1,
        
        # Training
        batch_size=16,
        learning_rate=0.001,
        temperature=0.05,
        num_negatives=4,
    )


@pytest.fixture
def batch_data(config):
    """Create a sample batch of training data."""
    batch_size = 16
    
    return {
        # User features
        'user_categorical': torch.randint(
            0, 10, (batch_size, config.num_user_categorical_features)
        ),
        'user_dense': torch.randn(batch_size, config.num_user_dense_features),
        'user_history': torch.randint(
            0, 1000, (batch_size, config.max_history_length)
        ),
        'history_mask': torch.ones(batch_size, config.max_history_length),
        
        # Item features
        'item_categorical': torch.randint(
            0, 10, (batch_size, config.num_item_categorical_features)
        ),
        'item_dense': torch.randn(batch_size, config.num_item_dense_features),
        
        # Labels (for training)
        'labels': torch.ones(batch_size),
    }


@pytest.fixture
def model(config):
    """Create a Two-Tower model instance."""
    return TwoTowerModel(config)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestTwoTowerConfig:
    """Tests for model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TwoTowerConfig()
        
        assert config.output_embedding_dim == 128
        assert config.temperature == 0.05
        assert config.dropout_rate == 0.1
        
    def test_config_validation(self):
        """Test configuration validation."""
        config = TwoTowerConfig(
            user_categorical_cardinalities=[100, 50],
            num_user_categorical_features=2,
        )
        
        assert len(config.user_categorical_cardinalities) == config.num_user_categorical_features
        
    def test_config_to_dict(self, config):
        """Test configuration serialization."""
        config_dict = asdict(config)
        
        assert 'output_embedding_dim' in config_dict
        assert config_dict['batch_size'] == 16


# =============================================================================
# MLP Tests
# =============================================================================

class TestMLP:
    """Tests for Multi-Layer Perceptron component."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=32,
            dropout_rate=0.1,
        )
        
        x = torch.randn(16, 64)
        output = mlp(x)
        
        assert output.shape == (16, 32)
        
    def test_mlp_no_hidden_layers(self):
        """Test MLP with direct input-to-output."""
        mlp = MLP(
            input_dim=64,
            hidden_dims=[],
            output_dim=32,
        )
        
        x = torch.randn(8, 64)
        output = mlp(x)
        
        assert output.shape == (8, 32)
        
    def test_mlp_dropout_in_training(self):
        """Test that dropout is applied during training."""
        mlp = MLP(
            input_dim=32,
            hidden_dims=[64],
            output_dim=16,
            dropout_rate=0.5,
        )
        
        mlp.train()
        x = torch.ones(100, 32)
        
        # Run multiple times and check for variation (due to dropout)
        outputs = [mlp(x).sum().item() for _ in range(5)]
        
        # With 50% dropout, outputs should vary
        assert len(set(outputs)) > 1


# =============================================================================
# History Encoder Tests
# =============================================================================

class TestHistoryEncoder:
    """Tests for attention-based history encoder."""
    
    def test_history_encoder_forward(self, config):
        """Test history encoder forward pass."""
        encoder = HistoryEncoder(
            num_items=1000,
            embedding_dim=32,
            output_dim=64,
            max_history_length=10,
            num_attention_heads=2,
        )
        
        history_ids = torch.randint(0, 1000, (8, 10))
        mask = torch.ones(8, 10)
        
        output = encoder(history_ids, mask)
        
        assert output.shape == (8, 64)
        
    def test_history_encoder_with_padding(self, config):
        """Test history encoder handles padding correctly."""
        encoder = HistoryEncoder(
            num_items=1000,
            embedding_dim=32,
            output_dim=64,
            max_history_length=10,
            num_attention_heads=2,
        )
        
        # Create batch with varying history lengths
        history_ids = torch.randint(0, 1000, (4, 10))
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 5 items
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 3 items
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10 items
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 item
        ], dtype=torch.float)
        
        output = encoder(history_ids, mask)
        
        assert output.shape == (4, 64)
        assert not torch.isnan(output).any()


# =============================================================================
# Tower Tests
# =============================================================================

class TestUserTower:
    """Tests for User Tower."""
    
    def test_user_tower_forward(self, config, batch_data):
        """Test user tower forward pass."""
        tower = UserTower(config)
        
        embeddings = tower(
            batch_data['user_categorical'],
            batch_data['user_dense'],
            batch_data['user_history'],
            batch_data['history_mask'],
        )
        
        assert embeddings.shape == (16, config.output_embedding_dim)
        
    def test_user_tower_normalized(self, config, batch_data):
        """Test that user embeddings are L2 normalized."""
        tower = UserTower(config)
        
        embeddings = tower(
            batch_data['user_categorical'],
            batch_data['user_dense'],
            batch_data['user_history'],
            batch_data['history_mask'],
        )
        
        # Check L2 normalization
        norms = torch.norm(embeddings, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones(16), rtol=1e-4, atol=1e-4)


class TestItemTower:
    """Tests for Item Tower."""
    
    def test_item_tower_forward(self, config, batch_data):
        """Test item tower forward pass."""
        tower = ItemTower(config)
        
        embeddings = tower(
            batch_data['item_categorical'],
            batch_data['item_dense'],
        )
        
        assert embeddings.shape == (16, config.output_embedding_dim)
        
    def test_item_tower_normalized(self, config, batch_data):
        """Test that item embeddings are L2 normalized."""
        tower = ItemTower(config)
        
        embeddings = tower(
            batch_data['item_categorical'],
            batch_data['item_dense'],
        )
        
        norms = torch.norm(embeddings, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones(16), rtol=1e-4, atol=1e-4)


# =============================================================================
# Full Model Tests
# =============================================================================

class TestTwoTowerModel:
    """Tests for complete Two-Tower model."""
    
    def test_model_forward(self, model, batch_data):
        """Test model forward pass."""
        user_emb, item_emb = model(
            user_categorical=batch_data['user_categorical'],
            user_dense=batch_data['user_dense'],
            user_history=batch_data['user_history'],
            history_mask=batch_data['history_mask'],
            item_categorical=batch_data['item_categorical'],
            item_dense=batch_data['item_dense'],
        )
        
        assert user_emb.shape == (16, 64)
        assert item_emb.shape == (16, 64)
        
    def test_model_compute_loss(self, model, batch_data):
        """Test loss computation with in-batch negatives."""
        loss = model.compute_loss(
            user_categorical=batch_data['user_categorical'],
            user_dense=batch_data['user_dense'],
            user_history=batch_data['user_history'],
            history_mask=batch_data['history_mask'],
            item_categorical=batch_data['item_categorical'],
            item_dense=batch_data['item_dense'],
        )
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive
        assert not torch.isnan(loss)
        
    def test_model_encode_users(self, model, batch_data):
        """Test user encoding for inference."""
        model.eval()
        
        with torch.no_grad():
            user_emb = model.encode_users(
                batch_data['user_categorical'],
                batch_data['user_dense'],
                batch_data['user_history'],
                batch_data['history_mask'],
            )
        
        assert user_emb.shape == (16, 64)
        
    def test_model_encode_items(self, model, batch_data):
        """Test item encoding for index building."""
        model.eval()
        
        with torch.no_grad():
            item_emb = model.encode_items(
                batch_data['item_categorical'],
                batch_data['item_dense'],
            )
        
        assert item_emb.shape == (16, 64)
        
    def test_model_similarity_scores(self, model, batch_data):
        """Test similarity computation."""
        model.eval()
        
        with torch.no_grad():
            user_emb = model.encode_users(
                batch_data['user_categorical'],
                batch_data['user_dense'],
                batch_data['user_history'],
                batch_data['history_mask'],
            )
            item_emb = model.encode_items(
                batch_data['item_categorical'],
                batch_data['item_dense'],
            )
            
            # Compute similarity
            scores = torch.matmul(user_emb, item_emb.T)
        
        assert scores.shape == (16, 16)
        # Scores should be in [-1, 1] for normalized embeddings
        assert scores.min() >= -1.0 - 1e-4
        assert scores.max() <= 1.0 + 1e-4


# =============================================================================
# Training Tests
# =============================================================================

class TestTwoTowerTrainer:
    """Tests for model trainer."""
    
    def test_trainer_initialization(self, config):
        """Test trainer initialization."""
        model = TwoTowerModel(config)
        trainer = TwoTowerTrainer(
            model=model,
            config=config,
            device='cpu',
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        
    def test_trainer_train_step(self, config, batch_data):
        """Test single training step."""
        model = TwoTowerModel(config)
        trainer = TwoTowerTrainer(
            model=model,
            config=config,
            device='cpu',
        )
        
        loss = trainer.train_step(batch_data)
        
        assert isinstance(loss, float)
        assert loss > 0
        
    @pytest.mark.parametrize("num_batches", [1, 3, 5])
    def test_trainer_multiple_steps(self, config, batch_data, num_batches):
        """Test multiple training steps reduce loss."""
        model = TwoTowerModel(config)
        trainer = TwoTowerTrainer(
            model=model,
            config=config,
            device='cpu',
        )
        
        losses = []
        for _ in range(num_batches):
            loss = trainer.train_step(batch_data)
            losses.append(loss)
        
        # Loss should generally decrease (or at least not explode)
        assert all(l < 100 for l in losses)


# =============================================================================
# FAISS Index Tests
# =============================================================================

class TestFAISSIndex:
    """Tests for FAISS index building."""
    
    def test_build_flat_index(self):
        """Test building a flat (exact) index."""
        embeddings = np.random.randn(1000, 64).astype(np.float32)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        index = build_faiss_index(
            embeddings=embeddings,
            index_type='flat',
            metric='inner_product',
        )
        
        assert index is not None
        assert index.ntotal == 1000
        
    def test_index_search(self):
        """Test index search returns correct results."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        index = build_faiss_index(embeddings, index_type='flat')
        
        # Search with the first embedding (should return itself as top result)
        query = embeddings[0:1]
        distances, indices = index.search(query, k=5)
        
        assert indices.shape == (1, 5)
        assert indices[0, 0] == 0  # First result should be the query itself
        
    def test_index_with_ivf(self):
        """Test building IVF index for approximate search."""
        embeddings = np.random.randn(10000, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        index = build_faiss_index(
            embeddings=embeddings,
            index_type='ivf',
            nlist=100,
            nprobe=10,
        )
        
        assert index is not None
        assert index.ntotal == 10000


# =============================================================================
# Integration Tests
# =============================================================================

class TestModelIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_training_and_inference_pipeline(self, config):
        """Test complete training and inference flow."""
        # Create model
        model = TwoTowerModel(config)
        
        # Generate synthetic training data
        train_data = {
            'user_categorical': torch.randint(0, 10, (100, config.num_user_categorical_features)),
            'user_dense': torch.randn(100, config.num_user_dense_features),
            'user_history': torch.randint(0, 1000, (100, config.max_history_length)),
            'history_mask': torch.ones(100, config.max_history_length),
            'item_categorical': torch.randint(0, 10, (100, config.num_item_categorical_features)),
            'item_dense': torch.randn(100, config.num_item_dense_features),
        }
        
        # Train for a few steps
        trainer = TwoTowerTrainer(model, config, device='cpu')
        
        for _ in range(3):
            trainer.train_step(train_data)
        
        # Generate item embeddings
        model.eval()
        with torch.no_grad():
            item_embeddings = model.encode_items(
                train_data['item_categorical'],
                train_data['item_dense'],
            ).numpy()
        
        # Build index
        index = build_faiss_index(item_embeddings, index_type='flat')
        
        # Query with a user
        with torch.no_grad():
            user_embedding = model.encode_users(
                train_data['user_categorical'][:1],
                train_data['user_dense'][:1],
                train_data['user_history'][:1],
                train_data['history_mask'][:1],
            ).numpy()
        
        # Search
        distances, indices = index.search(user_embedding, k=10)
        
        assert indices.shape == (1, 10)
        assert all(0 <= idx < 100 for idx in indices[0])


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_history(self, config):
        """Test handling of users with no history."""
        model = TwoTowerModel(config)
        
        # User with all-zero mask (no history)
        user_categorical = torch.randint(0, 10, (1, config.num_user_categorical_features))
        user_dense = torch.randn(1, config.num_user_dense_features)
        user_history = torch.zeros(1, config.max_history_length, dtype=torch.long)
        history_mask = torch.zeros(1, config.max_history_length)
        
        with torch.no_grad():
            embedding = model.encode_users(
                user_categorical, user_dense, user_history, history_mask
            )
        
        assert embedding.shape == (1, config.output_embedding_dim)
        assert not torch.isnan(embedding).any()
        
    def test_single_item_batch(self, config):
        """Test model with batch size of 1."""
        model = TwoTowerModel(config)
        
        item_categorical = torch.randint(0, 10, (1, config.num_item_categorical_features))
        item_dense = torch.randn(1, config.num_item_dense_features)
        
        with torch.no_grad():
            embedding = model.encode_items(item_categorical, item_dense)
        
        assert embedding.shape == (1, config.output_embedding_dim)
        
    def test_large_batch(self, config):
        """Test model with large batch size."""
        model = TwoTowerModel(config)
        
        batch_size = 512
        item_categorical = torch.randint(0, 10, (batch_size, config.num_item_categorical_features))
        item_dense = torch.randn(batch_size, config.num_item_dense_features)
        
        with torch.no_grad():
            embedding = model.encode_items(item_categorical, item_dense)
        
        assert embedding.shape == (batch_size, config.output_embedding_dim)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
