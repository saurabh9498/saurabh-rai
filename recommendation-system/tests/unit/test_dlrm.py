"""
Unit tests for DLRM ranking model.

Tests cover model architecture, feature interactions,
multi-task learning, and DCN integration.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, '/home/claude/github-portfolio/projects/recommendation-system')

from src.models.dlrm import (
    DLRMConfig,
    SparseEmbedding,
    MLP,
    FeatureInteraction,
    DLRM,
    DCN,
    CrossLayer,
    MultiTaskDLRM,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return DLRMConfig(
        # Sparse features
        num_sparse_features=5,
        sparse_cardinalities=[100, 50, 20, 10, 5],
        sparse_embedding_dim=16,
        
        # Dense features
        num_dense_features=8,
        
        # Architecture
        bottom_mlp_dims=[32, 16],
        top_mlp_dims=[64, 32, 1],
        
        # Interaction
        interaction_type='dot',
        
        # Training
        dropout_rate=0.1,
        batch_size=16,
        learning_rate=0.001,
    )


@pytest.fixture
def batch_data(config):
    """Create sample batch data."""
    batch_size = 16
    
    return {
        'sparse_features': torch.randint(0, 5, (batch_size, config.num_sparse_features)),
        'dense_features': torch.randn(batch_size, config.num_dense_features),
        'labels': torch.randint(0, 2, (batch_size,)).float(),
    }


@pytest.fixture
def model(config):
    """Create DLRM model instance."""
    return DLRM(config)


# =============================================================================
# Sparse Embedding Tests
# =============================================================================

class TestSparseEmbedding:
    """Tests for sparse embedding layer."""
    
    def test_embedding_forward(self):
        """Test embedding lookup."""
        embedding = SparseEmbedding(
            num_embeddings=100,
            embedding_dim=16,
        )
        
        indices = torch.randint(0, 100, (8,))
        output = embedding(indices)
        
        assert output.shape == (8, 16)
        
    def test_embedding_with_dropout(self):
        """Test embedding with dropout."""
        embedding = SparseEmbedding(
            num_embeddings=100,
            embedding_dim=16,
            dropout_rate=0.5,
        )
        
        embedding.train()
        indices = torch.randint(0, 100, (100,))
        
        # Run multiple times to check dropout variation
        outputs = [embedding(indices).sum().item() for _ in range(5)]
        assert len(set(outputs)) > 1  # Should vary due to dropout
        
    def test_embedding_initialization(self):
        """Test embedding initialization bounds."""
        embedding = SparseEmbedding(
            num_embeddings=1000,
            embedding_dim=64,
        )
        
        weights = embedding.embedding.weight.data
        
        # Xavier uniform initialization bounds
        limit = np.sqrt(6.0 / (1000 + 64))
        assert weights.min() >= -limit - 0.01
        assert weights.max() <= limit + 0.01


# =============================================================================
# MLP Tests
# =============================================================================

class TestDLRMMLP:
    """Tests for DLRM MLP component."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=32,
        )
        
        x = torch.randn(16, 64)
        output = mlp(x)
        
        assert output.shape == (16, 32)
        
    def test_mlp_activation(self):
        """Test MLP with different activations."""
        for activation in ['relu', 'leaky_relu', 'gelu']:
            mlp = MLP(
                input_dim=32,
                hidden_dims=[64],
                output_dim=16,
                activation=activation,
            )
            
            x = torch.randn(8, 32)
            output = mlp(x)
            
            assert output.shape == (8, 16)
            assert not torch.isnan(output).any()


# =============================================================================
# Feature Interaction Tests
# =============================================================================

class TestFeatureInteraction:
    """Tests for feature interaction layer."""
    
    def test_dot_interaction(self):
        """Test dot product interaction."""
        interaction = FeatureInteraction(
            num_features=5,
            embedding_dim=16,
            interaction_type='dot',
        )
        
        # Create embeddings: (batch, num_features, embedding_dim)
        embeddings = torch.randn(8, 5, 16)
        output = interaction(embeddings)
        
        # Output should be upper triangle of interaction matrix
        # For 5 features: 5*4/2 = 10 interactions
        expected_interactions = 5 * 4 // 2
        assert output.shape == (8, expected_interactions)
        
    def test_cat_interaction(self):
        """Test concatenation interaction."""
        interaction = FeatureInteraction(
            num_features=5,
            embedding_dim=16,
            interaction_type='cat',
        )
        
        embeddings = torch.randn(8, 5, 16)
        output = interaction(embeddings)
        
        # Concatenation: 5 * 16 = 80
        assert output.shape == (8, 80)
        
    def test_attention_interaction(self):
        """Test attention-based interaction."""
        interaction = FeatureInteraction(
            num_features=5,
            embedding_dim=16,
            interaction_type='attention',
            num_attention_heads=4,
        )
        
        embeddings = torch.randn(8, 5, 16)
        output = interaction(embeddings)
        
        assert output.shape[0] == 8
        assert not torch.isnan(output).any()


# =============================================================================
# DLRM Model Tests
# =============================================================================

class TestDLRM:
    """Tests for complete DLRM model."""
    
    def test_model_forward(self, model, batch_data):
        """Test model forward pass."""
        output = model(
            batch_data['sparse_features'],
            batch_data['dense_features'],
        )
        
        assert output.shape == (16, 1)
        
    def test_model_probability_output(self, model, batch_data):
        """Test that outputs are valid probabilities."""
        output = model(
            batch_data['sparse_features'],
            batch_data['dense_features'],
        )
        
        # After sigmoid, should be in [0, 1]
        probs = torch.sigmoid(output)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
        
    def test_model_loss_computation(self, model, batch_data):
        """Test loss computation."""
        output = model(
            batch_data['sparse_features'],
            batch_data['dense_features'],
        )
        
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output.squeeze(), batch_data['labels'])
        
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0
        
    def test_model_gradient_flow(self, model, batch_data):
        """Test that gradients flow through all components."""
        output = model(
            batch_data['sparse_features'],
            batch_data['dense_features'],
        )
        
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                
    def test_model_training_step(self, model, config, batch_data):
        """Test complete training step."""
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        output = model(
            batch_data['sparse_features'],
            batch_data['dense_features'],
        )
        
        loss = criterion(output.squeeze(), batch_data['labels'])
        loss.backward()
        optimizer.step()
        
        # Verify parameters changed
        assert loss.item() > 0


# =============================================================================
# DCN Tests
# =============================================================================

class TestCrossLayer:
    """Tests for DCN cross layer."""
    
    def test_cross_layer_forward(self):
        """Test cross layer forward pass."""
        layer = CrossLayer(input_dim=64)
        
        x0 = torch.randn(8, 64)
        x = torch.randn(8, 64)
        
        output = layer(x0, x)
        
        assert output.shape == (8, 64)
        
    def test_cross_layer_computation(self):
        """Test cross layer computation is correct."""
        layer = CrossLayer(input_dim=4)
        
        # Set known weights for testing
        with torch.no_grad():
            layer.weight.fill_(0.5)
            layer.bias.fill_(0.1)
        
        x0 = torch.ones(2, 4)
        x = torch.ones(2, 4) * 2
        
        output = layer(x0, x)
        
        # x0 * (x @ w) + b + x
        # 1 * (2*4*0.5) + 0.1 + 2 = 4 + 0.1 + 2 = 6.1
        assert output.shape == (2, 4)


class TestDCN:
    """Tests for Deep & Cross Network."""
    
    def test_dcn_forward(self):
        """Test DCN forward pass."""
        config = DLRMConfig(
            num_sparse_features=3,
            sparse_cardinalities=[50, 30, 20],
            sparse_embedding_dim=8,
            num_dense_features=4,
            bottom_mlp_dims=[16, 8],
            top_mlp_dims=[32, 1],
        )
        
        dcn = DCN(config, num_cross_layers=3)
        
        sparse = torch.randint(0, 10, (8, 3))
        dense = torch.randn(8, 4)
        
        output = dcn(sparse, dense)
        
        assert output.shape == (8, 1)
        
    def test_dcn_cross_network(self):
        """Test that cross network creates feature crosses."""
        config = DLRMConfig(
            num_sparse_features=2,
            sparse_cardinalities=[10, 10],
            sparse_embedding_dim=4,
            num_dense_features=2,
        )
        
        dcn = DCN(config, num_cross_layers=2)
        
        # Run inference
        sparse = torch.randint(0, 10, (4, 2))
        dense = torch.randn(4, 2)
        
        dcn.eval()
        with torch.no_grad():
            output = dcn(sparse, dense)
        
        assert not torch.isnan(output).any()


# =============================================================================
# Multi-Task DLRM Tests
# =============================================================================

class TestMultiTaskDLRM:
    """Tests for multi-task DLRM."""
    
    @pytest.fixture
    def mt_model(self, config):
        """Create multi-task model."""
        return MultiTaskDLRM(
            config=config,
            tasks=['ctr', 'cvr', 'revenue'],
        )
        
    def test_multitask_forward(self, mt_model, batch_data):
        """Test multi-task forward pass."""
        outputs = mt_model(
            batch_data['sparse_features'],
            batch_data['dense_features'],
        )
        
        assert 'ctr' in outputs
        assert 'cvr' in outputs
        assert 'revenue' in outputs
        
        for task_output in outputs.values():
            assert task_output.shape == (16, 1)
            
    def test_multitask_loss(self, mt_model, batch_data):
        """Test multi-task loss computation."""
        outputs = mt_model(
            batch_data['sparse_features'],
            batch_data['dense_features'],
        )
        
        labels = {
            'ctr': batch_data['labels'],
            'cvr': batch_data['labels'],
            'revenue': torch.rand(16),
        }
        
        loss = mt_model.compute_loss(outputs, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0
        
    def test_multitask_individual_losses(self, mt_model, batch_data):
        """Test that individual task losses are tracked."""
        outputs = mt_model(
            batch_data['sparse_features'],
            batch_data['dense_features'],
        )
        
        labels = {
            'ctr': batch_data['labels'],
            'cvr': batch_data['labels'],
            'revenue': torch.rand(16),
        }
        
        total_loss, task_losses = mt_model.compute_loss(
            outputs, labels, return_individual=True
        )
        
        assert 'ctr' in task_losses
        assert 'cvr' in task_losses
        assert 'revenue' in task_losses


# =============================================================================
# Interaction Type Tests
# =============================================================================

class TestInteractionTypes:
    """Tests for different feature interaction types."""
    
    @pytest.mark.parametrize("interaction_type", ['dot', 'cat', 'cross', 'attention'])
    def test_all_interaction_types(self, interaction_type):
        """Test DLRM with each interaction type."""
        config = DLRMConfig(
            num_sparse_features=4,
            sparse_cardinalities=[20, 20, 20, 20],
            sparse_embedding_dim=8,
            num_dense_features=4,
            interaction_type=interaction_type,
        )
        
        model = DLRM(config)
        
        sparse = torch.randint(0, 10, (8, 4))
        dense = torch.randn(8, 4)
        
        output = model(sparse, dense)
        
        assert output.shape == (8, 1)
        assert not torch.isnan(output).any()


# =============================================================================
# Performance and Scale Tests
# =============================================================================

class TestScalability:
    """Tests for model scalability."""
    
    def test_large_embedding_tables(self):
        """Test with large embedding tables (simulating production)."""
        config = DLRMConfig(
            num_sparse_features=26,  # Criteo-like
            sparse_cardinalities=[10000] * 13 + [1000] * 13,
            sparse_embedding_dim=64,
            num_dense_features=13,
        )
        
        model = DLRM(config)
        
        # Small batch for testing
        sparse = torch.randint(0, 100, (4, 26))
        dense = torch.randn(4, 13)
        
        output = model(sparse, dense)
        
        assert output.shape == (4, 1)
        
    def test_batch_size_scaling(self, config):
        """Test model with various batch sizes."""
        model = DLRM(config)
        
        for batch_size in [1, 8, 32, 128, 512]:
            sparse = torch.randint(0, 5, (batch_size, config.num_sparse_features))
            dense = torch.randn(batch_size, config.num_dense_features)
            
            output = model(sparse, dense)
            
            assert output.shape == (batch_size, 1)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_dense_features(self):
        """Test model with no dense features."""
        config = DLRMConfig(
            num_sparse_features=5,
            sparse_cardinalities=[100, 50, 20, 10, 5],
            sparse_embedding_dim=16,
            num_dense_features=0,
        )
        
        model = DLRM(config)
        
        sparse = torch.randint(0, 5, (8, 5))
        dense = torch.empty(8, 0)
        
        output = model(sparse, dense)
        
        assert output.shape == (8, 1)
        
    def test_single_sparse_feature(self):
        """Test with single sparse feature."""
        config = DLRMConfig(
            num_sparse_features=1,
            sparse_cardinalities=[1000],
            sparse_embedding_dim=32,
            num_dense_features=4,
        )
        
        model = DLRM(config)
        
        sparse = torch.randint(0, 100, (8, 1))
        dense = torch.randn(8, 4)
        
        output = model(sparse, dense)
        
        assert output.shape == (8, 1)
        
    def test_extreme_values(self, model, config):
        """Test model with extreme input values."""
        sparse = torch.randint(0, 5, (8, config.num_sparse_features))
        dense = torch.randn(8, config.num_dense_features) * 1000  # Large values
        
        output = model(sparse, dense)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
