"""
Embedding Layers and Utilities

Provides efficient embedding implementations for recommendation systems:
- Mixed-dimension embeddings
- Hashed embeddings for high-cardinality features
- Pre-trained embedding loading
- Embedding compression and quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import math


class MixedDimensionEmbedding(nn.Module):
    """
    Embedding layer with different dimensions per feature.
    
    Useful when features have different cardinalities and should
    use dimension proportional to log(cardinality).
    """
    
    def __init__(
        self,
        cardinalities: List[int],
        embedding_dims: Optional[List[int]] = None,
        base_dim: int = 16,
        max_dim: int = 64,
    ):
        super().__init__()
        
        self.num_features = len(cardinalities)
        
        # Calculate dimensions if not provided
        if embedding_dims is None:
            embedding_dims = [
                min(max_dim, max(base_dim, int(math.log2(card) * 4)))
                for card in cardinalities
            ]
        
        self.embedding_dims = embedding_dims
        self.total_dim = sum(embedding_dims)
        
        # Create embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim)
            for card, dim in zip(cardinalities, embedding_dims)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (batch, num_features) feature indices
            
        Returns:
            Concatenated embeddings (batch, total_dim)
        """
        embeds = [
            self.embeddings[i](indices[:, i])
            for i in range(self.num_features)
        ]
        return torch.cat(embeds, dim=-1)


class HashedEmbedding(nn.Module):
    """
    Hashed embedding for ultra-high cardinality features.
    
    Uses multiple hash functions to reduce collisions.
    Memory efficient but may have some information loss.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_hashes: int = 2,
        hash_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        
        # Create embedding tables for each hash function
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for _ in range(num_hashes)
        ])
        
        # Hash parameters (random primes)
        self.register_buffer(
            'hash_a',
            torch.tensor([2654435761, 805459861, 3266489917, 668265263][:num_hashes])
        )
        self.register_buffer(
            'hash_b', 
            torch.tensor([1, 2, 3, 5][:num_hashes])
        )
        
        # Aggregation weights
        if hash_weights:
            self.register_buffer('weights', torch.tensor(hash_weights))
        else:
            self.register_buffer('weights', torch.ones(num_hashes) / num_hashes)
        
        self._init_weights()
    
    def _init_weights(self):
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
    
    def _hash(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply hash function."""
        return ((x * self.hash_a[idx] + self.hash_b[idx]) % self.num_embeddings).long()
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (batch,) or (batch, seq_len) raw feature values
            
        Returns:
            Embeddings (batch, embedding_dim) or (batch, seq_len, embedding_dim)
        """
        embeddings = []
        
        for i, embedding in enumerate(self.embeddings):
            hashed_idx = self._hash(indices, i)
            embeddings.append(embedding(hashed_idx))
        
        # Weighted sum of hash embeddings
        stacked = torch.stack(embeddings, dim=0)  # (num_hashes, batch, ..., dim)
        weights = self.weights.view(-1, *([1] * (stacked.dim() - 1)))
        
        return (stacked * weights).sum(dim=0)


class QuantizedEmbedding(nn.Module):
    """
    Quantized embedding with learned codebook.
    
    Reduces memory usage while maintaining quality through
    product quantization.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_codebooks: int = 8,
        codebook_size: int = 256,
    ):
        super().__init__()
        
        assert embedding_dim % num_codebooks == 0
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.subvector_dim = embedding_dim // num_codebooks
        
        # Codebooks (num_codebooks, codebook_size, subvector_dim)
        self.codebooks = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, self.subvector_dim) * 0.01
        )
        
        # Assignment indices (num_embeddings, num_codebooks)
        self.register_buffer(
            'codes',
            torch.randint(0, codebook_size, (num_embeddings, num_codebooks))
        )
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (batch,) embedding indices
            
        Returns:
            Reconstructed embeddings (batch, embedding_dim)
        """
        # Get codes for each embedding
        codes = self.codes[indices]  # (batch, num_codebooks)
        
        # Gather from codebooks
        subvectors = []
        for i in range(self.num_codebooks):
            subvector = self.codebooks[i, codes[:, i]]  # (batch, subvector_dim)
            subvectors.append(subvector)
        
        return torch.cat(subvectors, dim=-1)
    
    def quantize_embeddings(self, full_embeddings: torch.Tensor):
        """Quantize full embeddings to codebook indices."""
        with torch.no_grad():
            # Split into subvectors
            subvectors = full_embeddings.view(
                -1, self.num_codebooks, self.subvector_dim
            )
            
            # Find nearest codebook entry for each subvector
            for i in range(self.num_codebooks):
                distances = torch.cdist(
                    subvectors[:, i],
                    self.codebooks[i]
                )
                self.codes[:, i] = distances.argmin(dim=-1)


class PretrainedEmbedding(nn.Module):
    """
    Wrapper for loading pre-trained embeddings (e.g., from Word2Vec, FastText).
    
    Supports freezing, fine-tuning, and optional projection.
    """
    
    def __init__(
        self,
        pretrained_weights: torch.Tensor,
        freeze: bool = False,
        projection_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weights,
            freeze=freeze,
            padding_idx=0,
        )
        
        self.projection = None
        if projection_dim:
            self.projection = nn.Linear(
                pretrained_weights.shape[1],
                projection_dim,
            )
        
        self.dropout = nn.Dropout(dropout)
    
    @classmethod
    def from_numpy(
        cls,
        weights: np.ndarray,
        **kwargs,
    ) -> 'PretrainedEmbedding':
        """Load from numpy array."""
        return cls(torch.from_numpy(weights).float(), **kwargs)
    
    @classmethod
    def from_file(
        cls,
        path: str,
        vocab: Dict[str, int],
        embedding_dim: int,
        **kwargs,
    ) -> 'PretrainedEmbedding':
        """Load from text file (Word2Vec format)."""
        weights = np.zeros((len(vocab), embedding_dim))
        
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                    idx = vocab[word]
                    weights[idx] = np.array(parts[1:], dtype=np.float32)
        
        return cls.from_numpy(weights, **kwargs)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(indices)
        x = self.dropout(x)
        
        if self.projection:
            x = self.projection(x)
        
        return x


class FeatureEmbedding(nn.Module):
    """
    Unified embedding layer for multiple feature types.
    
    Handles categorical, sequential, and pre-trained embeddings
    with automatic dimension alignment.
    """
    
    def __init__(
        self,
        feature_configs: List[Dict],
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            feature_configs: List of feature configurations
                Each config has: name, type (categorical/sequence/pretrained),
                cardinality, embedding_dim, and optional type-specific params
            output_dim: Project all features to this dim (optional)
        """
        super().__init__()
        
        self.feature_configs = feature_configs
        self.embeddings = nn.ModuleDict()
        
        total_dim = 0
        
        for config in feature_configs:
            name = config['name']
            feat_type = config.get('type', 'categorical')
            
            if feat_type == 'categorical':
                self.embeddings[name] = nn.Embedding(
                    config['cardinality'],
                    config['embedding_dim'],
                    padding_idx=config.get('padding_idx'),
                )
                total_dim += config['embedding_dim']
                
            elif feat_type == 'hashed':
                self.embeddings[name] = HashedEmbedding(
                    config['num_buckets'],
                    config['embedding_dim'],
                    config.get('num_hashes', 2),
                )
                total_dim += config['embedding_dim']
                
            elif feat_type == 'mixed':
                self.embeddings[name] = MixedDimensionEmbedding(
                    config['cardinalities'],
                    config.get('embedding_dims'),
                )
                total_dim += self.embeddings[name].total_dim
        
        self.total_dim = total_dim
        
        self.projection = None
        if output_dim:
            self.projection = nn.Linear(total_dim, output_dim)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Dict mapping feature name to tensor
            
        Returns:
            Combined embeddings (batch, total_dim or output_dim)
        """
        embeds = []
        
        for config in self.feature_configs:
            name = config['name']
            if name in features and name in self.embeddings:
                embed = self.embeddings[name](features[name])
                embeds.append(embed)
        
        combined = torch.cat(embeds, dim=-1)
        
        if self.projection:
            combined = self.projection(combined)
        
        return combined


def create_item_embeddings(
    num_items: int,
    embedding_dim: int,
    pretrained_path: Optional[str] = None,
) -> nn.Embedding:
    """Factory function for item embeddings."""
    
    if pretrained_path:
        weights = torch.load(pretrained_path)
        embedding = nn.Embedding.from_pretrained(weights, freeze=False)
    else:
        embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(embedding.weight)
        embedding.weight.data[0].zero_()  # Zero padding
    
    return embedding
