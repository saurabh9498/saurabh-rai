"""
Deep & Cross Network (DCN) Implementation

DCN-V2 architecture for learning explicit feature interactions.
Combines cross network for bounded-degree interactions with DNN for implicit patterns.

Reference: https://arxiv.org/abs/2008.13535
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


@dataclass
class DCNConfig:
    """Configuration for Deep & Cross Network."""
    
    # Input dimensions
    num_sparse_features: int = 26
    num_dense_features: int = 13
    sparse_cardinalities: List[int] = field(default_factory=lambda: [1000] * 26)
    sparse_embedding_dim: int = 64
    
    # Cross network
    num_cross_layers: int = 3
    cross_layer_type: str = "vector"  # "vector" or "matrix"
    
    # Deep network
    deep_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    deep_dropout: float = 0.1
    deep_activation: str = "relu"
    
    # Output
    output_dim: int = 1
    
    # Stacking mode
    stacking: str = "parallel"  # "parallel" or "stacked"


class CrossLayerV2(nn.Module):
    """
    Cross Layer V2 with vector or matrix formulation.
    
    Vector: x_{l+1} = x_0 * (W * x_l + b) + x_l
    Matrix: x_{l+1} = x_0 * (W * x_l + b) + x_l  (W is low-rank)
    """
    
    def __init__(
        self,
        input_dim: int,
        layer_type: str = "vector",
        low_rank_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.layer_type = layer_type
        self.input_dim = input_dim
        
        if layer_type == "vector":
            self.weight = nn.Parameter(torch.randn(input_dim))
            self.bias = nn.Parameter(torch.zeros(input_dim))
        else:  # matrix (low-rank)
            low_rank_dim = low_rank_dim or input_dim // 4
            self.U = nn.Parameter(torch.randn(input_dim, low_rank_dim))
            self.V = nn.Parameter(torch.randn(low_rank_dim, input_dim))
            self.bias = nn.Parameter(torch.zeros(input_dim))
            
            # Initialize
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
    
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: Original input (batch, input_dim)
            xl: Output from previous layer (batch, input_dim)
            
        Returns:
            Cross layer output (batch, input_dim)
        """
        if self.layer_type == "vector":
            # x_{l+1} = x_0 * (w * x_l + b) + x_l
            cross = x0 * (torch.sum(xl * self.weight, dim=-1, keepdim=True) + self.bias)
        else:
            # x_{l+1} = x_0 * (U @ V @ x_l + b) + x_l
            # Low-rank approximation of W = U @ V
            cross = x0 * (xl @ self.V.T @ self.U.T + self.bias)
        
        return cross + xl


class CrossNetwork(nn.Module):
    """
    Cross Network - stacked cross layers for explicit feature interactions.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        layer_type: str = "vector",
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CrossLayerV2(input_dim, layer_type)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            Cross network output (batch, input_dim)
        """
        x0 = x
        xl = x
        
        for layer in self.layers:
            xl = layer(x0, xl)
        
        return xl


class DeepNetwork(nn.Module):
    """
    Deep Network - MLP for implicit feature interactions.
    """
    
    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int],
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in layer_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = layer_dims[-1] if layer_dims else input_dim
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DCN(nn.Module):
    """
    Deep & Cross Network V2.
    
    Combines cross network for explicit bounded-degree feature interactions
    with deep network for implicit high-order interactions.
    """
    
    def __init__(self, config: DCNConfig):
        super().__init__()
        self.config = config
        
        # Sparse embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, config.sparse_embedding_dim)
            for cardinality in config.sparse_cardinalities
        ])
        
        # Input dimension after embedding
        sparse_output_dim = config.num_sparse_features * config.sparse_embedding_dim
        input_dim = sparse_output_dim + config.num_dense_features
        
        # Cross network
        self.cross_network = CrossNetwork(
            input_dim=input_dim,
            num_layers=config.num_cross_layers,
            layer_type=config.cross_layer_type,
        )
        
        # Deep network
        self.deep_network = DeepNetwork(
            input_dim=input_dim,
            layer_dims=config.deep_layers,
            dropout=config.deep_dropout,
            activation=config.deep_activation,
        )
        
        # Final projection
        if config.stacking == "parallel":
            final_dim = input_dim + self.deep_network.output_dim
        else:
            final_dim = self.deep_network.output_dim
        
        self.final_layer = nn.Linear(final_dim, config.output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sparse_features: Categorical feature indices (batch, num_sparse)
            dense_features: Numerical features (batch, num_dense)
            
        Returns:
            Predictions (batch, output_dim)
        """
        # Embed sparse features
        sparse_embeds = []
        for i, embedding in enumerate(self.embeddings):
            sparse_embeds.append(embedding(sparse_features[:, i]))
        
        sparse_concat = torch.cat(sparse_embeds, dim=-1)
        
        # Concatenate with dense features
        x = torch.cat([sparse_concat, dense_features], dim=-1)
        
        if self.config.stacking == "parallel":
            # Parallel: Cross and Deep networks run in parallel
            cross_out = self.cross_network(x)
            deep_out = self.deep_network(x)
            combined = torch.cat([cross_out, deep_out], dim=-1)
        else:
            # Stacked: Cross output feeds into Deep network
            cross_out = self.cross_network(x)
            combined = self.deep_network(cross_out)
        
        output = self.final_layer(combined)
        
        return output
    
    def compute_loss(
        self,
        sparse_features: torch.Tensor,
        dense_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute BCE loss."""
        logits = self.forward(sparse_features, dense_features)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
        return loss, torch.sigmoid(logits)


class DCNMix(nn.Module):
    """
    DCN-Mix: DCN with mixture of experts for cross layers.
    
    Uses multiple expert cross networks and a gating mechanism.
    """
    
    def __init__(
        self,
        config: DCNConfig,
        num_experts: int = 4,
    ):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        
        # Sparse embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, config.sparse_embedding_dim)
            for cardinality in config.sparse_cardinalities
        ])
        
        sparse_output_dim = config.num_sparse_features * config.sparse_embedding_dim
        input_dim = sparse_output_dim + config.num_dense_features
        
        # Multiple expert cross networks
        self.expert_networks = nn.ModuleList([
            CrossNetwork(input_dim, config.num_cross_layers, config.cross_layer_type)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1),
        )
        
        # Deep network
        self.deep_network = DeepNetwork(
            input_dim=input_dim,
            layer_dims=config.deep_layers,
            dropout=config.deep_dropout,
        )
        
        final_dim = input_dim + self.deep_network.output_dim
        self.final_layer = nn.Linear(final_dim, config.output_dim)
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        # Embed sparse features
        sparse_embeds = [emb(sparse_features[:, i]) for i, emb in enumerate(self.embeddings)]
        sparse_concat = torch.cat(sparse_embeds, dim=-1)
        x = torch.cat([sparse_concat, dense_features], dim=-1)
        
        # Gating weights
        gate_weights = self.gate(x)  # (batch, num_experts)
        
        # Expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.expert_networks
        ], dim=1)  # (batch, num_experts, input_dim)
        
        # Weighted combination
        cross_out = torch.sum(
            expert_outputs * gate_weights.unsqueeze(-1),
            dim=1
        )  # (batch, input_dim)
        
        # Deep network
        deep_out = self.deep_network(x)
        
        # Combine and output
        combined = torch.cat([cross_out, deep_out], dim=-1)
        output = self.final_layer(combined)
        
        return output
