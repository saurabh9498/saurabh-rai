"""
Session-Based Sequence Models

Implements sequential recommendation models that capture temporal patterns
and intra-session behavior for real-time personalization.

Includes:
- GRU4Rec: RNN-based session model
- SASRec: Self-attention based sequential model
- BERT4Rec: Bidirectional transformer for sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


@dataclass
class SequenceConfig:
    """Configuration for sequence models."""
    
    # Vocabulary
    num_items: int = 100000
    embedding_dim: int = 128
    max_sequence_length: int = 50
    
    # Model architecture
    model_type: str = "sasrec"  # "gru4rec", "sasrec", "bert4rec"
    
    # RNN settings (for GRU4Rec)
    num_rnn_layers: int = 2
    rnn_dropout: float = 0.2
    
    # Transformer settings (for SASRec/BERT4Rec)
    num_attention_heads: int = 4
    num_transformer_layers: int = 2
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    ffn_hidden_dim: int = 512
    
    # Training
    mask_prob: float = 0.15  # For BERT4Rec


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for sequences."""
    
    def __init__(self, max_len: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        position_embeds = self.position_embeddings(positions)
        return self.dropout(x + position_embeds)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking option."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        
        assert embedding_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.causal = causal
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal mask (prevent attending to future)
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Padding mask
        if attention_mask is not None:
            padding_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(2)) * float('-inf')
            scores = scores + padding_mask
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.output(context)


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(
            embedding_dim, num_heads, attention_dropout, causal
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(ffn_hidden_dim, embedding_dim),
            nn.Dropout(hidden_dropout),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(hidden_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class GRU4Rec(nn.Module):
    """
    GRU-based session recommendation model.
    
    Reference: Session-based Recommendations with Recurrent Neural Networks (ICLR 2016)
    """
    
    def __init__(self, config: SequenceConfig):
        super().__init__()
        self.config = config
        
        self.item_embedding = nn.Embedding(
            config.num_items + 1,  # +1 for padding
            config.embedding_dim,
            padding_idx=0,
        )
        
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=config.num_rnn_layers,
            dropout=config.rnn_dropout if config.num_rnn_layers > 1 else 0,
            batch_first=True,
        )
        
        self.output_layer = nn.Linear(config.embedding_dim, config.num_items)
    
    def forward(
        self,
        item_seq: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            item_seq: (batch, seq_len) item indices
            seq_lengths: (batch,) actual sequence lengths
            
        Returns:
            Logits over all items (batch, seq_len, num_items)
        """
        # Embed items
        x = self.item_embedding(item_seq)  # (batch, seq_len, embed_dim)
        
        # Pack if variable length
        if seq_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # GRU encoding
        output, _ = self.gru(x)
        
        # Unpack
        if seq_lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Project to item space
        logits = self.output_layer(output)
        
        return logits
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get all item embeddings for ANN search."""
        return self.item_embedding.weight[1:]  # Exclude padding


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation.
    
    Reference: Self-Attentive Sequential Recommendation (ICDM 2018)
    """
    
    def __init__(self, config: SequenceConfig):
        super().__init__()
        self.config = config
        
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            config.embedding_dim,
            padding_idx=0,
        )
        
        self.position_encoding = PositionalEncoding(
            config.max_sequence_length,
            config.embedding_dim,
            config.hidden_dropout,
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim,
                config.num_attention_heads,
                config.ffn_hidden_dim,
                config.attention_dropout,
                config.hidden_dropout,
                causal=True,
            )
            for _ in range(config.num_transformer_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        item_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            item_seq: (batch, seq_len) item indices
            attention_mask: (batch, seq_len) 1 for valid, 0 for padding
            
        Returns:
            Sequence representations (batch, seq_len, embed_dim)
        """
        # Embed items
        x = self.item_embedding(item_seq)
        x = self.position_encoding(x)
        
        # Generate attention mask from padding
        if attention_mask is None:
            attention_mask = (item_seq != 0).float()
        
        # Transformer layers
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        x = self.layer_norm(x)
        
        return x
    
    def predict(
        self,
        item_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get predictions for next item."""
        seq_repr = self.forward(item_seq, attention_mask)
        
        # Use last valid position for prediction
        last_repr = seq_repr[:, -1, :]  # (batch, embed_dim)
        
        # Score all items
        item_embeds = self.item_embedding.weight[1:]  # Exclude padding
        logits = torch.matmul(last_repr, item_embeds.T)
        
        return logits


class BERT4Rec(nn.Module):
    """
    BERT-style bidirectional model for sequential recommendation.
    
    Uses masked language model training objective.
    Reference: BERT4Rec (CIKM 2019)
    """
    
    def __init__(self, config: SequenceConfig):
        super().__init__()
        self.config = config
        
        # +2 for padding and mask tokens
        self.item_embedding = nn.Embedding(
            config.num_items + 2,
            config.embedding_dim,
            padding_idx=0,
        )
        self.mask_token_id = config.num_items + 1
        
        self.position_encoding = PositionalEncoding(
            config.max_sequence_length,
            config.embedding_dim,
            config.hidden_dropout,
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim,
                config.num_attention_heads,
                config.ffn_hidden_dim,
                config.attention_dropout,
                config.hidden_dropout,
                causal=False,  # Bidirectional
            )
            for _ in range(config.num_transformer_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.output_layer = nn.Linear(config.embedding_dim, config.num_items)
    
    def forward(
        self,
        item_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            item_seq: (batch, seq_len) with some positions masked
            attention_mask: (batch, seq_len)
            
        Returns:
            Logits for masked positions (batch, seq_len, num_items)
        """
        x = self.item_embedding(item_seq)
        x = self.position_encoding(x)
        
        if attention_mask is None:
            attention_mask = (item_seq != 0).float()
        
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        x = self.layer_norm(x)
        logits = self.output_layer(x)
        
        return logits
    
    def mask_sequence(
        self,
        item_seq: torch.Tensor,
        mask_prob: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mask random positions for training.
        
        Returns:
            masked_seq: Sequence with masks
            mask_positions: Boolean mask of which positions are masked
            labels: Original items at masked positions
        """
        batch_size, seq_len = item_seq.shape
        
        # Don't mask padding
        valid_mask = (item_seq != 0)
        
        # Random mask positions
        rand = torch.rand_like(item_seq.float())
        mask_positions = (rand < mask_prob) & valid_mask
        
        # Create masked sequence
        masked_seq = item_seq.clone()
        masked_seq[mask_positions] = self.mask_token_id
        
        # Labels (only for masked positions)
        labels = item_seq.clone()
        labels[~mask_positions] = -100  # Ignore in loss
        
        return masked_seq, mask_positions, labels


def create_sequence_model(config: SequenceConfig) -> nn.Module:
    """Factory function to create sequence model."""
    models = {
        "gru4rec": GRU4Rec,
        "sasrec": SASRec,
        "bert4rec": BERT4Rec,
    }
    
    if config.model_type not in models:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return models[config.model_type](config)
