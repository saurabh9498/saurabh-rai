"""
A compact DLRM (Deep Learning Recommendation Model) for real-time bid scoring.

Structure follows the DLRM paper (Naumov et al., 2019):
  dense features  -> bottom MLP -> 1 vector
  sparse features -> embedding lookups -> F vectors
  pairwise dot-product interaction of all F+1 vectors -> top MLP -> click probability

The model is split in two on purpose, because that is how it is served:

  * EmbeddingStore  - the sparse lookup tables. In production these are far too large
                      for GPU memory, so lookups happen OUTSIDE the inference engine:
                      a hot tier in VRAM, a cold tail in Redis (see sim/hot_cache_sim.py).
  * DenseScorer     - bottom MLP + interaction + top MLP. This is what gets exported to
                      ONNX and compiled to a TensorRT INT8 engine. It takes already
                      looked-up embedding vectors as input.

Sizes here are small so the whole thing runs on a laptop CPU. The production-scale
footprint is handled by the sizing scripts, not by instantiating a 1.2 TB model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class DLRMConfig:
    num_dense: int = 13                       # continuous features (bid floor, time-of-day, ...)
    num_sparse: int = 26                      # categorical features (user, device, publisher, ...)
    embedding_dim: int = 64
    vocab_size: int = 100_000                 # rows per table (tiny vs. production)
    bottom_mlp: list[int] = field(default_factory=lambda: [512, 256, 64])
    top_mlp: list[int] = field(default_factory=lambda: [512, 256, 1])


def _mlp(sizes: list[int], in_dim: int, final_activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, out_dim in enumerate(sizes):
        layers.append(nn.Linear(in_dim, out_dim))
        last = i == len(sizes) - 1
        if not last or final_activation:
            layers.append(nn.ReLU())
        in_dim = out_dim
    return nn.Sequential(*layers)


class EmbeddingStore(nn.Module):
    """One table per sparse feature. Served outside the engine in production."""

    def __init__(self, cfg: DLRMConfig):
        super().__init__()
        self.tables = nn.ModuleList(
            nn.Embedding(cfg.vocab_size, cfg.embedding_dim) for _ in range(cfg.num_sparse)
        )

    def forward(self, sparse_ids: torch.Tensor) -> torch.Tensor:
        # sparse_ids: [B, F] int64  ->  [B, F, D]
        return torch.stack([t(sparse_ids[:, i]) for i, t in enumerate(self.tables)], dim=1)


class DenseScorer(nn.Module):
    """Bottom MLP + dot interaction + top MLP. This is the exported/compiled part."""

    def __init__(self, cfg: DLRMConfig):
        super().__init__()
        if cfg.bottom_mlp[-1] != cfg.embedding_dim:
            raise ValueError("bottom MLP output must equal embedding_dim for the dot interaction")
        self.bottom = _mlp(cfg.bottom_mlp, cfg.num_dense, final_activation=True)
        n = cfg.num_sparse + 1
        self.num_pairs = n * (n - 1) // 2
        rows, cols = torch.triu_indices(n, n, offset=1)
        # Flattened upper-triangle indices: a plain gather, which ONNX and TensorRT map cleanly.
        self.register_buffer("pair_index", rows * n + cols, persistent=False)
        self.top = _mlp(cfg.top_mlp, cfg.embedding_dim + self.num_pairs, final_activation=False)

    def forward(self, dense: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        # dense: [B, num_dense] float, embeddings: [B, F, D] float
        x = self.bottom(dense)                                   # [B, D]
        feats = torch.cat([x.unsqueeze(1), embeddings], dim=1)   # [B, F+1, D]
        inter = torch.bmm(feats, feats.transpose(1, 2))          # [B, F+1, F+1]
        pairs = inter.flatten(1).index_select(1, self.pair_index)  # [B, pairs]
        logits = self.top(torch.cat([x, pairs], dim=1))          # [B, 1]
        return torch.sigmoid(logits)


class DLRM(nn.Module):
    """Full model, used for training and for checking the split is lossless."""

    def __init__(self, cfg: DLRMConfig | None = None):
        super().__init__()
        self.cfg = cfg or DLRMConfig()
        self.embeddings = EmbeddingStore(self.cfg)
        self.scorer = DenseScorer(self.cfg)

    def forward(self, dense: torch.Tensor, sparse_ids: torch.Tensor) -> torch.Tensor:
        return self.scorer(dense, self.embeddings(sparse_ids))


def synthetic_batch(cfg: DLRMConfig, batch: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(batch, cfg.num_dense, generator=g)
    sparse = torch.randint(0, cfg.vocab_size, (batch, cfg.num_sparse), generator=g)
    return dense, sparse
