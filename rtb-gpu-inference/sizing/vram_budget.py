"""
The 24 GB A10G VRAM budget, and what the hot-cache slice can hold.

Production split (per GPU):
    ~5 GB   INT8 TensorRT engine (dense layers)
    ~12 GB  runtime workspace: activations for 128-request batches, TensorRT scratch,
            CUDA context
    ~7 GB   hot embedding cache: the most-requested rows, looked up locally
    rest    cold tail of the embedding tables lives in Redis, one network hop away

Embedding footprint: 1.2 TB in FP32. INT8 post-training quantization is 4x smaller,
so ~300 GB. Row width is an assumption (128-dim), which fixes the row count; the
hot-cache fraction does not depend on it.

Run:  python sizing/vram_budget.py
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

GB = 1e9


@dataclass
class Budget:
    vram_gb: float = 24.0
    engine_gb: float = 5.0
    workspace_gb: float = 12.0
    hot_cache_gb: float = 7.0
    embedding_fp32_tb: float = 1.2
    embedding_dim: int = 128           # assumption; affects row count only

    def check(self) -> None:
        used = self.engine_gb + self.workspace_gb + self.hot_cache_gb
        if used > self.vram_gb + 1e-9:
            raise ValueError(f"budget {used} GB exceeds {self.vram_gb} GB VRAM")


def summarize(b: Budget) -> dict:
    b.check()
    fp32_bytes = b.embedding_fp32_tb * 1e12
    int8_bytes = fp32_bytes / 4                      # FP32 (4 B) -> INT8 (1 B)
    row_bytes_int8 = b.embedding_dim * 1             # 1 byte per element after INT8 PTQ
    total_rows = int8_bytes / row_bytes_int8
    hot_rows = b.hot_cache_gb * GB / row_bytes_int8
    return {
        "inputs": asdict(b),
        "embeddings_int8_gb": int8_bytes / GB,
        "cold_tail_in_redis_gb": (int8_bytes - b.hot_cache_gb * GB) / GB,
        "total_rows": total_rows,
        "hot_rows": hot_rows,
        "hot_fraction_of_rows": hot_rows / total_rows,
        "vram_headroom_gb": b.vram_gb - (b.engine_gb + b.workspace_gb + b.hot_cache_gb),
    }


def main() -> None:
    s = summarize(Budget())
    print(json.dumps(s, indent=2))
    print(
        f"\nINT8 embeddings: {s['embeddings_int8_gb']:.0f} GB. "
        f"Hot cache holds {s['hot_rows'] / 1e6:.1f}M of {s['total_rows'] / 1e9:.2f}B rows "
        f"({s['hot_fraction_of_rows']:.2%}); {s['cold_tail_in_redis_gb']:.0f} GB stays in Redis."
    )


if __name__ == "__main__":
    main()
