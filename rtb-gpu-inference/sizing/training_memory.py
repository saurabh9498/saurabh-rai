"""
Why a 1.2 TB DLRM does not fit an 8x H100 node naively, and the three changes that
made single-node training work.

Node: 8x H100 80 GB = 640 GB HBM3, ~3.35 TB/s HBM per GPU, 2 TB host DRAM,
PCIe Gen5 x16 ~64 GB/s per direction.

1. Memory: weights + gradients + Adam (2 states) in FP32 = 4x the model.
   Fix: row-wise sparse Adagrad on the embedding tables (one FP32 value per ROW, not per
   element), Adam only on the small dense layers; cold rows tiered to host DRAM via UVM.
2. Network: ad traffic is Zipfian, so random sharding sends most lookups to one or two
   GPUs and chokes the AllToAll. Fix: replicate the hottest ~3% of rows on every GPU.
3. Checkpoint I/O: torch.save goes GPU -> PCIe -> CPU bounce buffer -> disk.
   Fix: GPUDirect Storage straight to a local NVMe RAID 0.

Run:  python sizing/training_memory.py
"""
from __future__ import annotations

import json

TB, GB = 1e12, 1e9


def naive_adam_bytes(model_bytes: float) -> float:
    # FP32 weights + FP32 grads + Adam first and second moments
    return 4 * model_bytes


def rowwise_adagrad_state_bytes(model_bytes: float, dim: int) -> float:
    rows = model_bytes / (dim * 4)
    return rows * 4                                   # one FP32 accumulator per row


def summarize(model_tb: float = 1.2, dim: int = 128, hot_fraction: float = 0.03,
              gpus: int = 8, hbm_per_gpu_gb: float = 80,
              bounce_buffer_gbps: float = 5.0, gds_gbps: float = 45.0) -> dict:
    model = model_tb * TB
    hbm = gpus * hbm_per_gpu_gb * GB
    hot = hot_fraction * model
    return {
        "node_hbm_gb": hbm / GB,
        "naive_adam_training_tb": naive_adam_bytes(model) / TB,
        "naive_vs_hbm_ratio": naive_adam_bytes(model) / hbm,
        "rowwise_adagrad_state_gb": rowwise_adagrad_state_bytes(model, dim) / GB,
        "full_adam_state_on_embeddings_tb": 2 * model / TB,
        "hot_rows_gb": hot / GB,
        "hot_row_replication_hbm_cost_gb": hot * gpus / GB,
        "checkpoint_bounce_buffer_s": model / (bounce_buffer_gbps * GB),
        "checkpoint_gds_s": model / (gds_gbps * GB),
        "pcie_gen5_vs_hbm3_bandwidth_ratio": 3350 / 64,
    }


def main() -> None:
    s = summarize()
    print(json.dumps(s, indent=2))
    print(
        f"\nNaive Adam needs {s['naive_adam_training_tb']:.1f} TB against {s['node_hbm_gb']:.0f} GB HBM "
        f"({s['naive_vs_hbm_ratio']:.1f}x over). Row-wise Adagrad state: {s['rowwise_adagrad_state_gb']:.1f} GB "
        f"instead of {s['full_adam_state_on_embeddings_tb']:.1f} TB. Replicating the hot {s['hot_rows_gb']:.0f} GB "
        f"on 8 GPUs costs {s['hot_row_replication_hbm_cost_gb']:.0f} GB HBM. Checkpoint: "
        f"{s['checkpoint_bounce_buffer_s']:.0f} s via bounce buffer vs {s['checkpoint_gds_s']:.0f} s with GDS."
    )


if __name__ == "__main__":
    main()
