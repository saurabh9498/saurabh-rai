"""
KV-cache sizing and tensor-parallel (TP) selection for the edge LLM pods.

The KV cache stores, for every token of every live request, one key and one value vector
per layer per KV head. With grouped-query attention (GQA) the number of KV heads is much
smaller than the number of query heads, which is what keeps these numbers manageable.

    bytes per token  = 2 (K and V) x layers x kv_heads x head_dim x bytes per element
    bytes per stream = bytes per token x context length (prompt + generated tokens)
    pool             = bytes per stream x concurrent streams

Units: GB = 10^9 bytes, GiB = 2^30 bytes. Mixing them is the most common way these
calculations drift by a few percent, so both are printed.

Run:  python sizing/kv_cache.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass

GB, GiB = 1e9, 2 ** 30


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float        # billions of parameters
    layers: int
    kv_heads: int
    head_dim: int


LLAMA3_8B = Model("Llama-3-8B", 8.0, 32, 8, 128)
LLAMA3_70B = Model("Llama-3-70B", 70.0, 80, 8, 128)


def kv_bytes_per_token(m: Model, bytes_per_elem: float = 1.0) -> float:
    return 2 * m.layers * m.kv_heads * m.head_dim * bytes_per_elem


def kv_pool_bytes(m: Model, context_tokens: int, streams: int, bytes_per_elem: float = 1.0) -> float:
    return kv_bytes_per_token(m, bytes_per_elem) * context_tokens * streams


def vram_plan(m: Model, context_tokens: int, streams: int, tp: int, gpu_gb: float = 96.0,
              weight_bytes_per_param: float = 1.0, workspace_gb_per_gpu: float = 3.0,
              cuda_ctx_gb_per_gpu: float = 0.5) -> dict:
    """Static VRAM plan for one TP group. FP8 weights and FP8 KV cache by default."""
    if m.kv_heads % tp:
        raise ValueError(f"TP={tp} must divide {m.kv_heads} KV heads")
    weights = m.params_b * 1e9 * weight_bytes_per_param / GB
    kv = kv_pool_bytes(m, context_tokens, streams) / GB
    workspace = workspace_gb_per_gpu * tp
    ctx = cuda_ctx_gb_per_gpu * tp
    total = gpu_gb * tp
    used = weights + kv + workspace + ctx
    return {"model": m.name, "tp": tp, "pool_gb": total, "weights_gb": weights, "kv_cache_gb": round(kv, 2),
            "workspace_gb": workspace, "cuda_ctx_gb": ctx, "used_gb": round(used, 2),
            "headroom_gb": round(total - used, 2), "headroom_frac": round((total - used) / total, 3),
            "fits": used <= total}


def choose_tp(m: Model, context_tokens: int, streams: int, min_headroom_frac: float = 0.25,
              options=(1, 2, 4, 8), **kw) -> tuple[int, list[dict]]:
    """Smallest TP whose plan fits AND leaves the required headroom for bursts."""
    plans = [vram_plan(m, context_tokens, streams, tp, **kw) for tp in options if m.kv_heads % tp == 0]
    for p in plans:
        if p["fits"] and p["headroom_frac"] >= min_headroom_frac:
            return p["tp"], plans
    raise ValueError("no TP option meets the headroom requirement")


def main() -> None:
    out = {}
    for m, ctx, streams in ((LLAMA3_8B, 1024, 128), (LLAMA3_70B, 1280, 500)):
        per_tok = kv_bytes_per_token(m)
        per_stream = per_tok * ctx
        pool = per_stream * streams
        out[m.name] = {
            "kv_bytes_per_token": per_tok,
            "context_tokens": ctx,
            "per_stream": {"MB": round(per_stream / 1e6, 1), "MiB": round(per_stream / 2 ** 20, 1)},
            "streams": streams,
            "pool": {"GB": round(pool / GB, 2), "GiB": round(pool / GiB, 2)},
        }
    out["Llama-3-8B_TP1_plan"] = vram_plan(LLAMA3_8B, 1024, 128, tp=1, workspace_gb_per_gpu=4.0)
    tp, plans = choose_tp(LLAMA3_70B, 1280, 500)
    out["Llama-3-70B_tp_options"] = plans
    out["Llama-3-70B_chosen_tp"] = tp
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
