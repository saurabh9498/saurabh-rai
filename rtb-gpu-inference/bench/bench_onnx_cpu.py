"""
Benchmark harness for the exported dense scorer, run on CPU with ONNX Runtime.

What this measures: the shape of the latency/throughput trade-off as batch size grows -
the same trade-off Triton's dynamic batcher manages on the GPU. It does NOT measure A10G
or TensorRT performance; those numbers come from running build/build_engine.sh and
Triton's perf_analyzer on the target GPU.

Run:  python bench/bench_onnx_cpu.py [--out results/bench_cpu.json]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from dlrm import DLRMConfig  # noqa: E402
from export_onnx import export  # noqa: E402


def bench(session: ort.InferenceSession, cfg: DLRMConfig, batch: int,
          warmup: int = 20, iters: int = 300) -> dict:
    rng = np.random.default_rng(0)
    feeds = {
        "dense": rng.standard_normal((batch, cfg.num_dense), dtype=np.float32),
        "embeddings": rng.standard_normal((batch, cfg.num_sparse, cfg.embedding_dim), dtype=np.float32),
    }
    for _ in range(warmup):
        session.run(None, feeds)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        session.run(None, feeds)
        times.append((time.perf_counter() - t) * 1000)
    t = np.array(times)
    p50 = float(np.percentile(t, 50))
    return {"batch": batch, "p50_ms": round(p50, 3), "p99_ms": round(float(np.percentile(t, 99)), 3),
            "requests_per_s_at_p50": round(batch / (p50 / 1000))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()

    cfg = DLRMConfig()
    with tempfile.TemporaryDirectory() as d:
        path = export(pathlib.Path(d) / "dense_scorer.onnx", cfg)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        rows = [bench(sess, cfg, b) for b in (1, 8, 32, 128)]

    result = {
        "what": "ONNX Runtime CPU, dense scorer only (embeddings passed in). Not A10G/TensorRT numbers.",
        "host": {"machine": platform.machine(), "python": platform.python_version(),
                 "onnxruntime": ort.__version__, "intra_op_threads": args.threads},
        "results": rows,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
