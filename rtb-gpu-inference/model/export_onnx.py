"""
Export the DenseScorer to ONNX with a dynamic batch dimension.

This is step 1 of the build pipeline. Step 2 (build/build_engine.sh) compiles the ONNX
graph to a TensorRT INT8 engine on an A10G. The embedding tables are NOT exported: in
production they are served from the VRAM hot cache and Redis, outside the engine.

Usage:
    python model/export_onnx.py --out artifacts/dense_scorer.onnx
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dlrm import DLRM, DLRMConfig, synthetic_batch  # noqa: E402


def export(out: pathlib.Path, cfg: DLRMConfig | None = None, seed: int = 0) -> pathlib.Path:
    torch.manual_seed(seed)
    model = DLRM(cfg).eval()
    cfg = model.cfg
    dense, sparse = synthetic_batch(cfg, batch=8, seed=seed)
    with torch.no_grad():
        emb = model.embeddings(sparse)
        expected = model(dense, sparse)
        split = model.scorer(dense, emb)
    # The serving split must be exactly the same function as the full model.
    torch.testing.assert_close(split, expected)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.scorer,
        (dense, emb),
        str(out),
        input_names=["dense", "embeddings"],
        output_names=["click_prob"],
        dynamic_axes={"dense": {0: "batch"}, "embeddings": {0: "batch"}, "click_prob": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/dense_scorer.onnx"))
    args = ap.parse_args()
    path = export(args.out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
