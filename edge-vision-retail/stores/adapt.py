"""
One generic model plus a tiny per-store calibration, instead of one model per store.

Every store's cameras see the world differently: lighting, white balance, sensor gain,
lens dirt, compression noise. A model trained on one distribution loses accuracy in a store
whose images look different. The expensive fix is a model per store: collect and label
images at every store, train, version and deploy hundreds of models.

The cheap fix used here needs no labels. A network with batch normalisation (BN) keeps
running statistics of the activations it saw in training. Re-estimating those statistics
on a few hundred UNLABELLED images from a new store re-centres the network on that store's
input distribution (often called "AdaBN"). The per-store artefact is just those statistics:
a few hundred numbers, against the whole model for the per-store approach.

Stores are simulated by applying a different camera response to the test images: a gain,
an offset, sensor noise and a blur, chosen at random per store. Labelled images are used
only to SCORE each method, never by the calibration itself.

Three methods are compared on the same stores:
  generic         the clean-data model as is
  calibrated      generic model + BN statistics re-estimated on 200 unlabelled store images
  store-specific  generic model fine-tuned with 200 LABELLED store images (the costly option)

(Calibrating before the fine-tune makes no difference: training recomputes the batch
statistics from the store data anyway, so both start from the same weights and end equal.)

Calibration images are digits from the training split (the model saw them clean in
training) seen through the store's camera; scoring uses the held-out test split, with
independent sensor noise. The fine-tune baselines use one seed and fixed hyperparameters.

Run:  python stores/adapt.py [--out results/stores.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.cnn import accuracy, train  # noqa: E402
from quantize.ptq import fold_bn, quantized_model  # noqa: E402


def store_camera(seed: int):
    """A random but fixed camera response for one store."""
    g = torch.Generator().manual_seed(1000 + seed)
    u = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=g))   # noqa: E731
    gain, offset, noise, blur = u(0.4, 1.8), u(-0.35, 0.35), u(0.02, 0.12), u(0.0, 0.6)
    kernel = torch.tensor([[blur / 8] * 3, [blur / 8, 1 - blur, blur / 8], [blur / 8] * 3]).view(1, 1, 3, 3)

    def apply(x: torch.Tensor, noise_seed: int = 0) -> torch.Tensor:
        # The sensor noise differs per capture: calibration and scoring use different seeds,
        # so no image is ever scored with the exact noise it was calibrated or trained on.
        gen = torch.Generator().manual_seed(2000 + 7919 * seed + noise_seed)
        y = nn.functional.conv2d(x, kernel, padding=1)
        y = y * gain + offset + noise * torch.randn(y.shape, generator=gen)
        return y.clamp(-0.5, 2.0)                       # sensor saturation

    return apply, {"gain": round(gain, 2), "offset": round(offset, 2), "noise": round(noise, 3), "blur": round(blur, 2)}


@torch.no_grad()
def calibrate_bn(model: nn.Module, unlabelled: torch.Tensor) -> nn.Module:
    """Re-estimate every BN layer's running mean and variance on store images. No labels."""
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None                          # cumulative average over all batches
    m.train()
    for i in range(0, len(unlabelled), 50):
        m(unlabelled[i:i + 50])
    for mod in m.modules():                              # back to the default for any later training
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = 0.1
    return m.eval()


def fine_tune(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int = 30, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            idx = perm[i:i + 32]
            loss = nn.functional.cross_entropy(m(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def bn_stat_count(model: nn.Module) -> int:
    return sum(2 * mod.num_features for mod in model.modules() if isinstance(mod, nn.BatchNorm2d))


def run(stores: int = 12, unlabelled_per_store: int = 200, seed: int = 0, epochs: int = 40,
        store_specific: bool = True, trained=None) -> dict:
    model, (xtr, ytr, xte, yte) = trained or train(seed, epochs)
    rows = []
    for s in range(stores):
        cam, params = store_camera(s)
        # Each store contributes unlabelled training-split images for calibration and is
        # scored on the held-out test split - the two never overlap.
        x_cal = cam(xtr[s * 37 % 900: s * 37 % 900 + unlabelled_per_store], noise_seed=1)
        y_cal = ytr[s * 37 % 900: s * 37 % 900 + unlabelled_per_store]
        x_eval = cam(xte, noise_seed=2)
        calibrated = calibrate_bn(model, x_cal)
        q = quantized_model(fold_bn(calibrated), 8, True, 8, x_cal)          # per-store INT8 ranges
        row = {"store": s, "camera": params,
               "generic": accuracy(model, x_eval, yte),
               "calibrated": accuracy(calibrated, x_eval, yte),
               "calibrated_int8": accuracy(q, x_eval, yte)}
        if store_specific:
            row["store_specific"] = accuracy(fine_tune(model, x_cal, y_cal, seed=s), x_eval, yte)
        rows.append(row)
    keys = [k for k in ("generic", "calibrated", "calibrated_int8", "store_specific") if k in rows[0]]
    mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 4)          # noqa: E731
    worst = lambda k: round(min(r[k] for r in rows), 4)                     # noqa: E731
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "clean_test_accuracy": round(accuracy(model, xte, yte), 4),
        "stores": stores, "unlabelled_images_per_store": unlabelled_per_store,
        "mean_accuracy": {k: mean(k) for k in keys},
        "worst_store_accuracy": {k: worst(k) for k in keys},
        "per_store_artifact_values": {"calibration": bn_stat_count(model), "store_specific_model": n_params},
        "per_store": [{**r, **{k: round(r[k], 4) for k in keys}} for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = run()
    summary = {k: v for k, v in out.items() if k != "per_store"}
    print(json.dumps(summary, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
