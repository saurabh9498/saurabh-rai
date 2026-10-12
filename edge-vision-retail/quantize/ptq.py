"""
Post-training quantization (PTQ): per-tensor vs per-channel weight scales.

Deployment first folds each batch-norm layer into the convolution before it:

    w' = w * g / sqrt(var + eps)              (one factor per output channel)
    b' = (b - mean) * g / sqrt(var + eps) + beta

Folding multiplies each output channel by a different factor, so channels end up with very
different weight ranges. Quantization then maps floats to 8-bit integers with a scale:

    per-tensor   one scale for the whole layer   = max|w| over all channels / 127
    per-channel  one scale per output channel    = max|w| of that channel / 127

With one scale per tensor, the channel with the largest range sets the step size for every
channel, and small-range channels lose most of their resolution. Per-channel scales give
each channel its own step. Activations are quantized per tensor (unsigned 8-bit after ReLU)
with ranges taken from a calibration batch.

Everything here is simulated ("fake quantization": quantize, then dequantize in float), which
is how quantization accuracy is evaluated before compiling for an accelerator.

Run:  python quantize/ptq.py [--out results/quantization.json]
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


def fold_bn(model: nn.Module) -> nn.Module:
    """Return a copy with every Conv2d+BatchNorm2d pair merged into one Conv2d."""
    m = copy.deepcopy(model).eval()
    layers = list(m.features)
    out = []
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, nn.Conv2d) and i + 1 < len(layers) and isinstance(layers[i + 1], nn.BatchNorm2d):
            bn = layers[i + 1]
            factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            conv = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                             layer.stride, layer.padding, bias=True)
            with torch.no_grad():
                conv.weight.copy_(layer.weight * factor[:, None, None, None])
                b = layer.bias if layer.bias is not None else torch.zeros_like(bn.running_mean)
                conv.bias.copy_((b - bn.running_mean) * factor + bn.bias)
            out.append(conv)
            i += 2
        else:
            out.append(layer)
            i += 1
    m.features = nn.Sequential(*out)
    return m


def quantize_weight(w: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        maxabs = w.abs().flatten(1).max(1).values.clamp(min=1e-12).view(-1, *([1] * (w.dim() - 1)))
    else:
        maxabs = w.abs().max().clamp(min=1e-12)
    scale = maxabs / qmax
    return (w / scale).round().clamp(-qmax, qmax) * scale


class ActQuant(nn.Module):
    """Asymmetric 8-bit activation quantizer (scale + zero point); range set by calibration.

    After a ReLU the range starts at zero, so this is plain unsigned 8-bit. The input layer
    can see negative values (camera offset), which an unsigned-only quantizer would clip.
    """

    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits, self.lo, self.hi, self.calibrating = bits, None, None, False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            lo, hi = x.detach().min().clamp(max=0), x.detach().max()
            self.lo = lo if self.lo is None else torch.minimum(self.lo, lo)
            self.hi = hi if self.hi is None else torch.maximum(self.hi, hi)
            return x
        if self.hi is None:
            return x
        levels = 2 ** self.bits - 1
        scale = (self.hi - self.lo).clamp(min=1e-12) / levels
        zero = (-self.lo / scale).round()
        return ((x / scale).round() + zero).clamp(0, levels).sub(zero) * scale


def quantized_model(folded: nn.Module, w_bits: int, per_channel: bool, a_bits: int | None,
                    calib: torch.Tensor) -> nn.Module:
    q = copy.deepcopy(folded).eval()
    with torch.no_grad():
        for mod in q.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.weight.copy_(quantize_weight(mod.weight, w_bits, per_channel))
    if a_bits:
        layers = []
        for layer in q.features:
            layers.append(layer)
            if isinstance(layer, nn.ReLU):
                layers.append(ActQuant(a_bits))
        q.features = nn.Sequential(ActQuant(a_bits), *layers)
        observers = [m for m in q.modules() if isinstance(m, ActQuant)]
        for o in observers:
            o.calibrating = True
        with torch.no_grad():
            q(calib)
        for o in observers:
            o.calibrating = False
    return q


def layer_errors(folded: nn.Module, bits: int) -> list[dict]:
    """Relative weight error per layer, and how unequal the channel ranges are."""
    rows = []
    for name, mod in folded.named_modules():
        if isinstance(mod, nn.Conv2d):
            w = mod.weight.detach()
            ranges = w.abs().flatten(1).max(1).values
            err = {}
            for pc in (False, True):
                wq = quantize_weight(w, bits, pc)
                err["per_channel" if pc else "per_tensor"] = float(((wq - w) ** 2).sum() / (w ** 2).sum())
            rows.append({"layer": name, "channels": w.shape[0],
                         "channel_range_ratio": round(float(ranges.max() / ranges.min()), 1),
                         "rel_sq_error_per_tensor": f"{err['per_tensor']:.2e}",
                         "rel_sq_error_per_channel": f"{err['per_channel']:.2e}"})
    return rows


def range_sweep(bits: int = 8, channels: int = 64, fan_in: int = 288, seed: int = 0) -> list[dict]:
    """Weight error vs how unequal channel ranges are, on synthetic layers.

    The ratio is between the largest and smallest channel's standard deviation. This toy
    model's folded layers only differ about 2x between channels. Production
    backbones, especially depthwise convolutions, often differ by 10-100x. The sweep shows
    the per-tensor error growing with that spread while the per-channel error stays flat.
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    for ratio in (1, 4, 16, 64, 256):
        stds = torch.logspace(0, torch.log10(torch.tensor(float(ratio))), channels)
        w = torch.randn(channels, fan_in, generator=g) * stds[:, None]
        row = {"channel_range_ratio": ratio}
        for pc in (False, True):
            wq = quantize_weight(w, bits, pc)
            # Relative SQUARED error, measured channel by channel so small channels are not
            # hidden by big ones. Its square root is the relative RMS error.
            per_ch = ((wq - w) ** 2).sum(1) / (w ** 2).sum(1)
            key = "per_channel" if pc else "per_tensor"
            row[f"mean_channel_rel_error_{key}"] = f"{float(per_ch.mean()):.2e}"
            row[f"weights_rounded_to_zero_{key}"] = round(float((wq == 0).float().mean()), 3)
        rows.append(row)
    return rows


def run(seed: int = 0, epochs: int = 40, trained=None) -> dict:
    model, (xtr, _, xte, yte) = trained or train(seed, epochs)
    folded = fold_bn(model)
    calib = xtr[:256]
    res = {"fp32": accuracy(model, xte, yte), "fp32_bn_folded": accuracy(folded, xte, yte)}
    for bits in (8, 4):
        for pc in (False, True):
            q = quantized_model(folded, bits, pc, 8, calib)
            res[f"w{bits}_{'per_channel' if pc else 'per_tensor'}_a8"] = accuracy(q, xte, yte)
    return {"test_images": len(xte), "accuracy": {k: round(v, 4) for k, v in res.items()},
            "layer_errors_int8": layer_errors(folded, 8), "layer_errors_int4": layer_errors(folded, 4),
            "int8_error_vs_channel_range_ratio": range_sweep(8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = run()
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
