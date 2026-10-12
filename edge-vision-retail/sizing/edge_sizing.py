"""
Two sizing questions for an in-store accelerator, with every input stated.

1. Do four vision models fit on one device, in memory and in throughput?
2. How much local disk buys 72 hours of offline operation?

The inputs are ASSUMPTIONS for illustration - typical shapes for small INT8 vision models
and a mid-range edge accelerator, not vendor specifications or measurements. The point is
the method: list every model's memory and per-camera compute, add them up against the
device, and keep headroom. Replace the inputs with measured numbers for a real device.

Run:  python sizing/edge_sizing.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from edge.buffer import bytes_per_hour  # noqa: E402

MB = 2**20                         # "MB" in outputs means MiB


@dataclass(frozen=True)
class VisionModel:
    name: str
    weights_mb: float              # INT8 weights
    activations_mb: float          # working memory per in-flight batch
    tops_per_inference: float      # INT8 tera-operations per frame (x1e12)
    fps_per_camera: float          # how often it must run per camera


@dataclass(frozen=True)
class Device:
    memory_mb: float               # accelerator memory available to models (assumption)
    sustained_tops: float          # usable INT8 TOPS after real-world efficiency (assumption)
    runtime_overhead_mb: float = 512


# A detector of ~30 billion multiply-accumulates per 640x640 frame is ~0.06 tera-ops (2 ops/MAC).
MODELS = (
    VisionModel("person detector", 12, 96, 0.060, 15),
    VisionModel("shelf product classifier", 8, 64, 0.020, 2),
    VisionModel("queue / zone counter", 6, 48, 0.030, 5),
    VisionModel("re-identification embedder", 10, 64, 0.010, 15),
)
DEVICE = Device(memory_mb=8 * 1024, sustained_tops=40.0)
CAMERAS = 12
HEADROOM = 0.30                    # keep 30% free for bursts and model updates (A/B slots)


def packing(models=MODELS, device=DEVICE, cameras=CAMERAS, in_flight_batches: int = 2) -> dict:
    # Weights are held twice during an update (A/B), activations per in-flight batch.
    mem = device.runtime_overhead_mb + sum(2 * m.weights_mb + in_flight_batches * m.activations_mb for m in models)
    tops = sum(m.tops_per_inference * m.fps_per_camera * cameras for m in models)
    return {
        "models": len(models), "cameras": cameras,
        "memory_mb_needed": round(mem), "memory_mb_available": device.memory_mb,
        "memory_utilisation": round(mem / device.memory_mb, 3),
        "tops_needed": round(tops, 2), "tops_available": device.sustained_tops,
        "compute_utilisation": round(tops / device.sustained_tops, 3),
        "fits_with_headroom": mem <= device.memory_mb * (1 - HEADROOM) and tops <= device.sustained_tops * (1 - HEADROOM),
    }


# Events one store produces per hour (assumptions): (kind, events/hour, payload bytes)
STORE_RATES = (
    ("frame_meta", CAMERAS * 60 * 60 * 1, 120),   # one summary per camera per second
    ("count", CAMERAS * 60, 400),                  # per-zone counts per camera per minute
    ("alert", 20, 1_000),
)


def outage_budget(hours: float = 72, rates=STORE_RATES) -> dict:
    per_hour = bytes_per_hour(rates)
    return {"bytes_per_hour_mb": round(per_hour / MB, 2), "hours": hours,
            "disk_needed_mb": round(per_hour * hours / MB), "with_2x_margin_mb": round(2 * per_hour * hours / MB)}


def max_cameras(models=MODELS, device=DEVICE) -> int:
    """Most cameras one device can serve with all models running and the headroom kept."""
    n = 0
    while packing(models, device, n + 1)["fits_with_headroom"]:
        n += 1
    return n


def main() -> None:
    print(json.dumps({"assumptions": {"device": asdict(DEVICE), "models": [asdict(m) for m in MODELS],
                                      "cameras": CAMERAS, "headroom": HEADROOM},
                      "packing": packing(), "max_cameras_per_device": max_cameras(),
                      "max_cameras_detector_only": max_cameras(MODELS[:1]),
                      "offline_72h": outage_budget()}, indent=2))


if __name__ == "__main__":
    main()
