"""
Edge fleet layout for the two phases, and the per-replica load each layout implies.

Phase A - repatriation: inference moves from cloud A10Gs to bare-metal edge hubs with
  96 GB RTX PRO 6000 Blackwell GPUs, close to the exchanges.
  Per hub: 2 GPUs bidding (DLRM, independent TP=1 replicas) + 1 GPU context (Llama-3-8B, TP=1).
Phase B - scale-up: request volume doubles; a 70B model generates ad creative.
  Per hub: 2 GPUs bidding (TP=1 each) + 4 GPUs generative (Llama-3-70B, one TP=4 group).

Why independent TP=1 replicas for bidding: each card holds a full FP8 copy of the hot
embedding cache, so one GPU failing (or reloading a new engine) halves the hub's bidding
capacity instead of taking it to zero, and model pushes can be rolled one GPU at a time.

Run:  python sizing/fleet.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Phase:
    name: str
    hubs: int
    bidding_gpus_per_hub: int
    llm_gpus_per_hub: int
    llm_tp: int
    requests_per_day: float
    peak_to_avg: float


# Flash crowds (live sports, breaking news) push traffic to ~2.9x the daily average.
PEAK_TO_AVG = 50_000 / (1.5e9 / SECONDS_PER_DAY)

PHASE_A = Phase("A: repatriation", hubs=5, bidding_gpus_per_hub=2, llm_gpus_per_hub=1, llm_tp=1,
                requests_per_day=1.5e9, peak_to_avg=PEAK_TO_AVG)
PHASE_B = Phase("B: scale-up", hubs=10, bidding_gpus_per_hub=2, llm_gpus_per_hub=4, llm_tp=4,
                requests_per_day=3.0e9, peak_to_avg=PEAK_TO_AVG)


def summarize(p: Phase) -> dict:
    avg_qps = p.requests_per_day / SECONDS_PER_DAY
    peak_qps = avg_qps * p.peak_to_avg
    gpus_per_hub = p.bidding_gpus_per_hub + p.llm_gpus_per_hub
    return {
        "phase": p.name,
        "total_gpus": p.hubs * gpus_per_hub,
        "bidding_gpus": p.hubs * p.bidding_gpus_per_hub,
        "llm_groups": p.hubs * p.llm_gpus_per_hub // p.llm_tp,
        "avg_qps": round(avg_qps),
        "design_peak_qps": round(peak_qps),
        "peak_qps_per_hub": round(peak_qps / p.hubs),
        "peak_qps_per_bidding_replica": round(peak_qps / (p.hubs * p.bidding_gpus_per_hub)),
        "per_replica_if_one_gpu_fails": round(peak_qps / p.hubs),
    }


def main() -> None:
    print(json.dumps([summarize(PHASE_A), summarize(PHASE_B)], indent=2))


if __name__ == "__main__":
    main()
