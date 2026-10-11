"""
Latency budget against a 100 ms exchange timeout, and why the blended p99 is set by
the cold path.

Hot path (every embedding the request needs is in VRAM):
    network in 5 + Triton queue 5 + INT8 compute 25-30 + network out 5  ->  40-45 ms
Cold path adds one Redis round trip (~15 ms). All of a request's misses are fetched in a
single batched call (MGET), so one miss or three cost the same single hop.

Hit rate is per LOOKUP, not per request. Of the 26 sparse features, most tables are small
(device type, ad format, hour of day) and sit entirely in VRAM; they never miss. Only the
huge ID tables (user, device ID) have cold tails. A request misses if ANY of its lookups
into those tables misses:

    P(request goes cold) = 1 - hit_rate ** big_table_lookups

p99 is the latency 1% of requests exceed. If more than 1% of requests go cold, the p99
falls inside the cold path, so the blended p99 is bounded by the cold path's p99.

Run:  python sizing/latency_budget.py
"""
from __future__ import annotations

import json

SLA_MS = 100.0
STAGES_MS = {"network_in": 5.0, "triton_queue": 5.0, "compute_int8": 30.0, "network_out": 5.0}
REDIS_HOP_MS = 15.0
BIG_TABLE_LOOKUPS = 2          # user ID and device ID; the other tables fit fully in VRAM


def cold_request_fraction(lookup_hit_rate: float, lookups: int = BIG_TABLE_LOOKUPS) -> float:
    return 1 - lookup_hit_rate ** lookups


def hit_rate_for_hot_p99(lookups: int = BIG_TABLE_LOOKUPS, pct: float = 0.99) -> float:
    """Per-lookup hit rate needed so that at most (1 - pct) of requests go cold."""
    return pct ** (1 / lookups)


def blended_p99(hot_ms: float, cold_ms: float, cold_fraction: float, pct: float = 0.99) -> float:
    """Conservative bound: each path at its own p99; the cut lands in whichever path holds it."""
    return cold_ms if cold_fraction > (1 - pct) else hot_ms


def summarize(lookup_hit_rate: float = 0.98, lookups: int = BIG_TABLE_LOOKUPS) -> dict:
    hot = sum(STAGES_MS.values())
    cold = hot + REDIS_HOP_MS
    cold_frac = cold_request_fraction(lookup_hit_rate, lookups)
    p99 = blended_p99(hot, cold, cold_frac)
    return {
        "stages_ms": STAGES_MS,
        "hot_path_p99_ms": hot,
        "cold_path_p99_ms": cold,
        "lookup_hit_rate": lookup_hit_rate,
        "big_table_lookups_per_request": lookups,
        "cold_request_fraction": round(cold_frac, 4),
        "blended_p99_upper_bound_ms": p99,
        "headroom_ms": SLA_MS - p99,
        "lookup_hit_rate_needed_for_hot_p99": round(hit_rate_for_hot_p99(lookups), 5),
    }


def main() -> None:
    print(json.dumps(summarize(0.98), indent=2))
    print(json.dumps(summarize(0.98, lookups=26), indent=2))   # if every table could miss
    print(json.dumps(summarize(0.996), indent=2))


if __name__ == "__main__":
    main()
