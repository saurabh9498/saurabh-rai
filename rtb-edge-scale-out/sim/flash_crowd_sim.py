"""
Flash crowd vs. a static hot cache, and an adaptive promotion policy that recovers.

Each bidding GPU keeps a hot embedding cache built from recent history. A breaking news
event or a live game sends a burst of requests for rows that were cold an hour ago. A
static hot list keeps missing them, every miss is a Redis round trip, and p99 latency
climbs.

Adaptive policy (an LRU-2-style rule): if a cold row is fetched more than K times inside a
W-second window, promote it into a small reserved slice of the cache, evicting the least
recently used promoted row.

This is a simulation with stated assumptions, not a measurement of production traffic.
It shows the mechanism and how the recovery time depends on K, W and the reserved slice.

Run:  python sim/flash_crowd_sim.py
"""
from __future__ import annotations

import json
from collections import OrderedDict, defaultdict, deque

import numpy as np

N_ROWS = 200_000            # simulated row universe (scaled down)
HOT_ROWS = 4_000            # static hot list: top 2% by historical popularity
PROMO_SLOTS = 1_000         # extra slice reserved for promoted rows
QPS = 5_000                 # one bidding replica at design peak
ZIPF_S = 1.418              # calibrated so the 2% static list hits ~98% at this scaled-down row count
FLASH_AT, DURATION = 60, 180
FLASH_SHARE = 0.18          # share of traffic hitting the crowd's rows once it starts
FLASH_ROWS = 600            # distinct rows the crowd touches (never in the static list)
K, W = 5, 10.0              # promote after >K fetches in W seconds


def zipf_sampler(n: int, s: float, rng: np.random.Generator):
    w = 1.0 / np.arange(1, n + 1) ** s
    cdf = np.cumsum(w) / w.sum()
    return lambda k: np.searchsorted(cdf, rng.random(k))


def run(adaptive: bool, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    sample = zipf_sampler(N_ROWS, ZIPF_S, rng)
    flash_rows = np.arange(N_ROWS - FLASH_ROWS, N_ROWS)        # unpopular historically
    flash_pick = zipf_sampler(FLASH_ROWS, 1.0, rng)
    promoted: OrderedDict[int, None] = OrderedDict()
    fetches: dict[int, deque] = defaultdict(deque)
    hit_rate = []
    for sec in range(DURATION):
        ids = sample(QPS)
        if sec >= FLASH_AT:
            n_flash = int(QPS * FLASH_SHARE)
            ids[:n_flash] = flash_rows[flash_pick(n_flash)]
        hits = 0
        for i, row in enumerate(ids):
            t = sec + i / QPS
            if row < HOT_ROWS:
                hits += 1
            elif row in promoted:
                hits += 1
                promoted.move_to_end(row)
            elif adaptive:
                q = fetches[row]
                q.append(t)
                while q and t - q[0] > W:
                    q.popleft()
                if len(q) > K:
                    promoted[row] = None
                    del fetches[row]
                    if len(promoted) > PROMO_SLOTS:
                        promoted.popitem(last=False)
        hit_rate.append(hits / QPS)
    return hit_rate


def summarize(series: list[float]) -> dict:
    before = float(np.mean(series[FLASH_AT - 10:FLASH_AT]))
    first = series[FLASH_AT]
    settled = float(np.mean(series[-20:]))
    target = before - 0.005
    recover = next((s - FLASH_AT for s in range(FLASH_AT, DURATION) if series[s] >= target), None)
    return {"hit_before": round(before, 4), "hit_first_second_of_crowd": round(first, 4),
            "hit_settled": round(settled, 4), "seconds_to_recover_within_0.5pt": recover}


def main() -> None:
    static, adaptive = run(adaptive=False), run(adaptive=True)
    out = {"assumptions": {"rows": N_ROWS, "hot_rows": HOT_ROWS, "promo_slots": PROMO_SLOTS,
                           "qps": QPS, "zipf_s": ZIPF_S, "flash_share": FLASH_SHARE,
                           "flash_rows": FLASH_ROWS, "promote_after": f">{K} fetches in {W:.0f}s"},
           "static": summarize(static), "adaptive": summarize(adaptive)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
