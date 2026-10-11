"""
How much traffic a small VRAM hot cache absorbs under Zipfian (power-law) traffic.

The hot cache holds ~2.3% of embedding rows (sizing/vram_budget.py). Whether that serves
98% of lookups depends entirely on how skewed the traffic is. Ad traffic follows a Zipf
law: the k-th most popular row is requested with probability proportional to 1 / k^s.

This script answers: for a cache holding the top fraction f of N rows, what hit rate does
each skew s give - and what skew does a 98% hit rate imply?

Two methods, which must agree:
  * analytic  - ratio of generalized harmonic numbers H(fN, s) / H(N, s), valid at the
                production row count (billions of rows)
  * Monte Carlo - sample lookups from a Zipf distribution over a smaller N and count hits

Run:  python sim/hot_cache_sim.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sizing"))
from vram_budget import Budget, summarize as vram_summary  # noqa: E402

EULER_GAMMA = 0.5772156649015329


def harmonic(n: float, s: float) -> float:
    """Generalized harmonic number sum_{k=1..n} k^-s (Euler-Maclaurin approximation)."""
    if n < 1:
        return 0.0
    if abs(s - 1.0) < 1e-9:
        return math.log(n) + EULER_GAMMA + 1 / (2 * n)
    # sum_{k=1}^{n} k^-s ~= integral + endpoint terms + first correction
    integral = (n ** (1 - s) - 1) / (1 - s)
    return integral + 0.5 * (1 + n ** -s) + (s / 12) * (1 - n ** (-s - 1))


def analytic_hit_rate(n_rows: float, hot_fraction: float, s: float) -> float:
    return harmonic(hot_fraction * n_rows, s) / harmonic(n_rows, s)


def monte_carlo_hit_rate(n_rows: int, hot_fraction: float, s: float,
                         lookups: int = 2_000_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, n_rows + 1, dtype=np.float64) ** s
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    ranks = np.searchsorted(cdf, rng.random(lookups))       # 0-based popularity rank
    return float(np.mean(ranks < int(hot_fraction * n_rows)))


def skew_for_target(n_rows: float, hot_fraction: float, target: float) -> float:
    lo, hi = 0.5, 2.0
    for _ in range(60):                                      # bisection on s
        mid = (lo + hi) / 2
        if analytic_hit_rate(n_rows, hot_fraction, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    v = vram_summary(Budget())
    n_prod = v["total_rows"]               # 300 GB INT8 / 128 B per row
    f = v["hot_fraction_of_rows"]          # 7 GB hot cache / 300 GB
    skews = [0.8, 0.9, 1.0, 1.05, 1.1, 1.2]

    table = [{"zipf_s": s, "hit_rate": round(analytic_hit_rate(n_prod, f, s), 5)} for s in skews]
    s98 = skew_for_target(n_prod, f, 0.98)
    # Training-side observation: the top 3% of rows carried ~80% of lookups.
    s_train = skew_for_target(n_prod, 0.03, 0.80)

    # Cross-check the analytic formula against simulation on a smaller table.
    n_mc = 1_000_000
    check = [{"zipf_s": s,
              "analytic": round(analytic_hit_rate(n_mc, f, s), 4),
              "monte_carlo": round(monte_carlo_hit_rate(n_mc, f, s), 4)} for s in (0.9, 1.1)]

    out = {"rows": n_prod, "hot_fraction": f, "hit_rate_by_skew": table,
           "skew_needed_for_98pct": round(s98, 3),
           "skew_implied_by_top3pct_80pct": round(s_train, 3),
           "analytic_vs_monte_carlo_1M_rows": check}
    print(json.dumps(out, indent=2))
    print(f"\nWith {f:.2%} of rows cached, a 98% hit rate needs Zipf skew s >= {s98:.2f}.")


if __name__ == "__main__":
    main()
