#!/usr/bin/env python3
"""
WS0: When Does Disaggregation Pay?
DynoSim sweep — aggregated vs disaggregated prefill/decode under iso-GPU budget,
across workload profiles (chat / RAG / agentic), prefix-reuse ratios, and load levels.

Simulation study using NVIDIA Dynamo's DynoSim (mocker engine, default timing).
Relative behavior is the finding; absolute numbers are not hardware benchmarks.
"""
import csv
import io
import itertools
import json
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

from dynamo.llm import MockEngineArgs
from dynamo.replay.api import run_synthetic_trace_replay


def engine_args(**kw):
    """Build MockEngineArgs the same way the CLI does (from_json)."""
    return MockEngineArgs.from_json(json.dumps(kw))

BLOCK_SIZE = 64
SESSIONS = 200
PREFIX_GROUPS = 8
INTER_TURN_DELAY_MS = 250.0
ROUTER_MODE = "kv_router"
GPU_BUDGET = 4  # iso-budget: total simulated workers

PROFILES = {
    # name: (input_tokens, output_tokens, turns_per_session)
    "chat":    (512,  256, 3),   # short prompts, chatty decode
    "rag":     (4096, 512, 1),   # big stuffed context, single-shot
    "agentic": (8000, 128, 4),   # long re-sent context, short tool-call decode
}

SPLITS = {
    # name: (num_workers_agg, prefill, decode)  — agg uses num_workers, disagg uses p/d
    "agg_4w":  (4, None, None),
    "3p_1d":   (None, 3, 1),
    "2p_2d":   (None, 2, 2),
    "1p_3d":   (None, 1, 3),
}

PREFIX_RATIOS = [0.2, 0.6, 0.9]
CONCURRENCY = [8, 32, 128]

FIELDS = [
    "profile", "split", "prefix_ratio", "concurrency",
    "isl", "osl", "turns",
    "mean_ttft_ms", "p99_ttft_ms",
    "mean_itl_ms", "p99_itl_ms", "std_itl_ms",
    "mean_e2e_latency_ms", "p99_e2e_latency_ms",
    "output_throughput_tok_s", "request_throughput_rps",
    "prefix_cache_reused_ratio", "completed_requests", "wall_time_ms",
]


def report_to_dict(report):
    if isinstance(report, dict):
        return report
    for attr in ("to_dict", "as_dict"):
        if hasattr(report, attr):
            return getattr(report, attr)()
    if hasattr(report, "to_json"):
        return json.loads(report.to_json())
    return json.loads(json.dumps(report, default=lambda o: o.__dict__))


def run_one(profile, split, prefix_ratio, concurrency):
    isl, osl, turns = PROFILES[profile]
    agg_workers, n_prefill, n_decode = SPLITS[split]
    kwargs = dict(
        replay_concurrency=concurrency,
        replay_mode="offline",
        router_mode=ROUTER_MODE,
        arrival_interval_ms=1.0,
        turns_per_session=turns,
        shared_prefix_ratio=prefix_ratio,
        num_prefix_groups=PREFIX_GROUPS,
        inter_turn_delay_ms=INTER_TURN_DELAY_MS if turns > 1 else 0.0,
    )
    if agg_workers is not None:
        kwargs.update(
            num_workers=agg_workers,
            extra_engine_args=engine_args(block_size=BLOCK_SIZE),
        )
    else:
        kwargs.update(
            num_prefill_workers=n_prefill,
            num_decode_workers=n_decode,
            # NOTE: worker_type is required by the pip release but undocumented
            prefill_engine_args=engine_args(block_size=BLOCK_SIZE, worker_type="prefill"),
            decode_engine_args=engine_args(block_size=BLOCK_SIZE, worker_type="decode"),
        )
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        report = run_synthetic_trace_replay(isl, osl, SESSIONS, **kwargs)
    r = report_to_dict(report)
    row = {
        "profile": profile, "split": split,
        "prefix_ratio": prefix_ratio, "concurrency": concurrency,
        "isl": isl, "osl": osl, "turns": turns,
    }
    for f in FIELDS[7:]:
        row[f] = r.get(f)
    return row


def main():
    combos = list(itertools.product(PROFILES, SPLITS, PREFIX_RATIOS, CONCURRENCY))
    print(f"Sweep: {len(combos)} runs "
          f"({len(PROFILES)} profiles x {len(SPLITS)} splits x "
          f"{len(PREFIX_RATIOS)} prefix ratios x {len(CONCURRENCY)} concurrency levels)")
    rows, failures = [], []
    t0 = time.time()
    for i, (profile, split, pr, cc) in enumerate(combos, 1):
        tag = f"{profile}/{split}/prefix={pr}/cc={cc}"
        try:
            t = time.time()
            row = run_one(profile, split, pr, cc)
            rows.append(row)
            print(f"[{i:3d}/{len(combos)}] {tag:44s} ok "
                  f"({time.time()-t:5.1f}s)  tok/s={row['output_throughput_tok_s']:.0f}  "
                  f"p99_itl={row['p99_itl_ms']:.1f}ms")
        except Exception as e:
            failures.append((tag, str(e)))
            print(f"[{i:3d}/{len(combos)}] {tag:44s} FAILED: {e}")
    out = "results.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} ok, {len(failures)} failed, {time.time()-t0:.0f}s total -> {out}")
    if failures:
        with open("failures.log", "w") as fh:
            fh.write("\n".join(f"{t}: {e}" for t, e in failures))


if __name__ == "__main__":
    main()
