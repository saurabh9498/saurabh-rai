"""
Checkpoint cadence and hot-spare recovery for an hourly retrain on a 32-node H100 cluster.

The retrain must finish inside 60 minutes or the edge serves a stale model. With pipeline
parallelism all 32 nodes step together, so one failed node stops the whole job.

Two ways to choose a checkpoint interval:
  * Young/Daly - minimise expected wasted time: interval ~= sqrt(2 x checkpoint_cost x MTBF).
    With a few seconds of blocking cost and an MTBF of weeks this says "every few hours".
  * Deadline-driven - the job has a hard 60-minute window. What matters is the worst case:
    a failure late in the window must still finish on time. Lost work is up to one full
    interval, so the interval must be small.

Asynchronous GPUDirect Storage writes to an in-rack parallel filesystem make frequent
checkpoints nearly free, which is what lets the deadline argument win.

Run:  python training/recovery_math.py
"""
from __future__ import annotations

import json
import math

TB, GB = 1e12, 1e9


def young_daly_interval_s(checkpoint_cost_s: float, mtbf_s: float) -> float:
    return math.sqrt(2 * checkpoint_cost_s * mtbf_s)


def recovery(checkpoint_interval_min: float, restart_min: float, state_tb: float,
             read_gbps: float) -> dict:
    state_read_s = state_tb * TB / (read_gbps * GB)
    return {
        "state_read_s": round(state_read_s, 1),
        "restart_end_to_end_min": restart_min,
        "lost_work_avg_min": checkpoint_interval_min / 2,
        "lost_work_worst_min": checkpoint_interval_min,
        "total_avg_min": restart_min + checkpoint_interval_min / 2,
        "total_worst_min": restart_min + checkpoint_interval_min,
    }


def fits_deadline(train_min: float, deadline_min: float, worst_recovery_min: float) -> bool:
    return train_min + worst_recovery_min <= deadline_min


def main() -> None:
    gpus = 32 * 8
    out = {
        "cluster": {"nodes": 32, "gpus": gpus, "hbm_tb": gpus * 80 * GB / TB},
        "young_daly_interval_h_for_2wk_mtbf_10s_cost": round(
            young_daly_interval_s(10, 14 * 86_400) / 3600, 1),
        "recovery_with_3min_checkpoints": recovery(3, 2, 1.2, read_gbps=100),
        "recovery_with_60min_checkpoints": recovery(60, 2, 1.2, read_gbps=100),
    }
    # Example: a retrain that needs 50 minutes when nothing fails.
    out["deadline_check_50min_job"] = {
        "3min_checkpoints_fits_one_failure": fits_deadline(50, 60, out["recovery_with_3min_checkpoints"]["total_worst_min"]),
        "60min_checkpoints_fits_one_failure": fits_deadline(50, 60, out["recovery_with_60min_checkpoints"]["total_worst_min"]),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
