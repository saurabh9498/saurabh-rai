"""
Audience segmentation written once, run on GPU (RAPIDS cuDF) or CPU (pandas).

cuDF implements the pandas API on the GPU, so the same groupby / merge / filter code runs
on either. This module imports cuDF when it is available and falls back to pandas
otherwise. In this repo it runs on pandas; on a machine with RAPIDS installed the same
script runs on the GPU without code changes. For data larger than one GPU's memory, the
same operations go through dask_cudf across GPUs (not shown here).

Segments (recency / frequency / monetary, plus engagement):
    high_value      converted in the last 30 days and spent >= $100 in 90 days
    engaged_viewer  >= 5 video views in 30 days, no recent conversion
    lapsed          active 31-90 days ago, nothing in the last 30
    casual          everyone else with any activity

Run:  python segmentation/segment.py --data data/generated [--out results/segmentation.json]
Writes data/generated/segments.parquet, which the NL-to-SQL demo queries.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import time

try:
    import cudf as xdf          # GPU path
    ENGINE = "cudf"
except ImportError:             # CPU fallback, same API
    import pandas as xdf
    ENGINE = "pandas"


def segment(users, events):
    ev = events
    recent = ev[ev["days_ago"] <= 30]

    conv = ev[ev["event_type"] == "conversion"]
    spend_90 = conv.groupby("user_id")["spend_usd"].sum().rename("spend_90d")
    last_conv = conv.groupby("user_id")["days_ago"].min().rename("days_since_conversion")
    views_30 = (recent[recent["event_type"] == "video_view"]
                .groupby("user_id").size().rename("views_30d"))
    last_seen = ev.groupby("user_id")["days_ago"].min().rename("days_since_seen")

    feats = (users[["user_id", "region", "device"]]
             .merge(last_seen.reset_index(), on="user_id", how="inner")
             .merge(spend_90.reset_index(), on="user_id", how="left")
             .merge(last_conv.reset_index(), on="user_id", how="left")
             .merge(views_30.reset_index(), on="user_id", how="left"))
    feats["spend_90d"] = feats["spend_90d"].fillna(0.0)
    feats["views_30d"] = feats["views_30d"].fillna(0).astype("int64")
    feats["days_since_conversion"] = feats["days_since_conversion"].fillna(999)

    seg = xdf.Series("casual", index=feats.index, dtype="str")
    seg = seg.where(~(feats["days_since_seen"] > 30), "lapsed")
    seg = seg.where(~((feats["views_30d"] >= 5) & (feats["days_since_conversion"] > 30)), "engaged_viewer")
    seg = seg.where(~((feats["days_since_conversion"] <= 30) & (feats["spend_90d"] >= 100)), "high_value")
    feats["segment"] = seg
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/generated"))
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    users = xdf.read_parquet(args.data / "users.parquet")
    events = xdf.read_parquet(args.data / "events.parquet")
    t1 = time.perf_counter()
    feats = segment(users, events)
    counts = feats["segment"].value_counts()
    t2 = time.perf_counter()

    feats[["user_id", "segment", "spend_90d", "views_30d"]].to_parquet(args.data / "segments.parquet", index=False)
    sizes = {str(k): int(v) for k, v in counts.to_dict().items()}
    result = {
        "engine": ENGINE,
        "host": {"machine": platform.machine(), "cpus": os.cpu_count(), "python": platform.python_version()},
        "rows": {"users": int(len(users)), "events": int(len(events)), "segmented_users": int(len(feats))},
        "seconds": {"load": round(t1 - t0, 2), "segment": round(t2 - t1, 2)},
        "segment_sizes": dict(sorted(sizes.items())),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
