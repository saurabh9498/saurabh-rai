"""
Synthetic audience data: users and their ad/content events.

The shape mirrors an audience-segmentation workload: a users table and a much larger
events table (impressions, clicks, video views, conversions) with timestamps and spend.
Popularity follows a power law, as real engagement data does.

Run:  python data/generate_events.py --users 200000 --events 4000000 --out data/generated
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

EVENT_TYPES = np.array(["impression", "click", "video_view", "conversion"])
EVENT_P = np.array([0.80, 0.12, 0.07, 0.01])
REGIONS = np.array(["northeast", "southeast", "midwest", "southwest", "west"])
DEVICES = np.array(["ctv", "mobile", "desktop", "tablet"])
CATEGORIES = np.array(["sports", "news", "auto", "travel", "finance", "entertainment", "food"])


def generate(n_users: int, n_events: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({
        "user_id": np.arange(n_users, dtype=np.int64),
        "region": rng.choice(REGIONS, n_users),
        "device": rng.choice(DEVICES, n_users, p=[0.25, 0.45, 0.22, 0.08]),
        "signup_days_ago": rng.integers(1, 1500, n_users),
    })
    # Power-law activity: a small share of users generates most events.
    weights = 1.0 / np.arange(1, n_users + 1) ** 0.9
    weights /= weights.sum()
    events = pd.DataFrame({
        "user_id": rng.choice(n_users, n_events, p=weights).astype(np.int64),
        "event_type": rng.choice(EVENT_TYPES, n_events, p=EVENT_P),
        "category": rng.choice(CATEGORIES, n_events),
        "days_ago": rng.integers(0, 90, n_events).astype(np.int32),
    })
    is_conv = events["event_type"].to_numpy() == "conversion"
    events["spend_usd"] = np.where(is_conv, rng.gamma(2.0, 30.0, n_events), 0.0).round(2)
    return users, events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200_000)
    ap.add_argument("--events", type=int, default=4_000_000)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/generated"))
    args = ap.parse_args()
    users, events = generate(args.users, args.events)
    args.out.mkdir(parents=True, exist_ok=True)
    users.to_parquet(args.out / "users.parquet", index=False)
    events.to_parquet(args.out / "events.parquet", index=False)
    print(f"wrote {len(users):,} users and {len(events):,} events to {args.out}")


if __name__ == "__main__":
    main()
