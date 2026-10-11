#!/usr/bin/env bash
# Regenerate every result in results/ and run the test suite. CPU only, seconds.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python3 sizing/kv_cache.py          | tee results/kv_cache.json
python3 sizing/fleet.py             | tee results/fleet.json
python3 training/recovery_math.py   | tee results/recovery_math.json
python3 sim/flash_crowd_sim.py      | tee results/flash_crowd_sim.json
python3 -m pytest -q tests
