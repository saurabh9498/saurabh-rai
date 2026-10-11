#!/usr/bin/env bash
# Regenerate every result in results/ and run the test suite. CPU only, ~1 minute.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python3 sizing/vram_budget.py      | tee results/vram_budget.txt
python3 sizing/latency_budget.py   | tee results/latency_budget.txt
python3 sizing/training_memory.py  | tee results/training_memory.txt
python3 sim/hot_cache_sim.py       | tee results/hot_cache_sim.txt
python3 bench/bench_onnx_cpu.py --out results/bench_cpu.json 2>/dev/null
python3 -m pytest -q tests
