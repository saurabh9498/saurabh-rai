#!/usr/bin/env bash
# Train the stand-in model, run the quantization and per-store experiments, the sizing, and
# the tests. CPU only, ~1 minute.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python3 quantize/ptq.py --out results/quantization.json 2>/dev/null >/dev/null
python3 stores/adapt.py --out results/stores.json 2>/dev/null
python3 sizing/edge_sizing.py | tee results/sizing.json
python3 -m pytest -q tests
