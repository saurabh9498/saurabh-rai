#!/usr/bin/env bash
# Generate data, run the segmentation and the demo, and run the tests. CPU only, ~10 s.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python3 data/generate_events.py
python3 segmentation/segment.py --out results/segmentation.json
python3 demo.py | tee results/demo.txt
python3 -m pytest -q tests
