#!/usr/bin/env bash
# Run the demo conversations and the routing sweep, then the tests. CPU only, ~1 minute.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python3 demo.py | tee results/conversations.txt
python3 routing/prefix_routing_sim.py --out results/routing.json | tee results/routing.txt
python3 -m pytest -q tests
