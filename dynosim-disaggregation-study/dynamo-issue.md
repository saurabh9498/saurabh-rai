Apple Silicon: amd64 wheel segfaults under Docker emulation at mocker engine init; arm64 wheel works (+ 2 small docs gaps)

https://github.com/ai-dynamo/dynamo/issues/11228

---

## Environment

- **ai-dynamo:** 1.2.1 (pip)
- **Host:** macOS, Apple Silicon (M-series)
- **Container:** Docker Desktop, `python:3.12-slim`
- **Component:** DynoSim offline replay (`python -m dynamo.replay`)

## Summary

`ai-dynamo-runtime` publishes no macOS wheels, so Apple Silicon users naturally reach
for Docker. Under **amd64 emulation** (`--platform linux/amd64`), offline replay
segfaults reproducibly at mocker engine initialization — immediately after the
`KvManager initialized with event sink for DP rank 0` log lines. The **arm64 Linux
wheels work correctly**: the same script completed a 108-run replay sweep with 0
failures in ~20 s.

## Reproduction

Segfaults (amd64 emulation on Apple Silicon):

```bash
docker run --rm -it --platform linux/amd64 python:3.12-slim bash -c \
  "pip install ai-dynamo && python -m dynamo.replay \
     --input-tokens 8000 --output-tokens 128 --request-count 200 \
     --num-workers 4 --replay-mode offline --replay-concurrency 32 \
     --extra-engine-args '{\"block_size\":64}'"
# ...KvManager initialized... then: Segmentation fault
```

Works (native arm64):

```bash
docker run --rm -it --platform linux/arm64 python:3.12-slim bash -c \
  "pip install ai-dynamo && python -m dynamo.replay \
     --input-tokens 8000 --output-tokens 128 --request-count 200 \
     --num-workers 4 --replay-mode offline --replay-concurrency 32 \
     --extra-engine-args '{\"block_size\":64}'"
```

## The cache trap (docs suggestion)

If an amd64 image was ever pulled on the machine, Docker **silently reuses it from
cache** even when no `--platform` flag is passed — so the user keeps hitting the same
segfault with no indication why. The only tell is one warning line:

```
WARNING: The requested image's platform (linux/amd64) does not match the detected
host platform (linux/arm64/v8)
```

Suggested one-line addition to the DynoSim docs: *"On Apple Silicon, run via Docker
and pass `--platform linux/arm64` explicitly."* Given the growing ARM footprint
(Grace, Apple dev machines), this seems worth a sentence. If emulated-amd64 is simply
unsupported, detecting emulation and failing with a clear message at import time
would also solve it.

## Two small docs gaps (happy to split into separate issues)

1. **Disaggregated replay requires `worker_type` inside the engine args.** Passing
   `--num-prefill-workers` / `--num-decode-workers` alone raises
   `num_prefill_workers and num_decode_workers are only used for disagg replay`.
   The working invocation needs
   `--prefill-engine-args '{"worker_type":"prefill", ...}'` and
   `--decode-engine-args '{"worker_type":"decode", ...}'` — currently discoverable
   only through sequential error messages. The DynoSim runs guide doesn't mention it.

2. **Trace-format naming drift:** main-branch docs describe the agentic trace format
   as `agentic_mooncake`, while the released 1.2.1 CLI exposes
   `--trace-format {mooncake,applied_compute_agentic}`.

## Context

Found while running a small community study with DynoSim — a 108-run sweep of
aggregated vs. disaggregated splits across chat/RAG/agentic workload shapes:
https://github.com/saurabh9498/saurabh-rai/tree/main/dynosim-disaggregation-study

DynoSim is excellent; results reproduced across x86_64 and arm64
environments within noise. Happy to provide more detail or test fixes.


**Status (Jul 6):** After maintainer triage, the original single-replay repro was found
NOT to reproduce — the reliable reproducer is the in-process API sweep (sweep.py) under
Rosetta, invariant to VM memory. See the corrected matrix in the issue thread.
