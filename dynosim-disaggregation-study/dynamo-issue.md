Apple Silicon: amd64 wheel segfaults under Docker emulation at mocker engine init; arm64 wheel works (+ 2 small docs gaps)

https://github.com/ai-dynamo/dynamo/issues/11228

**Status (Jul 8, resolved):** Root-caused by the Dynamo team as an Apple Rosetta
translation bug (half-executed 32-byte AVX2 store in `AdmissionQueue::drain_ready`),
not a dynamo bug. Trigger: any multi-turn/prefix flag × `--num-workers > 1` under
Rosetta. Workarounds (arm64 platform flag, or QEMU fallback) being added to official docs.

**Status (Jul 8):** bisected to a one-flag reproducer — --turns-per-session under Rosetta emulation (SIGSEGV; QEMU and native arm64 unaffected). See issue thread.

**Status (Jul 6):** After maintainer triage, the original single-replay repro was found
NOT to reproduce — the reliable reproducer is the in-process API sweep (sweep.py) under
Rosetta, invariant to VM memory. See the corrected matrix in the issue thread.
