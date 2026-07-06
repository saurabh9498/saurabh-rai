# Project Explained: The DynoSim Disaggregation Study
### Why it was done, what it proves, how it works

---

## Part 1 — Why project exists at all

### Why this specific project

Four properties made this the right first move:

1. **It sits on the exact topic of the moment.** Disaggregated prefill/decode serving is
   the central architectural debate in LLM inference right now — it's why NVIDIA built
   Dynamo, why the llm-d project exists. It's the question every inference team is actively debating..

2. **Zero GPU cost.** DynoSim is a simulator: it runs on a laptop CPU in seconds. The
   alternative — real benchmarks — costs hundreds of dollars of cloud GPU time and weeks
   of setup. Project was publishable in one day.

---

## Part 2 — What was actually done (plain language)

**The question:** When you have a fixed budget of GPU workers (we used 4), you can
either let every worker do everything ("aggregated") or dedicate some workers to
*prefill* (processing the prompt) and others to *decode* (generating tokens) —
"disaggregated." Which is better? The honest answer is "it depends" — so we mapped
*what it depends on*.

**The method:** We defined three traffic types that mirror real applications —

| Profile | Prompt size | Output size | Turns | Represents |
|---|---|---|---|---|
| chat | 512 tokens | 256 tokens | 3 | consumer chatbot |
| RAG | 4,096 | 512 | 1 | retrieval-augmented Q&A |
| agentic | 8,000 | 128 | 4 | AI agent re-sending big tool context each turn |

— and simulated each one against four cluster configurations (all-aggregated 4W, and
three disaggregated splits: 3 prefill+1 decode, 2+2, 1+3), at three prefix-cache reuse
levels (20% / 60% / 90%) and three load levels (concurrency 8 / 32 / 128). That's
3 × 4 × 3 × 3 = **108 simulations**, each replaying 200 multi-turn sessions through
NVIDIA's simulator and recording latency and throughput statistics.

**The result:** three findings (detailed in Part 4), the headline being that **the
optimal prefill:decode ratio is not fixed — it shifts with your prefix-cache hit
rate**. That's a genuinely useful, non-obvious capacity-planning insight, and it's the
simulation-level argument for why NVIDIA built *dynamic* disaggregation and the Planner.

**The validation:** the full sweep was run independently on two different machines and
architectures (x86_64 in a cloud sandbox; arm64 via Docker on my MacBook) with
matching results — 7,446 vs 7,452 tok/s on the headline cell. Reproducibility is what
separates an experiment from a demo.

---

## Part 4 — The three findings, and why each matters

**Finding 1 — Chat: aggregated wins everywhere.** All 9 chat cells (every reuse level ×
every load) favored aggregated 4W on *both* throughput and p99 TTFT. Why: 512-token
prompts don't create enough prefill work to justify dedicating workers to it; splitting
the pool means some workers idle while others queue. *SA translation: don't sell
disaggregation to a chatbot customer; you'd be reducing their capacity.*

**Finding 2 — Agentic: disaggregation wins, and the split follows the cache.** With 8K
prompts, prefill dominates — at 20% prefix reuse the best config is 3 prefill + 1
decode. But raise reuse to 90% and most prefill work disappears into cache hits, so the
winner migrates to 2P+2D (12,069 tok/s vs aggregated's 9,719 — +24%). *Translation:
you can't size the P:D ratio without knowing the customer's prefix hit rate — and
because hit rates drift, static splits go stale, which is the business case for Dynamo's
Planner and dynamic disaggregation.*

**Finding 3 — The universal tradeoff: TTFT vs ITL stability.** Aggregated won p99
time-to-first-token in *all 27 comparison cells* — dedicating fewer workers to prefill
means prompts queue. But aggregated pays in **prefill-decode interference**: when an
8K-token prefill lands on a worker mid-decode, every active generation stream on that
worker stalls. At high load, aggregated p99 inter-token latency was 88–184ms; every
disaggregated split held 7–9ms. *SA translation: the question to ask a customer isn't
"do you want lower latency" but "which latency" — first token, or steady streaming? For
multi-turn agents that stream on every turn, steadiness usually wins.*

---

## Part 5 — The technology stack, piece by piece

### 5.1 NVIDIA Dynamo — the platform under study

**What:** NVIDIA's open-source distributed inference-serving framework — the layer that
sits above inference engines (vLLM, SGLang, TensorRT-LLM) and handles routing, worker
management, KV-cache awareness, disaggregation, and autoscaling across a GPU cluster.
**Significance here:** it is both the subject of the study and its strategic anchor —
Dynamo is NVIDIA's answer to the disaggregation era, the product at the center of NVIDIA's inference stack. **How used:** installed via `pip install ai-dynamo` (the Python package bundles
a compiled Rust core, `ai-dynamo-runtime`, that does the heavy lifting).

### 5.2 DynoSim — the simulator (the instrument)

**What:** Dynamo's simulation stack — not a separate product but the surface connecting
the mocker engine, replay runs, sweeps, Router simulation, and Planner simulation. It
answers "which topology should this workload use?" without touching a GPU.
**Significance:** it's what made a one-day, zero-dollar study possible, and NVIDIA's
own framing legitimizes the method: DynoSim "narrows the search space; it does not
replace real-hardware validation" — a sentence we quote in the README as our honest
framing. **How used:** through the offline replay entry point:

```bash
python -m dynamo.replay \
    --input-tokens 8000 --output-tokens 128 \      # agentic shape
    --request-count 200 --turns-per-session 4 \    # 200 sessions × 4 turns
    --shared-prefix-ratio 0.6 --num-prefix-groups 8 \
    --num-workers 4 --replay-mode offline \
    --router-mode kv_router \
    --extra-engine-args '{"block_size":64}' \
    --report-json report.json
```

Offline mode is pure simulation — no NATS, no etcd, no frontend, no GPUs. It prints an
AIPerf-style metrics table and writes a JSON report with ~70 statistics.

### 5.3 The mocker engine — how simulation without GPUs is credible

**What:** a discrete-event model of an inference engine. It doesn't run a neural
network; it *bookkeeps* what an engine would do: continuous batching decisions, KV-cache
block allocation and eviction, prefix-cache hits, preemption, and request lifecycle,
advancing a logical clock. Think flight simulator, not flight. **Significance:** this
is why relative comparisons are trustworthy even though absolute numbers aren't — the
*scheduling dynamics* (queueing, interference, cache behavior) are modeled faithfully;
only the per-step timing is a default model rather than calibrated GPU timing.
**Where it showed up in results:** the ITL-interference finding is pure scheduler
dynamics — exactly what the mocker models best.

### 5.4 Prefill/decode disaggregation — the concept under test

**What:** LLM inference has two phases with opposite hardware personalities. *Prefill*
processes the whole prompt at once — compute-bound, bursty. *Decode* generates one
token at a time — memory-bandwidth-bound, steady. Aggregated serving runs both on every
worker; disaggregated dedicates workers per phase so prefill bursts can't stall decode
streams. **How exercised in code:** the harness's central A/B:

```python
if aggregated:
    kwargs.update(num_workers=4,
                  extra_engine_args=engine_args(block_size=64))
else:  # e.g. 2 prefill + 2 decode
    kwargs.update(
        num_prefill_workers=2, num_decode_workers=2,
        # worker_type is REQUIRED but undocumented — first-user note #1
        prefill_engine_args=engine_args(block_size=64, worker_type="prefill"),
        decode_engine_args=engine_args(block_size=64, worker_type="decode"))
```

### 5.5 Prefix caching & the KV router — the moving part behind Finding 2

**What:** the KV cache stores attention state per token block; if a new request starts
with the same tokens as a cached one (system prompt, tool definitions, conversation
history), those blocks are reused and prefill work shrinks. Dynamo's **KV router**
makes routing cache-aware — it scores workers by how many of the request's blocks they
already hold (visible in logs as "effective cached blocks" and a per-worker "logit")
and routes to maximize reuse. **How used:** `--router-mode kv_router` throughout, and
prefix reuse was a swept dimension via the synthetic generator:

```python
turns_per_session=4,            # multi-turn: turn n+1 waits for turn n
shared_prefix_ratio=0.9,        # 90% of prompt shared within a prefix group
num_prefix_groups=8,            # 8 distinct "system prompts" across sessions
inter_turn_delay_ms=250.0,      # think-time between agent turns
```

This is what makes the traffic *agentic-shaped* rather than generic: long shared
prefixes re-sent every turn is precisely the tool-calling pattern.

### 5.6 The metrics — TTFT, ITL, and why percentiles

**TTFT** (time to first token) = responsiveness; dominated by prefill + queueing.
**ITL** (inter-token latency) = streaming smoothness; its *p99 and standard deviation*
expose interference that averages hide — aggregated's mean ITL looked fine (10.5ms)
while its p99 (138ms) revealed decode stalls. **Throughput (tok/s)** = capacity =
cost-per-token at fixed hardware. Every report carries mean/median/p75/p90/p95/p99/std
for each — we keyed the analysis on p99 because SLAs are written on tails, not means.

### 5.7 The sweep harness (`sweep.py`) — our actual code

**What:** ~150 lines that turn one simulator call into an experiment. Structure:
declare the design space as data, iterate the Cartesian product, call DynoSim's Python
API directly (faster than shelling out — no interpreter restart per run), extract 12
metrics per run, write CSV.

```python
PROFILES = {"chat": (512, 256, 3), "rag": (4096, 512, 1), "agentic": (8000, 128, 4)}
SPLITS   = {"agg_4w": (4, None, None), "3p_1d": (None, 3, 1),
            "2p_2d": (None, 2, 2),     "1p_3d": (None, 1, 3)}
PREFIX_RATIOS = [0.2, 0.6, 0.9]
CONCURRENCY   = [8, 32, 128]

for profile, split, pr, cc in itertools.product(PROFILES, SPLITS,
                                                PREFIX_RATIOS, CONCURRENCY):
    report = run_synthetic_trace_replay(isl, osl, SESSIONS, **kwargs)   # NVIDIA's API
    rows.append(extract_metrics(report))                               # ours
```

The subtle integration detail (first-user note #3): the API rejects plain dicts for
engine args; you must construct the Rust-backed type the way the CLI does internally:

```python
from dynamo.llm import MockEngineArgs
def engine_args(**kw):
    return MockEngineArgs.from_json(json.dumps(kw))
```

**Significance of keeping it small:** the harness is deliberately thin. The credibility
of the numbers rests on NVIDIA's simulator, not our code — a large custom framework
would *reduce* trust, not add it. SA-shaped work is experiment design + interpretation,
and the code volume reflects that.

### 5.8 Analysis & visualization — pandas + matplotlib (`plots.py`)

**What:** pandas for the winner analysis (group the 108 rows by cell, find the best
split per cell — this one-liner produced Finding 2's crossover map):

```python
winners = df.loc[df.groupby(['profile','prefix_ratio','concurrency'])
                   ['output_throughput_tok_s'].idxmax()]
```

matplotlib for the three charts: (1) throughput-vs-p99-TTFT scatter per profile — the
Pareto tradeoff view; (2) throughput vs prefix ratio per split for agentic — the
crossover chart, the study's signature figure; (3) p99 ITL bars on a **log scale** —
needed because the aggregated-vs-disagg gap spans an order of magnitude.

### 5.9 Docker & CPU architectures — the replication layer

**What:** `ai-dynamo-runtime` ships Linux-only compiled wheels, so my Mac needed a
Linux environment: Docker runs one in a lightweight VM. Apple Silicon adds the
architecture dimension — the M-series chip is **arm64**, while most published software
targets **x86_64 (amd64)**; Docker can run amd64 images through slow, imperfect
emulation. **What we learned the hard way (first-user note #4):** the emulated amd64
wheel segfaults inside Dynamo's Rust core at engine init; the native arm64 wheel works
perfectly; and Docker's image cache silently serves a previously-pulled amd64 image
even when you don't ask for it, hiding the cause. The fix:

```bash
docker run --rm -it --platform linux/arm64 -v "$PWD":/work -w /work \
  python:3.12-slim bash -c \
  "pip install ai-dynamo pandas matplotlib && RUST_LOG=off python3 sweep.py"
```

**Significance:** this debugging arc — macOS: no wheels → emulation: segfault → cache
trap → native ARM: success — became a filed upstream bug and is itself
evidence: platform pragmatics is daily work, and ARM fluency matters in a
Grace-CPU world.

---

## Part 6 — The intellectual moves worth internalizing

Beyond the artifacts, project modeled five habits that transfer to every project:

1. **Falsifiable hypotheses stated upfront.** We predicted H1/H2/H3 before running;
   H2 was partially *wrong* (high reuse didn't erase disagg's edge — it changed which
   split wins), and reporting that honestly is what makes the study credible.
2. **Iso-budget comparison.** Every config used exactly 4 workers. Without holding
   budget constant, "disaggregation is faster" is meaningless — you'd be comparing
   different amounts of hardware.
3. **Full-factorial over cherry-picking.** 108 cells means the boundaries of each
   conclusion are visible — you know where aggregated wins, not just that disagg
   "can" win.
4. **Honest framing as a feature.** "Simulation, not benchmark" in the first
   paragraph. Overclaiming is the fastest way to lose an expert audience; scoping
   claims precisely is how you earn one.
5. **Friction is content.** Every undocumented behavior and platform bug became a
   first-user note or an upstream issue — turning debugging cost into community
   contribution.

---

*Companion artifacts: the study itself
(github.com/saurabh9498/saurabh-rai/tree/main/dynosim-disaggregation-study), the
upstream issue draft (dynamo-issue.md).*
