# WS0 Explained: The DynoSim Disaggregation Study
### Why it was done, what it proves, how it works, and how to talk about it

---

## Part 1 — Why WS0 exists at all

### The problem it solves

Your pivot is from Product Manager to Solution Architect in AI infrastructure. The single
biggest objection any hiring panel will raise — spoken or unspoken — is: **"This person
has talked about GPU inference for years, but has he ever touched it himself?"** The
NVIDIA PMM loop ended partly on this axis: deep conceptual fluency, thin hands-on
evidence. A resume can claim anything; interviewers discount claims.

WS0 exists to convert one specific claim from *asserted* to *demonstrated*. Not "I
understand disaggregated serving" but "here is my public, reproducible experiment on
disaggregated serving, with a finding you can check in 25 seconds on your laptop." That
is a categorically different kind of evidence, and almost nobody applying for SA roles
has it.

### Why this specific project

Four properties made this the right first move:

1. **It sits on the exact topic of the moment.** Disaggregated prefill/decode serving is
   the central architectural debate in LLM inference right now — it's why NVIDIA built
   Dynamo, why the llm-d project exists, and what your NVIDIA panel spent whiteboard time
   on. A study here is automatically relevant to every company on your target list.

2. **Zero GPU cost.** DynoSim is a simulator: it runs on a laptop CPU in seconds. The
   alternative — real benchmarks — costs hundreds of dollars of cloud GPU time and weeks
   of setup. WS0 was publishable in one day.

3. **It opens a warm networking channel.** DynoSim's author (Vikram, a 1st-degree
   connection) publicly invited community experiments. Doing one gives you a
   substantive, no-ask reason to message the Dynamo team — infinitely better than
   "I admire your work."

4. **It feeds every other workstream.** The same material becomes: GitHub portfolio
   piece (WS5), a guide chapter on disaggregation (WS2), a LinkedIn post (WS4), and an
   interview story you own end-to-end (WS1). One effort, four outputs.

### What it signals to an SA hiring panel

Solution Architects don't build inference engines — they help customers **choose
configurations**: how many GPUs, aggregated or disaggregated, what worker split, what
SLA is achievable. WS0 is literally a miniature of that job: take a customer-shaped
question ("should my agentic workload use disaggregation?"), design an experiment with
vendor tooling, run it under constraints, and deliver a decision framework with charts.
It also demonstrates the SA's most underrated skill: **honest framing** — the README
explicitly says "simulation, not hardware benchmark," which is exactly the intellectual
honesty that builds customer trust.

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
architectures (x86_64 in a cloud sandbox; arm64 via Docker on your MacBook) with
matching results — 7,446 vs 7,452 tok/s on the headline cell. Reproducibility is what
separates an experiment from a demo.

**The publication:** everything — harness, data, charts, write-up, and four "first-user
notes" documenting undocumented behaviors — is public on your GitHub.

---

## Part 3 — The STAR story (interview-ready)

Use this when asked "tell me about a hands-on project," "how do you approach capacity
planning," or "how do you learn new technology."

**Situation.** "I was preparing for Solution Architect roles in AI inference, and the
hottest architectural question in the space is disaggregated serving — splitting
prefill and decode onto dedicated workers. Everyone quotes the vendor guidance that
disaggregation helps 'at scale with long prompts,' but I couldn't find anyone who had
mapped *where the crossover actually sits* — especially for agentic traffic, which has
a distinctive shape: very long, highly repetitive prompts and short outputs. NVIDIA had
just shipped DynoSim, a discrete-event simulator for exactly these questions, and the
author had publicly invited community experiments. Nobody had published one yet."

**Task.** "I set out to answer one falsifiable question: under a fixed budget of four
workers, at what combination of workload shape, prefix-cache reuse, and load does
disaggregation beat aggregation — and does the answer differ for agentic traffic? I
gave myself three constraints: zero GPU spend, full reproducibility by anyone in under
a minute, and honest framing — simulation results presented as simulation results, not
benchmarks."

**Action.** "I designed a full-factorial sweep: three workload profiles modeled on
chat, RAG, and agentic traffic; four worker splits at iso-budget; three prefix-reuse
ratios; three concurrency levels — 108 runs. I built a ~150-line Python harness on
DynoSim's replay API, using its synthetic multi-turn workload generator with shared-
prefix controls to model agents re-sending tool context. Along the way I hit three
undocumented behaviors — disaggregated mode silently requires a `worker_type` field in
the engine args, the Python API needs `MockEngineArgs` objects rather than dicts, and
the released trace-format name differs from the docs — which I documented as first-user
notes. When I replicated on my own MacBook I also uncovered and root-caused a platform
bug: the amd64 wheel segfaults under Docker emulation on Apple Silicon while the arm64
wheel works, with a nasty image-cache trap that hides the cause. I filed that upstream
with a minimal repro."

**Result.** "Three findings. For chat traffic, aggregated wins everywhere — short
prompts mean worker-splitting just strands capacity. For agentic traffic,
disaggregation wins on throughput, and — the finding I didn't expect — the optimal
prefill:decode ratio *shifts with prefix-cache hit rate*: prefill-heavy 3P+1D wins at
20% reuse, but at 90% reuse decode-heavy 2P+2D takes it, beating aggregated by 24%.
And universally, aggregation wins time-to-first-token while disaggregation wins
inter-token stability by ~20x — p99 ITL of 7–9ms versus 88–184ms — because it
eliminates prefill-decode interference. For multi-turn agents, that stability *is* the
user experience. The whole study reproduces on a laptop CPU in about 25 seconds, ran
identically on two architectures, and is public on my GitHub with the dataset and
harness. It also became my first contribution touchpoint with the Dynamo team — a
filed platform bug and docs gaps rather than a cold 'hi.'"

**One-line close if the interviewer wants the takeaway:** "The right disaggregation
split is a function of your cache hit rate — which is exactly why static topologies
lose to planner-driven dynamic disaggregation as workloads shift."

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
winner migrates to 2P+2D (12,069 tok/s vs aggregated's 9,719 — +24%). *SA translation:
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
Dynamo is NVIDIA's answer to the disaggregation era, the product your target teams
build. **How used:** installed via `pip install ai-dynamo` (the Python package bundles
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

**What:** `ai-dynamo-runtime` ships Linux-only compiled wheels, so your Mac needed a
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
trap → native ARM: success — became a filed upstream bug and is itself SA-grade
evidence: platform pragmatics is daily SA work, and ARM fluency matters in a
Grace-CPU world.

### 5.10 Git & GitHub — the publication layer

**What:** git for version history, GitHub as the public face. **How used:** clone →
copy artifacts → `git add` → `commit` → `push`; the study lives as a folder in your
portfolio monorepo with the README auto-rendering charts inline, linked as the flagship
from the top-level README. **Significance:** publication is what converts private work
into career evidence — the URL is now citable in applications, the Vikram DM, the
upstream issue, and LinkedIn.

---

## Part 6 — The intellectual moves worth internalizing

Beyond the artifacts, WS0 modeled five habits that transfer to every future project and
to the SA job itself:

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
   contribution and networking currency.

---

*Companion artifacts: the study itself
(github.com/saurabh9498/saurabh-rai/tree/main/dynosim-disaggregation-study), the
upstream issue draft (dynamo-issue-draft.md), and the Vikram outreach message
(in chat).*
