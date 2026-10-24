# Dynamic Disaggregation and the Planner

*Field Guide to AI Infrastructure — Disaggregated Serving, Post 2 of 2*
*Maps to outline sections 14.6 (Disaggregation) and 15.4 (NVIDIA Dynamo). [Post 1](./01-when-does-disaggregation-pay.md) covered when disaggregation pays at all; this post covers how many workers each side needs, and why that number has to be allowed to move.*

---

## The problem: a split tuned for yesterday's cache

Here is one cell from the [108-run DynoSim study](../../dynosim-disaggregation-study) behind this series. Agentic traffic, 128 concurrent sessions, 20% of each prompt already sitting in the prefix cache. Four simulated GPU workers, arranged four different ways:

| Split | Output throughput | p99 time to first token |
|---|---|---|
| Aggregated (all 4 do both) | 2,595 tok/s | **4.3 s** |
| 3 prefill + 1 decode | **2,728 tok/s** | 5.5 s |
| 2 prefill + 2 decode | 1,898 tok/s | 8.1 s |
| 1 prefill + 3 decode | 961 tok/s | 16.1 s |

Same hardware budget, same traffic. The decode-heavy split delivers a third of the throughput and makes the worst-off user wait sixteen seconds for the first token.

Now the uncomfortable part. 1 prefill + 3 decode is not a bad configuration. At 90% cache reuse and 32 concurrent sessions, it is the *best* split in the whole sweep. Someone who measured their traffic on a day when the cache was hot, picked 1P+3D, and shipped it would have made a correct decision. Then a new feature changes the prompts, the cache hit rate falls to 20%, and the same fleet ends up delivering a third of what the best split would — 65% behind at 128 sessions — without a single line of configuration changing.

Post 1 ended on the claim that the right prefill:decode ratio moves with the prefix-cache hit rate. This post covers why it moves, what a wrong ratio costs, and how NVIDIA Dynamo's Planner turns the ratio from a configuration value into a control loop.

## First principles

*Prefill, decode, TTFT, ITL and the basic disaggregation trade-off are introduced in [Post 1](./01-when-does-disaggregation-pay.md). Five more terms are needed here.*

**The prefill:decode (P:D) ratio.** In a disaggregated deployment, some GPU workers only process prompts (prefill) and some only generate tokens (decode). The P:D ratio is how many of each. "3P+1D" means three prefill workers and one decode worker. It is the single most important sizing decision in a disaggregated fleet, because it decides which pool becomes the bottleneck first.

**The prefix cache.** When a model reads a prompt, it builds a key-value (KV) cache: the intermediate attention state for every token. If the next request begins with the same tokens — the same system prompt, the same conversation history, the same retrieved documents — that state can be reused instead of recomputed. The reused portion is called the prefix.

**The prefix-cache hit rate.** The fraction of a prompt's tokens found already cached. Written *h* below. At *h* = 0.9, nine of every ten prompt tokens are reused, and only one in ten needs real prefill work. In the study, the measured reuse matched the configured shared-prefix ratio closely (0.20, 0.60 and 0.90).

**Iso-budget.** Comparing configurations under the same total number of GPUs. The study holds the fleet at four workers and only changes how they are divided. This isolates the ratio from the question of how big the fleet should be.

**A control loop.** A system that repeatedly measures an output, compares it with a target and adjusts an input — a thermostat. The Planner measures latency and traffic, compares latency with a service-level agreement (SLA), and adjusts worker counts.

## The mechanism, part 1: why the hit rate moves the ratio

Start with the work each pool has to do per second.

A prefill worker's job is the *uncached* part of each prompt. If requests arrive at rate *R* per second, each with an input sequence length (ISL) of *I* tokens and a hit rate *h*, the prefill pool must process:

> prefill tokens per second = *R* × *I* × (1 − *h*)

A decode worker's job is the response. With an output sequence length (OSL) of *O* tokens:

> decode tokens per second = *R* × *O*

Each pool's size is its work divided by what one worker can sustain. Call the per-worker prefill rate *T*ₚ and the per-worker decode rate *T*_d (both in tokens per second, both properties of the model and the GPU). Then:

> P : D = [ *I* × (1 − *h*) / *T*ₚ ] : [ *O* / *T*_d ]

Rearranged, the ratio has two factors:

> P / D = ( *I* / *O* ) × (1 − *h*) × ( *T*_d / *T*ₚ )

The last factor belongs to the hardware and the model. It is fixed once you have picked them, and it is what pre-deployment profiling measures. The first two factors belong to the traffic. The arrival rate *R* has cancelled out entirely: **load changes how big each pool must be, but under this simple model it does not change the ratio. The hit rate does, directly and linearly.**

For the study's agentic profile, *I* = 8,000 and *O* = 128, so *I*/*O* = 62.5. Moving from *h* = 0.2 to *h* = 0.9 takes (1 − *h*) from 0.8 to 0.1. **The required prefill share of the fleet falls by a factor of eight.** No other workload variable in the sweep moves the ratio that far.

Two things the simple model leaves out, and both show up in the data:

- **Granularity.** Four workers can only express three disaggregated ratios: 3:1, 2:2 and 1:3. The optimum is continuous; the fleet is not. Small fleets jump between ratios; large fleets can track the optimum closely.
- **Queueing.** The simple model assumes each pool runs at a steady rate. Under heavy concurrency, a pool that is "big enough on average" still builds a queue. That is why load *does* change the winner in practice, as the next section shows.

## The mechanism, part 2: how the Planner closes the loop

The Planner is the component of [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) that sets replica counts for the prefill and decode pools at runtime. Everything below is from the v1.2.1 release (the version the study used), read from the [design document](https://github.com/ai-dynamo/dynamo/blob/v1.2.1/docs/design-docs/planner-design.md) and [source](https://github.com/ai-dynamo/dynamo/tree/v1.2.1/components/src/dynamo/planner).

It has two scaling modes, which can run together:

- **Throughput-based scaling** sizes both pools from forecast traffic and pre-measured performance curves. It runs every 180 seconds by default and requires profiling data.
- **Load-based scaling** watches per-iteration engine metrics and nudges replica counts up or down by one, on a 5-second default interval. It needs no profiling data. In v1.2.1 those metrics come from vLLM and TensorRT-LLM; SGLang support is wired but not yet shipped in its runtime image.

When both are on, the throughput-based result acts as a floor and load-based scaling handles bursts above it. This post follows the throughput-based path, because that is the one that sets the ratio. One cycle has five steps:

**1. Measure.** The Planner queries Prometheus, the metrics database, against the Dynamo frontend's metrics endpoint. Per interval it reads average time to first token (TTFT), average inter-token latency (ITL), request count, and average ISL and OSL.

**2. Correct.** Before deployment, a profiling run measured what TTFT and ITL *should* be at each load level. The Planner compares reality with that prediction:

```text
prefill_correction = actual_ttft / expected_ttft
decode_correction  = actual_itl  / expected_itl
```

**This is where the hit rate enters.** The Planner never measures *h* directly. But when prefixes are cached, prefill does less work, TTFT comes in below what profiling predicted, and `prefill_correction` drops below 1. The design document lists prefix-cache hits explicitly as one of the effects these factors absorb, alongside request queueing, chunked prefill inside decode, and the gap between average and actual sequence lengths.

**3. Predict.** A load predictor forecasts the next interval's request count, ISL and OSL. Four are available: Constant (next equals current), ARIMA (the default; a statistical time-series model for trends and seasonality), Kalman (a filter suited to bursty traffic) and Prophet (for complex seasonality). Until enough history exists, the Planner falls back to Constant.

**4. Size.** The prefill pool:

```python
predicted_load   = next_requests * next_isl / interval * min(1, prefill_correction)
prefill_replicas = ceil(predicted_load / interpolated_throughput / gpus_per_engine)
```

Read it against the derivation above. `next_requests * next_isl / interval` is *R* × *I*. `min(1, prefill_correction)` stands in for (1 − *h*). Dividing by profiled throughput is dividing by *T*ₚ. The Planner is computing the left side of the ratio, with the hit rate inferred from latency rather than counted.

The decode pool is sized against the ITL target instead: the Planner looks up the highest per-GPU decode throughput that still meets the target at the predicted context length (ISL + OSL/2), then divides total decode work (*R* × *O*) by it.

**5. Act.** The Planner writes the new counts to the deployment through a connector. On Kubernetes it patches the DynamoGraphDeployment resource and lets the operator reconcile. A virtual connector exists for other orchestrators.

**The budget.** By default the Planner sizes each pool to meet its SLA and lets the total float, capped at `max_gpu_budget` (default 8 GPUs). Setting `min_gpu_budget` equal to `max_gpu_budget` pins the total, and the Planner then only redistributes workers between the two pools. That is exactly the iso-budget question the study asks. When the clamp applies, v1.2.1 scales both targets proportionally; the code notes an SLA-pressure-aware split as future work.

![The Planner control loop](./diagrams/02-planner-loop.svg)

## The numbers

All throughput figures below are medians of four complete runs of the sweep (the published CSV plus three re-runs on the same `ai-dynamo` 1.2.1 release, October 2026). Percentages are relative to the best split in that cell. Simulation results, not hardware benchmarks — see Methods.

**Agentic traffic (8,000-token prompts, 128-token responses, 4 turns), throughput by split:**

| Hit rate | Concurrency | Aggregated | 3P+1D | 2P+2D | 1P+3D | Winner |
|---|---|---|---|---|---|---|
| 0.2 | 32 | −2% | **best** | −3% | −51% | 3P+1D |
| 0.6 | 32 | −16% | −17% | **best** | −42% | 2P+2D |
| 0.9 | 32 | −12% | −9% | −4% | **best** | 1P+3D |
| 0.2 | 128 | −5% | **best** | −30% | −65% | 3P+1D |
| 0.6 | 128 | −9% | **best** | −24% | −61% | 3P+1D |
| 0.9 | 128 | −18% | −18% | **best** | −38% | 2P+2D |

![Agentic throughput by split as prefix reuse rises](../../dynosim-disaggregation-study/chart2_agentic_split_vs_prefix.png)

What to read from it:

- **At 32 sessions, the winner walks across the table exactly as the derivation predicts:** prefill-heavy at low reuse, balanced in the middle, decode-heavy at high reuse.
- **At 128 sessions, the walk is delayed.** 3P+1D still wins at 60% reuse, and the decode-heavy split never wins. With 128 concurrent sessions, one prefill worker is a queue even when each prompt is 90% cached: 1P+3D's p99 TTFT at 0.9 reuse is 2.1 s against 1.1 s for 2P+2D. Load does not move the ideal ratio in the simple model; it moves the penalty for undersizing a pool.
- **Aggregated serving never wins agentic throughput in these six cells.** At 90% reuse and 128 sessions, 2P+2D delivered about 12,060 tok/s against 9,716–9,906 for aggregated across the four runs: **+22% to +24%**.
- **The cost of a wrong static split is larger than the gain from the right one.** The best case is +22–24% over aggregated. The worst case for a reasonable-looking static split is −65%.

**What each static split risks, across the three hit rates:**

| Static split | Worst cell at 32 sessions | Worst cell at 128 sessions |
|---|---|---|
| 3P+1D | −17% (at 0.6 reuse) | −18% (at 0.9 reuse) |
| 2P+2D | −4% (at 0.9 reuse) | −30% (at 0.2 reuse) |
| 1P+3D | −51% (at 0.2 reuse) | −65% (at 0.2 reuse) |

At moderate load, 2P+2D is a safe static choice: never more than 4% from the best. At high load, **every static split loses at least 18% somewhere in a realistic range of hit rates.** That is the quantitative case for a planner: not that the best split is much better, but that no fixed split is safe once load is high and the cache hit rate is allowed to drift.

**Planner defaults, v1.2.1:** 180-second throughput adjustment interval; 5-second load-based interval; ARIMA predictor; 30-second startup delay before the first decision; GPU budget cap of 8; minimum of one worker per pool.

## Beyond the simulation: what public measurements show

DynoSim is a simulator. Three public results on real GPUs, published in September and October 2026, point the same way.

**A peer study where the fixed ratio lost every time (independent).** A public blueprint by a Principal Solutions Architect ran three matched Dynamo studies on 16 H200 GPUs, aggregated against disaggregated, same model and image. [Aggregated won all three](https://github.com/bhajian/serving-solutions): 5.5×, 1.62× and 1.34×. The author's own reconciliation is the useful part: *"each study fixed the P:D ratio at a value that was wrong for its workload."* A prefill-heavy run had one prefill worker where aggregated had two that could prefill; a decode-heavy run left its prefill worker idle, and its 1.34× gap is close to the 4:3 ratio of decode-capable workers. Disaggregation still won the metric Post 1 predicted: the worst inter-token gap was 1.1 s disaggregated against 40.4 s aggregated. The largest loss, 5.5×, had a second cause that belongs in this post: the disaggregated follow-up turns recomputed their 256K-token prefixes instead of reusing them. Lose the cache and no ratio saves you.

**A vendor result where moving the ratio paid 2.5× (vendor, MLPerf-submitted).** Crusoe's [MLPerf Inference v6.1 write-up](https://www.crusoe.ai/resources/blog/crusoes-mlperf-inference-v6-1-results-on-nvidia-gb200-nvl72) (16 September 2026) ran Qwen3-VL-235B-A22B on 16 GB200 NVL72 GPUs under Dynamo. Its first Interactive-scenario run used a 75:25 prefill:decode GPU split and measured 15.85 queries/s. Decode was the bottleneck; rebalancing to 50:50 and re-searching the sustainable query rate reached 39.85 queries/s. Two cautions: the gain combines the new split with a re-searched query rate, and the same post describes its final topology as six two-GPU prefill workers and four one-GPU decode workers, which reads as 12:4, not 50:50. Treat the direction as reported and the exact split as unreconciled.

**A small-scale replication of the ITL trade-off (vLLM project).** The vLLM blog's [practical guide to disaggregated serving](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide) (29 September 2026) ran Qwen2.5-7B with ~8K-token prompts on two L40S GPUs. At 0.4 requests/s, collocated p99 inter-token latency jumped to 169 ms while the prefill/decode pair held at 29 ms, never exceeding 52 ms up to 2 requests/s. The price was time to first token: median 2.2 s disaggregated against 0.7 s collocated, mostly because each prompt's ~470 MB of KV cache took about 1.3 s to reach the decode GPU on a PCIe pair without peer-to-peer copies.

**How high hit rates run in practice.** The hit rate *h* is the variable this post turns on, so it helps to know real values. DeepSeek reported [56.3% of a day's 608 billion input tokens hitting its on-disk KV cache](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) (February 2025, mixed production traffic). NVIDIA and PyTorch reported a prefix-cache hit rate above 94.5% on an [agentic benchmark under Dynamo](https://pytorch.org/blog/session-aware-agentic-inference-with-nvidia-dynamo/) (October 2026). Both sit inside the study's range.

## Worked example: following one agentic turn

Take a single turn in the agentic profile: an 8,000-token prompt (system instructions, tool definitions, conversation so far, the last tool result) and a 128-token response (the next tool call).

**At 20% reuse**, 6,400 of the 8,000 prompt tokens need prefill. Against 128 tokens of decode, that is 50 prefill tokens for every decode token. Prefill dominates. The best available split puts three workers on it: 3P+1D, which wins at both load levels.

**At 90% reuse**, only 800 tokens need prefill — 6.25 per decode token, eight times less prefill work for exactly the same traffic. Three prefill workers would now spend most of their time idle while one decode worker carries every generated token. The derivation says the fleet should swing toward decode. At 32 sessions it swings all the way to 1P+3D. At 128 sessions it stops at 2P+2D, because a single prefill worker serving 128 concurrent sessions still queues.

**What the Planner sees as this happens.** Nothing in its inputs says "the hit rate rose." Average ISL is still 8,000. What changes is TTFT: prefill is doing an eighth of the work profiling assumed, so observed TTFT drops well below the prediction and `prefill_correction` falls. The next sizing pass multiplies the prefill load estimate by that smaller factor, asks for fewer prefill replicas, and — if the budget is pinned — hands the freed GPUs to decode. The ratio follows the cache, one adjustment interval behind it.

Agentic workloads are where the hit rate swings hardest: each turn re-sends a growing history, and a new session or a changed system prompt resets what is cached.

## Trade-offs, and when not to use a planner

**Chat-shaped traffic.** Post 1's finding stands: for short prompts and longer responses, aggregated serving won every cell in the study. A planner that tunes the ratio of a disaggregated fleet does not help a workload that should not be disaggregated. In v1.2.1 the load-based mode can scale aggregated deployments too, but that is replica scaling, not ratio tuning.

**Stable traffic.** If the hit rate and the ISL/OSL mix genuinely do not move, measure once, pick the split and save the operational surface. Balanced splits are forgiving at moderate load.

**Profiling is a prerequisite.** Throughput-based scaling needs pre-deployment profiling for each model, GPU and parallelism combination, and the profile goes stale when any of them changes.

**The loop is slower than some traffic.** At the 180-second default, a burst that lasts a minute is over before the throughput path reacts. The load-based mode exists for that, with its own constraint: one replica of change per 5-second step, and only on backends that emit the per-iteration metrics (vLLM and TensorRT-LLM in this release).

**Scaling up pays the full cold start.** A new worker has to be scheduled, pull its container, load weights and register before it helps. If that takes longer than the adjustment interval, decisions overlap. The design document says so directly and recommends shorter intervals only for workloads with fast model loading. Weight loading is the part that is improving fastest. An [llm-d benchmark of ModelExpress](https://github.com/llm-d/llm-d/tree/v1.0.0/guides/modelexpress-p2p/benchmark-results/gke-a3u-roce-gpt-oss-120b.md) (August 2026), which copies weights GPU-to-GPU from a running worker, loaded 33.51 GB per rank in 0.70 s against 21–22 s from warm local NVMe on H200 nodes with 400 Gb/s RoCE. How fast a worker can change roles bounds how fast the ratio can move. (Cold starts get their own leaf in Part 16.)

**Chunked prefill is the alternative to compare against.** Splitting long prompts into chunks that interleave with decode attacks the same interference without a KV handoff. It is not free, and the right chunk size depends on the hardware. One [published course module](https://github.com/yiqiao-yin/deepspeed-course/pull/61) (October 2026) simulated a single 8K-token prompt landing on two live streams, using constants measured on a laptop GPU with a 0.6B-parameter model. Chunking at 256 tokens cut the worst decode stall from 310 ms to 116 ms but raised p99 time to first token 2.86× over the unchunked baseline; disaggregation cost 10% on the same metric but used more GPU-seconds. Its sizing rule: chunk = (decode step time − fixed per-pass overhead) ÷ per-token prefill time.

## How it fails in production

**Flapping.** *Symptom:* replica counts oscillating every interval; scaling events that never finish before the next one starts. *Cause:* adjustment interval shorter than real scaling time, or correction factors swinging during a traffic ramp. *Check first:* time from a scale-up decision to the new worker serving its first request, against the configured interval. The Planner logs a warning when decisions pile up; it does not queue them.

**Poisoned correction factors.** *Symptom:* the Planner sizes badly right after a deployment or restart, then slowly recovers. *Cause:* cold caches and warm-up effects distort observed TTFT and ITL, and the factors faithfully learn the distortion. *Check first:* the factor values over the first few intervals. The documented remedy is `--no-correction` while cold-start artifacts dominate.

**One number carrying two opposite signals.** *Symptom:* TTFT climbing under burst load while the prefill pool does not grow. *Cause (follows from the formula, not from documented incidents):* cache hits push `prefill_correction` down and queueing pushes it up, but the `min(1, …)` clamp ignores values above 1, so queueing cannot raise the prefill estimate through this path. *Check first:* whether load-based scaling, which exists for bursts, is enabled and supported on your backend.

**Averages hiding two workloads.** *Symptom:* SLA misses on one class of request while aggregate metrics look healthy. *Cause:* throughput-based sizing uses average ISL and OSL, which the design document lists as a known limitation for bimodal or heavy-tailed traffic. An agentic fleet that also serves short chat turns is exactly that shape. *Check first:* the ISL histogram, not the mean. Consider separate deployments per traffic class.

**A KV handoff far slower than the link.** *Symptom:* disaggregated TTFT much worse than the arithmetic predicts. *Cause:* the transfer path, not the link speed. A 10K-token prompt on a 70B-class model carries about 3 GB of KV cache, roughly 65 ms at a 400 Gb/s line rate. In the vLLM guide above, a 470 MB cache took about 1.3 s on a PCIe pair without peer-to-peer copies, and the authors attribute most of that to overhead around the copy. A transfer library that has quietly fallen back from RDMA to TCP produces the same symptom. *Check first:* measured KV transfer time per request against size ÷ line rate, and which transport the transfer library actually selected.

**Disaggregation that loses the prefix cache.** *Symptom:* follow-up turns as slow as first turns. *Cause:* the decode side or the router does not reuse prefixes across the split — the peer study above saw its 256K-token prefixes recomputed every turn, with the cause undiagnosed and decode-side caching off by default in that engine's disaggregated mode. *Check first:* per-turn cache hits in both modes before comparing them.

**Scaling that disturbs the cache it depends on.** *Symptom:* hit rate dropping right after scale events. *Cause (my reasoning, not documented behaviour):* a new worker starts with an empty cache, and requests routed to it miss prefixes that were warm elsewhere. The lower hit rate raises the prefill estimate on the next pass, which can trigger more scaling. Dynamo's KV-aware router routes on cached prefixes, which should limit this. *Check first:* hit rate per worker before and after a scale event.

## The interview version

*About 90 seconds, spoken.*

"In disaggregated serving, you split GPUs into prefill workers that read the prompt and decode workers that generate the response, and the ratio between them is the main sizing decision. The ratio depends mostly on how much of each prompt is already in the prefix cache. Prefill only has to process the uncached tokens, so when the cache hit rate goes from 20% to 90%, prefill work drops eightfold for the same traffic.

I measured this in a 108-run DynoSim study. At low reuse, three prefill and one decode was best. At high reuse, the winner moved toward decode-heavy splits. The bigger finding was the downside: a decode-heavy split that's best on a hot cache fell 65% behind the best split when reuse dropped to 20%, and at high load every static split lost at least 18% somewhere in the range.

That's the case for Dynamo's Planner. Every three minutes by default, it reads TTFT, ITL and traffic from Prometheus, compares latency with what profiling predicted, and resizes both pools. It never measures the hit rate directly — cache hits show up as TTFT coming in below prediction, which lowers the prefill estimate. So the ratio follows the cache, one interval behind. The limits are that it needs profiling up front, it works from averages, and it can't react faster than a new worker can start."

## Methods, honestly

The study's numbers are discrete-event simulation results from [NVIDIA Dynamo's DynoSim](https://docs.nvidia.com/dynamo/user-guides/dynosim) mocker with its default timing model — **not hardware benchmarks.** The relative behaviour across configurations is the finding; the absolute numbers are not. The study compares fixed splits and does not run the Planner; the link between the two is the derivation and the Planner's published formulas, not a closed-loop experiment. The sweep reproduces in about 25 seconds on a CPU; across four runs, 106 of 108 cells varied by less than 1%. Code and data are [in the study folder](../../dynosim-disaggregation-study). The public results in this post are each labelled with who measured them.

## Further reading

- **Dynamo Planner design document, v1.2.1** — architecture, the sizing formulas above, known limitations. [GitHub](https://github.com/ai-dynamo/dynamo/blob/v1.2.1/docs/design-docs/planner-design.md)
- **Dynamo Planner source, v1.2.1** — scaling modes, defaults (`config/defaults.py`) and budget clamping (`core/budget.py`). [GitHub](https://github.com/ai-dynamo/dynamo/tree/v1.2.1/components/src/dynamo/planner)
- **Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024** — the case for separating the phases and sizing each against its own latency target. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- **Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024** — phase splitting across machine pools, including mixed hardware. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
- **vLLM, "Taking vLLM Apart: A Practical Guide to Disaggregated Serving" (29 Sep 2026)** — measured ITL and TTFT trade-off on two GPUs, plus KV-size arithmetic. [vllm.ai](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide)
- **bhajian/serving-solutions, chapter 12 (2026)** — three matched aggregated-vs-disaggregated Dynamo studies on 16 H200s, with the reconciliation quoted above. [GitHub](https://github.com/bhajian/serving-solutions)
- **Dhanani, Li and Chung, "Session-Aware Agentic Inference with NVIDIA Dynamo" (8 Oct 2026)** — how the router, engine and KV-cache manager cooperate on agentic traffic. [PyTorch blog](https://pytorch.org/blog/session-aware-agentic-inference-with-nvidia-dynamo/)
- **When Does Disaggregation Pay? — the DynoSim study** — the 108-run sweep this post draws on. [GitHub](../../dynosim-disaggregation-study)

---

*Previous: [Post 1 — When Does Disaggregation Pay?](./01-when-does-disaggregation-pay.md)*
