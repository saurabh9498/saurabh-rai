# When Does Disaggregation Pay?
 
*Field Guide to AI Infrastructure — Disaggregated Serving, Post 1 of 2*
*Maps to outline sections 14.6 (Disaggregation) and 15.4 (NVIDIA Dynamo) — this post is the vendor-neutral half; Post 2 covers dynamic disaggregation and the Planner.*
 
---
 
If you've read anything about LLM serving in the last year, you've heard the pitch: split your GPUs into separate prefill and decode pools, and everything gets better. Disaggregation is the architecture behind NVIDIA Dynamo, the reason llm-d exists, and the default answer in half the system-design conversations happening in this industry right now.
 
The honest answer is less tidy: **whether disaggregation pays depends almost entirely on the shape of your workload** — and for one of the most common workload shapes on the planet, it doesn't pay at all.
 
I ran a 108-configuration simulation sweep to find the crossover: chat, RAG, and agentic traffic profiles, four worker splits under an identical GPU budget, three prefix-reuse rates, three load levels. The full study — code, data, charts, all reproducible on a CPU in about 25 seconds — is [on GitHub](https://github.com/saurabh9498/saurabh-rai/tree/main/dynosim-disaggregation-study). This post is what the results mean if you're the person deciding how to serve a model.
 
The short version: chat traffic and agentic traffic want *opposite* architectures. Everything else in this post is the why.
 
## The three-minute primer
 
*(Each of these gets a full treatment in Part 13 — Inference Mechanics. This is just enough to read the charts.)*
 
Every LLM request has two phases that could hardly be more different. **Prefill** processes your entire prompt in one shot — thousands of tokens crunched in parallel, saturating the GPU's compute units. **Decode** then generates the response one token at a time, and each step is mostly waiting on memory bandwidth while the GPU re-reads the model weights and the growing KV cache. Prefill is compute-bound; decode is memory-bound. Same model, same silicon, two workloads that stress entirely different parts of the chip.
 
That split is why serving has two latency metrics, not one. **Time to first token (TTFT)** is how long until the response starts appearing — dominated by prefill, plus any time spent waiting in a queue. **Inter-token latency (ITL)** is the gap between tokens once they're streaming — the decode heartbeat. A system can have excellent TTFT and terrible ITL, or the reverse, and users experience those failures very differently: slow TTFT feels like the system is ignoring you; jittery ITL feels like it's choking mid-sentence.
 
And when we measure either one, averages lie. A mean ITL of 15ms tells you nothing if every twentieth token stalls for 200ms. Production SLAs are written against **percentiles** — p99 means "the worst experience 1 in 100 requests gets" — because tail behavior, not typical behavior, is what users remember and what contracts are written against.
 
## What disaggregation actually is
 
In a conventional ("aggregated") deployment, every GPU worker does both jobs: it prefills incoming prompts and decodes ongoing responses, interleaved in the same batch. This is how vLLM and most serving engines run out of the box, and it has one glaring failure mode: **interference**. When a 8,000-token prompt lands on a worker that's mid-stream on twenty other conversations, that heavyweight prefill elbows into compute the decode steps needed — and every one of those twenty users sees their token stream stutter at once.
 
Disaggregation splits the fleet: some workers *only* prefill, some workers *only* decode. A finished prefill hands its KV cache — the model's working memory for that conversation — across the wire to a decode worker, which streams the response. Prefill spikes can no longer touch decode latency, because they're physically on different silicon.
 
The price of admission is that KV transfer between pools, plus a subtler cost that this study quantifies: **a split fleet is a divided fleet.** Under a fixed GPU budget, every worker you dedicate to prefill is a worker that can't decode, and vice versa. If your workload doesn't generate enough of both kinds of work to keep both pools busy, you've stranded capacity that an aggregated deployment would have used.
 
Two production-grade implementations of this idea exist today — NVIDIA Dynamo and the CNCF-governed llm-d — and Part 2 of this series digs into how they orchestrate it. This post stays at the architecture level: *when* is the split worth it, regardless of whose software does the splitting?
 
## Finding 1 — For chat traffic, don't bother
 
The chat profile in the sweep (512-token prompts, 256-token responses, 3 turns) is roughly the shape of every consumer assistant conversation ever had — and here the result was not close. **Aggregated won every single cell**: every prefix-reuse rate, every load level, on *both* throughput and p99 TTFT.
 
![Pareto frontier by workload profile](../../dynosim-disaggregation-study/chart1_pareto_by_profile.png)
 
The mechanism is the fleet-division cost from the last section, with nothing to offset it. Short prompts mean prefill is a small fraction of total work — there simply isn't enough prefill to keep a dedicated pool busy. Carve one worker out of four for prefill and it idles while the three decode workers run hot; the aggregated fleet, meanwhile, happily interleaves the small prefills into its decode batches and uses all four GPUs for everything.
 
This is worth sitting with, because chat is not an edge case — it's arguably the single most common LLM workload in production. If your traffic looks like chat, the trendy architecture is the wrong architecture, and the boring one is free capacity.
 
## Finding 3 — The tradeoff nobody escapes
 
Across all 27 workload cells, one pattern held without exception: **aggregated wins p99 TTFT, and disaggregation wins ITL stability.** Neither architecture escapes its half of the tradeoff; they just choose different victims.
 
![ITL stability: aggregated vs. disaggregated](../../dynosim-disaggregation-study/chart3_itl_stability.png)
 
The magnitude is what makes this an architectural decision rather than a tuning detail. At high concurrency (128 concurrent sessions), aggregated serving's p99 ITL landed between **88 and 184 ms** depending on profile — every so often, a stream visibly freezes because a big prefill just landed on its worker. Every disaggregated split, at the same load, held p99 ITL at **7–9 ms**. That's roughly a **20x difference in tail stability**, and it comes from exactly one design choice: prefill physically cannot interrupt decode when they don't share silicon.
 
So the real question an architect asks isn't "which is faster?" — it's **"which SLA am I paid to hit?"**
 
- If your product promise is *the response starts fast* — search-style interactions, single-turn Q&A — aggregated's TTFT edge is the one that matters.
- If your product promise is *the stream never stutters* — voice agents, coding assistants mid-generation, any multi-turn agent where the user watches tokens render dozens of times per session — a 184 ms freeze at p99 is a broken product, and disaggregation's 20x steadier tail is what you're buying.
For multi-turn agentic traffic, ITL stability compounds: a p99 hiccup that hits once per 100 requests hits almost every *session* when a session is 40 requests long. That arithmetic is why agentic serving keeps pulling toward disaggregation even where TTFT gets slightly worse.
 
## When disaggregation doesn't pay
 
The sweep plus first principles give a short checklist. Disaggregation is the wrong call when:
 
1. **Prompts are short relative to responses** (chat-shaped traffic). There isn't enough prefill work to justify a dedicated pool — Finding 1 is the data.
2. **Traffic is low or spiky.** A divided fleet strands capacity at low utilization; if your GPUs are at 30%, your problem is demand, not interference.
3. **KV transfer costs more than it saves.** The KV cache handoff between pools rides the interconnect. On NVLink-class links (~900 GB/s on Hopper, ~1.8 TB/s on Blackwell), a 1 GB cache moves in about a millisecond — effectively free. Over PCIe or commodity Ethernet, transfer time can approach or exceed the prefill time you're trying to protect, and the architecture defeats itself. *(Interconnects get their full treatment in Part 4.)*
4. **You're serving one user or a few.** Interference needs contention; a single stream has nothing to interfere with.
Notice what's *not* on the list: model size, GPU generation, or which serving engine you run. The decision is about workload shape and interconnect, not about hardware prestige.
 
## The decision in one pass
 
If you take one thing from this post, take the order of questions:
 
**1. What shape is the traffic?** Short-prompt, chat-like → aggregated, stop here. Long-prompt or multi-turn agentic → keep going.
**2. Which tail SLA is the product built on?** TTFT → aggregated keeps its edge. ITL stability → disaggregation is buying you ~20x.
**3. Can the interconnect carry the KV handoff for free?** NVLink-class → yes. PCIe/Ethernet-class → measure before committing.
**4. Is there enough sustained load to keep two pools busy?** If not, the split strands capacity no matter what the benchmarks promised.
 
Four questions, and three of them are about your workload rather than anyone's product sheet. That's the general lesson of the whole sweep: **disaggregation is not a feature you adopt, it's a bet on your traffic's shape** — and you should know the shape before you place it.
 
## The part I haven't told you yet
 
Everything above treats "disaggregated" as one thing. It isn't. A disaggregated fleet has a ratio — how many workers prefill versus decode — and the sweep's most interesting result is that **the right ratio isn't a constant. It moves with your prefix-cache hit rate**, and at the extreme (90% cache reuse, high load) the right split beat aggregated throughput by 24%.
 
Which means a static split is a snapshot of a moving target — and *that* is the actual argument for dynamic disaggregation and planner-driven scaling, the subject of Post 2.
 
## Methods, honestly
 
These are discrete-event simulation results from [NVIDIA Dynamo's DynoSim](https://docs.nvidia.com/dynamo/user-guides/dynosim) mocker engine with default timing models — **not hardware benchmarks, and not AIC-calibrated GPU timing**. The relative behavior across configurations is the finding; the absolute numbers are not. Per NVIDIA's own guidance, DynoSim narrows the search space; it doesn't replace real-hardware validation. The full 108-run sweep — code, results, charts — is [on GitHub](https://github.com/saurabh9498/saurabh-rai/tree/main/dynosim-disaggregation-study), reproduces on CPU in ~25 seconds, and was verified independently on two environments.
 
---
 
*Next: [Post 2 — Dynamic Disaggregation and the Planner], where the P:D ratio stops being a config value and starts being a control loop.*
