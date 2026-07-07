# Field Guide to AI Infrastructure
 
*A bottom-to-top guide to how the modern AI/semiconductor stack works — from sand to orbital inference — written while breaking into the field, for anyone trying to do the same.*
 
The guide is organized as ~26 Parts covering the full stack: fabrication → silicon → memory → interconnect → datacenter → software → inference mechanics → serving → orchestration → economics → careers. Parts are written in the order that matters for practitioners (serving and inference first), not reading order — so the table of contents below fills in non-linearly.
 
Where possible, each topic is grounded in a real deployment or a reproducible experiment rather than summarized from documentation. The first entries build on [a 108-run DynoSim disaggregation study](../dynosim-disaggregation-study) published in this repo.
 
## Published so far
 
| Post | Guide section | TL;DR |
|---|---|---|
| [When Does Disaggregation Pay?](./disaggregation/01-when-does-disaggregation-pay.md) | 14.6 / 15.4 (1 of 2) | Chat traffic and agentic traffic want opposite serving architectures. Aggregated wins p99 TTFT everywhere; disaggregation buys ~20x steadier inter-token latency. Four questions decide it. |
| *Dynamic Disaggregation and the Planner* — coming next | 14.6 / 15.4 (2 of 2) | The optimal prefill:decode ratio isn't a constant — it moves with your prefix-cache hit rate. Why static splits are fragile and planner-driven scaling exists. |
 
## Coming up
 
Inference mechanics (prefill/decode, latency metrics, batching) · Inference optimization (quantization, speculative decoding, KV-cache engineering) · Serving engines (vLLM, SGLang, TensorRT-LLM, Dynamo, llm-d) · Memory & the memory wall · Interconnect & networking · Orchestration & production
 
---
 
*Questions, corrections, or disagreements welcome — open an issue or find me on [LinkedIn](https://www.linkedin.com/in/saurabh-rai-aipm/).*
