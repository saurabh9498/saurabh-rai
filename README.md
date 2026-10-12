# Saurabh Rai

**Senior Solution Architect — GPU AI Infrastructure | Inference & Training at Scale | Kubernetes-Native GPU Platforms**

NVIDIA NCA-AIIO certified · Contributor to NVIDIA Dynamo and NVSentinel

---

I design AI infrastructure — training, inference, and the platform underneath both — and take it from requirements and sizing through PoC to production. Twelve years, starting as a software engineer, across ad tech, healthcare, retail and financial services.

This repo holds my open-source work, a published study, a field guide to the AI infrastructure stack, and reference implementations of patterns from production systems I've worked on. Each folder has its own README.

**Latest:** [Field Guide to AI Infrastructure](./field-guide) — Post 2, [*Dynamic Disaggregation and the Planner*](./field-guide/disaggregation/02-dynamic-disaggregation-and-the-planner.md).

---

## Open source

- **[NVIDIA/NVSentinel](https://github.com/NVIDIA/NVSentinel)** — filed [#1647](https://github.com/NVIDIA/NVSentinel/issues/1647) (demo scripts reported success without verifying the result), was assigned it, and authored the fix: [#1655](https://github.com/NVIDIA/NVSentinel/pull/1655), merged to main. [First-run notes](./nvsentinel-first-run-notes).
- **[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)** — filed and bisected [#11228](https://github.com/ai-dynamo/dynamo/issues/11228) (DynoSim segfault under Rosetta) down to a single flag. NVIDIA root-caused it as an Apple Rosetta translation defect; the documentation gaps I reported shipped in NVIDIA-authored [#11430](https://github.com/ai-dynamo/dynamo/pull/11430).
- **DynoSim study** — [108-run disaggregation study](./dynosim-disaggregation-study), published against NVIDIA's community-experiments invitation, with first-user notes fed back to the team.

---

## Study and field guide

| Folder | What it is | Stack |
|---|---|---|
| [`dynosim-disaggregation-study`](./dynosim-disaggregation-study) | **When does disaggregation pay?** 108 simulated runs of aggregated vs. disaggregated prefill/decode serving for chat, RAG and agentic workloads. The optimal prefill:decode split shifts with prefix-cache reuse (3P+1D at 20%, 2P+2D at 90%; +22–24% over aggregated), and on prefill-heavy traffic aggregated serving pays up to ~27× in p99 inter-token latency under load. Reproducible on CPU in ~25 s. | NVIDIA Dynamo, DynoSim, KV Router, Python |
| [`field-guide`](./field-guide) | A bottom-to-top guide to the AI infrastructure and semiconductor stack, in progress. Starts with disaggregated serving. | — |
| [`nvsentinel-first-run-notes`](./nvsentinel-first-run-notes) | Running NVSentinel without a GPU: pipeline walkthrough, five things the docs don't say, and the upstream issue that led to PR #1655. | NVSentinel, Kubernetes, KIND, DCGM, Helm, CEL |

## Architecture reconstructions

Production architectures rebuilt as runnable code, simulations, configuration and sizing math. Not client code; every number is labelled as measured, derived or designed.

| Folder | What it is | Stack |
|---|---|---|
| [`rtb-gpu-inference`](./rtb-gpu-inference) | **Phase 1.** A 1.2 TB DLRM trained on one 8×H100 node, compiled to INT8 TensorRT and served by Triton on 24 GB A10Gs under a 100 ms timeout: training-memory, VRAM and latency sizing, a hot-cache simulation under Zipfian traffic, ONNX export, Triton, Kubernetes and KEDA configuration. | PyTorch, ONNX, TensorRT, Triton, EKS, KEDA |
| [`rtb-edge-scale-out`](./rtb-edge-scale-out) | **Phases 2–3.** Bare-metal edge hubs with separate bidding and LLM GPUs, KV-cache sizing that forces TP=4 for a 70B model, a latched DCGM → BGP circuit breaker with tests, a flash-crowd cache simulation, and checkpoint-cadence math for hourly retraining on a 32-node H100 cluster. | Triton, TensorRT-LLM, DCGM, Prometheus, Arista eAPI, Slurm, BCM |
| [`gpu-analytics-nl2sql`](./gpu-analytics-nl2sql) | **GPU analytics.** Audience segmentation written once for RAPIDS cuDF or pandas, and a Builder + Judge NL-to-SQL layer: schema retrieval, role-based access enforced on the parsed SQL, a read-only and dry-run validation gate on a locked DuckDB, and log-probability confidence that asks a clarifying question instead of guessing. | RAPIDS cuDF, pandas, DuckDB, sqlglot |
| [`healthcare-dialog-agents`](./healthcare-dialog-agents) | **Healthcare scheduling agents.** A multi-turn booking agent that minimises PHI before text reaches the model or the logs, acts only on an explicit read-back, hands off on urgent symptoms, and keeps a hash-chained audit trail; plus a simulation of KV-cache-aware routing for multi-turn traffic, and on-prem NIM serving with egress locked down. | Python, NVIDIA NIM, Kubernetes NetworkPolicy |
| [`edge-vision-retail`](./edge-vision-retail) | **Edge computer vision.** Per-channel vs per-tensor INT8 quantization measured, one generic model plus a label-free per-store calibration measured against store-specific models, an offline-first store-and-forward buffer, staged fleet updates with local A/B rollback, and device sizing. | PyTorch, scikit-learn, SQLite |

## Reference implementations

Patterns from production work, rebuilt as standalone examples. Not client code.

| Folder | What it is | Stack |
|---|---|---|
| [`fraud-detection`](./fraud-detection) | Real-time fraud detection on streaming transactions across ACH, wire and card. | Spark Streaming, Kafka, PyTorch |

Older 2025 folders are kept in [`archive/`](./archive).

---

## Where these patterns come from

- **Real-time bidding** — CPU-to-GPU inference migration at billions of requests a day under a 100 ms SLA; later repatriated to bare-metal edge hubs, with training on a multi-node H100 SuperPOD. Reconstructed in [`rtb-gpu-inference`](./rtb-gpu-inference) and [`rtb-edge-scale-out`](./rtb-edge-scale-out).
- **GPU analytics** — RAPIDS segmentation over tens of millions of users, with a Builder + Judge natural-language-to-SQL layer. Reconstructed in [`gpu-analytics-nl2sql`](./gpu-analytics-nl2sql).
- **Healthcare conversational AI** — HIPAA-compliant patient scheduling agents across multiple health systems; multi-turn dialogue on Triton + NIM, deployed on-prem. Reconstructed in [`healthcare-dialog-agents`](./healthcare-dialog-agents).
- **Edge computer vision** — hundreds of unattended retail sites; accelerator chosen on workload fit and total cost. Reconstructed in [`edge-vision-retail`](./edge-vision-retail).
- **Financial services** — real-time fraud detection across payment channels, moved from batch to streaming.

---

## Technical focus

- **Inference and serving** — TensorRT, TensorRT-LLM, Triton, NVIDIA NIM, NVIDIA Dynamo, vLLM · quantization (INT8, FP8) · continuous batching · disaggregated serving
- **Training and data** — PyTorch, distributed training, NCCL · RAPIDS · Spark, Kafka
- **GPU platforms** — H100, Blackwell, A10G · CUDA · Kubernetes with the GPU Operator · Slurm · MIG
- **Networking and storage** — NVLink, InfiniBand, RDMA, GPUDirect Storage, NVMe
- **Reliability and observability** — DCGM, Prometheus, Grafana · autoscaling · failover
- **AI applications** — LLMs, RAG, multi-agent systems, recommender systems, computer vision
- **Languages and tools** — Python, CUDA, C++, SQL · Docker, Helm, Terraform, CI/CD
- **Cloud** — AWS, GCP, Azure

---

## Background

- **Senior Solution Architect**, Deloitte — Feb 2022 – present
- **Product Management Intern**, Tesla — Oct 2021 – Dec 2021
- **Product Management Intern**, Icertis — Jul 2021 – Sep 2021
- **Solution Architect**, Tata Consultancy Services — Apr 2016 – Aug 2020
- **Software Engineer**, Tata Consultancy Services — Jul 2013 – Mar 2016

**MS Business Analytics** — UC San Diego, Rady School of Management

**B.Tech Chemical Engineering** — SRM University, Chennai

---

## Connect

- **LinkedIn:** [linkedin.com/in/saurabh-rai-aipm](https://www.linkedin.com/in/saurabh-rai-aipm/)
- **Email:** rai.saurabh9491@gmail.com
