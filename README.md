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

## Reference implementations

Patterns from production work, rebuilt as standalone examples. Not client code.

| Folder | What it is | Stack |
|---|---|---|
| [`retail-vision-analytics`](./retail-vision-analytics) | Edge computer vision: INT8 quantization on constrained hardware, generic model plus per-store calibration. | YOLOv8, TensorRT, DeepStream |
| [`fraud-detection`](./fraud-detection) | Real-time fraud detection on streaming transactions across ACH, wire and card. | Spark Streaming, Kafka, PyTorch |

Older 2025 folders are kept in [`archive/`](./archive).

---

## Where these patterns come from

- **Real-time bidding** — CPU-to-GPU inference migration at billions of requests a day under a 100 ms SLA; later repatriated to bare-metal edge hubs, with training on a multi-node H100 SuperPOD. Reconstructed in [`rtb-gpu-inference`](./rtb-gpu-inference) and [`rtb-edge-scale-out`](./rtb-edge-scale-out).
- **GPU analytics** — RAPIDS segmentation over tens of millions of users, with a Builder + Judge natural-language-to-SQL layer. Reconstructed in [`gpu-analytics-nl2sql`](./gpu-analytics-nl2sql).
- **Healthcare conversational AI** — HIPAA-compliant patient scheduling agents across multiple health systems; multi-turn dialogue on Triton + NIM, deployed on-prem. Reconstructed in [`healthcare-dialog-agents`](./healthcare-dialog-agents).
- **Edge computer vision** — hundreds of unattended retail sites; accelerator chosen on workload fit and total cost.
- **Financial services** — real-time fraud detection across payment channels, moved from batch to streaming.

---

## Technical focus

**Programming** — Python (PyTorch, TensorFlow, JAX, NumPy, Pandas, XGBoost) · CUDA · C++ · Java · Shell · SQL

**GPU optimization** — warp divergence · occupancy · memory coalescing · roofline analysis · kernel fusion · Nsight · PyTorch Profiler

**Inference & serving** — TensorRT, TensorRT-LLM, Triton, NVIDIA NIM, NVIDIA Dynamo, vLLM, ONNX Runtime, NeMo, NGC, KServe · disaggregated serving (prefill/decode) · KV-cache management and KV-aware routing · PagedAttention · FlashAttention · continuous batching · speculative decoding · FP16/INT8/FP8/FP4/NVFP4 · PTQ/QAT/KL calibration · Merlin/HugeCTR · UVM tiering · hot-row replication · tensor/pipeline/3D parallelism · edge inference

**GPU & HPC** — DGX (A100, H100, B200) · H100 SuperPOD · A10G · RTX PRO 6000 Blackwell · MIG · RAPIDS (cuDF, Dask) · NCCL (AllReduce, AllToAll) · Slurm · Base Command Manager

**Networking & storage** — NVLink · NVSwitch · InfiniBand (Quantum-2) · RoCE · RDMA · PCIe Gen5 · BGP Anycast · Arista eAPI · GPUDirect Storage · NVMe · WEKA · Apache Iceberg · DPU/BlueField

**Kubernetes & IaC** — Kubernetes (EKS, GKE, bare metal) · GPU Operator · DRA (Dynamic Resource Allocation) · KEDA · Helm · Kubeflow · Slurm-on-K8s · Docker · containerd · Linux · blue-green rollouts · Terraform · Ansible · CI/CD · GitOps · MLflow

**Reliability & observability** — DCGM, dcgm-exporter, nvidia-smi · Prometheus, Alertmanager, Grafana, Thanos · eBPF · OpenTelemetry tail-based sampling · Xid triage · straggler and throttle detection · cordon/drain · fault quarantine and remediation · health-based autoscaling · circuit breakers · hot-spare recovery · fault tolerance for distributed serving · failure attribution · p50/p90/p99 latency profiling

**AI/ML & LLMs** — LLMs (Llama-3 8B/70B, BERT) · CLIP/ViT · DLRM · Hugging Face · LangChain · FastAPI · fine-tuning (LoRA/QLoRA/PEFT) · RLHF · RAG · vector databases (Pinecone, FAISS) · multi-agent systems (Builder + Judge) · conversational AI · NLP · computer vision · recommender systems · MLOps · A/B testing

**Cloud, data & compliance** — AWS (EC2 P4d/P5/G5, EKS, S3, ElastiCache, SageMaker) · GCP (Vertex AI) · Azure · Snowflake · Spark · Kafka · Airflow · HIPAA · PCI-DSS · SOC 2 · GDPR/CCPA · IAM · RBAC · signed model artifacts · confidential computing

**Solution architecture** — requirements discovery · capacity and VRAM sizing · TCO and cost-per-token modeling · CapEx vs OpEx · reference architectures · PoCs · build-vs-buy · REST and gRPC APIs · microservices · event-driven architecture · SDK development · HPC strategy · SDLC · Agile/Scrum

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
