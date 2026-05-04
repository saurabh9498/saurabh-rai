# Saurabh Rai

**Senior Product Manager | GPU Inference Infrastructure | Kubernetes, Distributed Systems, Agentic AI**

NVIDIA NCA-AIIO Certified · NVIDIA GenAI Professional (in progress) · NVIDIA Infrastructure Professional (in progress)

---

I build AI infrastructure where the cost of being wrong is measured in milliseconds and millions. 12 years across software engineering and product management, with deep technical fluency in inference stacks (TensorRT-LLM, Triton, Dynamo, vLLM), Kubernetes-native GPU orchestration, distributed serving, and cost-per-token economics.

This repo is a portfolio of projects I've shipped or built — production systems, reference architectures, and experiments. Each folder has its own README with the technical detail, results, and what I'd do differently.

---

## Projects

| Project | What it is | Stack |
|---|---|---|
| [`gpu-ml-pipeline`](./gpu-ml-pipeline) | GPU-accelerated ML inference pipeline. Cross-architecture compilation, TensorRT INT8 quantization, Triton on EKS with KEDA autoscaling on DCGM metrics. | NVIDIA Merlin, TensorRT, Triton, EKS, KEDA, DCGM |
| [`multi-agent-orchestration`](./multi-agent-orchestration) | Multi-agent NL-to-SQL system with Builder + Judge architecture and LogProb-based confidence clarification. Schema RAG via vector DB. | LangChain, RAPIDS, Pinecone, FastAPI |
| [`recommendation-system`](./recommendation-system) | GPU-accelerated recommendation engine. RAPIDS cuDF + Dask feature engineering, hybrid collaborative + content-based ranking, A/B testing harness. | NVIDIA Merlin, RAPIDS, PyTorch |
| [`conversational-ai`](./conversational-ai) | Production conversational AI patterns. Multi-turn dialogue, KV-cache-aware routing on Triton + NIM, on-prem deployment for regulated industries. | Triton, NIM, LangChain |
| [`retail-vision-analytics`](./retail-vision-analytics) | Edge AI computer vision for retail. INT8 quantization on constrained edge hardware (Qualcomm Cloud AI 100), per-store calibration approach replacing per-store models. | YOLOv8, TensorRT, DeepStream |
| [`fraud-detection`](./fraud-detection) | Real-time fraud detection on streaming transactions. Spark Streaming over batch, ML-enhanced detection across ACH, wire, card. | Spark Streaming, Kafka, PyTorch |

---

## Production highlights

These are projects where the patterns and architecture in this repo trace back to real production systems I've owned:

- **Real-Time Bidding Platform — 1.5B daily requests.** CPU→GPU inference migration on a 40-node A10G fleet. Triton on EKS with custom KEDA autoscaling on DCGM + Triton queue depth. p99 latency 180ms → 45ms. 52K peak QPS at 45ms SLA. Unit cost $2.23 → $1.55 per million requests. **$30M+ incremental annual revenue, 70x ROI.**
- **GPU-Accelerated Audience Segmentation — 30M+ users / 4.5B events.** LLM natural-language interface over RAPIDS backend with Builder + Judge multi-agent NL-to-SQL. Query time 4hr → 12min. **$3M annual CAC reduction.**
- **HIPAA-Compliant Conversational AI — 1.8M+ patients, 3 health systems.** NLP scheduling agents with multi-turn dialogue and KV-cache-aware routing on Triton + NIM, on-prem HIPAA architecture. **No-show 18% → 10%, $4M annual savings.**
- **Edge AI Computer Vision — 200+ retail stores.** Per-channel INT8 quantization on Qualcomm Cloud AI 100. Generic-model + per-store-calibration approach replaced 200 store-specific models. **Inventory accuracy 82% → 95%, 22x ROI.**

---

## Technical focus areas

**AI Inference & Serving** — TensorRT-LLM, Triton, NVIDIA NIM, NVIDIA Dynamo, vLLM, ONNX Runtime, NeMo · Disaggregated serving (prefill/decode) · KV-cache management & KV-aware routing · PagedAttention · Continuous batching · Speculative decoding · FlashAttention · PTQ/QAT · FP16/INT8/FP8/FP4/NVFP4 · Edge & on-device inference

**GPU & HPC Stack** — DGX (A100, H100, B200) · CUDA · NVIDIA Merlin · NVIDIA RAPIDS (cuDF, Dask) · MIG · NCCL · AllReduce · Slurm · BCM

**Networking & Interconnects** — NVLink · NVSwitch · InfiniBand · RoCE · RDMA · GPUDirect · NVMe · DPU/BlueField

**Kubernetes & IaC** — Kubernetes · GPU Operator · DRA · KEDA · Helm · Kubeflow · KServe · EKS · Slurm-on-K8s · Docker · Terraform · Ansible · CI/CD · GitOps

**AI/ML & LLMs** — PyTorch · TensorFlow · JAX · LLMs (Llama, BERT) · Hugging Face · LangChain · RAG · Vector DBs (Pinecone, FAISS) · Multi-Agent Systems (Builder + Judge) · LoRA/QLoRA/PEFT · RLHF

**Observability** — DCGM · nvidia-smi · Nsight · Prometheus · Grafana · MLflow

---

## Background

- **Senior Product Manager**, Deloitte (Feb 2022 – Present)
- **Product Management Intern**, Tesla (Oct 2021 – Dec 2021)
- **Product Manager / Software Engineer**, Tata Consultancy Services (Jul 2013 – Aug 2020)

**MS Business Analytics** — UC San Diego, Rady School of Management
**B.Tech Chemical Engineering** — SRM University, Chennai

---

## Connect

- **LinkedIn:** [linkedin.com/in/saurabh-rai-aipm](https://www.linkedin.com/in/saurabh-rai-aipm/)
- **Email:** rai.saurabh9491@gmail.com

---

*Building inference infrastructure that scales — happy to compare notes with anyone working on GPU-accelerated systems, distributed serving, or recommendation/personalization at scale.*
