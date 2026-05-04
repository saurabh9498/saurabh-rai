# Real-Time Fraud Detection — Reference Architecture

> Reference architecture for streaming fraud detection on payment transaction data. Demonstrates the pipeline structure (ingestion → feature engineering → ML scoring → rules engine → decision) used in production fraud platforms.

**This is a reference architecture, not a deployed system.** It demonstrates the streaming-first, feature-store-backed, ensemble-scoring pattern I worked with on production fraud detection at Tata Consultancy Services — a real-time platform processing 10M+ daily transactions across ACH, wire, and card channels (detection rate 62% → 85% at <200ms latency, false positives -30%, ~$18M annual fraud loss prevention). The TCS production system was built on Spark Streaming; this reference uses Kafka + a Python stream processor for clarity and broader educational value.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REAL-TIME FRAUD DETECTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  PAYMENT     │    │   KAFKA      │    │     STREAM PROCESSOR         │  │
│  │  GATEWAY     │───▶│   CLUSTER    │───▶│                              │  │
│  │              │    │  Partitioned │    │  Feature Engineering         │  │
│  └──────────────┘    │  by Card ID  │    │  • Velocity (1h/6h/24h/7d)   │  │
│                      └──────────────┘    │  • Aggregations              │  │
│                                          │  • Graph features            │  │
│  ┌──────────────┐                        │                              │  │
│  │   REDIS      │◀───────────────────────│  ML Ensemble                 │  │
│  │   FEATURE    │                        │  • XGBoost (tabular)         │  │
│  │   STORE      │                        │  • Neural Net (sequential)   │  │
│  │              │                        │  • Isolation Forest (novelty)│  │
│  └──────────────┘                        │                              │  │
│                                          │  Risk Score [0.0 — 1.0]      │  │
│                                          └──────────────┬───────────────┘  │
│                                                         │                   │
│                                                         ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  DECISION    │◀───│   RULES      │◀───│  Velocity / Pattern Rules    │  │
│  │  ENGINE      │    │   ENGINE     │    │  Blacklist / Geographic      │  │
│  │ Approve /    │    │              │    │                              │  │
│  │ Review /     │    └──────────────┘    └──────────────────────────────┘  │
│  │ Decline      │                                                          │
│  └──────────────┘                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Architecture

A few patterns this reference makes explicit:

**Streaming-first detection.** The cost of fraud detection is asymmetric — a 500ms batch-window detection delay is too late, because the fraudster has already completed the transaction. Streaming architectures (Spark Streaming in the TCS production system, Kafka + stream processor here) push detection inside the authorization window where it can actually prevent loss, not just report it.

**Feature store separation from the model.** Velocity features (transaction count over rolling windows, amount aggregates, unique merchant counts) need sub-millisecond retrieval at scoring time. Computing them per-request is too slow; pre-computing them in Redis with TTL-bounded keys hits the latency budget. This separation also lets feature engineering and model training evolve independently.

**Ensemble over single-model.** Different fraud patterns favor different models — gradient boosting catches feature-interaction patterns, neural networks catch sequential/embedding patterns, and isolation forests catch novel patterns the others have never seen. Ensembling produces a more robust risk score than any single model alone.

**Rules engine alongside ML.** Some fraud patterns are deterministic (high-value first transaction, geographic impossibility, blacklist matches) and don't benefit from probabilistic scoring. Keeping a separate rules engine for these patterns means the ML model focuses on the harder-to-codify cases, and compliance/business stakeholders can adjust rules without retraining.

**Decision tier with explicit thresholds.** Approve / Review / Decline tiers (not just binary fraud/not-fraud) let the system route ambiguous cases to manual review while auto-approving the clear majority and auto-declining the highest-risk minority. This is what makes fraud platforms cost-effective at scale.

---

## Component Structure

| Module | Purpose |
|---|---|
| `src/streaming/` | Kafka producer, consumer, stream processor |
| `src/features/` | Redis-backed feature store, velocity calculations, graph features |
| `src/models/` | XGBoost classifier, neural network, isolation forest, ensemble orchestrator |
| `src/api/` | FastAPI scoring endpoint with Pydantic schemas |
| `src/monitoring/` | Prometheus metrics, drift detection hooks, alerting |
| `configs/` | Model parameters, rules definitions, feature specifications |

The `configs/` directory parameterizes the system — model weights and thresholds, velocity rule definitions, feature window specifications. Production fraud platforms typically expand these significantly with per-channel logic (ACH vs wire vs card have different risk profiles), per-merchant-category rules, and regulatory-driven adjustments (AML/KYC overlays).

---

## Tech Stack

**Streaming:** Apache Kafka, Redis (feature store)
**ML Models:** XGBoost, PyTorch (neural network), scikit-learn (isolation forest)
**Serving:** FastAPI, Pydantic
**Observability:** Prometheus, Grafana
**Production Counterpart (TCS):** Spark Streaming-based platform with similar architectural patterns

---

## What's Here vs. Production

This reference architecture demonstrates the structure I work with on production fraud detection systems. The TCS production deployment that informed this work was internal and proprietary — that codebase is not public. What you can read here is the architectural thinking, component boundaries, and the streaming-first / ensemble / rules-alongside-ML approach that ships into a regulated payments environment.

If you're working on similar systems and want to compare notes — happy to talk.

---

## Related Work

- **GPU ML Pipeline** ([`../gpu-ml-pipeline`](../gpu-ml-pipeline)) — Triton serving and KEDA autoscaling patterns relevant to high-TPS scoring infrastructure
- **Recommendation System** ([`../recommendation-system`](../recommendation-system)) — Feature engineering and model serving patterns at scale

---

## License

MIT — see [LICENSE](./LICENSE)
