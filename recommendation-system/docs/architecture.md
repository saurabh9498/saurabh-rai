# Architecture Documentation

## System Overview

The Real-Time Personalization Engine is a production-grade recommendation system designed for high-throughput, low-latency serving at scale.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REAL-TIME SERVING LAYER                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   FastAPI   │──│   Feature    │──│  Retrieval  │──│    Ranking      │   │
│  │   Gateway   │  │    Store     │  │   Service   │  │    Service      │   │
│  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────────┘   │
│         │                │                 │                  │             │
│         ▼                ▼                 ▼                  ▼             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   Request   │  │    Redis     │  │   FAISS     │  │  Triton Server  │   │
│  │  Validator  │  │   Cluster    │  │    Index    │  │  (GPU Inference)│   │
│  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │    Data     │──│   Feature    │──│   Model     │──│     Model       │   │
│  │   Ingestion │  │  Engineering │  │  Training   │  │    Registry     │   │
│  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────────┘   │
│         │                │                 │                  │             │
│         ▼                ▼                 ▼                  ▼             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   Spark     │  │   NVTabular  │  │   PyTorch   │  │    MLflow       │   │
│  │   Streaming │  │   (GPU ETL)  │  │   + DDP     │  │   + S3          │   │
│  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Two-Tower Retrieval Model

The Two-Tower architecture enables efficient candidate retrieval from millions of items in sub-5ms latency.

#### Architecture

```
User Tower                              Item Tower
───────────                             ───────────
┌───────────────┐                       ┌───────────────┐
│  User ID      │                       │  Item ID      │
│  Demographics │                       │  Category     │
│  Context      │                       │  Attributes   │
└───────┬───────┘                       └───────┬───────┘
        │                                       │
        ▼                                       ▼
┌───────────────┐                       ┌───────────────┐
│   Embedding   │                       │   Embedding   │
│    Layers     │                       │    Layers     │
└───────┬───────┘                       └───────┬───────┘
        │                                       │
        ▼                                       ▼
┌───────────────┐                       ┌───────────────┐
│  History      │                       │  Text         │
│  Encoder      │                       │  Encoder      │
│  (Attention)  │                       │  (Optional)   │
└───────┬───────┘                       └───────┬───────┘
        │                                       │
        ▼                                       ▼
┌───────────────┐                       ┌───────────────┐
│     MLP       │                       │     MLP       │
└───────┬───────┘                       └───────┬───────┘
        │                                       │
        ▼                                       ▼
┌───────────────┐                       ┌───────────────┐
│  L2 Normalize │                       │  L2 Normalize │
└───────┬───────┘                       └───────┬───────┘
        │                                       │
        ▼                                       ▼
   128-dim User                           128-dim Item
    Embedding                              Embedding
        │                                       │
        └───────────────┬───────────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  Cosine Similarity │
              │   (via FAISS)     │
              └───────────────────┘
```

#### Key Design Decisions

1. **In-Batch Negative Sampling**: Uses other items in the batch as negatives, enabling efficient training without explicit negative sampling.

2. **Temperature-Scaled Softmax**: Controls the sharpness of the similarity distribution during training.

3. **Attention-Based History Encoding**: Captures sequential patterns and user preferences from interaction history.

4. **L2 Normalization**: Ensures embeddings lie on a unit sphere, enabling efficient cosine similarity computation.

### 2. DLRM Ranking Model

Deep Learning Recommendation Model (DLRM) for fine-grained ranking of retrieved candidates.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DLRM Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Dense Features          Sparse Features (Categorical)          │
│   [13 floats]             [26 indices]                           │
│        │                       │                                 │
│        ▼                       ▼                                 │
│   ┌─────────┐            ┌─────────────────────────────────┐    │
│   │ Bottom  │            │      Embedding Tables           │    │
│   │  MLP    │            │  (10M+ rows × 64 dim each)      │    │
│   │[512→64] │            └─────────────────────────────────┘    │
│   └────┬────┘                          │                         │
│        │                               │                         │
│        │         ┌─────────────────────┤                         │
│        │         │                     │                         │
│        ▼         ▼                     ▼                         │
│   ┌─────────────────────────────────────────┐                   │
│   │          Feature Interaction            │                   │
│   │     (Pairwise Dot Products)             │                   │
│   │  [64 + 26×64 = 1728 interactions]       │                   │
│   └─────────────────┬───────────────────────┘                   │
│                     │                                            │
│                     ▼                                            │
│            ┌────────────────┐                                   │
│            │    Top MLP     │                                   │
│            │ [1728→512→256→1]│                                  │
│            └────────┬───────┘                                   │
│                     │                                            │
│                     ▼                                            │
│                 Sigmoid                                          │
│               (CTR Score)                                        │
└─────────────────────────────────────────────────────────────────┘
```

#### Multi-Task Learning Extension

```
                    Shared Bottom
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │   CTR   │   │   CVR   │   │ Revenue │
      │  Tower  │   │  Tower  │   │  Tower  │
      └────┬────┘   └────┬────┘   └────┬────┘
           │             │             │
           ▼             ▼             ▼
        P(click)     P(convert)    E[revenue]
```

### 3. Feature Store

Real-time feature serving with sub-millisecond latency.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Feature Store                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │   Online Store  │     │  Offline Store  │                    │
│  │     (Redis)     │     │   (Parquet)     │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                              │
│           │    ┌──────────────────┘                              │
│           │    │                                                 │
│           ▼    ▼                                                 │
│  ┌─────────────────────────────────────────┐                    │
│  │          Feature Service API            │                    │
│  │  ┌───────────────┐  ┌───────────────┐   │                    │
│  │  │ get_user_     │  │ get_item_     │   │                    │
│  │  │   features()  │  │   features()  │   │                    │
│  │  └───────────────┘  └───────────────┘   │                    │
│  │  ┌───────────────┐  ┌───────────────┐   │                    │
│  │  │ batch_get()   │  │ compute_      │   │                    │
│  │  │               │  │  interaction()│   │                    │
│  │  └───────────────┘  └───────────────┘   │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  Feature Groups:                                                 │
│  ├── User Features (demographics, preferences)                   │
│  ├── Item Features (attributes, embeddings)                      │
│  ├── Context Features (time, device, location)                   │
│  └── Interaction Features (computed in real-time)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Serving Architecture

#### Request Flow

```
                           Request Flow
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. Request   2. Feature    3. Retrieval   4. Ranking   5. Response
│     │            │              │              │             │    │
│     ▼            ▼              ▼              ▼             ▼    │
│  ┌─────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────┐│
│  │Parse│───▶│ Feature │───▶│Two-Tower│───▶│  DLRM   │───▶│Build││
│  │ &   │    │  Fetch  │    │ + FAISS │    │ Ranking │    │JSON ││
│  │Valid│    │ (Redis) │    │(1000 c) │    │(Top 50) │    │     ││
│  └─────┘    └─────────┘    └─────────┘    └─────────┘    └─────┘│
│    │            │              │              │             │    │
│   0.5ms        1ms            3ms            2ms          0.5ms  │
│                                                                  │
│                    Total: ~7-8ms P50                             │
└─────────────────────────────────────────────────────────────────┘
```

#### Scaling Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Horizontal Scaling                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      Load Balancer                               │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                   │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│    ┌─────────┐       ┌─────────┐       ┌─────────┐              │
│    │ API Pod │       │ API Pod │       │ API Pod │              │
│    │  (GPU)  │       │  (GPU)  │       │  (GPU)  │              │
│    └────┬────┘       └────┬────┘       └────┬────┘              │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                   │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              │                         │                         │
│              ▼                         ▼                         │
│    ┌──────────────────┐      ┌──────────────────┐               │
│    │  Redis Cluster   │      │  FAISS Shards    │               │
│    │  (6 nodes, HA)   │      │  (Replicated)    │               │
│    └──────────────────┘      └──────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Raw Data   │───▶│   Feature    │───▶│   Training   │
│  (Events)    │    │  Engineering │    │   Dataset    │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Kafka     │    │  NVTabular   │    │   PyTorch    │
│   Streams    │    │  (GPU ETL)   │    │  DataLoader  │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Model Update Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Model Update Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Train    2. Validate   3. Export    4. Deploy    5. Monitor │
│     │            │            │            │            │        │
│     ▼            ▼            ▼            ▼            ▼        │
│  ┌─────┐    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │Train│───▶│Evaluate │──▶│  ONNX   │──▶│ Triton  │──▶│  A/B   │ │
│  │Model│    │ Metrics │  │ Export  │  │ Deploy  │  │  Test   │ │
│  └─────┘    └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
│                                                                  │
│  Frequency: Every 15 minutes for embeddings                      │
│             Every 24 hours for ranking model                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Component | Latency (P50) | Latency (P99) | Throughput |
|-----------|---------------|---------------|------------|
| Feature Store | 0.3ms | 1.2ms | 100K ops/s |
| Two-Tower Retrieval | 2.5ms | 5.0ms | 45K QPS |
| DLRM Ranking | 1.5ms | 3.0ms | 30K QPS |
| End-to-End | 7ms | 12ms | 20K QPS |

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| API | FastAPI | High-performance async API |
| Feature Store | Redis + Feast | Real-time feature serving |
| Retrieval | FAISS (GPU) | Approximate nearest neighbor |
| Inference | Triton | GPU model serving |
| Training | PyTorch + DDP | Distributed training |
| ETL | NVTabular | GPU-accelerated preprocessing |
| Orchestration | Kubernetes | Container orchestration |
| Monitoring | Prometheus + Grafana | Observability |

## Security Considerations

1. **API Authentication**: JWT tokens with rate limiting
2. **Data Encryption**: TLS in transit, encryption at rest
3. **Access Control**: RBAC for model and data access
4. **Audit Logging**: All predictions logged for compliance
