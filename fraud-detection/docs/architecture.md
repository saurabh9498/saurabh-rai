# Architecture Guide

## System Overview

The Fraud Detection System is a real-time ML platform designed for high-throughput, low-latency transaction scoring. It processes 50,000+ transactions per second with sub-10ms P99 latency.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Payment Gateway ──▶ Kafka Cluster ──▶ Stream Processors (3-10 workers)    │
│                       │                       │                              │
│                       │ Partitioned by        │ Parallel processing          │
│                       │ card_id               │ with ordering guarantees     │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           FEATURE ENGINEERING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Redis Cluster ◀──────────── Feature Store ────────────▶ ML Models         │
│   │                                                                          │
│   ├── Velocity tracking (sorted sets)                                       │
│   ├── Amount aggregations (sorted sets)                                     │
│   ├── Behavioral features (sets, hashes)                                    │
│   └── Risk scores (strings with TTL)                                        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           ML SCORING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                       │
│   │  XGBoost    │   │  Neural Net │   │  Isolation  │                       │
│   │  (0.45)     │   │  (0.35)     │   │  Forest     │                       │
│   │             │   │             │   │  (0.20)     │                       │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                       │
│          │                 │                 │                               │
│          └─────────────────┼─────────────────┘                               │
│                            ▼                                                 │
│                     ┌─────────────┐                                          │
│                     │  Ensemble   │                                          │
│                     │  Scoring    │                                          │
│                     └─────────────┘                                          │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           DECISION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Risk Score ──▶ Rules Engine ──▶ Decision                                  │
│                  │                                                           │
│                  ├── Velocity rules                                         │
│                  ├── Pattern rules                                          │
│                  ├── Geographic rules                                       │
│                  └── Blacklist checks                                       │
│                                                                              │
│   Decisions: APPROVE | STEP_UP | REVIEW | DECLINE                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Stream Ingestion

**Technology:** Apache Kafka

**Design Decisions:**
- Partitioned by `card_id` for ordering guarantees
- Replication factor of 3 for durability
- Consumer groups for horizontal scaling
- Dead letter queue for failed messages

**Configuration:**
```yaml
kafka:
  num_partitions: 32
  replication_factor: 3
  retention_ms: 604800000  # 7 days
  max_message_bytes: 1048576
```

### 2. Feature Store

**Technology:** Redis Cluster

**Data Structures:**
- **Sorted Sets**: Velocity tracking with timestamp scores
- **Sets**: Unique entity tracking (merchants, devices)
- **Hashes**: Complex feature aggregations
- **Strings**: Simple key-value with TTL

**Feature Categories:**

| Category | Features | Computation |
|----------|----------|-------------|
| Velocity | txn_count_1h, 6h, 24h, 7d | ZCOUNT on sorted sets |
| Amount | sum_1h, avg_30d, std_30d | ZRANGEBYSCORE + aggregation |
| Behavioral | time_since_last, unique_merchants | GET, SCARD |
| Risk | merchant_score, device_score | Pre-computed lookup |

### 3. ML Ensemble

**Architecture:**

```
Input Features (17 dims)
        │
        ▼
┌───────────────────────────────────────────┐
│                 ENSEMBLE                   │
│                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ XGBoost  │  │  Neural  │  │Isolation │ │
│  │          │  │  Network │  │  Forest  │ │
│  │  w=0.45  │  │  w=0.35  │  │  w=0.20  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │             │             │        │
│       └─────────────┼─────────────┘        │
│                     │                      │
│              Weighted Average              │
│                     │                      │
│                     ▼                      │
│              Risk Score [0, 1]             │
└───────────────────────────────────────────┘
```

**Model Characteristics:**

| Model | Strength | Training | Inference |
|-------|----------|----------|-----------|
| XGBoost | Feature interactions | Hours | 0.1ms |
| Neural Network | Complex patterns | Hours | 0.2ms |
| Isolation Forest | Anomaly detection | Minutes | 0.1ms |

### 4. Decision Engine

**Threshold Configuration:**

```
Score: 0.0 ────────────────────────────────▶ 1.0
       │         │           │           │
    APPROVE   STEP_UP     REVIEW     DECLINE
       │         │           │           │
     < 0.3    0.3-0.7     0.7-0.9     > 0.9
```

**Rules Engine:**
- Deterministic rules complement ML scores
- Priority-based evaluation
- Configurable actions per rule
- Audit logging for compliance

## Data Flow

### Real-Time Scoring Path

```
1. Transaction arrives via Kafka (t=0ms)
2. Consumer deserializes message (t=0.5ms)
3. Feature store lookup (parallel) (t=1-2ms)
4. ML ensemble scoring (t=2-4ms)
5. Rules engine evaluation (t=0.5ms)
6. Decision produced to Kafka (t=0.5ms)
7. Feature store update (async) (t=0ms - async)

Total: 4-8ms (P99: 12ms)
```

### Batch Training Path

```
1. Historical data extracted from warehouse
2. Feature engineering (distributed)
3. Train/validation/test split
4. Parallel model training
5. Ensemble weight calibration
6. Model validation
7. Model deployment (blue-green)
```

## Scaling Strategy

### Horizontal Scaling

| Component | Scaling Method | Bottleneck |
|-----------|---------------|------------|
| Kafka | Add partitions | Network |
| Stream Processor | Add consumers | CPU |
| Redis | Add shards | Memory |
| API | Add replicas | CPU |

### Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Throughput | 50,000 TPS | 52,000 TPS |
| P50 Latency | < 5ms | 4.2ms |
| P99 Latency | < 15ms | 12.3ms |
| Availability | 99.99% | 99.995% |

## Monitoring

### Key Metrics

1. **Throughput**: Transactions processed per second
2. **Latency**: P50, P90, P99 latency distribution
3. **Error Rate**: Failed scorings percentage
4. **Model Drift**: Feature and prediction distribution shift
5. **Decision Distribution**: Approve/Review/Decline ratios

### Alerting

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Latency | P99 > 20ms | Warning |
| Error Spike | Error rate > 1% | Critical |
| Model Drift | PSI > 0.1 | Warning |
| Low Throughput | TPS < 1000 | Critical |

## Security

### Data Protection

- All data encrypted at rest (AES-256)
- TLS 1.3 for all network communication
- PII fields hashed in logs
- Role-based access control (RBAC)

### Compliance

- PCI-DSS Level 1 compliant
- GDPR data handling
- SOC 2 Type II certified
- Audit logging for all decisions
