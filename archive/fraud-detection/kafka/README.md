# Kafka Configuration

This directory contains Kafka topic configurations and related settings for the fraud detection streaming pipeline.

## Topics Overview

| Topic | Purpose | Partitions | Retention |
|-------|---------|------------|-----------|
| `transactions.incoming` | Raw transaction ingestion | 12 | 7 days |
| `transactions.scored` | Scored transactions output | 12 | 30 days |
| `fraud.alerts` | High-risk alerts | 6 | 90 days |
| `transactions.dlq` | Dead letter queue | 3 | 30 days |
| `audit.predictions` | Audit trail | 6 | 365 days |
| `features.updates` | Feature store sync | 12 | Compacted |

## Setup

### Create Topics

```bash
# Using Kafka CLI
kafka-topics.sh --create \
    --bootstrap-server localhost:9092 \
    --topic transactions.incoming \
    --partitions 12 \
    --replication-factor 3 \
    --config retention.ms=604800000 \
    --config compression.type=lz4

# Or using the setup script
python scripts/setup_kafka.py --config kafka/topics.yaml
```

### Verify Topics

```bash
kafka-topics.sh --describe \
    --bootstrap-server localhost:9092 \
    --topic transactions.incoming
```

## Message Formats

### Transaction Input

```json
{
  "transaction_id": "txn_abc123",
  "user_id": "usr_xyz789",
  "amount": 150.00,
  "currency": "USD",
  "merchant_id": "mrc_store001",
  "merchant_category": "retail",
  "timestamp": "2024-01-15T10:30:00Z",
  "device_fingerprint": "fp_device123",
  "ip_address_hash": "a1b2c3d4",
  "location": {"lat": 37.7749, "lon": -122.4194}
}
```

### Scored Transaction Output

```json
{
  "transaction_id": "txn_abc123",
  "risk_score": 0.15,
  "decision": "APPROVE",
  "model_scores": {
    "xgboost": 0.12,
    "neural_net": 0.18,
    "isolation_forest": 0.10
  },
  "processing_time_ms": 8,
  "timestamp": "2024-01-15T10:30:00.008Z"
}
```

## Consumer Configuration

For optimal performance with the fraud detection consumer:

```python
consumer_config = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'fraud-detection-consumer',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,  # Manual commit for exactly-once
    'max.poll.records': 500,
    'max.poll.interval.ms': 300000,
}
```

## Monitoring

Key Kafka metrics to monitor:

- `kafka_consumer_group_lag`: Consumer lag (alert if > 10,000)
- `kafka_topic_partition_current_offset`: Current offset
- `kafka_server_broker_topic_metrics_bytes_in_per_sec`: Throughput

## Troubleshooting

### High Consumer Lag

1. Check consumer health: `kafka-consumer-groups.sh --describe --group fraud-detection-consumer`
2. Verify partition assignment
3. Scale consumer instances if needed
4. Check for slow message processing

### Message Delivery Failures

1. Check DLQ for failed messages
2. Review error logs
3. Verify schema compatibility
4. Check network connectivity

## Security

All topics require authentication. See `acls` section in `topics.yaml` for permission matrix.
