# API Reference

## Overview

The Fraud Detection API provides REST endpoints for real-time transaction scoring.

**Base URL:** `http://localhost:8000`

**Content-Type:** `application/json`

---

## Endpoints

### Health Check

#### `GET /health`

Check API health status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 86400.5,
  "version": "1.0.0"
}
```

---

### Score Transaction

#### `POST /v1/score`

Score a single transaction for fraud.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `transaction_id` | string | Yes | Unique transaction identifier |
| `card_id` | string | Yes | Card/account identifier |
| `amount` | float | Yes | Transaction amount (> 0) |
| `merchant_id` | string | Yes | Merchant identifier |
| `merchant_category` | string | No | Merchant category code |
| `timestamp` | string | No | ISO 8601 timestamp |
| `channel` | string | No | Transaction channel |
| `ip_address` | string | No | Client IP address |
| `device_id` | string | No | Device identifier |
| `location` | object | No | Lat/lon coordinates |

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_abc123",
    "card_id": "card_xyz789",
    "amount": 150.00,
    "merchant_id": "merch_001",
    "merchant_category": "retail",
    "timestamp": "2024-01-15T10:30:00Z",
    "channel": "online",
    "ip_address": "192.168.1.1",
    "device_id": "device_123"
  }'
```

**Response:**

```json
{
  "transaction_id": "txn_abc123",
  "risk_score": 0.15,
  "decision": "approve",
  "xgboost_score": 0.12,
  "neural_net_score": 0.18,
  "isolation_forest_score": 0.10,
  "risk_factors": [],
  "latency_ms": 5.2
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `risk_score` | float | Combined risk score (0-1) |
| `decision` | string | Decision: approve/step_up/review/decline |
| `xgboost_score` | float | XGBoost model score |
| `neural_net_score` | float | Neural network score |
| `isolation_forest_score` | float | Anomaly detection score |
| `risk_factors` | array | Identified risk factors |
| `latency_ms` | float | Processing time in milliseconds |

---

### Batch Score

#### `POST /v1/score/batch`

Score multiple transactions in a batch.

**Request Body:**

```json
{
  "transactions": [
    {
      "transaction_id": "txn_001",
      "card_id": "card_abc",
      "amount": 100.00,
      "merchant_id": "merch_001"
    },
    {
      "transaction_id": "txn_002",
      "card_id": "card_xyz",
      "amount": 250.00,
      "merchant_id": "merch_002"
    }
  ]
}
```

**Response:**

```json
{
  "results": [
    {
      "transaction_id": "txn_001",
      "risk_score": 0.08,
      "decision": "approve",
      "latency_ms": 3.1
    },
    {
      "transaction_id": "txn_002",
      "risk_score": 0.45,
      "decision": "step_up",
      "latency_ms": 3.2
    }
  ],
  "total_latency_ms": 12.5,
  "avg_latency_ms": 6.25
}
```

**Limits:**
- Maximum 100 transactions per batch
- Timeout: 30 seconds

---

### Model Information

#### `GET /v1/model/info`

Get information about the loaded model.

**Response:**

```json
{
  "model_type": "ensemble",
  "version": "1.0.0",
  "models": ["xgboost", "neural_net", "isolation_forest"],
  "weights": {
    "xgboost": 0.45,
    "neural_net": 0.35,
    "isolation_forest": 0.20
  },
  "thresholds": {
    "approve": 0.3,
    "review": 0.7,
    "decline": 0.9
  }
}
```

---

### Metrics

#### `GET /metrics`

Prometheus metrics endpoint.

**Response:** Prometheus text format

```
# HELP fraud_requests_total Total fraud scoring requests
# TYPE fraud_requests_total counter
fraud_requests_total{endpoint="score",status="success"} 1523456

# HELP fraud_request_latency_seconds Request latency
# TYPE fraud_request_latency_seconds histogram
fraud_request_latency_seconds_bucket{endpoint="score",le="0.005"} 1200000
```

---

## Decision Logic

### Risk Score Interpretation

| Score Range | Decision | Action |
|-------------|----------|--------|
| 0.0 - 0.3 | `approve` | Auto-approve transaction |
| 0.3 - 0.7 | `step_up` | Require additional authentication |
| 0.7 - 0.9 | `review` | Queue for manual review |
| 0.9 - 1.0 | `decline` | Auto-decline transaction |

### Risk Factors

Possible risk factors returned:

| Factor | Description |
|--------|-------------|
| `High risk across multiple models` | Multiple models flag high risk |
| `Transaction pattern is highly unusual` | Anomaly detection triggered |
| `High velocity: many transactions in last hour` | Velocity limit exceeded |
| `First transaction with this merchant` | New merchant relationship |
| `First transaction from this device` | New device detected |
| `Amount significantly above average` | Amount deviation > 3σ |

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 422 | Validation Error - Schema violation |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

### Example Errors

**Validation Error (422):**

```json
{
  "detail": [
    {
      "loc": ["body", "amount"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

---

## Rate Limiting

| Tier | Limit | Window |
|------|-------|--------|
| Standard | 1,000 req/min | Per API key |
| Premium | 10,000 req/min | Per API key |
| Enterprise | Unlimited | - |

---

## SDKs

### Python Client

```python
from fraud_detection import FraudClient

client = FraudClient("http://localhost:8000")

result = client.score({
    "transaction_id": "txn_001",
    "card_id": "card_abc",
    "amount": 100.0,
    "merchant_id": "merch_001"
})

print(f"Risk: {result.risk_score}, Decision: {result.decision}")
```

### cURL

```bash
# Score transaction
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"txn_001","card_id":"card_abc","amount":100,"merchant_id":"merch_001"}'
```
