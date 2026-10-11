# API Reference

## Base URL

```
Production: https://api.recommendations.example.com/v1
Development: http://localhost:8000
```

## Authentication

All API requests require a Bearer token in the Authorization header:

```bash
curl -H "Authorization: Bearer <your-api-key>" \
     https://api.recommendations.example.com/v1/recommend
```

---

## Endpoints

### Get Recommendations

Retrieve personalized recommendations for a user.

**Endpoint:** `POST /recommend`

#### Request

```json
{
  "user_id": "string",
  "num_recommendations": 10,
  "context": {
    "device": "mobile|desktop|tablet",
    "page": "home|category|product|cart",
    "session_id": "string",
    "referrer": "string"
  },
  "filters": {
    "category": "string",
    "min_price": 0.0,
    "max_price": 1000.0,
    "in_stock": true,
    "brands": ["brand1", "brand2"]
  },
  "exclude_items": ["item_id_1", "item_id_2"],
  "diversity_factor": 0.3
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | Unique user identifier |
| `num_recommendations` | integer | No | Number of items to return (default: 10, max: 100) |
| `context` | object | No | Request context for personalization |
| `filters` | object | No | Filtering criteria |
| `exclude_items` | array | No | Item IDs to exclude from results |
| `diversity_factor` | float | No | 0.0-1.0, higher = more diverse results |

#### Response

```json
{
  "request_id": "req_abc123",
  "user_id": "user_456",
  "items": [
    {
      "item_id": "item_789",
      "score": 0.95,
      "rank": 1,
      "reason": "Based on your recent browsing",
      "metadata": {
        "category": "Electronics",
        "price": 299.99,
        "in_stock": true
      }
    }
  ],
  "latency_ms": 8.2,
  "model_version": "v2.3.1"
}
```

#### Example

```bash
curl -X POST https://api.recommendations.example.com/v1/recommend \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "user_id": "user_12345",
    "num_recommendations": 5,
    "context": {
      "device": "mobile",
      "page": "home"
    }
  }'
```

---

### Batch Recommendations

Get recommendations for multiple users in a single request.

**Endpoint:** `POST /recommend/batch`

#### Request

```json
{
  "user_ids": ["user_1", "user_2", "user_3"],
  "num_recommendations": 10,
  "context": {
    "device": "mobile"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_ids` | array | Yes | List of user IDs (max: 100) |
| `num_recommendations` | integer | No | Items per user (default: 10) |
| `context` | object | No | Shared context for all users |

#### Response

```json
{
  "request_id": "batch_xyz789",
  "results": {
    "user_1": {
      "items": [...],
      "latency_ms": 5.2
    },
    "user_2": {
      "items": [...],
      "latency_ms": 4.8
    }
  },
  "total_latency_ms": 15.3
}
```

---

### Similar Items

Find items similar to a given item.

**Endpoint:** `GET /similar/{item_id}`

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `item_id` | string | Yes | Source item ID |
| `k` | integer | No | Number of similar items (default: 10) |
| `exclude_same_category` | boolean | No | Exclude items from same category |

#### Response

```json
{
  "item_id": "item_123",
  "items": [
    {
      "item_id": "item_456",
      "score": 0.92,
      "rank": 1
    }
  ],
  "latency_ms": 3.1
}
```

#### Example

```bash
curl "https://api.recommendations.example.com/v1/similar/item_123?k=5"
```

---

### Record Feedback

Record user interactions for model improvement.

**Endpoint:** `POST /feedback`

#### Request

```json
{
  "user_id": "string",
  "item_id": "string",
  "event_type": "click|impression|add_to_cart|purchase",
  "context": {
    "position": 3,
    "page": "home",
    "recommendation_id": "req_abc123"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | User identifier |
| `item_id` | string | Yes | Item identifier |
| `event_type` | string | Yes | Type of interaction |
| `context` | object | No | Additional context |
| `timestamp` | string | No | ISO 8601 timestamp |

#### Response

```json
{
  "status": "recorded",
  "event_id": "evt_abc123"
}
```

---

### Health Check

Check API health status.

**Endpoint:** `GET /health`

#### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "redis": "healthy",
    "triton": "healthy",
    "faiss_index": "healthy"
  }
}
```

---

### Metrics

Get Prometheus metrics.

**Endpoint:** `GET /metrics`

Returns metrics in Prometheus text format:

```
# HELP recommendation_requests_total Total recommendation requests
# TYPE recommendation_requests_total counter
recommendation_requests_total{status="success"} 1234567

# HELP recommendation_latency_seconds Recommendation latency
# TYPE recommendation_latency_seconds histogram
recommendation_latency_seconds_bucket{le="0.01"} 98765
recommendation_latency_seconds_bucket{le="0.05"} 123456
```

---

## Error Responses

### Error Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid user_id format",
    "details": {
      "field": "user_id",
      "constraint": "must be non-empty string"
    }
  },
  "request_id": "req_abc123"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 422 | Invalid request parameters |
| `USER_NOT_FOUND` | 404 | User ID not found |
| `ITEM_NOT_FOUND` | 404 | Item ID not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Dependent service unavailable |
| `TIMEOUT` | 504 | Request timeout |

---

## Rate Limits

| Tier | Requests/second | Burst |
|------|-----------------|-------|
| Free | 10 | 20 |
| Pro | 100 | 200 |
| Enterprise | 1000 | 2000 |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312200
```

---

## SDKs

### Python

```python
from recommendation_client import RecommendationClient

client = RecommendationClient(api_key="your-api-key")

# Get recommendations
recs = client.recommend(
    user_id="user_123",
    num_recommendations=10,
    context={"device": "mobile"}
)

for item in recs.items:
    print(f"{item.item_id}: {item.score:.2f}")
```

### JavaScript

```javascript
import { RecommendationClient } from '@example/recommendation-sdk';

const client = new RecommendationClient({ apiKey: 'your-api-key' });

const recs = await client.recommend({
  userId: 'user_123',
  numRecommendations: 10,
  context: { device: 'mobile' }
});

recs.items.forEach(item => {
  console.log(`${item.itemId}: ${item.score.toFixed(2)}`);
});
```

---

## Webhooks

Configure webhooks to receive real-time notifications.

### Events

| Event | Description |
|-------|-------------|
| `model.updated` | New model deployed |
| `experiment.started` | A/B test started |
| `experiment.completed` | A/B test completed |

### Payload

```json
{
  "event": "model.updated",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "model_version": "v2.3.1",
    "previous_version": "v2.3.0",
    "metrics": {
      "auc": 0.773,
      "ndcg_10": 0.445
    }
  }
}
```
