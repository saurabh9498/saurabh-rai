# API Reference

Complete REST API documentation for the Retail Vision Analytics platform.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API requests require an API key in the header:

```
X-API-Key: your-api-key-here
```

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/system/health` | Detailed system status |
| GET | `/system/metrics` | Performance metrics |
| GET | `/streams` | List camera streams |
| POST | `/streams` | Add new stream |
| GET | `/streams/{id}` | Get stream details |
| DELETE | `/streams/{id}` | Remove stream |
| GET | `/analytics/summary` | Analytics summary |
| GET | `/analytics/journeys` | Customer journeys |
| GET | `/analytics/queues` | Queue metrics |
| GET | `/analytics/heatmap` | Heatmap data |
| GET | `/alerts` | List alerts |
| POST | `/alerts` | Create alert |
| PATCH | `/alerts/{id}` | Update alert |

---

## System Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### System Health

```
GET /api/v1/system/health
```

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 86400,
  "timestamp": "2024-01-15T10:30:00Z",
  "cpu_util_percent": 45.2,
  "ram_util_percent": 62.1,
  "gpu_util_percent": 78.5,
  "disk_util_percent": 35.0,
  "temperature_celsius": 58.0,
  "streams_active": 4,
  "streams_total": 5,
  "fps_total": 119.88,
  "detections_per_second": 85.3,
  "inference_latency_ms": 4.5,
  "cloud_connected": true,
  "pending_uploads": 12,
  "last_sync_time": "2024-01-15T10:29:30Z"
}
```

---

## Camera Stream Endpoints

### List Streams

```
GET /api/v1/streams
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| status | string | Filter by status (online/offline/error) |
| store_id | string | Filter by store ID |

**Response:**
```json
[
  {
    "config": {
      "stream_id": "cam-entrance-1",
      "name": "Entrance Camera 1",
      "uri": "rtsp://192.168.1.10:554/stream1",
      "protocol": "rtsp",
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "enabled": true,
      "store_id": "store-001",
      "location": "entrance"
    },
    "status": "online",
    "fps_actual": 29.97,
    "frames_processed": 125000,
    "detections_total": 45000,
    "last_frame_time": "2024-01-15T10:30:00Z"
  }
]
```

### Add Stream

```
POST /api/v1/streams
```

**Request Body:**
```json
{
  "stream_id": "cam-aisle-3",
  "name": "Aisle 3 Camera",
  "uri": "rtsp://192.168.1.13:554/stream1",
  "protocol": "rtsp",
  "width": 1920,
  "height": 1080,
  "fps": 30,
  "enabled": true,
  "store_id": "store-001",
  "location": "aisle"
}
```

### Delete Stream

```
DELETE /api/v1/streams/{stream_id}
```

---

## Analytics Endpoints

### Analytics Summary

```
GET /api/v1/analytics/summary
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| time_range | string | 1h, 6h, 24h, 7d, 30d, or custom |
| start_time | datetime | Custom start (ISO format) |
| end_time | datetime | Custom end (ISO format) |
| store_id | string | Filter by store |

**Response:**
```json
{
  "time_range": "24h",
  "start_time": "2024-01-14T10:30:00Z",
  "end_time": "2024-01-15T10:30:00Z",
  "total_visitors": 1250,
  "peak_visitors": 85,
  "peak_time": "2024-01-15T12:30:00Z",
  "avg_visitors_per_hour": 52.1,
  "conversion_rate": 0.42,
  "avg_dwell_time_seconds": 485,
  "cart_usage_rate": 0.65,
  "avg_queue_length": 4.2,
  "avg_wait_time_seconds": 145,
  "total_abandonments": 23,
  "busiest_zone": "aisle-1",
  "zone_traffic": {
    "entrance": 1250,
    "aisle-1": 890,
    "aisle-2": 720,
    "checkout": 525
  }
}
```

### Customer Journeys

```
GET /api/v1/analytics/journeys
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| stream_id | string | Filter by stream |
| start_time | datetime | Start time filter |
| end_time | datetime | End time filter |
| converted_only | boolean | Only converted customers |
| page | integer | Page number (default: 1) |
| page_size | integer | Items per page (default: 20) |

**Response:**
```json
{
  "items": [
    {
      "journey_id": "journey-000123",
      "track_id": 123,
      "stream_id": "cam-entrance-1",
      "start_time": "2024-01-15T10:15:00Z",
      "end_time": "2024-01-15T10:28:00Z",
      "duration_seconds": 780,
      "zones_visited": ["entrance", "aisle-1", "aisle-2", "checkout"],
      "zone_dwell_times": {
        "entrance": 30,
        "aisle-1": 240,
        "aisle-2": 180,
        "checkout": 330
      },
      "entry_point": "entrance",
      "exit_point": "checkout",
      "converted": true,
      "cart_detected": true
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20,
  "pages": 8
}
```

### Queue Metrics

```
GET /api/v1/analytics/queues
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| lane_id | string | Filter by lane |
| stream_id | string | Filter by stream |
| hours | integer | Hours of data (default: 24) |

**Response:**
```json
[
  {
    "lane_id": "checkout-1",
    "stream_id": "cam-checkout-1",
    "timestamp": "2024-01-15T10:30:00Z",
    "queue_length": 5,
    "avg_wait_time_seconds": 145,
    "max_wait_time_seconds": 280,
    "service_rate": 1.2,
    "abandonment_count": 1,
    "staffing_recommendation": 2
  }
]
```

### Heatmap Data

```
GET /api/v1/analytics/heatmap
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| stream_id | string | Stream ID (required) |
| time_range | string | Time range (default: 1h) |
| resolution | integer | Grid resolution (default: 96) |

**Response:**
```json
{
  "stream_id": "cam-entrance-1",
  "start_time": "2024-01-15T09:30:00Z",
  "end_time": "2024-01-15T10:30:00Z",
  "resolution": [96, 54],
  "data": [[0.1, 0.2, ...], ...],
  "hotspots": [
    {
      "x": 0.15,
      "y": 0.85,
      "intensity": 0.95,
      "label": "Entrance area"
    }
  ],
  "max_value": 1.0
}
```

---

## Alert Endpoints

### List Alerts

```
GET /api/v1/alerts
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| status | string | active, acknowledged, resolved |
| severity | string | info, warning, critical |
| stream_id | string | Filter by stream |
| limit | integer | Max alerts (default: 50) |

**Response:**
```json
[
  {
    "alert_id": "alert-0001",
    "alert_type": "queue_length_exceeded",
    "severity": "warning",
    "status": "active",
    "stream_id": "cam-checkout-1",
    "timestamp": "2024-01-15T10:25:00Z",
    "message": "Queue length exceeded threshold",
    "details": {
      "threshold": 8,
      "actual": 12
    },
    "acknowledged_at": null,
    "acknowledged_by": null,
    "resolved_at": null
  }
]
```

### Create Alert

```
POST /api/v1/alerts
```

**Request Body:**
```json
{
  "alert_type": "custom_alert",
  "severity": "warning",
  "stream_id": "cam-aisle-1",
  "message": "Suspicious activity detected",
  "details": {
    "location": "Electronics section",
    "confidence": 0.85
  }
}
```

### Update Alert

```
PATCH /api/v1/alerts/{alert_id}
```

**Request Body:**
```json
{
  "status": "acknowledged",
  "acknowledged_by": "operator@store.com"
}
```

---

## WebSocket Endpoints

### Real-time Detections

```
ws://localhost:8000/ws/detections
```

**Message Format:**
```json
{
  "stream_id": "cam-entrance-1",
  "frame_number": 12500,
  "timestamp": "2024-01-15T10:30:00.123Z",
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox": {
        "x": 150,
        "y": 200,
        "width": 80,
        "height": 180
      },
      "track_id": 42
    }
  ]
}
```

### Real-time Alerts

```
ws://localhost:8000/ws/alerts
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 409 | Conflict |
| 500 | Internal Server Error |

---

## Rate Limiting

- 1000 requests per minute per API key
- WebSocket connections limited to 100 per API key

Headers returned:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1705315860
```
