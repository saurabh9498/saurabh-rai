# API Reference

This document provides detailed documentation for the Multi-Agent AI System REST API.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API supports API key authentication via header:

```
Authorization: Bearer <your-api-key>
```

## Endpoints

### Health Check

#### GET /health

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Query Endpoints

#### POST /api/v1/query

Execute a multi-agent query.

**Request Body:**
```json
{
  "query": "What are the key trends in our customer support tickets?",
  "agents": ["research", "analyst"],
  "context": {
    "time_range": "last_30_days",
    "department": "support"
  },
  "output_format": "detailed",
  "stream": false
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | The user's question or task |
| `agents` | array | No | Specific agents to use (default: auto-select) |
| `context` | object | No | Additional context for the query |
| `output_format` | string | No | "summary" or "detailed" (default: "detailed") |
| `stream` | boolean | No | Enable streaming response (default: false) |

**Response:**
```json
{
  "id": "query_abc123",
  "status": "completed",
  "execution_time": 4.2,
  "result": {
    "summary": "Analysis of support tickets reveals...",
    "detailed_analysis": "...",
    "visualizations": [],
    "sources": [],
    "confidence": 0.92
  },
  "agent_traces": [
    {
      "agent": "research",
      "actions": [],
      "duration": 1.8
    }
  ]
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized
- `500`: Server error

---

### Document Ingestion

#### POST /api/v1/ingest

Ingest documents into the knowledge base.

**Request Body (multipart/form-data):**
```
files: [file1.pdf, file2.txt]
metadata: {"source": "manual_upload", "category": "documentation"}
chunking_strategy: "recursive"
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | files | Yes | One or more files to ingest |
| `metadata` | JSON | No | Metadata to attach to documents |
| `chunking_strategy` | string | No | "fixed", "recursive", "sentence", "paragraph" |

**Response:**
```json
{
  "status": "success",
  "documents_processed": 2,
  "chunks_created": 45,
  "embeddings_generated": 45,
  "collection": "knowledge_base"
}
```

---

#### POST /api/v1/ingest/text

Ingest raw text documents.

**Request Body:**
```json
{
  "documents": [
    {
      "content": "Document content here...",
      "metadata": {"source": "api", "title": "Doc 1"}
    }
  ],
  "chunking_strategy": "recursive"
}
```

---

### RAG Queries

#### POST /api/v1/rag/query

Query the knowledge base directly.

**Request Body:**
```json
{
  "query": "What is our refund policy?",
  "top_k": 5,
  "filters": {
    "source": "policy_documents"
  },
  "use_reranking": true
}
```

**Response:**
```json
{
  "query": "What is our refund policy?",
  "results": [
    {
      "content": "Refund policy excerpt...",
      "score": 0.95,
      "metadata": {"source": "policy.pdf", "page": 3}
    }
  ],
  "context": "Assembled context for LLM..."
}
```

---

### Agent Management

#### GET /api/v1/agents

List available agents.

**Response:**
```json
{
  "agents": [
    {
      "name": "research",
      "description": "Information retrieval and synthesis",
      "tools": ["rag_query", "web_search", "summarize"],
      "status": "available"
    },
    {
      "name": "analyst",
      "description": "Data analysis and visualization",
      "tools": ["query_data", "statistical_analysis"],
      "status": "available"
    }
  ]
}
```

---

### Conversation Management

#### POST /api/v1/conversations

Create a new conversation.

**Response:**
```json
{
  "conversation_id": "conv_xyz789",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### GET /api/v1/conversations/{conversation_id}

Get conversation history.

#### DELETE /api/v1/conversations/{conversation_id}

Delete a conversation.

---

### Feedback

#### POST /api/v1/feedback

Submit feedback for a query response.

**Request Body:**
```json
{
  "query_id": "query_abc123",
  "rating": 5,
  "feedback_type": "positive",
  "comment": "Very helpful response"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The query parameter is required",
    "details": {}
  }
}
```

**Error Codes:**
- `INVALID_REQUEST`: Bad request parameters
- `UNAUTHORIZED`: Invalid or missing API key
- `NOT_FOUND`: Resource not found
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error

---

## Rate Limiting

- Default: 100 requests per minute
- Streaming endpoints: 20 concurrent connections

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705315800
```

---

## SDK Usage

### Python

```python
from multi_agent import MultiAgentClient

client = MultiAgentClient(api_key="your-key")

# Execute query
result = client.query(
    "Analyze customer feedback trends",
    agents=["research", "analyst"]
)

print(result.summary)
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest updates?"}'
```
