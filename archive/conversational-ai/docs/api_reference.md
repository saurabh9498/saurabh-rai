# API Reference

Complete API documentation for the Conversational AI Assistant.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [REST API](#rest-api)
- [WebSocket API](#websocket-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## Overview

The API provides two interfaces:

| Interface | Protocol | Use Case |
|-----------|----------|----------|
| REST API | HTTP/HTTPS | Text-based chat, configuration |
| WebSocket | WS/WSS | Real-time voice streaming |

**Base URL:** `https://api.example.com/v1`

**WebSocket URL:** `wss://api.example.com/ws`

---

## Authentication

### API Key Authentication

Include your API key in the request header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.example.com/v1/chat
```

### JWT Authentication (WebSocket)

For WebSocket connections, obtain a JWT token first:

```bash
# Get JWT token
curl -X POST https://api.example.com/v1/auth/token \
     -H "Authorization: Bearer YOUR_API_KEY"

# Connect with JWT
wscat -c "wss://api.example.com/ws/chat?token=YOUR_JWT_TOKEN"
```

---

## REST API

### Health Check

Check service health status.

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "asr": "ready",
    "nlu": "ready",
    "dialog": "ready",
    "tts": "ready"
  },
  "uptime_seconds": 3600
}
```

---

### Chat (Text)

Send a text message and receive a response.

```
POST /chat
```

**Request Body:**

```json
{
  "text": "Book a flight to Paris",
  "session_id": "sess_abc123",
  "context": {
    "user_name": "John",
    "preferences": {}
  }
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | Yes | User's input text |
| session_id | string | No | Session ID for multi-turn (auto-generated if omitted) |
| context | object | No | Additional context |

**Response:**

```json
{
  "response_id": "resp_xyz789",
  "session_id": "sess_abc123",
  "text": "I'd be happy to help you book a flight to Paris. When would you like to travel?",
  "intent": {
    "name": "book_flight",
    "confidence": 0.95
  },
  "entities": [
    {
      "type": "destination",
      "value": "paris",
      "text": "Paris",
      "start": 20,
      "end": 25
    }
  ],
  "sentiment": {
    "label": "neutral",
    "score": 0.8
  },
  "dialog_state": {
    "turn_count": 1,
    "slots": {
      "destination": "paris"
    },
    "missing_slots": ["date", "origin"]
  },
  "audio_url": null,
  "metadata": {
    "processing_time_ms": 150
  }
}
```

---

### Speech (Audio Input)

Send audio and receive text/audio response.

```
POST /speech
```

**Request Body:**

```json
{
  "audio_base64": "UklGRiQA...",
  "audio_format": "wav",
  "sample_rate": 16000,
  "session_id": "sess_abc123",
  "return_audio": true
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_base64 | string | Yes | Base64-encoded audio |
| audio_format | string | No | Format: wav, mp3, webm (default: wav) |
| sample_rate | int | No | Sample rate in Hz (default: 16000) |
| session_id | string | No | Session ID |
| return_audio | bool | No | Return TTS audio (default: false) |

**Response:**

```json
{
  "response_id": "resp_xyz789",
  "session_id": "sess_abc123",
  "transcription": {
    "text": "Book a flight to Paris",
    "confidence": 0.95,
    "language": "en"
  },
  "text": "I'd be happy to help you book a flight to Paris.",
  "intent": {
    "name": "book_flight",
    "confidence": 0.95
  },
  "audio_base64": "UklGRiQA...",
  "audio_format": "wav",
  "metadata": {
    "asr_time_ms": 320,
    "nlu_time_ms": 28,
    "tts_time_ms": 180,
    "total_time_ms": 550
  }
}
```

---

### Session Management

#### Get Session

```
GET /sessions/{session_id}
```

**Response:**

```json
{
  "session_id": "sess_abc123",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:05:00Z",
  "turn_count": 5,
  "state": {
    "current_intent": "book_flight",
    "slots": {
      "destination": "paris",
      "date": "tomorrow"
    }
  },
  "history": [
    {
      "turn": 1,
      "user": "Book a flight to Paris",
      "assistant": "When would you like to travel?"
    }
  ]
}
```

#### Delete Session

```
DELETE /sessions/{session_id}
```

**Response:**

```json
{
  "message": "Session deleted successfully",
  "session_id": "sess_abc123"
}
```

---

### NLU Analysis

Analyze text without dialog processing.

```
POST /nlu/analyze
```

**Request Body:**

```json
{
  "text": "Set a reminder for tomorrow at 9am to call mom"
}
```

**Response:**

```json
{
  "text": "Set a reminder for tomorrow at 9am to call mom",
  "intent": {
    "name": "set_reminder",
    "confidence": 0.98
  },
  "entities": [
    {
      "type": "date",
      "value": "tomorrow",
      "text": "tomorrow",
      "start": 22,
      "end": 30
    },
    {
      "type": "time",
      "value": "09:00",
      "text": "9am",
      "start": 34,
      "end": 37
    },
    {
      "type": "task",
      "value": "call mom",
      "text": "call mom",
      "start": 41,
      "end": 49
    }
  ],
  "sentiment": {
    "label": "neutral",
    "score": 0.75
  },
  "tokens": ["Set", "a", "reminder", "for", "tomorrow", "at", "9am", "to", "call", "mom"]
}
```

---

### TTS Synthesis

Synthesize text to speech.

```
POST /tts/synthesize
```

**Request Body:**

```json
{
  "text": "Hello! How can I help you today?",
  "voice": "default",
  "speed": 1.0,
  "format": "wav",
  "ssml": false
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | Yes | Text to synthesize |
| voice | string | No | Voice ID (default: "default") |
| speed | float | No | Speed multiplier 0.5-2.0 (default: 1.0) |
| format | string | No | Output format: wav, mp3 (default: wav) |
| ssml | bool | No | Parse as SSML (default: false) |

**Response:**

```json
{
  "audio_base64": "UklGRiQA...",
  "format": "wav",
  "sample_rate": 22050,
  "duration_seconds": 2.5,
  "metadata": {
    "voice": "default",
    "processing_time_ms": 180
  }
}
```

---

### Model Information

```
GET /models
```

**Response:**

```json
{
  "models": {
    "asr": {
      "name": "whisper-base",
      "version": "20231117",
      "languages": ["en", "es", "fr", "de", "ja"]
    },
    "nlu": {
      "intent_classifier": "distilbert-intent-v1",
      "entity_extractor": "spacy-en-core-web-sm"
    },
    "tts": {
      "name": "coqui-tacotron2-ddc",
      "voices": ["default", "female-1", "male-1"]
    }
  }
}
```

---

## WebSocket API

### Connection

Connect to the WebSocket endpoint:

```javascript
const ws = new WebSocket('wss://api.example.com/ws/chat?token=YOUR_JWT_TOKEN');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Message Types

#### Client → Server

**Start Session:**
```json
{
  "type": "session.start",
  "session_id": "sess_abc123",
  "config": {
    "language": "en",
    "return_audio": true
  }
}
```

**Send Text:**
```json
{
  "type": "text.input",
  "text": "Hello",
  "session_id": "sess_abc123"
}
```

**Send Audio Chunk:**
```json
{
  "type": "audio.chunk",
  "audio_base64": "UklGRiQA...",
  "is_final": false
}
```

**End Audio:**
```json
{
  "type": "audio.end"
}
```

#### Server → Client

**Session Started:**
```json
{
  "type": "session.started",
  "session_id": "sess_abc123"
}
```

**Transcription (Partial):**
```json
{
  "type": "transcription.partial",
  "text": "Book a fli...",
  "is_final": false
}
```

**Transcription (Final):**
```json
{
  "type": "transcription.final",
  "text": "Book a flight to Paris",
  "confidence": 0.95
}
```

**Response:**
```json
{
  "type": "response",
  "text": "I'd be happy to help you book a flight.",
  "intent": "book_flight",
  "audio_base64": "UklGRiQA..."
}
```

**Error:**
```json
{
  "type": "error",
  "code": "INVALID_AUDIO",
  "message": "Audio format not supported"
}
```

### Streaming Audio Example

```javascript
// Initialize audio capture
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

// Send audio chunks
mediaRecorder.ondataavailable = async (event) => {
  const reader = new FileReader();
  reader.onloadend = () => {
    const base64 = reader.result.split(',')[1];
    ws.send(JSON.stringify({
      type: 'audio.chunk',
      audio_base64: base64,
      is_final: false
    }));
  };
  reader.readAsDataURL(event.data);
};

// Start recording
mediaRecorder.start(100); // 100ms chunks

// Stop and finalize
mediaRecorder.stop();
ws.send(JSON.stringify({ type: 'audio.end' }));
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request body",
    "details": [
      {
        "field": "text",
        "message": "Field is required"
      }
    ]
  },
  "request_id": "req_abc123"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 400 | Invalid request parameters |
| UNAUTHORIZED | 401 | Invalid or missing API key |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMITED | 429 | Too many requests |
| ASR_ERROR | 500 | Speech recognition failed |
| NLU_ERROR | 500 | NLU processing failed |
| TTS_ERROR | 500 | Speech synthesis failed |
| INTERNAL_ERROR | 500 | Internal server error |

---

## Rate Limiting

### Limits

| Tier | Requests/min | WebSocket Connections |
|------|--------------|----------------------|
| Free | 60 | 1 |
| Pro | 600 | 10 |
| Enterprise | 6000 | 100 |

### Rate Limit Headers

```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 599
X-RateLimit-Reset: 1705312800
```

### Rate Limit Response

```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "retry_after": 60
  }
}
```

---

## SDK Examples

### Python

```python
from conversational_ai import Client

client = Client(api_key="YOUR_API_KEY")

# Text chat
response = client.chat("Book a flight to Paris")
print(response.text)

# Voice chat
with open("audio.wav", "rb") as f:
    response = client.speech(f, return_audio=True)
    print(response.transcription)
    response.save_audio("response.wav")
```

### JavaScript

```javascript
import { ConversationalAI } from 'conversational-ai-sdk';

const client = new ConversationalAI({ apiKey: 'YOUR_API_KEY' });

// Text chat
const response = await client.chat('Book a flight to Paris');
console.log(response.text);

// Streaming
const stream = client.streamChat();
stream.on('partial', (text) => console.log('Partial:', text));
stream.on('response', (response) => console.log('Final:', response));
stream.sendAudio(audioBlob);
```

---

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:

```
GET /openapi.json
```

Interactive documentation:

```
GET /docs      # Swagger UI
GET /redoc     # ReDoc
```
