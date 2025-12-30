# System Architecture

This document describes the architecture of the Conversational AI Assistant, a production-grade voice-enabled AI system.

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Component Details](#component-details)
- [Scalability Considerations](#scalability-considerations)
- [Performance Optimizations](#performance-optimizations)

---

## Overview

The Conversational AI Assistant is a modular, scalable system that processes voice and text input to provide intelligent responses. It supports real-time streaming, multi-turn conversations, and can be deployed on edge devices or cloud infrastructure.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Applications                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Mobile    │  │    Web      │  │   Desktop   │  │    IoT      │    │
│  │    App      │  │   Browser   │  │    App      │  │   Device    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼───────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            API Gateway                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Load Balancer  │  Rate Limiter  │  Auth  │  Request Router     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│        REST API               │   │       WebSocket API            │
│   (Synchronous Requests)      │   │   (Real-time Streaming)        │
└───────────────┬───────────────┘   └───────────────┬───────────────┘
                │                                   │
                └───────────────┬───────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Processing Pipeline                              │
│                                                                          │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │     ASR     │──▶│     NLU     │──▶│   Dialog    │──▶│     TTS     │ │
│  │  (Whisper)  │   │  (Intent/   │   │   Manager   │   │  (Coqui)    │ │
│  │             │   │   Entity)   │   │             │   │             │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
│         │                 │                 │                 │         │
│         ▼                 ▼                 ▼                 ▼         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Shared Services                             │   │
│  │   Session Store  │  Feature Store  │  Model Registry  │  Cache   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Layer                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │    Redis    │   │  PostgreSQL │   │     S3      │   │ Prometheus  │ │
│  │  (Sessions) │   │   (Logs)    │   │  (Models)   │   │  (Metrics)  │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. API Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| REST API | FastAPI | Synchronous request handling |
| WebSocket | FastAPI + Starlette | Real-time bidirectional streaming |
| Load Balancer | NGINX / AWS ALB | Traffic distribution |
| Rate Limiter | Redis + Sliding Window | API protection |

### 2. Processing Pipeline

| Component | Technology | Latency Target |
|-----------|------------|----------------|
| ASR | OpenAI Whisper | < 500ms (1s audio) |
| NLU | Transformers | < 50ms |
| Dialog Manager | Rule-based + LLM | < 100ms |
| TTS | Coqui TTS | < 300ms (10 words) |

### 3. Data Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| Session Store | Redis | Active conversation state |
| Conversation Logs | PostgreSQL | Historical data, analytics |
| Model Storage | S3 / MinIO | Model artifacts |
| Metrics | Prometheus + Grafana | Observability |

---

## Data Flow

### Voice Input Flow

```
1. Client captures audio
2. Audio streamed via WebSocket (16kHz, 16-bit PCM)
3. Voice Activity Detection (VAD) identifies speech
4. ASR transcribes audio chunks
5. NLU extracts intent and entities
6. Dialog Manager updates state and generates response
7. TTS synthesizes response audio
8. Audio streamed back to client
```

### Text Input Flow

```
1. Client sends text via REST API
2. NLU extracts intent and entities
3. Dialog Manager updates state and generates response
4. Response returned as JSON
```

---

## Component Details

### Automatic Speech Recognition (ASR)

```
┌─────────────────────────────────────────────────────────────┐
│                      ASR Pipeline                            │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Audio   │──▶│   VAD    │──▶│ Whisper  │──▶│  Post-   │ │
│  │  Buffer  │   │ (Silero) │   │  Model   │   │ Process  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                              │
│  Features:                                                   │
│  • Streaming with partial results                           │
│  • Multi-language support                                   │
│  • Noise robustness                                         │
│  • Speaker diarization (optional)                           │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```yaml
asr:
  model_size: base  # tiny, base, small, medium, large
  language: en
  sample_rate: 16000
  chunk_size: 1024
  vad_threshold: 0.5
  beam_size: 5
```

### Natural Language Understanding (NLU)

```
┌─────────────────────────────────────────────────────────────┐
│                      NLU Pipeline                            │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │   Text   │──▶│  Intent  │──▶│  Entity  │──▶│Sentiment │ │
│  │ Preproc  │   │ Classify │   │ Extract  │   │ Analyze  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                              │
│  Models:                                                     │
│  • Intent: DistilBERT fine-tuned                            │
│  • NER: spaCy + custom rules                                │
│  • Sentiment: VADER + transformer                           │
└─────────────────────────────────────────────────────────────┘
```

**Supported Intents:**
- `greeting`, `goodbye`, `help`
- `book_flight`, `check_weather`, `set_reminder`
- `play_music`, `control_device`, `get_news`
- `fallback` (catch-all)

### Dialog Management

```
┌─────────────────────────────────────────────────────────────┐
│                   Dialog Manager                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  State Tracker                        │  │
│  │  • Turn count     • Intent history                   │  │
│  │  • Slot values    • Context window                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Dialog Policy                        │  │
│  │  • Rule-based for task completion                    │  │
│  │  • Slot filling logic                                │  │
│  │  • Fallback handling                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Response Generator                      │  │
│  │  • Template-based responses                          │  │
│  │  • Dynamic slot filling                              │  │
│  │  • LLM fallback (optional)                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Text-to-Speech (TTS)

```
┌─────────────────────────────────────────────────────────────┐
│                      TTS Pipeline                            │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │   Text   │──▶│   SSML   │──▶│  Coqui   │──▶│  Audio   │ │
│  │ Normalize│   │  Parser  │   │   TTS    │   │ Streamer │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                              │
│  Features:                                                   │
│  • Multiple voices                                          │
│  • SSML support (emphasis, pauses, pitch)                  │
│  • Streaming audio output                                   │
│  • Caching for repeated phrases                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Scalability Considerations

### Horizontal Scaling

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   API Pod 1     │ │   API Pod 2     │ │   API Pod N     │
│  (ASR + NLU +   │ │  (ASR + NLU +   │ │  (ASR + NLU +   │
│   Dialog + TTS) │ │   Dialog + TTS) │ │   Dialog + TTS) │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Redis Cluster  │
                    └─────────────────┘
```

### Resource Requirements

| Component | CPU | Memory | GPU |
|-----------|-----|--------|-----|
| ASR (Whisper base) | 2 cores | 2GB | Optional |
| NLU | 1 core | 1GB | - |
| Dialog | 0.5 core | 512MB | - |
| TTS | 1 core | 2GB | Optional |
| **Total per pod** | **4.5 cores** | **5.5GB** | **Optional** |

### Bottleneck Analysis

1. **ASR**: Most compute-intensive; GPU acceleration recommended
2. **TTS**: Second most intensive; can be cached
3. **WebSocket connections**: Memory-bound; use connection pooling

---

## Performance Optimizations

### Model Optimizations

| Technique | Component | Speedup |
|-----------|-----------|---------|
| TensorRT | ASR, NLU | 2-4x |
| ONNX Runtime | NLU | 1.5-2x |
| Model Quantization | All | 2x (INT8) |
| Batching | NLU | 3-5x |

### Caching Strategy

```python
# Response caching hierarchy
1. In-memory LRU cache (hot responses)
2. Redis cache (session-scoped)
3. CDN cache (static TTS audio)
```

### Latency Targets

| Metric | Target | Current |
|--------|--------|---------|
| ASR (1s audio) | < 500ms | 320ms |
| NLU inference | < 50ms | 28ms |
| Dialog processing | < 50ms | 15ms |
| TTS (10 words) | < 300ms | 180ms |
| End-to-end (voice) | < 1.5s | 1.1s |

---

## Security Considerations

### Data Protection

- Audio data encrypted in transit (TLS 1.3)
- PII detection and redaction in logs
- Session data encrypted at rest
- GDPR-compliant data retention

### Access Control

- API key authentication
- JWT tokens for WebSocket
- Role-based access control
- Rate limiting per user/API key

---

## Monitoring & Observability

### Metrics (Prometheus)

```
# Request metrics
conversational_ai_requests_total{endpoint, status}
conversational_ai_request_duration_seconds{endpoint}

# Component metrics
conversational_ai_asr_duration_seconds
conversational_ai_nlu_duration_seconds
conversational_ai_tts_duration_seconds

# Resource metrics
conversational_ai_active_sessions
conversational_ai_model_memory_bytes
```

### Logging

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "session_id": "sess_abc123",
  "component": "nlu",
  "intent": "book_flight",
  "confidence": 0.95,
  "latency_ms": 28
}
```

---

## Future Enhancements

1. **Multi-language support** - Expand beyond English
2. **Emotion detection** - Detect user emotion from voice
3. **Custom wake word** - "Hey Assistant" activation
4. **LLM integration** - GPT/Claude for complex responses
5. **Edge deployment** - Raspberry Pi / Jetson Nano support
