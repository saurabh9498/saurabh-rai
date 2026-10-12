# Conversational AI — Reference Architecture

> Reference architecture for production conversational AI systems. Demonstrates component structure, configuration patterns, and API surface for a streaming voice pipeline (ASR → NLU → Dialog Management → TTS).

**This is a reference architecture, not a deployed system.** It shows the patterns I work with on enterprise conversational AI — multi-turn dialogue, slot filling, on-prem-friendly component design. The patterns here informed my production work at Deloitte on a HIPAA-compliant patient engagement platform (1.8M+ patients, 3 health systems, NLP scheduling agents on NVIDIA Triton + NIM, no-show rates 18% → 10%, $4M annual savings).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONVERSATIONAL AI PLATFORM                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │    Audio     │───▶│   WebSocket  │───▶│   Streaming  │───▶│   Client   │ │
│  │    Input     │    │   Gateway    │    │   Response   │    │   Device   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   ▲                            │
│         ▼                   ▼                   │                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                         SPEECH PIPELINE                                  ││
│  │  ┌─────────────────────────────────────────────────────────────────────┐││
│  │  │                      ASR (Speech-to-Text)                           │││
│  │  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │││
│  │  │  │   Audio     │   │   Whisper   │   │   Streaming             │   │││
│  │  │  │   VAD       │──▶│   Encoder   │──▶│   Decoder               │   │││
│  │  │  │   (Silero)  │   │   (GPU)     │   │   (Beam Search)         │   │││
│  │  │  └─────────────┘   └─────────────┘   └─────────────────────────┘   │││
│  │  │            Voice Activity Detection → Transcription in <500ms       │││
│  │  └─────────────────────────────────────────────────────────────────────┘││
│  │                                    │                                     ││
│  │                                    ▼                                     ││
│  │  ┌─────────────────────────────────────────────────────────────────────┐││
│  │  │                      NLU (Understanding)                            │││
│  │  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │││
│  │  │  │   Intent    │   │   Entity    │   │   Sentiment             │   │││
│  │  │  │   Classifier│   │   Extractor │   │   Analysis              │   │││
│  │  │  │   (BERT)    │   │   (NER)     │   │   (RoBERTa)             │   │││
│  │  │  └─────────────┘   └─────────────┘   └─────────────────────────┘   │││
│  │  │            Intent → Entities → User Emotional State                 │││
│  │  └─────────────────────────────────────────────────────────────────────┘││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                         DIALOG MANAGEMENT                                ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  ││
│  │  │  State Tracker  │  │  Policy Engine  │  │  Response Generator     │  ││
│  │  │  (Belief State) │  │  (Rule + ML)    │  │  (Retrieval + LLM)      │  ││
│  │  │  Context Memory │  │  Action Select  │  │  Template Filling       │  ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  ││
│  │     Track context for 12+ turns with slot filling and confirmation      ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                      TTS (Text-to-Speech)                                ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  ││
│  │  │  Text Normalizer│  │  VITS/Coqui     │  │  Audio Streamer         │  ││
│  │  │  (Numbers,      │──▶│  Neural TTS     │──▶│  (Chunked Output)       │  ││
│  │  │   Abbreviations)│  │  Multi-Speaker  │  │  WebSocket Push         │  ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  ││
│  │            Natural speech synthesis with prosody and emotion control     ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Component Structure

### Speech Pipeline

| Module | Purpose |
|---|---|
| `src/asr/` | Streaming ASR with Whisper + Silero VAD |
| `src/nlu/` | Intent classification, entity extraction, sentiment |
| `src/dialog/` | Belief state tracking, rule-based policy, template responses |
| `src/tts/` | Text normalization, neural TTS, audio streaming |
| `src/api/` | FastAPI service with WebSocket streaming |

### Configuration

The `configs/` directory shows how each pipeline stage is parameterized:
- `asr_config.yaml` — model selection, VAD thresholds, sample rate
- `nlu_config.yaml` — intent model, NER model, confidence thresholds
- `dialog_config.yaml` — state tracker history depth, policy type, response generator
- `tts_config.yaml` — voice selection, sample rate, streaming chunk size
- `intents.yaml` — intent taxonomy and slot definitions

Configs are intentionally lightweight — production systems typically expand these significantly per-domain (e.g., adding HIPAA-specific PII redaction, region-specific compliance flags, or per-tenant dialog policies).

---

## Why This Architecture

A few patterns this reference makes explicit:

**Streaming-first throughout the pipeline.** ASR emits partial transcriptions; NLU runs incrementally; TTS streams audio chunks. This minimizes user-perceived latency vs request-response designs.

**Component isolation.** Each stage (ASR, NLU, Dialog, TTS) is a swap-out boundary. In the Deloitte healthcare deployment, this isolation let us update the dialog policy weekly while EHR integrations stayed on a quarterly release cadence — a critical separation when one side of the system has clinical-IT change-control requirements and the other doesn't.

**On-prem-friendly design.** No required external API calls for inference; all models can be served on Triton/NIM in the same network as the application. This matters in regulated industries (HIPAA, financial services) where data sovereignty is a hard constraint.

**KV-cache awareness in the dialog layer.** For LLM-backed response generation in multi-turn conversations, routing requests to the same inference replica that holds the conversation prefix in cache is a 5-10x latency win. The dialog state tracker exposes hooks for cache-aware routing decisions; the actual routing happens at the inference serving layer (Triton/Dynamo).

---

## Tech Stack

**Speech & Language Models:** OpenAI Whisper (ASR), Silero VAD, BERT/RoBERTa (NLU), VITS/Coqui (TTS)
**Serving:** FastAPI, WebSocket streaming, Docker
**Configuration:** YAML-driven, env-overridable
**Inference (production):** Triton Inference Server, NVIDIA NIM (in the related Deloitte deployment)

---

## What's Here vs. Production

This reference architecture demonstrates the structure and component contracts I work with on production conversational AI systems. The production deployment that informed this work was internal to Deloitte and a healthcare client — that codebase is not public. What you can read here is the architectural thinking, configuration approach, and component boundaries that ship cleanly into a regulated-industry deployment.

If you're working on similar systems and want to compare notes — happy to talk.

---

## Related Work

- **Multi-Agent Orchestration** ([`../multi-agent-orchestration`](../multi-agent-orchestration)) — Builder + Judge agent pattern for NL-to-SQL, applicable to dialog-driven query systems
- **GPU ML Pipeline** ([`../gpu-ml-pipeline`](../gpu-ml-pipeline)) — Triton serving and KEDA autoscaling patterns relevant to inference-heavy conversational systems

---

## License

MIT — see [LICENSE](./LICENSE)
