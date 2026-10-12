# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-language support (Spanish, French, German)
- Emotion detection from voice
- Custom wake word detection
- LLM-powered response generation

---

## [1.0.0] - 2024-12-28

### Added
- **Automatic Speech Recognition (ASR)**
  - Whisper integration (tiny, base, small, medium, large)
  - Streaming audio input with Voice Activity Detection (VAD)
  - Audio preprocessing (noise reduction, normalization)
  - Multi-format support (WAV, MP3, FLAC, WebM)

- **Natural Language Understanding (NLU)**
  - Intent classification with transformer models
  - Named Entity Recognition (NER)
  - Sentiment analysis
  - Configurable intent definitions via YAML

- **Dialog Management**
  - Rule-based dialog policy
  - Dialog state tracking across turns
  - Context management with slot filling
  - Multi-turn conversation support

- **Text-to-Speech (TTS)**
  - Coqui TTS integration
  - Multiple voice options
  - SSML parsing for expressive speech
  - Real-time audio streaming

- **API & Infrastructure**
  - FastAPI REST endpoints
  - WebSocket for real-time streaming
  - Docker and Docker Compose setup
  - GPU-accelerated inference option

- **Documentation**
  - Architecture documentation with diagrams
  - API reference with OpenAPI spec
  - Deployment guide
  - Quick start guide

### Performance
- ASR latency: < 500ms for 1s audio
- NLU inference: < 50ms
- TTS latency: < 300ms for 10 words
- End-to-end: < 1.5s voice-to-voice

---

## [0.3.0] - 2024-12-20

### Added
- WebSocket streaming for real-time conversations
- Voice Activity Detection (VAD) with Silero
- Audio streaming output for TTS

### Changed
- Migrated to async/await for all I/O operations
- Improved dialog state serialization

### Fixed
- Memory leak in audio buffer handling
- WebSocket reconnection issues

---

## [0.2.0] - 2024-12-10

### Added
- TTS integration with Coqui TTS
- SSML support for expressive speech
- Sentiment analysis in NLU pipeline
- Prometheus metrics endpoint

### Changed
- Upgraded Whisper to latest version
- Improved intent classification accuracy

### Fixed
- Audio format conversion edge cases
- Entity extraction for compound names

---

## [0.1.0] - 2024-12-01

### Added
- Initial project structure
- Whisper ASR integration
- Basic NLU with intent classification
- Simple dialog manager
- FastAPI endpoints
- Docker development environment
- Unit test framework

---

[Unreleased]: https://github.com/yourusername/conversational-ai/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/conversational-ai/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/yourusername/conversational-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yourusername/conversational-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/conversational-ai/releases/tag/v0.1.0
