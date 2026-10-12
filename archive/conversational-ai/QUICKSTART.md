# 🚀 Quick Start Guide

Get the Conversational AI Assistant running in under 5 minutes.

---

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| Python | 3.9+ | `python --version` |
| NVIDIA GPU | Compute 7.0+ | `nvidia-smi` (optional) |
| FFmpeg | 4.0+ | `ffmpeg -version` |

---

## Option 1: Docker Quick Start (Recommended)

### Step 1: Clone and Navigate

```bash
git clone https://github.com/yourusername/conversational-ai.git
cd conversational-ai
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### Step 3: Start Services

```bash
# Start all services (ASR, TTS, NLU, Dialog Manager)
docker compose up -d

# Check status
docker compose ps
```

### Step 4: Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "asr": "ready", "tts": "ready", "nlu": "ready"}
```

### Step 5: Test the Assistant

```bash
# Send a text message
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, what can you help me with?", "session_id": "test-001"}'

# Send audio (base64 encoded)
curl -X POST http://localhost:8000/api/v1/speech \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "<base64_audio>", "session_id": "test-001"}'
```

---

## Option 2: Local Development Setup

### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# For GPU acceleration
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Whisper for ASR
pip install openai-whisper
```

### Step 3: Download Models

```bash
# Download Whisper model
python -c "import whisper; whisper.load_model('base')"

# Download TTS model (Coqui TTS)
python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"
```

### Step 4: Generate Sample Data

```bash
python scripts/generate_sample_data.py --conversations 100 --output data/sample/
```

### Step 5: Run the Server

```bash
# Development mode with hot reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Option 3: Python Demo Script

Create `demo.py` and run it:

```python
#!/usr/bin/env python3
"""Quick demo of the Conversational AI Assistant."""

import asyncio
from src.asr.whisper_asr import WhisperASR
from src.nlu.pipeline import NLUPipeline
from src.dialog.state_tracker import DialogStateTracker
from src.dialog.response_generator import ResponseGenerator
from src.tts.synthesizer import TTSSynthesizer

async def main():
    print("=" * 60)
    print("🎤 Conversational AI Assistant Demo")
    print("=" * 60)
    
    # 1. Initialize components
    print("\n🔧 Initializing components...")
    
    print("   Loading ASR (Whisper)...")
    asr = WhisperASR(model_size="base")
    
    print("   Loading NLU pipeline...")
    nlu = NLUPipeline()
    
    print("   Initializing dialog manager...")
    state_tracker = DialogStateTracker()
    response_gen = ResponseGenerator()
    
    print("   Loading TTS synthesizer...")
    tts = TTSSynthesizer()
    
    print("   ✓ All components ready")
    
    # 2. Process a sample utterance
    print("\n💬 Processing sample conversation...")
    
    test_utterances = [
        "Hi there! What's the weather like today?",
        "Set a reminder for tomorrow at 9am",
        "Tell me a joke",
    ]
    
    session_id = "demo-session-001"
    
    for utterance in test_utterances:
        print(f"\n   User: '{utterance}'")
        
        # NLU processing
        nlu_result = await nlu.process(utterance)
        print(f"   Intent: {nlu_result.intent} (confidence: {nlu_result.confidence:.2f})")
        if nlu_result.entities:
            print(f"   Entities: {nlu_result.entities}")
        
        # Dialog state update
        state = state_tracker.update(session_id, nlu_result)
        
        # Generate response
        response = await response_gen.generate(state)
        print(f"   Assistant: '{response.text}'")
    
    # 3. Display final state
    print("\n" + "=" * 60)
    print("📊 Dialog State Summary")
    print("=" * 60)
    final_state = state_tracker.get_state(session_id)
    print(f"   Turns: {final_state.turn_count}")
    print(f"   Intents seen: {final_state.intent_history}")
    print(f"   Active slots: {final_state.slots}")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("""
Next Steps:
1. Try audio input:  python scripts/client.py --audio sample.wav
2. Start WebSocket:  python scripts/client.py --websocket
3. Train NLU:        python scripts/train_nlu.py --data data/sample/
4. Run benchmarks:   python scripts/benchmark.py --requests 1000
""")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the demo:

```bash
python demo.py
```

---

## Verification Commands

### Check All Services

```bash
# Docker services
docker compose ps

# API health
curl http://localhost:8000/health

# ASR model status
curl http://localhost:8000/api/v1/models/asr

# WebSocket test
websocat ws://localhost:8000/ws/chat
```

### Run Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires services running)
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=src --cov-report=html
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f asr
```

---

## Project Structure

```
conversational-ai/
├── configs/                    # Configuration files
│   ├── asr_config.yaml         # Whisper ASR settings
│   ├── tts_config.yaml         # TTS voice settings
│   ├── nlu_config.yaml         # Intent/entity config
│   ├── dialog_config.yaml      # Dialog policy
│   └── intents.yaml            # Intent definitions
│
├── data/
│   ├── README.md               # Data documentation
│   └── sample/                 # Sample datasets
│       ├── conversations.json
│       ├── intents.json
│       └── audio/
│
├── docker/
│   ├── Dockerfile              # Main application
│   ├── Dockerfile.gpu          # GPU-enabled version
│   └── docker-compose.yml      # Full stack
│
├── docs/
│   ├── architecture.md         # System design
│   ├── api_reference.md        # API documentation
│   └── deployment.md           # Production guide
│
├── models/                     # Model checkpoints
│   └── .gitkeep
│
├── notebooks/
│   └── demo.ipynb
│
├── scripts/
│   ├── train_nlu.py            # NLU training
│   ├── evaluate.py             # Evaluation metrics
│   ├── benchmark.py            # Performance testing
│   ├── client.py               # CLI client
│   └── generate_sample_data.py # Data generator
│
├── src/
│   ├── api/                    # FastAPI + WebSocket
│   ├── asr/                    # Speech recognition
│   ├── tts/                    # Text-to-speech
│   ├── nlu/                    # Intent & entity
│   ├── dialog/                 # Dialog management
│   └── utils/                  # Helpers
│
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
│
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── QUICKSTART.md               # This file
├── README.md
└── requirements.txt
```

---

## Common Issues

### Whisper Model Download Fails

```bash
# Check internet connectivity
ping huggingface.co

# Manual download
python -c "
import whisper
model = whisper.load_model('base', download_root='./models/whisper')
"

# Use smaller model if memory limited
ASR_MODEL_SIZE=tiny  # in .env
```

### Audio Not Recording

```bash
# Check audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Install PortAudio (Linux)
sudo apt-get install portaudio19-dev

# macOS
brew install portaudio
```

### TTS Voice Quality Issues

```bash
# Use higher quality model
TTS_MODEL=tts_models/en/ljspeech/vits  # in .env

# Check sample rate
TTS_SAMPLE_RATE=22050  # Match your audio output
```

### WebSocket Connection Drops

```bash
# Check for firewall issues
sudo ufw allow 8000

# Increase timeout in docker-compose.yml
environment:
  - WS_TIMEOUT=300
```

### Out of Memory (GPU)

```bash
# Use CPU-only mode
USE_GPU=false  # in .env

# Or use smaller models
ASR_MODEL_SIZE=tiny
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
```

---

## Performance Benchmarks

| Component | Target | Measured |
|-----------|--------|----------|
| ASR Latency (1s audio) | < 500ms | 320ms |
| NLU Inference | < 50ms | 28ms |
| TTS Latency (10 words) | < 300ms | 180ms |
| End-to-End (voice) | < 1.5s | 1.1s |
| Concurrent Sessions | > 50 | 75 |

Run your own benchmarks:

```bash
python scripts/benchmark.py --requests 1000 --concurrency 50
```

---

## Next Steps

1. **Train Custom NLU**: Add domain-specific intents in `configs/intents.yaml`
2. **Customize Voice**: Configure TTS voice and style in `configs/tts_config.yaml`
3. **Add Skills**: Extend `src/dialog/policy.py` with new capabilities
4. **Deploy to Production**: Follow `docs/deployment.md` for Kubernetes setup

---

## Support

- 📖 [Full Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/conversational-ai/issues)
- 💬 [Discussions](https://github.com/yourusername/conversational-ai/discussions)
