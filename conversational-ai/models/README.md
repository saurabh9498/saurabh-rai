# Models Directory

This directory stores downloaded and trained model artifacts for the Conversational AI Assistant.

## Directory Structure

```
models/
├── whisper/              # Whisper ASR models
│   ├── base/
│   ├── small/
│   └── medium/
├── tts/                  # Text-to-Speech models
│   ├── tacotron2-DDC/
│   └── vits/
├── nlu/                  # NLU models
│   ├── intent_classifier/
│   ├── entity_extractor/
│   └── sentiment/
├── spacy/                # spaCy language models
│   └── en_core_web_sm/
└── README.md
```

## Downloading Models

Use the download script to fetch all required models:

```bash
# Download all models with defaults
python scripts/download_models.py

# Download specific model sizes
python scripts/download_models.py --asr-model small --tts-model vits

# Download only specific components
python scripts/download_models.py --components asr nlu

# List available models
python scripts/download_models.py --list

# Verify installed models
python scripts/download_models.py --verify
```

## Model Details

### Whisper ASR Models

| Model | Size | VRAM | Speed | Use Case |
|-------|------|------|-------|----------|
| tiny | 39MB | ~1GB | ~32x | Testing |
| base | 74MB | ~1GB | ~16x | Development |
| small | 244MB | ~2GB | ~6x | Production (CPU) |
| medium | 769MB | ~5GB | ~2x | Production (GPU) |
| large-v3 | 1.5GB | ~10GB | ~1x | High accuracy |

### TTS Models

| Model | Quality | Speed | Notes |
|-------|---------|-------|-------|
| tacotron2-DDC | Good | Fast | Default, reliable |
| vits | Excellent | Medium | More natural |
| glow-tts | Good | Fast | Lightweight |

### NLU Models

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| Intent | DistilBERT | 250MB | Fine-tuned |
| Entity | spaCy NER | 12MB | en_core_web_sm |
| Sentiment | VADER + BERT | 150MB | Hybrid approach |

## Storage Locations

Models are cached in these locations:

```bash
# Whisper cache
~/.cache/whisper/
# or set WHISPER_CACHE_DIR

# Hugging Face cache
~/.cache/huggingface/
# or set HF_HOME

# TTS cache
~/.cache/TTS/

# spaCy models
# Installed in site-packages
```

## Docker Volume

In Docker, models are stored in a persistent volume:

```yaml
volumes:
  - model-cache:/app/models
```

## Custom Models

To use custom fine-tuned models:

```python
# Custom intent classifier
from src.nlu.intent_classifier import IntentClassifier
classifier = IntentClassifier(model_path='models/nlu/custom_intent/')

# Custom TTS voice
from src.tts.synthesizer import TTSSynthesizer
tts = TTSSynthesizer(model_path='models/tts/custom_voice/')
```

## Model Versioning

For production, use model versioning:

```
models/
├── whisper/
│   ├── v1.0/
│   └── v1.1/
└── nlu/
    ├── v1.0/
    └── v2.0/
```

## .gitignore

Model files are excluded from git. Use Git LFS for version control:

```bash
git lfs track "*.pt"
git lfs track "*.bin"
git lfs track "*.onnx"
```

Or use cloud storage (S3, GCS) for model artifacts.
