# Data Directory

This directory contains datasets for the Conversational AI Assistant.

---

## Directory Structure

```
data/
├── README.md                 # This file
├── sample/                   # Sample data for testing
│   ├── conversations.json    # Multi-turn dialogs (100 conversations)
│   ├── intents.json          # Intent training examples
│   ├── entities.json         # Entity annotations
│   └── audio/                # Sample audio files
│       ├── sample_01.wav
│       ├── sample_02.wav
│       └── ...
├── raw/                      # Raw data (not tracked in git)
│   ├── transcripts/
│   ├── audio/
│   └── annotations/
├── processed/                # Preprocessed data (not tracked)
│   ├── train.json
│   ├── valid.json
│   └── test.json
└── models/                   # Trained model artifacts (not tracked)
    ├── intent_classifier/
    └── entity_extractor/
```

---

## Data Schemas

### Conversations (`conversations.json`)

```json
{
  "conversation_id": "conv_001",
  "session_id": "session_abc123",
  "turns": [
    {
      "turn_id": 1,
      "speaker": "user",
      "text": "Hi, I need help booking a flight",
      "timestamp": "2024-01-15T10:30:00Z",
      "audio_file": "audio/conv_001_turn_01.wav",
      "asr_confidence": 0.95
    },
    {
      "turn_id": 2,
      "speaker": "assistant",
      "text": "I'd be happy to help you book a flight. Where would you like to go?",
      "timestamp": "2024-01-15T10:30:02Z",
      "intent": "flight_booking",
      "entities": [],
      "slots_filled": {}
    }
  ],
  "metadata": {
    "duration_seconds": 45,
    "num_turns": 8,
    "outcome": "success",
    "domain": "travel"
  }
}
```

### Intents (`intents.json`)

```json
{
  "intent": "book_flight",
  "examples": [
    "I want to book a flight",
    "Book me a ticket to New York",
    "I need to fly to London next week",
    "Can you help me find flights?"
  ],
  "slots": ["origin", "destination", "date", "passengers"],
  "responses": [
    "I'd be happy to help you book a flight. Where would you like to go?",
    "Sure! What's your destination?"
  ]
}
```

### Entities (`entities.json`)

```json
{
  "entity_type": "city",
  "examples": [
    {"text": "New York", "value": "NYC", "synonyms": ["NY", "New York City"]},
    {"text": "Los Angeles", "value": "LAX", "synonyms": ["LA", "L.A."]},
    {"text": "London", "value": "LHR", "synonyms": ["London UK"]}
  ]
}
```

### Audio Files

| Format | Sample Rate | Channels | Bit Depth |
|--------|-------------|----------|-----------|
| WAV | 16000 Hz | Mono | 16-bit |
| Duration | 1-30 seconds | - | - |

---

## Data Setup Options

### Option 1: Generate Sample Data (Recommended)

```bash
# Generate synthetic conversations and audio
python scripts/generate_sample_data.py \
    --conversations 100 \
    --audio-samples 50 \
    --output data/sample/
```

### Option 2: Use Public Datasets

```bash
# MultiWOZ (task-oriented dialogs)
wget https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2.zip
unzip MultiWOZ_2.2.zip -d data/raw/multiwoz/

# Common Voice (ASR training)
# See: https://commonvoice.mozilla.org/en/datasets

# SLURP (spoken language understanding)
# See: https://zenodo.org/record/4274930
```

### Option 3: Connect to Annotation Platform

```bash
# Configure data source in .env
ANNOTATION_PLATFORM=labelstudio
LABELSTUDIO_URL=http://localhost:8080
LABELSTUDIO_API_KEY=your-api-key

# Sync annotations
python scripts/sync_annotations.py --source labelstudio --target data/raw/
```

---

## Sample Data Statistics

| Dataset | Count | Size | Description |
|---------|-------|------|-------------|
| conversations.json | 100 | 250 KB | Multi-turn dialogs |
| intents.json | 25 | 45 KB | Intent definitions |
| entities.json | 50 | 30 KB | Entity types |
| audio/*.wav | 50 | 15 MB | Sample utterances |

### Intent Distribution

| Intent | Examples | Percentage |
|--------|----------|------------|
| greeting | 150 | 15% |
| book_flight | 120 | 12% |
| check_weather | 100 | 10% |
| set_reminder | 95 | 9.5% |
| play_music | 90 | 9% |
| get_news | 85 | 8.5% |
| other | 360 | 36% |

### Entity Statistics

| Entity Type | Unique Values | Examples |
|-------------|---------------|----------|
| city | 50 | New York, London, Tokyo |
| date | N/A | tomorrow, next Friday |
| time | N/A | 9am, noon, 3:30pm |
| person | 100 | John, Sarah, Dr. Smith |
| number | N/A | one, 5, twenty |

---

## Audio Data Guidelines

### Recording Quality Requirements

- **Sample Rate**: 16000 Hz (16 kHz)
- **Channels**: Mono
- **Format**: WAV (PCM 16-bit)
- **SNR**: > 20 dB recommended
- **Duration**: 1-30 seconds per utterance

### Preprocessing Pipeline

```bash
# Convert audio to required format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav

# Batch convert
for f in *.mp3; do
    ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"
done
```

---

## Privacy & Compliance

⚠️ **Important:** Sample data is 100% synthetic and contains no real user information.

For production data:
- All PII is removed or anonymized
- Audio files contain only synthetic speech
- No real user conversations included
- Compliant with GDPR, CCPA, and COPPA
- Data retention: Configurable per policy

### Consent Requirements

For real user data collection:
1. Explicit opt-in consent required
2. Clear disclosure of data usage
3. Right to deletion upon request
4. Audio recordings encrypted at rest

---

## Data Quality Checks

Run validation before training:

```bash
# Validate all data files
python scripts/validate_data.py --path data/sample/

# Expected output:
# ✓ conversations.json: 100 valid conversations
# ✓ intents.json: 25 intents, 1000 examples
# ✓ entities.json: 50 entity types
# ✓ Audio files: 50 valid, 0 corrupted
# ✓ Schema validation passed
# ✓ No duplicate examples found
```

---

## Updating Sample Data

To regenerate sample data with different parameters:

```bash
# Larger dataset for training
python scripts/generate_sample_data.py \
    --conversations 1000 \
    --intents 50 \
    --audio-samples 500 \
    --output data/large/

# Domain-specific data
python scripts/generate_sample_data.py \
    --domain healthcare \
    --conversations 200 \
    --output data/healthcare/
```

---

## Data Versioning

We use DVC for data versioning:

```bash
# Track data with DVC
dvc add data/sample/

# Push to remote storage
dvc push

# Pull data on new machine
dvc pull
```
