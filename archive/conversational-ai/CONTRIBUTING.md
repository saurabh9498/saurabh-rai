# Contributing to Conversational AI Assistant

Thank you for your interest in contributing to the Conversational AI Assistant! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what's best for the community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- FFmpeg (for audio processing)
- Git

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/conversational-ai.git
cd conversational-ai
git remote add upstream https://github.com/ORIGINAL_OWNER/conversational-ai.git
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Or install in editable mode
pip install -e ".[dev]"
```

### 3. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1 portaudio19-dev

# macOS
brew install ffmpeg portaudio

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
```

### 4. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env with your local settings
```

### 6. Download Models

```bash
# Download Whisper model for ASR
python -c "import whisper; whisper.load_model('base')"

# Download TTS model (optional)
python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"
```

---

## Project Structure

```
conversational-ai/
├── src/
│   ├── api/              # FastAPI endpoints and WebSocket
│   │   ├── main.py       # Application entry point
│   │   ├── routes.py     # REST endpoints
│   │   ├── schemas.py    # Pydantic models
│   │   └── websocket.py  # WebSocket handlers
│   ├── asr/              # Automatic Speech Recognition
│   │   ├── whisper_asr.py    # Whisper integration
│   │   ├── audio_processor.py # Audio preprocessing
│   │   ├── streaming.py      # Streaming ASR
│   │   └── vad.py            # Voice Activity Detection
│   ├── nlu/              # Natural Language Understanding
│   │   ├── intent_classifier.py  # Intent detection
│   │   ├── entity_extractor.py   # NER
│   │   ├── pipeline.py           # NLU pipeline
│   │   └── sentiment.py          # Sentiment analysis
│   ├── dialog/           # Dialog Management
│   │   ├── state_tracker.py      # State tracking
│   │   ├── policy.py             # Dialog policy
│   │   ├── context_manager.py    # Context handling
│   │   └── response_generator.py # Response generation
│   ├── tts/              # Text-to-Speech
│   │   ├── synthesizer.py    # TTS synthesis
│   │   ├── ssml_parser.py    # SSML parsing
│   │   └── audio_streamer.py # Audio streaming
│   └── utils/            # Utilities
│       ├── audio.py      # Audio utilities
│       ├── logging.py    # Logging config
│       └── metrics.py    # Prometheus metrics
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── conftest.py       # Pytest fixtures
├── configs/              # Configuration files
├── data/                 # Data and samples
├── docker/               # Docker files
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks
└── scripts/              # Utility scripts
```

---

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

```python
# Good: Use type hints
def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    language: str = "en",
) -> TranscriptionResult:
    """Transcribe audio to text.
    
    Args:
        audio: Audio samples as numpy array
        sample_rate: Audio sample rate in Hz
        language: Language code (ISO 639-1)
        
    Returns:
        TranscriptionResult with text and metadata
    """
    ...

# Good: Use dataclasses for data structures
@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    language: str
    segments: List[Segment]
```

### Code Formatting

```bash
# Format code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/
```

### Docstrings

Use Google-style docstrings:

```python
def process_utterance(
    text: str,
    session_id: str,
    context: Optional[Dict] = None,
) -> DialogResponse:
    """Process a user utterance and generate response.
    
    This function handles the complete NLU -> Dialog -> Response pipeline.
    
    Args:
        text: User's input text
        session_id: Unique session identifier
        context: Optional context from previous turns
        
    Returns:
        DialogResponse containing the assistant's response and updated state
        
    Raises:
        SessionNotFoundError: If session_id is invalid
        NLUError: If intent classification fails
        
    Example:
        >>> response = process_utterance("Book a flight to Paris", "sess_123")
        >>> print(response.text)
        "I'd be happy to help you book a flight to Paris..."
    """
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_nlu.py -v

# Run tests matching pattern
pytest -k "test_intent" -v

# Run only unit tests
pytest tests/unit/ -v

# Run integration tests (requires services)
RUN_INTEGRATION_TESTS=true pytest tests/integration/ -v
```

### Writing Tests

```python
# tests/unit/test_nlu.py
import pytest
from src.nlu.intent_classifier import IntentClassifier

class TestIntentClassifier:
    """Tests for intent classification."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return IntentClassifier(model_path="models/intent")
    
    def test_greeting_intent(self, classifier):
        """Test classification of greeting intents."""
        result = classifier.classify("Hello there!")
        
        assert result.intent == "greeting"
        assert result.confidence > 0.8
    
    @pytest.mark.parametrize("text,expected_intent", [
        ("Book a flight", "book_flight"),
        ("What's the weather", "check_weather"),
        ("Set a reminder", "set_reminder"),
    ])
    def test_various_intents(self, classifier, text, expected_intent):
        """Test multiple intent types."""
        result = classifier.classify(text)
        assert result.intent == expected_intent
```

### Test Coverage Requirements

- Minimum 70% overall coverage
- New features must include tests
- Critical paths require 90%+ coverage

---

## Submitting Changes

### Branch Naming

```
feature/add-multilingual-support
bugfix/fix-audio-buffer-overflow
docs/update-api-reference
refactor/simplify-dialog-state
```

### Commit Messages

Follow conventional commits:

```
feat(asr): add streaming transcription support

- Implement chunked audio processing
- Add Voice Activity Detection
- Support real-time partial results

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat(component): description"
   ```

3. **Keep branch updated**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature
   # Open PR on GitHub
   ```

5. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No new warnings
   ```

---

## Review Process

### What We Look For

1. **Functionality**: Does it work correctly?
2. **Tests**: Are changes well-tested?
3. **Performance**: Any performance implications?
4. **Security**: Any security concerns?
5. **Documentation**: Is it documented?
6. **Style**: Does it follow our standards?

### Timeline

- Initial review: 2-3 business days
- Follow-up reviews: 1-2 business days
- Complex changes may take longer

---

## Component-Specific Guidelines

### ASR (Speech Recognition)

- Test with various audio qualities
- Consider different accents/languages
- Validate sample rate handling
- Test streaming edge cases

### NLU (Natural Language Understanding)

- Include diverse training examples
- Test entity extraction thoroughly
- Validate confidence thresholds
- Consider edge cases (empty input, very long input)

### Dialog Management

- Test multi-turn conversations
- Validate state persistence
- Test slot filling logic
- Consider context window limits

### TTS (Text-to-Speech)

- Test various text inputs
- Validate SSML parsing
- Check audio quality
- Test streaming output

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Features**: Open a Feature Request issue
- **Security**: Email security@example.com

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Thanked in documentation

Thank you for contributing! 🎉
