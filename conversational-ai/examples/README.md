# Examples

Ready-to-run example scripts demonstrating common use cases for the Conversational AI Assistant.

## Available Examples

### 1. Basic Chat (`basic_chat.py`)

Interactive text-based chatbot demonstrating the full NLU → Dialog → Response pipeline.

```bash
# Interactive mode
python examples/basic_chat.py

# With custom session
python examples/basic_chat.py --session-id user123

# Single message mode
python examples/basic_chat.py --single-message "What's the weather like?"
```

**Interactive Commands:**
- `quit` - Exit the chat
- `reset` - Clear conversation history
- `history` - Show conversation history
- `debug` - Toggle debug mode

**What it demonstrates:**
- NLU intent classification and entity extraction
- Dialog state tracking across turns
- Context-aware response generation
- Session management

---

### 2. Voice Assistant (`voice_assistant.py`)

Full voice-enabled assistant with speech recognition and synthesis.

```bash
# Default settings
python examples/voice_assistant.py

# With specific models
python examples/voice_assistant.py \
    --asr-model small \
    --tts-model vits \
    --tts-voice female

# Process audio file
python examples/voice_assistant.py \
    --audio-file input.wav \
    --output response.wav

# Text mode (no audio hardware required)
python examples/voice_assistant.py --text-mode
```

**What it demonstrates:**
- Real-time speech recognition with Whisper
- Voice activity detection (VAD)
- Text-to-speech synthesis
- End-to-end voice conversation loop

**Requirements:**
```bash
pip install pyaudio soundfile
# On Ubuntu: sudo apt-get install portaudio19-dev
```

---

### 3. Custom Intent (`custom_intent.py`)

Define and train custom intents for domain-specific applications.

```bash
# Train custom model
python examples/custom_intent.py --train --epochs 15

# Test single message
python examples/custom_intent.py --test "Schedule a meeting tomorrow at 3pm"

# Interactive testing
python examples/custom_intent.py --interactive
```

**Built-in Custom Intents:**
| Intent | Description | Example |
|--------|-------------|---------|
| `schedule_meeting` | Create a new meeting | "Set up a meeting with John" |
| `cancel_meeting` | Cancel existing meeting | "Cancel my 3pm appointment" |
| `check_calendar` | View schedule | "What's on my calendar today?" |
| `reschedule_meeting` | Move a meeting | "Push back standup by 30 mins" |

**What it demonstrates:**
- Defining custom intents with training examples
- Training intent classifiers
- Custom entity extraction patterns
- Domain-specific response templates
- Intent handlers with slot filling

---

## Prerequisites

Before running examples, ensure you have:

1. **Installed dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Downloaded models:**
   ```bash
   python scripts/download_models.py
   ```

3. **For voice examples:**
   ```bash
   pip install pyaudio soundfile
   # Download ASR model
   python scripts/download_models.py --components asr
   ```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download required models
python scripts/download_models.py --components nlu

# Run basic chat example
python examples/basic_chat.py
```

## Example Output

### Basic Chat
```
You: What's the weather in San Francisco?

Bot: The weather in San Francisco is currently 65°F and partly cloudy.
     [Intent: check_weather (94%)]
     [Entities: location=San Francisco]

You: How about tomorrow?

Bot: Tomorrow in San Francisco: High of 68°F, low of 55°F with sunny skies.
     [Intent: check_weather (89%)]
```

### Voice Assistant
```
🎤 Listening...
📝 Transcribing...
You said: Set a reminder for 5pm

🔊 Speaking: I'll set a reminder for 5pm. What would you like to be reminded about?
```

## Integration Examples

### Python API

```python
from examples.basic_chat import ChatBot

# Initialize
bot = ChatBot()

# Process messages
response = bot.process_message("Book a flight to New York")
print(response["text"])
print(f"Intent: {response['intent']}")
```

### Custom Intent Handler

```python
from examples.custom_intent import CalendarIntentHandler, CUSTOM_RESPONSES

handler = CalendarIntentHandler(CUSTOM_RESPONSES)

# Handle detected intent
response = handler.handle(
    intent="schedule_meeting",
    entities=[{"type": "datetime", "value": "tomorrow at 3pm"}],
    state={}
)
print(response["text"])
```

## Adding New Examples

To add your own example:

1. Create a new file in `examples/`
2. Add project root to path:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```
3. Import from `src/` modules
4. Add CLI with argparse
5. Update this README

## Troubleshooting

**Import errors:**
```bash
# Run from project root
cd conversational-ai
python examples/basic_chat.py
```

**Model not found:**
```bash
# Download models first
python scripts/download_models.py
```

**No audio device:**
```bash
# Use text mode
python examples/voice_assistant.py --text-mode
```

**PyAudio installation issues:**
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows - use pre-built wheel
pip install pipwin
pipwin install pyaudio
```
