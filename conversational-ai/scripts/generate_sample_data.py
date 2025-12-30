#!/usr/bin/env python3
"""
Generate synthetic sample data for the Conversational AI Assistant.

This script creates realistic synthetic datasets for development and testing,
including conversations, intents, entities, and audio samples.

Usage:
    python scripts/generate_sample_data.py --conversations 100 --output data/sample/
    python scripts/generate_sample_data.py --audio-samples 50 --seed 42
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# =============================================================================
# Intent Definitions
# =============================================================================

INTENTS = {
    "greeting": {
        "examples": [
            "Hi there",
            "Hello",
            "Hey",
            "Good morning",
            "Good afternoon",
            "What's up",
            "Hi, how are you",
            "Hello there",
            "Hey, what's going on",
            "Howdy",
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! I'm here to help. What do you need?",
        ],
        "slots": [],
    },
    "goodbye": {
        "examples": [
            "Bye",
            "Goodbye",
            "See you later",
            "Talk to you soon",
            "Thanks, bye",
            "That's all, thanks",
            "I'm done",
            "Exit",
            "Quit",
            "Close",
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! Let me know if you need anything else.",
        ],
        "slots": [],
    },
    "book_flight": {
        "examples": [
            "I want to book a flight",
            "Book me a ticket to {destination}",
            "I need to fly to {destination} on {date}",
            "Find flights from {origin} to {destination}",
            "Help me book a flight for {passengers} people",
            "I'm looking for a flight next {date}",
            "Can you search for flights to {destination}",
            "Book a round trip to {destination}",
            "One way ticket to {destination} please",
            "Flight to {destination} for {date}",
        ],
        "responses": [
            "I'd be happy to help you book a flight. Where would you like to go?",
            "Sure! What's your destination?",
            "Let me help you find flights. Where are you flying from and to?",
        ],
        "slots": ["origin", "destination", "date", "passengers"],
    },
    "check_weather": {
        "examples": [
            "What's the weather like",
            "How's the weather in {city}",
            "Will it rain {date}",
            "Weather forecast for {city}",
            "Is it going to be sunny {date}",
            "Tell me the temperature in {city}",
            "What's the forecast for {date}",
            "Do I need an umbrella {date}",
            "How cold is it in {city}",
            "Weather update for {city}",
        ],
        "responses": [
            "Let me check the weather for you.",
            "I'll look up the forecast right now.",
            "Checking the weather conditions...",
        ],
        "slots": ["city", "date"],
    },
    "set_reminder": {
        "examples": [
            "Set a reminder for {time}",
            "Remind me to {task} at {time}",
            "Create a reminder for {date}",
            "Don't let me forget to {task}",
            "Alert me at {time} to {task}",
            "Schedule a reminder for {date} at {time}",
            "I need a reminder about {task}",
            "Remind me {date} to {task}",
            "Set an alarm for {time}",
            "Wake me up at {time}",
        ],
        "responses": [
            "I'll set that reminder for you.",
            "Reminder created! I'll notify you at the right time.",
            "Got it! I'll remind you.",
        ],
        "slots": ["time", "date", "task"],
    },
    "play_music": {
        "examples": [
            "Play some music",
            "Play {song} by {artist}",
            "I want to listen to {genre}",
            "Put on some {genre} music",
            "Play my {playlist} playlist",
            "Shuffle my music",
            "Play something relaxing",
            "Can you play {artist}",
            "I'm in the mood for {genre}",
            "Start playing {song}",
        ],
        "responses": [
            "Playing music for you now.",
            "Starting your music...",
            "Here's some music you might enjoy.",
        ],
        "slots": ["song", "artist", "genre", "playlist"],
    },
    "get_news": {
        "examples": [
            "What's in the news",
            "Tell me the latest news",
            "News about {topic}",
            "What's happening in {location}",
            "Any updates on {topic}",
            "Read me the headlines",
            "What's trending today",
            "Sports news",
            "Tech news update",
            "Breaking news",
        ],
        "responses": [
            "Here are the latest headlines.",
            "Let me get you caught up on the news.",
            "Here's what's happening right now.",
        ],
        "slots": ["topic", "location"],
    },
    "control_device": {
        "examples": [
            "Turn on the {device}",
            "Turn off the {device}",
            "Set {device} to {value}",
            "Dim the lights to {value} percent",
            "Make it warmer in here",
            "Lower the {device}",
            "Increase the volume",
            "Mute the {device}",
            "Switch on {device} in {room}",
            "Adjust the thermostat to {value}",
        ],
        "responses": [
            "Done! I've adjusted your device.",
            "Your device settings have been updated.",
            "I've made that change for you.",
        ],
        "slots": ["device", "value", "room"],
    },
    "get_help": {
        "examples": [
            "Help",
            "What can you do",
            "Show me what you can do",
            "I need help",
            "How does this work",
            "What are your capabilities",
            "Guide me",
            "Instructions please",
            "How do I use this",
            "Tell me your features",
        ],
        "responses": [
            "I can help you with many things! Try asking about weather, reminders, or music.",
            "I'm your AI assistant. I can book flights, check weather, set reminders, and more!",
            "Here's what I can do for you...",
        ],
        "slots": [],
    },
    "fallback": {
        "examples": [
            "asdfghjkl",
            "I don't know",
            "blah blah",
            "test test",
            "random stuff",
        ],
        "responses": [
            "I'm not sure I understand. Could you rephrase that?",
            "I didn't quite catch that. Can you try again?",
            "Sorry, I'm not sure what you mean. Try asking differently.",
        ],
        "slots": [],
    },
}

# Entity values for slot filling
ENTITIES = {
    "city": ["New York", "Los Angeles", "Chicago", "London", "Paris", "Tokyo", "Sydney", "Berlin", "Toronto", "Miami"],
    "destination": ["New York", "Los Angeles", "London", "Paris", "Tokyo", "Sydney", "Rome", "Barcelona", "Dubai", "Singapore"],
    "origin": ["San Francisco", "Seattle", "Boston", "Denver", "Atlanta", "Dallas", "Phoenix", "Portland", "Austin", "Nashville"],
    "date": ["tomorrow", "next Monday", "this Friday", "next week", "in two days", "December 15th", "January 3rd", "the 20th"],
    "time": ["9am", "10:30", "noon", "3pm", "5:30pm", "7 o'clock", "morning", "evening", "at night", "in an hour"],
    "task": ["call mom", "buy groceries", "submit report", "exercise", "take medication", "water plants", "pay bills", "meeting prep"],
    "artist": ["The Beatles", "Taylor Swift", "Drake", "Adele", "Ed Sheeran", "Coldplay", "Beyoncé", "Bruno Mars"],
    "song": ["Bohemian Rhapsody", "Shape of You", "Blinding Lights", "Bad Guy", "Uptown Funk", "Happy", "Shake It Off"],
    "genre": ["jazz", "rock", "pop", "classical", "hip hop", "country", "electronic", "R&B", "indie", "folk"],
    "playlist": ["workout", "chill", "focus", "party", "sleep", "road trip", "morning coffee", "evening wind down"],
    "device": ["lights", "TV", "thermostat", "speaker", "fan", "air conditioner", "heater", "blinds"],
    "room": ["living room", "bedroom", "kitchen", "bathroom", "office", "garage", "basement"],
    "value": ["50", "75", "100", "low", "medium", "high", "25%", "maximum"],
    "topic": ["politics", "technology", "sports", "entertainment", "business", "health", "science", "weather"],
    "location": ["United States", "Europe", "Asia", "local", "worldwide", "my city"],
    "passengers": ["1", "2", "3", "4", "5", "just me", "two of us", "family of four"],
}


class ConversationGenerator:
    """Generates synthetic conversation data."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.conversation_counter = 0

    def fill_slots(self, template: str) -> Tuple[str, Dict[str, str]]:
        """Fill slot placeholders in template with random values."""
        filled = template
        extracted_entities = {}

        for slot_name, values in ENTITIES.items():
            placeholder = "{" + slot_name + "}"
            if placeholder in filled:
                value = random.choice(values)
                filled = filled.replace(placeholder, value, 1)
                extracted_entities[slot_name] = value

        return filled, extracted_entities

    def generate_turn(
        self,
        turn_id: int,
        speaker: str,
        intent: str = None,
        base_time: datetime = None,
    ) -> Dict[str, Any]:
        """Generate a single conversation turn."""
        if speaker == "user":
            intent_data = INTENTS[intent]
            template = random.choice(intent_data["examples"])
            text, entities = self.fill_slots(template)
            
            return {
                "turn_id": turn_id,
                "speaker": "user",
                "text": text,
                "timestamp": (base_time + timedelta(seconds=turn_id * 3)).isoformat() + "Z",
                "intent": intent,
                "entities": entities,
                "asr_confidence": round(random.uniform(0.85, 0.99), 2),
            }
        else:
            intent_data = INTENTS[intent]
            response = random.choice(intent_data["responses"])
            
            return {
                "turn_id": turn_id,
                "speaker": "assistant",
                "text": response,
                "timestamp": (base_time + timedelta(seconds=turn_id * 3 + 1)).isoformat() + "Z",
                "intent": intent,
                "entities": {},
                "slots_filled": {},
            }

    def generate_conversation(self) -> Dict[str, Any]:
        """Generate a complete multi-turn conversation."""
        self.conversation_counter += 1
        conv_id = f"conv_{self.conversation_counter:04d}"
        session_id = f"session_{uuid.uuid4().hex[:12]}"

        # Start with greeting, then random intents, end with goodbye
        num_exchanges = random.randint(2, 5)
        intent_sequence = ["greeting"]
        
        # Add random middle intents
        middle_intents = [k for k in INTENTS.keys() if k not in ["greeting", "goodbye", "fallback"]]
        for _ in range(num_exchanges - 2):
            intent_sequence.append(random.choice(middle_intents))
        
        intent_sequence.append("goodbye")

        base_time = datetime(2024, 1, 1) + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(8, 22),
            minutes=random.randint(0, 59),
        )

        turns = []
        turn_id = 1

        for intent in intent_sequence:
            # User turn
            turns.append(self.generate_turn(turn_id, "user", intent, base_time))
            turn_id += 1

            # Assistant turn
            turns.append(self.generate_turn(turn_id, "assistant", intent, base_time))
            turn_id += 1

        duration = len(turns) * 3  # Approximate seconds

        return {
            "conversation_id": conv_id,
            "session_id": session_id,
            "turns": turns,
            "metadata": {
                "duration_seconds": duration,
                "num_turns": len(turns),
                "outcome": random.choice(["success", "success", "success", "partial", "abandoned"]),
                "domain": random.choice(["general", "travel", "smart_home", "entertainment"]),
                "intents_used": intent_sequence,
            },
        }

    def generate_intents_data(self) -> List[Dict[str, Any]]:
        """Generate intent training data."""
        intents_data = []

        for intent_name, intent_info in INTENTS.items():
            # Generate multiple filled examples
            examples = []
            for template in intent_info["examples"]:
                # Generate 3 variations of each template
                for _ in range(3):
                    filled, _ = self.fill_slots(template)
                    examples.append(filled)

            intents_data.append({
                "intent": intent_name,
                "examples": examples,
                "slots": intent_info["slots"],
                "responses": intent_info["responses"],
            })

        return intents_data

    def generate_entities_data(self) -> List[Dict[str, Any]]:
        """Generate entity training data."""
        entities_data = []

        for entity_type, values in ENTITIES.items():
            entity_examples = []
            for value in values:
                entity_examples.append({
                    "text": value,
                    "value": value.lower().replace(" ", "_"),
                    "synonyms": [],  # Could add synonyms here
                })

            entities_data.append({
                "entity_type": entity_type,
                "examples": entity_examples,
            })

        return entities_data


def generate_audio_metadata(num_samples: int, output_dir: Path) -> List[Dict[str, Any]]:
    """Generate metadata for sample audio files (actual audio generation requires additional deps)."""
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_metadata = []

    for i in range(num_samples):
        # Pick a random utterance
        intent = random.choice(list(INTENTS.keys()))
        template = random.choice(INTENTS[intent]["examples"])

        # Fill slots
        text = template
        for slot_name, values in ENTITIES.items():
            placeholder = "{" + slot_name + "}"
            if placeholder in text:
                text = text.replace(placeholder, random.choice(values), 1)

        filename = f"sample_{i+1:03d}.wav"
        duration = round(len(text.split()) * 0.3 + random.uniform(0.5, 1.0), 2)

        audio_metadata.append({
            "filename": filename,
            "text": text,
            "intent": intent,
            "duration_seconds": duration,
            "sample_rate": 16000,
            "channels": 1,
        })

        # Create placeholder file (empty for now)
        (audio_dir / filename).touch()

    return audio_metadata


def save_data(
    conversations: List[Dict],
    intents: List[Dict],
    entities: List[Dict],
    audio_metadata: List[Dict],
    output_dir: Path,
) -> None:
    """Save all generated data to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save conversations
    with open(output_dir / "conversations.json", "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"    ✓ conversations.json ({len(conversations)} conversations)")

    # Save intents
    with open(output_dir / "intents.json", "w") as f:
        json.dump(intents, f, indent=2)
    print(f"    ✓ intents.json ({len(intents)} intents)")

    # Save entities
    with open(output_dir / "entities.json", "w") as f:
        json.dump(entities, f, indent=2)
    print(f"    ✓ entities.json ({len(entities)} entity types)")

    # Save audio metadata
    with open(output_dir / "audio_manifest.json", "w") as f:
        json.dump(audio_metadata, f, indent=2)
    print(f"    ✓ audio_manifest.json ({len(audio_metadata)} samples)")


def print_statistics(
    conversations: List[Dict],
    intents: List[Dict],
    entities: List[Dict],
) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)

    print("\n💬 Conversations:")
    print(f"  Total conversations: {len(conversations)}")
    total_turns = sum(c["metadata"]["num_turns"] for c in conversations)
    print(f"  Total turns: {total_turns}")
    print(f"  Avg turns per conversation: {total_turns / len(conversations):.1f}")

    print("\n🎯 Intents:")
    print(f"  Total intents: {len(intents)}")
    total_examples = sum(len(i["examples"]) for i in intents)
    print(f"  Total examples: {total_examples}")

    print("\n📦 Entities:")
    print(f"  Total entity types: {len(entities)}")
    total_values = sum(len(e["examples"]) for e in entities)
    print(f"  Total entity values: {total_values}")

    print("\n  Intent distribution in conversations:")
    intent_counts = {}
    for conv in conversations:
        for intent in conv["metadata"]["intents_used"]:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {intent}: {count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversational AI data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_sample_data.py
  python generate_sample_data.py --conversations 500 --audio-samples 100
  python generate_sample_data.py --output data/large/ --seed 123
        """,
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=100,
        help="Number of conversations (default: 100)",
    )
    parser.add_argument(
        "--audio-samples",
        type=int,
        default=50,
        help="Number of audio samples (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample"),
        help="Output directory (default: data/sample)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🎤 Conversational AI - Sample Data Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Conversations: {args.conversations}")
    print(f"  Audio samples: {args.audio_samples}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print()

    # Initialize generator
    generator = ConversationGenerator(seed=args.seed)

    # Generate data
    print("Generating data...")

    print("  Generating conversations...")
    conversations = [generator.generate_conversation() for _ in range(args.conversations)]

    print("  Generating intents...")
    intents = generator.generate_intents_data()

    print("  Generating entities...")
    entities = generator.generate_entities_data()

    print("  Generating audio metadata...")
    audio_metadata = generate_audio_metadata(args.audio_samples, args.output)

    # Save data
    print(f"\nSaving to {args.output}/...")
    save_data(conversations, intents, entities, audio_metadata, args.output)

    # Print statistics
    print_statistics(conversations, intents, entities)

    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
