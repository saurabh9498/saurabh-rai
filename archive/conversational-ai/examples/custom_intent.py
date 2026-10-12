#!/usr/bin/env python3
"""
Custom Intent Example

This example demonstrates how to:
1. Define custom intents and entities
2. Train a custom intent classifier
3. Add domain-specific response templates
4. Integrate custom handlers

Usage:
    python examples/custom_intent.py --train
    python examples/custom_intent.py --test "Schedule a meeting tomorrow at 3pm"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlu.intent_classifier import IntentClassifier
from src.nlu.entity_extractor import EntityExtractor
from src.dialog.response_generator import ResponseGenerator
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Intent Definitions
# =============================================================================

CUSTOM_INTENTS = {
    "schedule_meeting": {
        "description": "User wants to schedule a meeting",
        "examples": [
            "Schedule a meeting",
            "Set up a meeting with John",
            "Can you book a conference room",
            "I need to arrange a meeting tomorrow",
            "Create a meeting invite",
            "Set up a call with the team",
            "Book a meeting for next Monday",
            "Schedule a 30-minute sync",
            "I want to set up a meeting at 2pm",
            "Arrange a meeting with marketing"
        ],
        "required_entities": ["datetime"],
        "optional_entities": ["person", "duration", "location"]
    },
    "cancel_meeting": {
        "description": "User wants to cancel a meeting",
        "examples": [
            "Cancel my meeting",
            "Delete the 3pm meeting",
            "Remove the meeting with Sarah",
            "Cancel tomorrow's standup",
            "I need to cancel my appointment",
            "Please cancel the team sync",
            "Remove the meeting from my calendar",
            "Delete my 10am call"
        ],
        "required_entities": [],
        "optional_entities": ["datetime", "person", "meeting_id"]
    },
    "check_calendar": {
        "description": "User wants to check their schedule",
        "examples": [
            "What's on my calendar",
            "Show me my schedule",
            "What meetings do I have today",
            "Am I free tomorrow at 2pm",
            "Check my availability",
            "What's my schedule for next week",
            "Do I have any meetings",
            "Show me today's appointments"
        ],
        "required_entities": [],
        "optional_entities": ["datetime"]
    },
    "reschedule_meeting": {
        "description": "User wants to move a meeting to a different time",
        "examples": [
            "Reschedule my meeting",
            "Move the 3pm meeting to 4pm",
            "Change the time of my appointment",
            "Push back the standup by 30 minutes",
            "Reschedule tomorrow's call to Friday",
            "Move my meeting with John to next week",
            "Change the team sync to 11am"
        ],
        "required_entities": ["datetime"],
        "optional_entities": ["person", "meeting_id"]
    }
}

CUSTOM_ENTITIES = {
    "meeting_id": {
        "description": "Meeting identifier",
        "patterns": [
            r"meeting #(\d+)",
            r"appointment (\d+)",
            r"MTG-(\d+)"
        ]
    },
    "duration": {
        "description": "Meeting duration",
        "patterns": [
            r"(\d+)\s*(?:minute|min|hour|hr)s?",
            r"(half|quarter)\s*hour",
            r"(\d+:\d+)"
        ],
        "examples": ["30 minutes", "1 hour", "45 mins", "half hour"]
    }
}

CUSTOM_RESPONSES = {
    "schedule_meeting": {
        "ask_datetime": [
            "When would you like to schedule the meeting?",
            "What time works best for you?"
        ],
        "ask_attendees": [
            "Who should I invite to the meeting?",
            "Who will be attending?"
        ],
        "confirm": [
            "I'll schedule a meeting for {datetime}. Should I send the invite?",
            "Meeting confirmed for {datetime} with {attendees}."
        ],
        "success": [
            "Your meeting has been scheduled for {datetime}.",
            "All set! Meeting created for {datetime}."
        ]
    },
    "cancel_meeting": {
        "ask_which": [
            "Which meeting would you like to cancel?",
            "I see multiple meetings. Which one?"
        ],
        "confirm": [
            "Are you sure you want to cancel the {meeting_name} meeting?",
            "I'll cancel the meeting at {datetime}. Is that correct?"
        ],
        "success": [
            "The meeting has been cancelled.",
            "Done! I've removed the meeting from your calendar."
        ]
    },
    "check_calendar": {
        "no_meetings": [
            "You have no meetings scheduled for {datetime}.",
            "Your calendar is clear for {datetime}."
        ],
        "has_meetings": [
            "You have {count} meeting(s) scheduled:\n{meetings}",
            "Here's your schedule for {datetime}:\n{meetings}"
        ]
    },
    "reschedule_meeting": {
        "ask_new_time": [
            "What time would you like to move the meeting to?",
            "When should I reschedule it for?"
        ],
        "confirm": [
            "I'll move the meeting from {old_time} to {new_time}. Is that correct?",
            "Rescheduling to {new_time}. Should I update the invite?"
        ],
        "success": [
            "Meeting rescheduled to {datetime}.",
            "Done! The meeting is now at {datetime}."
        ]
    }
}


# =============================================================================
# Custom Intent Trainer
# =============================================================================

class CustomIntentTrainer:
    """Train custom intent classification model."""
    
    def __init__(self, intents: Dict[str, dict]):
        """
        Initialize trainer.
        
        Args:
            intents: Intent definitions with examples
        """
        self.intents = intents
        self.classifier = IntentClassifier()
    
    def prepare_training_data(self) -> tuple:
        """Prepare training data from intent definitions."""
        texts = []
        labels = []
        
        for intent_name, intent_data in self.intents.items():
            for example in intent_data["examples"]:
                texts.append(example)
                labels.append(intent_name)
        
        return texts, labels
    
    def train(
        self,
        output_path: str = "models/nlu/custom_intent/",
        epochs: int = 10,
        batch_size: int = 8
    ) -> dict:
        """
        Train custom intent classifier.
        
        Args:
            output_path: Path to save trained model
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training metrics
        """
        logger.info("Preparing training data...")
        texts, labels = self.prepare_training_data()
        
        logger.info(f"Training on {len(texts)} examples for {len(self.intents)} intents")
        
        # Train classifier
        metrics = self.classifier.train(
            texts=texts,
            labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
        # Save model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.classifier.save(output_path)
        
        # Save intent definitions
        with open(output_path / "intents.json", 'w') as f:
            json.dump(self.intents, f, indent=2)
        
        logger.info(f"Model saved to {output_path}")
        
        return metrics
    
    def evaluate(self, test_texts: List[str], test_labels: List[str]) -> dict:
        """Evaluate classifier on test data."""
        return self.classifier.evaluate(test_texts, test_labels)


# =============================================================================
# Custom Intent Handler
# =============================================================================

class CalendarIntentHandler:
    """Handler for calendar-related intents."""
    
    def __init__(self, responses: Dict[str, dict]):
        """
        Initialize handler.
        
        Args:
            responses: Response templates
        """
        self.responses = responses
        self.response_generator = ResponseGenerator()
        
        # Mock calendar data
        self.calendar = [
            {"id": 1, "title": "Team Standup", "datetime": "2024-01-15T09:00:00", "attendees": ["team"]},
            {"id": 2, "title": "1:1 with Manager", "datetime": "2024-01-15T14:00:00", "attendees": ["manager"]},
            {"id": 3, "title": "Project Review", "datetime": "2024-01-16T10:00:00", "attendees": ["team", "stakeholders"]}
        ]
    
    def handle(self, intent: str, entities: List[dict], state: dict) -> dict:
        """
        Handle a calendar intent.
        
        Args:
            intent: Detected intent name
            entities: Extracted entities
            state: Current dialog state
            
        Returns:
            Response with action
        """
        handlers = {
            "schedule_meeting": self._handle_schedule,
            "cancel_meeting": self._handle_cancel,
            "check_calendar": self._handle_check,
            "reschedule_meeting": self._handle_reschedule
        }
        
        handler = handlers.get(intent, self._handle_unknown)
        return handler(entities, state)
    
    def _handle_schedule(self, entities: List[dict], state: dict) -> dict:
        """Handle schedule_meeting intent."""
        datetime_entity = self._find_entity(entities, "datetime")
        
        if not datetime_entity:
            return {
                "text": self._get_response("schedule_meeting", "ask_datetime"),
                "action": "request_slot",
                "slot": "datetime"
            }
        
        # Simulate scheduling
        response_text = self._get_response(
            "schedule_meeting", 
            "success",
            datetime=datetime_entity["value"]
        )
        
        return {
            "text": response_text,
            "action": "complete",
            "result": {"scheduled": True, "datetime": datetime_entity["value"]}
        }
    
    def _handle_cancel(self, entities: List[dict], state: dict) -> dict:
        """Handle cancel_meeting intent."""
        datetime_entity = self._find_entity(entities, "datetime")
        
        if datetime_entity:
            response_text = self._get_response(
                "cancel_meeting",
                "success"
            )
            return {"text": response_text, "action": "complete"}
        
        # Ask which meeting
        return {
            "text": self._get_response("cancel_meeting", "ask_which"),
            "action": "request_slot",
            "slot": "meeting_id"
        }
    
    def _handle_check(self, entities: List[dict], state: dict) -> dict:
        """Handle check_calendar intent."""
        datetime_entity = self._find_entity(entities, "datetime")
        date_str = datetime_entity["value"] if datetime_entity else "today"
        
        # Mock calendar check
        meetings = self.calendar
        
        if meetings:
            meetings_text = "\n".join(
                f"• {m['title']} at {m['datetime']}"
                for m in meetings
            )
            response_text = self._get_response(
                "check_calendar",
                "has_meetings",
                count=len(meetings),
                meetings=meetings_text,
                datetime=date_str
            )
        else:
            response_text = self._get_response(
                "check_calendar",
                "no_meetings",
                datetime=date_str
            )
        
        return {"text": response_text, "action": "inform"}
    
    def _handle_reschedule(self, entities: List[dict], state: dict) -> dict:
        """Handle reschedule_meeting intent."""
        datetime_entity = self._find_entity(entities, "datetime")
        
        if not datetime_entity:
            return {
                "text": self._get_response("reschedule_meeting", "ask_new_time"),
                "action": "request_slot",
                "slot": "datetime"
            }
        
        response_text = self._get_response(
            "reschedule_meeting",
            "success",
            datetime=datetime_entity["value"]
        )
        
        return {"text": response_text, "action": "complete"}
    
    def _handle_unknown(self, entities: List[dict], state: dict) -> dict:
        """Handle unknown intent."""
        return {
            "text": "I'm not sure how to help with that. I can help you schedule, cancel, check, or reschedule meetings.",
            "action": "fallback"
        }
    
    def _find_entity(self, entities: List[dict], entity_type: str) -> dict:
        """Find entity by type."""
        for entity in entities:
            if entity.get("type") == entity_type:
                return entity
        return None
    
    def _get_response(self, intent: str, response_type: str, **kwargs) -> str:
        """Get response template and fill placeholders."""
        import random
        
        templates = self.responses.get(intent, {}).get(response_type, [])
        if not templates:
            return "I understand. Let me help you with that."
        
        template = random.choice(templates)
        return template.format(**kwargs) if kwargs else template


# =============================================================================
# Main Functions
# =============================================================================

def train_custom_model(output_path: str, epochs: int = 10):
    """Train a custom intent classification model."""
    trainer = CustomIntentTrainer(CUSTOM_INTENTS)
    
    print("Training custom intent classifier...")
    metrics = trainer.train(output_path=output_path, epochs=epochs)
    
    print(f"\nTraining complete!")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
    print(f"  F1 Score: {metrics.get('f1', 0):.2%}")
    print(f"  Model saved to: {output_path}")


def test_custom_model(text: str, model_path: str = "models/nlu/custom_intent/"):
    """Test the custom intent model on input text."""
    
    # Load classifier
    classifier = IntentClassifier()
    classifier.load(model_path)
    
    # Load entity extractor
    entity_extractor = EntityExtractor()
    
    # Initialize handler
    handler = CalendarIntentHandler(CUSTOM_RESPONSES)
    
    # Classify intent
    intent_result = classifier.predict(text)
    
    # Extract entities
    entities = entity_extractor.extract(text)
    
    print(f"\nInput: {text}")
    print(f"\nIntent: {intent_result['intent']} ({intent_result['confidence']:.2%})")
    
    if entities:
        print(f"Entities:")
        for entity in entities:
            print(f"  - {entity['type']}: {entity['value']}")
    
    # Handle intent
    response = handler.handle(intent_result['intent'], entities, {})
    
    print(f"\nResponse: {response['text']}")
    print(f"Action: {response['action']}")


def interactive_test(model_path: str = "models/nlu/custom_intent/"):
    """Interactive testing mode."""
    
    # Load components
    classifier = IntentClassifier()
    classifier.load(model_path)
    
    entity_extractor = EntityExtractor()
    handler = CalendarIntentHandler(CUSTOM_RESPONSES)
    
    print("\n" + "="*50)
    print("Custom Intent Tester")
    print("="*50)
    print("Type a message to test intent classification.")
    print("Type 'quit' to exit.")
    print("="*50 + "\n")
    
    while True:
        try:
            text = input("You: ").strip()
            
            if not text:
                continue
            
            if text.lower() == 'quit':
                break
            
            # Classify
            intent_result = classifier.predict(text)
            entities = entity_extractor.extract(text)
            
            # Handle
            response = handler.handle(intent_result['intent'], entities, {})
            
            print(f"\n[Intent: {intent_result['intent']} ({intent_result['confidence']:.0%})]")
            if entities:
                print(f"[Entities: {', '.join(f'{e['type']}={e['value']}' for e in entities)}]")
            print(f"Bot: {response['text']}\n")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Custom intent classifier example"
    )
    
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--train",
        action="store_true",
        help="Train custom intent model"
    )
    action_group.add_argument(
        "--test",
        type=str,
        help="Test with a single message"
    )
    action_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive testing mode"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/nlu/custom_intent/",
        help="Path to model directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs"
    )
    
    args = parser.parse_args()
    
    if args.train:
        train_custom_model(args.model_path, args.epochs)
    elif args.test:
        test_custom_model(args.test, args.model_path)
    elif args.interactive:
        interactive_test(args.model_path)


if __name__ == "__main__":
    main()
