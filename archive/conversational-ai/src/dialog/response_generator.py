"""Response Generation."""

from typing import Dict, Any, List


TEMPLATES = {
    "greeting": ["Hello! How can I help you?", "Hi there! What can I do for you?"],
    "goodbye": ["Goodbye!", "Have a great day!"],
    "slot_request": {
        "location": "What location?",
        "date": "What date?",
        "time": "What time?",
        "duration": "For how long?",
    },
    "confirmation": "I'll help you with that.",
    "fallback": "I'm not sure I understood. Could you rephrase?",
}


class ResponseGenerator:
    """Template-based response generation."""
    
    def generate(self, action: Dict[str, Any]) -> str:
        action_type = action.get("action", "fallback")
        
        if action_type == "request_slot":
            slot = action.get("slot", "")
            templates = TEMPLATES.get("slot_request", {})
            return templates.get(slot, f"What is the {slot}?")
        
        template = TEMPLATES.get(action_type, TEMPLATES["fallback"])
        if isinstance(template, list):
            import random
            return random.choice(template)
        return template
