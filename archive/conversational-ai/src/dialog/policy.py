"""Dialog Policy for action selection."""

from typing import Dict, Any, Optional
from .state_tracker import DialogState


class DialogPolicy:
    """Rule-based dialog policy."""
    
    def select_action(self, state: DialogState) -> Dict[str, Any]:
        """Select next dialog action based on state."""
        if state.active_intent is None:
            return {"action": "greet", "response_type": "greeting"}
        
        missing = state.get_missing_required_slots()
        if missing:
            slot = list(missing)[0]
            return {
                "action": "request_slot",
                "slot": slot,
                "response_type": "slot_request",
            }
        
        if state.is_complete():
            return {
                "action": "execute",
                "intent": state.active_intent,
                "slots": state.get_filled_slots(),
                "response_type": "confirmation",
            }
        
        return {"action": "clarify", "response_type": "clarification"}
