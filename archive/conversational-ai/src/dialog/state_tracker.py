"""
Dialog State Tracker

Maintains conversation state across turns including slots, intents, and context.
Implements belief state tracking for task-oriented dialog.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json
import copy
import logging

logger = logging.getLogger(__name__)


@dataclass
class Slot:
    """A dialog slot with value and metadata."""
    name: str
    value: Any
    confidence: float = 1.0
    source: str = "user"  # user, system, default
    turn_filled: int = 0
    confirmed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "turn_filled": self.turn_filled,
            "confirmed": self.confirmed,
        }


@dataclass
class Turn:
    """A single conversation turn."""
    turn_id: int
    user_utterance: str
    system_response: str
    intent: str
    entities: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_utterance": self.user_utterance,
            "system_response": self.system_response,
            "intent": self.intent,
            "entities": self.entities,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DialogState:
    """
    Complete dialog state.
    
    Tracks:
    - Current slots and their values
    - Conversation history
    - Active intent/task
    - User preferences
    """
    session_id: str
    turn_count: int = 0
    active_intent: Optional[str] = None
    
    # Slot tracking
    slots: Dict[str, Slot] = field(default_factory=dict)
    required_slots: Set[str] = field(default_factory=set)
    optional_slots: Set[str] = field(default_factory=set)
    
    # History
    history: List[Turn] = field(default_factory=list)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "active_intent": self.active_intent,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "history": [t.to_dict() for t in self.history[-10:]],  # Last 10 turns
            "context": self.context,
            "user_preferences": self.user_preferences,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }
    
    def get_slot_value(self, slot_name: str) -> Optional[Any]:
        """Get value of a slot."""
        slot = self.slots.get(slot_name)
        return slot.value if slot else None
    
    def get_filled_slots(self) -> Dict[str, Any]:
        """Get all filled slot values."""
        return {
            name: slot.value
            for name, slot in self.slots.items()
            if slot.value is not None
        }
    
    def get_missing_required_slots(self) -> Set[str]:
        """Get required slots that are not yet filled."""
        filled = set(
            name for name, slot in self.slots.items()
            if slot.value is not None
        )
        return self.required_slots - filled
    
    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        return len(self.get_missing_required_slots()) == 0


class DialogStateTracker:
    """
    Tracks and updates dialog state across conversation turns.
    
    Features:
    - Slot filling with confirmation
    - Context carryover
    - Intent switching handling
    - State persistence
    """
    
    def __init__(self):
        self.states: Dict[str, DialogState] = {}
        self.slot_schemas = self._load_slot_schemas()
    
    def _load_slot_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load slot schemas for different intents."""
        return {
            "book_flight": {
                "required": ["origin", "destination", "date"],
                "optional": ["time", "class", "passengers"],
            },
            "book_hotel": {
                "required": ["location", "check_in", "check_out"],
                "optional": ["room_type", "guests", "price_range"],
            },
            "set_timer": {
                "required": ["duration"],
                "optional": ["name"],
            },
            "set_reminder": {
                "required": ["time", "message"],
                "optional": ["repeat"],
            },
            "get_weather": {
                "required": [],
                "optional": ["location", "date"],
            },
            "play_music": {
                "required": [],
                "optional": ["song", "artist", "album", "genre", "playlist"],
            },
            "send_message": {
                "required": ["recipient", "message"],
                "optional": [],
            },
        }
    
    def get_or_create_state(self, session_id: str) -> DialogState:
        """Get existing state or create new one."""
        if session_id not in self.states:
            self.states[session_id] = DialogState(session_id=session_id)
            logger.info(f"Created new dialog state: {session_id}")
        
        return self.states[session_id]
    
    def update(
        self,
        session_id: str,
        user_utterance: str,
        intent: str,
        entities: List[Dict[str, Any]],
        system_response: str = "",
    ) -> DialogState:
        """
        Update dialog state with new turn.
        
        Args:
            session_id: Session identifier
            user_utterance: User's input
            intent: Detected intent
            entities: Extracted entities
            system_response: System's response
            
        Returns:
            Updated DialogState
        """
        state = self.get_or_create_state(session_id)
        state.turn_count += 1
        state.last_updated = datetime.now()
        
        # Handle intent change
        if intent != state.active_intent and intent not in ["fallback", "yes", "no"]:
            self._handle_intent_change(state, intent)
        
        # Update slots from entities
        for entity in entities:
            self._update_slot(state, entity)
        
        # Handle confirmation intents
        if intent == "yes" and state.context.get("pending_confirmation"):
            self._confirm_pending(state)
        elif intent == "no" and state.context.get("pending_confirmation"):
            self._reject_pending(state)
        
        # Add turn to history
        turn = Turn(
            turn_id=state.turn_count,
            user_utterance=user_utterance,
            system_response=system_response,
            intent=intent,
            entities=entities,
        )
        state.history.append(turn)
        
        # Keep history manageable
        if len(state.history) > 20:
            state.history = state.history[-20:]
        
        return state
    
    def _handle_intent_change(self, state: DialogState, new_intent: str):
        """Handle transition to new intent."""
        logger.info(f"Intent change: {state.active_intent} -> {new_intent}")
        
        state.active_intent = new_intent
        
        # Load slot schema for new intent
        schema = self.slot_schemas.get(new_intent, {})
        state.required_slots = set(schema.get("required", []))
        state.optional_slots = set(schema.get("optional", []))
        
        # Clear slots (but preserve some context)
        preserved = {"location", "date"}  # Slots to preserve across intent changes
        state.slots = {
            k: v for k, v in state.slots.items()
            if k in preserved and v.confirmed
        }
    
    def _update_slot(self, state: DialogState, entity: Dict[str, Any]):
        """Update slot with entity value."""
        entity_type = entity.get("type")
        value = entity.get("value")
        confidence = entity.get("confidence", 0.9)
        
        # Map entity types to slot names
        slot_name = self._map_entity_to_slot(entity_type, state.active_intent)
        
        if slot_name:
            existing = state.slots.get(slot_name)
            
            # Update if new or higher confidence
            if not existing or confidence > existing.confidence or not existing.confirmed:
                state.slots[slot_name] = Slot(
                    name=slot_name,
                    value=value,
                    confidence=confidence,
                    source="user",
                    turn_filled=state.turn_count,
                    confirmed=False,
                )
                logger.debug(f"Updated slot {slot_name} = {value}")
    
    def _map_entity_to_slot(self, entity_type: str, intent: str) -> Optional[str]:
        """Map entity type to slot name based on intent."""
        # Direct mappings
        direct = {
            "location": "location",
            "date": "date",
            "time": "time",
            "duration": "duration",
            "number": "number",
            "person": "recipient",
        }
        
        if entity_type in direct:
            return direct[entity_type]
        
        # Intent-specific mappings
        intent_mappings = {
            "book_flight": {
                "location": "destination",  # Could also be origin
            },
            "play_music": {
                "person": "artist",
            },
        }
        
        if intent in intent_mappings:
            return intent_mappings[intent].get(entity_type)
        
        return entity_type
    
    def _confirm_pending(self, state: DialogState):
        """Confirm pending slot values."""
        pending = state.context.get("pending_confirmation", {})
        
        for slot_name, value in pending.items():
            if slot_name in state.slots:
                state.slots[slot_name].confirmed = True
                logger.debug(f"Confirmed slot {slot_name} = {value}")
        
        state.context.pop("pending_confirmation", None)
    
    def _reject_pending(self, state: DialogState):
        """Reject pending slot values."""
        pending = state.context.get("pending_confirmation", {})
        
        for slot_name in pending:
            if slot_name in state.slots:
                state.slots.pop(slot_name)
                logger.debug(f"Rejected slot {slot_name}")
        
        state.context.pop("pending_confirmation", None)
    
    def set_slot(
        self,
        session_id: str,
        slot_name: str,
        value: Any,
        source: str = "system",
    ):
        """Manually set a slot value."""
        state = self.get_or_create_state(session_id)
        
        state.slots[slot_name] = Slot(
            name=slot_name,
            value=value,
            source=source,
            turn_filled=state.turn_count,
            confirmed=source == "system",
        )
    
    def request_confirmation(
        self,
        session_id: str,
        slots: Dict[str, Any],
    ):
        """Request confirmation for slot values."""
        state = self.get_or_create_state(session_id)
        state.context["pending_confirmation"] = slots
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get full context for response generation."""
        state = self.get_or_create_state(session_id)
        
        return {
            "intent": state.active_intent,
            "slots": state.get_filled_slots(),
            "missing_slots": list(state.get_missing_required_slots()),
            "is_complete": state.is_complete(),
            "turn_count": state.turn_count,
            "history": [
                {"user": t.user_utterance, "system": t.system_response}
                for t in state.history[-5:]
            ],
            "preferences": state.user_preferences,
        }
    
    def clear_state(self, session_id: str):
        """Clear state for a session."""
        if session_id in self.states:
            del self.states[session_id]
            logger.info(f"Cleared dialog state: {session_id}")
    
    def export_state(self, session_id: str) -> Optional[str]:
        """Export state as JSON."""
        if session_id in self.states:
            return json.dumps(self.states[session_id].to_dict(), indent=2)
        return None
