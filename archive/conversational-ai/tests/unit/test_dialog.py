"""Unit tests for dialog management."""

import pytest
from src.dialog.state_tracker import DialogStateTracker, DialogState, Slot
from src.dialog.policy import DialogPolicy
from src.dialog.response_generator import ResponseGenerator


class TestDialogStateTracker:
    """Tests for dialog state tracking."""
    
    def test_create_state(self):
        tracker = DialogStateTracker()
        
        state = tracker.get_or_create_state("test-session")
        
        assert state.session_id == "test-session"
        assert state.turn_count == 0
        assert state.active_intent is None
    
    def test_update_state(self):
        tracker = DialogStateTracker()
        
        state = tracker.update(
            session_id="test-session",
            user_utterance="Set a timer for 5 minutes",
            intent="set_timer",
            entities=[{"type": "duration", "value": "5 minutes"}],
        )
        
        assert state.turn_count == 1
        assert state.active_intent == "set_timer"
    
    def test_slot_filling(self):
        tracker = DialogStateTracker()
        
        # First turn - provide duration
        state = tracker.update(
            session_id="test-session",
            user_utterance="Set a timer for 5 minutes",
            intent="set_timer",
            entities=[{"type": "duration", "value": "5 minutes"}],
        )
        
        assert "duration" in state.slots
        assert state.slots["duration"].value == "5 minutes"
    
    def test_context_preservation(self):
        tracker = DialogStateTracker()
        
        # Update twice
        tracker.update("session", "hello", "greeting", [])
        state = tracker.update("session", "what time is it", "get_time", [])
        
        assert len(state.history) == 2


class TestDialogPolicy:
    """Tests for dialog policy."""
    
    def test_select_action_greet(self):
        policy = DialogPolicy()
        state = DialogState(session_id="test")
        
        action = policy.select_action(state)
        
        assert action["action"] == "greet"
    
    def test_select_action_request_slot(self):
        policy = DialogPolicy()
        state = DialogState(session_id="test")
        state.active_intent = "set_timer"
        state.required_slots = {"duration"}
        
        action = policy.select_action(state)
        
        assert action["action"] == "request_slot"
        assert action["slot"] == "duration"


class TestResponseGenerator:
    """Tests for response generation."""
    
    def test_generate_greeting(self):
        generator = ResponseGenerator()
        
        response = generator.generate({"action": "greeting"})
        
        assert isinstance(response, str)
        assert len(response) > 0
