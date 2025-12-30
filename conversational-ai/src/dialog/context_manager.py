"""Context Management for multi-turn conversations."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class ContextManager:
    """Manages conversation context across turns."""
    
    def __init__(self, max_context_turns: int = 10):
        self.max_turns = max_context_turns
        self.contexts: Dict[str, Dict] = {}
    
    def update(self, session_id: str, key: str, value: Any):
        if session_id not in self.contexts:
            self.contexts[session_id] = {"created": datetime.now()}
        self.contexts[session_id][key] = value
    
    def get(self, session_id: str, key: str, default: Any = None) -> Any:
        ctx = self.contexts.get(session_id, {})
        return ctx.get(key, default)
    
    def clear(self, session_id: str):
        self.contexts.pop(session_id, None)
    
    def cleanup_old(self, max_age_hours: int = 24):
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            sid for sid, ctx in self.contexts.items()
            if ctx.get("created", datetime.now()) < cutoff
        ]
        for sid in to_remove:
            del self.contexts[sid]
