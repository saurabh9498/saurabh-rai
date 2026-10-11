"""
Context Management for Multi-Agent System.

This module handles conversation context, agent state, and memory
management across multi-turn interactions and agent collaborations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
from collections import deque


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    AGENT = "agent"


@dataclass
class Message:
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "message_id": self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            metadata=data.get("metadata", {}),
            message_id=data.get("message_id", str(uuid.uuid4()))
        )


@dataclass
class AgentAction:
    """Record of an agent action."""
    agent_name: str
    action_type: str
    input_data: Any
    output_data: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error
        }


@dataclass
class ConversationContext:
    """Context for a single conversation/session."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    agent_actions: List[AgentAction] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Context window management
    max_messages: int = 50
    max_tokens: int = 8000
    
    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        
        # Trim old messages if needed
        self._trim_context()
        
        return message
    
    def add_agent_action(self, action: AgentAction) -> None:
        """Record an agent action."""
        self.agent_actions.append(action)
        self.updated_at = datetime.utcnow()
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM input."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.messages
            if msg.role in [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT]
        ]
    
    def get_recent_context(self, n_messages: int = 10) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-n_messages:]
    
    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)
    
    def _trim_context(self) -> None:
        """Trim context to stay within limits."""
        if len(self.messages) > self.max_messages:
            # Keep system message if present
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            other_messages = [m for m in self.messages if m.role != MessageRole.SYSTEM]
            
            # Keep most recent messages
            keep_count = self.max_messages - len(system_messages)
            self.messages = system_messages + other_messages[-keep_count:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [m.to_dict() for m in self.messages],
            "agent_actions": [a.to_dict() for a in self.agent_actions],
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Deserialize context from dictionary."""
        context = cls(
            conversation_id=data.get("conversation_id", str(uuid.uuid4())),
            variables=data.get("variables", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))
        )
        
        for msg_data in data.get("messages", []):
            context.messages.append(Message.from_dict(msg_data))
        
        return context


class ContextManager:
    """
    Manages multiple conversation contexts and provides
    utilities for context manipulation.
    """
    
    def __init__(self, max_contexts: int = 100):
        self._contexts: Dict[str, ConversationContext] = {}
        self._max_contexts = max_contexts
        self._lru_queue: deque = deque(maxlen=max_contexts)
    
    def create_context(self, conversation_id: Optional[str] = None) -> ConversationContext:
        """Create a new conversation context."""
        context = ConversationContext(
            conversation_id=conversation_id or str(uuid.uuid4())
        )
        
        self._contexts[context.conversation_id] = context
        self._lru_queue.append(context.conversation_id)
        
        # Evict old contexts if at capacity
        self._evict_if_needed()
        
        return context
    
    def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get an existing context by ID."""
        context = self._contexts.get(conversation_id)
        if context:
            # Update LRU
            if conversation_id in self._lru_queue:
                self._lru_queue.remove(conversation_id)
            self._lru_queue.append(conversation_id)
        return context
    
    def get_or_create_context(self, conversation_id: str) -> ConversationContext:
        """Get existing context or create new one."""
        context = self.get_context(conversation_id)
        if not context:
            context = self.create_context(conversation_id)
        return context
    
    def delete_context(self, conversation_id: str) -> bool:
        """Delete a context."""
        if conversation_id in self._contexts:
            del self._contexts[conversation_id]
            if conversation_id in self._lru_queue:
                self._lru_queue.remove(conversation_id)
            return True
        return False
    
    def _evict_if_needed(self) -> None:
        """Evict oldest contexts if at capacity."""
        while len(self._contexts) > self._max_contexts:
            oldest_id = self._lru_queue.popleft()
            if oldest_id in self._contexts:
                del self._contexts[oldest_id]
    
    def export_context(self, conversation_id: str) -> Optional[str]:
        """Export context as JSON string."""
        context = self.get_context(conversation_id)
        if context:
            return json.dumps(context.to_dict(), indent=2)
        return None
    
    def import_context(self, json_data: str) -> ConversationContext:
        """Import context from JSON string."""
        data = json.loads(json_data)
        context = ConversationContext.from_dict(data)
        self._contexts[context.conversation_id] = context
        self._lru_queue.append(context.conversation_id)
        return context


# Global context manager instance
context_manager = ContextManager()
