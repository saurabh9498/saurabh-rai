"""
Core module for Multi-Agent AI System.

Provides configuration, LLM abstraction, and context management.
"""

from .config import settings, get_settings, Settings
from .llm import LLMFactory, get_llm, BaseLLM, LLMResponse, LLMProvider
from .context import (
    ContextManager,
    ConversationContext,
    Message,
    MessageRole,
    AgentAction,
    context_manager
)

__all__ = [
    # Config
    "settings",
    "get_settings",
    "Settings",
    # LLM
    "LLMFactory",
    "get_llm",
    "BaseLLM",
    "LLMResponse",
    "LLMProvider",
    # Context
    "ContextManager",
    "ConversationContext",
    "Message",
    "MessageRole",
    "AgentAction",
    "context_manager",
]
