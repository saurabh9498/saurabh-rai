"""
Multi-Agent AI System.

Enterprise-grade multi-agent orchestration with RAG capabilities.
"""

from .core import settings, get_llm, context_manager
from .agents import OrchestratorAgent, ResearchAgent, AnalystAgent, CodeAgent
from .rag import RAGPipeline, Document

__version__ = "1.0.0"

__all__ = [
    # Core
    "settings",
    "get_llm",
    "context_manager",
    # Agents
    "OrchestratorAgent",
    "ResearchAgent",
    "AnalystAgent",
    "CodeAgent",
    # RAG
    "RAGPipeline",
    "Document",
    # Version
    "__version__",
]
