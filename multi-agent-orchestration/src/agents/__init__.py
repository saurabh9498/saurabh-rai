"""
Agents module for Multi-Agent AI System.

Provides specialized agents for different tasks:
- ResearchAgent: Information retrieval and synthesis
- AnalystAgent: Data analysis and visualization
- CodeAgent: Code generation and review
- OrchestratorAgent: Multi-agent coordination
"""

from .base_agent import BaseAgent, Tool, AgentResult, AgentStatus
from .research_agent import ResearchAgent
from .analyst_agent import AnalystAgent
from .code_agent import CodeAgent
from .orchestrator import OrchestratorAgent, WorkflowResult, TaskPlan

__all__ = [
    # Base
    "BaseAgent",
    "Tool",
    "AgentResult",
    "AgentStatus",
    # Specialized Agents
    "ResearchAgent",
    "AnalystAgent",
    "CodeAgent",
    # Orchestration
    "OrchestratorAgent",
    "WorkflowResult",
    "TaskPlan",
]
