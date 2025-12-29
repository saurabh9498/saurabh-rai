"""
Tools Module.

Provides tools that agents can use to interact with
external systems and services.
"""

from .registry import (
    ToolRegistry,
    ToolDefinition,
    ToolCategory,
    registry,
    tool
)
from .web_search import WebSearchTool, web_search
from .database import DatabaseTool, database_query
from .code_executor import CodeExecutor, execute_code, ExecutionResult

__all__ = [
    # Registry
    "ToolRegistry",
    "ToolDefinition",
    "ToolCategory",
    "registry",
    "tool",
    # Web Search
    "WebSearchTool",
    "web_search",
    # Database
    "DatabaseTool",
    "database_query",
    # Code Execution
    "CodeExecutor",
    "execute_code",
    "ExecutionResult",
]
