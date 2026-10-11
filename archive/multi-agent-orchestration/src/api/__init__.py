"""
API Module for Multi-Agent AI System.

Provides REST API endpoints for the multi-agent system.
"""

from .main import app, create_app
from .routes import router
from .schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    AgentsResponse,
    HealthResponse
)

__all__ = [
    "app",
    "create_app",
    "router",
    "QueryRequest",
    "QueryResponse",
    "IngestRequest",
    "IngestResponse",
    "AgentsResponse",
    "HealthResponse",
]
