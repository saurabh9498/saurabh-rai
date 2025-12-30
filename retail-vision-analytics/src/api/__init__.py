"""
Retail Vision Analytics - REST API Module.

This module provides the FastAPI-based REST API for:
- Analytics data retrieval
- Camera/stream management
- Alert management
- Real-time WebSocket streaming
- System health monitoring

Usage:
    uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
"""

from .routes import app

__all__ = ["app"]
