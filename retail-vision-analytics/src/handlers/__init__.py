"""
Error Handlers Module.

Centralized error handling for the Retail Vision Analytics system.
"""

from .error_handler import (
    RetailVisionError,
    ConfigurationError,
    ModelError,
    InferenceError,
    StreamError,
    StorageError,
    APIError,
    ErrorHandler,
    error_handler,
)

__all__ = [
    "RetailVisionError",
    "ConfigurationError",
    "ModelError",
    "InferenceError",
    "StreamError",
    "StorageError",
    "APIError",
    "ErrorHandler",
    "error_handler",
]
