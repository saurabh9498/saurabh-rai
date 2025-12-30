"""Logging configuration."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", json_format: bool = False):
    """Configure logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if json_format:
        import json
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                })
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers = [handler]
    
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
