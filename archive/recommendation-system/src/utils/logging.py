"""
Structured logging configuration for the recommendation system.

Provides JSON-formatted logs for production and human-readable logs for development.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from contextvars import ContextVar
import traceback

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add context variables
        if request_id := request_id_var.get():
            log_data['request_id'] = request_id
        if user_id := user_id_var.get():
            log_data['user_id'] = user_id
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info),
            }
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for development."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Build prefix with context
        prefix_parts = [f'{color}{record.levelname}{self.RESET}']
        
        if request_id := request_id_var.get():
            prefix_parts.append(f'[{request_id[:8]}]')
        
        prefix = ' '.join(prefix_parts)
        
        # Format message
        message = f"{prefix} {record.name}: {record.getMessage()}"
        
        # Add exception if present
        if record.exc_info:
            message += '\n' + ''.join(traceback.format_exception(*record.exc_info))
        
        return message


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that includes context in all log messages."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]):
        extra = kwargs.get('extra', {})
        
        # Add context variables
        if request_id := request_id_var.get():
            extra['request_id'] = request_id
        if user_id := user_id_var.get():
            extra['user_id'] = user_id
        
        kwargs['extra'] = extra
        return msg, kwargs


def setup_logging(
    level: str = 'INFO',
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (for production)
        log_file: Optional file path for logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = ColoredFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (always JSON for parsing)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def get_logger(name: str) -> ContextLogger:
    """
    Get a context-aware logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        ContextLogger instance
    """
    return ContextLogger(logging.getLogger(name), {})


def set_request_context(request_id: str, user_id: Optional[str] = None):
    """Set context for the current request."""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context():
    """Clear request context."""
    request_id_var.set(None)
    user_id_var.set(None)


class LogContext:
    """Context manager for request logging."""
    
    def __init__(self, request_id: str, user_id: Optional[str] = None):
        self.request_id = request_id
        self.user_id = user_id
        self._token_request_id = None
        self._token_user_id = None
    
    def __enter__(self):
        self._token_request_id = request_id_var.set(self.request_id)
        if self.user_id:
            self._token_user_id = user_id_var.set(self.user_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        request_id_var.reset(self._token_request_id)
        if self._token_user_id:
            user_id_var.reset(self._token_user_id)


# Metric logging helpers
def log_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None,
):
    """Log a metric value."""
    extra = {
        'metric_name': metric_name,
        'metric_value': value,
        'metric_tags': tags or {},
    }
    logger.info(f"Metric: {metric_name}={value}", extra={'extra_fields': extra})


def log_latency(
    logger: logging.Logger,
    operation: str,
    latency_ms: float,
    success: bool = True,
):
    """Log operation latency."""
    extra = {
        'operation': operation,
        'latency_ms': latency_ms,
        'success': success,
    }
    logger.info(
        f"Latency: {operation} took {latency_ms:.2f}ms",
        extra={'extra_fields': extra}
    )
