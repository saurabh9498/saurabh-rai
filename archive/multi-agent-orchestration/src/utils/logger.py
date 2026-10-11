"""
Logger Utility

Provides structured logging for the multi-agent system.
Supports console, file, and JSON logging with context injection.
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import time

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
agent_name_var: ContextVar[Optional[str]] = ContextVar("agent_name", default=None)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON for easy parsing and analysis.
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add context variables
        if request_id := request_id_var.get():
            log_data["request_id"] = request_id
        if user_id := user_id_var.get():
            log_data["user_id"] = user_id
        if agent_name := agent_name_var.get():
            log_data["agent_name"] = agent_name
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "exc_info", "exc_text", "thread", "threadName",
                    "message", "asctime",
                }
            }
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for better readability.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, fmt: Optional[str] = None):
        default_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        super().__init__(fmt or default_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        # Add context to message
        extras = []
        if request_id := request_id_var.get():
            extras.append(f"req={request_id[:8]}")
        if agent_name := agent_name_var.get():
            extras.append(f"agent={agent_name}")
        
        if extras:
            record.msg = f"[{' '.join(extras)}] {record.msg}"
        
        # Apply color
        color = self.COLORS.get(record.levelname, "")
        formatted = super().format(record)
        
        if color:
            return f"{color}{formatted}{self.RESET}"
        return formatted


class AgentLogger:
    """
    Logger wrapper with agent-specific functionality.
    
    Provides context-aware logging for multi-agent systems.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log_with_context(
        self,
        level: int,
        msg: str,
        *args,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log with additional context."""
        extra = extra or {}
        extra["logger_name"] = self.name
        
        # Add timing if available
        if "duration_ms" in kwargs:
            extra["duration_ms"] = kwargs.pop("duration_ms")
        
        self.logger.log(level, msg, *args, extra=extra, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        kwargs["exc_info"] = True
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def log_agent_action(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> None:
        """Log an agent action with structured data."""
        status = "completed" if success else "failed"
        msg = f"Agent action: {action} - {status}"
        
        extra = {"action": action, "status": status}
        if details:
            extra["details"] = details
        
        level = logging.INFO if success else logging.ERROR
        self._log_with_context(level, msg, extra=extra)
    
    def log_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
        success: bool = True,
    ) -> None:
        """Log an LLM API call."""
        status = "success" if success else "failed"
        msg = f"LLM call to {model}: {status} ({duration_ms:.0f}ms)"
        
        extra = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_ms": duration_ms,
            "success": success,
        }
        
        level = logging.INFO if success else logging.ERROR
        self._log_with_context(level, msg, extra=extra)
    
    def log_retrieval(
        self,
        query: str,
        num_results: int,
        duration_ms: float,
    ) -> None:
        """Log a RAG retrieval operation."""
        msg = f"Retrieval: {num_results} results in {duration_ms:.0f}ms"
        
        extra = {
            "query_preview": query[:100],
            "num_results": num_results,
            "duration_ms": duration_ms,
        }
        
        self._log_with_context(logging.INFO, msg, extra=extra)


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level
        log_file: Path to log file (optional)
        json_format: Use JSON formatting
        max_bytes: Max size per log file
        backup_count: Number of backup files to keep
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())
    
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str) -> AgentLogger:
    """Get a logger instance for the given name."""
    return AgentLogger(name)


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> None:
    """Set context variables for the current request."""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if agent_name:
        agent_name_var.set(agent_name)


def clear_request_context() -> None:
    """Clear all context variables."""
    request_id_var.set(None)
    user_id_var.set(None)
    agent_name_var.set(None)


def log_execution_time(logger: Optional[AgentLogger] = None) -> Callable:
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time()
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        _logger = logger or get_logger(func.__module__)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                _logger.debug(
                    f"{func.__name__} completed",
                    duration_ms=duration_ms,
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                _logger.error(
                    f"{func.__name__} failed: {e}",
                    duration_ms=duration_ms,
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                _logger.debug(
                    f"{func.__name__} completed",
                    duration_ms=duration_ms,
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                _logger.error(
                    f"{func.__name__} failed: {e}",
                    duration_ms=duration_ms,
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Initialize logging on import
setup_logging()
