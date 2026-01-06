"""
Centralized Error Handler.

Provides custom exceptions, error handling decorators, and recovery mechanisms
for the Retail Vision Analytics system.

Usage:
    from src.handlers import ErrorHandler, InferenceError, error_handler
    
    # Using decorator
    @error_handler(max_retries=3, fallback=default_result)
    def process_frame(frame):
        ...
    
    # Using context manager
    with ErrorHandler(component="detector"):
        result = model.infer(image)
    
    # Raising custom errors
    raise InferenceError("TensorRT engine failed", details={"gpu_memory": "OOM"})
"""

import functools
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

# Type variable for generic return types
T = TypeVar("T")


# =============================================================================
# Custom Exceptions
# =============================================================================

class RetailVisionError(Exception):
    """Base exception for Retail Vision Analytics."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def __str__(self) -> str:
        base = f"[{self.code}] {self.message}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base += f" ({details_str})"
        return base


class ConfigurationError(RetailVisionError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, code="CONFIG_ERROR", details=details, **kwargs)


class ModelError(RetailVisionError):
    """Model loading and initialization errors."""
    
    def __init__(self, message: str, model_path: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if model_path:
            details["model_path"] = model_path
        super().__init__(message, code="MODEL_ERROR", details=details, **kwargs)


class InferenceError(RetailVisionError):
    """Inference and prediction errors."""
    
    def __init__(
        self,
        message: str,
        batch_size: Optional[int] = None,
        input_shape: Optional[tuple] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if batch_size:
            details["batch_size"] = batch_size
        if input_shape:
            details["input_shape"] = input_shape
        super().__init__(message, code="INFERENCE_ERROR", details=details, **kwargs)


class StreamError(RetailVisionError):
    """Video stream errors."""
    
    def __init__(
        self,
        message: str,
        stream_id: Optional[str] = None,
        stream_uri: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if stream_id:
            details["stream_id"] = stream_id
        if stream_uri:
            details["stream_uri"] = stream_uri
        super().__init__(message, code="STREAM_ERROR", details=details, **kwargs)


class StorageError(RetailVisionError):
    """Storage and database errors."""
    
    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if storage_type:
            details["storage_type"] = storage_type
        if operation:
            details["operation"] = operation
        super().__init__(message, code="STORAGE_ERROR", details=details, **kwargs)


class APIError(RetailVisionError):
    """API and HTTP errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["status_code"] = status_code
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, code="API_ERROR", details=details, **kwargs)


class GPUError(RetailVisionError):
    """GPU and CUDA errors."""
    
    def __init__(
        self,
        message: str,
        gpu_id: Optional[int] = None,
        cuda_error: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if gpu_id is not None:
            details["gpu_id"] = gpu_id
        if cuda_error:
            details["cuda_error"] = cuda_error
        super().__init__(
            message, code="GPU_ERROR", details=details, recoverable=False, **kwargs
        )


# =============================================================================
# Error Severity
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Error Record
# =============================================================================

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    
    error_type: str
    message: str
    timestamp: datetime
    component: str
    severity: ErrorSeverity
    traceback: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


# =============================================================================
# Error Handler Class
# =============================================================================

class ErrorHandler:
    """
    Centralized error handler with logging, metrics, and recovery.
    
    Usage:
        handler = ErrorHandler(component="detector")
        
        # As context manager
        with handler:
            result = model.infer(image)
        
        # Manual handling
        try:
            result = model.infer(image)
        except Exception as e:
            handler.handle(e)
    """
    
    _instance: Optional["ErrorHandler"] = None
    _error_history: List[ErrorRecord] = []
    _max_history: int = 1000
    
    def __init__(
        self,
        component: str = "unknown",
        logger: Optional[logging.Logger] = None,
        raise_on_error: bool = False,
        notify_callback: Optional[Callable[[ErrorRecord], None]] = None,
    ):
        self.component = component
        self.logger = logger or logging.getLogger(f"retail_vision.{component}")
        self.raise_on_error = raise_on_error
        self.notify_callback = notify_callback
        self._error_count = 0
        self._last_error: Optional[ErrorRecord] = None
    
    @classmethod
    def get_instance(cls) -> "ErrorHandler":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(component="global")
        return cls._instance
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with error handling."""
        if exc_type is not None:
            self.handle(exc_val, exc_tb)
            return not self.raise_on_error
        return False
    
    def handle(
        self,
        error: Exception,
        tb: Optional[Any] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> ErrorRecord:
        """
        Handle an error.
        
        Args:
            error: The exception to handle
            tb: Optional traceback object
            severity: Error severity level
        
        Returns:
            ErrorRecord with error details
        """
        # Get traceback
        if tb is None:
            tb_str = traceback.format_exc()
        else:
            tb_str = "".join(traceback.format_tb(tb))
        
        # Determine error type and details
        if isinstance(error, RetailVisionError):
            error_type = error.code
            message = error.message
            details = error.details
        else:
            error_type = type(error).__name__
            message = str(error)
            details = {}
        
        # Create error record
        record = ErrorRecord(
            error_type=error_type,
            message=message,
            timestamp=datetime.now(),
            component=self.component,
            severity=severity,
            traceback=tb_str,
            details=details,
        )
        
        # Update state
        self._error_count += 1
        self._last_error = record
        
        # Add to history
        ErrorHandler._error_history.append(record)
        if len(ErrorHandler._error_history) > ErrorHandler._max_history:
            ErrorHandler._error_history.pop(0)
        
        # Log error
        self._log_error(record)
        
        # Notify if callback set
        if self.notify_callback:
            try:
                self.notify_callback(record)
            except Exception as e:
                self.logger.warning(f"Error notification failed: {e}")
        
        return record
    
    def _log_error(self, record: ErrorRecord):
        """Log error with appropriate level."""
        log_message = (
            f"[{record.error_type}] {record.message} "
            f"(component={record.component})"
        )
        
        if record.details:
            log_message += f" details={record.details}"
        
        if record.severity == ErrorSeverity.DEBUG:
            self.logger.debug(log_message)
        elif record.severity == ErrorSeverity.INFO:
            self.logger.info(log_message)
        elif record.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        elif record.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
            if record.traceback:
                self.logger.debug(f"Traceback:\n{record.traceback}")
        elif record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            if record.traceback:
                self.logger.error(f"Traceback:\n{record.traceback}")
    
    @property
    def error_count(self) -> int:
        """Get total error count."""
        return self._error_count
    
    @property
    def last_error(self) -> Optional[ErrorRecord]:
        """Get last error record."""
        return self._last_error
    
    @classmethod
    def get_error_history(
        cls,
        component: Optional[str] = None,
        error_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ErrorRecord]:
        """
        Get error history with optional filters.
        
        Args:
            component: Filter by component
            error_type: Filter by error type
            since: Filter errors since datetime
            limit: Maximum records to return
        
        Returns:
            List of ErrorRecords
        """
        records = cls._error_history.copy()
        
        if component:
            records = [r for r in records if r.component == component]
        if error_type:
            records = [r for r in records if r.error_type == error_type]
        if since:
            records = [r for r in records if r.timestamp >= since]
        
        return records[-limit:]
    
    @classmethod
    def get_error_summary(cls) -> Dict[str, Any]:
        """Get summary of error statistics."""
        history = cls._error_history
        
        if not history:
            return {"total_errors": 0}
        
        # Count by type
        by_type: Dict[str, int] = {}
        for record in history:
            by_type[record.error_type] = by_type.get(record.error_type, 0) + 1
        
        # Count by component
        by_component: Dict[str, int] = {}
        for record in history:
            by_component[record.component] = by_component.get(record.component, 0) + 1
        
        # Count by severity
        by_severity: Dict[str, int] = {}
        for record in history:
            sev = record.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {
            "total_errors": len(history),
            "by_type": by_type,
            "by_component": by_component,
            "by_severity": by_severity,
            "oldest": history[0].timestamp.isoformat() if history else None,
            "newest": history[-1].timestamp.isoformat() if history else None,
        }
    
    @classmethod
    def clear_history(cls):
        """Clear error history."""
        cls._error_history.clear()


# =============================================================================
# Error Handler Decorator
# =============================================================================

def error_handler(
    max_retries: int = 0,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    fallback: Optional[T] = None,
    exceptions: tuple = (Exception,),
    component: str = "unknown",
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    reraise: bool = False,
) -> Callable:
    """
    Decorator for error handling with retry support.
    
    Args:
        max_retries: Maximum retry attempts (0 = no retries)
        retry_delay: Initial delay between retries (seconds)
        exponential_backoff: Use exponential backoff for retries
        fallback: Value to return on failure
        exceptions: Tuple of exceptions to catch
        component: Component name for logging
        severity: Error severity level
        reraise: Re-raise exception after handling
    
    Usage:
        @error_handler(max_retries=3, fallback=None)
        def process_frame(frame):
            return model.infer(frame)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = ErrorHandler(component=component)
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    
                    if attempt < max_retries:
                        # Calculate delay
                        delay = retry_delay
                        if exponential_backoff:
                            delay = retry_delay * (2 ** attempt)
                        
                        handler.logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        # Final failure
                        handler.handle(e, severity=severity)
            
            if reraise and last_error:
                raise last_error
            
            return fallback
        
        return wrapper
    return decorator


# =============================================================================
# Recovery Strategies
# =============================================================================

class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error: Exception) -> bool:
        """Check if this strategy can recover from the error."""
        return False
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt recovery."""
        raise NotImplementedError


class RetryStrategy(RecoveryStrategy):
    """Retry-based recovery strategy."""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    def can_recover(self, error: Exception) -> bool:
        if isinstance(error, RetailVisionError):
            return error.recoverable
        return True
    
    def recover(
        self,
        error: Exception,
        context: Dict[str, Any],
    ) -> Any:
        func = context.get("func")
        args = context.get("args", ())
        kwargs = context.get("kwargs", {})
        
        if not func:
            raise ValueError("No function in context")
        
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay * (attempt + 1))
                return func(*args, **kwargs)
            except Exception:
                continue
        
        raise error


class FallbackStrategy(RecoveryStrategy):
    """Fallback value recovery strategy."""
    
    def __init__(self, fallback_value: Any):
        self.fallback_value = fallback_value
    
    def can_recover(self, error: Exception) -> bool:
        return True
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        return self.fallback_value


# =============================================================================
# Utility Functions
# =============================================================================

def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    **kwargs,
) -> Optional[T]:
    """
    Safely execute a function, returning default on error.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        return default


@contextmanager
def suppress_errors(
    *exceptions: Type[Exception],
    logger: Optional[logging.Logger] = None,
):
    """
    Context manager to suppress specific exceptions.
    
    Usage:
        with suppress_errors(ValueError, TypeError):
            risky_operation()
    """
    try:
        yield
    except exceptions as e:
        if logger:
            logger.debug(f"Suppressed error: {e}")


def format_exception(error: Exception, include_traceback: bool = True) -> str:
    """Format exception for display."""
    parts = [f"{type(error).__name__}: {error}"]
    
    if isinstance(error, RetailVisionError) and error.details:
        parts.append(f"Details: {error.details}")
    
    if include_traceback:
        tb = traceback.format_exc()
        if tb and tb != "NoneType: None\n":
            parts.append(f"Traceback:\n{tb}")
    
    return "\n".join(parts)


# =============================================================================
# Module Initialization
# =============================================================================

# Create global error handler instance
_global_handler = ErrorHandler.get_instance()


def get_handler() -> ErrorHandler:
    """Get global error handler."""
    return _global_handler


def handle_error(error: Exception, **kwargs) -> ErrorRecord:
    """Handle error with global handler."""
    return _global_handler.handle(error, **kwargs)
