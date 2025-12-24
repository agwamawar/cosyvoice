"""Logging configuration and utilities.

Provides a complete logging system with:
- JSON formatter for production (structured logs)
- Pretty formatter for development (colored, readable)
- Log context using contextvars for request-scoped data
- Named loggers for different components
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any

# Context variable for request-scoped logging data
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})

# ANSI color codes for pretty formatting
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
}


class LogContext:
    """Context manager for request-scoped logging data."""

    def __init__(self, **kwargs: Any) -> None:
        self._token = None
        self._data = kwargs

    def __enter__(self) -> "LogContext":
        current = _log_context.get().copy()
        current.update(self._data)
        self._token = _log_context.set(current)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _log_context.reset(self._token)

    def update(self, **kwargs: Any) -> None:
        """Update the current context with additional data."""
        current = _log_context.get().copy()
        current.update(kwargs)
        _log_context.set(current)


def get_log_context() -> dict[str, Any]:
    """Get the current logging context."""
    return _log_context.get().copy()


def set_log_context(**kwargs: Any) -> None:
    """Set values in the current logging context."""
    current = _log_context.get().copy()
    current.update(kwargs)
    _log_context.set(current)


def clear_log_context() -> None:
    """Clear the current logging context."""
    _log_context.set({})


class ContextFilter(logging.Filter):
    """Filter that injects context variables into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _log_context.get()
        for key, value in context.items():
            setattr(record, key, value)
        # Always include request_id if present
        if not hasattr(record, "request_id"):
            record.request_id = context.get("request_id", "-")
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for production structured logging."""

    def __init__(self, service_name: str = "tts-backend") -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        # Base log data
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        # Add request_id if present
        request_id = getattr(record, "request_id", None)
        if request_id and request_id != "-":
            log_data["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add location info
        log_data["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add any extra context fields
        context = _log_context.get()
        for key, value in context.items():
            if key not in log_data:
                log_data[key] = value

        # Add extra fields from the record
        skip_fields = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "exc_info", "exc_text", "thread", "threadName", "taskName",
            "message", "request_id",
        }
        for key, value in record.__dict__.items():
            if key not in skip_fields and not key.startswith("_"):
                try:
                    json.dumps(value)  # Test if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, default=str)


class PrettyFormatter(logging.Formatter):
    """Pretty formatter for development with colors."""

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Get colors
        if self.use_colors:
            level_color = COLORS.get(record.levelname, "")
            reset = COLORS["RESET"]
            dim = COLORS["DIM"]
            bold = COLORS["BOLD"]
        else:
            level_color = reset = dim = bold = ""

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # Format level (padded)
        level = f"{level_color}{record.levelname:<8}{reset}"

        # Format logger name (shortened)
        logger_name = record.name
        if len(logger_name) > 20:
            parts = logger_name.split(".")
            logger_name = ".".join(p[0] for p in parts[:-1]) + "." + parts[-1]

        # Get request_id from context
        request_id = getattr(record, "request_id", None)
        if not request_id or request_id == "-":
            request_id = _log_context.get().get("request_id", "")

        # Build the message
        parts = [
            f"{dim}{timestamp}{reset}",
            level,
            f"{dim}{logger_name:<20}{reset}",
        ]

        if request_id:
            parts.append(f"{dim}[{request_id[:8]}]{reset}")

        parts.append(f"{bold}{record.getMessage()}{reset}")

        message = " │ ".join(parts)

        # Add extra context on new lines
        context = _log_context.get()
        extras = {k: v for k, v in context.items() if k != "request_id"}

        # Add record extras
        skip_fields = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "exc_info", "exc_text", "thread", "threadName", "taskName",
            "message", "request_id",
        }
        for key, value in record.__dict__.items():
            if key not in skip_fields and not key.startswith("_"):
                extras[key] = value

        if extras:
            extras_str = " ".join(f"{dim}{k}={reset}{v}" for k, v in extras.items())
            message += f"\n         {extras_str}"

        # Add exception info
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


class ContextLogger:
    """Logger wrapper that supports structured logging with context."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        self._static_context: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> "ContextLogger":
        """Create a new logger with additional static context."""
        new_logger = ContextLogger(self._logger.name)
        new_logger._static_context = {**self._static_context, **kwargs}
        return new_logger

    def _format_message(self, message: str, kwargs: dict[str, Any]) -> str:
        """Format message with inline key-value pairs for pretty output."""
        if kwargs:
            extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} | {extras}"
        return message

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Log a message with context."""
        extra = {**self._static_context, **kwargs}
        # For JSON format, pass extras directly; for pretty, format inline
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        extra = {**self._static_context, **kwargs}
        self._logger.exception(message, extra=extra)


# Pre-configured named loggers
_loggers: dict[str, ContextLogger] = {}


def get_logger(name: str) -> ContextLogger:
    """
    Get a context-aware logger instance.

    Named loggers available:
    - api: API request/response logging
    - inference: Model inference logging
    - audio: Audio processing logging
    - system: System/startup logging

    Args:
        name: Logger name (e.g., 'api', 'inference', 'src.api.routers.synthesis')

    Returns:
        ContextLogger instance
    """
    if name not in _loggers:
        _loggers[name] = ContextLogger(name)
    return _loggers[name]


def setup_logging(
    level: str = "INFO",
    format: str = "pretty",
    service_name: str = "tts-backend",
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ('json' or 'pretty')
        service_name: Name of the service for log context
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Add context filter
    console_handler.addFilter(ContextFilter())

    # Set formatter based on format type
    if format == "json":
        console_handler.setFormatter(JSONFormatter(service_name=service_name))
    else:
        console_handler.setFormatter(PrettyFormatter())

    root_logger.addHandler(console_handler)

    # Configure named loggers
    for logger_name in ["api", "inference", "audio", "system"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Log startup
    system_logger = get_logger("system")
    system_logger.info(
        "Logging initialized",
        log_level=level,
        log_format=format,
        service=service_name,
    )
