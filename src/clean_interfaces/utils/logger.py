"""Structured logging configuration with OpenTelemetry compatibility.

This module provides a structured logging setup using structlog that is compatible
with OpenTelemetry tracing when available. It supports multiple output formats
and can be configured for different environments.
"""

import datetime
import inspect
import logging
import logging.handlers
import sys
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Protocol, TypeVar

import structlog
from structlog.types import EventDict, Processor

from clean_interfaces.utils.settings import get_settings

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])

# OpenTelemetry exporter integration removed to avoid test freezes.


class LoggerProtocol(Protocol):
    """Protocol defining the logger interface."""

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message."""
        ...

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log critical message."""
        ...

    def bind(self, **kwargs: Any) -> "LoggerProtocol":
        """Bind context to logger."""
        ...


def add_timestamp(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to log entries."""
    event_dict["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    return event_dict


def add_caller_info(
    _logger: Any,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add caller information to log entries."""
    # Get caller frame (skip structlog internal frames)

    frame = inspect.currentframe()
    if frame is not None:
        # Skip through structlog frames to find actual caller
        while frame and frame.f_code.co_filename.find("structlog") != -1:
            frame = frame.f_back

        if frame and frame.f_back:
            caller_frame = frame.f_back
            event_dict["caller"] = {
                "filename": Path(caller_frame.f_code.co_filename).name,
                "function": caller_frame.f_code.co_name,
                "line": caller_frame.f_lineno,
            }

    return event_dict


def add_opentelemetry_context(
    _logger: Any,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add OpenTelemetry trace context if available."""
    try:
        # Try to import OpenTelemetry and get current span
        from opentelemetry import trace  # type: ignore[import-untyped]

        span = trace.get_current_span()  # type: ignore[attr-defined]
        if span and span.is_recording():  # type: ignore[attr-defined]
            span_context = span.get_span_context()  # type: ignore[attr-defined]
            if span_context.is_valid:  # type: ignore[attr-defined]
                event_dict["trace_id"] = format(span_context.trace_id, "032x")  # type: ignore[attr-defined]
                event_dict["span_id"] = format(span_context.span_id, "016x")  # type: ignore[attr-defined]
                event_dict["trace_flags"] = format(span_context.trace_flags, "02x")  # type: ignore[attr-defined]
    except ImportError:
        # OpenTelemetry not available, continue without trace context
        pass
    except Exception:
        # Any other error, continue without trace context
        pass

    return event_dict


def _build_processors(
    include_timestamp: bool,
    include_caller: bool,
    include_otel_context: bool,
) -> list[Processor]:
    """Build the processor chain for structlog."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
    ]

    if include_timestamp:
        processors.append(add_timestamp)

    if include_caller:
        processors.append(add_caller_info)

    if include_otel_context:
        processors.append(add_opentelemetry_context)

    return processors


def _add_format_processors(processors: list[Processor], log_format: str) -> None:
    """Add format-specific processors to the chain."""
    if log_format == "json":
        processors.extend(
            [
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
        )
    elif log_format == "console":
        processors.extend(
            [
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )
    else:  # plain format
        processors.extend(
            [
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.KeyValueRenderer(),
            ],
        )


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: str | None = None,
    include_timestamp: bool = True,
    include_caller: bool = False,
    include_otel_context: bool = True,
) -> None:
    """Configure structured logging with the specified options.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('json', 'console', or 'plain')
        log_file: Optional file path for log output
        include_timestamp: Whether to include timestamps in log entries
        include_caller: Whether to include caller information
        include_otel_context: Whether to include OpenTelemetry trace context if present

    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Build processor chain
    processors = _build_processors(
        include_timestamp,
        include_caller,
        include_otel_context,
    )

    # Add format-specific processors
    _add_format_processors(processors, log_format)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handlers: list[logging.Handler] = []

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=handlers,
        force=True,  # Force reconfiguration
    )


def get_logger(name: str) -> LoggerProtocol:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically the module name

    Returns:
        Configured logger instance

    """
    return structlog.get_logger(name)  # type: ignore[return-value]


def shutdown_logging() -> None:
    """Shutdown logging and clean up resources."""
    # No special cleanup required after removing OTEL exporters
    return


def log_performance(logger: LoggerProtocol) -> Callable[[F], F]:
    """Log function performance metrics.

    Args:
        logger: Logger instance to use for performance logging

    Returns:
        Decorator function

    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            exception_info = None

            try:
                return func(*args, **kwargs)
            except Exception as e:
                exception_info = f"{type(e).__name__}: {e}"
                raise
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000

                log_data = {
                    "function_name": func.__name__,
                    "duration_ms": round(duration_ms, 3),
                }

                if exception_info:
                    log_data["exception"] = exception_info
                    logger.error("Function execution failed", **log_data)
                else:
                    logger.info("Function execution completed", **log_data)

        return wrapper  # type: ignore[return-value]

    return decorator


# Convenience function for common usage patterns
def setup_application_logging(
    app_name: str,
    environment: str = "development",
    log_file: str | None = None,
) -> LoggerProtocol:
    """Set up logging for an application with sensible defaults.

    Args:
        app_name: Application name for the logger
        environment: Environment ('development', 'production', 'testing')
        log_file: Optional log file path
        use_otel: Whether to enable OpenTelemetry export based on settings

    Returns:
        Configured application logger

    """
    # Load settings for default log configuration
    settings = get_settings()

    if environment == "production":
        configure_logging(
            log_level="INFO",
            log_format="json",
            log_file=log_file or (settings.log_file_path if settings else None),
            include_timestamp=True,
            include_caller=False,
            include_otel_context=True,
        )
    elif environment == "testing":
        configure_logging(
            log_level="WARNING",
            log_format="json",
            log_file=log_file,
            include_timestamp=True,
            include_caller=False,
            include_otel_context=False,
        )
    else:  # development
        configure_logging(
            log_level="DEBUG",
            log_format="console",
            log_file=log_file or (settings.log_file_path if settings else None),
            include_timestamp=True,
            include_caller=True,
            include_otel_context=True,
        )

    return get_logger(app_name)
