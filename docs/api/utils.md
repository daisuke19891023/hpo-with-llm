# Utilities Module

The `utils` module provides utility functions and classes for file handling, logging, and settings management.

## File Handler

File operations with encoding support and structured logging.

### FileHandler

```python
from clean_interfaces.utils.file_handler import FileHandler

class FileHandler(BaseComponent):
    """Handle file operations with encoding support."""

    def __init__(self, encoding: str = "utf-8") -> None:
        """Initialize with encoding.

        Args:
            encoding: Default file encoding
        """
```

#### Methods

##### read_text()

```python
def read_text(
    self,
    file_path: Path | str,
    encoding: str | None = None
) -> str:
    """Read text file.

    Args:
        file_path: Path to file
        encoding: Override default encoding

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file encoding is wrong
    """
```

##### write_text()

```python
def write_text(
    self,
    file_path: Path | str,
    content: str,
    encoding: str | None = None,
    create_parents: bool = True
) -> None:
    """Write text file.

    Args:
        file_path: Path to file
        content: Content to write
        encoding: Override default encoding
        create_parents: Create parent directories
    """
```

##### read_json()

```python
def read_json(
    self,
    file_path: Path | str,
    encoding: str | None = None
) -> dict[str, Any]:
    """Read JSON file.

    Returns:
        Parsed JSON data

    Raises:
        json.JSONDecodeError: If JSON is invalid
    """
```

##### write_json()

```python
def write_json(
    self,
    file_path: Path | str,
    data: dict[str, Any],
    encoding: str | None = None,
    indent: int = 2,
    ensure_ascii: bool = False,
    create_parents: bool = True
) -> None:
    """Write JSON file with formatting options."""
```

##### read_yaml()

```python
def read_yaml(
    self,
    file_path: Path | str,
    encoding: str | None = None
) -> dict[str, Any]:
    """Read YAML file."""
```

##### write_yaml()

```python
def write_yaml(
    self,
    file_path: Path | str,
    data: dict[str, Any],
    encoding: str | None = None,
    default_flow_style: bool = False,
    create_parents: bool = True
) -> None:
    """Write YAML file."""
```

### Convenience Functions

```python
from clean_interfaces.utils.file_handler import (
    read_text, write_text,
    read_json, write_json,
    read_yaml, write_yaml
)

# Quick file operations
content = read_text("file.txt")
write_json("data.json", {"key": "value"})
```

## Logging

Structured logging configuration. OpenTelemetry exporter integration has been removed; trace context may still be included if OpenTelemetry is present.

### configure_logging()

```python
from clean_interfaces.utils.logger import configure_logging

def configure_logging(
    level: str = "INFO",
    format: str = "json",
    log_file: Path | None = None,
) -> None:
    """Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("json" or "console")
        log_file: Optional log file path

    """
```

### get_logger()

```python
from clean_interfaces.utils.logger import get_logger

def get_logger(name: str | None = None) -> BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name (defaults to caller's module)

    Returns:
        Configured structlog logger

    Example:
        logger = get_logger(__name__)
        logger.info("starting", component="MyApp")
    """
```

### log_performance

```python
from clean_interfaces.utils.logger import log_performance

@log_performance
def slow_operation(data: list[int]) -> int:
    """Decorator logs function execution time."""
    return sum(data)
```

## OpenTelemetry Exporter (removed)

The OpenTelemetry exporter components were removed to improve stability. Logs can still include trace context if OTEL tracing is active.

## Settings

Application settings management with environment variable support.

### LoggingSettings

```python
from clean_interfaces.utils.settings import LoggingSettings

class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_file: Path | None = Field(default=None)
    otel_export_enabled: bool = Field(default=False)
    otel_export_mode: str = Field(default="file")
    otel_log_file: Path | None = Field(default=None)
    otel_endpoint: str | None = Field(default=None)
    otel_headers: str | None = Field(default=None)
    otel_timeout: int = Field(default=30, ge=1, le=300)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
```

### InterfaceSettings

```python
from clean_interfaces.utils.settings import InterfaceSettings

class InterfaceSettings(BaseSettings):
    """Interface configuration settings."""

    interface_type: InterfaceType = Field(
        default=InterfaceType.CLI,
        description="Type of interface to use"
    )
```

### get_logging_settings()

```python
from clean_interfaces.utils.settings import get_logging_settings

def get_logging_settings() -> LoggingSettings:
    """Get logging settings (singleton).

    Returns:
        Cached logging settings instance
    """
```

### get_interface_settings()

```python
from clean_interfaces.utils.settings import get_interface_settings

def get_interface_settings() -> InterfaceSettings:
    """Get interface settings (singleton).

    Returns:
        Cached interface settings instance
    """
```

## Usage Examples

### File Operations

```python
from clean_interfaces.utils.file_handler import FileHandler

# Context manager usage
with FileHandler(encoding="utf-8") as handler:
    # Read operations
    text = handler.read_text("input.txt")
    data = handler.read_json("config.json")
    config = handler.read_yaml("settings.yaml")

    # Write operations
    handler.write_text("output.txt", "Hello World")
    handler.write_json("data.json", {"key": "value"})
    handler.write_yaml("config.yaml", {"debug": True})
```

### Logging Setup

```python
from clean_interfaces.utils.logger import configure_logging, get_logger
from clean_interfaces.utils.settings import get_logging_settings

# Configure from settings
settings = get_logging_settings()
configure_logging(
    level=settings.log_level,
    format=settings.log_format,
    log_file=settings.log_file
)

# Get logger
logger = get_logger(__name__)
logger.info("application_started", version="1.0.0")
```

### OpenTelemetry Integration

```python
from clean_interfaces.utils.otel_exporter import create_exporters
from clean_interfaces.utils.settings import get_logging_settings

settings = get_logging_settings()
exporters = create_exporters(settings)

# Configure logging with exporters
configure_logging(
    otel_export_enabled=True,
    otel_exporters=exporters
)
```

## Best Practices

1. **Use context managers**: Use `with FileHandler()` for automatic cleanup
2. **Handle encoding**: Always specify encoding for text files
3. **Structure logs**: Use keyword arguments for structured logging
4. **Cache settings**: Use getter functions for settings singletons
5. **Configure early**: Set up logging before other operations

## See Also

-   [Logging Guide](../guides/logging.md) - Detailed logging configuration
-   [Environment Variables](../guides/environment.md) - Settings configuration
-   **Base Components** - Component base classes in the base module
