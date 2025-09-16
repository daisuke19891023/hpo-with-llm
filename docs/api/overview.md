# API Reference Overview

This section provides detailed API documentation for all Clean Interfaces modules and components.

## Module Structure

The Clean Interfaces package is organized into the following modules:

### Core Modules

-   **[app](app.md)**: Main application entry point and coordination
-   **base**: Base classes for components with logging
-   **types**: Type definitions and enums

### Interface Modules

-   **[interfaces.base](interfaces.md#base-interface)**: Abstract base interface
-   **[interfaces.cli](interfaces.md#cli-interface)**: Command-line interface implementation
-   **[interfaces.restapi](interfaces.md#rest-api-interface)**: REST API interface implementation
-   **[interfaces.factory](interfaces.md#interface-factory)**: Factory for creating interfaces

### Model Modules

-   **[models.api](models.md#api-models)**: REST API request/response models
-   **[models.io](models.md#io-models)**: Input/output models for CLI

### Utility Modules

-   **[utils.file_handler](utils.md#file-handler)**: File operations with encoding support
-   **[utils.logger](utils.md#logging)**: Structured logging configuration

-   **[utils.settings](utils.md#settings)**: Application settings management

## Import Structure

The package provides convenient imports:

```python
# Main application
from clean_interfaces.app import Application, create_app, run_app

# Base components
from clean_interfaces.base import BaseComponent

# Types
from clean_interfaces.types import InterfaceType

# Interfaces
from clean_interfaces.interfaces import (
    BaseInterface,
    CLIInterface,
    RestAPIInterface,
    create_interface
)

# Models
from clean_interfaces.models.api import (
    HealthResponse,
    WelcomeResponse,
    ErrorResponse
)
from clean_interfaces.models.io import WelcomeMessage

# Utilities
from clean_interfaces.utils.file_handler import FileHandler
from clean_interfaces.utils.logger import configure_logging, get_logger
from clean_interfaces.utils.settings import (
    get_logging_settings,
    get_interface_settings
)
```

## Design Principles

The API follows these design principles:

1. **Type Safety**: All public APIs use type hints
2. **Dependency Injection**: Components receive dependencies through constructors
3. **Interface Segregation**: Small, focused interfaces
4. **Single Responsibility**: Each module has a clear, single purpose
5. **Open/Closed**: Easy to extend with new interfaces without modifying core

## Common Patterns

### Context Management

Many components use context managers for resource cleanup:

```python
with FileHandler() as handler:
    content = handler.read_text("file.txt")
```

### Structured Logging

All components inherit from `BaseComponent` for consistent logging:

```python
class MyComponent(BaseComponent):
    def process(self) -> None:
        self.logger.info("processing", step="start")
```

### Settings Management

Settings use Pydantic for validation and environment variable loading:

```python
from clean_interfaces.utils.settings import get_logging_settings

settings = get_logging_settings()
print(settings.log_level)  # From LOG_LEVEL env var
```

## Error Handling

The API uses specific exception types:

-   `ValueError`: For invalid arguments or configuration
-   `FileNotFoundError`: When files don't exist
-   `json.JSONDecodeError`: For invalid JSON
-   `yaml.YAMLError`: For invalid YAML

## Thread Safety

Most components are thread-safe for read operations. Write operations may require external synchronization.

## Next Steps

-   Explore specific module documentation
-   Review the [Application](app.md) module for the main entry point
-   Check [Interfaces](interfaces.md) for available interface types
-   See [Utilities](utils.md) for helper functions
