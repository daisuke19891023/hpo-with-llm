# Application Module

The `app` module provides the main application entry point and coordination logic.

## Overview

The application module handles:
- Application initialization
- Interface creation based on configuration
- Environment variable loading
- Graceful error handling

## Classes

### Application

The main application class that coordinates the entire system.

```python
from clean_interfaces.app import Application

class Application(BaseComponent):
    """Main application class."""
    
    def __init__(self, dotenv_path: Path | None = None) -> None:
        """Initialize the application.
        
        Args:
            dotenv_path: Optional path to .env file
        """
```

#### Methods

##### run()

```python
def run(self) -> None:
    """Run the application with the configured interface.
    
    Raises:
        ValueError: If interface type is invalid
        Exception: Any exception from the interface
    """
```

Executes the application:
1. Loads environment variables from dotenv file (if specified)
2. Creates the appropriate interface based on INTERFACE_TYPE
3. Runs the interface
4. Handles errors gracefully

## Functions

### create_app()

```python
def create_app(dotenv_path: Path | None = None) -> Application:
    """Create an application instance.
    
    Args:
        dotenv_path: Optional path to .env file
        
    Returns:
        Application: Configured application instance
        
    Example:
        app = create_app(Path(".env.production"))
        app.run()
    """
```

Factory function for creating application instances.

### run_app()

```python
def run_app(dotenv_path: Path | None = None) -> None:
    """Create and run the application.
    
    Args:
        dotenv_path: Optional path to .env file
        
    Example:
        # Run with default settings
        run_app()
        
        # Run with specific env file
        run_app(Path(".env.development"))
    """
```

Convenience function that creates and runs the application in one call.

## Usage Examples

### Basic Usage

```python
from clean_interfaces.app import run_app

# Run with default configuration
run_app()
```

### With Environment File

```python
from pathlib import Path
from clean_interfaces.app import create_app

# Create app with specific env file
app = create_app(Path(".env.production"))

# Run the application
app.run()
```

### As Script Entry Point

```python
# main.py
from pathlib import Path
from clean_interfaces.app import run_app
import typer

def main(dotenv: Path | None = None) -> None:
    """Run the application."""
    run_app(dotenv)

if __name__ == "__main__":
    typer.run(main)
```

### Error Handling

```python
from clean_interfaces.app import Application

app = Application()

try:
    app.run()
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Application error: {e}")
```

## Environment Variables

The application respects these environment variables:

- `INTERFACE_TYPE`: Type of interface to use (cli, restapi)
- `LOG_LEVEL`: Logging level
- `LOG_FORMAT`: Logging format
- Other interface-specific variables

## Integration Points

The application module integrates with:

- **[Interfaces](interfaces.md)**: Creates and runs the selected interface
- **[Settings](utils.md#settings)**: Loads configuration from environment
- **[Logging](utils.md#logging)**: Provides structured logging
- **Base Components**: Inherits logging capabilities from BaseComponent class

## Best Practices

1. **Use factory functions**: Prefer `create_app()` over direct instantiation
2. **Handle errors**: Wrap `run()` calls in try-except blocks
3. **Configure via environment**: Use environment variables for configuration
4. **Load dotenv files**: Use dotenv files for environment-specific settings

## See Also

- [Interfaces Module](interfaces.md) - Available interface implementations
- [Configuration Guide](../configuration.md) - Configuration options
- [Environment Variables](../guides/environment.md) - Environment setup