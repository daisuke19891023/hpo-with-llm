# Interfaces Module

The `interfaces` module provides different interface implementations for the Clean Interfaces application.

## Overview

The interfaces module implements the Strategy pattern, allowing the application to switch between different interface types at runtime.

## Base Interface

### BaseInterface

Abstract base class for all interfaces.

```python
from abc import ABC, abstractmethod
from clean_interfaces.interfaces.base import BaseInterface

class BaseInterface(ABC):
    """Abstract base interface."""
    
    @abstractmethod
    def run(self) -> None:
        """Run the interface."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get interface name."""
        pass
```

All interfaces must implement:
- `run()`: Main execution method
- `name`: Property returning the interface name

## CLI Interface

### CLIInterface

Command-line interface implementation using Typer.

```python
from clean_interfaces.interfaces.cli import CLIInterface

class CLIInterface(BaseInterface):
    """Command-line interface implementation."""
    
    def __init__(self) -> None:
        """Initialize CLI interface with Typer app."""
        self.app = typer.Typer()
        self._setup_commands()
```

#### Features

- Interactive command-line interface
- Built-in help system
- Command completion support
- Colored output

#### Commands

##### welcome

```python
@app.command()
def welcome() -> None:
    """Display welcome message."""
```

Displays the application welcome message.

#### Usage

```bash
# Run CLI interface
export INTERFACE_TYPE=cli
python -m clean_interfaces.main

# Get help
python -m clean_interfaces.main --help
```

## REST API Interface

### RestAPIInterface

REST API implementation using FastAPI.

```python
from clean_interfaces.interfaces.restapi import RestAPIInterface

class RestAPIInterface(BaseInterface):
    """REST API interface implementation."""
    
    def __init__(self) -> None:
        """Initialize REST API with FastAPI app."""
        self.app = FastAPI(
            title="Clean Interfaces API",
            version="1.0.0"
        )
        self._setup_routes()
```

#### Features

- RESTful API endpoints
- Automatic API documentation
- Request/response validation
- CORS support (configurable)

#### Endpoints

##### GET /

Redirects to API documentation.

##### GET /health

```python
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
```

Returns application health status.

##### GET /api/v1/welcome

```python
@app.get("/api/v1/welcome", response_model=WelcomeResponse)
def welcome() -> WelcomeResponse:
    """Welcome endpoint."""
```

Returns welcome message and version.

#### Usage

```bash
# Run REST API
export INTERFACE_TYPE=restapi
python -m clean_interfaces.main

# Or with uvicorn
uvicorn clean_interfaces.main:app --reload
```

## MCP Interface

### MCPInterface

MCP (Model Context Protocol) implementation using FastMCP.

```python
from clean_interfaces.interfaces.mcp import MCPInterface

class MCPInterface(BaseInterface):
    """MCP interface implementation."""

    def __init__(self) -> None:
        """Initialize MCP interface with FastMCP app."""
        self.mcp = FastMCP(
            name="clean-interfaces-mcp",
            title="Clean Interfaces MCP",
            description="A clean interface MCP implementation",
            version="1.0.0",
        )
        self._setup_commands()
```

#### Features

- Standardized communication with LLMs
- Tool and resource discovery
- Asynchronous operation support

#### Tools

##### welcome

```python
@mcp.tool()
def welcome() -> str:
    """Display welcome message."""
```

Returns a welcome message string.

#### Usage

```bash
# Run MCP interface
export INTERFACE_TYPE=mcp
python -m clean_interfaces.main
```

## Interface Factory

### create_interface()

Factory function for creating interfaces.

```python
from clean_interfaces.interfaces.factory import create_interface
from clean_interfaces.types import InterfaceType

def create_interface(interface_type: InterfaceType) -> BaseInterface:
    """Create interface based on type.
    
    Args:
        interface_type: Type of interface to create
        
    Returns:
        BaseInterface: Interface instance
        
    Raises:
        ValueError: If interface type is not supported
        
    Example:
        interface = create_interface(InterfaceType.CLI)
        interface.run()
    """
```

#### Supported Types

- `InterfaceType.CLI`: Creates CLIInterface
- `InterfaceType.RESTAPI`: Creates RestAPIInterface
- `InterfaceType.MCP`: Creates MCPInterface

## Usage Examples

### Direct Interface Creation

```python
from clean_interfaces.interfaces import CLIInterface, RestAPIInterface

# Create CLI interface
cli = CLIInterface()
cli.run()

# Create REST API interface
api = RestAPIInterface()
api.run()
```

### Using Factory

```python
from clean_interfaces.interfaces import create_interface
from clean_interfaces.types import InterfaceType

# Create based on enum
interface = create_interface(InterfaceType.RESTAPI)
interface.run()

# Create based on settings
from clean_interfaces.utils.settings import get_interface_settings
settings = get_interface_settings()
interface = create_interface(settings.interface_type)
interface.run()
```

### Custom Interface Implementation

```python
from clean_interfaces.interfaces import BaseInterface

class CustomInterface(BaseInterface):
    """Custom interface implementation."""
    
    @property
    def name(self) -> str:
        return "custom"
    
    def run(self) -> None:
        print("Running custom interface")
```

## Configuration

Interfaces can be configured via environment variables:

### CLI Configuration

- `LOG_LEVEL`: Set logging verbosity
- `LOG_FORMAT`: Set output format

### REST API Configuration

- `HOST`: Bind host (default: 0.0.0.0)
- `PORT`: Bind port (default: 8000)
- `LOG_LEVEL`: API logging level

## Best Practices

1. **Use the factory**: Use `create_interface()` instead of direct instantiation
2. **Handle interface errors**: Wrap `run()` calls in error handlers
3. **Extend carefully**: When adding new interfaces, follow the BaseInterface contract
4. **Configure via environment**: Use environment variables for runtime configuration

## See Also

- [Application Module](app.md) - Main application coordination
- [Models Module](models.md) - Data models for interfaces
- **Interface Types** - InterfaceType enumeration in the types module