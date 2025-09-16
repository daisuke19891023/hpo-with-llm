# MCP Interface Guide

The Model Context Protocol (MCP) interface allows language models to interact with the application.

## Overview

The MCP interface is built using [FastMCP](https://github.com/jlowin/fastmcp), providing:

- A standardized way to expose tools and resources to LLMs.
- Automatic schema generation from Python type hints.
- Support for multiple transport protocols (stdio, http, sse).

## Running the MCP Interface

### Basic Usage

```bash
# Run with default settings (stdio transport)
uv run python -m clean_interfaces.main
```

### Setting MCP as Default

To run the MCP interface, set the `INTERFACE_TYPE` environment variable:

```ini
# .env
INTERFACE_TYPE=mcp
```

## MCP Features

### Tools

Tools are functions that can be called by an MCP client.

#### Welcome Tool

The `welcome` tool returns a welcome message.

An MCP client can call this tool and receive the message:

```
"Welcome to Clean Interfaces!\nType --help for more information"
```

## Configuration for MCP

### Recommended Development Settings

```ini
# dev.env for MCP development
INTERFACE_TYPE=mcp
LOG_LEVEL=DEBUG
LOG_FORMAT=console
```

### Production MCP Settings

```ini
# prod.env for MCP production
INTERFACE_TYPE=mcp
LOG_LEVEL=WARNING
LOG_FORMAT=json
LOG_FILE_PATH=/var/log/mcp.log
```

## MCP Development

### Adding New Tools

To add new tools to the MCP interface, modify the `MCPInterface` class:

```python
# src/clean_interfaces/interfaces/mcp.py

from fastmcp import FastMCP
from .base import BaseInterface

class MCPInterface(BaseInterface):
    def __init__(self) -> None:
        super().__init__()
        self.mcp = FastMCP(name="clean-interfaces-mcp")
        self._setup_commands()

    def _setup_commands(self) -> None:
        @self.mcp.tool()
        def my_new_tool(param1: str) -> str:
            """A new tool that does something."""
            return f"You sent: {param1}"
```

## Next Steps

- Learn about the [CLI Interface](cli.md)
- Learn about the [REST API Interface](restapi.md)
- Configure [Logging](logging.md)
- Explore [Environment Variables](environment.md)
- Read the [API Reference](../api/interfaces.md)
