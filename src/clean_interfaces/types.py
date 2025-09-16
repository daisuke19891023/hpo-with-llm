"""Type definitions for clean interfaces."""

from enum import Enum


class InterfaceType(str, Enum):
    """Available interface types."""

    CLI = "cli"
    RESTAPI = "restapi"
    MCP = "mcp"
