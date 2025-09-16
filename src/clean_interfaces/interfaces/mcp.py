"""MCP interface implementation using FastMCP."""

from fastmcp import FastMCP

from clean_interfaces.models.io import WelcomeMessage

from .base import BaseInterface


class MCPInterface(BaseInterface):
    """MCP Interface implementation."""

    def __init__(self) -> None:
        """Initialize the MCP interface."""
        super().__init__()
        self.mcp = FastMCP(name="clean-interfaces-mcp")
        self._setup_commands()

    @property
    def name(self) -> str:
        """Get the interface name.

        Returns:
            str: The interface name

        """
        return "MCP"

    def _setup_commands(self) -> None:
        """Set up MCP commands."""

        @self.mcp.tool()
        def welcome() -> str:  # pyright: ignore [reportUnusedFunction]
            """Display welcome message."""
            msg = WelcomeMessage()
            return f"{msg.message}\n{msg.hint}"

    def run(self) -> None:
        """Run the MCP interface."""
        self.logger.info("Starting MCP server")
        self.mcp.run()
