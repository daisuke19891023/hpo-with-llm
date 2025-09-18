"""Tests for MCP interface implementation."""

from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP

from clean_interfaces.interfaces.base import BaseInterface
from clean_interfaces.interfaces.mcp import MCPInterface


class TestMCPInterface:
    """Test MCP interface functionality."""

    def test_mcp_interface_inherits_base(self) -> None:
        """Test that MCPInterface inherits from BaseInterface."""
        assert issubclass(MCPInterface, BaseInterface)

    def test_mcp_interface_has_name(self) -> None:
        """Test that MCPInterface has correct name."""
        mcp = MCPInterface()
        assert mcp.name == "MCP"

    def test_mcp_interface_has_fastmcp_app(self) -> None:
        """Test that MCPInterface has FastMCP app."""
        mcp = MCPInterface()
        assert hasattr(mcp, "mcp")
        assert isinstance(mcp.mcp, FastMCP)

    @pytest.mark.asyncio  # pyright: ignore [reportUnknownMemberType, reportUntypedFunctionDecorator, reportAttributeAccessIssue]
    async def test_mcp_welcome_command(self) -> None:
        """Test MCP welcome command functionality."""
        mcp = MCPInterface()
        tools = await mcp.mcp.get_tools()
        assert "welcome" in tools
        assert "reflect_hpo" in tools

    @patch("fastmcp.FastMCP.run")
    def test_mcp_run_method(self, mock_run: MagicMock) -> None:
        """Test MCP run method executes fastmcp app."""
        mcp = MCPInterface()
        mcp.run()
        mock_run.assert_called_once()

    def test_mcp_reflect_tool_returns_payload(self) -> None:
        """Ensure the reflect_hpo tool executes an optimisation loop."""
        mcp = MCPInterface()
        payload = mcp.reflect_hpo(
            task_description="Tune prompts",
            mode="baseline",
            max_trials=1,
            direction="maximize",
        )

        assert isinstance(payload, dict)
        reflection = cast(dict[str, Any], payload["reflection"])
        assert reflection["mode"] == "baseline"
        assert "summary" in reflection
        trials = cast(Sequence[Any], payload["trials"])
        assert len(list(trials)) > 0
