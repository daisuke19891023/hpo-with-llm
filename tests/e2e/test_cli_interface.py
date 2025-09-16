"""E2E tests for CLI interface functionality."""

import re
import subprocess
import sys

import pytest


class TestCLIInterfaceE2E:
    """E2E tests for CLI interface."""

    @staticmethod
    def strip_ansi_codes(text: str) -> str:
        """Remove ANSI escape codes from text."""
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        return ansi_escape.sub("", text)

    def test_cli_welcome_message(self) -> None:
        """Test that CLI displays welcome message on startup."""
        # Run the CLI command
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "clean_interfaces.main"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Welcome to Clean Interfaces!" in result.stdout
        assert "Type --help for more information" in result.stdout

    def test_cli_help_command(self) -> None:
        """Test that CLI displays help message with --help flag."""
        # Run the CLI command with --help
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "clean_interfaces.main", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        # Strip ANSI codes for proper assertion
        clean_output = self.strip_ansi_codes(result.stdout)
        assert "Usage:" in clean_output
        assert "Options" in clean_output  # Typer uses "Options" without colon
        assert "help" in clean_output  # Check without dashes as they may vary

    def test_cli_with_interface_type_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that CLI respects INTERFACE_TYPE environment variable."""
        # Set environment variable
        monkeypatch.setenv("INTERFACE_TYPE", "cli")

        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "clean_interfaces.main"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Welcome to Clean Interfaces!" in result.stdout
