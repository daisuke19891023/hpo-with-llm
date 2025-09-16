"""Fixtures and utilities for CLI E2E tests using pexpect."""

import os
import sys
from collections.abc import Generator
from typing import Any

import pexpect
import pytest


class CLIRunner:
    """Helper class for running CLI commands with pexpect."""

    def __init__(self, timeout: int = 10) -> None:
        """Initialize the CLI runner.

        Args:
            timeout: Default timeout for expect operations in seconds

        """
        self.timeout = timeout
        self.process: Any | None = None

    def run(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        logfile: Any = None,
    ) -> Any:
        """Run a CLI command.

        Args:
            command: Command to run (defaults to clean-interfaces)
            args: Command line arguments
            env: Environment variables
            logfile: Optional file-like object to log all output

        Returns:
            pexpect.spawn[str]: The spawned process

        """
        if command is None:
            # Use the installed script entry point
            command = sys.executable
            # Add -u flag to disable buffering
            base_args = ["-u", "-m", "clean_interfaces.main"]
        else:
            base_args = []

        if args is None:
            args = []

        full_args = base_args + args

        # Merge environment variables - always use os.environ as base
        # This ensures Python can find modules and system libraries
        cmd_env = {**os.environ, **env} if env else os.environ.copy()

        # Spawn the process
        # Note: pexpect accepts dict[str, str] despite type hints
        # Pass env only if explicitly provided
        spawn_kwargs: dict[str, Any] = {
            "args": full_args,
            "encoding": "utf-8",
            "timeout": self.timeout,
            "echo": False,  # Disable echo to avoid duplicate output
            "dimensions": (24, 80),  # Set terminal dimensions
        }
        if env:
            spawn_kwargs["env"] = cmd_env
        if logfile is not None:
            spawn_kwargs["logfile"] = logfile

        self.process = pexpect.spawn(command, **spawn_kwargs)

        # Increase the delaybeforesend to give the process more time
        self.process.delaybeforesend = 0.1

        # Wait for the process to start and produce initial output
        import time

        time.sleep(0.5)  # Wait for process initialization

        # Ensure the process is running
        if not self.process.isalive():
            raise RuntimeError("Process failed to start")

        return self.process

    def expect(self, pattern: str | list[str], timeout: int | None = None) -> int:
        """Expect a pattern in the output.

        Args:
            pattern: Pattern(s) to expect
            timeout: Timeout override

        Returns:
            int: Index of matched pattern

        """
        if self.process is None:
            msg = "No process is running"
            raise RuntimeError(msg)

        return self.process.expect(pattern, timeout=timeout or self.timeout)

    def send(self, text: str) -> None:
        """Send text to the process.

        Args:
            text: Text to send

        """
        if self.process is None:
            msg = "No process is running"
            raise RuntimeError(msg)

        self.process.send(text)

    def sendline(self, line: str = "") -> None:
        """Send a line to the process.

        Args:
            line: Line to send (newline is added automatically)

        """
        if self.process is None:
            msg = "No process is running"
            raise RuntimeError(msg)

        self.process.sendline(line)

    def close(self) -> None:
        """Close the process."""
        if self.process:
            self.process.close()
            self.process = None

    @property
    def output(self) -> str:
        """Get the current output buffer.

        Returns:
            str: The output buffer content

        """
        if self.process is None:
            return ""
        before = self.process.before
        return str(before) if before is not None else ""

    @property
    def exitstatus(self) -> int | None:
        """Get the exit status of the process.

        Returns:
            int | None: Exit status or None if still running

        """
        if self.process is None:
            return None
        return self.process.exitstatus


@pytest.fixture
def cli_runner() -> Generator[CLIRunner]:
    """Provide a CLI runner for tests.

    Yields:
        CLIRunner: A CLI runner instance

    """
    runner = CLIRunner()
    yield runner
    runner.close()


@pytest.fixture
def clean_env() -> dict[str, str]:
    """Provide a clean environment for CLI tests.

    Returns:
        dict[str, str]: Clean environment variables

    """
    # Start with minimal environment
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        # Critical for pexpect: disable Python output buffering
        "PYTHONUNBUFFERED": "1",
        # Disable any color/formatting that might interfere with tests
        "NO_COLOR": "1",
        "TERM": "dumb",
        # Set log level to ERROR to reduce output
        "LOG_LEVEL": "ERROR",
        # Use plain log format for tests to avoid JSON output
        "LOG_FORMAT": "plain",
        # Disable file logging
        "LOG_FILE_PATH": "",
        # Force UTF-8 encoding
        "PYTHONIOENCODING": "utf-8",
    }

    # Add Python path to ensure our package is importable
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]

    return env
