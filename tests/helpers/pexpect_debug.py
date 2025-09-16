"""Pexpect debug helpers for tests."""

import os
import sys
from typing import Any

import pexpect


def run_cli_with_debug(
    command: str,
    env: dict[str, str] | None = None,
    timeout: int = 5,
    debug: bool | None = None,
) -> tuple[str, int]:
    """Run CLI command with optional debug output.

    Args:
        command: Command to run
        env: Environment variables
        timeout: Timeout in seconds
        debug: Enable debug output (defaults to PYTEST_DEBUG env var)

    Returns:
        Tuple of (output, exitstatus)

    """
    # Check debug mode from environment if not explicitly set
    if debug is None:
        debug = os.environ.get("PYTEST_DEBUG", "").lower() in ("1", "true", "yes")

    if debug:
        print(f"\n[DEBUG] Running command: {command}")
        print(f"[DEBUG] Environment: {env}")
        print(f"[DEBUG] Timeout: {timeout}s")
        print("-" * 80)

    try:
        # Use pexpect.run for simple command execution
        result = pexpect.run(  # type: ignore[attr-defined]
            command,
            env=env,  # type: ignore[arg-type]
            withexitstatus=True,
            timeout=timeout,
            encoding="utf-8",
        )

        # Handle return value
        if isinstance(result, tuple):  # type: ignore[arg-type]
            output, exitstatus = result  # type: ignore[misc]
            output = (
                str(output) if output is not None else ""  # type: ignore[arg-type]
            )  # Ensure output is string
        else:
            output = str(result)
            exitstatus = 0

        if debug:
            print(f"[DEBUG] Exit status: {exitstatus}")
            print(f"[DEBUG] Output length: {len(output)} chars")
            print("[DEBUG] Output:")
            print("-" * 80)
            print(output)
            print("-" * 80)

        return output, exitstatus

    except pexpect.TIMEOUT as e:
        if debug:
            print(f"[DEBUG] TIMEOUT after {timeout}s")
            print(f"[DEBUG] Exception: {e}")
            print(f"[DEBUG] Buffer before timeout: {getattr(e, 'buffer', 'N/A')}")
        raise
    except Exception as e:
        if debug:
            print(f"[DEBUG] Exception: {type(e).__name__}: {e}")
        raise


def spawn_cli_with_debug(
    command: str,
    env: dict[str, str] | None = None,
    timeout: int = 30,
    debug: bool | None = None,
) -> Any:
    """Spawn CLI process with optional debug output for interactive testing.

    Args:
        command: Command to run
        env: Environment variables
        timeout: Timeout in seconds
        debug: Enable debug output (defaults to PYTEST_DEBUG env var)

    Returns:
        pexpect.spawn object

    """
    # Check debug mode from environment if not explicitly set
    if debug is None:
        debug = os.environ.get("PYTEST_DEBUG", "").lower() in ("1", "true", "yes")

    if debug:
        print(f"\n[DEBUG] Spawning: {command}")
        print(f"[DEBUG] Environment: {env}")
        print(f"[DEBUG] Timeout: {timeout}s")
        print("-" * 80)

    # Create spawn object
    child: Any = pexpect.spawn(  # type: ignore[no-untyped-call]
        command,
        env=env,  # type: ignore[arg-type]
        timeout=timeout,
        encoding="utf-8",
    )

    # Enable logging if debug mode
    if debug:
        child.logfile_read = sys.stdout

    return child
