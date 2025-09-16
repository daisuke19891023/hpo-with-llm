"""Main entry point for clean_interfaces."""

import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from clean_interfaces.app import run_app


def main(
    dotenv: Annotated[
        Path | None,
        typer.Option(
            "--dotenv",
            "-e",
            help="Path to .env file to load environment variables from",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> NoReturn:
    """Execute the main function with optional dotenv file."""
    # Save original argv
    original_argv = sys.argv

    # Remove main.py's own options from sys.argv so they don't interfere with CLI
    # Keep script name and any args that aren't --dotenv/-e related
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ("--dotenv", "-e"):
            # Skip this option and its value
            i += 2
        elif sys.argv[i].startswith("--dotenv="):
            # Skip this option
            i += 1
        else:
            # Keep this argument for the CLI interface
            new_argv.append(sys.argv[i])
            i += 1

    sys.argv = new_argv

    try:
        run_app(dotenv_path=dotenv)
    finally:
        sys.argv = original_argv

    sys.exit(0)


if __name__ == "__main__":
    typer.run(main)
