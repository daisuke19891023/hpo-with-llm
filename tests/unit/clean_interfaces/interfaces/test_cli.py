"""Tests for CLI interface implementation."""

import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import typer

from typer.testing import CliRunner

from clean_interfaces.interfaces.base import BaseInterface
from clean_interfaces.interfaces.cli import CLIInterface


class TestCLIInterface:
    """Test CLI interface functionality."""

    def test_cli_interface_inherits_base(self) -> None:
        """Test that CLIInterface inherits from BaseInterface."""
        assert issubclass(CLIInterface, BaseInterface)

    def test_cli_interface_has_name(self) -> None:
        """Test that CLIInterface has correct name."""
        cli = CLIInterface()
        assert cli.name == "CLI"

    def test_cli_interface_has_typer_app(self) -> None:
        """Test that CLIInterface has Typer app."""
        cli = CLIInterface()
        assert hasattr(cli, "app")
        assert isinstance(cli.app, typer.Typer)

    def test_cli_welcome_command(self) -> None:
        """Test CLI welcome command functionality."""
        cli = CLIInterface()

        # Mock the console output
        with patch("clean_interfaces.interfaces.cli.console") as mock_console:
            cli.welcome()

            # Check that welcome message was printed (should be called twice)
            assert mock_console.print.call_count == 2
            # First call is the welcome message
            first_call = mock_console.print.call_args_list[0][0]
            assert "Welcome to Clean Interfaces!" in str(first_call)
            # Second call is the hint
            second_call = mock_console.print.call_args_list[1][0]
            assert "Type --help for more information" in str(second_call)

    def test_cli_run_method(self) -> None:
        """Test CLI run method executes typer app."""
        cli = CLIInterface()

        # Mock the typer app
        cli.app = MagicMock()

        cli.run()

        cli.app.assert_called_once()

    def test_cli_hpo_command_executes(self) -> None:
        """Ensure the run-hpo command executes without error."""
        cli = CLIInterface()
        runner = CliRunner()

        with runner.isolated_filesystem():
            config_path = Path("search_space.yaml")
            config_path.write_text(
                textwrap.dedent(
                    """
                    parameters:
                      - name: temperature
                        type: float
                        lower: 0.0
                        upper: 1.0
                        location:
                          kind: environment
                          variable: CLI_TEMPERATURE
                      - name: max_output_tokens
                        type: int
                        lower: 64
                        upper: 128
                        step: 64
                        location:
                          kind: cli_argument
                          flag: --max-output-tokens
                    """,
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            result = runner.invoke(
                cli.app,
                [
                    "run-hpo",
                    "--task",
                    "Optimize prompts",
                    "--max-trials",
                    "2",
                    "--search-space-config",
                    str(config_path),
                ],
            )

        assert result.exit_code == 0
        cleaned_output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        assert "Executed 2 trials" in cleaned_output

    def test_cli_reflect_command_emits_summary(self) -> None:
        """Ensure the reflect-hpo command runs and reports a summary."""
        cli = CLIInterface()
        runner = CliRunner()

        with runner.isolated_filesystem():
            config_path = Path("search_space.yaml")
            config_path.write_text(
                textwrap.dedent(
                    """
                    parameters:
                      - name: temperature
                        type: float
                        lower: 0.0
                        upper: 1.0
                      - name: max_output_tokens
                        type: int
                        lower: 128
                        upper: 256
                        step: 64
                      - name: use_tooling
                        type: bool
                    """,
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            result = runner.invoke(
                cli.app,
                [
                    "reflect-hpo",
                    "--task",
                    "Optimise prompts",
                    "--max-trials",
                    "1",
                    "--mode",
                    "baseline",
                    "--search-space-config",
                    str(config_path),
                ],
            )

        assert result.exit_code == 0
        cleaned_output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        assert "Reflection summary" in cleaned_output
        assert "Suggested hyperparameters" in cleaned_output
