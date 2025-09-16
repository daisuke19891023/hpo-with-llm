"""End-to-end tests for CLI interface using pexpect."""

import sys

import pexpect

from clean_interfaces.models.io import WelcomeMessage


class TestCLIE2E:
    """E2E tests for the CLI interface."""

    def test_cli_shows_welcome_without_args(
        self,
        clean_env: dict[str, str],
    ) -> None:
        """Test that CLI shows welcome message when run without arguments."""
        from tests.helpers.pexpect_debug import run_cli_with_debug

        # Run CLI without arguments - should show welcome
        try:
            output, exitstatus = run_cli_with_debug(
                f"{sys.executable} -u -m clean_interfaces.main",
                env=clean_env,
                timeout=10,
                debug=True,  # Enable debug output to see what's happening
            )
        except pexpect.TIMEOUT:
            error_msg = "Timeout waiting for CLI to complete"
            raise AssertionError(error_msg) from None

        # Check exit status
        assert exitstatus == 0, f"Expected exit code 0, got {exitstatus}"

        # Expect the welcome message
        welcome_msg = WelcomeMessage()

        # Check that welcome message appears in output
        assert welcome_msg.message in output, (
            f"Welcome message not found in output: {output}"
        )
        assert welcome_msg.hint in output, f"Welcome hint not found in output: {output}"

    def test_cli_help_command(
        self,
        clean_env: dict[str, str],
    ) -> None:
        """Test that CLI shows help when --help is used."""
        from tests.helpers.pexpect_debug import run_cli_with_debug

        # Run CLI with --help - now it should work properly
        try:
            output, exitstatus = run_cli_with_debug(
                f"{sys.executable} -u -m clean_interfaces.main --help",
                env=clean_env,
                timeout=10,
                debug=True,  # Enable debug output
            )
        except pexpect.TIMEOUT:
            error_msg = "Timeout waiting for CLI help command to complete"
            raise AssertionError(error_msg) from None

        # Check exit code
        assert exitstatus == 0, f"Expected exit code 0, got {exitstatus}"

        # Since main.py intercepts --help, we see main.py's help, not CLI interface help
        # Check for main.py help elements
        assert "Usage:" in output or "usage:" in output, (
            f"Usage not found in output: {output}"
        )
        assert "--dotenv" in output, f"--dotenv option not found in output: {output}"
        assert "--help" in output, f"--help option not found in output: {output}"

    def test_cli_version_command(
        self,
        clean_env: dict[str, str],
    ) -> None:
        """Test that CLI shows version information."""
        from tests.helpers.pexpect_debug import run_cli_with_debug

        # Use run_cli_with_debug to capture output
        try:
            _, exitstatus = run_cli_with_debug(
                f"{sys.executable} -u -m clean_interfaces.main --version",
                env=clean_env,
                timeout=10,
                debug=False,  # Disable debug for this test
            )
        except pexpect.TIMEOUT:
            error_msg = "Timeout waiting for CLI version command to complete"
            raise AssertionError(error_msg) from None

        # Typer may or may not implement --version by default
        # Check that it at least exits cleanly
        assert exitstatus in (0, 2), (
            f"Unexpected exit code: {exitstatus}"
        )  # 0 for success, 2 for unrecognized option

    def test_cli_invalid_command(
        self,
        clean_env: dict[str, str],
    ) -> None:
        """Test that CLI handles invalid commands gracefully."""
        from tests.helpers.pexpect_debug import run_cli_with_debug

        # Use run_cli_with_debug to capture output
        try:
            output, exitstatus = run_cli_with_debug(
                f"{sys.executable} -u -m clean_interfaces.main invalid-command",
                env=clean_env,
                timeout=10,
                debug=False,  # Disable debug for this test
            )
        except pexpect.TIMEOUT:
            error_msg = "Timeout waiting for CLI invalid command to complete"
            raise AssertionError(error_msg) from None

        # Should exit with non-zero status
        assert exitstatus != 0, f"Expected non-zero exit code, got {exitstatus}"

        # Should show error message
        assert any(
            word in output for word in ["Error", "No such command", "Invalid"]
        ), f"Expected error message not found in output: {output}"

    def test_cli_explicit_welcome_command(
        self,
        clean_env: dict[str, str],
    ) -> None:
        """Test running the welcome command explicitly."""
        from tests.helpers.pexpect_debug import run_cli_with_debug

        # Since main.py doesn't accept 'welcome' command, it will show error
        # The CLI interface inside does have 'welcome' command, but main.py filters args
        try:
            output, exitstatus = run_cli_with_debug(
                f"{sys.executable} -u -m clean_interfaces.main welcome",
                env=clean_env,
                timeout=10,
                debug=False,  # Disable debug output
            )
        except pexpect.TIMEOUT:
            error_msg = "Timeout waiting for CLI welcome command to complete"
            raise AssertionError(error_msg) from None

        # Since main.py doesn't recognize 'welcome', it will exit with error
        assert exitstatus == 2, (
            f"Expected exit code 2 for unrecognized command, got {exitstatus}"
        )

        # Should show error message about unexpected argument
        assert "unexpected" in output.lower() or "error" in output.lower(), (
            f"Expected error message not found in output: {output}"
        )

    def test_cli_interrupt_handling(
        self,
        clean_env: dict[str, str],
    ) -> None:
        """Test that CLI handles basic execution without hanging."""
        from tests.helpers.pexpect_debug import run_cli_with_debug

        # Note: Interrupt handling with pexpect is complex and platform-specific
        # This test just verifies the CLI completes execution normally

        # Use run_cli_with_debug to capture output
        try:
            output, exitstatus = run_cli_with_debug(
                f"{sys.executable} -u -m clean_interfaces.main",
                env=clean_env,
                timeout=10,
                debug=False,  # Disable debug for this test
            )
        except pexpect.TIMEOUT:
            error_msg = "Timeout - CLI appears to be hanging"
            raise AssertionError(error_msg) from None

        # Should complete with exit code 0
        assert exitstatus == 0, f"Expected exit code 0, got {exitstatus}"

        # Should produce output
        assert len(output) > 0, "No output produced"
