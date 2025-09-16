"""Tests for I/O models."""

from clean_interfaces.models.io import WelcomeMessage


class TestWelcomeMessage:
    """Test WelcomeMessage model."""

    def test_welcome_message_model(self) -> None:
        """Test WelcomeMessage model validation."""
        msg = WelcomeMessage(message="Test", hint="Test hint")
        assert msg.message == "Test"
        assert msg.hint == "Test hint"

    def test_welcome_message_default_values(self) -> None:
        """Test WelcomeMessage default values."""
        # Test default message and hint
        msg_all_defaults = WelcomeMessage()
        assert msg_all_defaults.message == "Welcome to Clean Interfaces!"
        assert msg_all_defaults.hint == "Type --help for more information"

        # Test default hint with custom message
        msg_custom_message = WelcomeMessage(message="Test")
        assert msg_custom_message.message == "Test"
        assert msg_custom_message.hint == "Type --help for more information"
