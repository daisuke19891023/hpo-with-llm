"""Tests for application module."""

from unittest.mock import MagicMock, patch

import pytest

from clean_interfaces.app import Application, create_app, run_app


class TestApplication:
    """Test Application class."""

    @patch("clean_interfaces.app.load_dotenv")
    @patch("clean_interfaces.app.configure_logging")
    @patch("clean_interfaces.app.get_logger")
    @patch("clean_interfaces.app.get_settings")
    @patch("clean_interfaces.app.get_interface_settings")
    @patch("clean_interfaces.app.InterfaceFactory")
    def test_application_initialization(
        self,
        mock_factory_class: MagicMock,
        mock_get_interface_settings: MagicMock,
        mock_get_settings: MagicMock,
        mock_get_logger: MagicMock,
        mock_configure_logging: MagicMock,
        mock_load_dotenv: MagicMock,
    ) -> None:
        """Test Application initialization."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.log_level = "INFO"
        mock_settings.log_format = "json"
        mock_settings.log_file_path = None
        mock_get_settings.return_value = mock_settings

        mock_interface_settings = MagicMock()
        mock_interface_settings.model_dump.return_value = {"interface_type": "cli"}
        mock_get_interface_settings.return_value = mock_interface_settings

        mock_interface = MagicMock()
        mock_interface.name = "CLI"
        mock_factory = MagicMock()
        mock_factory.create_from_settings.return_value = mock_interface
        mock_factory_class.return_value = mock_factory

        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create application
        app = Application()

        # Verify load_dotenv was called for default .env
        mock_load_dotenv.assert_called_once_with(override=True)

        # Verify logging was configured
        mock_configure_logging.assert_called_once_with(
            log_level="INFO",
            log_format="json",
            log_file=None,
        )

        # Verify interface was created
        mock_factory.create_from_settings.assert_called_once()
        assert app.interface == mock_interface

    @patch("clean_interfaces.app.load_dotenv")
    @patch("clean_interfaces.app.configure_logging")
    @patch("clean_interfaces.app.get_logger")
    @patch("clean_interfaces.app.get_settings")
    @patch("clean_interfaces.app.get_interface_settings")
    @patch("clean_interfaces.app.InterfaceFactory")
    def test_application_initialization_with_dotenv(
        self,
        mock_factory_class: MagicMock,
        mock_get_interface_settings: MagicMock,
        mock_get_settings: MagicMock,
        mock_get_logger: MagicMock,
        mock_configure_logging: MagicMock,
        mock_load_dotenv: MagicMock,
    ) -> None:
        """Test Application initialization with dotenv path."""
        from pathlib import Path

        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.log_level = "DEBUG"
        mock_settings.log_format = "console"
        mock_settings.log_file_path = None
        mock_get_settings.return_value = mock_settings

        mock_interface_settings = MagicMock()
        mock_interface_settings.model_dump.return_value = {"interface_type": "cli"}
        mock_get_interface_settings.return_value = mock_interface_settings

        mock_interface = MagicMock()
        mock_interface.name = "CLI"
        mock_factory = MagicMock()
        mock_factory.create_from_settings.return_value = mock_interface
        mock_factory_class.return_value = mock_factory

        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create application with dotenv path
        dotenv_path = Path("test.env")
        app = Application(dotenv_path=dotenv_path)

        # Verify load_dotenv was called with specific path
        mock_load_dotenv.assert_called_once_with(dotenv_path, override=True)

        # Verify logging was configured
        mock_configure_logging.assert_called_once_with(
            log_level="DEBUG",
            log_format="console",
            log_file=None,
        )

        # Verify interface was created
        mock_factory.create_from_settings.assert_called_once()
        assert app.interface == mock_interface

    @patch("clean_interfaces.app.load_dotenv")
    @patch("clean_interfaces.app.configure_logging")
    @patch("clean_interfaces.app.get_logger")
    @patch("clean_interfaces.app.get_settings")
    @patch("clean_interfaces.app.get_interface_settings")
    @patch("clean_interfaces.app.InterfaceFactory")
    def test_application_run(
        self,
        mock_factory_class: MagicMock,
        mock_get_interface_settings: MagicMock,
        mock_get_settings: MagicMock,
        mock_get_logger: MagicMock,
        mock_configure_logging: MagicMock,  # noqa: ARG002
        mock_load_dotenv: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test Application run method."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.log_level = "INFO"
        mock_settings.log_format = "json"
        mock_settings.log_file_path = None
        mock_get_settings.return_value = mock_settings

        mock_interface_settings = MagicMock()
        mock_interface_settings.model_dump.return_value = {"interface_type": "cli"}
        mock_get_interface_settings.return_value = mock_interface_settings

        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_interface = MagicMock()
        mock_interface.name = "CLI"
        mock_factory = MagicMock()
        mock_factory.create_from_settings.return_value = mock_interface
        mock_factory_class.return_value = mock_factory

        # Create and run application
        app = Application()
        app.run()

        # Verify interface was run
        mock_interface.run.assert_called_once()

    @patch("clean_interfaces.app.load_dotenv")
    @patch("clean_interfaces.app.configure_logging")
    @patch("clean_interfaces.app.get_logger")
    @patch("clean_interfaces.app.get_settings")
    @patch("clean_interfaces.app.get_interface_settings")
    @patch("clean_interfaces.app.InterfaceFactory")
    def test_application_run_with_exception(
        self,
        mock_factory_class: MagicMock,
        mock_get_interface_settings: MagicMock,
        mock_get_settings: MagicMock,
        mock_get_logger: MagicMock,
        mock_configure_logging: MagicMock,  # noqa: ARG002
        mock_load_dotenv: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test Application run method with exception."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.log_level = "INFO"
        mock_settings.log_format = "json"
        mock_settings.log_file_path = None
        mock_get_settings.return_value = mock_settings

        mock_interface_settings = MagicMock()
        mock_interface_settings.model_dump.return_value = {"interface_type": "cli"}
        mock_get_interface_settings.return_value = mock_interface_settings

        # Mock logger to prevent actual logging
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_interface = MagicMock()
        mock_interface.name = "CLI"
        mock_interface.run.side_effect = RuntimeError("Test error")
        mock_factory = MagicMock()
        mock_factory.create_from_settings.return_value = mock_interface
        mock_factory_class.return_value = mock_factory

        # Create application and verify exception is raised
        app = Application()
        with pytest.raises(RuntimeError, match="Test error"):
            app.run()

        # Verify interface was attempted to run
        mock_interface.run.assert_called_once()

        # Verify logger was called
        mock_logger.error.assert_called_once_with(
            "Application error",
            error="Test error",
        )
        mock_logger.info.assert_any_call("Application shutting down")


def test_create_app() -> None:
    """Test create_app factory function."""
    with patch("clean_interfaces.app.Application") as mock_app_class:
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app

        result = create_app()

        assert result == mock_app
        mock_app_class.assert_called_once_with(dotenv_path=None)


def test_create_app_with_dotenv() -> None:
    """Test create_app factory function with dotenv path."""
    from pathlib import Path

    with patch("clean_interfaces.app.Application") as mock_app_class:
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        dotenv_path = Path("test.env")

        result = create_app(dotenv_path=dotenv_path)

        assert result == mock_app
        mock_app_class.assert_called_once_with(dotenv_path=dotenv_path)


def test_run_app() -> None:
    """Test run_app function."""
    with patch("clean_interfaces.app.create_app") as mock_create_app:
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        run_app()

        mock_create_app.assert_called_once_with(dotenv_path=None)
        mock_app.run.assert_called_once()


def test_run_app_with_dotenv() -> None:
    """Test run_app function with dotenv path."""
    from pathlib import Path

    with patch("clean_interfaces.app.create_app") as mock_create_app:
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        dotenv_path = Path("test.env")

        run_app(dotenv_path=dotenv_path)

        mock_create_app.assert_called_once_with(dotenv_path=dotenv_path)
        mock_app.run.assert_called_once()
