"""Unit tests for the BaseComponent class."""

from unittest.mock import Mock, patch

from clean_interfaces.base import BaseComponent
from clean_interfaces.utils.logger import LoggerProtocol


class TestBaseComponent:
    """Test cases for BaseComponent class."""

    @patch("clean_interfaces.base.get_logger")
    def test_logger_initialization(self, mock_get_logger: Mock) -> None:
        """Test that logger is properly initialized with class name."""
        # Arrange
        mock_logger = Mock(spec=LoggerProtocol)
        mock_get_logger.return_value = mock_logger

        # Act
        instance = BaseComponent()

        # Assert
        mock_get_logger.assert_called_once_with(
            "clean_interfaces.base.BaseComponent",
        )
        assert instance.logger is mock_logger

    @patch("clean_interfaces.base.get_logger")
    def test_logger_direct_access(self, mock_get_logger: Mock) -> None:
        """Test direct access to logger methods."""
        # Arrange
        mock_logger = Mock(spec=LoggerProtocol)
        mock_get_logger.return_value = mock_logger
        instance = BaseComponent()

        # Act
        instance.logger.debug("Debug message", user="test")
        instance.logger.info("Info message", status="success")
        instance.logger.warning("Warning message", threshold=90)
        instance.logger.error("Error message", error_code="E001")
        instance.logger.critical("Critical message", system="database")

        # Assert
        mock_logger.debug.assert_called_once_with("Debug message", user="test")
        mock_logger.info.assert_called_once_with("Info message", status="success")
        mock_logger.warning.assert_called_once_with("Warning message", threshold=90)
        mock_logger.error.assert_called_once_with("Error message", error_code="E001")
        mock_logger.critical.assert_called_once_with(
            "Critical message",
            system="database",
        )

    @patch("clean_interfaces.base.get_logger")
    def test_inheritance(self, mock_get_logger: Mock) -> None:
        """Test that classes inheriting from BaseComponent get proper logger name."""
        # Arrange
        mock_logger = Mock(spec=LoggerProtocol)
        mock_get_logger.return_value = mock_logger

        class CustomService(BaseComponent):
            """Test service class."""

        # Act
        instance = CustomService()

        # Assert
        mock_get_logger.assert_called_once_with(
            "tests.unit.clean_interfaces.test_base.CustomService",
        )
        assert instance.logger is mock_logger

    @patch("clean_interfaces.base.get_logger")
    def test_logger_bind(self, mock_get_logger: Mock) -> None:
        """Test logger bind method can be used."""
        # Arrange
        mock_logger = Mock(spec=LoggerProtocol)
        mock_bound_logger = Mock(spec=LoggerProtocol)
        mock_logger.bind.return_value = mock_bound_logger
        mock_get_logger.return_value = mock_logger
        instance = BaseComponent()

        # Act
        bound_logger = instance.logger.bind(request_id="123")

        # Assert
        mock_logger.bind.assert_called_once_with(request_id="123")
        assert bound_logger is mock_bound_logger
