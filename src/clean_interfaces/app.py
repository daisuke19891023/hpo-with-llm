"""Application builder for clean interfaces.

This module handles the construction and configuration of the application,
including interface selection, storage configuration, and future database setup.
"""

from pathlib import Path

from dotenv import load_dotenv

from clean_interfaces.interfaces.factory import InterfaceFactory
from clean_interfaces.utils.logger import configure_logging, get_logger
from clean_interfaces.utils.settings import get_interface_settings, get_settings


class Application:
    """Main application class that orchestrates components."""

    def __init__(self, dotenv_path: Path | None = None) -> None:
        """Initialize the application.

        Args:
            dotenv_path: Optional path to .env file to load

        """
        # Load environment variables from .env file if provided
        if dotenv_path:
            load_dotenv(dotenv_path, override=True)
        else:
            # Load from default .env file if it exists
            load_dotenv(override=True)

        # Configure logging first
        settings = get_settings()

        # Configure logging
        configure_logging(
            log_level=settings.log_level,
            log_format=settings.log_format,
            log_file=settings.log_file_path,
        )
        self.logger = get_logger(__name__)

        # Initialize interface
        self.interface_factory = InterfaceFactory()
        self.interface = self.interface_factory.create_from_settings()

        self.logger.info(
            "Application initialized",
            interface=self.interface.name,
            settings=get_interface_settings().model_dump(),
            dotenv_loaded=str(dotenv_path) if dotenv_path else "default",
        )

    def run(self) -> None:
        """Run the application."""
        self.logger.info("Starting application", interface=self.interface.name)

        try:
            self.interface.run()
        except Exception as e:
            self.logger.error("Application error", error=str(e))
            raise
        finally:
            self.logger.info("Application shutting down")


def create_app(dotenv_path: Path | None = None) -> Application:
    """Create an application instance.

    Args:
        dotenv_path: Optional path to .env file to load

    Returns:
        Application: Configured application instance

    """
    return Application(dotenv_path=dotenv_path)


def run_app(dotenv_path: Path | None = None) -> None:
    """Create and run the application.

    Args:
        dotenv_path: Optional path to .env file to load

    """
    app = create_app(dotenv_path=dotenv_path)
    app.run()
