"""Base component class for clean interfaces.

This module provides a base component class that all other classes can inherit from
to get common functionality including structured logging capabilities.
"""

from .utils.logger import LoggerProtocol, get_logger


class BaseComponent:
    """Base component providing common functionality.

    All classes inheriting from this base component will have access to:
    - Structured logging through the self.logger attribute
    - Additional functionality can be added in the future
    """

    def __init__(self) -> None:
        """Initialize the base component."""
        # Initialize logger using the class's module and name
        self.logger: LoggerProtocol = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
