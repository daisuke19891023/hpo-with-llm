"""Base interface class for all interfaces."""

from abc import ABC, abstractmethod

from clean_interfaces.base import BaseComponent


class BaseInterface(ABC, BaseComponent):
    """Abstract base class for all interfaces."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the interface.

        Returns:
            str: The interface name

        """

    @abstractmethod
    def run(self) -> None:
        """Run the interface.

        This method should contain the main logic for starting
        and running the interface.
        """
