"""Factory pattern implementation for creating interfaces."""

from clean_interfaces.types import InterfaceType
from clean_interfaces.utils.settings import get_interface_settings

from .base import BaseInterface
from .cli import CLIInterface
from .mcp import MCPInterface
from .restapi import RestAPIInterface


class InterfaceFactory:
    """Factory for creating interface instances."""

    def create(self, interface_type: InterfaceType) -> BaseInterface:
        """Create an interface instance based on the type.

        Args:
            interface_type: The type of interface to create

        Returns:
            BaseInterface: The created interface instance

        Raises:
            ValueError: If the interface type is unknown

        """
        if interface_type == InterfaceType.CLI:
            return CLIInterface()
        if interface_type == InterfaceType.RESTAPI:
            return RestAPIInterface()
        if interface_type == InterfaceType.MCP:
            return MCPInterface()

        msg = f"Unknown interface type: {interface_type}"
        raise ValueError(msg)

    def create_from_settings(self) -> BaseInterface:
        """Create an interface instance from settings.

        Returns:
            BaseInterface: The created interface instance

        """
        settings = get_interface_settings()
        interface_type = InterfaceType(settings.interface_type)
        return self.create(interface_type)
