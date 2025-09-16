"""Tests for interface factory pattern."""

from unittest.mock import patch

import pytest

from clean_interfaces.interfaces.base import BaseInterface
from clean_interfaces.interfaces.factory import InterfaceFactory
from clean_interfaces.types import InterfaceType


class TestInterfaceFactory:
    """Test interface factory functionality."""

    def test_factory_creates_cli_interface(self) -> None:
        """Test that factory creates CLI interface when type is CLI."""
        factory = InterfaceFactory()
        interface = factory.create(InterfaceType.CLI)

        assert interface is not None
        assert isinstance(interface, BaseInterface)
        assert hasattr(interface, "run")
        assert interface.name == "CLI"

    def test_factory_creates_restapi_interface(self) -> None:
        """Test that factory creates RestAPI interface when type is RESTAPI."""
        factory = InterfaceFactory()
        interface = factory.create(InterfaceType.RESTAPI)

        assert interface is not None
        assert isinstance(interface, BaseInterface)
        assert hasattr(interface, "run")
        assert interface.name == "RestAPI"

    def test_factory_respects_interface_type_setting(self) -> None:
        """Test that factory uses interface type from settings."""
        with patch(
            "clean_interfaces.interfaces.factory.get_interface_settings",
        ) as mock_settings:
            mock_settings.return_value.interface_type = InterfaceType.CLI

            factory = InterfaceFactory()
            interface = factory.create_from_settings()

            assert interface is not None
            assert isinstance(interface, BaseInterface)

    def test_factory_raises_for_unknown_interface(self) -> None:
        """Test that factory raises error for unknown interface type."""
        factory = InterfaceFactory()

        with pytest.raises(ValueError, match="Unknown interface type"):
            factory.create("unknown")  # type: ignore[arg-type]

    def test_cli_interface_execution(self) -> None:
        """Test CLI interface can be executed."""
        factory = InterfaceFactory()
        interface = factory.create(InterfaceType.CLI)

        # Mock the typer app to avoid actual execution
        with patch.object(interface, "app") as mock_app:
            interface.run()
            mock_app.assert_called_once()
