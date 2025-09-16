"""Tests for type definitions."""

from enum import Enum

from clean_interfaces.types import InterfaceType


class TestInterfaceType:
    """Test InterfaceType enum."""

    def test_interface_type_is_enum(self) -> None:
        """Test that InterfaceType is an Enum."""
        assert issubclass(InterfaceType, Enum)

    def test_interface_type_has_cli(self) -> None:
        """Test that InterfaceType has CLI option."""
        assert hasattr(InterfaceType, "CLI")
        assert InterfaceType.CLI.value == "cli"

    def test_interface_type_has_restapi(self) -> None:
        """Test that InterfaceType has RESTAPI option."""
        assert hasattr(InterfaceType, "RESTAPI")
        assert InterfaceType.RESTAPI.value == "restapi"

    def test_interface_type_values(self) -> None:
        """Test InterfaceType values are correct."""
        assert InterfaceType.CLI.value == "cli"
        assert InterfaceType.RESTAPI.value == "restapi"

    def test_interface_type_from_string(self) -> None:
        """Test creating InterfaceType from string."""
        cli_interface = InterfaceType("cli")
        assert cli_interface == InterfaceType.CLI

        restapi_interface = InterfaceType("restapi")
        assert restapi_interface == InterfaceType.RESTAPI
