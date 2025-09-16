"""Tests for base interface class."""

from abc import ABC

import pytest

from clean_interfaces.interfaces.base import BaseInterface


class TestBaseInterface:
    """Test base interface functionality."""

    def test_base_interface_is_abstract(self) -> None:
        """Test that BaseInterface is abstract."""
        assert issubclass(BaseInterface, ABC)

    def test_base_interface_has_run_method(self) -> None:
        """Test that BaseInterface has abstract run method."""
        assert hasattr(BaseInterface, "run")
        # Check that it's abstract by verifying it's in __abstractmethods__
        assert "run" in BaseInterface.__abstractmethods__

    def test_base_interface_has_name_property(self) -> None:
        """Test that BaseInterface has abstract name property."""
        assert hasattr(BaseInterface, "name")
        # Check that it's abstract by verifying it's in __abstractmethods__
        assert "name" in BaseInterface.__abstractmethods__

    def test_cannot_instantiate_base_interface(self) -> None:
        """Test that BaseInterface cannot be instantiated directly."""
        # This should raise TypeError because of abstract methods
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseInterface()  # type: ignore[abstract]
