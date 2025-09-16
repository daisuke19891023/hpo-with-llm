"""Interfaces package for clean interfaces."""

from .base import BaseInterface
from .cli import CLIInterface
from .factory import InterfaceFactory
from .restapi import RestAPIInterface

__all__ = ["BaseInterface", "CLIInterface", "InterfaceFactory", "RestAPIInterface"]
