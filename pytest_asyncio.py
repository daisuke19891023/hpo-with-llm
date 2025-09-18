"""Minimal pytest plugin providing asyncio test support."""

from __future__ import annotations

import asyncio

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register asyncio marker for environments without pytest-asyncio."""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test to run with an asyncio event loop",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute async test functions using a fresh event loop."""
    test_func = pyfuncitem.obj
    if asyncio.iscoroutinefunction(test_func):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(test_func(**pyfuncitem.funcargs))
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        return True
    return None
