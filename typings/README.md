# Type Stubs

This directory contains type stubs for third-party packages that don't provide their own type information or are not available in the Pyright environment when running inside nox sessions.

## Structure

- `nox/` - Type stubs for nox task runner
- `pytest/` - Type stubs for pytest testing framework
- `fastapi/` - Type stubs for FastAPI web framework
- `httpx/` - Type stubs for httpx HTTP client
- `uvicorn/` - Type stubs for uvicorn ASGI server

## Purpose

These stubs are necessary because:
1. When Pyright runs inside nox virtual environments, it cannot find the nox package itself
2. Test dependencies like pytest are only installed in test environments
3. This allows us to maintain strict type checking without using `# type: ignore` comments

## Maintenance

When updating dependencies or adding new ones that require type stubs:
1. Create a new directory for the package
2. Add `__init__.pyi` with the necessary type definitions
3. Only include the types actually used in the codebase