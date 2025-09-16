# Clean Interfaces

Welcome to **Clean Interfaces** - a flexible Python application framework with multiple interface types and comprehensive logging support.

## Features

-   🚀 **Multiple Interface Types**: Support for CLI and REST API interfaces
-   ⚙️ **Flexible Configuration**: Environment-based configuration with `.env` file support
-   📝 **Structured Logging**: Advanced logging with OpenTelemetry integration
-   🐍 **Modern Python**: Built with Python 3.13+ and modern tooling
-   ✅ **Comprehensive Testing**: Unit, API, and E2E test coverage
-   🔍 **Type Safety**: Full type hints with strict Pyright checking
-   🎨 **Code Quality**: Automated linting and formatting with Ruff
-   📦 **Dependency Management**: Managed with uv for fast, reliable builds

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/clean-interfaces.git
cd clean-interfaces

# Install dependencies
uv sync

# Copy environment configuration
cp .env.example .env

# Run the application (CLI mode)
uv run python -m clean_interfaces.main

# Run with custom environment file
uv run python -m clean_interfaces.main --dotenv prod.env

# Run REST API mode
INTERFACE_TYPE=restapi uv run python -m clean_interfaces.main
```

## Project Overview

Clean Interfaces provides a clean, extensible architecture for building Python applications with multiple interface types. Whether you need a command-line tool, a REST API, or both, Clean Interfaces has you covered.

### Key Components

-   **Interface System**: Factory pattern for creating different interface types
-   **Configuration Management**: Pydantic-based settings with environment variable support
-   **Logging System**: Structured logging with multiple output formats and OpenTelemetry integration
-   **Type Safety**: Full type annotations with strict checking

## Documentation Structure

-   **[Getting Started](installation.md)**: Installation and quick start guides
-   **[User Guide](guides/cli.md)**: Detailed guides for using the framework
-   **[API Reference](api/overview.md)**: Complete API documentation
-   **[Development](development/contributing.md)**: Contributing and development guides

## Development Commands

```bash
# Run tests
nox -s test

# Run linting
nox -s lint

# Format code
nox -s format_code

# Type checking
nox -s typing

# Run all CI checks
nox -s ci

# Build documentation
nox -s docs
```

## License

This project is licensed under the MIT License.
