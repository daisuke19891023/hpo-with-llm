# Installation

This guide covers the installation of Clean Interfaces.

## Requirements

-   Python 3.13 or higher
-   uv (Python package manager)

## Installing uv

If you don't have uv installed, you can install it using:

=== "Linux/macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

## Installation Steps

### From Source

1. **Clone the repository**

    ```bash
    git clone https://github.com/your-username/clean-interfaces.git
    cd clean-interfaces
    ```

2. **Create virtual environment and install dependencies**

    ```bash
    uv sync
    ```

3. **Install with optional dependencies** (if needed)

    ```bash
    # Install with documentation tools
    uv sync --extra docs

    # Install with development tools
    uv sync --extra dev

    # Install all extras
    uv sync --all-extras
    ```

### From PyPI

!!! note "Coming Soon"
PyPI package installation will be available once the package is published.

```bash
# Basic installation
uv pip install clean-interfaces

# With extras
uv pip install "clean-interfaces[docs]"
```

## Configuration

1. **Copy the example configuration**

    ```bash
    cp .env.example .env
    ```

2. **Edit `.env` with your configuration**

    ```ini
    # Example configuration
    INTERFACE_TYPE=cli
    LOG_LEVEL=INFO
    LOG_FORMAT=console
    ```

## Verification

Verify the installation by running:

```bash
# Show help
uv run python -m clean_interfaces.main --help

# Run the application
uv run python -m clean_interfaces.main
```

You should see output indicating that the application is running successfully.

## Troubleshooting

### Common Issues

#### ImportError

If you encounter import errors, ensure you're running the command with `uv run`:

```bash
# Wrong
python -m clean_interfaces.main

# Correct
uv run python -m clean_interfaces.main
```

#### Environment Variables Not Loading

Make sure your `.env` file is in the project root directory, or specify it explicitly:

```bash
uv run python -m clean_interfaces.main --dotenv /path/to/your/.env
```

## Next Steps

-   Read the [Quick Start](quickstart.md) guide
-   Learn about [Configuration](configuration.md)
-   Explore the [CLI Interface](guides/cli.md) or [REST API Interface](guides/restapi.md)
