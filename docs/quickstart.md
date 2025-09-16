# Quick Start

Get up and running with Clean Interfaces in minutes!

## Prerequisites

Before you begin, ensure you have:

-   ✅ Python 3.13 or higher installed
-   ✅ uv package manager installed
-   ✅ Git (for cloning the repository)

## 5-Minute Setup

### 1. Get the Code

```bash
git clone https://github.com/your-username/clean-interfaces.git
cd clean-interfaces
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure the Application

```bash
cp .env.example .env
```

### 4. Run Your First Command

=== "CLI Interface"

    ```bash
    uv run python -m clean_interfaces.main
    ```

=== "REST API Interface"

    ```bash
    INTERFACE_TYPE=restapi uv run python -m clean_interfaces.main
    ```

## Basic Usage Examples

### Using the CLI Interface

The CLI interface provides an interactive command-line experience:

```bash
# Run with default settings
uv run python -m clean_interfaces.main

# Run with debug logging
LOG_LEVEL=DEBUG uv run python -m clean_interfaces.main

# Run with custom environment file
uv run python -m clean_interfaces.main --dotenv dev.env
```

### Using the REST API Interface

The REST API interface starts a FastAPI server:

```bash
# Start the API server
INTERFACE_TYPE=restapi uv run python -m clean_interfaces.main

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Testing the API

Once the REST API is running, you can test it:

```bash
# Get a welcome message
curl http://localhost:8000/

# Check the health endpoint
curl http://localhost:8000/health
```

## Configuration Examples

### Development Setup

Create a `dev.env` file:

```ini
INTERFACE_TYPE=cli
LOG_LEVEL=DEBUG
LOG_FORMAT=console
```

Run with development settings:

```bash
uv run python -m clean_interfaces.main --dotenv dev.env
```

### Production Setup

Create a `prod.env` file:

```ini
INTERFACE_TYPE=restapi
LOG_LEVEL=WARNING
LOG_FORMAT=json
LOG_FILE_PATH=/var/log/app.log
OTEL_LOGS_EXPORT_MODE=otlp
OTEL_ENDPOINT=http://collector:4317
```

Run with production settings:

```bash
uv run python -m clean_interfaces.main --dotenv prod.env
```

## Understanding the Output

### Console Logging (Development)

```
2025-07-20 10:30:45 [INFO] clean_interfaces.app: Application initialized
2025-07-20 10:30:45 [INFO] clean_interfaces.app: Starting CLI interface
2025-07-20 10:30:45 [DEBUG] clean_interfaces.interfaces.cli: CLI ready
```

### JSON Logging (Production)

```json
{
    "timestamp": "2025-07-20T10:30:45.123Z",
    "level": "info",
    "logger": "clean_interfaces.app",
    "message": "Application initialized",
    "interface": "restapi"
}
```

## Common Tasks

### Changing Log Level

```bash
# Debug level for detailed output
LOG_LEVEL=DEBUG uv run python -m clean_interfaces.main

# Error level for production
LOG_LEVEL=ERROR uv run python -m clean_interfaces.main
```

### Switching Interfaces

```bash
# CLI Interface (default)
INTERFACE_TYPE=cli uv run python -m clean_interfaces.main

# REST API Interface
INTERFACE_TYPE=restapi uv run python -m clean_interfaces.main
```

### Using Different Configurations

```bash
# Development
uv run python -m clean_interfaces.main --dotenv dev.env

# Testing
uv run python -m clean_interfaces.main --dotenv test.env

# Production
uv run python -m clean_interfaces.main --dotenv prod.env
```

## What's Next?

Now that you have Clean Interfaces running:

1. **Explore the Interfaces**

    - [CLI Interface Guide](guides/cli.md)
    - [REST API Interface Guide](guides/restapi.md)

2. **Learn About Configuration**

    - [Configuration Guide](configuration.md)
    - [Environment Variables](guides/environment.md)

3. **Understand Logging**

    - [Logging Guide](guides/logging.md)

4. **Start Developing**
    - [Development Guide](development/contributing.md)
    - [API Reference](api/overview.md)

## Getting Help

-   📖 Check the [full documentation](index.md)
-   🐛 Report issues on [GitHub](https://github.com/your-username/clean-interfaces/issues)
-   💬 Ask questions in [Discussions](https://github.com/your-username/clean-interfaces/discussions)
