# Environment Variables Guide

Clean Interfaces uses environment variables for configuration, following the 12-factor app methodology.

## Overview

Environment variables provide a clean way to configure the application without modifying code. They can be set directly or loaded from `.env` files.

## Core Configuration

### Interface Selection

```bash
# Choose the interface type
export INTERFACE_TYPE=cli     # CLI interface (default)
export INTERFACE_TYPE=restapi # REST API interface
```

### Logging Configuration

```bash
# Log level
export LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log format
export LOG_FORMAT=json    # json (default) or console

# File logging
export LOG_FILE=/var/log/app.log  # Path to log file
```

### OpenTelemetry Configuration

OpenTelemetry exporter configuration has been removed. The application ignores `OTEL_*` variables; trace context may still be included if OTEL is present.

## Using .env Files

Create a `.env` file in your project root:

```bash
# .env
INTERFACE_TYPE=restapi
LOG_LEVEL=INFO
LOG_FORMAT=json
OTEL_EXPORT_ENABLED=true
OTEL_EXPORT_MODE=file
OTEL_LOG_FILE=./logs/otel.log
```

Load it when running the application:

```bash
# Using the CLI option
python -m clean_interfaces.main --dotenv .env

# Or with python-dotenv auto-loading
python -m clean_interfaces.main
```

## Environment-Specific Configuration

Use different `.env` files for different environments:

```bash
# Development
python -m clean_interfaces.main --dotenv .env.development

# Production
python -m clean_interfaces.main --dotenv .env.production

# Testing
python -m clean_interfaces.main --dotenv .env.test
```

## REST API Specific Variables

When using the REST API interface:

```bash
# Server configuration
export HOST=0.0.0.0
export PORT=8000

# CORS settings (if implemented)
export CORS_ORIGINS="http://localhost:3000,https://myapp.com"
```

## Security Best Practices

1. **Never commit `.env` files**: Add them to `.gitignore`
2. **Use strong values**: Generate secure random values for secrets
3. **Validate inputs**: Always validate environment variable values
4. **Document requirements**: List all required variables in `.env.example`
5. **Use secrets management**: In production, use proper secrets management tools

## Example .env.example

Create a `.env.example` file to document available variables:

```bash
# Interface Configuration
INTERFACE_TYPE=cli

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=

# OpenTelemetry Configuration (deprecated)
# OTEL_* variables are ignored by the application

# REST API Configuration (when INTERFACE_TYPE=restapi)
HOST=0.0.0.0
PORT=8000
```

## Loading Order

Environment variables are loaded in this order (later overrides earlier):

1. System environment variables
2. `.env` file (if present in current directory)
3. Explicitly specified `--dotenv` file
4. Command-line arguments (if applicable)

## Debugging

To debug environment variable issues:

```bash
# Print all environment variables
python -c "import os; print(dict(os.environ))"

# Check specific variable
echo $INTERFACE_TYPE

# Run with debug logging
LOG_LEVEL=DEBUG python -m clean_interfaces.main
```

## Next Steps

-   Learn about [Logging](logging.md) configuration options
-   Explore the [REST API](restapi.md) environment variables
-   See the [Configuration](../configuration.md) overview
