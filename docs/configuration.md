# Configuration Guide

Clean Interfaces uses environment variables for configuration, with support for `.env` files and command-line options.

## Configuration Methods

### 1. Environment Variables

Set environment variables directly in your shell:

```bash
export INTERFACE_TYPE=restapi
export LOG_LEVEL=DEBUG
uv run python -m clean_interfaces.main
```

### 2. Default .env File

Create a `.env` file in the project root:

```ini
# .env
INTERFACE_TYPE=cli
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### 3. Custom .env Files

Use the `--dotenv` option to specify custom configuration files:

```bash
uv run python -m clean_interfaces.main --dotenv production.env
```

## Configuration Options

### Core Settings

| Variable         | Description      | Default | Options          |
| ---------------- | ---------------- | ------- | ---------------- |
| `INTERFACE_TYPE` | Interface to use | `cli`   | `cli`, `restapi` |

### Logging Configuration

| Variable        | Description       | Default | Options                                         |
| --------------- | ----------------- | ------- | ----------------------------------------------- |
| `LOG_LEVEL`     | Logging verbosity | `INFO`  | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FORMAT`    | Output format     | `json`  | `json`, `console`, `plain`                      |
| `LOG_FILE_PATH` | Log file location | None    | Any valid file path                             |

### OpenTelemetry Configuration

OpenTelemetry exporter configuration has been removed. `OTEL_*` variables are not used by the application. Trace context (if OTEL is present) may appear in logs but no export is performed.

## Configuration Precedence

Configuration values are loaded in the following order (later values override earlier ones):

1. Default values in code
2. `.env` file in project root
3. Custom `.env` file (if specified with `--dotenv`)
4. Environment variables
5. Command-line arguments

## Environment-Specific Configurations

### Development Configuration

```ini
# dev.env
INTERFACE_TYPE=cli
LOG_LEVEL=DEBUG
LOG_FORMAT=console
# OpenTelemetry exporter removed
```

**Features:**

-   Verbose logging for debugging
-   Human-readable console output
-   Local file logging only

### Testing Configuration

```ini
# test.env
INTERFACE_TYPE=cli
LOG_LEVEL=WARNING
LOG_FORMAT=json
LOG_FILE_PATH=tests/logs/test.log
```

**Features:**

-   Reduced logging noise
-   Structured output for parsing
-   Separate test log file

### Production Configuration

```ini
# prod.env
INTERFACE_TYPE=restapi
LOG_LEVEL=ERROR
LOG_FORMAT=json
LOG_FILE_PATH=/var/log/clean-interfaces/app.log
# OpenTelemetry exporter removed
```

**Features:**

-   Minimal logging overhead
-   Structured JSON for log aggregation
-   OpenTelemetry export for monitoring
-   Persistent log files

## Advanced Configuration

### Dynamic Configuration

You can dynamically set configuration:

```python
import os

# Set configuration programmatically
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['INTERFACE_TYPE'] = 'restapi'

# Then run the application
from clean_interfaces.app import run_app
run_app()
```

### Configuration Validation

All configuration values are validated on startup:

-   **Type validation**: Ensures correct data types
-   **Value validation**: Checks for valid options
-   **Range validation**: Validates numeric ranges

Invalid configurations will raise clear error messages:

```
ValueError: Invalid log level: TRACE. Must be one of {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
```

### Configuration Debugging

To debug configuration issues:

1. **Check loaded values**:

    ```bash
    LOG_LEVEL=DEBUG uv run python -m clean_interfaces.main
    ```

2. **Verify .env file**:

    ```bash
    cat .env
    ```

3. **Check environment**:
    ```bash
    env | grep -E '^(LOG_|OTEL_|INTERFACE_)'
    ```

## Best Practices

### 1. Use Environment-Specific Files

Create separate configuration files for each environment:

```
.env.development
.env.testing
.env.staging
.env.production
```

### 2. Never Commit Secrets

Add `.env` files to `.gitignore`:

```gitignore
# Environment files
.env
.env.*
!.env.example
```

### 3. Document All Options

Keep `.env.example` updated with all available options:

```ini
# Copy this to .env and configure
INTERFACE_TYPE=cli
LOG_LEVEL=INFO
# Add more options...
```

### 4. Use Descriptive Names

For custom configurations:

```bash
# Good
uv run python -m clean_interfaces.main --dotenv config/production.env

# Less clear
uv run python -m clean_interfaces.main --dotenv prod.env
```

### 5. Validate Early

Test configuration in development:

```bash
# Test production config locally
uv run python -m clean_interfaces.main --dotenv prod.env
```

## Troubleshooting

### Configuration Not Loading

**Problem**: Changes to `.env` not taking effect

**Solution**:

-   Ensure file is in project root
-   Check file permissions
-   Use `--dotenv` to specify path explicitly

### Invalid Configuration Values

**Problem**: Application fails with configuration errors

**Solution**:

-   Check spelling of option names
-   Verify values match allowed options
-   Review error message for valid choices

### Environment Variable Conflicts

**Problem**: Unexpected configuration values

**Solution**:

-   Check system environment variables
-   Use `env | grep PATTERN` to debug
-   Explicitly unset conflicting variables

## Next Steps

-   Learn about [Logging Configuration](guides/logging.md)
-   Explore [Environment Variables](guides/environment.md)
-   Read about [Interface Types](guides/cli.md)
