# CLI Interface Guide

The Command Line Interface (CLI) provides an interactive way to work with Clean Interfaces.

## Overview

The CLI interface is built using [Typer](https://typer.tiangolo.com/), providing:

- Rich terminal output with colors and formatting
- Interactive prompts and confirmations
- Automatic help generation
- Command completion support

## Running the CLI

### Basic Usage

```bash
# Run with default settings
uv run python -m clean_interfaces.main

# Run with custom environment
uv run python -m clean_interfaces.main --dotenv custom.env

# Show help
uv run python -m clean_interfaces.main --help
```

### Setting CLI as Default

The CLI is the default interface, but you can explicitly set it:

```ini
# .env
INTERFACE_TYPE=cli
```

## CLI Features

### Interactive Mode

When you run the CLI, it starts in interactive mode:

```
Welcome to Clean Interfaces CLI!
Type 'help' for available commands.

> help
Available commands:
  - hello: Display a welcome message
  - status: Show application status
  - exit: Exit the application
```

### Command Examples

#### Hello Command

```
> hello
Hello from Clean Interfaces!
Current time: 2025-07-20 10:30:45
```

#### Status Command

```
> status
Application Status:
- Interface: CLI
- Version: 0.1.0
- Uptime: 00:05:23
- Log Level: INFO
```

## Configuration for CLI

### Recommended Development Settings

```ini
# dev.env for CLI development
INTERFACE_TYPE=cli
LOG_LEVEL=DEBUG
LOG_FORMAT=console  # Human-readable output
```

### Production CLI Settings

```ini
# prod.env for CLI production
INTERFACE_TYPE=cli
LOG_LEVEL=WARNING
LOG_FORMAT=json
LOG_FILE_PATH=/var/log/cli.log
```

## Advanced CLI Usage

### Command-Line Arguments

The main entry point supports these arguments:

```bash
# Specify custom .env file
uv run python -m clean_interfaces.main --dotenv /path/to/config.env

# Short form
uv run python -m clean_interfaces.main -e config.env
```

### Environment Overrides

Override specific settings without modifying files:

```bash
# Temporary debug mode
LOG_LEVEL=DEBUG uv run python -m clean_interfaces.main

# Change log format
LOG_FORMAT=plain uv run python -m clean_interfaces.main
```

## CLI Development

### Adding New Commands

To add new commands to the CLI, modify the `CLIInterface` class:

```python
# src/clean_interfaces/interfaces/cli.py

class CLIInterface(BaseInterface):
    def run(self) -> None:
        """Run the CLI interface."""
        # Add your command logic here
```

### Custom Prompts

The CLI uses rich prompts for user interaction:

```python
from rich.prompt import Prompt, Confirm

# Text input
name = Prompt.ask("Enter your name")

# Confirmation
if Confirm.ask("Do you want to continue?"):
    # Continue processing
```

### Styling Output

Use Rich for formatted output:

```python
from rich.console import Console
from rich.table import Table

console = Console()

# Colored output
console.print("Success!", style="green bold")

# Tables
table = Table(title="Status")
table.add_column("Property")
table.add_column("Value")
table.add_row("Interface", "CLI")
console.print(table)
```

## Error Handling

The CLI provides user-friendly error messages:

```
> invalid_command
Error: Unknown command 'invalid_command'
Type 'help' for available commands.

> exit
Goodbye!
```

## Logging in CLI Mode

### Console Logging

With `LOG_FORMAT=console`, logs appear inline:

```
2025-07-20 10:30:45 [INFO] Starting CLI interface
2025-07-20 10:30:46 [DEBUG] Command received: hello
2025-07-20 10:30:46 [INFO] Executing hello command
```

### JSON Logging

With `LOG_FORMAT=json`, logs are structured:

```json
{"timestamp": "2025-07-20T10:30:45Z", "level": "info", "message": "Starting CLI interface"}
```

## Tips and Tricks

### 1. Enable Debug Logging

For troubleshooting:

```bash
LOG_LEVEL=DEBUG LOG_FORMAT=console uv run python -m clean_interfaces.main
```

### 2. Pipe Output

Useful for scripting:

```bash
echo "hello" | uv run python -m clean_interfaces.main
```

### 3. Custom Configurations

Create task-specific configs:

```ini
# data-import.env
INTERFACE_TYPE=cli
LOG_LEVEL=INFO
LOG_FILE_PATH=imports.log
```

### 4. Shell Integration

Add an alias for convenience:

```bash
alias ci='uv run python -m clean_interfaces.main'
```

## Common Issues

### Terminal Encoding

If you see encoding issues:

```bash
export PYTHONIOENCODING=utf-8
uv run python -m clean_interfaces.main
```

### Color Output

If colors don't appear:

```bash
# Force color output
export FORCE_COLOR=1
uv run python -m clean_interfaces.main
```

### Input Buffer

For large inputs:

```bash
# Increase buffer size
export PYTHONUNBUFFERED=1
uv run python -m clean_interfaces.main
```

## Next Steps

- Learn about the [REST API Interface](restapi.md)
- Configure [Logging](logging.md)
- Explore [Environment Variables](environment.md)
- Read the [API Reference](../api/interfaces.md)