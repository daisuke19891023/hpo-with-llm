# Code Style Guide

This guide defines the coding standards and conventions for the Clean Interfaces project.

## Overview

We prioritize:

-   **Readability**: Code should be easy to understand
-   **Consistency**: Follow established patterns
-   **Type Safety**: Use type hints everywhere
-   **Documentation**: Document public APIs

## Tools

We use these tools to enforce code quality:

-   **[Ruff](https://docs.astral.sh/ruff/)**: Linting and formatting
-   **[Pyright](https://github.com/microsoft/pyright)**: Type checking
-   **[Black](https://black.readthedocs.io/)**: Code formatting (via Ruff)
-   **[isort](https://pycqa.github.io/isort/)**: Import sorting (via Ruff)

## Python Version

We target Python 3.13+ and use modern Python features:

```python
# Use modern type hints
def process(items: list[str]) -> dict[str, int]:
    pass

# Use match statements
match command:
    case "start":
        start_process()
    case "stop":
        stop_process()
    case _:
        raise ValueError(f"Unknown command: {command}")
```

## Code Formatting

### General Rules

-   Line length: 88 characters (Black default)
-   Indentation: 4 spaces
-   Use double quotes for strings
-   Add trailing commas in multiline structures

### Import Organization

Imports are organized by isort:

```python
# Standard library
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

# Third-party packages
import pytest
import typer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local imports
from clean_interfaces.base import BaseComponent
from clean_interfaces.models.api import WelcomeResponse
from clean_interfaces.utils.logger import get_logger
```

### Function and Class Definitions

```python
# Functions with type hints
def calculate_total(
    items: list[dict[str, float]],
    tax_rate: float = 0.08,
    discount: float = 0.0,
) -> float:
    """Calculate total with tax and discount.

    Args:
        items: List of items with 'price' key
        tax_rate: Tax rate as decimal
        discount: Discount amount

    Returns:
        Total amount after tax and discount
    """
    subtotal = sum(item.get("price", 0.0) for item in items)
    total = subtotal * (1 + tax_rate) - discount
    return max(0.0, total)


# Classes with proper structure
class OrderProcessor(BaseComponent):
    """Process orders with validation and logging."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize processor with configuration.

        Args:
            config: Processor configuration
        """
        super().__init__()
        self.config = config
        self._validator = OrderValidator()

    def process(self, order: Order) -> ProcessResult:
        """Process a single order.

        Args:
            order: Order to process

        Returns:
            Processing result

        Raises:
            ValidationError: If order is invalid
            ProcessingError: If processing fails
        """
        self.logger.info("processing_order", order_id=order.id)

        # Validate order
        self._validator.validate(order)

        # Process order
        try:
            result = self._execute_processing(order)
            self.logger.info("order_processed", order_id=order.id)
            return result
        except Exception as e:
            self.logger.error(
                "processing_failed",
                order_id=order.id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to process order {order.id}") from e
```

## Type Hints

### Always Use Type Hints

```python
# Good - fully typed
def fetch_user(user_id: int, include_profile: bool = False) -> User | None:
    """Fetch user by ID."""
    pass

# Bad - no type hints
def fetch_user(user_id, include_profile=False):
    """Fetch user by ID."""
    pass
```

### Common Type Patterns

```python
from typing import Any, TypeVar, Generic, Protocol
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

# Type variables
T = TypeVar("T")
TConfig = TypeVar("TConfig", bound=BaseModel)

# Generic classes
class Repository(Generic[T]):
    """Generic repository for entities."""

    def save(self, entity: T) -> T:
        """Save entity."""
        pass

    def find_by_id(self, id: int) -> T | None:
        """Find entity by ID."""
        pass

# Protocols for structural typing
class Processor(Protocol):
    """Protocol for processors."""

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process data."""
        ...

# Union types
ConfigValue = str | int | float | bool | None
ConfigDict = dict[str, ConfigValue | dict[str, ConfigValue]]

# Callable types
ErrorHandler = Callable[[Exception], None]
Middleware = Callable[[Request], Response]
```

### Avoid Any When Possible

```python
# Bad - using Any unnecessarily
def process_data(data: Any) -> Any:
    return data["value"]

# Good - specific types
def process_data(data: dict[str, str]) -> str:
    return data["value"]

# Good - when Any is necessary, document why
def load_plugin(name: str) -> Any:
    """Load plugin dynamically.

    Returns Any because plugin types are not known at compile time.
    """
    module = importlib.import_module(name)
    return module.Plugin()
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def complex_operation(
    data: list[dict[str, Any]],
    config: Config,
    *,
    validate: bool = True,
    timeout: float | None = None,
) -> OperationResult:
    """Perform complex operation on data.

    This function processes the input data according to the provided
    configuration. It supports validation and timeout controls.

    Args:
        data: List of data items to process. Each item should contain
            'id' and 'value' keys.
        config: Configuration object controlling the operation behavior.
        validate: Whether to validate input data before processing.
            Defaults to True.
        timeout: Maximum time in seconds for the operation. None means
            no timeout. Defaults to None.

    Returns:
        OperationResult containing:
            - processed_count: Number of successfully processed items
            - failed_items: List of items that failed processing
            - duration: Total processing time in seconds

    Raises:
        ValidationError: If validate=True and data is invalid.
        TimeoutError: If operation exceeds timeout.
        ConfigError: If configuration is invalid.

    Example:
        >>> config = Config(mode="fast")
        >>> result = complex_operation(
        ...     [{"id": 1, "value": "test"}],
        ...     config,
        ...     timeout=30.0
        ... )
        >>> print(result.processed_count)
        1

    Note:
        This operation is CPU-intensive and should be run in a
        background task for large datasets.
    """
```

### Comments

Write self-documenting code that rarely needs comments:

```python
# Bad - comment explains what
# Increment counter by one
counter += 1

# Good - comment explains why
# Retry counter is incremented before the attempt to ensure
# we don't exceed max_retries even if an exception occurs
retry_count += 1
```

### Type Aliases

Document complex type aliases:

```python
# Type alias for user preferences mapping
# Maps preference key to value and last updated timestamp
UserPreferences = dict[str, tuple[Any, datetime]]

# Configuration for batch processing
# Includes batch size, timeout, and retry settings
BatchConfig = dict[str, int | float | bool]
```

## Error Handling

### Specific Exceptions

```python
# Good - specific exception types
def validate_email(email: str) -> None:
    """Validate email format.

    Raises:
        ValueError: If email format is invalid
    """
    if "@" not in email:
        raise ValueError(f"Invalid email format: {email}")

# Good - exception chaining
try:
    data = json.loads(json_string)
except json.JSONDecodeError as e:
    raise ValidationError(f"Invalid JSON: {e}") from e
```

### Error Messages

```python
# Good - informative error messages
if not isinstance(value, (int, float)):
    raise TypeError(
        f"Expected numeric value, got {type(value).__name__}: {value!r}"
    )

# Good - include context
if user_id not in self.users:
    raise KeyError(
        f"User {user_id} not found. "
        f"Available users: {list(self.users.keys())}"
    )
```

## Naming Conventions

### General Rules

-   **Classes**: PascalCase (`UserProfile`, `OrderProcessor`)
-   **Functions/Methods**: snake_case (`calculate_total`, `get_user`)
-   **Constants**: UPPER_SNAKE_CASE (`MAX_RETRIES`, `DEFAULT_TIMEOUT`)
-   **Private**: Leading underscore (`_internal_method`)
-   **Type Variables**: PascalCase, often prefixed with T (`T`, `TConfig`)

### Descriptive Names

```python
# Bad - unclear abbreviations
def calc_amt(p, t, r):
    return p * (1 + r) ** t

# Good - clear, descriptive names
def calculate_compound_interest(
    principal: float,
    time_years: float,
    interest_rate: float,
) -> float:
    """Calculate compound interest."""
    return principal * (1 + interest_rate) ** time_years
```

## Testing Style

### Test Organization

```python
class TestOrderProcessor:
    """Tests for OrderProcessor."""

    @pytest.fixture
    def processor(self) -> OrderProcessor:
        """Create processor instance for testing."""
        config = {"max_items": 100}
        return OrderProcessor(config)

    @pytest.fixture
    def sample_order(self) -> Order:
        """Create sample order for testing."""
        return Order(
            id="123",
            items=[{"name": "Widget", "price": 9.99}],
        )

    def test_process_valid_order(
        self,
        processor: OrderProcessor,
        sample_order: Order,
    ) -> None:
        """Test processing a valid order."""
        result = processor.process(sample_order)

        assert result.status == "completed"
        assert result.order_id == "123"
        assert result.total == 9.99
```

### Test Naming

```python
# Pattern: test_<method>_<scenario>_<expected_result>

def test_calculate_total_with_empty_list_returns_zero() -> None:
    """Test that calculating total of empty list returns zero."""
    assert calculate_total([]) == 0.0

def test_validate_email_with_invalid_format_raises_value_error() -> None:
    """Test that invalid email format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid email format"):
        validate_email("not-an-email")
```

## Performance Considerations

### Efficient Code

```python
# Good - generator for memory efficiency
def process_large_file(file_path: Path) -> Iterator[ProcessedLine]:
    """Process large file line by line."""
    with open(file_path) as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():  # Skip empty lines
                yield ProcessedLine(line_num, line.strip())

# Good - set for O(1) lookups
VALID_COMMANDS = {"start", "stop", "restart", "status"}

def is_valid_command(command: str) -> bool:
    """Check if command is valid."""
    return command in VALID_COMMANDS
```

## Configuration

### Ruff Configuration

In `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["ALL"]
ignore = [
    "D100",  # Missing module docstring
    "D104",  # Missing public package docstring
    "COM812", # Trailing comma missing
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow assert in tests
```

### Pyright Configuration

In `pyproject.toml`:

```toml
[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"
reportUnknownMemberType = false
reportUnknownArgumentType = false
```

## Pre-commit Hooks

Ensure code quality before commits:

```yaml
# .pre-commit-config.yaml
repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.8.0
      hooks:
          - id: ruff
            args: [--fix]
          - id: ruff-format

    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v5.0.0
      hooks:
          - id: trailing-whitespace
          - id: end-of-file-fixer
          - id: check-yaml
          - id: check-added-large-files
```

## See Also

-   [Contributing Guide](contributing.md) - How to contribute
-   [Testing Guide](testing.md) - Testing best practices
-   [Ruff Documentation](https://docs.astral.sh/ruff/) - Linter documentation
-   [Pyright Documentation](https://github.com/microsoft/pyright) - Type checker docs
