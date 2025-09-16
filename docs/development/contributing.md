# Contributing Guide

Thank you for considering contributing to Clean Interfaces! This guide will help you get started.

## Getting Started

### Prerequisites

-   Python 3.13 or higher
-   [uv](https://github.com/astral-sh/uv) for dependency management
-   Git for version control

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/your-username/clean-interfaces.git
cd clean-interfaces
```

2. **Run the setup script**

```bash
./setup.sh
```

This will:

-   Create a virtual environment
-   Install all dependencies
-   Set up pre-commit hooks
-   Initialize the project structure

3. **Activate the virtual environment**

```bash
source .venv/bin/activate
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:

-   `feature/add-oauth-support`
-   `fix/memory-leak-in-parser`
-   `docs/update-api-reference`

### 2. Make Your Changes

Follow the 4-phase development approach:

1. **Explore**: Understand existing code
2. **Plan**: Design your implementation
3. **Implement**: Write code using TDD
4. **Commit**: Create atomic commits

### 3. Write Tests

All code must have tests:

```bash
# Run tests
nox -s test

# Run specific test file
nox -s test -- tests/unit/test_your_module.py

# Run with coverage
nox -s coverage
```

Test requirements:

-   Minimum 80% code coverage
-   Write tests before implementation (TDD)
-   Include unit, integration, and E2E tests

### 4. Check Code Quality

Run all quality checks:

```bash
# Run all checks
nox

# Individual checks
nox -s lint      # Linting
nox -s typing    # Type checking
nox -s format_code  # Formatting
nox -s security  # Security checks
```

### 5. Update Documentation

-   Update docstrings for new functions/classes
-   Update relevant documentation in `docs/`
-   Add examples for new features

Build documentation locally:

```bash
nox -s docs
# View at http://localhost:8000
python -m http.server -d site 8000
```

### 6. Commit Your Changes

Follow [conventional commits](https://www.conventionalcommits.org/):

```bash
# Stage changes
git add -A

# Commit with conventional format
git commit -m "feat: add new validation feature"
```

Commit types:

-   `feat`: New feature
-   `fix`: Bug fix
-   `docs`: Documentation changes
-   `style`: Code style changes
-   `refactor`: Code refactoring
-   `test`: Test additions/changes
-   `chore`: Build/tooling changes

## Code Style

### Python Style Guide

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```python
# Good
def calculate_total(items: list[dict[str, float]]) -> float:
    """Calculate total price of items.

    Args:
        items: List of items with 'price' key

    Returns:
        Total price
    """
    return sum(item.get("price", 0.0) for item in items)

# Bad
def calc(i):
    t = 0
    for x in i:
        t += x["price"]
    return t
```

### Type Hints

Always use type hints:

```python
from typing import Any
from pathlib import Path

def process_file(
    file_path: Path,
    encoding: str = "utf-8",
    options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Process a file with given options."""
    if options is None:
        options = {}
    # Implementation
```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: int,
    optional: bool = False
) -> dict[str, Any]:
    """Brief description of function.

    Longer description if needed. Can span multiple lines
    and include examples.

    Args:
        param1: Description of param1
        param2: Description of param2
        optional: Description of optional parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result["status"])
        success
    """
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestMyComponent:
    """Test suite for MyComponent."""

    @pytest.fixture
    def component(self) -> MyComponent:
        """Create component instance for testing."""
        return MyComponent()

    def test_normal_operation(self, component: MyComponent) -> None:
        """Test normal operation scenario."""
        result = component.process("input")
        assert result == "expected"

    def test_error_handling(self, component: MyComponent) -> None:
        """Test error handling."""
        with pytest.raises(ValueError, match="Invalid input"):
            component.process("")

    @patch("module.external_service")
    def test_with_mock(
        self,
        mock_service: Mock,
        component: MyComponent
    ) -> None:
        """Test with mocked external service."""
        mock_service.return_value = "mocked"
        result = component.process_with_service()
        assert result == "mocked"
        mock_service.assert_called_once()
```

### Test Categories

1. **Unit Tests** (`tests/unit/`)

    - Test individual functions/methods
    - Mock external dependencies
    - Fast execution

2. **Integration Tests** (`tests/integration/`)

    - Test component interactions
    - Use real dependencies where possible
    - Medium execution time

3. **E2E Tests** (`tests/e2e/`)
    - Test complete workflows
    - Minimal mocking
    - Slower execution

## Pull Request Process

1. **Update your branch**

```bash
git fetch origin
git rebase origin/main
```

2. **Push your changes**

```bash
git push origin feature/your-feature-name
```

3. **Create Pull Request**

-   Use a descriptive title
-   Reference any related issues
-   Include a detailed description
-   Add screenshots for UI changes

4. **PR Checklist**

-   [ ] Tests pass (`nox -s test`)
-   [ ] Code is formatted (`nox -s format_code`)
-   [ ] Type checks pass (`nox -s typing`)
-   [ ] Documentation updated
-   [ ] Commit messages follow conventions
-   [ ] PR description is complete

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. GitHub Actions publishes to PyPI

## Getting Help

-   Open an issue for bugs/features
-   Join discussions for questions
-   Check existing issues first
-   Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## See Also

-   [Testing Guide](testing.md) - Detailed testing information
-   [Code Style Guide](code-style.md) - Style conventions
-   [Development Setup](../installation.md) - Installation instructions
