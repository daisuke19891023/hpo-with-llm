# Testing Guide

Comprehensive testing is crucial for maintaining code quality in Clean Interfaces.

## Testing Philosophy

We follow Test-Driven Development (TDD) principles:

1. **Write tests first**: Define behavior before implementation
2. **Red-Green-Refactor**: Fail, pass, improve
3. **Test at multiple levels**: Unit, integration, and E2E
4. **Maintain high coverage**: Minimum 80% code coverage

## Test Structure

```
tests/
├── unit/           # Isolated component tests
├── integration/    # Component interaction tests
├── e2e/           # End-to-end workflow tests
├── fixtures/      # Test data and fixtures
├── helpers/       # Test utilities
└── conftest.py    # Shared pytest configuration
```

## Running Tests

### Using Nox

```bash
# Run all tests
nox -s test

# Run specific test categories
nox -s test_unit
nox -s test_api
nox -s test_e2e

# Run with coverage report
nox -s coverage

# Run all quality checks
nox -s ci
```

### Using Pytest Directly

```bash
# Run all tests
uv run pytest

# Run specific file
uv run pytest tests/unit/test_app.py

# Run specific test
uv run pytest tests/unit/test_app.py::TestApplication::test_run

# Run with options
uv run pytest -xvs  # Stop on first failure, verbose, no capture
uv run pytest -k "test_file"  # Run tests matching pattern
```

## Writing Tests

### Unit Tests

Test individual components in isolation:

```python
# tests/unit/clean_interfaces/utils/test_file_handler.py
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from clean_interfaces.utils.file_handler import FileHandler


class TestFileHandler:
    """Unit tests for FileHandler."""
    
    @pytest.fixture
    def handler(self) -> FileHandler:
        """Create FileHandler instance."""
        return FileHandler(encoding="utf-8")
    
    def test_read_text_success(self, handler: FileHandler) -> None:
        """Test successful text file reading."""
        mock_data = "Hello, World!"
        
        with patch("builtins.open", mock_open(read_data=mock_data)):
            result = handler.read_text("test.txt")
            
        assert result == mock_data
    
    def test_read_text_file_not_found(self, handler: FileHandler) -> None:
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            handler.read_text("nonexistent.txt")
    
    @pytest.mark.parametrize("encoding", ["utf-8", "cp932", "shift_jis"])
    def test_encoding_support(self, encoding: str) -> None:
        """Test different encoding support."""
        handler = FileHandler(encoding=encoding)
        assert handler.encoding == encoding
```

### Integration Tests

Test component interactions:

```python
# tests/integration/test_app_integration.py
import pytest
from pathlib import Path

from clean_interfaces.app import Application
from clean_interfaces.types import InterfaceType


class TestApplicationIntegration:
    """Integration tests for Application."""
    
    @pytest.fixture
    def temp_env_file(self, tmp_path: Path) -> Path:
        """Create temporary .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("INTERFACE_TYPE=cli\nLOG_LEVEL=DEBUG")
        return env_file
    
    def test_app_with_env_file(
        self,
        temp_env_file: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test application with .env file loading."""
        # Clear environment
        monkeypatch.delenv("INTERFACE_TYPE", raising=False)
        
        # Create and run app
        app = Application(dotenv_path=temp_env_file)
        
        # Verify environment was loaded
        import os
        assert os.getenv("INTERFACE_TYPE") == "cli"
        assert os.getenv("LOG_LEVEL") == "DEBUG"
```

### E2E Tests

Test complete workflows:

```python
# tests/e2e/test_cli_workflow.py
import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIWorkflow:
    """E2E tests for CLI workflow."""
    
    def test_complete_cli_workflow(self, tmp_path: Path) -> None:
        """Test complete CLI usage workflow."""
        # Create test data
        input_file = tmp_path / "input.json"
        input_file.write_text('{"message": "test"}')
        
        # Run CLI command
        result = subprocess.run(
            [sys.executable, "-m", "clean_interfaces.main", "--help"],
            capture_output=True,
            text=True,
            check=False
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "--dotenv" in result.stdout
```

## Test Fixtures

### Conftest.py

Shared fixtures in `tests/conftest.py`:

```python
import pytest
from pathlib import Path
from typing import Generator

from clean_interfaces.app import Application


@pytest.fixture
def app() -> Application:
    """Create application instance."""
    return Application()


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield data_dir
    # Cleanup if needed


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean environment for each test."""
    # Remove any existing settings
    for key in ["INTERFACE_TYPE", "LOG_LEVEL", "LOG_FORMAT"]:
        monkeypatch.delenv(key, raising=False)
```

### Custom Fixtures

Create reusable test components:

```python
# tests/fixtures/factories.py
from clean_interfaces.models.api import WelcomeResponse


def create_welcome_response(**kwargs: Any) -> WelcomeResponse:
    """Create WelcomeResponse with defaults."""
    defaults = {
        "message": "Test Welcome",
        "hint": "Test Hint",
        "version": "0.0.1"
    }
    return WelcomeResponse(**{**defaults, **kwargs})
```

## Mocking Strategies

### Mocking External Services

```python
from unittest.mock import patch, Mock

class TestExternalService:
    """Test external service interactions."""
    
    @patch("requests.get")
    def test_api_call(self, mock_get: Mock) -> None:
        """Test API call handling."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Test code that uses requests
        result = fetch_data("https://api.example.com")
        
        # Verify
        assert result == {"status": "ok"}
        mock_get.assert_called_once_with("https://api.example.com")
```

### Mocking File System

```python
from unittest.mock import mock_open, patch

def test_file_operations() -> None:
    """Test file operations with mocks."""
    mock_data = "file content"
    
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with open("test.txt") as f:
            content = f.read()
    
    assert content == mock_data
```

## Testing Best Practices

### 1. Test Naming

Use descriptive test names:

```python
# Good
def test_file_handler_reads_utf8_encoded_file_successfully() -> None:
    pass

def test_file_handler_raises_error_for_invalid_json() -> None:
    pass

# Bad
def test_read() -> None:
    pass

def test_error() -> None:
    pass
```

### 2. Arrange-Act-Assert

Structure tests clearly:

```python
def test_user_creation() -> None:
    """Test user creation process."""
    # Arrange
    user_data = {"name": "John", "email": "john@example.com"}
    service = UserService()
    
    # Act
    user = service.create_user(**user_data)
    
    # Assert
    assert user.name == "John"
    assert user.email == "john@example.com"
    assert user.id is not None
```

### 3. Parametrized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize(
    "input_value,expected",
    [
        ("hello", "HELLO"),
        ("World", "WORLD"),
        ("123", "123"),
        ("", ""),
        (None, ""),
    ]
)
def test_uppercase_conversion(input_value: str | None, expected: str) -> None:
    """Test uppercase conversion with various inputs."""
    result = convert_to_uppercase(input_value)
    assert result == expected
```

### 4. Test Isolation

Ensure tests don't affect each other:

```python
class TestDatabase:
    """Database tests with isolation."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, db: Database) -> Generator[None, None, None]:
        """Setup and teardown for each test."""
        db.begin_transaction()
        yield
        db.rollback()
    
    def test_user_creation(self, db: Database) -> None:
        """Test creating user in database."""
        # Test runs in transaction that's rolled back
        db.create_user("test@example.com")
        assert db.get_user_count() == 1
```

### 5. Testing Exceptions

Test error conditions thoroughly:

```python
def test_validation_errors() -> None:
    """Test validation error handling."""
    with pytest.raises(ValueError, match="Invalid email format"):
        validate_email("not-an-email")
    
    with pytest.raises(TypeError, match="Email must be string"):
        validate_email(123)
```

## Coverage Requirements

Maintain minimum 80% code coverage:

```bash
# Check coverage
nox -s coverage

# Generate HTML report
uv run pytest --cov=src --cov-report=html
# Open htmlcov/index.html
```

Coverage configuration in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/migrations/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
    "pass",
    "except ImportError:",
]
```

## Performance Testing

For performance-critical code:

```python
import pytest
import time


@pytest.mark.performance
def test_processing_performance() -> None:
    """Test processing performance."""
    data = list(range(10000))
    
    start = time.time()
    result = process_large_dataset(data)
    duration = time.time() - start
    
    assert len(result) == 10000
    assert duration < 1.0  # Should complete in under 1 second
```

## Debugging Tests

### Using pytest debugging

```bash
# Drop into debugger on failure
uv run pytest --pdb

# Drop into debugger at specific point
def test_complex_logic() -> None:
    result = complex_function()
    import pdb; pdb.set_trace()  # Debugger starts here
    assert result == expected
```

### Verbose output

```bash
# Show print statements
uv run pytest -s

# Show detailed output
uv run pytest -vv

# Show local variables on failure
uv run pytest -l
```

## See Also

- [Contributing Guide](contributing.md) - Development workflow
- [Code Style Guide](code-style.md) - Coding standards
- [pytest Documentation](https://docs.pytest.org/) - Official pytest docs