# Models Module

The `models` module provides data models for request/response handling and input/output operations.

## Overview

All models use Pydantic for:
- Data validation
- Serialization/deserialization
- Type safety
- Automatic documentation generation

## API Models

Models for REST API request and response handling.

### HealthResponse

Health check response model.

```python
from clean_interfaces.models.api import HealthResponse
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
```

#### Fields

- `status` (str): Health status, defaults to "healthy"
- `timestamp` (datetime): Current timestamp

#### Example

```python
health = HealthResponse()
print(health.model_dump_json())
# {"status": "healthy", "timestamp": "2025-01-20T12:00:00"}
```

### WelcomeResponse

Welcome endpoint response model.

```python
from clean_interfaces.models.api import WelcomeResponse

class WelcomeResponse(BaseModel):
    """Welcome response model."""
    
    message: str = Field(
        default="Welcome to Clean Interfaces!",
        description="Welcome message"
    )
    hint: str = Field(
        default="Type --help for more information",
        description="Usage hint"
    )
    version: str = Field(
        default="0.1.0",
        description="API version"
    )
```

#### Fields

- `message` (str): Welcome message
- `hint` (str): Usage hint for users
- `version` (str): API version string

#### Example

```python
welcome = WelcomeResponse(version="1.2.0")
print(welcome.model_dump())
# {
#     "message": "Welcome to Clean Interfaces!",
#     "hint": "Type --help for more information",
#     "version": "1.2.0"
# }
```

### ErrorResponse

Error response model for API errors.

```python
from clean_interfaces.models.api import ErrorResponse

class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: str | None = Field(
        default=None,
        description="Additional error details"
    )
```

#### Fields

- `error` (str): Main error message (required)
- `detail` (str | None): Optional additional details

#### Example

```python
error = ErrorResponse(
    error="Resource not found",
    detail="The requested user ID does not exist"
)
print(error.model_dump_json(exclude_none=True))
# {"error": "Resource not found", "detail": "The requested user ID does not exist"}
```

## IO Models

Models for CLI input/output operations.

### WelcomeMessage

Welcome message model for CLI display.

```python
from clean_interfaces.models.io import WelcomeMessage

class WelcomeMessage(BaseModel):
    """Welcome message for CLI."""
    
    message: str = Field(
        default="Welcome to Clean Interfaces!",
        description="Main welcome message"
    )
    hint: str = Field(
        default="Type --help for more information",
        description="Usage hint"
    )
```

#### Fields

- `message` (str): Main welcome message
- `hint` (str): Usage hint for CLI users

#### Example

```python
from clean_interfaces.models.io import WelcomeMessage
import typer

welcome = WelcomeMessage()
typer.echo(welcome.message)
typer.echo(welcome.hint)
```

## Model Features

### Validation

All models validate input data:

```python
from clean_interfaces.models.api import ErrorResponse

# Valid
error = ErrorResponse(error="Something went wrong")

# Invalid - missing required field
try:
    error = ErrorResponse()
except ValidationError as e:
    print(e)
```

### Serialization

Models support various serialization formats:

```python
from clean_interfaces.models.api import HealthResponse

health = HealthResponse()

# To dictionary
data = health.model_dump()

# To JSON string
json_str = health.model_dump_json()

# To JSON with custom options
json_pretty = health.model_dump_json(indent=2)

# Exclude None values
json_clean = health.model_dump_json(exclude_none=True)
```

### Schema Generation

Models can generate JSON Schema:

```python
from clean_interfaces.models.api import WelcomeResponse

schema = WelcomeResponse.model_json_schema()
print(schema)
# {
#     "title": "WelcomeResponse",
#     "type": "object",
#     "properties": {...}
# }
```

## Custom Models

Creating custom models:

```python
from pydantic import BaseModel, Field
from datetime import datetime

class CustomRequest(BaseModel):
    """Custom request model."""
    
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "John Doe",
                    "age": 30,
                    "email": "john@example.com"
                }
            ]
        }
    )
```

## Integration with FastAPI

Models integrate seamlessly with FastAPI:

```python
from fastapi import FastAPI
from clean_interfaces.models.api import WelcomeResponse, ErrorResponse

app = FastAPI()

@app.get("/welcome", response_model=WelcomeResponse)
def get_welcome() -> WelcomeResponse:
    return WelcomeResponse(version="2.0.0")

@app.get(
    "/error",
    response_model=ErrorResponse,
    responses={404: {"model": ErrorResponse}}
)
def get_error() -> ErrorResponse:
    return ErrorResponse(error="Example error")
```

## Best Practices

1. **Use type hints**: Always specify field types
2. **Add descriptions**: Document fields with descriptions
3. **Validate data**: Use Field constraints for validation
4. **Provide examples**: Add examples in model_config
5. **Handle optionals**: Use `| None` for optional fields
6. **Exclude None**: Use `exclude_none=True` when serializing

## See Also

- [Interfaces Module](interfaces.md) - Uses models for API responses
- [Pydantic Documentation](https://docs.pydantic.dev/) - Detailed Pydantic features
- [API Reference Overview](overview.md) - Complete API documentation