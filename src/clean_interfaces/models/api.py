"""API response models for RestAPI interface."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(
        default="healthy",
        description="Health status of the API",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Current server timestamp",
    )


class WelcomeResponse(BaseModel):
    """Welcome message response model."""

    message: str = Field(
        default="Welcome to Clean Interfaces!",
        description="Welcome message",
    )
    hint: str = Field(
        default="Type --help for more information",
        description="Hint for users",
    )
    interface: str = Field(
        default="RestAPI",
        description="Current interface type",
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(
        description="Error message",
    )
    detail: str | None = Field(
        default=None,
        description="Detailed error information",
    )
    status_code: int = Field(
        description="HTTP status code",
    )


class SwaggerAnalysisResponse(BaseModel):
    """Swagger UI source code analysis response model."""

    interfaces: list[str] = Field(
        description="List of discovered interface classes",
    )
    models: list[str] = Field(
        description="List of discovered model classes",
    )
    endpoints: list[str] = Field(
        description="List of discovered API endpoints",
    )
    documentation_files: list[str] = Field(
        description="List of documentation files found",
    )
    summary: dict[str, int] = Field(
        description="Summary statistics of analysis",
    )


class DynamicContentMetadata(BaseModel):
    """Dynamic content generation metadata model."""

    source_files_analyzed: int = Field(
        description="Number of source files analyzed",
    )
    documentation_files_found: int = Field(
        description="Number of documentation files found",
    )
    interfaces_discovered: int = Field(
        description="Number of interfaces discovered",
    )
    models_discovered: int = Field(
        description="Number of models discovered",
    )
    endpoints_analyzed: int = Field(
        description="Number of endpoints analyzed",
    )
    generation_timestamp: str = Field(
        description="Timestamp of content generation",
    )
