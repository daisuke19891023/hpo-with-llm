"""Tests for API models."""

from datetime import UTC, datetime

from clean_interfaces.models.api import ErrorResponse, HealthResponse, WelcomeResponse


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_health_response_defaults(self) -> None:
        """Test HealthResponse with default values."""
        response = HealthResponse()
        assert response.status == "healthy"
        assert isinstance(response.timestamp, datetime)

    def test_health_response_custom_values(self) -> None:
        """Test HealthResponse with custom values."""
        timestamp = datetime.now(tz=UTC)
        response = HealthResponse(status="degraded", timestamp=timestamp)
        assert response.status == "degraded"
        assert response.timestamp == timestamp


class TestWelcomeResponse:
    """Test WelcomeResponse model."""

    def test_welcome_response_defaults(self) -> None:
        """Test WelcomeResponse with default values."""
        response = WelcomeResponse()
        assert response.message == "Welcome to Clean Interfaces!"
        assert response.hint == "Type --help for more information"
        assert response.interface == "RestAPI"

    def test_welcome_response_custom_values(self) -> None:
        """Test WelcomeResponse with custom values."""
        response = WelcomeResponse(
            message="Custom welcome",
            hint="Custom hint",
            interface="API",
        )
        assert response.message == "Custom welcome"
        assert response.hint == "Custom hint"
        assert response.interface == "API"


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_error_response_minimal(self) -> None:
        """Test ErrorResponse with minimal values."""
        response = ErrorResponse(error="Not found", status_code=404)
        assert response.error == "Not found"
        assert response.status_code == 404
        assert response.detail is None

    def test_error_response_with_detail(self) -> None:
        """Test ErrorResponse with detail."""
        response = ErrorResponse(
            error="Validation error",
            detail="Invalid input format",
            status_code=400,
        )
        assert response.error == "Validation error"
        assert response.detail == "Invalid input format"
        assert response.status_code == 400
