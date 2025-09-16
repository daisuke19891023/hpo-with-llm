"""E2E tests for RestAPI interface functionality."""

import pytest
from fastapi.testclient import TestClient


class TestRestAPIInterfaceE2E:
    """E2E tests for RestAPI interface."""

    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        """Create test client with RestAPI interface."""
        # Set environment variable to use RestAPI interface
        monkeypatch.setenv("INTERFACE_TYPE", "restapi")

        # Import here to ensure environment variable is set
        from clean_interfaces.interfaces.factory import InterfaceFactory
        from clean_interfaces.interfaces.restapi import RestAPIInterface
        from clean_interfaces.types import InterfaceType

        factory = InterfaceFactory()
        interface = factory.create(InterfaceType.RESTAPI)

        # Type narrow to RestAPIInterface to access app attribute
        assert isinstance(interface, RestAPIInterface)

        # Get the FastAPI app from the interface
        return TestClient(interface.app)

    def test_health_endpoint(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_welcome_endpoint(self, client: TestClient) -> None:
        """Test welcome message endpoint."""
        response = client.get("/api/v1/welcome")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Welcome to Clean Interfaces!"
        assert data["hint"] == "Type --help for more information"
        assert data["interface"] == "RestAPI"

    def test_root_redirect(self, client: TestClient) -> None:
        """Test root path redirects to API documentation."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307  # Temporary redirect
        assert response.headers["location"] == "/docs"

    def test_api_documentation(self, client: TestClient) -> None:
        """Test API documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger-ui" in response.text.lower()

    def test_openapi_schema(self, client: TestClient) -> None:
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Clean Interfaces API"
        assert "/health" in schema["paths"]
        assert "/api/v1/welcome" in schema["paths"]
