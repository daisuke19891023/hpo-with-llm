"""Tests for RestAPI interface implementation."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from clean_interfaces.interfaces.base import BaseInterface
from clean_interfaces.interfaces.restapi import RestAPIInterface


class TestRestAPIInterface:
    """Test RestAPI interface functionality."""

    def test_restapi_interface_inherits_base(self) -> None:
        """Test that RestAPIInterface inherits from BaseInterface."""
        assert issubclass(RestAPIInterface, BaseInterface)

    def test_restapi_interface_has_name(self) -> None:
        """Test that RestAPIInterface has correct name."""
        api = RestAPIInterface()
        assert api.name == "RestAPI"

    def test_restapi_interface_has_fastapi_app(self) -> None:
        """Test that RestAPIInterface has FastAPI app."""
        api = RestAPIInterface()
        assert hasattr(api, "app")
        assert isinstance(api.app, FastAPI)

    def test_restapi_interface_app_title(self) -> None:
        """Test that FastAPI app has correct title."""
        api = RestAPIInterface()
        assert api.app.title == "Clean Interfaces API"
        assert api.app.version == "1.0.0"

    def test_restapi_interface_has_endpoints(self) -> None:
        """Test that RestAPIInterface has required endpoints."""
        api = RestAPIInterface()
        routes = [route.path for route in api.app.routes]  # type: ignore[attr-defined]
        assert "/health" in routes
        assert "/api/v1/welcome" in routes
        assert "/" in routes  # Root redirect

    @patch("clean_interfaces.interfaces.restapi.uvicorn")
    def test_restapi_run_method(self, mock_uvicorn: MagicMock) -> None:
        """Test RestAPI run method configures uvicorn."""
        api = RestAPIInterface()

        # Mock to prevent actual server start
        api.run()

        # Verify uvicorn.run was called with correct parameters
        mock_uvicorn.run.assert_called_once()
        call_args = mock_uvicorn.run.call_args

        # Check app parameter
        assert call_args[0][0] == api.app

        # Check kwargs
        kwargs = call_args[1]
        assert kwargs.get("host") == "0.0.0.0"  # noqa: S104
        assert kwargs.get("port") == 8000
        assert kwargs.get("log_config") is not None

    def test_restapi_interface_initialization_logs(self) -> None:
        """Test that RestAPIInterface logs initialization."""
        with patch("clean_interfaces.base.get_logger") as mock_get_logger:
            # Setup mock logger
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            # Create instance which will trigger logging
            RestAPIInterface()

            # Check that logger was used during initialization
            assert mock_logger.info.called


class TestRestAPISwaggerUIEndpoints:
    """Test Swagger UI endpoints in RestAPI interface."""

    def test_swagger_ui_endpoint_exists(self) -> None:
        """Test that Swagger UI endpoint is registered."""
        api = RestAPIInterface()
        routes = [route.path for route in api.app.routes]  # type: ignore[attr-defined]
        assert "/api/v1/swagger-ui" in routes

    def test_swagger_ui_schema_endpoint_exists(self) -> None:
        """Test that Swagger UI schema endpoint is registered."""
        api = RestAPIInterface()
        routes = [route.path for route in api.app.routes]  # type: ignore[attr-defined]
        assert "/api/v1/swagger-ui/schema" in routes

    def test_swagger_ui_analysis_endpoint_exists(self) -> None:
        """Test that Swagger UI analysis endpoint is registered."""
        api = RestAPIInterface()
        routes = [route.path for route in api.app.routes]  # type: ignore[attr-defined]
        assert "/api/v1/swagger-ui/analysis" in routes

    def test_swagger_ui_returns_html(self) -> None:
        """Test that Swagger UI endpoint returns HTML response."""
        api = RestAPIInterface()
        client = TestClient(api.app)

        response = client.get("/api/v1/swagger-ui")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    def test_swagger_ui_schema_returns_json(self) -> None:
        """Test that Swagger UI schema endpoint returns JSON response."""
        api = RestAPIInterface()
        client = TestClient(api.app)

        response = client.get("/api/v1/swagger-ui/schema")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        schema = response.json()
        assert isinstance(schema, dict)
        assert "info" in schema

    def test_swagger_ui_analysis_returns_json(self) -> None:
        """Test that Swagger UI analysis endpoint returns JSON response."""
        api = RestAPIInterface()
        client = TestClient(api.app)

        response = client.get("/api/v1/swagger-ui/analysis")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        analysis = response.json()
        assert isinstance(analysis, dict)
        assert "interfaces" in analysis
        assert "models" in analysis

    def test_swagger_ui_schema_includes_dynamic_content_metadata(self) -> None:
        """Test that schema includes dynamic content generation metadata."""
        api = RestAPIInterface()
        client = TestClient(api.app)

        response = client.get("/api/v1/swagger-ui/schema")
        schema = response.json()

        assert "info" in schema
        assert "dynamic_content" in schema["info"]
        assert "source_files_analyzed" in schema["info"]["dynamic_content"]
        assert "documentation_files_found" in schema["info"]["dynamic_content"]

    def test_swagger_ui_analysis_includes_interface_information(self) -> None:
        """Test that analysis includes interface type information."""
        api = RestAPIInterface()
        client = TestClient(api.app)

        response = client.get("/api/v1/swagger-ui/analysis")
        analysis = response.json()

        assert "interfaces" in analysis
        assert isinstance(analysis["interfaces"], list)
        assert len(analysis["interfaces"]) > 0
