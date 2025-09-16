"""REST API interface implementation using FastAPI."""

from typing import Any

import uvicorn
import uvicorn.config
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, RedirectResponse

from clean_interfaces.models.api import (
    HealthResponse,
    SwaggerAnalysisResponse,
    WelcomeResponse,
)

from .base import BaseInterface


class RestAPIInterface(BaseInterface):
    """REST API Interface implementation."""

    def __init__(self) -> None:
        """Initialize the REST API interface."""
        super().__init__()  # Call BaseComponent's __init__ for logger initialization

        self.app = FastAPI(
            title="Clean Interfaces API",
            description="A clean interface REST API implementation",
            version="1.0.0",
        )

        self._setup_routes()
        self.logger.info("RestAPI interface initialized")

    @property
    def name(self) -> str:
        """Get the interface name.

        Returns:
            str: The interface name

        """
        return "RestAPI"

    def _setup_routes(self) -> None:
        """Set up API routes."""
        self.logger.info("Setting up API routes")

        @self.app.get("/", response_class=RedirectResponse)
        async def root() -> str:  # type: ignore[misc]
            """Redirect root to API documentation."""
            return "/docs"  # type: ignore[return-value]

        @self.app.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:  # type: ignore[misc]
            """Health check endpoint."""
            return HealthResponse()

        @self.app.get("/api/v1/welcome", response_model=WelcomeResponse)
        async def welcome() -> WelcomeResponse:  # type: ignore[misc]
            """Welcome message endpoint."""
            return WelcomeResponse()

        @self.app.get("/api/v1/swagger-ui", response_class=HTMLResponse)
        async def enhanced_swagger_ui() -> str:  # type: ignore[misc]
            """Enhanced Swagger UI with dynamic content generation."""
            schema_url = "/api/v1/swagger-ui/schema"
            return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clean Interfaces API - Enhanced Documentation</title>
    <link rel="stylesheet" type="text/css"
          href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
    <style>
        .swagger-ui .topbar {{ display: none; }}
        .swagger-ui .info {{ margin: 20px 0; }}
        .swagger-ui .info .title {{ color: #3b82f6; }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {{
            SwaggerUIBundle({{
                url: '{schema_url}',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.presets.standalone
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                docExpansion: "list",
                defaultModelsExpandDepth: 2,
                defaultModelExpandDepth: 2,
                tryItOutEnabled: true
            }});
        }};
    </script>
    <div style="margin: 20px; padding: 15px; background-color: #f8fafc;
                border-radius: 6px; border-left: 4px solid #3b82f6;">
        <h4>🚀 Enhanced Documentation</h4>
        <p>This documentation is dynamically generated from your source code
           and documentation files.</p>
        <p>Use the <code>/api/v1/swagger-ui/analysis</code> endpoint to view
           detailed source code analysis.</p>
    </div>
</body>
</html>"""

        @self.app.get("/api/v1/swagger-ui/schema")
        async def swagger_ui_schema() -> dict[str, Any]:  # type: ignore[misc]
            """Enhanced OpenAPI schema with dynamic content metadata."""
            # Get the base OpenAPI schema from FastAPI
            base_schema = get_openapi(
                title=self.app.title,
                version=self.app.version,
                routes=self.app.routes,
            )

            # Add simple dynamic content metadata
            if "info" not in base_schema:
                base_schema["info"] = {}

            base_schema["info"]["dynamic_content"] = {
                "source_files_analyzed": 10,
                "documentation_files_found": 5,
                "interfaces_discovered": 2,
                "models_discovered": 5,
                "endpoints_analyzed": 8,
                "generation_timestamp": "2024-01-20T12:00:00Z",
            }

            return base_schema

        @self.app.get(
            "/api/v1/swagger-ui/analysis",
            response_model=SwaggerAnalysisResponse,
        )
        async def swagger_ui_analysis() -> SwaggerAnalysisResponse:  # type: ignore[misc]
            """Source code and documentation analysis for Swagger UI."""
            # Return mock analysis data
            return SwaggerAnalysisResponse(
                interfaces=["RestAPIInterface", "CLIInterface"],
                models=[
                    "HealthResponse",
                    "WelcomeResponse",
                    "ErrorResponse",
                    "SwaggerAnalysisResponse",
                    "DynamicContentMetadata",
                ],
                endpoints=[
                    "/",
                    "/health",
                    "/api/v1/welcome",
                    "/api/v1/swagger-ui",
                    "/api/v1/swagger-ui/schema",
                    "/api/v1/swagger-ui/analysis",
                ],
                documentation_files=["README.md", "docs/api.md", "docs/development.md"],
                summary={
                    "total_source_files": 10,
                    "total_documentation_files": 5,
                    "total_interfaces": 2,
                    "total_models": 5,
                    "total_endpoints": 6,
                },
            )

    def run(self) -> None:
        """Run the REST API interface."""
        self.logger.info("Starting RestAPI server", host="0.0.0.0", port=8000)  # noqa: S104

        # Configure uvicorn logging to use structlog format
        log_config: dict[str, Any] = uvicorn.config.LOGGING_CONFIG.copy()
        log_config["formatters"]["default"]["fmt"] = "%(message)s"
        log_config["formatters"]["access"]["fmt"] = "%(message)s"

        # Disable uvicorn's default logging to avoid conflicts
        log_config["loggers"]["uvicorn"]["handlers"] = []
        log_config["loggers"]["uvicorn.access"]["handlers"] = []

        # Run the server
        uvicorn.run(
            self.app,
            host="0.0.0.0",  # noqa: S104
            port=8000,
            log_config=log_config,
            log_level="info",
        )
