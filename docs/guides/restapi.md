# REST API Interface Guide

The REST API interface provides HTTP endpoints for interacting with the Clean Interfaces application.

## Overview

The REST API interface is built using FastAPI and provides a RESTful web service with automatic API documentation.

## Running the REST API

To run the application with the REST API interface:

```bash
# Set the interface type
export INTERFACE_TYPE=restapi

# Run the application
python -m clean_interfaces.main

# Or use uvicorn directly
uvicorn clean_interfaces.main:app --reload
```

## Available Endpoints

### Root Endpoint

- **GET `/`** - Redirects to the API documentation

### Health Check

- **GET `/health`** - Returns the health status of the application

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-20T12:00:00Z"
}
```

### Welcome Endpoint

- **GET `/api/v1/welcome`** - Returns a welcome message

```bash
curl http://localhost:8000/api/v1/welcome
```

Response:
```json
{
  "message": "Welcome to Clean Interfaces!",
  "hint": "Type --help for more information",
  "version": "0.1.0"
}
```

## API Documentation

The REST API automatically generates interactive documentation:

- **Swagger UI**: Available at `http://localhost:8000/docs`
- **ReDoc**: Available at `http://localhost:8000/redoc`

## Configuration

The REST API can be configured using environment variables:

- `HOST`: Host to bind to (default: "0.0.0.0")
- `PORT`: Port to bind to (default: 8000)
- `LOG_LEVEL`: Logging level (default: "INFO")

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses follow a consistent format:

```json
{
  "error": "Resource not found",
  "detail": "The requested endpoint does not exist"
}
```

## Authentication

Currently, the REST API does not implement authentication. This can be added as needed for your specific use case.

## Next Steps

- Explore the [API Reference](../api/interfaces.md) for detailed interface documentation
- Learn about [Logging](logging.md) configuration
- Configure [Environment Variables](environment.md)