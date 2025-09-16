Overview
========

Clean Interfaces is a flexible Python application framework designed to provide multiple interface types (CLI, REST API) with a clean, extensible architecture.

Architecture
------------

The project follows a modular architecture with clear separation of concerns:

.. code-block:: text

   src/clean_interfaces/
   ├── interfaces/         # Interface implementations
   │   ├── base.py        # Base interface class
   │   ├── cli.py         # CLI interface using Typer
   │   ├── factory.py     # Interface factory pattern
   │   └── restapi.py     # REST API interface using FastAPI
   ├── models/            # Data models
   ├── utils/             # Utility modules
   │   ├── logger.py      # Structured logging
   │   └── settings.py    # Configuration management
   └── app.py            # Application entry point

Key Components
--------------

Interface System
~~~~~~~~~~~~~~~~

The interface system uses a factory pattern to create different interface types:

* **CLI Interface**: Built with Typer for rich command-line interactions
* **REST API Interface**: Built with FastAPI for high-performance web APIs

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

All configuration is managed through environment variables and Pydantic settings:

* Environment-based configuration
* Type validation and coercion
* Support for ``.env`` files
* Custom dotenv file support via ``--dotenv`` option

Logging System
~~~~~~~~~~~~~~

Comprehensive logging with multiple output formats:

* Structured JSON logging for production
* Colored console output for development
* OpenTelemetry integration for observability
* File and OTLP export options

Design Principles
-----------------

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Dependency Injection**: Components are loosely coupled through interfaces
3. **Configuration as Code**: All settings are validated and typed
4. **Test-Driven Development**: Comprehensive test coverage at all levels
5. **Type Safety**: Full type hints with strict checking