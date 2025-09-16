Quick Start Guide
=================

This guide will help you get started with Clean Interfaces quickly.

Basic Usage
-----------

Running the CLI Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

The default interface is the command-line interface:

.. code-block:: bash

   # Run with default settings
   uv run python -m clean_interfaces.main

   # Run with custom environment file
   uv run python -m clean_interfaces.main --dotenv prod.env

Running the REST API Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the REST API interface:

.. code-block:: bash

   # Set interface type and run
   INTERFACE_TYPE=restapi uv run python -m clean_interfaces.main

   # Or use a custom .env file
   echo "INTERFACE_TYPE=restapi" > api.env
   uv run python -m clean_interfaces.main --dotenv api.env

Configuration Options
---------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Configure the application using environment variables:

.. code-block:: bash

   # Set logging level
   export LOG_LEVEL=DEBUG

   # Set log format
   export LOG_FORMAT=console

   # Set interface type
   export INTERFACE_TYPE=cli

Using .env Files
~~~~~~~~~~~~~~~~

Create a ``.env`` file for persistent configuration:

.. code-block:: ini

   # .env file
   INTERFACE_TYPE=cli
   LOG_LEVEL=INFO
   LOG_FORMAT=json
   # OpenTelemetry exporter was removed; OTEL_* variables are ignored

Multiple Environments
~~~~~~~~~~~~~~~~~~~~~

Use different configuration files for different environments:

.. code-block:: bash

   # Development
   uv run python -m clean_interfaces.main --dotenv dev.env

   # Production
   uv run python -m clean_interfaces.main --dotenv prod.env

   # Testing
   uv run python -m clean_interfaces.main --dotenv test.env

Example Configurations
----------------------

Development Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   # dev.env
   INTERFACE_TYPE=cli
   LOG_LEVEL=DEBUG
   LOG_FORMAT=console
   # OTEL_LOGS_EXPORT_MODE is ignored

Production Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   # prod.env
   INTERFACE_TYPE=restapi
   LOG_LEVEL=WARNING
   LOG_FORMAT=json
   LOG_FILE_PATH=/var/log/clean-interfaces/app.log
   # OpenTelemetry exporter removed

Logging Examples
----------------

The application supports multiple log formats:

JSON Format
~~~~~~~~~~~

.. code-block:: json

   {
     "timestamp": "2025-07-20T10:30:45.123Z",
     "level": "info",
     "logger": "clean_interfaces.app",
     "message": "Application started",
     "interface": "cli"
   }

Console Format
~~~~~~~~~~~~~~

.. code-block:: text

   2025-07-20 10:30:45 [INFO] clean_interfaces.app: Application started interface=cli

Next Steps
----------

* Explore the :doc:`api/index` for detailed API documentation
* Learn about :doc:`development` for contributing
* Check the configuration reference in the README
