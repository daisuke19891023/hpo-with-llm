Installation
============

This guide covers the installation of Clean Interfaces.

Requirements
------------

* Python 3.13 or higher
* uv (Python package manager)

Installing uv
~~~~~~~~~~~~~

If you don't have uv installed, you can install it using:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Or on Windows:

.. code-block:: powershell

   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

Installation Steps
------------------

From Source
~~~~~~~~~~~

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-username/clean-interfaces.git
      cd clean-interfaces

2. Create virtual environment and install dependencies:

   .. code-block:: bash

      uv sync

3. Install with optional dependencies:

   .. code-block:: bash

      # Install with documentation tools
      uv sync --extra docs

      # Install with development tools
      uv sync --extra dev

      # Install all extras
      uv sync --all-extras

From PyPI
~~~~~~~~~

Once published to PyPI, you can install using:

.. code-block:: bash

   uv pip install clean-interfaces

   # Or with extras
   uv pip install "clean-interfaces[docs]"

Configuration
-------------

1. Copy the example configuration:

   .. code-block:: bash

      cp .env.example .env

2. Edit ``.env`` with your configuration:

   .. code-block:: bash

      # Example configuration
      INTERFACE_TYPE=cli
      LOG_LEVEL=INFO
      LOG_FORMAT=console

Verification
------------

Verify the installation by running:

.. code-block:: bash

   # Show help
   uv run python -m clean_interfaces.main --help

   # Run the application
   uv run python -m clean_interfaces.main

Next Steps
----------

* Read the :doc:`quickstart` guide
* Explore the :doc:`api/index`
* Learn about :doc:`development`
