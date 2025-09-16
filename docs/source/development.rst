Development Guide
=================

This guide covers development setup and contribution guidelines for Clean Interfaces.

Development Setup
-----------------

1. Clone the repository::

    git clone https://github.com/your-username/clean-interfaces.git
    cd clean-interfaces

2. Install development dependencies::

    uv sync --extra dev

3. Install pre-commit hooks::

    uv run pre-commit install

Development Workflow
--------------------

Running Tests
~~~~~~~~~~~~~

::

    # Run all tests
    nox -s test

    # Run specific test types
    nox -s test_unit
    nox -s test_api
    nox -s test_e2e

    # Run with coverage
    nox -s coverage

Code Quality
~~~~~~~~~~~~

::

    # Run linting
    nox -s lint

    # Format code
    nox -s format_code

    # Type checking
    nox -s typing

    # All CI checks
    nox -s ci

Contribution Guidelines
-----------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run all quality checks
5. Submit a pull request

Coding Standards
----------------

* Use Python 3.13+ type hints
* Follow PEP 8 style guide
* Write comprehensive tests
* Document all public APIs
* Use conventional commits
