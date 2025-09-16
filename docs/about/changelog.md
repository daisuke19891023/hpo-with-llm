# Changelog

All notable changes to Clean Interfaces will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with `setup.sh` script
- CLI interface using Typer
- REST API interface using FastAPI
- Structured logging with structlog
- OpenTelemetry integration for log export
- File handling utilities with encoding support
- Comprehensive test suite with >80% coverage
- Type hints throughout the codebase
- Documentation with MkDocs
- Pre-commit hooks for code quality
- GitHub Actions for CI/CD
- Support for Python 3.12+

### Changed
- Migrated from poetry to uv for dependency management
- Updated all dependencies to latest stable versions

### Fixed
- ANSI escape code handling in CLI tests for CI compatibility
- Type checking errors in pexpect debug helper

## [0.1.0] - 2025-01-20

### Added
- Initial release of Clean Interfaces
- Multiple interface support (CLI and REST API)
- Flexible configuration via environment variables
- Structured logging with multiple output formats
- OpenTelemetry export capabilities
- Comprehensive testing framework
- Full type hint coverage
- Detailed documentation

## Version History

### Versioning Strategy

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Process

1. Update version in `pyproject.toml`
2. Update this CHANGELOG.md
3. Create git tag: `git tag -a v0.1.0 -m "Release version 0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions automatically publishes to PyPI

### Deprecation Policy

- Deprecated features will be marked with warnings for at least one minor version
- Breaking changes will only occur in major version updates
- Migration guides will be provided for breaking changes

## Contributing

See [Contributing Guide](../development/contributing.md) for how to contribute changes.

[Unreleased]: https://github.com/your-username/clean-interfaces/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/clean-interfaces/releases/tag/v0.1.0