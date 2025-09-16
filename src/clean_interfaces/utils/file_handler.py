"""File handler utility module for reading and writing various file formats.

This module provides a comprehensive FileHandler class and convenience functions
for handling files with different encodings (UTF-8, CP932) and formats (JSON, YAML).
It includes proper error handling with structured logging integration.
"""

import json
from pathlib import Path
from typing import Any, Literal

import yaml

from clean_interfaces.utils.logger import get_logger

logger = get_logger(__name__)

EncodingType = Literal["utf-8", "cp932", "shift_jis"]


class FileHandler:
    """Generic file handler with support for multiple encodings and formats."""

    def __init__(self, encoding: EncodingType = "utf-8") -> None:
        """Initialize FileHandler with default encoding.

        Args:
            encoding: Default encoding to use for file operations.
                     Options: "utf-8", "cp932", "shift_jis"

        """
        self.encoding = encoding
        logger.debug("FileHandler initialized", encoding=encoding)

    def read_text(
        self,
        path: str | Path,
        encoding: EncodingType | None = None,
    ) -> str:
        """Read text file with specified encoding.

        Args:
            path: Path to the file to read
            encoding: File encoding (uses instance default if not specified)

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If the file doesn't exist
            UnicodeDecodeError: If the file cannot be decoded with specified encoding
            IOError: For other I/O related errors

        """
        path = Path(path)
        use_encoding = encoding or self.encoding

        logger.debug(
            "Reading text file",
            path=str(path),
            encoding=use_encoding,
        )

        try:
            with path.open("r", encoding=use_encoding) as f:
                content = f.read()
            logger.info(
                "Text file read successfully",
                path=str(path),
                size=len(content),
                encoding=use_encoding,
            )
            return content
        except FileNotFoundError:
            logger.error("File not found", path=str(path))
            raise
        except UnicodeDecodeError as e:
            logger.error(
                "Failed to decode file",
                path=str(path),
                encoding=use_encoding,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "Error reading file",
                path=str(path),
                error=str(e),
            )
            raise
        else:
            return content

    def write_text(
        self,
        path: str | Path,
        content: str,
        encoding: EncodingType | None = None,
        create_parents: bool = True,
    ) -> None:
        """Write text to file with specified encoding.

        Args:
            path: Path to write the file to
            content: Content to write
            encoding: File encoding (uses instance default if not specified)
            create_parents: Whether to create parent directories if they don't exist

        Raises:
            UnicodeEncodeError: If the content cannot be encoded with specified encoding
            IOError: For other I/O related errors

        """
        path = Path(path)
        use_encoding = encoding or self.encoding

        logger.debug(
            "Writing text file",
            path=str(path),
            encoding=use_encoding,
            content_length=len(content),
        )

        try:
            if create_parents and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug("Created parent directories", path=str(path.parent))

            with path.open("w", encoding=use_encoding) as f:
                f.write(content)

            logger.info(
                "Text file written successfully",
                path=str(path),
                size=len(content),
                encoding=use_encoding,
            )
        except UnicodeEncodeError as e:
            logger.error(
                "Failed to encode content",
                path=str(path),
                encoding=use_encoding,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "Error writing file",
                path=str(path),
                error=str(e),
            )
            raise

    def read_json(
        self,
        path: str | Path,
        encoding: EncodingType | None = None,
    ) -> Any:
        """Read JSON file and return parsed content.

        Args:
            path: Path to the JSON file
            encoding: File encoding (uses instance default if not specified)

        Returns:
            Parsed JSON content as Python object

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            UnicodeDecodeError: If the file cannot be decoded with specified encoding

        """
        path = Path(path)
        use_encoding = encoding or self.encoding

        logger.debug("Reading JSON file", path=str(path), encoding=use_encoding)

        try:
            content = self.read_text(path, encoding=use_encoding)  # type: ignore[arg-type]
            data = json.loads(content)
            logger.info(
                "JSON file parsed successfully",
                path=str(path),
                data_type=type(data).__name__,
            )
        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON format",
                path=str(path),
                error=str(e),
                line=e.lineno,
                column=e.colno,
            )
            raise
        else:
            return data

    def write_json(
        self,
        path: str | Path,
        data: Any,
        encoding: EncodingType | None = None,
        indent: int = 2,
        ensure_ascii: bool = False,
        sort_keys: bool = False,
        create_parents: bool = True,
    ) -> None:
        """Write data to JSON file.

        Args:
            path: Path to write the JSON file to
            data: Data to serialize to JSON
            encoding: File encoding (uses instance default if not specified)
            indent: Number of spaces for indentation (None for compact output)
            ensure_ascii: Whether to escape non-ASCII characters
            sort_keys: Whether to sort dictionary keys
            create_parents: Whether to create parent directories if they don't exist

        Raises:
            TypeError: If the data cannot be serialized to JSON
            IOError: For I/O related errors

        """
        path = Path(path)
        use_encoding = encoding or self.encoding

        logger.debug(
            "Writing JSON file",
            path=str(path),
            encoding=use_encoding,
            data_type=type(data).__name__,
        )

        try:
            json_content = json.dumps(
                data,
                indent=indent,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
            )
            self.write_text(
                path,
                json_content,
                encoding=(
                    use_encoding
                    if use_encoding in ["utf-8", "cp932", "shift_jis"]
                    else "utf-8"
                ),  # type: ignore[arg-type]
                create_parents=create_parents,
            )
            logger.info("JSON file written successfully", path=str(path))
        except TypeError as e:
            logger.error(
                "Failed to serialize data to JSON",
                path=str(path),
                data_type=type(data).__name__,
                error=str(e),
            )
            raise

    def read_yaml(
        self,
        path: str | Path,
        encoding: EncodingType | None = None,
    ) -> Any:
        """Read YAML file and return parsed content.

        Args:
            path: Path to the YAML file
            encoding: File encoding (uses instance default if not specified)

        Returns:
            Parsed YAML content as Python object

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file contains invalid YAML
            UnicodeDecodeError: If the file cannot be decoded with specified encoding

        """
        path = Path(path)
        use_encoding = encoding or self.encoding

        logger.debug("Reading YAML file", path=str(path), encoding=use_encoding)

        try:
            content = self.read_text(path, encoding=use_encoding)  # type: ignore[arg-type]
            data = yaml.safe_load(content)
            logger.info(
                "YAML file parsed successfully",
                path=str(path),
                data_type=type(data).__name__,
            )
        except yaml.YAMLError as e:
            logger.error(
                "Invalid YAML format",
                path=str(path),
                error=str(e),
            )
            raise
        else:
            return data

    def write_yaml(
        self,
        path: str | Path,
        data: Any,
        encoding: EncodingType | None = None,
        default_flow_style: bool = False,
        sort_keys: bool = False,
        create_parents: bool = True,
    ) -> None:
        """Write data to YAML file.

        Args:
            path: Path to write the YAML file to
            data: Data to serialize to YAML
            encoding: File encoding (uses instance default if not specified)
            default_flow_style: Use flow style (False for block style)
            sort_keys: Whether to sort dictionary keys
            create_parents: Whether to create parent directories if they don't exist

        Raises:
            yaml.YAMLError: If the data cannot be serialized to YAML
            IOError: For I/O related errors

        """
        path = Path(path)
        use_encoding = encoding or self.encoding

        logger.debug(
            "Writing YAML file",
            path=str(path),
            encoding=use_encoding,
            data_type=type(data).__name__,
        )

        try:
            yaml_content = yaml.dump(
                data,
                default_flow_style=default_flow_style,
                sort_keys=sort_keys,
                allow_unicode=True,
            )
            self.write_text(
                path,
                yaml_content,
                encoding=(
                    use_encoding
                    if use_encoding in ["utf-8", "cp932", "shift_jis"]
                    else "utf-8"
                ),  # type: ignore[arg-type]
                create_parents=create_parents,
            )
            logger.info("YAML file written successfully", path=str(path))
        except yaml.YAMLError as e:
            logger.error(
                "Failed to serialize data to YAML",
                path=str(path),
                data_type=type(data).__name__,
                error=str(e),
            )
            raise


# Convenience functions using default UTF-8 encoding
_default_handler = FileHandler(encoding="utf-8")


def read_text(path: str | Path, encoding: EncodingType = "utf-8") -> str:
    """Read text file with specified encoding.

    Args:
        path: Path to the file to read
        encoding: File encoding (default: utf-8)

    Returns:
        File contents as string

    """
    return _default_handler.read_text(path, encoding)


def write_text(
    path: str | Path,
    content: str,
    encoding: EncodingType = "utf-8",
    create_parents: bool = True,
) -> None:
    """Write text to file with specified encoding.

    Args:
        path: Path to write the file to
        content: Content to write
        encoding: File encoding (default: utf-8)
        create_parents: Whether to create parent directories if they don't exist

    """
    _default_handler.write_text(path, content, encoding, create_parents)


def read_json(path: str | Path, encoding: EncodingType = "utf-8") -> Any:
    """Read JSON file and return parsed content.

    Args:
        path: Path to the JSON file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed JSON content as Python object

    """
    return _default_handler.read_json(path, encoding)


def write_json(
    path: str | Path,
    data: Any,
    encoding: EncodingType = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    create_parents: bool = True,
) -> None:
    """Write data to JSON file.

    Args:
        path: Path to write the JSON file to
        data: Data to serialize to JSON
        encoding: File encoding (default: utf-8)
        indent: Number of spaces for indentation
        ensure_ascii: Whether to escape non-ASCII characters
        sort_keys: Whether to sort dictionary keys
        create_parents: Whether to create parent directories if they don't exist

    """
    _default_handler.write_json(
        path,
        data,
        encoding,
        indent,
        ensure_ascii,
        sort_keys,
        create_parents,
    )


def read_yaml(path: str | Path, encoding: EncodingType = "utf-8") -> Any:
    """Read YAML file and return parsed content.

    Args:
        path: Path to the YAML file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed YAML content as Python object

    """
    return _default_handler.read_yaml(path, encoding)


def write_yaml(
    path: str | Path,
    data: Any,
    encoding: EncodingType = "utf-8",
    default_flow_style: bool = False,
    sort_keys: bool = False,
    create_parents: bool = True,
) -> None:
    """Write data to YAML file.

    Args:
        path: Path to write the YAML file to
        data: Data to serialize to YAML
        encoding: File encoding (default: utf-8)
        default_flow_style: Use flow style (False for block style)
        sort_keys: Whether to sort dictionary keys
        create_parents: Whether to create parent directories if they don't exist

    """
    _default_handler.write_yaml(
        path,
        data,
        encoding,
        default_flow_style,
        sort_keys,
        create_parents,
    )
