"""Sphinx configuration file for Clean Interfaces documentation."""

import sys
from datetime import UTC, datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Project information
project = "Clean Interfaces"
copyright = f"{datetime.now(tz=UTC).year}, Daisuke Okamoto"  # noqa: A001
author = "Daisuke Okamoto"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []

# HTML output options
html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "description": "A flexible Python application framework "
    "with multiple interface types",
    "github_user": "your-username",
    "github_repo": "clean-interfaces",
    "fixed_sidebar": True,
}

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Napoleon configuration (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
}

# Coverage extension configuration
coverage_skip_undoc_in_source = True
