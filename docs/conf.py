"""Sphinx configuration for skyplot."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = "SkyPlot"
author = "Shamik Ghosh"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "shibuya"
html_static_path = ["_static"]

autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
