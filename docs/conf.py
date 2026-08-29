"""Sphinx configuration for skyplot."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

project = "SkyPlot"
author = "Shamik Ghosh"
copyright = f"{datetime.now().year}, {author}"
release = (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip()

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
