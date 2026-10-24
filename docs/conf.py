"""Sphinx configuration for the maxkcut research package."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "maxkcut"
copyright = "2026, maxkcut contributors"
author = "maxkcut contributors"
release = "0.1.0"
language = "en"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3
html_theme = "furo"
html_title = "maxkcut | Research documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563b8",
        "color-brand-content": "#2563b8",
        "color-background-primary": "#ffffff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#75a9f7",
        "color-brand-content": "#75a9f7",
    },
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
