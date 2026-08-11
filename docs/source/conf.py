import datetime
import importlib
import os
import sys

from sphinxawesome_theme.postprocess import Icons
from sphinx.util import logging

# Add the path to your project's root directory and local extensions
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("_ext"))

from implementations import build_docs_context, flatten_implementation_toc

logger = logging.getLogger(__name__)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Bioverse"
copyright = f"{datetime.datetime.now().year}, Bioverse Contributors."
author = "Tim Kucera"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Enable automatic generation of API docs
autosummary_generate = True
autosummary_imported_members = True

# Add autodoc settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True


# Add this to exclude certain patterns from warnings
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_autosummary/*",  # This will exclude autosummary generated files from warnings
]

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_title = "Bioverse"
html_short_title = "Bioverse"

_docs_baseurl = os.environ.get("BIOVERSE_DOCS_BASEURL", "")
if not _docs_baseurl and os.environ.get("GITHUB_ACTIONS") == "true":
    _docs_baseurl = "https://borgwardtlab.github.io/bioverse/"
if _docs_baseurl:
    html_baseurl = _docs_baseurl

_paper_url = os.environ.get(
    "BIOVERSE_PAPER_URL",
    f"{_docs_baseurl.rstrip('/')}/citation.html" if _docs_baseurl else "citation.html",
)

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["mode.js"]
html_permalinks_icon = Icons.permalinks_icon

html_theme_options = {
    "awesome_external_links": True,
    "main_nav_links": {
        "Paper": _paper_url,
        "GitHub": "https://github.com/BorgwardtLab/bioverse",
    },
}

importlib.import_module("bioverse")


def setup(app):
    docs_context = build_docs_context()
    app.config.bioverse_docs_context = docs_context

    for category, data in docs_context.items():
        if "classes" in data:
            logger.info(
                "Discovered %d %s implementations for documentation.",
                len(data["classes"]),
                category,
            )
        elif "configs" in data:
            logger.info(
                "Discovered %d %s configurations for documentation.",
                len(data["configs"]),
                category,
            )

    def rst_jinja_render(app, docname, source):
        if "{%" not in source[0]:
            return
        source[0] = app.builder.templates.render_string(
            source[0],
            {"bioverse": app.config.bioverse_docs_context},
        )

    app.connect("source-read", rst_jinja_render)
    app.connect("build-finished", flatten_implementation_toc, priority=100)
