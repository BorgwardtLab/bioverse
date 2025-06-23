import datetime
import importlib
import os
import sys

from sphinxawesome_theme.postprocess import Icons

# Add the path to your project's root directory
sys.path.insert(0, os.path.abspath("../.."))

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
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["mode.js"]
html_permalinks_icon = Icons.permalinks_icon

bioverse = importlib.import_module("bioverse")


def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {"bioverse": {"adapters": {"classes": bioverse.adapters.__all__}}}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect("source-read", rst_jinja_render)
