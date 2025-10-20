# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CODS"
copyright = "2025, Léo Andéol, Luca Mossina"
author = "Léo Andéol, Luca Mossina"

version = "0.3"
release = "0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# Napoleon for Google/NumPy docstrings
extensions.append("sphinx.ext.napoleon")
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# Napoleon for Google/NumPy docstrings
extensions.append("sphinx.ext.napoleon")
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
