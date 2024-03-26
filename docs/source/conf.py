# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../"))


project = 'aquapointer'
copyright = '2024, Unitary Fund, Pasqal, and Qubit Pharmaceuticals'
author = 'Unitary Fund, Pasqal, and Qubit Pharmaceuticals'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


sys.path.append(os.path.abspath("sphinxext"))
extensions = [ "sphinx.ext.autodoc",
            "sphinx.ext.autosummary",
            "sphinx_autodoc_typehints"
            ]

master_doc = "index"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
