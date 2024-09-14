# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../Core'))
sys.path.insert(0, os.path.abspath('../../DataProcessing'))
sys.path.insert(0, os.path.abspath('../../SystemSimulation'))
sys.path.insert(0, os.path.abspath('../../Test'))
import Core as Orion


project = Orion.__title__
copyright = Orion.__author__
author = Orion.__author__
version = Orion.__version__
release = version

html_theme_options = {
    'prev_next_buttons_location': None,
    'sticky_navigation': True,
    'navigation_depth': 6,
    'includehidden': True,
    'display_version': False,
}

# Project logo 
# html_logo = "_static\logo\logo2_logo.png"

html_show_copyright = True
html_show_sphinx = False
html_show_sourcelink = False
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'  # Order members by the order they appear in the source code

html_title = f"{project} {version}"
html_last_updated_fmt = '%b %d, %Y'
html_use_modindex = True
html_copy_source = False
html_domain_indices = False


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    # 'sphinx.ext.intersphinx', if linked to git lab to host multiple version's docs
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinxcontrib.mermaid',
    'sphinx_togglebutton'
]

html_context = {
  'current_version' : version,
  'GitHub' : [["Orion's GitHub", Orion.__github_url__]]
#   'versions' : [["1.0", "link to 1.0"], ["2.0", "link to 2.0"]],
#   'current_language': 'en',
#   'languages': [["en", "link to en"], ["de", "link to de"]]
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'renku'  
templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = ["mycss.css"]
html_js_files = ["myjs.js"]

def setup(app):
   app.add_css_file(r"sphinx\source\_static\mycss.css")
   app.add_js_file(r"sphinx\source\_static\myjs.js")
