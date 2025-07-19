# -- Path setup --------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'immuno-compass'
copyright = '2025, Wanxiang Shen'
author = 'Wanxiang Shen, Thinh H. Nguyen, Michelle M. Li, Yepeng Huang, Intae Moon, Nitya Nair, Daniel Marbach, and Marinka Zitnik'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
master_doc = "index"
source_suffix = ['.rst', '.md']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon',
    'nbsphinx',
    # 'sphinx_gallery.gen_gallery',
    # 'bokeh.sphinxext.bokeh_plot',
    # 'myst_parser',   
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("http://scikit-learn.org/stable/", None),
}

templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": -1,
    "collapse_navigation": False,
    'globaltoc_collapse': False,
    'globaltoc_maxdepth': -1,
    "logo_only": True,
}
html_logo = "../images/compass_logo.png"
html_static_path = ['_static']
