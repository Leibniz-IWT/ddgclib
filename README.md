[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8010952.svg)](https://doi.org/10.5281/zenodo.8010952)

# ddgclib
Experimental library for discrete differential geometry curvature in fluid simulations, especially useful for problems with complex and changing topologies.

# Installation

## Conda (recommended)

Create a conda environment called `ddg` with all dependencies:

```bash
conda env create -f environment.yml
conda activate ddg
pip install -e .
```

To update an existing `ddg` environment after pulling new changes:

```bash
conda env update -f environment.yml --prune
```

## pip only

```bash
pip install -e .
```

## Dependencies

**Required:** `numpy`, `scipy`, [`hyperct`](https://github.com/stefan-endres/hyperct)

**Optional:**
- Visualization: `polyscope`, `matplotlib` — install with `pip install -e ".[vis]"`
- Data: `pandas` — install with `pip install -e ".[data]"`
- GPU acceleration: `torch` — install with `pip install -e ".[gpu]"`
- Development: `pytest`, `pytest-cov` — install with `pip install -e ".[dev]"`

Python 3.9+ required (development uses Python 3.13).

# Running Tests

```bash
# Run all tests
pytest ddgclib/tests/ -v

# Run fast tests only (skip slow convergence studies)
pytest ddgclib/tests/ -v -m "not slow"

# Run manuscript tutorial tests only
pytest ddgclib/tests/test_manuscript_tutorials.py -v
```

PyCharm run configurations are included in `.idea/runConfigurations/`.

# Tutorials and Manuscript figures

The tutorials progressively introduce the main functionality of the library and the most important functions in the source code, building up to show how the test cases from the manuscript can be reproduced:

- [Tutorial 1: Capillary rise](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%201%20Capillary%20rise.ipynb)
- [Tutorial 2: Particle-particle bridge](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%202%20particle-particle%20bridge.ipynb)
- [Tutorial 3: Sessile droplet comparison](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%203%20Sessile%20droplet.ipynb)

The original code for the manuscripts are contained in the `v0.3.1-alpha` tag. Currently we are refactoring the library so that some code might be broken. The published figures can be generated using the code inside `./manuscript_figures`

# Building the Documentation Locally

The documentation uses [Sphinx](https://www.sphinx-doc.org/) with the Read the Docs theme and autodoc for API reference generation.

## 1. Install documentation dependencies

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

## 2. Scaffold the docs (first time only)

```bash
mkdir -p docs
sphinx-quickstart docs \
  --sep \
  --project "ddgclib" \
  --author "Stefan Endres, Lutz Mädler, Ianto Cannon, Sonyi Deng, Marcello Zani" \
  --release "0.4.3" \
  --language en \
  --extensions sphinx.ext.autodoc,sphinx.ext.napoleon,sphinx.ext.viewcode,sphinx.ext.mathjax,sphinx_autodoc_typehints
```

Then edit `docs/source/conf.py` to use the RTD theme:

```python
html_theme = "sphinx_rtd_theme"
```

And ensure the package is importable by adding to the top of `conf.py`:

```python
import os, sys
sys.path.insert(0, os.path.abspath("../.."))
```

## 3. Build the HTML docs

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`.

## 4. View locally

Open the docs in your browser:

```bash
# Linux
xdg-open docs/build/html/index.html

# macOS
open docs/build/html/index.html

# Or start a local server
python -m http.server 8000 --directory docs/build/html
# then visit http://localhost:8000
```

## 5. Live-reload during editing (optional)

For a live-reloading dev server that rebuilds on file changes:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/build/html
# then visit http://127.0.0.1:8000
```
