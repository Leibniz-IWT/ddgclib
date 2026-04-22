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

# Documentation

The documentation uses [Sphinx](https://www.sphinx-doc.org/) with the Read the Docs theme and includes user guides, API reference, and tutorials.

**Contents:**

- **Getting Started** — installation, quick start
- **User Guide** — simulation workflow, multiphase flows, DEM, mean curvature flow
- **API Reference** — operators, multiphase, EOS, BCs, ICs, integrators, geometry, visualization, DEM

## Build the docs

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
cd docs
make html
```

The built HTML is in `docs/build/html/`. Open with:

```bash
xdg-open docs/build/html/index.html   # Linux
open docs/build/html/index.html        # macOS
```

For live-reload during editing:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/build/html
# visit http://127.0.0.1:8000
```
