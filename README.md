[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8010952.svg)](https://doi.org/10.5281/zenodo.8010952)

# ddgclib
Experimental library for discrete differential geometry curvature in fluid simulations, especially for useful for problems with complex and changing topologies.

# Installation
setup.py files are planned for a future release, currently the test cases can by run by cloning the entire repository:

`$ git clone https://github.com/Leibniz-IWT/ddgclib/`

# Tutorials and Manuscript figures

The tutorials progressively introduce the main functionality of the library and the most important functions in the source code, building up to show how the test cases from the manuscript can be reproduced:

- [Tutorial 1: Capillary rise](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%201%20Capillary%20rise.ipynb)
- [Tutorial 2: Particle-particle bridge](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%202%20particle-particle%20bridge.ipynb)
- [Tutorial 3: Sessile droplet comparison](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%203%20Sessile%20droplet.ipynb)

The original code for the manuscripts are contained in the `v0.3.1-alpha` tag. Currently we are refactoring the library so that some code might be broken. The published figures can be generated using the code inside `./manuscript_figures`
