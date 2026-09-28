Installation
============

Requirements
------------

- Python 3.9 or later (development uses Python 3.13)
- NumPy, SciPy
- `hyperct <https://github.com/stefan-endres/hyperct>`_ (simplicial complex backend)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/Leibniz-IWT/ddgclib.git
   cd ddgclib
   pip install -e .

Optional dependencies
---------------------

Install extras for visualization, data handling, GPU support, or development:

.. code-block:: bash

   # 3D visualization (polyscope + matplotlib)
   pip install -e ".[vis]"

   # Data handling (pandas)
   pip install -e ".[data]"

   # GPU acceleration (PyTorch)
   pip install -e ".[gpu]"

   # Development tools (pytest)
   pip install -e ".[dev]"

   # Everything
   pip install -e ".[vis,data,gpu,dev]"

Conda environment
-----------------

A full conda environment is provided in ``environment.yml``:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate ddg

Building documentation
----------------------

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

   cd docs
   make html

The built HTML is in ``docs/build/html/``. For live-reload during editing:

.. code-block:: bash

   pip install sphinx-autobuild
   sphinx-autobuild source build/html

Verifying the installation
--------------------------

Run the fast test suite to confirm everything works:

.. code-block:: bash

   pytest ddgclib/tests/ -v -m "not slow"

This runs ~400 tests in under 20 seconds.
