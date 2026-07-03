# Third-party notice for Figure 3-8

This folder includes author-created Python scripts for reproducing Figure 3-8
and several third-party dependencies used by those scripts.

Bundled source trees:

- `common/_external_irl_quadratic_cutting/`: Interface Reconstruction Library
  source used to build the local Evrard-type paraboloid-clipping helper. The
  folder includes its original `LICENSE.txt` and README. The license is
  MPL-2.0.
- `common/_external_eigen/`: Eigen C++ headers used by the IRL build. The
  folder includes its original `LICENSE`, `COPYING.*`, and README files.
- `neatplot-main/`: plotting-style helper with its own MIT `LICENSE`.

Not bundled:

- Python packages listed in `requirements.txt`, including `overlap`, are
  installed from PyPI into `_python_deps/` when the recompute scripts run.
  The `_python_deps/` folder is platform-specific and intentionally omitted
  from this package.

The files named by literature year, such as `evrard_2023.py`,
`strobl_2016.py`, and `xie_xiao_2017.py`, are author-created wrappers used for
the manuscript comparison. They implement adapted volume-operator comparisons
against the same PL baseline and should not be cited as complete copies of the
original authors' full solver codes.
