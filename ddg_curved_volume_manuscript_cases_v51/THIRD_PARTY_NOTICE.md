# Third-party notice

This archive contains author-created reproducibility scripts together with a
small number of third-party source trees needed to rebuild comparison helpers
for Figure 3-8. Third-party source trees are included only to make the figure
reproducible and retain their original license files.

## Bundled third-party source trees

- `Figure_3-8_VolumeOperatorComparison/common/_external_irl_quadratic_cutting/`

  Interface Reconstruction Library (IRL), used to build the local
  paraboloid-clipping helper for the Evrard-type comparison. The source tree
  includes its own `LICENSE.txt` and README. IRL is licensed under MPL-2.0.

- `Figure_3-8_VolumeOperatorComparison/common/_external_eigen/`

  Eigen C++ template library, used as a header dependency when building the
  IRL helper. The source tree includes its own `LICENSE`, `COPYING.*`, and
  README files. Eigen is primarily licensed under MPL-2.0, with additional
  notices for selected components as provided by Eigen.

- `Figure_3-8_VolumeOperatorComparison/neatplot-main/`

  Local plotting style package used by the manuscript figures. The folder
  includes its own MIT `LICENSE`.

## Python packages

Python packages such as `numpy`, `pandas`, `matplotlib`, `meshio`, `overlap`,
`pypdf`, `reportlab`, and `Pillow` are not vendored in this archive. They are
installed from PyPI into a local `_python_deps/` folder when the recomputation
scripts run. Their licenses are governed by their respective upstream packages.

## Literature-comparison wrappers

The files `evrard_2023.py`, `strobl_2016.py`, and `xie_xiao_2017.py` are
author-created comparison wrappers for the manuscript's Figure 3-8
reproducibility test. They should be read as adapted volume-operator
comparisons against a common PL baseline, not as full redistributed releases of
the original papers' complete simulation codes.

## Original v45 release

This v51 package is based on the released v45 Zenodo archive:

Deng, S., Endres, S. C., and Maedler, L. (2026). *DDG curved-volume manuscript
cases*, version v45. Zenodo. https://doi.org/10.5281/zenodo.19135931
