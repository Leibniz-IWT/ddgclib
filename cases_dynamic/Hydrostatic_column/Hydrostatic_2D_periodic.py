"""2D hydrostatic column with free-slip side walls (no wall meniscus).

Same column as ``Hydrostatic_2D.py``, but the side-wall vertices slide
along the wall (``ddgclib._boundary_conditions.FreeSlipWallBC``) instead
of being frozen, so the solution is one-dimensional and the comparison
with the analytical profile has no wall layer.  Only the bottom is
frozen.

The file keeps its historical name.  The column is NOT periodic: the
library periodic connectivity (``connectivity='periodic'``) cannot carry
a single-phase EOS column yet (seam dual volumes and dual-face closure,
see README.md), so the side walls are symmetry walls, which is what this
runner always did with case-local code.

The run goes through ``ddgclib.methods.PRESETS['hydrostatic_2D_periodic']``
on the library ``symplectic_euler`` integrator.  All shared code is in
``src/_column.py``.

    python cases_dynamic/Hydrostatic_column/Hydrostatic_2D_periodic.py
    python cases_dynamic/Hydrostatic_column/Hydrostatic_2D_periodic.py --arm remap

Outputs: ``results/hydrostatic_2D_periodic[_remap]/`` and
``fig/hydrostatic_2D_periodic[_remap]_*``.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from cases_dynamic.Hydrostatic_column.src._column import run_case  # noqa: E402

if __name__ == '__main__':
    run_case('hydrostatic_2D_periodic')
