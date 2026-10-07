"""2D hydrostatic column: weakly compressible settling under gravity.

Unit square of 145 vertices (refinement 3), gravity along y, no-slip
bottom and side walls (frozen vertices), free surface on top, gauge
pressure, linear Tait EOS with c0 = 10 sqrt(g H).  The run goes through
``ddgclib.methods.PRESETS['hydrostatic_2D']`` on the library
``symplectic_euler`` integrator; gravity is the ``body_force`` of
``SolverMethods.dudt_fn``.  All shared code is in ``src/_column.py``.

    python cases_dynamic/Hydrostatic_column/Hydrostatic_2D.py
    python cases_dynamic/Hydrostatic_column/Hydrostatic_2D.py --arm remap

``--arm remap`` runs the reconnecting arm (``delaunay_material`` +
conservative remap) instead of the preset.  Outputs:
``results/hydrostatic_2D[_remap]/`` (snapshots, methods.json,
summary.json) and ``fig/hydrostatic_2D[_remap]_*``.  See README.md.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from cases_dynamic.Hydrostatic_column.src._column import run_case  # noqa: E402

if __name__ == '__main__':
    run_case('hydrostatic_2D')
