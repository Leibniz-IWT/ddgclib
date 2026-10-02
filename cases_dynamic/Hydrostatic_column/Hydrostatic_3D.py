"""3D hydrostatic column: weakly compressible settling under gravity.

Unit cube of 189 vertices (refinement 2), gravity along z, no-slip bottom
and side walls (frozen vertices), free surface on top, gauge pressure,
linear Tait EOS with c0 = 10 sqrt(g H).  The run goes through
``ddgclib.methods.PRESETS['hydrostatic_3D']`` on the library
``symplectic_euler`` integrator; gravity is the ``body_force`` of
``SolverMethods.dudt_fn``.  All shared code is in ``src/_column.py``.

    python cases_dynamic/Hydrostatic_column/Hydrostatic_3D.py
    python cases_dynamic/Hydrostatic_column/Hydrostatic_3D.py --arm remap

The default horizon is 40 acoustic times (the 3D ``p_ij`` dual faces are
evaluated per force call, so this is the slowest of the four).  Outputs:
``results/hydrostatic_3D[_remap]/`` (snapshots, methods.json,
summary.json) and ``fig/hydrostatic_3D[_remap]_*``.  See README.md.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from cases_dynamic.Hydrostatic_column.src._column import run_case  # noqa: E402

if __name__ == '__main__':
    run_case('hydrostatic_3D')
