"""1D hydrostatic column: weakly compressible settling under gravity.

A 10 m column of 33 vertices (refinement 4), frozen bottom vertex, free top
vertex, gauge pressure, linear Tait EOS with c0 = 10 sqrt(g H).  The run
goes through ``ddgclib.methods.PRESETS['hydrostatic_1D']`` on the library
``symplectic_euler`` integrator; gravity is the ``body_force`` of
``SolverMethods.dudt_fn``.  All shared code is in ``src/_column.py``.

    python cases_dynamic/Hydrostatic_column/Hydrostatic_1D.py
    python cases_dynamic/Hydrostatic_column/Hydrostatic_1D.py --ic equilibrium --n-tac 40

The default horizon is 200 acoustic times: the uniform-density start rings
at the fundamental mode, whose viscous damping time is 52 acoustic times
at this resolution.

Outputs: ``results/hydrostatic_1D/`` (snapshots, methods.json,
summary.json) and ``fig/hydrostatic_1D_*``.  See README.md.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from cases_dynamic.Hydrostatic_column.src._column import run_case  # noqa: E402

if __name__ == '__main__':
    run_case('hydrostatic_1D')
