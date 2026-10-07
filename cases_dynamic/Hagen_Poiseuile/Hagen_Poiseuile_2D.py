"""2D planar Poiseuille flow developing from a plug (Lagrangian).

Channel ``[0, L] x [0, D]`` with no-slip walls at ``y = 0`` and ``y = D``.
Plug flow ``U_avg`` enters at ``x = 0`` and develops under the prescribed
pressure field ``P = G (L - x)`` towards ``u_x(y) = G / (2 mu) y (D - y)``.
The mesh moves with the fluid: an upstream buffer of prescribed plug motion
feeds the inlet (``PeriodicInletBufferedBC``), a buffer behind the outlet
removes the vertices (``OutletBufferedDeleteBC``), the walls are frozen by
membership.  The run goes through ``ddgclib.methods.PRESETS['hagen_poiseuille_2D']``
(symplectic Euler, per-step Delaunay, ``frozen_set='membership'``,
``viscous_flux='simplex_gradient'``) on the library integrator.  Shared
code: ``src/_run.py`` (runner body), ``src/_setup.py:setup_poiseuille_developing``
(mesh, BCs, ICs), ``src/_metrics.py`` (measurements); shipped parameters:
``CASES`` in ``src/_run.py``.

    python cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py --headless
    python cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py --viscous-flux two_point --tag two_point

``--viscous-flux``, ``--pressure-flux`` and ``--frozen-set`` run the preset
with that axis replaced (an A/B arm); ``--tag`` keeps an arm out of the
shipped results.  Outputs: ``results/hagen_poiseuille_2D[_<tag>]/``
(``summary.json``, ``methods.json``, snapshots, final state) and
``fig/hagen_poiseuille_2D[_<tag>]_*``.  See README.md.
"""
import os
import sys

# Repository root first, so that the live hyperct tree (symlink in the
# root) is imported and not an installed wheel.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from cases_dynamic.Hagen_Poiseuile.src._run import run_case  # noqa: E402

if __name__ == '__main__':
    run_case('hagen_poiseuille_2D')
