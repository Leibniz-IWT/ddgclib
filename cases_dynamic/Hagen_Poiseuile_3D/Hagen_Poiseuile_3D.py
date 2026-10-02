"""3D Hagen-Poiseuille pipe flow developing from a plug (Lagrangian).

Pipe of radius ``R`` along z with a no-slip wall.  Plug flow ``U_avg``
enters at ``z = 0`` and develops under the prescribed pressure field
``P = G (L - z)`` towards ``u_z(r) = U_max (1 - (r / R)^2)``,
``U_max = G R^2 / (4 mu) = 2 U_avg``.  Same construction as the 2D channel
(``cases_dynamic/Hagen_Poiseuile``): upstream buffer of prescribed plug
motion, buffer behind the outlet, walls frozen by membership.  The run goes
through ``ddgclib.methods.PRESETS['hagen_poiseuille_3D']`` (symplectic
Euler, per-step Delaunay, ``frozen_set='membership'``,
``pressure_flux='simplex_gradient'``, ``viscous_flux='simplex_gradient'``)
on the library integrator; there is no case-local retopology.  Shared code:
``cases_dynamic/Hagen_Poiseuile/src/_run.py``, ``_setup.py``, ``_metrics.py``.

    python cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py --headless
    python cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py --n-refine 1 --steps 300

Outputs: ``results/hagen_poiseuille_3D[_<tag>]/`` (``summary.json``,
``methods.json``, snapshots, final state) and
``fig/hagen_poiseuille_3D[_<tag>]_*``.  Interactive replay:
``python -m ddgclib.scripts.view_polyscope --snapshots
cases_dynamic/Hagen_Poiseuile_3D/results/hagen_poiseuille_3D/snapshots``
or ``python visualize_hp3d.py``.  See README.md.
"""
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from cases_dynamic.Hagen_Poiseuile.src._run import CASES, run_case  # noqa: E402

# Shipped parameters, by name (visualize_hp3d.py imports them).
_CASE = CASES['hagen_poiseuille_3D']
flow_axis = 2
D = _CASE['D']
R = D / 2
L = _CASE['L']
rho = _CASE['rho']
mu = _CASE['mu']
U_avg = _CASE['U_avg']
U_max = 2 * U_avg
Re_D = rho * U_avg * D / mu
G = 8 * mu * U_avg / R**2
n_refine = _CASE['n_refine']

if __name__ == '__main__':
    # Degenerate (coplanar) tetrahedra of the Delaunay rebuild at the wall
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            module='hyperct.ddg._geometry')
    run_case('hagen_poiseuille_3D')
