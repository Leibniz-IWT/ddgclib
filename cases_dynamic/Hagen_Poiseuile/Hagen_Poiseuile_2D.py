"""
2D Planar Poiseuille (Hagen-Poiseuille) Developing Flow

Simulates a 2D channel flow between parallel plates that develops from
an initial uniform plug flow toward the analytical parabolic Poiseuille
profile under a constant pressure gradient.

Physics:
    - Channel: [0, L] x [0, D]  (flow in x, walls at y=0 and y=D)
    - No-slip walls at y=0 and y=D
    - Open outlet: vertices that exit x >= L + buffer are deleted
    - Periodic inlet: ghost mesh injects new vertices at x=0
    - Constant pressure gradient G = -dP/dx driving the flow
    - Analytical steady-state: u_x(y) = (G / 2mu) * y * (D - y)

Uses the Cauchy stress tensor pipeline (ddgclib.operators.stress.dudt_i)
with the symplectic_euler integrator (semi-implicit, Lagrangian mesh),
through the preset ``hagen_poiseuille_2D`` (METHODS.md).  The walls are
frozen by MEMBERSHIP (``frozen_set='membership'``, laneL): with the old
hull rule the whole wall was released at step 1248, when two outlet
buffer vertices drifted past the wall line.

Usage (from this directory):
    python Hagen_Poiseuile_2D.py                       # 3000 steps, plots
    python Hagen_Poiseuile_2D.py --headless --steps 1400
    python Hagen_Poiseuile_2D.py --headless --frozen-set hull --tag hull

``--frozen-set`` runs the preset with that value replaced (an A/B arm);
``--tag`` sends the outputs to ``results/<tag>/`` so that an arm does
not overwrite the shipped run.  Every run writes ``wall_report.json``:
the wall vertices at the start and what became of them.

Visualization: run ``python visualize_hp2d.py`` after this script completes.


TODO: On fixing inlet:

1. Standard Inlet BC for Developing Poiseuille Flow
The existing test setup in _setup.py and test_case_hagen_poiseuille.py uses the Eulerian approach — euler_velocity_only on a fixed mesh with just NoSlipWallBC. No PeriodicInletBC at all:


# From test — standard approach
euler_velocity_only(HC, bV, mock_accel, dt=0.001, n_steps=10,
                    dim=2, bc_set=bc_set)
For developing Poiseuille flow the standard approach is a fixed mesh where only velocity evolves.

2. Why Vertices Accumulate (the core bug)
The vertex counts I measured confirm the problem:

Mesh	Domain	n_refine	Vertices
HP main	[0, 15] × [0, 1]	3	145
HP unit (inlet)	[0, 0.15] × [0, 1]	3	145
hyperct vertex count depends only on n_refine, not domain size. Both meshes have exactly 145 vertices. The unit mesh covers an area 100x smaller, so its vertex density is 100x higher than the main mesh.

Each injection cycle dumps 145 vertices into a 0.15-wide strip near x=0. Over 3000 steps (~20 cycles): ~2900 new vertices, all crammed near the inlet. That's the "hundreds to thousands of vertices close together."

3. Hash / merge_all Issue
Confirmed from hyperct source — mesh.V[tuple(gv.x_a)] always auto-creates on cache miss (no distance tolerance, exact tuple match). Since main mesh vertices have moved even slightly from the ghost positions, every injection creates a new vertex. And merge_all(cdist=1e-10) is far too tight to catch the overlap.

Recommendation
For the developing Poiseuille case, the simplest correct approach is to use euler_velocity_only (Eulerian fixed mesh) like the test does — no PeriodicInletBC needed. The symplectic_euler + PeriodicInletBC approach is designed for free-surface/moving-boundary problems, not channel flow on a fixed domain.

If you do want the Lagrangian symplectic_euler with inlet injection, the fix is to set inlet_layer_thickness ≈ L / 2^n_refine (the main mesh x-spacing) so the unit mesh density matches. For L=15, n_refine=3: inlet_layer_thickness ≈ 1.875.

Would you like me to switch Hagen_Poiseuile_2D.py to the Eulerian euler_velocity_only approach (matching the test), or fix the Lagrangian inlet parameters?

"""

import argparse
import json
import os
import pickle
import sys
import numpy as np

# Repository root first, so that the live hyperct tree (symlink in the
# root) is imported and not an installed wheel.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from ddgclib.methods import PRESETS, record_methods
from ddgclib.data import StateHistory, save_state

# Local parameters
from src._params import (
    L, G, mu, rho, D, U_avg, U_max, cdist,
    print_params,
)
from src._setup import (
    setup_poiseuille_2d_lagrangian, wall_snapshot, wall_report,
)

ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
ap.add_argument('--steps', type=int, default=3000, help='time steps (dt = 0.01)')
ap.add_argument('--headless', action='store_true',
                help='no blocking mesh plots')
ap.add_argument('--frozen-set', default=None, choices=['hull', 'membership'],
                help='A/B arm: the preset with frozen_set replaced')
ap.add_argument('--workers', type=int, default=None,
                help='dudt worker processes (default: the preset value)')
ap.add_argument('--tag', default='', help='write to results/<tag>/')
args = ap.parse_args()

# Output directories
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results', args.tag)
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RESULTS, exist_ok=True)

print_params()

# ============================================================
# Steps 1-3: Domain, boundary conditions, initial conditions
# ============================================================
# Channel [0, L] x [0, D], walls at y=0 and y=D; outlet buffer, periodic
# inlet ghost, positional no-slip walls; plug flow at U_avg.  See
# src/_setup.py:setup_poiseuille_2d_lagrangian.

d = 2  # spatial dimension
n_refine = 1

HC, bV, bc_set, wall_criterion, params = setup_poiseuille_2d_lagrangian(
    L=L, D=D, U_avg=U_avg, rho=rho, mu=mu, G=G, n_refine=n_refine,
    buffer_width=2.0, cdist=cdist,
)
poiseuille_ic = params['poiseuille_ic']
if not args.headless:
    HC.plot_complex()

n_verts = sum(1 for _ in HC.V)
walls_start = wall_snapshot(HC, wall_criterion)
print(f"\nMesh: {n_verts} vertices, {n_refine} refinements")
print(f"BCs: {len(walls_start)} wall, outlet at x={L:.1f} "
      f"(+ buffer {params['buffer_width']:.1f}), periodic inlet (period=1.0)")
print(f"ICs applied: plug flow u_x={U_avg:.3f} m/s, P gradient G={G:.5f} Pa/m")

# ============================================================
# Step 4: Dynamic Integration
# ============================================================
# Symplectic (semi-implicit) Euler: updates velocity first, then
# position using the NEW velocity.  This is a Lagrangian scheme —
# vertices move with the flow.

# Solver methods (METHODS.md): Lagrangian symplectic Euler, per-step
# Delaunay, walls frozen by membership (narrowed from the hull by
# boundary_filter), 20 dudt workers (safe: pressure_model=None, no EOS
# side effects).
methods = PRESETS['hagen_poiseuille_2D']
changes = {}
if args.frozen_set is not None:
    changes['frozen_set'] = args.frozen_set
if args.workers is not None:
    changes['workers'] = args.workers if args.workers > 1 else None
if changes:
    methods = methods.replace(**changes)
print(methods.describe())
dudt_fn = methods.dudt_fn(HC, mu=mu)

# Time stepping parameters
dt = 0.01
n_steps = args.steps
record_every = 25
save_every = 500  # save state to disk every 500 steps

history = StateHistory(fields=['u', 'p'], record_every=record_every)

# Number of vertices on the wall lines after every step.  It drops when
# wall vertices are released and drift off the line (the hull rule).
n_wall_log = []


def callback(step, t, HC, bV=None, diagnostics=None):
    history.callback(step, t, HC, bV, diagnostics)
    n_wall_log.append(sum(1 for v in HC.V if wall_criterion(v)))


print(f"\nRunning: dt={dt}, n_steps={n_steps}, t_final={dt*n_steps:.2f}")
print(f"Recording every {record_every} steps ({n_steps // record_every} snapshots)")
print(f"Saving state to {_RESULTS}/ every {save_every} steps")
if not args.headless:
    HC.plot_complex()
t_final = methods.integrate(
    HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
    bc_set=bc_set,
    boundary_filter=wall_criterion,  # only wall vertices are frozen, not inlet/outlet
    callback=callback,
    save_every=save_every,
    save_dir=_RESULTS,
)
record_methods(os.path.join(_RESULTS, 'methods.json'), methods, HC,
               extra={'dt': dt, 'n_steps': n_steps, 'mu': mu, 'G': G,
                      'boundary_filter': 'wall_criterion (|y|<1e-10 or |y-D|<1e-10)'})

print(f"Simulation complete: t = {t_final:.4f}")

# Wall vertices: where they were at the start and what became of them.
report = wall_report(HC, bV, walls_start)
drops = [i for i in range(1, len(n_wall_log))
         if n_wall_log[i] < n_wall_log[i - 1]]
report.update(
    frozen_set=methods.frozen_set, n_steps=n_steps, dt=dt,
    n_on_wall_lines_first=n_wall_log[0] if n_wall_log else None,
    n_on_wall_lines_min=min(n_wall_log, default=None),
    n_on_wall_lines_end=n_wall_log[-1] if n_wall_log else None,
    first_step_wall_count_drops=drops[0] if drops else None,
    n_vertices_end=sum(1 for _ in HC.V), n_frozen_end=len(bV),
)
with open(os.path.join(_RESULTS, 'wall_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nWalls ({methods.frozen_set}): {report['n_wall_start']} at the start, "
      f"{report['n_frozen']} still frozen, {report['n_moved']} moved "
      f"(max displacement {report['max_displacement']:.3e}); "
      f"wall count first drops at step {report['first_step_wall_count_drops']}")

# ============================================================`
# Step 5: Save Results and Analyze
# ============================================================
# All visualization is in visualize_hp2d.py (separate script).

# 5a: Save final state
save_state(HC, bV, t=t_final, fields=['u', 'p', 'm'],
           path=os.path.join(_RESULTS, 'hp2d_final_state.json'),
           extra_meta={'case': 'hagen_poiseuille_2d', 'mu': mu, 'G': G,
                       'Re_D': rho * U_avg * D / mu})
print(f"Final state saved to {_RESULTS}/hp2d_final_state.json")

# 5b: Save history (for animation in vis script)
history_path = os.path.join(_RESULTS, 'hp2d_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history, f)
print(f"History saved to {history_path} ({history.n_snapshots} snapshots)")

# 5c: Error metrics at channel midpoint
x_mid = L / 2.0
tol = L / (2**n_refine) * 0.6
mid_verts = sorted(
    [v for v in HC.V if abs(v.x_a[0] - x_mid) < tol],
    key=lambda v: v.x_a[1]
)

# Velocity comparison at midplane cross-section
ux_num = np.array([v.u[0] for v in mid_verts])
print(f"\nVelocity at x=L/2:")
print(f"  U_max analytical = {U_max:.6f}, U_max numerical = "
      f"{max(ux_num) if len(ux_num) > 0 else float('nan'):.6f}")

# 5d: Integrated comparison (velocity difference tensor on final mesh)
print(f"\nIntegrated velocity comparison (Du_DDG vs ∫ ∇u dV):")
try:
    from hyperct.ddg import dual_cell_polygon_2d
    from ddgclib.analytical._divergence_theorem import integrated_gradient_2d_vector
    from ddgclib.operators.stress import velocity_difference_tensor, cache_dual_volumes
    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize

    # The inlet may have injected vertices in the last BC pass; they have
    # no dual cell until the next retopology.
    _retopologize(HC, bV, d, boundary_filter=wall_criterion,
                  frozen_set=methods.frozen_set)
    cache_dual_volumes(HC, dim=d)
    interior_final = [v for v in HC.V if v not in bV]
    du_errors = []
    for v in interior_final:
        try:
            Du_num = velocity_difference_tensor(v, HC, dim=d)
            polygon = dual_cell_polygon_2d(v)
            u_callable = lambda x, _ic=poiseuille_ic: np.array([
                _ic.analytical_velocity(x), 0.0
            ])
            Du_ana = integrated_gradient_2d_vector(u_callable, polygon)
            du_errors.append(np.linalg.norm(Du_num - Du_ana))
        except Exception:
            pass

    if du_errors:
        print(f"  max|Du_DDG - Du_ana| = {max(du_errors):.6e}")
        print(f"  mean|Du_DDG - Du_ana| = {np.mean(du_errors):.6e}")
    else:
        print("  No interior vertices for integrated comparison")

    # 5e: Integrated pressure comparison
    from ddgclib.analytical._integrated_comparison import (
        integrated_pressure_error,
        integrated_l2_norm,
        compare_stress_force,
    )
    P_analytical = lambda x: -G * x[0]  # P(x) = -G*x (Poiseuille)
    int_p_errs = integrated_pressure_error(
        HC, interior_final, P_analytical=P_analytical, dim=d,
    )
    int_p_l2 = integrated_l2_norm(
        HC, interior_final, P_analytical=P_analytical, dim=d,
    )
    print(f"\nIntegrated pressure comparison:")
    if int_p_errs:
        print(f"  max|p*V - ∫P dV| = {max(int_p_errs):.6e}")
        print(f"  Integrated L2 norm = {int_p_l2:.6e}")

    # 5f: Force balance diagnostic
    force_diag = compare_stress_force(HC, interior_final, dim=d, mu=mu)
    print(f"  Force balance: max|F| = {force_diag['max_F']:.4e}, "
          f"median|F| = {force_diag['median_F']:.4e}")

except ImportError as e:
    print(f"  Skipping integrated comparison: {e}")

print(f"\nRun 'python visualize_hp2d.py' to generate all plots and animations.")
