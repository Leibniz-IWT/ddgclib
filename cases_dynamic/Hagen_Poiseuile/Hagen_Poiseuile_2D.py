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
with the symplectic_euler integrator (semi-implicit, Lagrangian mesh).

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

import os
import pickle
import numpy as np
from functools import partial

from hyperct import Complex
from hyperct.ddg import compute_vd

# ddgclib modules
from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    PositionalNoSlipWallBC,
    OutletBufferedDeleteBC,
    PeriodicInletBC,
    identify_boundary_vertices,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    UniformVelocity,
    LinearPressureGradient,
    UniformMass,
    PoiseuillePlanar,
)

from ddgclib.geometry._complex_operations import extrude

from ddgclib.operators.stress import dudt_i
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory, save_state

# Local parameters
from src._params import (
    L, r, G, mu, rho, D, U_avg, U_max,
    inlet_layer_thickness, outlet_buffer, cdist,
    print_params,
)

# Output directories
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RESULTS, exist_ok=True)

print_params()

# ============================================================
# Step 1: Domain Setup
# ============================================================
# Create a 2D rectangular channel [0, L] x [0, D].
# Flow is in the x-direction, walls at y=0 and y=D.

d = 2  # spatial dimension
n_refine = 1

#TODO: Rebuild extrude method to get a more regular mesh for this simple geometry:
HC_unit = Complex(d, domain=[(0.0, 1), (0.0, D)])
HC_unit.triangulate()
for _ in range(n_refine):
    HC_unit.refine_all()

#HC_unit.plot_complex()

# Extrude to L
#HC = extrude(HC_unit, L, axis=1, cdist=1e-10)
HC = extrude(HC_unit, L, axis=0, cdist=1e-10)
print(help(HC.plot_complex))
HC.plot_complex()

n_verts = sum(1 for _ in HC.V)
print(f"\nMesh: {n_verts} vertices, {n_refine} refinements")

# Tag boundary vertices (required before compute_vd)
if 0:
    bV = identify_boundary_vertices(HC, lambda v: (
        abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L) < 1e-14 or
        abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - D) < 1e-14
    ))
else:
    bV = HC.boundary(HC.V)

for v in HC.V:
    v.boundary = v in bV

# Compute barycentric dual mesh (needed by stress tensor operators)
compute_vd(HC, method="barycentric")

print(f"Boundary vertices: {len(bV)}")

# ============================================================
# Step 2: Boundary Conditions (Lagrangian)
# ============================================================
# Wall criterion: no-slip at y=0 and y=D (position-based, detects
# newly injected vertices automatically).
wall_tol = 1e-10
wall_criterion = lambda v: (
    abs(v.x_a[1]) < wall_tol or abs(v.x_a[1] - D) < wall_tol
)

# Build the unit mesh for the periodic inlet (a thin strip [0, thickness] x [0, D])
#TODO: unit mesh should be  domain=[(0.0, 1), (0.0, D)]
if 0:
    unit_mesh = Complex(d, domain=[(0.0, inlet_layer_thickness), (0.0, D)])
    unit_mesh.triangulate()
    for _ in range(n_refine):
        unit_mesh.refine_all()
else:
    HC_unit = Complex(d, domain=[(0.0, 1), (0.0, D)])
    HC_unit.triangulate()
    for _ in range(n_refine):
        HC_unit.refine_all()

    unit_mesh = HC_unit

#HC_unit.plot_complex()
# Set ICs on the unit mesh: plug flow velocity + linear pressure + mass
if 0:
    unit_ic = CompositeIC(
        UniformVelocity(u_vec=np.array([U_avg, 0.0])),
        LinearPressureGradient(G=G, axis=0, P_ref=0.0),
        UniformMass(total_volume=inlet_layer_thickness * D, rho=rho),
    )
    unit_bV = identify_boundary_vertices(unit_mesh, lambda v: (
        abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - inlet_layer_thickness) < 1e-14 or
        abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - D) < 1e-14
    ))
    unit_ic.apply(unit_mesh, unit_bV)
else:
    unit_ic = CompositeIC(
        UniformVelocity(u_vec=np.array([U_avg, 0.0])),
        LinearPressureGradient(G=G, axis=0, P_ref=0.0),
        UniformMass(total_volume=1 * D, rho=rho),
        #UniformMass(total_volume=inlet_layer_thickness  * D, rho=rho),
    )

    unit_bV = HC.boundary(HC.V)
    for v in unit_bV:
        v.boundary = v in bV

    unit_ic.apply(unit_mesh, unit_bV)

# Assemble BCs:
#   1. PositionalNoSlipWallBC — zeros velocity at wall positions, adds to bV
#   2. OutletDeleteBC — deletes vertices past x = L + buffer
#   3. PeriodicInletBC — ghost mesh injects new vertices at x = 0
bc_set = BoundaryConditionSet()
bc_set.add(
    PositionalNoSlipWallBC(criterion_fn=wall_criterion, dim=d, bV=bV),
    None,  # scans all vertices
)
print(f'L = {L+ outlet_buffer}')
bc_set.add(
    OutletBufferedDeleteBC(
        outlet_pos=L, buffer_width=2.0, axis=0, bV=bV,
    ),
    None,  # scans all vertices
)
print(f'inlet_layer_thickness = {inlet_layer_thickness}')
bc_set.add(
    PeriodicInletBC(
        unit_mesh=unit_mesh,
        velocity=U_avg,
        axis=0,
        inlet_pos=0.0,
        cdist=cdist,
        fields=['u', 'p', 'm'],
        period=1.0,  # must match unit mesh x-span (see bc_demo)
    ),
    None,  # manages its own ghost mesh
)

n_wall = sum(1 for v in HC.V if wall_criterion(v))
print(f"BCs: {n_wall} wall, outlet at x={L + outlet_buffer:.1f}, "
      f"periodic inlet (period={inlet_layer_thickness})")

# ============================================================
# Step 3: Initial Conditions
# ============================================================
# Start from uniform plug flow at U_avg — the flow will develop
# toward the analytical parabolic profile under the BCs.

ic = CompositeIC(
    UniformVelocity(u_vec=np.array([U_avg, 0.0])),    # plug flow in x
    LinearPressureGradient(G=G, axis=0, P_ref=0.0),   # linear P(x) = -G*x
    UniformMass(total_volume=L * D, rho=rho),
)
ic.apply(HC, bV)

# Enforce BCs on initial state (walls should be zero)
bc_set.apply_all(HC, bV, dt=0.0)

# Build the analytical profile object for comparison
poiseuille_ic = PoiseuillePlanar(
    G=G, mu=mu, y_lb=0.0, y_ub=D,
    flow_axis=0, normal_axis=1, dim=d,
)

print(f"ICs applied: plug flow u_x={U_avg:.3f} m/s, P gradient G={G:.5f} Pa/m")

# ============================================================
# Step 4: Dynamic Integration
# ============================================================
# Symplectic (semi-implicit) Euler: updates velocity first, then
# position using the NEW velocity.  This is a Lagrangian scheme —
# vertices move with the flow.

# Bind physics parameters via partial (avoids HC keyword conflict)
dudt_fn = partial(dudt_i, dim=d, mu=mu, HC=HC)

# Time stepping parameters
dt = 0.01
#n_steps = 2000 * 3
n_steps =  3000
record_every = 25
save_every = 500  # save state to disk every 500 steps

history = StateHistory(fields=['u', 'p'], record_every=record_every)

print(f"\nRunning: dt={dt}, n_steps={n_steps}, t_final={dt*n_steps:.2f}")
print(f"Recording every {record_every} steps ({n_steps // record_every} snapshots)")
print(f"Saving state to {_RESULTS}/ every {save_every} steps")
HC.plot_complex()
t_final = symplectic_euler(
    HC, bV, dudt_fn,
    dt=dt, n_steps=n_steps, dim=d,
    bc_set=bc_set,
    boundary_filter=wall_criterion,  # only freeze wall vertices, not inlet/outlet
    callback=history.callback,
    save_every=save_every,
    save_dir=_RESULTS,
    workers=20
)

print(f"Simulation complete: t = {t_final:.4f}")

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

errors = []
for v in mid_verts:
    u_anal = poiseuille_ic.analytical_velocity(v.x_a)
    errors.append(abs(v.u[0] - u_anal))

max_err = max(errors) if errors else float('nan')
l2_err = np.sqrt(np.mean(np.array(errors)**2)) if errors else float('nan')
ux_num = np.array([v.u[0] for v in mid_verts])
print(f"\nError at x=L/2 vs analytical: max={max_err:.6e}, L2={l2_err:.6e}")
print(f"U_max analytical = {U_max:.6f}, U_max numerical = "
      f"{max(ux_num) if len(ux_num) > 0 else float('nan'):.6f}")
print(f"\nRun 'python visualize_hp2d.py' to generate all plots and animations.")
