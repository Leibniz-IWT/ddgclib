"""Test script for the NEW OutletBufferedDeleteBC.

Runs a short HP2D simulation and reports per-step diagnostics to
compare outlet behaviour against the old OutletDeleteBC.

Usage:
    cd cases_dynamic/Hagen_Poiseuile
    python test_outlet_new_bc.py
"""

import numpy as np
from functools import partial

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    PositionalNoSlipWallBC,
    OutletBufferedDeleteBC,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    UniformVelocity,
    LinearPressureGradient,
    UniformMass,
)
from ddgclib.operators.stress import dudt_i
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.geometry._complex_operations import extrude

# --- Parameters (match HP2D case) ---
L, D, d = 15.0, 1.0, 2
U_avg, rho = 0.1, 1.0
r = D / 2
mu = (rho * U_avg * D) / 100.0
G = 8 * mu * U_avg / r**2
buffer_width = 2.0
n_refine = 1
dt = 0.01
n_steps = 200
report_every = 25

# --- Build mesh ---
HC_unit = Complex(d, domain=[(0.0, 1), (0.0, D)])
HC_unit.triangulate()
for _ in range(n_refine):
    HC_unit.refine_all()
HC = extrude(HC_unit, L, axis=0, cdist=1e-10)

bV = HC.boundary(HC.V)
for v in HC.V:
    v.boundary = v in bV
compute_vd(HC, method='barycentric')

# --- Wall criterion ---
wall_tol = 1e-10
wall_criterion = lambda v: (
    abs(v.x_a[1]) < wall_tol or abs(v.x_a[1] - D) < wall_tol
)

# --- BCs: NEW approach (OutletBufferedDeleteBC) ---
bc_set = BoundaryConditionSet()
bc_set.add(
    PositionalNoSlipWallBC(criterion_fn=wall_criterion, dim=d, bV=bV),
    None,
)
outlet_bc = OutletBufferedDeleteBC(
    outlet_pos=L, buffer_width=buffer_width, axis=0, bV=bV,
)
bc_set.add(outlet_bc, None)

# --- Initial conditions ---
ic = CompositeIC(
    UniformVelocity(u_vec=np.array([U_avg, 0.0])),
    LinearPressureGradient(G=G, axis=0, P_ref=0.0),
    UniformMass(total_volume=L * D, rho=rho),
)
ic.apply(HC, bV)
bc_set.apply_all(HC, bV, dt=0.0)

dudt_fn = partial(dudt_i, dim=d, mu=mu, HC=HC)

# --- Callback ---
print(f"{'Step':>6}  {'t':>7}  {'Verts':>5}  {'bV':>3}  "
      f"{'min_ux':>10}  {'max_ux':>10}  {'backward':>8}  "
      f"{'on_bdy_outlet':>13}  {'buf_size':>8}")
print("-" * 100)


def callback(step, t, HC, bV, diag):
    if step % report_every != 0:
        return
    # Outlet-adjacent domain vertices (before outlet_pos)
    outlet_zone = [v for v in HC.V
                   if L - 2.0 < v.x_a[0] <= L
                   and not wall_criterion(v)]
    n = sum(1 for _ in HC.V)
    backward = sum(1 for v in HC.V
                   if v.u[0] < -0.01 and not wall_criterion(v))

    # Check if domain vertices near outlet are on topological boundary
    topo_bdy = HC.boundary()
    on_bdy = sum(1 for v in topo_bdy
                 if L - 1.0 < v.x_a[0] <= L and not wall_criterion(v))

    buf_size = len(outlet_bc._buffer)

    if outlet_zone:
        min_ux = min(v.u[0] for v in outlet_zone)
        max_ux = max(v.u[0] for v in outlet_zone)
    else:
        min_ux = max_ux = float('nan')

    print(f"{step:6d}  {t:7.3f}  {n:5d}  {len(bV):3d}  "
          f"{min_ux:10.4f}  {max_ux:10.4f}  {backward:8d}  "
          f"{on_bdy:13d}  {buf_size:8d}")


# --- Run ---
print(f"\nNEW BC: OutletBufferedDeleteBC(outlet_pos={L}, buffer_width={buffer_width})")
print(f"Running {n_steps} steps, dt={dt}\n")

t_final = symplectic_euler(
    HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=d,
    bc_set=bc_set, boundary_filter=wall_criterion,
    callback=callback, workers=1,
)

n_final = sum(1 for _ in HC.V)
backward_final = sum(1 for v in HC.V
                     if v.u[0] < -0.01 and not wall_criterion(v))
buf_final = len(outlet_bc._buffer)

print(f"\nFinal: {n_final} verts, t={t_final:.4f}, "
      f"backward={backward_final}, buffer={buf_final}")

# --- Position drift check ---
print("\nBuffer vertex position check:")
for vid, (v, frozen_u, correct_pos) in list(outlet_bc._buffer.items())[:5]:
    actual = v.x_a[0]
    expected = correct_pos[0]
    drift = abs(actual - expected)
    print(f"  x={actual:.6f}, correct={expected:.6f}, drift={drift:.2e}, "
          f"frozen_ux={frozen_u[0]:.4f}")
