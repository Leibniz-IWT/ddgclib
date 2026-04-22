#!/usr/bin/env python
"""
ddgclib Dynamic Framework — Feature Demonstration

This script shows all the new features in action with three mini-examples:

  1. Hydrostatic column (1D) — pressure IC, equilibrium check
  2. Poiseuille channel (2D) — velocity IC, BCs, integrator, visualization
  3. Save/load round-trip + history queries

Run from the project root:
    python cases_dynamic/Template/example_features_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (works headless)
import matplotlib.pyplot as plt

from hyperct import Complex

# Output directory: fig/ next to this script
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
os.makedirs(_FIG, exist_ok=True)


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# Example 1: Hydrostatic Column (1D)
separator("Example 1: Hydrostatic Column (1D)")

from ddgclib._boundary_conditions import (
    identify_cube_boundaries,
    BoundaryConditionSet,
    NoSlipWallBC,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    ZeroVelocity,
    HydrostaticPressure,
    UniformMass,
)

# Build 1D mesh on [0, 1]
HC1 = Complex(1, domain=[(0.0, 1.0)])
HC1.triangulate()
HC1.refine_all()
HC1.refine_all()
HC1.refine_all()

bV1 = identify_cube_boundaries(HC1, 0.0, 1.0, dim=1)

# Physical parameters
rho = 1000.0   # kg/m^3
g = 9.81       # m/s^2
h = 1.0        # column height

# Apply hydrostatic IC: P(x) = rho*g*(h - x), u = 0
ic = CompositeIC(
    ZeroVelocity(dim=1),
    HydrostaticPressure(rho=rho, g=g, axis=0, h_ref=h, P_ref=0.0),
    UniformMass(total_volume=h, rho=rho),
)
ic.apply(HC1, bV1)

# Verify
print("Hydrostatic pressure profile:")
for v in sorted(HC1.V, key=lambda v: v.x_a[0]):
    x = v.x_a[0]
    P_analytical = rho * g * (h - x)
    P_computed = v.p
    error = abs(P_computed - P_analytical)
    marker = " <-- boundary" if v in bV1 else ""
    print(f"  x={x:.4f}  P={P_computed:9.2f}  analytical={P_analytical:9.2f}  "
          f"err={error:.2e}{marker}")

# Plot
from ddgclib.visualization import plot_scalar_field_1d

fig1, ax1 = plot_scalar_field_1d(
    HC1, field='p',
    analytical_fn=lambda x: rho * g * (h - x),
    xlabel='Height x [m]',
    ylabel='Pressure P [Pa]',
    title='Hydrostatic Column (1D)',
    label='DDG mesh',
)
fig1.savefig(os.path.join(_FIG, 'example1_hydrostatic.png'), dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {_FIG}/example1_hydrostatic.png")


# Example 2: Poiseuille Channel Flow (2D)
separator("Example 2: Poiseuille Channel Flow (2D)")

from ddgclib._boundary_conditions import (
    identify_boundary_vertices,
    DirichletPressureBC,
)
from ddgclib.initial_conditions import (
    LinearPressureGradient,
    PoiseuillePlanar,
)
from ddgclib.dynamic_integrators import euler_velocity_only
from ddgclib.data import StateHistory

# Build 2D mesh on [0, 2] x [0, 1]
L, H = 2.0, 1.0
HC2 = Complex(2, domain=[(0.0, L), (0.0, H)])
HC2.triangulate()
HC2.refine_all()
HC2.refine_all()

bV2 = identify_boundary_vertices(HC2, lambda v: (
    abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L) < 1e-14 or
    abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
))
bV_walls = identify_boundary_vertices(
    HC2, lambda v: abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
)
bV_outlet = identify_boundary_vertices(HC2, lambda v: abs(v.x_a[0] - L) < 1e-14)

print(f"2D mesh: {sum(1 for _ in HC2.V)} vertices, {len(bV2)} boundary")

# BCs: no-slip walls + zero pressure outlet
bc_set = (BoundaryConditionSet()
          .add(NoSlipWallBC(dim=2), bV_walls)
          .add(DirichletPressureBC(0.0), bV_outlet))

# IC: start from rest, linear pressure gradient drives the flow
G = 2.0    # pressure gradient
mu = 0.5   # viscosity

ic2 = CompositeIC(
    ZeroVelocity(dim=2),
    LinearPressureGradient(G=G, axis=0),
    UniformMass(total_volume=L * H, rho=1.0),
)
ic2.apply(HC2, bV2)

# Mock acceleration for demo (Laplacian-like diffusion + pressure drive)
def channel_accel(v, dim=2, mu=0.5, G=2.0, **kwargs):
    """Simple acceleration: pressure drive + viscous diffusion."""
    a = np.zeros(dim)
    # Pressure gradient drives flow in x
    a[0] += G / v.m
    # Viscous diffusion (discrete Laplacian approximation)
    if v.nn:
        avg_u = np.mean([nb.u[:dim] for nb in v.nn], axis=0)
        a += mu * (avg_u - v.u[:dim])
    return a

# Run simulation with history recording
history = StateHistory(fields=['u', 'p'], record_every=100)

print("Running simulation (1000 steps)...")
t_final = euler_velocity_only(
    HC2, bV2, channel_accel,
    dt=0.001, n_steps=1000, dim=2,
    bc_set=bc_set,
    callback=history.callback,
    mu=mu, G=G,
)
print(f"Done: t = {t_final:.4f}, {history.n_snapshots} snapshots")

# Show final velocity at centerline (y = 0.5)
print("\nVelocity profile at x=1.0, varying y:")
for v in sorted(HC2.V, key=lambda v: v.x_a[1]):
    if abs(v.x_a[0] - 1.0) < 0.3:
        marker = " <-- wall" if v in bV_walls else ""
        print(f"  y={v.x_a[1]:.4f}  u_x={v.u[0]:.6f}  u_y={v.u[1]:.6f}{marker}")

# Plot results: mesh + pressure + velocity
from ddgclib.visualization import plot_scalar_field_2d, plot_vector_field_2d, plot_mesh_2d

fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_mesh_2d(HC2, bV=bV2, ax=axes[0], title='Mesh with boundary')
plot_scalar_field_2d(HC2, field='p', ax=axes[1], title=f'Pressure at t={t_final:.3f}')
plot_vector_field_2d(HC2, bV=bV2, ax=axes[2], title=f'Velocity at t={t_final:.3f}',
                     scale=None)
fig2.tight_layout()
fig2.savefig(os.path.join(_FIG, 'example2_poiseuille.png'), dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {_FIG}/example2_poiseuille.png")


# Example 3: Save/Load + History Queries
separator("Example 3: Save/Load + History Queries")

from ddgclib.data import save_state, load_state

# Save the final state
path = save_state(
    HC2, bV2, t=t_final,
    fields=['u', 'p', 'm'],
    path=os.path.join(_FIG, 'example_state.json'),
    extra_meta={'case': 'poiseuille_2d', 'G': G, 'mu': mu},
)
print(f"State saved to: {path}")

# Load it back
HC_loaded, bV_loaded, meta = load_state(path)
print(f"State loaded: t={meta['time']}, dim={meta['dim']}, "
      f"case={meta.get('case')}, G={meta.get('G')}")
print(f"  Vertices: {sum(1 for _ in HC_loaded.V)}, "
      f"Boundary: {len(bV_loaded)}")

# Verify a field round-trips correctly
v_orig = next(iter(HC2.V))
v_key = tuple(float(x) for x in v_orig.x_a)
v_loaded = None
for v in HC_loaded.V:
    if tuple(float(x) for x in v.x_a) == v_key:
        v_loaded = v
        break
if v_loaded is not None:
    print(f"\n  Round-trip check at {v_key}:")
    print(f"    P original = {v_orig.p:.6f}")
    print(f"    P loaded   = {v_loaded.p:.6f}")
    print(f"    u original = {v_orig.u}")
    print(f"    u loaded   = {v_loaded.u}")

# Query history: time series at a specific vertex
print(f"\nHistory: {history.n_snapshots} snapshots at times {history.times}")

# Pick an interior vertex near the center
center_v = min(
    (v for v in HC2.V if v not in bV2),
    key=lambda v: np.linalg.norm(v.x_a - np.array([1.0, 0.5]))
)
center_key = tuple(float(x) for x in center_v.x_a)
times, u_vals = history.query_vertex(center_key, 'u')
if times:
    print(f"\nVelocity history at center vertex {center_key}:")
    for t, u in zip(times, u_vals):
        print(f"  t={t:.4f}  u_x={u[0]:.6f}")


# Example 4: DynamicSimulation runner (alternative to direct calls)
separator("Example 4: DynamicSimulation Runner")

from ddgclib.dynamic_integrators import DynamicSimulation, SimulationParams

# Reset mesh to initial state
ic2.apply(HC2, bV2)

params = SimulationParams(dt=0.001, n_steps=200, dim=2, mu=mu, extra={'G': G})
history2 = StateHistory(fields=['u'], record_every=100)

sim = (DynamicSimulation(HC2, bV2, params)
       .set_initial_conditions(ic2)         # re-applies IC before running
       .set_boundary_conditions(bc_set)
       .set_integrator(euler_velocity_only)
       .set_acceleration_fn(channel_accel))

t = sim.run(callback=history2.callback)
print(f"DynamicSimulation finished: t = {t:.4f}")
print(f"Snapshots recorded: {history2.n_snapshots}")


# Example 5: Adaptive time stepping
separator("Example 5: Adaptive Euler with CFL")

from ddgclib.dynamic_integrators import euler_adaptive

# Reset
ic2.apply(HC2, bV2)
# Give some initial velocity to trigger CFL
for v in HC2.V:
    if v not in bV2:
        v.u = np.array([0.5, 0.0])

dt_log = []
def log_callback(step, t, hc, bv, diag):
    if diag and 'dt' in diag:
        dt_log.append((t, diag['dt']))

t_adapt = euler_adaptive(
    HC2, bV2, channel_accel,
    dt_initial=0.01, t_end=0.05, dim=2,
    bc_set=bc_set,
    cfl_target=0.5,
    callback=log_callback,
    mu=mu, G=G,
)

print(f"Adaptive Euler reached t = {t_adapt:.6f} in {len(dt_log)} steps")
if dt_log:
    dts = [d[1] for d in dt_log]
    print(f"  dt range: [{min(dts):.6f}, {max(dts):.6f}]")
    print(f"  First 5 steps: {['%.6f' % d for d in dts[:5]]}")


# Summary
separator("Summary of New Features Used")

print("""
Modules demonstrated:
  ddgclib.initial_conditions     — CompositeIC, ZeroVelocity, HydrostaticPressure,
                                   PoiseuillePlanar, LinearPressureGradient, UniformMass
  ddgclib._boundary_conditions   — identify_boundary_vertices, identify_cube_boundaries,
                                   BoundaryConditionSet, NoSlipWallBC, DirichletPressureBC
  ddgclib.dynamic_integrators    — euler_velocity_only, euler_adaptive,
                                   DynamicSimulation, SimulationParams
  ddgclib.data                   — save_state, load_state, StateHistory
  ddgclib.visualization          — plot_scalar_field_1d, plot_scalar_field_2d,
                                   plot_vector_field_2d, plot_mesh_2d

Output files (in cases_dynamic/Template/fig/):
  fig/example1_hydrostatic.png  — 1D hydrostatic pressure plot
  fig/example2_poiseuille.png   — 2D Poiseuille mesh + pressure + velocity
  fig/example_state.json        — Saved simulation state (JSON)
""")
