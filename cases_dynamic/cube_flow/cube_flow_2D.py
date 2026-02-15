#!/usr/bin/env python
"""
Cube flow case study: 2D uniform flow through a square domain.

Demonstrates the full Lagrangian simulation pipeline:
  - Mesh construction and refinement
  - Uniform velocity + linear pressure gradient ICs
  - PeriodicInletBC (vertex injection) + OutletDeleteBC (vertex removal)
  - euler() integrator (Lagrangian: moves mesh with the flow)
  - Time-based state recording with disk persistence
  - Static and animated visualization

Run from the project root::

    python cases_dynamic/cube_flow/cube_flow_2D.py

Output files (in cases_dynamic/cube_flow/):
    fig/initial_state_2D.png    -- Initial pressure + velocity snapshot
    fig/final_state_2D.png      -- Final pressure + velocity snapshot
    fig/cube_flow_2D.gif        -- Animated flow (matplotlib)
    results/snapshots/          -- Per-timestep JSON state files
    results/final_state_2D.json -- Final state (loadable via load_state)

Tuneable parameters (modify at top of each section):
    n_refine  -- mesh refinement passes (default 1; increase for finer mesh)
    u_inlet   -- inlet velocity [m/s] (default 0.05)
    G         -- pressure gradient [Pa/m] (default 0.001)
    dt        -- time step [s] (default 0.01)
    t_end     -- simulation duration [s] (default 4.0)
    record_every_t -- snapshot interval [s] (default 0.1)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Ensure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')
os.makedirs(_FIG, exist_ok=True)

# --- Tuneable visualization parameter ---
face_alpha = 0.6  # Transparency for filled triangle faces (0 = hidden, 1 = opaque)


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# Step 1: Setup (domain, ICs, BCs)
separator("Step 1: Setup")

from cases_dynamic.cube_flow.src._setup import setup_cube_flow

dim = 2
L = 1.0
HC, bV, ic, bc_set, unit_mesh, params = setup_cube_flow(
    dim=dim, n_refine=1, L=L, u_inlet=0.05, G=0.001, mu=0.1, rho=1.0,
)
ic.apply(HC, bV)

n_verts = sum(1 for _ in HC.V)
print(f"Mesh: {n_verts} vertices, {len(bV)} boundary")
print(f"Params: dim={params['dim']}, L={params['L']}, "
      f"u_inlet={params['u_inlet']}, G={params['G']}, mu={params['mu']}")

# Verify initial state
print("\nInitial vertex state (first 5):")
for i, v in enumerate(sorted(HC.V, key=lambda v: tuple(v.x_a))):
    if i >= 5:
        break
    marker = " <-- boundary" if v in bV else ""
    print(f"  x={v.x_a}  u={v.u}  p={v.p:.4f}  m={v.m:.4f}{marker}")


# Step 2: Plot initial state
separator("Step 2: Plot initial state")

from ddgclib.visualization import plot_fluid

fig_init, axes_init = plot_fluid(
    HC, bV, t=0.0, face_alpha=face_alpha,
    save_path=os.path.join(_FIG, 'initial_state_2D.png'),
)
print(f"Initial state plotted: {_FIG}/initial_state_2D.png")


# Step 3: Define acceleration function

# Default: mock Laplacian (simple, no compute_vd needed)
def dudt_fn(v, dim=2, mu=0.1, G=0.001, **kwargs):
    """Pressure drive + viscous diffusion (Laplacian-like mock)."""
    a = np.zeros(dim)
    # Pressure gradient drives flow in x
    a[0] += G / v.m
    # Viscous diffusion: discrete Laplacian approximation
    if v.nn:
        avg_u = np.mean([nb.u[:dim] for nb in v.nn], axis=0)
        a += mu * (avg_u - v.u[:dim])
    return a

# To use real DDG operators instead of the mock:
# from hyperct.ddg import compute_vd
# from ddgclib.operators.gradient import acceleration
# compute_vd(HC)  # Required: compute dual mesh
# dudt_fn = acceleration  # Pass dim=dim, mu=mu as dudt_kwargs


# Step 4: Run simulation
separator("Step 4: Run simulation (4 seconds)")

from ddgclib._boundary_conditions import identify_cube_boundaries
from ddgclib.data import StateHistory
from ddgclib.dynamic_integrators import euler

dt = 0.01
t_end = 4.0
n_steps = int(t_end / dt)

history = StateHistory(
    fields=['u', 'p'],
    record_every_t=0.1,                          # ~40 snapshots (adjust to 0.01 for finer)
    save_dir=os.path.join(_RESULTS, 'snapshots'),
)


def simulation_callback(step, t, HC, bV, diagnostics):
    """Rebuild bV after vertex injection/deletion, then record history."""
    bV.clear()
    bV.update(identify_cube_boundaries(HC, 0.0, L, dim=dim))
    history.callback(step, t, HC, bV, diagnostics)
    if step % 1000 == 0:
        n = sum(1 for _ in HC.V)
        print(f"  step={step:5d}  t={t:.3f}  vertices={n}  "
              f"snapshots={history.n_snapshots}")


print(f"Parameters: dt={dt}, n_steps={n_steps}, t_end={t_end}")
print(f"Recording every 0.01s to {_RESULTS}/snapshots/")

t_final = euler(
    HC, bV, dudt_fn,
    dt=dt, n_steps=n_steps, dim=dim,
    bc_set=bc_set,
    callback=simulation_callback,
    mu=params['mu'], G=params['G'],
)

print(f"\nDone: t_final={t_final:.4f}, {history.n_snapshots} snapshots recorded")
n_verts_final = sum(1 for _ in HC.V)
print(f"Final mesh: {n_verts_final} vertices, {len(bV)} boundary")


# Step 5: Post-process and visualize
separator("Step 5: Post-process")

from ddgclib.data import save_state

# Save final state
save_state(
    HC, bV, t=t_final, fields=['u', 'p', 'm'],
    path=os.path.join(_RESULTS, 'final_state_2D.json'),
    extra_meta={'case': 'cube_flow_2D', **{k: v for k, v in params.items()
                                            if not isinstance(v, set)}},
)
print(f"Final state saved: {_RESULTS}/final_state_2D.json")

# Plot final state
fig_final, axes_final = plot_fluid(
    HC, bV, t=t_final, face_alpha=face_alpha,
    save_path=os.path.join(_FIG, 'final_state_2D.png'),
)
print(f"Final state plotted: {_FIG}/final_state_2D.png")

# Animated GIF
from ddgclib.visualization import dynamic_plot_fluid

print(f"\nGenerating animation from {history.n_snapshots} snapshots...")
try:
    anim = dynamic_plot_fluid(
        history, HC, bV, face_alpha=face_alpha,
        save_path=os.path.join(_FIG, 'cube_flow_2D.gif'),
        fps=20, writer='pillow',
    )
    print(f"Animation saved: {_FIG}/cube_flow_2D.gif")
except Exception as e:
    print(f"Animation failed (non-critical): {e}")


# Summary
separator("Summary")

print(f"""
Cube Flow 2D Case Study Complete
  Simulation time:  {t_final:.2f} s
  Time step:        {dt}
  Snapshots:        {history.n_snapshots}
  Final vertices:   {n_verts_final}

Output files:
  {_FIG}/initial_state_2D.png    — Initial pressure + velocity
  {_FIG}/final_state_2D.png      — Final pressure + velocity
  {_FIG}/cube_flow_2D.gif        — Animated flow
  {_RESULTS}/snapshots/          — Per-timestep JSON states
  {_RESULTS}/final_state_2D.json — Final state (loadable)
""")
