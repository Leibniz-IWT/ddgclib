#!/usr/bin/env python
"""
Cube flow case study: 3D uniform flow through a cube domain.

Demonstrates the Lagrangian simulation pipeline in 3D:
  - PeriodicInletBC + OutletDeleteBC on a cube [0, L]^3
  - euler() integrator (mesh advects with velocity)
  - Time-based state recording and visualization
  - Optional polyscope 3D visualization

Run from the project root::

    python cases_dynamic/cube_flow/cube_flow_3D.py

Output files (in cases_dynamic/cube_flow/):
    fig/initial_state_3D.png            -- Initial pressure + velocity snapshot
    fig/final_state_3D.png              -- Final pressure + velocity snapshot
    fig/cube_flow_3D.gif                -- Animated flow (matplotlib)
    fig/cube_flow_3D_polyscope.mp4      -- Polyscope animation (if installed)
    fig/ps_frames_3D/                   -- Polyscope per-frame screenshots
    results/snapshots_3D/               -- Per-timestep JSON state files
    results/final_state_3D.json         -- Final state (loadable via load_state)

Note: 3D simulation is significantly slower than 1D/2D due to larger meshes.
      Use n_refine=0 for faster iteration during development.

Tuneable parameters: see cube_flow_2D.py docstring.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

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


# Step 1: Setup
separator("Step 1: Setup (3D)")

from cases_dynamic.cube_flow.src._setup import setup_cube_flow

dim = 3
L = 1.0
HC, bV, ic, bc_set, unit_mesh, params = setup_cube_flow(
    dim=dim, n_refine=1, L=L, u_inlet=0.05, G=0.001, mu=0.1, rho=1.0,
)
ic.apply(HC, bV)

n_verts = sum(1 for _ in HC.V)
print(f"Mesh: {n_verts} vertices, {len(bV)} boundary")
print(f"Params: dim={params['dim']}, L={params['L']}, "
      f"u_inlet={params['u_inlet']}, G={params['G']}")


# Step 2: Plot initial state
separator("Step 2: Plot initial state")

from ddgclib.visualization import plot_fluid

fig_init, axes_init = plot_fluid(
    HC, bV, t=0.0, face_alpha=face_alpha,
    save_path=os.path.join(_FIG, 'initial_state_3D.png'),
)
print(f"Initial state plotted: {_FIG}/initial_state_3D.png")

# Polyscope snapshot of initial state (optional)
try:
    from ddgclib.visualization import plot_fluid_ps
    plot_fluid_ps(
        HC, bV, t=0.0,
        save_path=os.path.join(_FIG, 'initial_state_3D_ps.png'),
    )
    print(f"Polyscope snapshot: {_FIG}/initial_state_3D_ps.png")
except ImportError:
    print("polyscope not installed, skipping initial polyscope snapshot")
except Exception as e:
    print(f"Polyscope initial snapshot failed (non-critical): {e}")


# Step 3: Acceleration function

def dudt_fn(v, dim=3, mu=0.1, G=0.001, **kwargs):
    """Pressure drive + viscous diffusion (3D)."""
    a = np.zeros(dim)
    a[0] += G / v.m
    if v.nn:
        avg_u = np.mean([nb.u[:dim] for nb in v.nn], axis=0)
        a += mu * (avg_u - v.u[:dim])
    return a

# To use real DDG operators instead of the mock:
# from hyperct.ddg import compute_vd
# from ddgclib.operators.gradient import acceleration
# compute_vd(HC)
# dudt_fn = acceleration


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
    save_dir=os.path.join(_RESULTS, 'snapshots_3D'),
)


def simulation_callback(step, t, HC, bV, diagnostics):
    bV.clear()
    bV.update(identify_cube_boundaries(HC, 0.0, L, dim=dim))
    history.callback(step, t, HC, bV, diagnostics)
    if step % 1000 == 0:
        n = sum(1 for _ in HC.V)
        print(f"  step={step:5d}  t={t:.3f}  vertices={n}")


print(f"Parameters: dt={dt}, n_steps={n_steps}")
t_final = euler(
    HC, bV, dudt_fn,
    dt=dt, n_steps=n_steps, dim=dim,
    bc_set=bc_set,
    callback=simulation_callback,
    mu=params['mu'], G=params['G'],
)

print(f"\nDone: t_final={t_final:.4f}, {history.n_snapshots} snapshots")


# Step 5: Post-process
separator("Step 5: Post-process")

from ddgclib.data import save_state

save_state(
    HC, bV, t=t_final, fields=['u', 'p', 'm'],
    path=os.path.join(_RESULTS, 'final_state_3D.json'),
)
print(f"Final state saved: {_RESULTS}/final_state_3D.json")

fig_final, axes_final = plot_fluid(
    HC, bV, t=t_final, face_alpha=face_alpha,
    save_path=os.path.join(_FIG, 'final_state_3D.png'),
)
print(f"Final state plotted: {_FIG}/final_state_3D.png")

# Matplotlib animation
try:
    from ddgclib.visualization import dynamic_plot_fluid
    anim = dynamic_plot_fluid(
        history, HC, bV, face_alpha=face_alpha,
        save_path=os.path.join(_FIG, 'cube_flow_3D.gif'),
        fps=20, writer='pillow',
    )
    print(f"Animation saved: {_FIG}/cube_flow_3D.gif")
except Exception as e:
    print(f"Matplotlib animation failed (non-critical): {e}")

# Polyscope animation (optional, uses surface mesh with triangles)
try:
    from ddgclib.visualization import dynamic_plot_fluid_polyscope
    ps_frame_dir = os.path.join(_FIG, 'ps_frames_3D')
    frames = dynamic_plot_fluid_polyscope(
        history, HC,
        scalar_fields=['p'],
        vector_fields=['u'],
        frame_dir=ps_frame_dir,
        video_path=os.path.join(_FIG, 'cube_flow_3D_polyscope.mp4'),
        fps=20,
    )
    print(f"Polyscope frames saved: {ps_frame_dir}/ ({len(frames)} frames)")
except ImportError:
    print("polyscope not installed, skipping 3D polyscope visualization")
except Exception as e:
    print(f"Polyscope visualization failed (non-critical): {e}")

print(f"\n3D cube flow complete: {history.n_snapshots} snapshots, "
      f"t_final={t_final:.2f}")
