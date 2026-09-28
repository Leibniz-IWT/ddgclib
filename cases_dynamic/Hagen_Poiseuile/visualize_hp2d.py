"""
Visualization for 2D Hagen-Poiseuille case study.

Loads the final simulation state and periodic snapshots from results/,
then generates all plots and animations.  Each field is plotted on its
own separate figure for clarity.

Edit ZOOM_XLIM / ZOOM_YLIM below to change the zoom window, then re-run.

Usage:
    cd cases_dynamic/Hagen_Poiseuile
    python visualize_hp2d.py
"""

import os
import glob
import json
import pickle

import numpy as np
from scipy.spatial import Delaunay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from hyperct.ddg import compute_vd

from ddgclib.initial_conditions import PoiseuillePlanar
from ddgclib.data import load_state, StateHistory
from ddgclib.visualization import (
    plot_scalar_field_2d,
    plot_vector_field_2d,
    plot_mesh_2d,
    plot_interpolated_scalar_2d,
    plot_interpolated_vector_2d,
    plot_fluid,
    plot_dual,
    dynamic_plot_fluid,
)

# Local parameters
from src._params import L, r, G, mu, rho, D, U_avg, U_max


# ============ CONFIGURABLE ZOOM WINDOW ============
ZOOM_XLIM = (L / 2 - 1.0, L / 2 + 1.0)
#ZOOM_XLIM = (0, L / 2 + 2)
ZOOM_XLIM = (0, L)
ZOOM_YLIM = (0.0, D)
# ==================================================

DPI = 150

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')
os.makedirs(_FIG, exist_ok=True)

n_refine = 1  # must match simulation script
d = 2


def prepare_loaded_state(HC, dim=2):
    """Retriangulate a loaded state and compute duals.

    Saved states store edges from the simulation snapshot, but these may
    not form a valid Delaunay triangulation after loading.  This function
    disconnects all edges, rebuilds the Delaunay, tags boundaries, and
    computes the barycentric dual mesh — the same sequence as
    ``_retopologize`` in the integrators.
    """
    verts = list(HC.V)
    if len(verts) < dim + 1:
        return

    # Disconnect all edges
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)

    # Delaunay retriangulation
    coords = np.array([v.x_a[:dim] for v in verts])
    tri = Delaunay(coords)
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                verts[simplex[i]].connect(verts[simplex[j]])

    # Recompute boundary and tag
    dV = HC.boundary()
    for v in HC.V:
        v.boundary = v in dV

    # Compute barycentric duals
    compute_vd(HC, method="barycentric")


# ============================================================
# Load simulation results
# ============================================================
print("Loading final state...")
HC, bV, meta = load_state(os.path.join(_RESULTS, 'hp2d_final_state.json'))
t_final = meta['time']

# Retriangulate and compute duals (saved edges may not form valid triangulation)
prepare_loaded_state(HC, dim=d)

n_verts = sum(1 for _ in HC.V)
print(f"Loaded: t={t_final:.4f}, dim={d}, {n_verts} vertices, {len(bV)} boundary")

# Analytical profile for comparison
poiseuille_ic = PoiseuillePlanar(
    G=G, mu=mu, y_lb=0.0, y_ub=D,
    flow_axis=0, normal_axis=1, dim=d,
)


# ============================================================
# Load history for animations
# ============================================================
history = None
history_path = os.path.join(_RESULTS, 'hp2d_history.pkl')
if os.path.exists(history_path):
    print("Loading history from pickle...")
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    print(f"  {history.n_snapshots} snapshots loaded")
else:
    # Fall back: reconstruct from saved state JSON files
    print("Reconstructing history from state files...")
    state_files = sorted(glob.glob(os.path.join(_RESULTS, 'state_*.json')))
    if state_files:
        history = StateHistory(fields=['u', 'p'], record_every=1)
        for path in state_files:
            with open(path) as f:
                state = json.load(f)
            t = state.get('time', 0.0)
            snapshot = {}
            for vdata in state['vertices']:
                key = tuple(vdata['coords'])
                snapshot[key] = {}
                for field in ('u', 'p'):
                    if field in vdata:
                        val = vdata[field]
                        snapshot[key][field] = (
                            np.array(val) if isinstance(val, list) else val
                        )
            history._snapshots.append((t, snapshot, {}))
        print(f"  {history.n_snapshots} snapshots from state files")
    else:
        print("  No history data found; animations will be skipped")


# ============================================================
# Plot 0: First timestep — confirm initial domain geometry
# ============================================================
print("Plotting first timestep mesh...")
state_files_sorted = sorted(glob.glob(os.path.join(_RESULTS, 'state_*.json')))
if state_files_sorted:
    HC_t0, bV_t0, meta_t0 = load_state(state_files_sorted[0])
    prepare_loaded_state(HC_t0, dim=d)

    t0_time = meta_t0.get('time', 0.0)
    n_t0 = sum(1 for _ in HC_t0.V)

    # Full domain mesh
    fig, ax = plot_mesh_2d(HC_t0, bV=bV_t0,
                           title=f'First saved state — t={t0_time:.4f}, {n_t0} verts',
                           xlim=(-0.5, L + 0.5), ylim=(-0.1, D + 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, 'hp2d_first_timestep.png'), dpi=DPI)
    plt.close(fig)
    print(f"  -> {_FIG}/hp2d_first_timestep.png")

    # Inlet zoom
    fig, ax = plot_mesh_2d(HC_t0, bV=bV_t0,
                           title=f'First state — inlet zoom (t={t0_time:.4f})',
                           xlim=(-0.2, 2.0), ylim=(-0.1, D + 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, 'hp2d_first_timestep_inlet.png'), dpi=DPI)
    plt.close(fig)
    print(f"  -> {_FIG}/hp2d_first_timestep_inlet.png")

    # Outlet zoom
    fig, ax = plot_mesh_2d(HC_t0, bV=bV_t0,
                           title=f'First state — outlet zoom (t={t0_time:.4f})',
                           xlim=(L - 2.0, L + 0.5), ylim=(-0.1, D + 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, 'hp2d_first_timestep_outlet.png'), dpi=DPI)
    plt.close(fig)
    print(f"  -> {_FIG}/hp2d_first_timestep_outlet.png")

    del HC_t0, bV_t0  # free memory
else:
    print("  No state files found; skipping first timestep plot")


# ============================================================
# Plot 1: Velocity profile vs analytical solution
# ============================================================
print("Plotting velocity profile...")
x_mid = L / 2.0
tol = L / (2**n_refine) * 0.6
mid_verts = sorted(
    [v for v in HC.V if abs(v.x_a[0] - x_mid) < tol],
    key=lambda v: v.x_a[1]
)

y_num = np.array([v.x_a[1] for v in mid_verts])
ux_num = np.array([v.u[0] for v in mid_verts])

y_anal = np.linspace(0.0, D, 200)
ux_anal = (G / (2 * mu)) * y_anal * (D - y_anal)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ux_anal, y_anal, 'k-', linewidth=2, label='Analytical (steady)')
ax.plot(ux_num, y_num, 'ro', markersize=5, label=f'Numerical (t={t_final:.1f})')
ax.set_xlabel('$u_x$ [m/s]')
ax.set_ylabel('$y$ [m]')
ax.set_title(f'Velocity Profile at x = L/2 (Re_D = {rho*U_avg*D/mu:.0f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_velocity_profile.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_velocity_profile.png")


# ============================================================
# Plot 2: Primal mesh
# ============================================================
print("Plotting primal mesh...")
fig, ax = plot_mesh_2d(HC, bV=bV, title='Primal Mesh (zoomed)',
                       xlim=ZOOM_XLIM, ylim=ZOOM_YLIM)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_mesh.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_mesh.png")


# ============================================================
# Plot 3: Pressure field (scatter)
# ============================================================
print("Plotting pressure field...")
fig, ax = plot_scalar_field_2d(HC, field='p', title='Pressure (zoomed)',
                               xlim=ZOOM_XLIM, ylim=ZOOM_YLIM)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_pressure.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_pressure.png")


# ============================================================
# Plot 4: Velocity field (quiver)
# ============================================================
print("Plotting velocity field...")
fig, ax = plot_vector_field_2d(HC, bV=bV, title='Velocity (zoomed)',
                               xlim=ZOOM_XLIM, ylim=ZOOM_YLIM)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_velocity.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_velocity.png")


# ============================================================
# Plot 5: Interpolated pressure (tricontourf)
# ============================================================
print("Plotting interpolated pressure...")
fig, ax = plot_interpolated_scalar_2d(
    HC, field='p', title='Pressure (pointwise, zoomed)',
    xlim=ZOOM_XLIM, ylim=ZOOM_YLIM, cmap='viridis',
)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_pressure_interpolated.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_pressure_interpolated.png")


# ============================================================
# Plot 6: Interpolated u_x velocity (tricontourf)
# ============================================================
print("Plotting interpolated velocity...")
fig, ax = plot_interpolated_vector_2d(
    HC, component=0, title='$u_x$ velocity (pointwise, zoomed)',
    xlim=ZOOM_XLIM, ylim=ZOOM_YLIM, cmap='coolwarm',
)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_velocity_interpolated.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_velocity_interpolated.png")


# ============================================================
# Plot 7: Dual complex
# ============================================================
print("Plotting dual complex...")
fig, ax = plot_dual(HC, title='Dual Mesh (zoomed)', save_path=None,
                    xlim=ZOOM_XLIM, ylim=ZOOM_YLIM)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_dual_mesh.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_dual_mesh.png")


# ============================================================
# Plot 8: Dual complex with pressure overlay
# ============================================================
print("Plotting dual complex with pressure...")
fig, ax = plot_dual(HC, scalar_field='p',
                    title='Dual + Pressure (zoomed)', save_path=None,
                    xlim=ZOOM_XLIM, ylim=ZOOM_YLIM)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_dual_pressure.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_dual_pressure.png")


# ============================================================
# Plot 9: Dual complex with velocity overlay
# ============================================================
print("Plotting dual complex with velocity...")
fig, ax = plot_dual(HC, vector_field='u',
                    title='Dual + Velocity (zoomed)', save_path=None,
                    xlim=ZOOM_XLIM, ylim=ZOOM_YLIM)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'hp2d_dual_velocity.png'), dpi=DPI)
plt.close(fig)
print(f"  -> {_FIG}/hp2d_dual_velocity.png")


# ============================================================
# Plot 10: Fluid snapshot (plot_fluid — pressure + velocity)
# ============================================================
print("Plotting fluid snapshot...")
plot_fluid(
    HC, bV=bV, t=t_final,
    save_path=os.path.join(_FIG, 'hp2d_fluid_zoomed.png'),
    xlim=ZOOM_XLIM, ylim=ZOOM_YLIM,
)
plt.close('all')
print(f"  -> {_FIG}/hp2d_fluid_zoomed.png")


# ============================================================
# Animation 1: Primal fluid evolution
# ============================================================
if history is not None and history.n_snapshots > 1:
    print("Creating primal fluid animation...")
    anim = dynamic_plot_fluid(
        history, HC, bV=bV,
        scalar_field='p', vector_field='u',
        save_path=os.path.join(_FIG, 'hp2d_evolution.gif'),
        xlim=ZOOM_XLIM, ylim=ZOOM_YLIM,
        fps=5, dpi=100,
    )
    plt.close('all')
    print(f"  -> {_FIG}/hp2d_evolution.gif")
else:
    print("Skipping primal animation (no history data)")


# ============================================================
# Animation 2: Dual complex evolution
# ============================================================
def create_dual_animation(state_dir, fig_dir, zoom_xlim, zoom_ylim,
                          fps=5, dpi=100):
    """Create dual mesh animation from saved state files.

    Each frame loads a state file, reconstructs the Complex, computes
    the dual mesh, and renders a dual plot.  Frames are compiled into
    a GIF.
    """
    from hyperct.ddg.plot_dual import plot_dual_mesh_2D

    state_files = sorted(glob.glob(os.path.join(state_dir, 'state_*.json')))
    if len(state_files) < 2:
        print("  Not enough state files for dual animation")
        return

    frame_dir = os.path.join(fig_dir, '_dual_frames')
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    for i, sf in enumerate(state_files):
        hc_i, bv_i, meta_i = load_state(sf)
        prepare_loaded_state(hc_i, dim=2)

        fig_i, ax_i = plot_dual_mesh_2D(hc_i, ax=None, show=False)
        ax_i.set_xlim(zoom_xlim)
        ax_i.set_ylim(zoom_ylim)
        ax_i.set_title(f'Dual mesh — t = {meta_i["time"]:.4f} s')
        plt.tight_layout()

        fpath = os.path.join(frame_dir, f'dual_{i:04d}.png')
        fig_i.savefig(fpath, dpi=dpi, bbox_inches='tight')
        plt.close(fig_i)
        frame_paths.append(fpath)
        print(f"  Frame {i+1}/{len(state_files)}")

    # Compile frames to GIF
    try:
        from PIL import Image
        gif_path = os.path.join(fig_dir, 'hp2d_dual_evolution.gif')
        images = [Image.open(p) for p in frame_paths]
        if images:
            images[0].save(
                gif_path, save_all=True, append_images=images[1:],
                duration=int(1000 / fps), loop=0,
            )
            print(f"  -> {gif_path}")
    except ImportError:
        print("  PIL not available; dual GIF not compiled (frames saved)")


state_files_exist = len(glob.glob(os.path.join(_RESULTS, 'state_*.json'))) >= 2
if state_files_exist:
    print("Creating dual complex animation...")
    create_dual_animation(_RESULTS, _FIG, ZOOM_XLIM, ZOOM_YLIM)
else:
    print("Skipping dual animation (no state files)")


# ============================================================
# Error summary
# ============================================================
errors = []
for v in mid_verts:
    u_anal = poiseuille_ic.analytical_velocity(v.x_a)
    errors.append(abs(v.u[0] - u_anal))

max_err = max(errors) if errors else float('nan')
l2_err = np.sqrt(np.mean(np.array(errors)**2)) if errors else float('nan')
print(f"\nError at x=L/2 vs analytical: max={max_err:.6e}, L2={l2_err:.6e}")
print(f"U_max analytical = {U_max:.6f}, U_max numerical = "
      f"{max(ux_num) if len(ux_num) > 0 else float('nan'):.6f}")
print("\nAll plots saved to", _FIG)
