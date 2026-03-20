"""
Boundary Condition Demo: Periodic Inlet + Open Outlet

Visualizes the PeriodicInletBC and OutletDeleteBC working together,
independent of the stress tensor solver.  Vertices are advected at a
constant velocity in the x-direction.  The periodic inlet continuously
injects new vertices from a ghost mesh, and the outlet deletes vertices
that exit the domain.

Two side-by-side panels show:
  Left:  the full domain [0, L_domain] x [0, H] with the main mesh
  Right: the ghost (upstream) mesh that feeds the inlet

Usage:
    cd cases_dynamic/bc_demo
    python bc_demo.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from hyperct import Complex

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    PeriodicInletBC,
    OutletDeleteBC,
    PositionalNoSlipWallBC,
    identify_boundary_vertices,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    UniformVelocity,
    UniformMass,
)

# ============================================================
# Parameters
# ============================================================
H = 1.0           # channel height
L_period = 1.0    # periodic unit cell length (inlet period)
L_domain = 3.0    # total domain length (outlet at x = L_domain)
U = 0.5           # constant advection velocity [m/s]
dt = 0.05         # time step
n_steps = 120     # total steps (enough for ~2 full periods to enter)
n_refine = 1      # mesh refinement level

d = 2  # spatial dimension

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
os.makedirs(_FIG, exist_ok=True)

# ============================================================
# Step 1: Build the main domain mesh [0, L_domain] x [0, H]
# ============================================================
HC = Complex(d, domain=[(0.0, L_domain), (0.0, H)])
HC.triangulate()
for _ in range(n_refine):
    HC.refine_all()

bV = identify_boundary_vertices(HC, lambda v: (
    abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L_domain) < 1e-14 or
    abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
))
for v in HC.V:
    v.boundary = v in bV

# Set ICs on main mesh: uniform velocity to the right
main_ic = CompositeIC(
    UniformVelocity(u_vec=np.array([U, 0.0])),
    UniformMass(total_volume=L_domain * H, rho=1.0),
)
main_ic.apply(HC, bV)
# Set pressure to zero for field tracking
for v in HC.V:
    v.p = 0.0

n_main = sum(1 for _ in HC.V)
print(f"Main mesh: {n_main} vertices, domain [0, {L_domain}] x [0, {H}]")

# ============================================================
# Step 2: Build the unit mesh for the periodic inlet
# ============================================================
unit_mesh = Complex(d, domain=[(0.0, L_period), (0.0, H)])
unit_mesh.triangulate()
for _ in range(n_refine):
    unit_mesh.refine_all()

unit_bV = identify_boundary_vertices(unit_mesh, lambda v: (
    abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L_period) < 1e-14 or
    abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
))
unit_ic = CompositeIC(
    UniformVelocity(u_vec=np.array([U, 0.0])),
    UniformMass(total_volume=L_period * H, rho=1.0),
)
unit_ic.apply(unit_mesh, unit_bV)
for v in unit_mesh.V:
    v.p = 0.0

n_unit = sum(1 for _ in unit_mesh.V)
print(f"Unit mesh: {n_unit} vertices, period = {L_period}")

# ============================================================
# Step 3: Set up BCs
# ============================================================
wall_tol = 1e-10
wall_criterion = lambda v: (
    abs(v.x_a[1]) < wall_tol or abs(v.x_a[1] - H) < wall_tol
)

inlet_bc = PeriodicInletBC(
    unit_mesh=unit_mesh,
    velocity=U,
    axis=0,
    inlet_pos=0.0,
    fields=['u', 'p', 'm'],
    period=L_period,
)

outlet_bc = OutletDeleteBC(outlet_pos=L_domain, axis=0, bV=bV)

wall_bc = PositionalNoSlipWallBC(
    criterion_fn=wall_criterion, dim=d, bV=bV,
)

bc_set = BoundaryConditionSet()
bc_set.add(wall_bc, None)
bc_set.add(outlet_bc, None)
bc_set.add(inlet_bc, None)

print(f"BCs: outlet at x={L_domain}, periodic inlet (period={L_period})")

# ============================================================
# Plotting helpers
# ============================================================
def get_edges(mesh):
    """Extract edge segments from mesh for LineCollection."""
    edges = []
    seen = set()
    for v in mesh.V:
        for nb in v.nn:
            key = frozenset((v.x, nb.x))
            if key not in seen:
                edges.append([v.x_a[:2], nb.x_a[:2]])
                seen.add(key)
    return edges


def plot_frame(step, t, HC, ghost_mesh, fig_dir):
    """Plot main domain + ghost mesh side by side."""
    fig, (ax_main, ax_ghost) = plt.subplots(1, 2, figsize=(14, 4),
                                             gridspec_kw={'width_ratios': [3, 1.5]})

    # --- Main domain ---
    coords = np.array([v.x_a[:2] for v in HC.V])
    # Color by x-position to see advection
    colors_main = coords[:, 0] if len(coords) > 0 else []

    if len(coords) > 0:
        sc = ax_main.scatter(coords[:, 0], coords[:, 1], c=colors_main,
                             cmap='coolwarm', s=12, zorder=3,
                             vmin=-L_period, vmax=L_domain + L_period)

    # Draw edges
    edges = get_edges(HC)
    if edges:
        lc = LineCollection(edges, colors='gray', linewidths=0.3, alpha=0.4)
        ax_main.add_collection(lc)

    # Mark boundaries
    ax_main.axvline(x=0.0, color='blue', linewidth=2, linestyle='--',
                    label='Inlet (x=0)')
    ax_main.axvline(x=L_domain, color='red', linewidth=2, linestyle='--',
                    label='Outlet (x=L)')

    # Mark wall vertices
    wall_x = [v.x_a[0] for v in HC.V if wall_criterion(v)]
    wall_y = [v.x_a[1] for v in HC.V if wall_criterion(v)]
    if wall_x:
        ax_main.scatter(wall_x, wall_y, c='green', s=20, marker='s',
                        zorder=4, label='Wall (no-slip)')

    n_verts = sum(1 for _ in HC.V)
    ax_main.set_xlim(-0.3, L_domain + 0.3)
    ax_main.set_ylim(-0.15, H + 0.15)
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.set_title(f'Main Domain — step {step}, t = {t:.3f} s, '
                      f'{n_verts} vertices')
    ax_main.set_aspect('equal')
    ax_main.legend(loc='upper right', fontsize=7)
    ax_main.grid(True, alpha=0.2)

    # --- Ghost mesh ---
    ghost_coords = np.array([v.x_a[:2] for v in ghost_mesh.V])
    if len(ghost_coords) > 0:
        ax_ghost.scatter(ghost_coords[:, 0], ghost_coords[:, 1],
                         c='orange', s=15, zorder=3, label='Ghost vertices')

    ghost_edges = get_edges(ghost_mesh)
    if ghost_edges:
        lc_g = LineCollection(ghost_edges, colors='orange', linewidths=0.5,
                              alpha=0.5)
        ax_ghost.add_collection(lc_g)

    ax_ghost.axvline(x=0.0, color='blue', linewidth=2, linestyle='--',
                     label='Inlet (x=0)')
    ax_ghost.set_xlim(-L_period - 0.3, 0.5)
    ax_ghost.set_ylim(-0.15, H + 0.15)
    ax_ghost.set_xlabel('x')
    ax_ghost.set_ylabel('y')
    n_ghost = sum(1 for _ in ghost_mesh.V)
    ax_ghost.set_title(f'Ghost Mesh — {n_ghost} vertices')
    ax_ghost.set_aspect('equal')
    ax_ghost.legend(loc='upper right', fontsize=7)
    ax_ghost.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(fig_dir, f'frame_{step:04d}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


# ============================================================
# Step 4: Run the advection loop (no solver — just BC demo)
# ============================================================
print(f"\nRunning: dt={dt}, n_steps={n_steps}, U={U}")
print(f"Expected period crossing time: {L_period / U:.2f} s")

frame_dir = os.path.join(_FIG, '_frames')
os.makedirs(frame_dir, exist_ok=True)
frame_paths = []

t = 0.0
# Initial frame
fp = plot_frame(0, t, HC, inlet_bc.ghost, frame_dir)
frame_paths.append(fp)

for step in range(1, n_steps + 1):
    # Advect all non-wall interior vertices at constant velocity
    for v in list(HC.V):
        if not wall_criterion(v):
            pos = v.x_a.copy()
            pos[0] += U * dt
            # Preserve bV membership
            if v in bV:
                bV.remove(v)
                HC.V.move(v, tuple(pos))
                bV.add(v)
            else:
                HC.V.move(v, tuple(pos))

    # Apply BCs: wall -> outlet delete -> periodic inlet inject
    diagnostics = bc_set.apply_all(HC, bV, dt)

    t += dt

    n_verts = sum(1 for _ in HC.V)
    n_deleted = diagnostics.get('bc_1_OutletDeleteBC', 0)
    n_injected = diagnostics.get('bc_2_PeriodicInletBC', 0)
    n_wall = diagnostics.get('bc_0_PositionalNoSlipWallBC', 0)

    if step % 5 == 0 or n_injected > 0 or n_deleted > 0:
        print(f"  step {step:4d}  t={t:.3f}  verts={n_verts:4d}  "
              f"injected={n_injected}  deleted={n_deleted}  wall={n_wall}")

    # Save frame every 2 steps for animation
    if step % 2 == 0:
        fp = plot_frame(step, t, HC, inlet_bc.ghost, frame_dir)
        frame_paths.append(fp)

# Final frame
fp = plot_frame(n_steps, t, HC, inlet_bc.ghost, frame_dir)
frame_paths.append(fp)

print(f"\nDone: t_final = {t:.3f} s, {sum(1 for _ in HC.V)} vertices remaining")
print(f"Saved {len(frame_paths)} frames to {frame_dir}/")

# ============================================================
# Step 5: Compile GIF
# ============================================================
gif_path = os.path.join(_FIG, 'bc_demo.gif')
try:
    from PIL import Image
    images = [Image.open(p) for p in frame_paths]
    if images:
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=150, loop=0,
        )
        print(f"Animation saved to {gif_path}")
except ImportError:
    print("PIL not available; frames saved but GIF not compiled")

# ============================================================
# Step 6: Summary plot (first + last frame side by side)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Read first and last frames as images
from matplotlib.image import imread
img_first = imread(frame_paths[0])
img_last = imread(frame_paths[-1])

ax1.imshow(img_first)
ax1.set_title('t = 0')
ax1.axis('off')

ax2.imshow(img_last)
ax2.set_title(f't = {t:.3f}')
ax2.axis('off')

plt.suptitle('Periodic Inlet + Open Outlet BC Demo', fontsize=14)
plt.tight_layout()
summary_path = os.path.join(_FIG, 'bc_demo_summary.png')
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Summary plot saved to {summary_path}")
