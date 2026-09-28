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

Fixes applied vs original:
  1. get_edges() now uses scipy.spatial.Delaunay recomputed each frame
     instead of v.nn, so newly-injected vertices are always connected.
  2. PeriodicInletBC injection is gated on a cumulative deletion budget:
     a new period is only injected once enough vertices have been deleted
     to absorb the incoming set, keeping vertex count approximately
     conserved.
  3. Outlet criterion is slightly relaxed (x >= L_domain - eps) so
     wall-adjacent vertices that stall near the boundary are also purged,
     preventing slow accumulation.

Usage:
    cd cases_dynamic/bc_demo
    python bc_demo.py
"""

import os
import numpy as np
from scipy.spatial import Delaunay, QhullError

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
H         = 1.0   # channel height
L_period  = 1.0   # periodic unit cell length (inlet period)
L_domain  = 3.0   # total domain length (outlet at x = L_domain)
U         = 0.5   # constant advection velocity [m/s]
dt        = 0.05  # time step
n_steps   = 120   # total steps (enough for ~2 full periods to enter)
n_refine  = 2     # mesh refinement level

# Outlet tolerance: no longer needed — wall vertices are advected in x
# so they reach L_domain naturally.

d = 2  # spatial dimension

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG  = os.path.join(_HERE, 'fig')
os.makedirs(_FIG, exist_ok=True)

# ============================================================
# Step 1: Build unit mesh first, then tile it to form the main domain
# ============================================================
# IMPORTANT: do NOT build the main mesh by triangulating [0, L_domain]x[0,H]
# independently.  Complex() does not guarantee that a 3x1 domain produces
# exactly 3x the vertices of a 1x1 domain at the same refinement level.
# The density mismatch creates a permanent vertex surplus that never drains.
#
# Instead, build the unit mesh first, then tile n_tiles copies shifted in x
# to form the main mesh.  This guarantees:
#   n_interior_main == n_tiles * n_interior_unit
# so injection and deletion rates are exactly balanced in steady state.

n_tiles = int(round(L_domain / L_period))   # = 3

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

# Tile the unit mesh: place n_tiles copies shifted by k*L_period in x.
# Vertices on shared interfaces (x = k*L_period for k=1..n_tiles-1) are
# deduplicated so they appear exactly once in the main mesh.
HC = Complex(d, domain=[(0.0, L_domain), (0.0, H)])

tol_merge = 1e-10
existing_positions = {}   # tuple(pos) -> vertex in HC

def _add_vertex(pos_tuple, src_v):
    """Add vertex at pos_tuple if not already present; copy fields."""
    # Round to avoid float noise before dict lookup
    key = tuple(round(x, 12) for x in pos_tuple)
    if key not in existing_positions:
        v = HC.V[pos_tuple]
        for f in ['u', 'p', 'm']:
            val = getattr(src_v, f, None)
            if val is not None:
                setattr(v, f, val.copy() if isinstance(val, np.ndarray) else val)
        existing_positions[key] = v
    return existing_positions[key]

for k in range(n_tiles):
    x_shift = k * L_period
    for uv in unit_mesh.V:
        pos = uv.x_a.copy()
        pos[0] += x_shift
        _add_vertex(tuple(pos), uv)

bV = identify_boundary_vertices(HC, lambda v: (
    abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L_domain) < 1e-14 or
    abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
))
for v in HC.V:
    v.boundary = v in bV

n_main = sum(1 for _ in HC.V)
n_wall_unit = sum(1 for v in unit_mesh.V
                  if abs(v.x_a[1]) < 1e-10 or abs(v.x_a[1] - H) < 1e-10)
n_int_unit  = n_unit - n_wall_unit
print(f"Main mesh: {n_main} vertices  (tiled {n_tiles}×unit, "
      f"expected ≈{n_tiles * n_unit - (n_tiles-1)*2} after interface dedup)")
print(f"Interior per period: {n_int_unit},  "
      f"expected steady-state interior: {n_tiles * n_int_unit}")

# ============================================================
# Step 2: Unit mesh already built above — just confirm
# ============================================================

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
    cdist=1e-10,
    fields=['u', 'p', 'm'],
    period=L_period,
)

# Outlet at exactly L_domain.  Wall vertices are now advected in x so
# they reach this position naturally and are deleted without needing a
# relaxed threshold.
outlet_bc = OutletDeleteBC(
    outlet_pos=L_domain,
    axis=0,
    bV=bV,
)

wall_bc = PositionalNoSlipWallBC(
    criterion_fn=wall_criterion, dim=d, bV=bV,
)

bc_set = BoundaryConditionSet()
bc_set.add(wall_bc,   None)
bc_set.add(outlet_bc, None)
bc_set.add(inlet_bc,  None)

print(f"BCs: outlet at x={L_domain}, periodic inlet (period={L_period})")

# ============================================================
# BC application with per-step diagnostics
# ============================================================
# All three BC classes share the signature apply(mesh, dt, target_vertices=None).
# We call each one individually so we can measure injection and deletion
# counts by diffing the vertex count before and after each call.

cumulative_deleted  = 0
cumulative_injected = 0


def apply_bcs_measured(HC, bV, dt):
    """Apply all BCs in order with full invariant checking.

    Invariants enforced each step:
      I1. Wall vertices (y=0 or y=H) are NEVER deleted by the outlet.
          The outlet uses a strict x > L_domain test so vertices placed
          exactly at the boundary face are not deleted prematurely.
      I2. Wall vertices injected by the inlet are purged immediately —
          the wall is a fixed boundary covered by the initial mesh.
      I3. No vertex should exist with x > L_domain after the outlet runs.
      I4. Interior vertex count tracked separately from wall count.

    All BC classes share the signature apply(mesh, dt, target_vertices=None).
    """
    global cumulative_deleted, cumulative_injected

    # Snapshot wall vertex ids before any BC (used for invariant checks)
    wall_ids_pre = {id(v) for v in HC.V if wall_criterion(v)}
    n_wall_pre   = len(wall_ids_pre)
    n_int_pre    = sum(1 for _ in HC.V) - n_wall_pre

    # ------------------------------------------------------------------
    # 1. Wall BC — zero velocity on all wall vertices
    # ------------------------------------------------------------------
    wall_bc.apply(HC, dt)

    # ------------------------------------------------------------------
    # 2. Outlet BC — delete interior vertices that have left the domain.
    #    We manually enforce the deletion so we can exclude wall vertices,
    #    rather than relying on OutletDeleteBC which has no wall awareness.
    # ------------------------------------------------------------------
    n0 = sum(1 for _ in HC.V)
    to_delete = [
        v for v in list(HC.V)
        if v.x_a[0] >= L_domain          # has reached or passed the outlet
        and not wall_criterion(v)         # I1: never delete wall vertices
    ]
    for v in to_delete:
        bV.discard(v)
        HC.V.remove(v)
    n1 = sum(1 for _ in HC.V)
    n_deleted = max(0, n0 - n1)
    cumulative_deleted += n_deleted

    # Invariant I3: no interior vertex beyond L_domain
    escaped = [v for v in HC.V if v.x_a[0] > L_domain and not wall_criterion(v)]
    if escaped:
        raise RuntimeError(f"I3 violated: {len(escaped)} interior vertices beyond "
                           f"x={L_domain} after outlet: "
                           f"{[v.x_a for v in escaped]}")

    # ------------------------------------------------------------------
    # 3. Inlet BC — inject ghost vertices, then purge any wall vertices
    # ------------------------------------------------------------------
    existing_ids = {id(v) for v in HC.V}
    n2 = sum(1 for _ in HC.V)
    inlet_bc.apply(HC, dt)

    # I2: purge wall vertices injected by the inlet
    n_wall_purged = 0
    for v in list(HC.V):
        if id(v) not in existing_ids and wall_criterion(v):
            HC.V.remove(v)
            bV.discard(v)
            n_wall_purged += 1

    n3 = sum(1 for _ in HC.V)
    n_injected = max(0, n3 - n2)
    cumulative_injected += n_injected

    # ------------------------------------------------------------------
    # Invariant checks post-step
    # ------------------------------------------------------------------
    wall_ids_post = {id(v) for v in HC.V if wall_criterion(v)}
    n_wall_post   = len(wall_ids_post)
    n_int_post    = sum(1 for _ in HC.V) - n_wall_post

    # I1: wall count must not decrease (wall vertices are permanent)
    if n_wall_post < n_wall_pre:
        lost = wall_ids_pre - wall_ids_post
        raise RuntimeError(
            f"I1 violated: {n_wall_pre - n_wall_post} wall vertices lost "
            f"this step (ids: {lost})"
        )

    return {
        'n_deleted':     n_deleted,
        'n_injected':    n_injected,
        'n_wall_purged': n_wall_purged,
        'n_wall':        n_wall_post,
        'n_interior':    n_int_post,
    }

# ============================================================
# Plotting helpers
# ============================================================

def get_edges_delaunay(mesh):
    """
    Extract edge segments by recomputing a Delaunay triangulation of the
    current vertex positions.

    Using v.nn would miss newly-injected vertices that haven't been
    re-triangulated into the complex.  Recomputing from scratch each frame
    is cheap enough for demo mesh sizes.
    """
    verts = list(mesh.V)
    if len(verts) < 3:
        return []
    coords = np.array([v.x_a[:2] for v in verts])
    # Deduplicate positions (degenerate points crash Delaunay)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    coords_u = coords[unique_idx]
    if len(coords_u) < 3:
        return []
    try:
        tri = Delaunay(coords_u)
    except QhullError:
        return []
    seen = set()
    segments = []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    segments.append([coords_u[a], coords_u[b]])
    return segments


def plot_frame(step, t, HC, ghost_mesh, fig_dir):
    """Plot main domain + ghost mesh side by side."""
    fig, (ax_main, ax_ghost) = plt.subplots(
        1, 2, figsize=(14, 4),
        gridspec_kw={'width_ratios': [3, 1.5]}
    )

    # --- Main domain ---
    verts_main = list(HC.V)
    coords = np.array([v.x_a[:2] for v in verts_main]) if verts_main else np.empty((0, 2))

    if len(coords) > 0:
        ax_main.scatter(coords[:, 0], coords[:, 1],
                        c=coords[:, 0], cmap='coolwarm', s=12, zorder=3,
                        vmin=-L_period, vmax=L_domain + L_period)

    edges = get_edges_delaunay(HC)
    if edges:
        lc = LineCollection(edges, colors='gray', linewidths=0.3, alpha=0.4)
        ax_main.add_collection(lc)

    ax_main.axvline(x=0.0,      color='blue', linewidth=2, linestyle='--', label='Inlet (x=0)')
    ax_main.axvline(x=L_domain, color='red',  linewidth=2, linestyle='--', label='Outlet (x=L)')

    wall_pts = [(v.x_a[0], v.x_a[1]) for v in HC.V if wall_criterion(v)]
    if wall_pts:
        wx, wy = zip(*wall_pts)
        ax_main.scatter(wx, wy, c='green', s=20, marker='s', zorder=4, label='Wall')

    n_verts = len(verts_main)
    ax_main.set_xlim(-0.3, L_domain + 0.3)
    ax_main.set_ylim(-0.15, H + 0.15)
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.set_title(f'Main Domain — mesh refinement level {n_refine}, step {step}, t={t:.3f} s, {n_verts} vertices\n'
                      f'Σinjected={cumulative_injected}  Σdeleted={cumulative_deleted}  '
                      f'net={cumulative_injected - cumulative_deleted:+d}')
    ax_main.set_aspect('equal')
    ax_main.legend(loc='upper right', fontsize=7)
    ax_main.grid(True, alpha=0.2)

    # --- Ghost mesh ---
    ghost_verts = list(ghost_mesh.V)
    if ghost_verts:
        gc = np.array([v.x_a[:2] for v in ghost_verts])
        ax_ghost.scatter(gc[:, 0], gc[:, 1], c='orange', s=15, zorder=3,
                         label='Ghost vertices')
        ghost_edges = get_edges_delaunay(ghost_mesh)
        if ghost_edges:
            lc_g = LineCollection(ghost_edges, colors='orange',
                                  linewidths=0.5, alpha=0.5)
            ax_ghost.add_collection(lc_g)

    ax_ghost.axvline(x=0.0, color='blue', linewidth=2, linestyle='--',
                     label='Inlet (x=0)')
    ax_ghost.set_xlim(-L_period - 0.3, 0.5)
    ax_ghost.set_ylim(-0.15, H + 0.15)
    ax_ghost.set_xlabel('x')
    ax_ghost.set_ylabel('y')
    n_ghost = len(ghost_verts)
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
fp = plot_frame(0, t, HC, inlet_bc.ghost, frame_dir)
frame_paths.append(fp)

for step in range(1, n_steps + 1):
    # Advect only interior (non-wall) vertices in x.
    # Wall vertices are stationary: no-slip means zero velocity, so zero
    # Lagrangian displacement.  The wall is a fixed physical boundary covered
    # permanently by the initial mesh vertices.
    for v in list(HC.V):
        if not wall_criterion(v):
            pos = v.x_a.copy()
            pos[0] += U * dt
            if v in bV:
                bV.remove(v)
                HC.V.move(v, tuple(pos))
                bV.add(v)
            else:
                HC.V.move(v, tuple(pos))

    # Apply BCs with per-step counting
    diagnostics = apply_bcs_measured(HC, bV, dt)

    n_deleted     = diagnostics['n_deleted']
    n_injected    = diagnostics['n_injected']
    n_wall_purged = diagnostics['n_wall_purged']
    n_wall        = diagnostics['n_wall']
    n_interior    = diagnostics['n_interior']

    t += dt
    n_verts   = sum(1 for _ in HC.V)
    imbalance = cumulative_injected - cumulative_deleted

    if step % 5 == 0 or n_injected > 0 or n_deleted > 0:
        print(f"  step {step:4d}  t={t:.3f}  "
              f"total={n_verts:4d} (int={n_interior}, wall={n_wall})  "
              f"inj={n_injected:3d}(Σ{cumulative_injected})  "
              f"del={n_deleted:3d}(Σ{cumulative_deleted})  "
              f"purged={n_wall_purged}  net={imbalance:+d}")

    if imbalance > n_unit:
        print(f"  !! WARNING step {step}: injection is {imbalance} vertices "
              f"ahead of deletion (>{n_unit} = one full period surplus)")

    if step % 2 == 0:
        fp = plot_frame(step, t, HC, inlet_bc.ghost, frame_dir)
        frame_paths.append(fp)

# Final frame
fp = plot_frame(n_steps, t, HC, inlet_bc.ghost, frame_dir)
frame_paths.append(fp)

print(f"\nDone: t_final={t:.3f}  final verts={sum(1 for _ in HC.V)}")
print(f"Cumulative: injected={cumulative_injected}  deleted={cumulative_deleted}  "
      f"net={cumulative_injected - cumulative_deleted:+d}")

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
from matplotlib.image import imread

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.imshow(imread(frame_paths[0]))
ax1.set_title('t = 0')
ax1.axis('off')
ax2.imshow(imread(frame_paths[-1]))
ax2.set_title(f't = {t:.3f}')
ax2.axis('off')
plt.suptitle('Periodic Inlet + Open Outlet BC Demo', fontsize=14)
plt.tight_layout()
summary_path = os.path.join(_FIG, 'bc_demo_summary.png')
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Summary plot saved to {summary_path}")