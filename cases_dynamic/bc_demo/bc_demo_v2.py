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

Dual-mesh strategy
------------------
Barycentric duals are used throughout.

``compute_vd(HC, method='barycentric')``   from ``hyperct.ddg``
    Populates ``v.vd`` on every primal vertex and ``HC.Vd`` with the
    barycentric dual vertex cache.  Requires ``v.nn`` to be populated.

``_rebuild_nn_from_delaunay(HC)``   from ``ddgclib.visualization.unified``
    Computes a single fresh Delaunay triangulation of the current vertex
    positions, wires the adjacency into every ``v.nn`` set, and returns
    ``(tri, simplex_verts)``.  Called ONLY when the vertex count changes
    (injection / deletion events).  Between events the cached
    ``simplex_verts`` (vertex-reference tuples) give stable connectivity.

Connectivity cache
------------------
``_topo`` tracks vertex counts.  When a count changes:
  1. ``_rebuild_nn_from_delaunay`` rebuilds ``v.nn`` and stores
     ``simplex_verts`` (list of ``(va, vb, vc)`` vertex-object triples).
  2. ``v.boundary`` is refreshed for newly injected vertices.
``compute_vd`` is called just before each plot so dual positions reflect
the current primal positions (vertices have moved by advection since the
last topology rebuild).

Frame rendering (``plot_frame``)
---------------------------------
Layer 0  ``_render_faces_from_simplex_verts`` — light-blue primal faces
Layer 1  ``_render_edges_from_simplex_verts`` — primal edge line collection
Layer 2  ``_render_dual_from_vd``             — barycentric dual edges/verts
Layer 3  ``ax.scatter``                        — primal verts coloured by x

All four helpers are from ``ddgclib.visualization.unified``.
``plot_dual_mesh_2D`` from ``hyperct.ddg.plot_dual`` is used for the
ghost-mesh panel (its ``v.nn`` is populated by the original triangulation).

Usage:
    cd cases_dynamic/bc_demo
    python bc_demo.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# --- barycentric dual computation (hyperct.ddg) ----------------------------
from hyperct.ddg import compute_vd                          # barycentric by default
from hyperct.ddg.plot_dual import plot_dual_mesh_2D         # for ghost-mesh panel

# --- visualization helpers (ddgclib.visualization.unified) -----------------
# path: from ddgclib.visualization.unified
from ddgclib.visualization.unified import (
    _rebuild_nn_from_delaunay,          # builds v.nn + returns (tri, simplex_verts)
    _render_faces_from_simplex_verts,   # Layer 0: primal faces (vertex refs)
    _render_edges_from_simplex_verts,   # Layer 1: primal edges (vertex refs)
    _render_dual_from_vd,               # Layer 2: barycentric dual from v.vd
)

# ============================================================
# Parameters
# ============================================================
H         = 1.0   # channel height
L_period  = 1.0   # periodic unit cell length (inlet period)
L_domain  = 3.0   # total domain length
U         = 0.5   # constant advection velocity [m/s]
dt        = 0.05  # time step
n_steps   = 120   # total steps (~2 full periods)
n_refine  = 2     # mesh refinement level

d = 2

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG  = os.path.join(_HERE, 'fig')
os.makedirs(_FIG, exist_ok=True)

# ============================================================
# Step 1: Build unit mesh, tile to form main domain
# ============================================================
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

HC = Complex(d, domain=[(0.0, L_domain), (0.0, H)])
existing_positions = {}

def _add_vertex(pos_tuple, src_v):
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
print(f"Main mesh: {n_main} vertices  (tiled {n_tiles}x unit)")
print(f"Interior per period: {n_int_unit}")

# ============================================================
# Step 2: Set up BCs
# ============================================================
wall_tol = 1e-10
wall_criterion = lambda v: (
    abs(v.x_a[1]) < wall_tol or abs(v.x_a[1] - H) < wall_tol
)

inlet_bc = PeriodicInletBC(
    unit_mesh=unit_mesh, velocity=U, axis=0, inlet_pos=0.0,
    cdist=1e-10, fields=['u', 'p', 'm'], period=L_period,
)
outlet_bc = OutletDeleteBC(outlet_pos=L_domain, axis=0, bV=bV)
wall_bc   = PositionalNoSlipWallBC(criterion_fn=wall_criterion, dim=d, bV=bV)

bc_set = BoundaryConditionSet()
bc_set.add(wall_bc, None)
bc_set.add(outlet_bc, None)
bc_set.add(inlet_bc, None)

print(f"BCs: outlet at x={L_domain}, periodic inlet (period={L_period})")

# ============================================================
# Connectivity cache — rebuilt only on topology change
# ============================================================
# simplex_verts : list of (va, vb, vc) vertex-object triples.
# Holds references so v.x_a is always current without re-triangulating.

_topo = {
    'n_main':   -1,
    'sv_main':  [],    # simplex_verts for HC
    'n_ghost':  -1,
    'sv_ghost': [],    # simplex_verts for inlet_bc.ghost
}


def _refresh_main_topo():
    """Rebuild v.nn and simplex_verts for HC when vertex count changed."""
    _, sv = _rebuild_nn_from_delaunay(HC)
    _topo['sv_main'] = sv
    _topo['n_main']  = sum(1 for _ in HC.V)
    # refresh v.boundary for any freshly injected vertices
    for v in HC.V:
        v.boundary = v in bV


def _refresh_ghost_topo(ghost_mesh):
    """Rebuild topology and mark boundaries for the ghost mesh."""
    if not ghost_mesh.V:
        return

    # 1. Rebuild connectivity (v.nn)
    _, sv = _rebuild_nn_from_delaunay(ghost_mesh)
    _topo['sv_ghost'] = sv
    _topo['n_ghost'] = sum(1 for _ in ghost_mesh.V)

    # 2. IDENTIFY BOUNDARIES:
    # Vertices at x=0, x=-L_period, y=0, or y=H are boundary.
    # This prevents compute_vd from trying to "close" duals on the edges.
    for v in ghost_mesh.V:
        x, y = v.x_a[0], v.x_a[1]
        v.boundary = (
                abs(x) < 1e-10 or
                abs(x + L_period) < 1e-10 or
                abs(y) < 1e-10 or
                abs(y - H) < 1e-10
        )

    # 3. Compute dual (only for interior vertices)
    global _ghost_vd_ok
    try:
        compute_vd(ghost_mesh, method='barycentric')
        _ghost_vd_ok = True
    except Exception:
        # For extremely sparse meshes (like 5 verts), a dual may still be
        # topologically impossible. We catch it silently to keep the logs clean.
        _ghost_vd_ok = False


# --- Initialization Section ---
# Initialise both meshes
_refresh_main_topo()
_refresh_ghost_topo(inlet_bc.ghost)



# ============================================================
# BC application with per-step diagnostics
# ============================================================
cumulative_deleted  = 0
cumulative_injected = 0


def apply_bcs_measured(HC, bV, dt):
    global cumulative_deleted, cumulative_injected

    wall_ids_pre = {id(v) for v in HC.V if wall_criterion(v)}
    n_wall_pre   = len(wall_ids_pre)

    # 1. Wall BC
    wall_bc.apply(HC, dt)

    # 2. Outlet BC — interior vertices only (I1: preserve wall verts)
    n0 = sum(1 for _ in HC.V)
    to_delete = [v for v in list(HC.V)
                 if v.x_a[0] >= L_domain and not wall_criterion(v)]
    for v in to_delete:
        bV.discard(v)
        HC.V.remove(v)
    n_deleted = max(0, n0 - sum(1 for _ in HC.V))
    cumulative_deleted += n_deleted

    escaped = [v for v in HC.V if v.x_a[0] > L_domain and not wall_criterion(v)]
    if escaped:
        raise RuntimeError(
            f"I3 violated: {len(escaped)} interior vertices beyond x={L_domain}")

    # 3. Inlet BC — inject, then purge any wall vertices
    existing_ids = {id(v) for v in HC.V}
    n2 = sum(1 for _ in HC.V)
    inlet_bc.apply(HC, dt)

    n_wall_purged = 0
    for v in list(HC.V):
        if id(v) not in existing_ids and wall_criterion(v):
            HC.V.remove(v)
            bV.discard(v)
            n_wall_purged += 1

    n_injected = max(0, sum(1 for _ in HC.V) - n2)
    cumulative_injected += n_injected

    wall_ids_post = {id(v) for v in HC.V if wall_criterion(v)}
    n_wall_post   = len(wall_ids_post)
    if n_wall_post < n_wall_pre:
        raise RuntimeError(
            f"I1 violated: {n_wall_pre - n_wall_post} wall vertices lost")

    return {
        'n_deleted':     n_deleted,
        'n_injected':    n_injected,
        'n_wall_purged': n_wall_purged,
        'n_wall':        n_wall_post,
        'n_interior':    sum(1 for _ in HC.V) - n_wall_post,
    }


# ============================================================
# Frame plotting
# ============================================================

def plot_frame(step, t, HC, ghost_mesh, fig_dir):
    """
    Render one animation frame.

    Main domain (left panel)
    ------------------------
    Layer 0  _render_faces_from_simplex_verts — light-blue triangle fill.
             Uses cached simplex_verts (vertex references) so positions
             are current without re-triangulating.
    Layer 1  _render_edges_from_simplex_verts — primal edges, same cache.
    Layer 2  _render_dual_from_vd             — barycentric dual edges and
             vertices from v.vd (populated by compute_vd called just before
             plot_frame in the main loop).
    Layer 3  ax.scatter                        — primal vertices coloured
             by x-coordinate (coolwarm) so advection is visible; wall
             vertices highlighted in green on top.

    Ghost mesh (right panel)
    ------------------------
    plot_dual_mesh_2D from hyperct.ddg.plot_dual renders the full
    primal + dual picture (v.nn is already populated from triangulation).
    A coloured scatter is added on top for consistency.
    """
    fig, (ax_main, ax_ghost) = plt.subplots(
        1, 2, figsize=(14, 4),
        gridspec_kw={'width_ratios': [3, 1.5]},
    )

    # ------------------------------------------------------------------ #
    # LEFT: main domain                                                    #
    # ------------------------------------------------------------------ #
    sv_main = _topo['sv_main']

    # Layer 0: primal face fill (behind everything)
    _render_faces_from_simplex_verts(sv_main, ax_main, face_alpha=0.18)

    # Layer 1: primal edges
    _render_edges_from_simplex_verts(sv_main, ax_main,
                                      edge_color=(0.25, 0.25, 0.25),
                                      linewidth=0.5)

    # Layer 2: barycentric dual edges + dual vertices from v.vd
    _render_dual_from_vd(
        HC, ax_main,
        clip_box=(-0.05, L_domain + 0.05, -0.05, H + 0.05),
        color='tab:orange', lw=0.85, alpha=0.65,
    )

    # Layer 3: primal vertex scatter coloured by x-position
    verts_main = list(HC.V)
    if verts_main:
        coords = np.array([v.x_a[:2] for v in verts_main])
        ax_main.scatter(
            coords[:, 0], coords[:, 1],
            c=coords[:, 0], cmap='coolwarm', s=16, zorder=6,
            vmin=-L_period, vmax=L_domain + L_period,
        )
        wall_mask = np.array([wall_criterion(v) for v in verts_main])
        if wall_mask.any():
            wc = coords[wall_mask]
            ax_main.scatter(wc[:, 0], wc[:, 1],
                            c='green', s=24, marker='s', zorder=7,
                            label='Wall')

    ax_main.axvline(x=0.0,      color='blue', lw=2, ls='--', label='Inlet (x=0)')
    ax_main.axvline(x=L_domain, color='red',  lw=2, ls='--', label='Outlet (x=L)')

    n_verts = len(verts_main)
    ax_main.set_xlim(-0.3, L_domain + 0.3)
    ax_main.set_ylim(-0.15, H + 0.15)
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.set_title(
        f'Main Domain — refine={n_refine}, step={step}, '
        f't={t:.3f} s, {n_verts} verts\n'
        f'Sinj={cumulative_injected}  Sdel={cumulative_deleted}  '
        f'net={cumulative_injected - cumulative_deleted:+d}   '
        f'gray=primal  orange=barycentric dual'
    )
    ax_main.set_aspect('equal')
    ax_main.legend(loc='upper right', fontsize=7)
    ax_main.grid(True, alpha=0.15)

    # ------------------------------------------------------------------ #
    # RIGHT: ghost mesh                                                    #
    # Use plot_dual_mesh_2D (hyperct.ddg.plot_dual) for primal+dual,     #
    # then overlay coloured scatter.                                       #
    # ------------------------------------------------------------------ #
    if _ghost_vd_ok:
        # plot_dual_mesh_2D draws primal edges (blue), dual edges (orange),
        # primal vertices (blue circles), dual vertices (orange circles),
        # and primal-to-dual connections (green dashed).
        plot_dual_mesh_2D(ghost_mesh, ax=ax_ghost, show=False)
    else:
        # Fallback: render with simplex_verts helpers
        _render_faces_from_simplex_verts(_topo['sv_ghost'], ax_ghost, face_alpha=0.18)

        # Use the RGB equivalent of 'tab:blue' (or any 3-float tuple)
        _render_edges_from_simplex_verts(_topo['sv_ghost'], ax_ghost,
                                         edge_color=(0.12, 0.47, 0.71), linewidth=0.5)

    ghost_verts = list(ghost_mesh.V)
    if ghost_verts:
        gc = np.array([v.x_a[:2] for v in ghost_verts])
        ax_ghost.scatter(gc[:, 0], gc[:, 1],
                         c='tab:orange', s=16, zorder=6,
                         label='Ghost vertices')

    ax_ghost.axvline(x=0.0, color='blue', lw=2, ls='--', label='Inlet (x=0)')
    ax_ghost.set_xlim(-L_period - 0.3, 0.5)
    ax_ghost.set_ylim(-0.15, H + 0.15)
    ax_ghost.set_xlabel('x')
    ax_ghost.set_ylabel('y')
    ax_ghost.set_title(f'Ghost Mesh — {len(ghost_verts)} vertices\n'
                       f'blue=primal  orange=barycentric dual')
    ax_ghost.set_aspect('equal')
    ax_ghost.legend(loc='upper right', fontsize=7)
    ax_ghost.grid(True, alpha=0.15)

    plt.tight_layout()
    path = os.path.join(fig_dir, f'frame_{step:04d}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


# ============================================================
# Step 3: Run the advection loop
# ============================================================
print(f"\nRunning: dt={dt}, n_steps={n_steps}, U={U}")
print(f"Expected period crossing time: {L_period / U:.2f} s")

frame_dir = os.path.join(_FIG, '_frames')
os.makedirs(frame_dir, exist_ok=True)
frame_paths = []

# Compute duals for initial state then plot frame 0
compute_vd(HC, method='barycentric')
t = 0.0
fp = plot_frame(0, t, HC, inlet_bc.ghost, frame_dir)
frame_paths.append(fp)

for step in range(1, n_steps + 1):
    # -- advect interior vertices in x -----------------------------------
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

    # -- apply BCs -------------------------------------------------------
    diagnostics = apply_bcs_measured(HC, bV, dt)
    n_deleted     = diagnostics['n_deleted']
    n_injected    = diagnostics['n_injected']
    n_wall_purged = diagnostics['n_wall_purged']
    n_wall        = diagnostics['n_wall']
    n_interior    = diagnostics['n_interior']

    # -- rebuild connectivity when topology changed ----------------------
    n_now = sum(1 for _ in HC.V)
    if n_now != _topo['n_main'] or n_injected > 0 or n_deleted > 0:
        _refresh_main_topo()
        _topo['n_main'] = n_now  # Ensure the tracker stays updated

        # ADD THIS: Refresh the ghost panel when injection occurs
        if n_injected > 0:
            _refresh_ghost_topo(inlet_bc.ghost)

    t += dt
    n_verts   = sum(1 for _ in HC.V)
    imbalance = cumulative_injected - cumulative_deleted

    if step % 5 == 0 or n_injected > 0 or n_deleted > 0:
        print(f"  step {step:4d}  t={t:.3f}  "
              f"total={n_verts:4d} (int={n_interior}, wall={n_wall})  "
              f"inj={n_injected:3d}(S{cumulative_injected})  "
              f"del={n_deleted:3d}(S{cumulative_deleted})  "
              f"purged={n_wall_purged}  net={imbalance:+d}")

    if imbalance > n_unit:
        print(f"  !! WARNING step {step}: injection {imbalance} vertices ahead "
              f"of deletion (>{n_unit} = one full period surplus)")

    if step % 2 == 0:
        # Recompute barycentric duals from current primal positions
        # (positions changed by advection → dual barycenters must update)
        compute_vd(HC, method='barycentric')
        fp = plot_frame(step, t, HC, inlet_bc.ghost, frame_dir)
        frame_paths.append(fp)

# Final frame
compute_vd(HC, method='barycentric')
fp = plot_frame(n_steps, t, HC, inlet_bc.ghost, frame_dir)
frame_paths.append(fp)

print(f"\nDone: t_final={t:.3f}  final verts={sum(1 for _ in HC.V)}")
print(f"Cumulative: injected={cumulative_injected}  "
      f"deleted={cumulative_deleted}  "
      f"net={cumulative_injected - cumulative_deleted:+d}")

# ============================================================
# Step 4: Compile GIF
# ============================================================
gif_path = os.path.join(_FIG, 'bc_demo.gif')
try:
    from PIL import Image
    images = [Image.open(p) for p in frame_paths]
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=150, loop=0)
        print(f"Animation saved to {gif_path}")
except ImportError:
    print("PIL not available; frames saved but GIF not compiled")

# ============================================================
# Step 5: Summary plot (first + last frame)
# ============================================================
from matplotlib.image import imread

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.imshow(imread(frame_paths[0]));  ax1.set_title('t = 0');        ax1.axis('off')
ax2.imshow(imread(frame_paths[-1])); ax2.set_title(f't = {t:.3f}'); ax2.axis('off')
plt.suptitle('Periodic Inlet + Open Outlet BC Demo — barycentric duals', fontsize=14)
plt.tight_layout()
summary_path = os.path.join(_FIG, 'bc_demo_summary.png')
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Summary plot saved to {summary_path}")