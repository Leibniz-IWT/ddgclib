"""
Incompressible Poiseuille Flow — Plug Flow → Parabolic Development
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay, QhullError, cKDTree, Voronoi

from hyperct import Complex
from ddgclib._boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionSet,
    PeriodicInletBC,
    PositionalNoSlipWallBC,
    identify_boundary_vertices,
)


# Output settings

FRAME_DPI       = 180
FRAME_FIG_W     = 22
FRAME_FIG_H     = 10
SUMMARY_DPI     = 160
GIF_DURATION_MS = 80
FRAME_EVERY     = 5

matplotlib.rcParams.update({
    'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.titlesize': 16,
    'lines.linewidth': 2.0, 'axes.linewidth': 1.2, 'grid.linewidth': 0.8,
})


# Physical parameters

H        = 1.0      # channel height  [m]
L_period = 1.0      # periodic unit-cell length  [m]
L_domain = 3.0      # total domain length  [m]
mu       = 1.0      # dynamic viscosity  [Pa s]
rho      = 1.0      # density  [kg/m^3]
dPdx     = -8.0     # pressure gradient [Pa/m]  (negative -> drives +x)
nu       = mu / rho  # kinematic viscosity = 1.0

U_max    = -dPdx * H**2 / (8.0 * mu)    # = 1.0 m/s
U_mean   = (2.0 / 3.0) * U_max           # = 2/3 m/s

tau_visc = H**2 / (np.pi**2 * nu)        # approx 0.1013 s

# ── Adaptive time-stepping limits ───────────────────────────
DT_INITIAL = 0.005
DT_MAX     = 0.005
DT_MIN     = 1e-6
CFL_TARGET = 0.4    # advective CFL:   dt < CFL * dy_min / U_max
VN_COEFF   = 0.20   # von Neumann:     dt < VN  * dy_min^2 / nu
T_END      = 8.0

N_REFINE   = 1
DIM        = 2
WALL_TOL   = 1e-10

# Snapshot target times for profile comparison plot
SNAP_TIMES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

_HERE   = os.path.dirname(os.path.abspath(__file__))
_FIG    = os.path.join(_HERE, 'fig', 'poiseuille_dev')
_FRAMES = os.path.join(_FIG, '_frames')
os.makedirs(_FRAMES, exist_ok=True)

print(f"Viscous timescale tau = {tau_visc:.4f} s  "
      f"(99% developed at approx 5tau = {5*tau_visc:.3f} s)")
print(f"\nPoiseuille flow:  H={H}, L={L_domain}, mu={mu}, rho={rho}")
print(f"  dP/dx = {dPdx}  ->  U_max = {U_max:.4f} m/s,"
      f"  U_mean = {U_mean:.4f} m/s")
print(f"  Re = {rho * U_mean * H / mu:.1f}")
print(f"  t_end = {T_END:.2f} s  approx  {T_END/tau_visc:.1f} tau")


# Analytical solutions


def u_poiseuille(y):
    """Steady-state Poiseuille parabola."""
    return U_max * (1.0 - (2.0 * y / H - 1.0) ** 2)


def p_analytical(x):
    return dPdx * float(x)


# ── Precompute Fourier coefficients for zero-velocity start ───────────
#   Impulsive start from rest driven by body force -dPdx/rho:
#     u(y,t) = u_P(y) - sum_n B_n sin(n pi y/H) exp(-n^2 pi^2 nu t/H^2)
#     u(y,0) = 0  =>  u_P(y) = sum_n B_n sin(n pi y/H)
#     B_n = (2/H) int_0^H u_P(y) sin(n pi y/H) dy
_N_FOURIER = 40
_y_quad    = np.linspace(0.0, H, 4000)
_B_n       = np.empty(_N_FOURIER)
for _n in range(1, _N_FOURIER + 1):
    _intgd     = u_poiseuille(_y_quad) * np.sin(_n * np.pi * _y_quad / H)
    _B_n[_n-1] = (2.0 / H) * np.trapezoid(_intgd, _y_quad)
print(f"  Fourier coefficients precomputed (N={_N_FOURIER}, zero-start)")


def u_transient(y, t, n_terms=_N_FOURIER):
    """Exact transient: rest start -> Poiseuille (Fourier series)."""
    u = u_poiseuille(y)
    for n in range(1, min(n_terms, _N_FOURIER) + 1):
        decay = np.exp(-n**2 * np.pi**2 * nu * t / H**2)
        u = u - _B_n[n-1] * np.sin(n * np.pi * y / H) * decay
    return u



# Small helpers


def wall_criterion(v):
    y = float(v.x_a[1])
    return y < WALL_TOL or y > H - WALL_TOL


def _safe_u(v):
    """Return a clean (2,) float64 velocity for any vertex state."""
    u = np.asarray(getattr(v, 'u', np.zeros(2)), dtype=float).ravel()
    if u.size >= 2: return u[:2]
    if u.size == 1: return np.array([u[0], 0.0])
    return np.zeros(2)


def _safe_move(v, pos2, HC, bV):
    """Move vertex while keeping bV membership consistent."""
    key = tuple(float(c) for c in pos2)
    if v in bV:
        bV.discard(v)
        HC.V.move(v, key)
        bV.add(v)
    else:
        HC.V.move(v, key)


def domain_mass(HC):
    return sum(getattr(v, 'm', 0.0) for v in HC.V if not wall_criterion(v))


def domain_momentum_x(HC):
    return sum(
        getattr(v, 'm', 0.0) * float(_safe_u(v)[0])
        for v in HC.V if not wall_criterion(v)
    )


def compile_gif(paths, out, dur=GIF_DURATION_MS):
    try:
        from PIL import Image
        imgs = [Image.open(p) for p in paths]
        if imgs:
            imgs[0].save(out, save_all=True, append_images=imgs[1:],
                         duration=dur, loop=0)
            print(f"  GIF -> {out}")
    except ImportError:
        print("  PIL not available — GIF skipped")


def get_edges_delaunay(mesh):
    verts = list(mesh.V)
    if len(verts) < 3:
        return []
    coords = np.array([v.x_a[:2] for v in verts])
    _, uid = np.unique(coords, axis=0, return_index=True)
    cu = coords[uid]
    if len(cu) < 3:
        return []
    try:
        tri = Delaunay(cu)
    except QhullError:
        return []
    seen, segs = set(), []
    for s in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = int(s[i]), int(s[j])
                k = (min(a, b), max(a, b))
                if k not in seen:
                    seen.add(k)
                    segs.append([cu[a], cu[b]])
    return segs



# Safe retopologization  (Delaunay only, no compute_vd)


def _retopologize_safe(HC, bV):
    """
    Rebuild Delaunay triangulation so v.nn is valid after Lagrangian
    vertex movement.  Updates bV to contain ONLY wall vertices.

    Deliberately does NOT call compute_vd (barycentric dual cells)
    because that function can crash on irregular / transient meshes.
    The dudt_fn below uses only v.nn (+ KDTree fallback), so no
    dual cells are required.
    """
    verts = list(HC.V)
    if len(verts) < DIM + 1:
        return

    # Disconnect all edges
    for v in verts:
        for nb in list(v.nn):
            try:
                v.disconnect(nb)
            except Exception:
                pass

    # Delaunay triangulation on current vertex positions
    coords = np.array([v.x_a[:DIM] for v in verts], dtype=float)
    try:
        tri = Delaunay(coords)
    except QhullError:
        return

    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                try:
                    verts[simplex[i]].connect(verts[simplex[j]])
                except Exception:
                    pass

    # bV <- wall vertices only  (inlet/outlet vertices advect freely)
    bV.clear()
    for v in HC.V:
        is_wall = wall_criterion(v)
        v.boundary = is_wall
        if is_wall:
            bV.add(v)



# Momentum forcing  du/dt = f_body + nu * Laplacian(u)

#
# Why body force ONLY — no separate grad_p:
#   The stored pressure field is  p = dPdx * x  (linear).
#   Its gradient is exactly  dp/dx = dPdx.
#   The body force is         f_body = -dPdx / rho.
#   These are identical in magnitude and opposite in sign to -grad_p/rho.
#   Adding both would double the forcing, causing velocity divergence.
#   We use f_body as the sole pressure representation.
#
# Wall neighbors (v.u = 0) ARE included in the Laplacian stencil.
# This is intentional: no-slip enters as a Dirichlet condition for
# the viscous diffusion operator, driving the near-wall velocity
# toward zero and the interior toward the parabolic profile.

# Module-level reference to HC so the KDTree fallback can access it.
_HC_ref = None


def dudt_fn(v):
    """
    Return acceleration du/dt = f_body + nu * d^2u/dy^2.

    For channel flow the velocity field u(y,t) is homogeneous in x
    (plug flow and its relaxation to the Poiseuille parabola depend
    only on y and t).  Computing a full 2D Laplacian over Lagrangian
    particles causes two problems:
      - Particles cluster near the outlet as they accelerate, creating
        large spurious d^2u/dx^2 terms.
      - The weighted estimator amplifies these, blowing up velocities
        and collapsing the adaptive dt.

    Solution: use only the y-direction (wall-normal) second derivative
    with an asymmetric 3-point stencil over the nearest neighbours
    strictly above and strictly below v in y.  Wall vertices (u=0) are
    included as neighbours — this is the correct Dirichlet condition
    that diffuses momentum toward the no-slip walls.

    Stencil:
        d^2u/dy^2 ~= 2 / (dy_up + dy_dn)
                       * [ (u_up - u_v)/dy_up + (u_dn - u_v)/dy_dn ]
    which reduces to the standard second-order FD formula when
    dy_up == dy_dn.
    """
    u_v    = _safe_u(v)
    f_body = np.array([-dPdx / rho, 0.0])

    if _HC_ref is None:
        return f_body

    y_v = float(v.x_a[1])

    # Collect all vertices (wall + interior) separated by y-position
    above, below = [], []
    for w in _HC_ref.V:
        if w is v:
            continue
        dy = float(w.x_a[1]) - y_v
        if dy > 1e-12:
            above.append((dy, w))
        elif dy < -1e-12:
            below.append((-dy, w))

    if not above or not below:
        # At or next to a wall — body force only; no-slip BC handles the rest
        return f_body

    # Nearest neighbour above and below (minimum |dy|)
    dy_up, v_up = min(above, key=lambda x: x[0])
    dy_dn, v_dn = min(below, key=lambda x: x[0])

    u_up = _safe_u(v_up)
    u_dn = _safe_u(v_dn)

    # Asymmetric 3-point d^2u/dy^2
    denom = dy_up + dy_dn
    if denom < 1e-14:
        return f_body

    d2u_dy2 = (2.0 / denom) * (
        (u_up - u_v) / dy_up + (u_dn - u_v) / dy_dn
    )

    return f_body + nu * d2u_dy2



# Custom outlet BC (simple deletion)


class SimpleOutletBC(BoundaryCondition):
    """Delete non-wall vertices that have exited the domain (x >= x_max)."""

    def __init__(self, x_max):
        super().__init__()
        self.x_max = float(x_max)

    def apply(self, mesh, dt, target_vertices=None):
        to_del = [v for v in list(mesh.V)
                  if not wall_criterion(v)
                  and float(v.x_a[0]) >= self.x_max]
        for v in to_del:
            mesh.V.remove(v)
        return len(to_del)



# Ghost mesh update (call AFTER each step, before next step)


def update_inlet_ghost(unit_mesh, HC):
    """
    Sync unit_mesh vertex velocities to the current inlet-region profile
    of HC via inverse-distance weighting in y.

    This ensures injected particles carry the developing (not the
    original plug-flow) velocity profile.  The ghost is re-cloned from
    unit_mesh whenever it is fully depleted, so unit_mesh must be
    up-to-date before that re-clone.
    """
    inlet_verts = [v for v in HC.V
                   if not wall_criterion(v)
                   and float(v.x_a[0]) <= L_period + 1e-10]
    if not inlet_verts:
        return

    ys = np.array([float(v.x_a[1]) for v in inlet_verts])
    us = np.array([_safe_u(v) for v in inlet_verts])  # (N, 2)

    for gv in unit_mesh.V:
        if wall_criterion(gv):
            gv.u = np.zeros(2)
            continue
        gy = float(gv.x_a[1])
        dy = np.abs(ys - gy)
        w  = 1.0 / (dy + 1e-6)
        w /= w.sum()
        gv.u = (w[:, None] * us).sum(axis=0)



# Mesh construction — PLUG FLOW initial conditions


def build_unit_mesh():
    um = Complex(DIM, domain=[(0.0, L_period), (0.0, H)])
    um.triangulate()
    for _ in range(N_REFINE):
        um.refine_all()
    for v in um.V:
        v.u = np.zeros(2)   # zero-start: all vertices at rest
        v.p = p_analytical(v.x_a[0])
        v.m = 0.0
    return um


def build_tiled_mesh(unit_mesh, n_tiles):
    HC   = Complex(DIM, domain=[(0.0, L_domain), (0.0, H)])
    seen = {}
    for k in range(n_tiles):
        for uv in unit_mesh.V:
            pos = uv.x_a.copy()
            pos[0] += k * L_period
            key = tuple(round(c, 12) for c in pos)
            if key not in seen:
                v = HC.V[tuple(pos)]
                v.u = np.zeros(2)   # zero-start: all vertices at rest
                v.p = p_analytical(pos[0])
                v.m = 0.0
                seen[key] = v
    bV = set()
    for v in HC.V:
        if wall_criterion(v):
            bV.add(v)
            v.boundary = True
        else:
            v.boundary = False
    return HC, bV



# Error diagnostics


def compute_errors(HC, t):
    """Return (L2_transient, Linf_transient, L2_ss, Linf_ss)."""
    errs_tr, errs_ss = [], []
    for v in HC.V:
        if wall_criterion(v):
            continue
        y     = float(v.x_a[1])
        u_num = float(_safe_u(v)[0])
        u_tr  = float(u_transient(y, t, n_terms=20))
        u_ss  = float(u_poiseuille(y))
        errs_tr.append((u_num - u_tr)**2)
        errs_ss.append((u_num - u_ss)**2)

    def _s(e):
        if not e: return 0.0, 0.0
        return float(np.sqrt(np.mean(e))), float(np.sqrt(max(e)))

    return _s(errs_tr) + _s(errs_ss)



# Frame renderer




# CFD diagnostic helpers


def _ccw_sort(pts):
    """Sort polygon vertices counter-clockwise around their centroid."""
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles  = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return pts[np.argsort(angles)]


def _sh_clip(polygon, x_min, x_max, y_min, y_max):
    """
    Sutherland-Hodgman polygon clipping to an axis-aligned rectangle.
    Input polygon vertices are first sorted CCW so the inside test is
    consistent regardless of the original Voronoi vertex ordering.
    Returns clipped (M,2) array or None.  Pure numpy, no shapely.
    """
    def _clip_edge(pts, p1, p2):
        def inside(p):
            return (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0]) >= 0
        def intersect(a, b):
            dx1, dy1 = b[0]-a[0], b[1]-a[1]
            dx2, dy2 = p2[0]-p1[0], p2[1]-p1[1]
            denom = dx1*dy2 - dy1*dx2
            if abs(denom) < 1e-14:
                return a
            t = ((p1[0]-a[0])*dy2 - (p1[1]-a[1])*dx2) / denom
            return np.array([a[0]+t*dx1, a[1]+t*dy1])
        out, prev = [], pts[-1]
        for curr in pts:
            if inside(curr):
                if not inside(prev):
                    out.append(intersect(prev, curr))
                out.append(curr)
            elif inside(prev):
                out.append(intersect(prev, curr))
            prev = curr
        return out

    # Sort polygon CCW, then clip against each half-plane.
    # Each clip edge is oriented so the INTERIOR of the rectangle
    # lies to the LEFT of the directed edge (standard S-H convention):
    #   left edge:   (x_min,y_max)->(x_min,y_min)  interior is +x side
    #   bottom edge: (x_min,y_min)->(x_max,y_min)  interior is +y side
    #   right edge:  (x_max,y_min)->(x_max,y_max)  interior is -x side
    #   top edge:    (x_max,y_max)->(x_min,y_max)  interior is -y side
    arr = np.asarray(polygon, dtype=float)
    pts = list(_ccw_sort(arr))
    for p1, p2 in [
        (np.array([x_min, y_max]), np.array([x_min, y_min])),
        (np.array([x_min, y_min]), np.array([x_max, y_min])),
        (np.array([x_max, y_min]), np.array([x_max, y_max])),
        (np.array([x_max, y_max]), np.array([x_min, y_max])),
    ]:
        pts = _clip_edge(pts, p1, p2)
        if not pts:
            return None
    return np.array(pts)


def _poly_area(pts):
    """Shoelace area of polygon (N,2)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_dual_cells(HC):
    """
    Compute clipped Voronoi dual cells for all vertices.
    Returns list of (polygon_xy, area, is_wall) tuples.
    Clips to domain [0, L_domain] x [0, H].
    Pure numpy — no shapely required.
    """
    verts = list(HC.V)
    if len(verts) < 4:
        return []
    coords = np.array([v.x_a[:2] for v in verts])
    # Mirror points at all four boundaries to force finite Voronoi cells
    mirrors = np.vstack([
        coords + np.array([ 2*L_domain, 0]),
        coords + np.array([-2*L_domain, 0]),
        coords + np.array([0,  2*H]),
        coords + np.array([0, -2*H]),
    ])
    try:
        vor = Voronoi(np.vstack([coords, mirrors]))
    except Exception:
        return []

    results = []
    for i, v in enumerate(verts):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            results.append((None, 0.0, wall_criterion(v)))
            continue
        poly_pts = vor.vertices[region]
        clipped  = _sh_clip(poly_pts, 0.0, L_domain, 0.0, H)
        if clipped is None or len(clipped) < 3:
            results.append((None, 0.0, wall_criterion(v)))
            continue
        results.append((_ccw_sort(clipped), _poly_area(clipped), wall_criterion(v)))
    return results


def compute_vorticity(HC):
    """
    Compute vorticity omega_z = -du_x/dy for each interior vertex.
    Uses nearest-above / nearest-below 3-point stencil (same as solver).
    Returns dict {vertex: omega_z}.
    """
    all_verts = list(HC.V)
    omega = {}
    for v in all_verts:
        if wall_criterion(v):
            omega[id(v)] = 0.0
            continue
        y_v = float(v.x_a[1])
        u_v = float(_safe_u(v)[0])
        above = [(float(w.x_a[1]) - y_v, w) for w in all_verts
                 if float(w.x_a[1]) - y_v > 1e-12]
        below = [(y_v - float(w.x_a[1]), w) for w in all_verts
                 if y_v - float(w.x_a[1]) > 1e-12]
        if not above or not below:
            omega[id(v)] = 0.0
            continue
        dy_up, v_up = min(above, key=lambda x: x[0])
        dy_dn, v_dn = min(below, key=lambda x: x[0])
        u_up = float(_safe_u(v_up)[0])
        u_dn = float(_safe_u(v_dn)[0])
        denom = dy_up + dy_dn
        if denom < 1e-14:
            omega[id(v)] = 0.0
            continue
        # du/dy by asymmetric FD; omega_z = -du/dy for this 2D flow
        dudy = (u_up - u_dn) / denom
        omega[id(v)] = -dudy
    return omega


def compute_wss(HC):
    """
    Wall shear stress tau_w = mu * du/dy at both walls.

    Method: first-order finite difference from the no-slip wall.
        tau_w_bot = mu * u_i / y_i   (where y_i is small, u_wall=0)
        tau_w_top = mu * u_i / (H - y_i)

    Average over the N_near = 3 closest-to-wall vertices to reduce
    single-particle noise.  Using points very close to the wall
    (rather than a large region) avoids integrating the non-linear
    part of the parabola which biases the slope estimate downward.

    Analytical SS: tau_w = mu * 4 * U_max / H = 4 Pa.
    Returns (tau_bottom, tau_top).
    """
    interior = sorted(
        [(float(v.x_a[1]), float(_safe_u(v)[0]))
         for v in HC.V if not wall_criterion(v)],
        key=lambda p: p[0]
    )
    if not interior:
        return 0.0, 0.0

    # Collect DISTINCT y-levels (tiling gives multiple vertices at same y).
    # Average u at each y-level, then apply Richardson extrapolation.
    from collections import defaultdict
    y_to_us = defaultdict(list)
    for y, u in interior:
        y_to_us[round(y, 8)].append(u)
    y_levels = sorted(y_to_us.keys())           # ascending: wall → centre
    u_means  = [float(np.mean(y_to_us[y])) for y in y_levels]

    def _richardson_bot(y_lev, u_lev):
        """Richardson extrapolation to y=0 using two distinct y-levels."""
        # Find first two distinct y-levels with meaningful gap
        for i in range(len(y_lev)):
            for j in range(i+1, len(y_lev)):
                y1, y2 = y_lev[i], y_lev[j]
                u1, u2 = u_lev[i], u_lev[j]
                if y2 - y1 > 1e-8 and y1 > 1e-10:
                    # tau_w = mu * (y2*u1/y1 - y1*u2/y2) / (y2-y1)
                    return mu * (y2*u1/y1 - y1*u2/y2) / (y2 - y1)
        # Fallback: first-order
        if y_lev and y_lev[0] > 1e-10:
            return mu * u_lev[0] / y_lev[0]
        return 0.0

    def _richardson_top(y_lev, u_lev):
        """Richardson extrapolation to y=H using two distinct y-levels."""
        y_lev_r = [H - y for y in reversed(y_lev)]
        u_lev_r = list(reversed(u_lev))
        for i in range(len(y_lev_r)):
            for j in range(i+1, len(y_lev_r)):
                d1, d2 = y_lev_r[i], y_lev_r[j]
                u1, u2 = u_lev_r[i], u_lev_r[j]
                if d2 - d1 > 1e-8 and d1 > 1e-10:
                    return mu * (d2*u1/d1 - d1*u2/d2) / (d2 - d1)
        if y_lev_r and y_lev_r[0] > 1e-10:
            return mu * u_lev_r[0] / y_lev_r[0]
        return 0.0

    tau_bot = _richardson_bot(y_levels, u_means)
    tau_top = _richardson_top(y_levels, u_means)
    return float(tau_bot), float(tau_top)


def compute_mass_flux(HC, x_slice, tol=None):
    """
    Mass flux: integral rho * u_x(y) dy over [0, H].

    Since u(y,t) is x-homogeneous in this Lagrangian DEC solver
    (the momentum equation has no x-derivatives), we use ALL interior
    vertices sorted by y rather than a thin x-slice.  A thin slice
    contains only ~3-4 vertices, giving a very noisy estimate.

    The x_slice argument is kept for API compatibility but ignored.
    Analytical SS: rho * U_mean * H = 2/3 kg/m/s.
    """
    from collections import defaultdict
    y_to_us = defaultdict(list)
    for v in HC.V:
        if not wall_criterion(v):
            y_to_us[round(float(v.x_a[1]), 8)].append(float(_safe_u(v)[0]))
    if not y_to_us:
        return 0.0
    # Average u at each distinct y-level (correct for tiled Lagrangian mesh)
    y_lev = sorted(y_to_us.keys())
    u_lev = [float(np.mean(y_to_us[y])) for y in y_lev]
    # Add wall BCs
    ys_full = np.array([0.0] + y_lev + [H])
    us_full = np.array([0.0] + u_lev + [0.0])
    return float(rho * np.trapezoid(us_full, ys_full))


def compute_KE_dissipation(HC):
    """
    Kinetic energy:   KE = (1/2) rho L_domain integral_0^H u(y)^2 dy
    Viscous diss.:    eps = mu * L_domain * integral_0^H (du/dy)^2 dy

    Both are computed by averaging over DISTINCT y-levels (so that
    multiple Lagrangian particles at the same y-level in the tiled mesh
    contribute only once to the quadrature), then integrating over y
    with wall BCs.  This gives physically meaningful values comparable
    to the analytical integrals.

    Analytical SS:  KE = rho*L*(8/15)*H*U_max^2/2 = 0.8 J
                    eps = mu*L*(16/3)*H             = 16 W
    """
    from collections import defaultdict
    omega_map = compute_vorticity(HC)

    y_to_u2  = defaultdict(list)   # u^2 at each y-level
    y_to_om2 = defaultdict(list)   # omega_z^2 at each y-level

    for v in HC.V:
        if wall_criterion(v):
            continue
        y_key = round(float(v.x_a[1]), 8)
        u_i   = _safe_u(v)
        omz   = omega_map.get(id(v), 0.0)
        y_to_u2[y_key].append(float(np.dot(u_i, u_i)))
        y_to_om2[y_key].append(omz**2)

    if not y_to_u2:
        return 0.0, 0.0

    y_lev  = sorted(y_to_u2.keys())
    u2_lev = [float(np.mean(y_to_u2[y]))  for y in y_lev]
    om2_lev= [float(np.mean(y_to_om2[y])) for y in y_lev]

    # Integrate over y with wall BCs (u=0, du/dy finite at walls)
    # For KE: add u²=0 at walls
    ys_ke  = np.array([0.0] + y_lev + [H])
    u2s_ke = np.array([0.0] + u2_lev + [0.0])
    KE     = 0.5 * rho * L_domain * float(np.trapezoid(u2s_ke, ys_ke))

    # For dissipation: (du/dy)² at walls can be estimated by Richardson
    # at bottom: (du/dy)² = (tau_w_bot / mu)²
    tau_b, tau_t = compute_wss(HC)
    om2_wall_bot = (tau_b / mu)**2
    om2_wall_top = (tau_t / mu)**2

    ys_ep  = np.array([0.0] + y_lev + [H])
    om2s   = np.array([om2_wall_bot] + om2_lev + [om2_wall_top])
    eps    = mu * L_domain * float(np.trapezoid(om2s, ys_ep))

    return float(KE), float(eps)

def plot_frame(step, t, HC, fig_dir,
               cum_inj, cum_del, l2_tr, l2_ss,
               hist_t, hist_l2_tr, hist_l2_ss,
               hist_tau_wall=None, hist_u_cl=None):
    """
    Commercial-CFD-style frame: 2 rows x 3 columns.
    [0,0] Velocity field u_x (colored scatter + Delaunay mesh)
    [0,1] Voronoi dual-cell map (cell area)
    [0,2] Velocity profiles at x = L/4, L/2, 3L/4  vs analytical
    [1,0] Vorticity field  omega_z = -du/dy
    [1,1] Centerline velocity + wall shear stress history
    [1,2] Residuals: L2 error convergence
    """
    fig = plt.figure(figsize=(FRAME_FIG_W * 1.1, FRAME_FIG_H * 1.2))
    gs  = fig.add_gridspec(2, 3, width_ratios=[2.5, 1.8, 2],
                           height_ratios=[1, 1], hspace=0.50, wspace=0.40)
    ax_vel  = fig.add_subplot(gs[0, 0])
    ax_dual = fig.add_subplot(gs[1, 0])
    ax_vort = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, 1])
    ax_cl   = fig.add_subplot(gs[0, 2])
    ax_err  = fig.add_subplot(gs[1, 2])

    verts  = list(HC.V)
    y_fine = np.linspace(0, H, 300)
    n_int  = sum(1 for v in HC.V if not wall_criterion(v))
    n_wall = len(verts) - n_int

    # ── [0,0] Velocity field ──────────────────────────────────
    if verts:
        coords = np.array([v.x_a[:2] for v in verts])
        ux_arr = np.array([float(_safe_u(v)[0]) for v in verts])
        sc = ax_vel.scatter(coords[:, 0], coords[:, 1],
                            c=ux_arr, cmap='RdYlGn', s=40, zorder=3,
                            vmin=0.0, vmax=U_max * 1.05)
        fig.colorbar(sc, ax=ax_vel, fraction=0.03, pad=0.02,
                     label='$u_x$ [m/s]')
    edges = get_edges_delaunay(HC)
    if edges:
        ax_vel.add_collection(
            LineCollection(edges, colors='gray', linewidths=0.4, alpha=0.3))
    ax_vel.axvline(0.0,      color='royalblue', lw=1.5, ls='--')
    ax_vel.axvline(L_domain, color='firebrick',  lw=1.5, ls='--')
    ax_vel.set_xlim(-0.1, L_domain + 0.1)
    ax_vel.set_ylim(-0.08, H + 0.08)
    ax_vel.set_xlabel('x [m]'); ax_vel.set_ylabel('y [m]')
    ax_vel.set_title(
        f'$u_x$ field  |  t={t:.3f}s ({t/tau_visc:.1f}$\tau$)  '
        f'N={len(verts)}', fontsize=11)
    ax_vel.set_aspect('equal'); ax_vel.grid(True, alpha=0.15)

    # ── [1,0] Voronoi dual-cell map ───────────────────────────
    dual = compute_dual_cells(HC)
    areas = [d[1] for d in dual if d[0] is not None and not d[2]]
    area_min = min(areas) if areas else 1e-6
    area_max = max(areas) if areas else 1.0
    cmap_dual = plt.cm.viridis
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    int_patches, int_areas_pc, wall_patches_list = [], [], []
    for (xy, area, is_wall) in dual:
        if xy is None:
            continue
        if is_wall:
            wall_patches_list.append(MplPoly(xy, closed=True))
        else:
            int_patches.append(MplPoly(xy, closed=True))
            int_areas_pc.append(area)
    if wall_patches_list:
        pc_wall = PatchCollection(wall_patches_list, facecolor='lightgray',
                                  edgecolor='k', linewidths=0.5, alpha=0.6)
        ax_dual.add_collection(pc_wall)
    if int_patches:
        pc_int = PatchCollection(int_patches, cmap=cmap_dual, alpha=0.85,
                                 edgecolor='k', linewidths=0.5)
        pc_int.set_array(np.array(int_areas_pc))
        pc_int.set_clim(area_min, area_max)
        ax_dual.add_collection(pc_int)
    # Overlay vertex positions
    if verts:
        ax_dual.scatter(coords[:, 0], coords[:, 1],
                        c='white', s=15, zorder=4, edgecolors='k', lw=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap_dual,
         norm=plt.Normalize(vmin=area_min, vmax=area_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax_dual, fraction=0.03, pad=0.02,
                 label='Cell area [m$^2$]')
    ax_dual.set_xlim(-0.1, L_domain + 0.1)
    ax_dual.set_ylim(-0.08, H + 0.08)
    ax_dual.set_xlabel('x [m]'); ax_dual.set_ylabel('y [m]')
    ax_dual.set_title('Voronoi dual-cell map', fontsize=11)
    ax_dual.set_aspect('equal'); ax_dual.grid(True, alpha=0.15)

    # ── [0,1] Vorticity field  omega_z = -du/dy ───────────────
    omega_map = compute_vorticity(HC)
    if verts:
        omz_arr = np.array([omega_map.get(id(v), 0.0) for v in verts])
        vlim = max(abs(omz_arr).max(), 1e-6)
        sc2 = ax_vort.scatter(coords[:, 0], coords[:, 1],
                              c=omz_arr, cmap='RdBu_r', s=40, zorder=3,
                              vmin=-vlim, vmax=vlim)
        fig.colorbar(sc2, ax=ax_vort, fraction=0.03, pad=0.02,
                     label=r'$\omega_z$ [1/s]')
    if edges:
        ax_vort.add_collection(
            LineCollection(edges, colors='gray', linewidths=0.3, alpha=0.2))
    # Analytical vorticity at steady state: omega_z_SS = -du_P/dy = 4(2y/H-1)*U_max/H
    omz_ss_mid = -float(4 * U_max * (2*0.5/H - 1) / H)  # = 0 at centreline
    ax_vort.set_xlim(-0.1, L_domain + 0.1)
    ax_vort.set_ylim(-0.08, H + 0.08)
    ax_vort.set_xlabel('x [m]'); ax_vort.set_ylabel('y [m]')
    ax_vort.set_title(r'Vorticity  $\omega_z = -\partial u/\partial y$', fontsize=11)
    ax_vort.set_aspect('equal'); ax_vort.grid(True, alpha=0.15)

    # ── [1,1] Multi-station velocity profiles ─────────────────
    colors_st = ['royalblue', 'darkorange', 'green']
    x_stations = [L_domain/4, L_domain/2, 3*L_domain/4]
    for x_st, col in zip(x_stations, colors_st):
        pts = [(float(_safe_u(v)[0]), float(v.x_a[1]))
               for v in HC.V
               if not wall_criterion(v)
               and abs(float(v.x_a[0]) - x_st) < L_period * 0.6]
        if pts:
            us_st, ys_st = zip(*sorted(pts, key=lambda p: p[1]))
            ax_prof.plot(us_st, ys_st, 'o', color=col, ms=5,
                         label=f'x={x_st:.2f}m', zorder=4)
    ax_prof.plot(u_poiseuille(y_fine), y_fine, 'k-', lw=2.5,
                 label='SS', zorder=10)
    ax_prof.plot(u_transient(y_fine, t, n_terms=20), y_fine,
                 'k--', lw=1.5, alpha=0.5, label=f'Exact t={t:.2f}s')
    ax_prof.set_xlabel('$u_x$ [m/s]'); ax_prof.set_ylabel('y [m]')
    ax_prof.set_title('Velocity profiles at x = L/4, L/2, 3L/4', fontsize=11)
    ax_prof.set_xlim(-0.05, U_max * 1.15)
    ax_prof.set_ylim(-0.02, H + 0.02)
    ax_prof.legend(fontsize=8, loc='upper left'); ax_prof.grid(True, alpha=0.3)

    # ── [0,2] Centerline velocity + wall shear stress ─────────
    tau_w_analytical = mu * 4.0 * U_max / H   # = 4 Pa at SS
    ax_cl.axhline(U_max, color='royalblue', lw=1.2, ls='--',
                  label=f'$U_{{max}}$={U_max:.2f} m/s (SS)')
    if hist_t and hist_u_cl:
        ax_cl.plot(hist_t, hist_u_cl, 'b-', lw=2, label='$u_{cl}(t)$ numerical')
    # Mark current
    # Centreline: weighted average near y=H/2
    _dy_band_f = H / 4.0
    _cl_f = [(abs(float(v.x_a[1]) - H/2), float(_safe_u(v)[0]))
             for v in HC.V
             if not wall_criterion(v)
             and abs(float(v.x_a[1]) - H/2) < _dy_band_f]
    cl_now = None
    if _cl_f:
        _w_f  = np.array([1.0 / (p[0] + 1e-6) for p in _cl_f])
        cl_now = float(np.dot(_w_f, [p[1] for p in _cl_f]) / _w_f.sum())
        ax_cl.scatter([t], [cl_now], c='blue', s=60, zorder=5)
    ax_cl.set_xlabel('t [s]'); ax_cl.set_ylabel('$u_{cl}$ [m/s]', color='b')
    ax_cl.tick_params(axis='y', labelcolor='b')
    ax_cl2 = ax_cl.twinx()
    ax_cl2.axhline(tau_w_analytical, color='crimson', lw=1.2, ls='--',
                     label=r'tau_w SS=' + f'{tau_w_analytical:.1f} Pa')
    if hist_t and hist_tau_wall:
        ax_cl2.plot(hist_t, hist_tau_wall, 'r-', lw=2,
                    label='tau_w(t) numerical')
    tau_now, _ = compute_wss(HC)
    if hist_t:
        ax_cl2.scatter([t], [tau_now], c='red', s=60, zorder=5)
    ax_cl2.set_ylabel('tau_w [Pa]', color='r')
    ax_cl2.tick_params(axis='y', labelcolor='r')
    lines1, labs1 = ax_cl.get_legend_handles_labels()
    lines2, labs2 = ax_cl2.get_legend_handles_labels()
    ax_cl.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper left')
    ax_cl.set_title('Centreline velocity  &  Wall shear stress', fontsize=11)
    ax_cl.grid(True, alpha=0.3)

    # ── [1,2] Residuals / convergence ─────────────────────────
    if hist_t:
        ax_err.semilogy(hist_t, hist_l2_tr, 'b-',  lw=2,
                        label='$L_2$ vs transient')
        ax_err.semilogy(hist_t, hist_l2_ss, 'r--', lw=2,
                        label='$L_2$ vs Poiseuille SS')
    ax_err.axvline(tau_visc,   color='gray', lw=1.2, ls=':',  label='tau')
    ax_err.axvline(5*tau_visc, color='gray', lw=1.8, ls='--', label='5 tau')
    ax_err.set_xlabel('t [s]'); ax_err.set_ylabel('$L_2$ velocity error [m/s]')
    ax_err.set_title(f'Residuals: $L_2$={l2_ss:.2e}', fontsize=11)
    ax_err.legend(fontsize=8); ax_err.grid(True, which='both', alpha=0.3)
    if hist_l2_ss and max(hist_l2_ss) > 0:
        ymin = max(1e-14, min(hist_l2_ss) * 0.5)
        ymax = max(hist_l2_ss) * 2.0
        if ymax > ymin:
            ax_err.set_ylim(ymin, ymax)

    fig.suptitle(
        f'Lagrangian DEC  |  Poiseuille flow  |  '
        f't = {t:.3f} s  ({t/tau_visc:.1f}$\tau$)  |  '
        f'Re = {rho*U_mean*H/mu:.1f}',
        fontsize=13, y=1.01)
    plt.tight_layout()
    fpath = os.path.join(fig_dir, f'frame_{step:06d}.png')
    plt.savefig(fpath, dpi=FRAME_DPI, bbox_inches='tight')
    plt.close(fig)
    return fpath



# Build meshes and boundary conditions

n_tiles   = int(round(L_domain / L_period))
unit_mesh = build_unit_mesh()
HC, bV    = build_tiled_mesh(unit_mesh, n_tiles)

n_unit   = sum(1 for _ in unit_mesh.V)
n_wall_u = sum(1 for v in unit_mesh.V if wall_criterion(v))
n_int_u  = n_unit - n_wall_u
n_main   = sum(1 for _ in HC.V)
n_int_0  = sum(1 for v in HC.V if not wall_criterion(v))

M_total  = rho * L_domain * H          # target total domain mass
m_per_v  = M_total / max(n_int_0, 1)   # fixed mass per interior vertex

# Assign initial masses
for v in HC.V:
    v.m = m_per_v if not wall_criterion(v) else 0.0
for v in unit_mesh.V:
    v.m = (rho * L_period * H / max(n_int_u, 1)
           if not wall_criterion(v) else 0.0)

print(f"\nUnit mesh : {n_unit} verts  (wall={n_wall_u}, interior={n_int_u})")
print(f"Main mesh : {n_main} verts  (tiled {n_tiles}x unit)")
print(f"n_int_0   = {n_int_0},   m_per_v = {m_per_v:.4f}")
print(f"M_total   = {domain_mass(HC):.6f}  (target {M_total:.6f})")

# Give dudt_fn access to HC for its KDTree fallback
_HC_ref = HC

# Boundary conditions — periodic wrap replaces ghost inlet + outlet
# The wall BC is still needed to zero-out wall velocities after advection.
wall_bc = PositionalNoSlipWallBC(criterion_fn=wall_criterion, dim=DIM, bV=bV)


# Diagnostic storage

ts = dict(
    t=[], step=[], dt_used=[],
    n_verts=[], n_interior=[], n_wall=[],
    cum_inj=[], cum_del=[],
    l2_transient=[], linf_transient=[],
    l2_ss=[], linf_ss=[],
    mass_total=[], momentum_x=[], momentum_expected=[],
    KE=[], dissipation=[], tau_wall=[], u_centerline=[],
    mass_flux_inlet=[], mass_flux_mid=[], mass_flux_outlet=[],
)

profile_snaps = []
_snap_done    = set()


def save_profile_snap(t_now, HC_mesh, label):
    pts = sorted(
        [(float(v.x_a[1]), float(_safe_u(v)[0]))
         for v in HC_mesh.V if not wall_criterion(v)],
        key=lambda p: p[0],
    )
    if pts:
        profile_snaps.append((
            t_now,
            np.array([p[0] for p in pts]),
            np.array([p[1] for p in pts]),
            label,
        ))



# Initial diagnostics

l2_tr0, li_tr0, l2_ss0, li_ss0 = compute_errors(HC, 0.0)
M0 = domain_mass(HC)
P0 = domain_momentum_x(HC)

print(f"\nInitial: M={M0:.4f}  Px={P0:.4f}")
print(f"  L2 vs transient : {l2_tr0:.4e}  (plug matches t=0 exact solution)")
print(f"  L2 vs SS        : {l2_ss0:.4e}  (large: plug != parabola)")

save_profile_snap(0.0, HC, 't=0 (rest, u=0)')

frame_paths = []
fp = plot_frame(0, 0.0, HC, _FRAMES,
                0, 0, l2_tr0, l2_ss0, [], [], [], [], [])
frame_paths.append(fp)

print(f"\n{'step':>6}  {'t':>7}  {'t/tau':>6}  {'dt':>8}  {'N':>5}  "
      f"{'L2_tr':>10}  {'L2_ss':>10}  {'M/M0':>8}  {'dPx%':>8}  "
      f"{'inj':>5}  {'del':>5}  {'net':>6}")


# FVM mass state

# Each interior Lagrangian vertex represents a control volume
# of fixed mass  m_cell = rho * L_domain * H / n_int_0.
# The periodic-wrap BC keeps N = n_int_0 exactly at all times,
# so M = n_int_0 * m_cell = M_total at every step.
m_cell = M_total / n_int_0


# Adaptive Euler time loop

t          = 0.0
dt         = DT_INITIAL
step       = 0
cum_inj    = 0
cum_del    = 0
P_expected = P0

while t < T_END - 1e-14:

    dt = min(dt, T_END - t)   # don't overshoot

    # 1. Refresh bV: wall vertices only (no Delaunay rebuild needed —
    #    dudt_fn uses a direct y-search over all vertices, not v.nn)
    bV.clear()
    for _v in HC.V:
        _is_wall = wall_criterion(_v)
        _v.boundary = _is_wall
        if _is_wall:
            bV.add(_v)

    # 2. Expected momentum increment from body force
    M_fluid    = domain_mass(HC)
    P_expected += (-dPdx / rho) * M_fluid * dt

    # 3. Identify interior vertices (not in bV = not a wall)
    interior = [v for v in HC.V if v not in bV]

    # 4. Compute accelerations at time level n  (snapshot u before update)
    accels = {v: dudt_fn(v) for v in interior}

    # 5. Explicit Euler update  (u and x updated simultaneously from level n)
    updates = {}
    for v in interior:
        u_old = _safe_u(v)
        x_new = v.x_a[:2] + dt * u_old        # Lagrangian: x advances at old u
        u_new = u_old + dt * accels[v]
        updates[v] = (x_new, u_new)

    for v, (x_new, u_new) in updates.items():
        v.u = u_new
        _safe_move(v, x_new, HC, bV)

    # Update linear pressure field to new vertex positions
    for v in HC.V:
        v.p = p_analytical(v.x_a[0])

    # 6. Apply BCs
    # 6a. Wall BC: zero velocity on wall vertices
    wall_bc.apply(HC, dt)

    # 6b. Periodic wrap: any interior vertex that has crossed x = L_domain
    #     is immediately repositioned to x' = x - L_domain (same y, same m).
    #     Its velocity is interpolated from current interior vertices near
    #     x = 0 at the same y-level, so it re-enters with the developing
    #     inlet profile rather than its own (outlet) velocity.
    #     This guarantees: net = 0 and M = const at every step.
    n_wrap = 0

    # Collect wrapping candidates first so the inlet_pts lookup is
    # built from the CURRENT (pre-wrap) inlet, not yet-wrapped vertices.
    wrap_candidates = [v for v in list(HC.V)
                       if not wall_criterion(v) and float(v.x_a[0]) >= L_domain]

    # Build inlet profile lookup from vertices currently near x = 0
    # (before any wrapping disturbs them).
    inlet_pts = [(float(v.x_a[1]), _safe_u(v))
                 for v in HC.V
                 if not wall_criterion(v)
                 and float(v.x_a[0]) <= L_period + 1e-10]

    for v in wrap_candidates:
        x   = float(v.x_a[0])
        y_v = float(v.x_a[1])

        # Place the wrapped vertex at x' = (x mod L_domain) + WRAP_OFFSET.
        # WRAP_OFFSET > 0 ensures x' never lands exactly on x = 0 (the
        # inlet face), which would coincide with existing vertex positions
        # and cause the HC.V dictionary to silently merge/drop the vertex.
        WRAP_OFFSET = 1e-4          # small but >> position tolerance (~1e-10)
        x_new = (x % L_domain) + WRAP_OFFSET
        # Clamp: if modulo gave exactly 0 (rare float coincidence), offset
        if x_new < WRAP_OFFSET * 0.5:
            x_new = WRAP_OFFSET

        # Interpolate inlet velocity at this y-level
        if inlet_pts:
            ys_in = np.array([p[0] for p in inlet_pts])
            us_in = np.array([p[1] for p in inlet_pts])
            dy    = np.abs(ys_in - y_v)
            w     = 1.0 / (dy + 1e-6)
            w    /= w.sum()
            u_in  = (w[:, None] * us_in).sum(axis=0)
        else:
            u_in = np.zeros(2)

        _safe_move(v, np.array([x_new, y_v]), HC, bV)
        v.u = u_in
        v.p = p_analytical(x_new)
        # mass unchanged: same vertex, same m_cell -> M stays exactly M_total
        n_wrap += 1

    n_del_stp = n_wrap    # "wrapped" = logically recycled at outlet
    n_inj_stp = n_wrap    # immediately re-appeared at inlet
    cum_del  += n_del_stp
    cum_inj  += n_inj_stp

    # 9. Advance time counter
    t    += dt
    step += 1

    # 10. Diagnostics
    l2_tr, li_tr, l2_ss, li_ss = compute_errors(HC, t)
    M_now  = domain_mass(HC)
    Px_now = domain_momentum_x(HC)
    n_v    = sum(1 for _ in HC.V)
    n_int  = sum(1 for v in HC.V if not wall_criterion(v))
    n_w    = n_v - n_int
    dPx_pct = abs(Px_now - P_expected) / max(abs(P_expected), 1e-12) * 100.0

    ts['t'].append(t);                ts['step'].append(step)
    ts['dt_used'].append(dt);         ts['n_verts'].append(n_v)
    ts['n_interior'].append(n_int);   ts['n_wall'].append(n_w)
    ts['cum_inj'].append(cum_inj);    ts['cum_del'].append(cum_del)
    ts['l2_transient'].append(l2_tr); ts['linf_transient'].append(li_tr)
    ts['l2_ss'].append(l2_ss);        ts['linf_ss'].append(li_ss)
    ts['mass_total'].append(M_now)
    ts['momentum_x'].append(Px_now)
    ts['momentum_expected'].append(P_expected)
    # Extra CFD diagnostics
    _KE, _eps = compute_KE_dissipation(HC)
    _tau_bot, _ = compute_wss(HC)
    # Centreline velocity: inverse-distance-weighted average of all interior
    # vertices within a y-band around H/2. Avoids single-point noise.
    _dy_band = H / 4.0
    _cl_verts = [(abs(float(v.x_a[1]) - H/2), float(_safe_u(v)[0]))
                 for v in HC.V
                 if not wall_criterion(v)
                 and abs(float(v.x_a[1]) - H/2) < _dy_band]
    if _cl_verts:
        _w_cl  = np.array([1.0 / (p[0] + 1e-6) for p in _cl_verts])
        _u_cl  = float(np.dot(_w_cl, [p[1] for p in _cl_verts]) / _w_cl.sum())
    else:
        _u_cl = 0.0
    _mf_in  = compute_mass_flux(HC, 0.0   + L_period*0.5)
    _mf_mid = compute_mass_flux(HC, L_domain/2)
    _mf_out = compute_mass_flux(HC, L_domain - L_period*0.5)
    ts['KE'].append(_KE)
    ts['dissipation'].append(_eps)
    ts['tau_wall'].append(_tau_bot)
    ts['u_centerline'].append(_u_cl)
    ts['mass_flux_inlet'].append(_mf_in)
    ts['mass_flux_mid'].append(_mf_mid)
    ts['mass_flux_outlet'].append(_mf_out)

    # Profile snapshots at target times
    for t_snap in SNAP_TIMES:
        if t_snap not in _snap_done and abs(t - t_snap) < dt * 0.6:
            save_profile_snap(t, HC, f't={t:.2f} s ({t/tau_visc:.1f}tau)')
            _snap_done.add(t_snap)

    # Console log
    if step % 20 == 0 or n_inj_stp or n_del_stp:
        print(f"  {step:>5d}  {t:>7.3f}  {t/tau_visc:>6.2f}  "
              f"{dt:>8.5f}  {n_v:>5d}  "
              f"{l2_tr:>10.3e}  {l2_ss:>10.3e}  "
              f"{M_now/M_total:>8.5f}  {dPx_pct:>8.3f}%  "
              f"{n_inj_stp:>5d}  {n_del_stp:>5d}  {cum_inj-cum_del:>+6d}")

    # Frame
    if step % FRAME_EVERY == 0:
        fp = plot_frame(step, t, HC, _FRAMES,
                        cum_inj, cum_del, l2_tr, l2_ss,
                        ts['t'], ts['l2_transient'], ts['l2_ss'],
                        ts['tau_wall'], ts['u_centerline'])
        frame_paths.append(fp)

    # 11. Adaptive dt bounded by y-direction stability limits.
    #     For the y-only Laplacian stencil, the relevant length scale
    #     is dy_min (minimum y-gap between any two distinct y-levels).
    #     CFL uses U_max (advective transport in x, not y).
    all_int = [v for v in HC.V if not wall_criterion(v)]
    if all_int:
        # Collect all distinct y-positions (wall + interior)
        all_y  = sorted(set(round(float(v.x_a[1]), 10) for v in HC.V))
        dy_min = min(all_y[i+1] - all_y[i] for i in range(len(all_y)-1))                  if len(all_y) > 1 else H
        u_max  = max((float(np.linalg.norm(_safe_u(v))) for v in all_int),
                     default=1e-10)
        dt_cfl  = CFL_TARGET * dy_min / max(u_max, 1e-10)
        dt_visc = VN_COEFF   * dy_min**2 / nu
        dt = float(np.clip(min(dt_cfl, dt_visc), DT_MIN, DT_MAX))
    else:
        dt = DT_MAX

# ── Final frame and snapshot ──
fp = plot_frame(step, t, HC, _FRAMES,
                cum_inj, cum_del,
                ts['l2_transient'][-1], ts['l2_ss'][-1],
                ts['t'], ts['l2_transient'], ts['l2_ss'],
                ts['tau_wall'], ts['u_centerline'])
if fp not in frame_paths:
    frame_paths.append(fp)
save_profile_snap(t, HC, f't={t:.1f} s (final, {t/tau_visc:.0f}tau)')

print(f"\nDone:  t={t:.3f} s ({t/tau_visc:.1f}tau),  {step} steps")
print(f"  Final L2 vs SS        = {ts['l2_ss'][-1]:.3e}  (-> 0 at steady state)")
print(f"  Final L2 vs transient = {ts['l2_transient'][-1]:.3e}")
print(f"  Mass conservation     : M_final/M0 = {domain_mass(HC)/M_total:.6f}")
print(f"  Net vertices          : {cum_inj - cum_del:+d}")


# Animation

compile_gif(frame_paths, os.path.join(_FIG, 'animation.gif'))


# Summary plot 1: error convergence, mass, vertex count

t_arr  = np.array(ts['t'])
M_arr  = np.array(ts['mass_total'])
Px_arr = np.array(ts['momentum_x'])
Pe_arr = np.array(ts['momentum_expected'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Error
ax = axes[0]
ax.semilogy(t_arr, ts['l2_transient'], 'b-',  lw=2,
            label='$L_2$ vs transient')
ax.semilogy(t_arr, ts['l2_ss'],        'r--', lw=2,
            label='$L_2$ vs Poiseuille SS')
ax.axvline(tau_visc,   color='gray', lw=1.2, ls=':',
           label=f'tau={tau_visc:.3f} s')
ax.axvline(5*tau_visc, color='gray', lw=1.8, ls='--',
           label=f'5tau={5*tau_visc:.2f} s')
ax.set_xlabel('t  [s]'); ax.set_ylabel('$L_2$ velocity error  [m/s]')
ax.set_title('Error convergence  (plug -> Poiseuille)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)

# Mass
ax = axes[1]
ax.plot(t_arr, M_arr / M_total * 100.0, 'b-', lw=2, label='domain mass / $M_0$')
ax.axhline(100.0, color='red', lw=1.5, ls='--', label='100%')
ax.set_xlabel('t  [s]'); ax.set_ylabel('% of $M_0$')
ax.set_title('Mass conservation  (fixed assignment)', fontsize=13)
ax.set_ylim(0, 150)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Vertex count + injection/deletion/net
ax = axes[2]

# Compute per-step injection and deletion from cumulative arrays
cum_inj_arr = np.array(ts['cum_inj'])
cum_del_arr = np.array(ts['cum_del'])
net_arr     = cum_inj_arr - cum_del_arr

# Left y-axis: vertex counts
color_tot  = 'black'
color_int  = 'royalblue'
color_wall = 'green'
ax.plot(t_arr, ts['n_verts'],    '-',  color=color_tot,  lw=2,   label='N total')
ax.plot(t_arr, ts['n_interior'], '-',  color=color_int,  lw=1.5, label='N interior')
ax.plot(t_arr, ts['n_wall'],     '--', color=color_wall, lw=1.5, label='N wall')
ax.axhline(n_main, color='steelblue', ls=':', lw=1.2, label=f'$N_0$={n_main}')
ax.set_xlabel('t  [s]')
ax.set_ylabel('Vertex count', color='black')
ax.tick_params(axis='y', labelcolor='black')

# Right y-axis: cumulative injected, deleted, net
ax2 = ax.twinx()
ax2.plot(t_arr, cum_inj_arr, 's-', color='darkorange',  lw=1.2, ms=2,
         label='$\\Sigma$ injected')
ax2.plot(t_arr, cum_del_arr, 'v-', color='crimson',     lw=1.2, ms=2,
         label='$\\Sigma$ deleted')
ax2.plot(t_arr, net_arr,     'D-', color='purple',      lw=1.8, ms=2,
         label='net (inj$-$del)')
ax2.axhline(0, color='purple', ls=':', lw=0.8)
ax2.set_ylabel('Cumulative count', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Merge legends from both axes
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left',
          framealpha=0.85)
ax.set_title('Vertex count  +  injection / deletion / net', fontsize=12)
ax.grid(True, alpha=0.3)

plt.suptitle('Rest (u=0) -> Poiseuille  |  Simulation Summary', fontsize=15)
plt.tight_layout()
p_err = os.path.join(_FIG, 'error_vs_time.png')
plt.savefig(p_err, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p_err}")


# Summary plot 2: velocity profile development

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
y_fine   = np.linspace(0, H, 400)
cmap_dev = plt.cm.plasma

# Left: numerical snapshots
ax = axes[0]
ax.plot(u_poiseuille(y_fine), y_fine, 'k-', lw=3.5,
        label='Poiseuille SS', zorder=10)
ax.axvline(0.0, color='orange', lw=1.6, ls=':',
           label='IC: $u_x=0$ (rest)')
if profile_snaps:
    cols = cmap_dev(np.linspace(0.1, 0.9, len(profile_snaps)))
    for (t_s, y_s, u_s, lbl), col in zip(profile_snaps, cols):
        idx = np.argsort(y_s)
        ax.plot(u_s[idx], y_s[idx], '-o', color=col, lw=1.6, ms=3,
                label=lbl, alpha=0.85)
ax.set_xlabel('$u_x$  [m/s]', fontsize=13); ax.set_ylabel('y  [m]', fontsize=13)
ax.set_title('Velocity development  (numerical, u(y,0)=0)', fontsize=13)
ax.set_xlim(-0.05, U_max * 1.20); ax.set_ylim(-0.02, H + 0.02)
ax.legend(fontsize=9, loc='upper left'); ax.grid(True, alpha=0.3)

# Right: analytical transient
ax = axes[1]
ax.plot(u_poiseuille(y_fine), y_fine, 'k-', lw=3.5,
        label='Poiseuille SS', zorder=10)
ax.axvline(0.0, color='orange', lw=1.6, ls=':',
           label='IC: $u_x=0$ (rest)')
snap_t_an = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
cols_an   = cmap_dev(np.linspace(0.1, 0.9, len(snap_t_an)))
for t_s, col in zip(snap_t_an, cols_an):
    u_t = u_transient(y_fine, t_s)
    ax.plot(u_t, y_fine, '-', color=col, lw=1.8,
            label=f't={t_s:.2f} s ({t_s/tau_visc:.2f}tau)', alpha=0.85)
ax.set_xlabel('$u_x$  [m/s]', fontsize=13); ax.set_ylabel('y  [m]', fontsize=13)
ax.set_title('Velocity development  (analytical, impulsive start)', fontsize=13)
ax.set_xlim(-0.05, U_max * 1.20); ax.set_ylim(-0.02, H + 0.02)
ax.legend(fontsize=9, loc='upper left'); ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Rest (u=0) -> Poiseuille  |  tau = {tau_visc:.3f} s  |  '
    f'$U_{{max}}$ = {U_max:.2f} m/s  |  $U_{{mean}}$ = {U_mean:.2f} m/s',
    fontsize=14)
plt.tight_layout()
p_dev = os.path.join(_FIG, 'development_profiles.png')
plt.savefig(p_dev, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p_dev}")


# Summary plot 3: momentum budget

# The analytical transient momentum is:
#   P_x(t) = M_total * U_mean_inst(t)
# where U_mean_inst is the domain-averaged velocity at time t.
# From the Fourier series:  U_mean(t) = (1/H) integral_0^H u(y,t) dy
# = U_mean_ss * [1 - sum_n (8/((2n-1)^2 pi^2)) exp(-(2n-1)^2 pi^2 nu t/H^2)]
# (odd terms only — even Fourier modes cancel in the mean)
# Analytical body force: F_body = (-dPdx/rho) * M_total  [N]
# Analytical wall drag:  F_drag(t) = 2 * tau_w(t) * L_domain  [N]
# Net force = F_body - F_drag = dP_x/dt

t_fine = np.linspace(0, T_END, 2000)
y_quad = np.linspace(0, H, 500)
P_analytical = np.array([
    M_total / H * np.trapezoid(u_transient(y_quad, tt, n_terms=20), y_quad)
    for tt in t_fine
])
# Analytical wall shear: tau_w(t) = sum of Fourier series wall gradient
tau_analytical = np.array([
    mu * abs(float(np.gradient(u_transient(y_quad, tt, n_terms=20), y_quad)[0]))
    for tt in t_fine
])
F_body_const = (-dPdx / rho) * M_total   # constant body force [N]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: actual Px(t) vs analytical transient
ax = axes[0]
ax.plot(t_fine, P_analytical, 'k-', lw=2.5, label='Analytical P(t)')
ax.plot(t_arr,  Px_arr,       'b-', lw=2,   label='Numerical P(t) = sum m_i u_xi')
ax.axhline(M_total * U_mean, color='k', lw=1.2, ls='--',
           label=f'SS: M * U_mean = {M_total*U_mean:.3f}')
ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':',  label='tau')
ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--', label='5 tau')
ax.set_xlabel('t [s]'); ax.set_ylabel('x-momentum [kg m/s]')
ax.set_title('Momentum development (numerical vs analytical transient)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Right: force balance — body force vs wall drag
ax = axes[1]
ax.axhline(F_body_const, color='b', lw=2.5, ls='--',
           label=f'Body force F = {F_body_const:.2f} N (const)')
ax.plot(t_fine, 2*tau_analytical*L_domain, 'r-', lw=2,
        label='Analytical wall drag 2*tau_w(t)*L')
if ts['tau_wall']:
    ax.plot(t_arr, 2*np.array(ts['tau_wall'])*L_domain, 'r:', lw=2,
            label='Numerical wall drag')
ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':',  label='tau')
ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--', label='5 tau')
ax.set_xlabel('t [s]'); ax.set_ylabel('Force [N]')
ax.set_title('Force balance: body force vs wall drag (equal at steady state)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.suptitle('Momentum Budget  |  Force Balance', fontsize=14)
plt.tight_layout()
p_mom = os.path.join(_FIG, 'mass_momentum.png')
plt.savefig(p_mom, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p_mom}")

print("\nAll outputs written to:", _FIG)


# Summary plot 4: Wall shear stress + centreline velocity

tau_ss  = mu * 4.0 * U_max / H          # analytical SS: 4 Pa
u_cl_ss = U_max                          # centreline SS: 1 m/s

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(t_arr, ts['u_centerline'], 'b-', lw=2, label='Numerical $u_{cl}(t)$')
ax.axhline(u_cl_ss, color='k', lw=1.5, ls='--',
           label=f'SS $U_{{max}}$={u_cl_ss:.2f} m/s')
ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':')
ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--')
ax.set_xlabel('t [s]'); ax.set_ylabel('$u_{cl}$ [m/s]')
ax.set_title('Centreline velocity development', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_arr, ts['tau_wall'], 'r-', lw=2, label='Numerical tau_w(t)')
ax.axhline(tau_ss, color='k', lw=1.5, ls='--',
           label=f'SS $\\tau_w$={tau_ss:.1f} Pa')
ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':',  label='$\\tau$')
ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--', label='$5\\tau$')
ax.set_xlabel('t [s]'); ax.set_ylabel('tau_w [Pa]')
ax.set_title('Wall shear stress development', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t_arr, ts['mass_flux_inlet'],  'b-',  lw=2, label=r'Inlet  ($x\approx 0$)')
ax.plot(t_arr, ts['mass_flux_mid'],    'g--', lw=2, label='Mid    (x=L/2)')
ax.plot(t_arr, ts['mass_flux_outlet'], 'r:',  lw=2, label=r'Outlet ($x\approx L$)')
ax.axhline(rho * U_mean * H, color='k', lw=1.5, ls='--',
           label=f'SS flux={rho*U_mean*H:.4f}')
ax.set_xlabel('t [s]'); ax.set_ylabel(r'$\dot{m}$ [kg/m/s]')
ax.set_title('Mass flux across cross-sections', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.suptitle('Wall Shear Stress  |  Centreline Velocity  |  Mass Flux', fontsize=15)
plt.tight_layout()
p_wss = os.path.join(_FIG, 'wss_centerline_flux.png')
plt.savefig(p_wss, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p_wss}")


# Summary plot 5: Kinetic energy + viscous dissipation + pressure

# Analytical SS kinetic energy:  KE_ss = int_0^H 0.5*rho*u_P(y)^2 dy * L_domain
#   = 0.5*rho*L_domain * int_0^H [U_max*(1-(2y/H-1)^2)]^2 dy
#   = 0.5*rho*L_domain*U_max^2 * (8/15)*H
y_q  = np.linspace(0, H, 2000)
KE_ss = 0.5 * rho * L_domain * np.trapezoid(u_poiseuille(y_q)**2, y_q)
# Analytical SS dissipation: eps_ss = mu * int (du_P/dy)^2 dy * L_domain
dudy_ss = np.gradient(u_poiseuille(y_q), y_q)
eps_ss  = mu * L_domain * np.trapezoid(dudy_ss**2, y_q)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(t_arr, ts['KE'], 'b-', lw=2, label='KE numerical')
ax.axhline(KE_ss, color='k', lw=1.5, ls='--',
           label=f'SS KE={KE_ss:.4f} J')
ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':')
ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--')
ax.set_xlabel('t [s]'); ax.set_ylabel('KE [J]')
ax.set_title('Kinetic energy  (1/2) * sum_i m_i * |u_i|^2', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_arr, ts['dissipation'], 'r-', lw=2, label='Dissipation (numerical)')
ax.axhline(eps_ss, color='k', lw=1.5, ls='--',
           label=f'SS = {eps_ss:.4f} W')
ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':',  label='tau')
ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--', label='5 tau')
ax.set_xlabel('t [s]')
ax.set_ylabel('Viscous dissipation rate [W]')
ax.set_title('Viscous dissipation rate', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Pressure field at final state: scatter p vs x, compare with analytical
ax = axes[2]
fin_verts = [(float(v.x_a[0]), float(getattr(v, 'p', 0.0)))
             for v in HC.V if not wall_criterion(v)]
if fin_verts:
    xp, pp = zip(*sorted(fin_verts))
    ax.scatter(xp, pp, c='royalblue', s=30, zorder=3, label='Numerical p')
x_line = np.linspace(0, L_domain, 200)
ax.plot(x_line, dPdx * x_line, 'k--', lw=2,
        label=f'Analytical  p={dPdx:.0f}x')
ax.set_xlabel('x [m]'); ax.set_ylabel('p [Pa]')
ax.set_title('Pressure field (should be linear)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.suptitle('Energy Budget  |  Viscous Dissipation  |  Pressure Field', fontsize=15)
plt.tight_layout()
p_energy = os.path.join(_FIG, 'energy_dissipation_pressure.png')
plt.savefig(p_energy, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p_energy}")


# Summary plot 6: Dual-cell areas at final state + vorticity profile

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: dual cell area distribution
dual_final = compute_dual_cells(HC)
int_areas  = [d[1] for d in dual_final if d[0] is not None and not d[2]]
wall_areas = [d[1] for d in dual_final if d[0] is not None and d[2]]
ax = axes[0]
if int_areas:
    ax.hist(int_areas,  bins=12, color='royalblue', alpha=0.7,
            label=f'Interior (n={len(int_areas)})', edgecolor='k')
if wall_areas:
    ax.hist(wall_areas, bins=8,  color='gray', alpha=0.6,
            label=f'Wall (n={len(wall_areas)})', edgecolor='k')
ax.axvline(M_total / rho / n_int_0, color='red', lw=2, ls='--',
           label=f'Target cell = {M_total/rho/n_int_0:.4f} m$^2$')
ax.set_xlabel('Voronoi cell area [m$^2$]')
ax.set_ylabel('Count')
ax.set_title('Dual-cell area distribution (final)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Right: vorticity profile at final state (numerical vs analytical)
ax = axes[1]
omega_final = compute_vorticity(HC)
# Analytical vorticity for Poiseuille: omega_z = -du_P/dy = 4*U_max*(2y/H-1)/H
y_fine2 = np.linspace(0, H, 300)
omz_an = -np.gradient(u_poiseuille(y_fine2), y_fine2)
ax.plot(omz_an, y_fine2, 'k-', lw=2.5, label='Analytical SS')
# Numerical: group by y, take mean at each y level
num_pts = [(float(v.x_a[1]), omega_final.get(id(v), 0.0))
           for v in HC.V if not wall_criterion(v)]
if num_pts:
    num_pts.sort(key=lambda p: p[0])
    yn = np.array([p[0] for p in num_pts])
    on = np.array([p[1] for p in num_pts])
    ax.scatter(on, yn, c='royalblue', s=40, zorder=4, label='Numerical')
ax.axvline(0, color='gray', lw=0.8, ls=':')
ax.set_xlabel(r'$\omega_z = -\partial u/\partial y$ [1/s]')
ax.set_ylabel('y [m]')
ax.set_title('Vorticity profile at final state', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.suptitle('Dual-Cell Quality  |  Vorticity Profile  |  Final State', fontsize=15)
plt.tight_layout()
p_dual = os.path.join(_FIG, 'dual_cells_vorticity.png')
plt.savefig(p_dual, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p_dual}")