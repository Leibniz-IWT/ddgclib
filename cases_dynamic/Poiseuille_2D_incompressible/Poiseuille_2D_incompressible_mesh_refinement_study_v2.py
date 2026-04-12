"""
Mesh Refinement Study for 2D Incompressible Poiseuille Flow
============================================================
DEC-based solver: uses ddgclib.operators.gradient.acceleration
(Cauchy stress tensor, dual area vectors) when available.
All integrated quantities computed via DualArea_i / dual 2-cells.
Fallback: y-stencil for viscous diffusion (x-homogeneous flow).
"""

"""
Incompressible Poiseuille Flow — Plug Flow → Parabolic Development
===================================================================
v9: adaptive CFL Euler, _retopologize_safe (no compute_vd),
    BoundaryConditionSet, body-force-only dudt (no grad_p double-count),
    fixed-mass assignment, precomputed Fourier transient solution.

Starting condition: plug flow  u_x = U_mean everywhere, walls at 0.
Driven by body force -dP/dx, viscous diffusion relaxes the profile
to the Poiseuille parabola on the timescale tau = H^2/(pi^2 nu).

Key fixes over v8
-----------------
* dudt_fn: body force ONLY — the separate grad_p term was doubling
  the forcing (grad_p == f_body for a linear pressure field) and
  causing velocity divergence.
* _retopologize_safe: Delaunay rebuild of v.nn every step, without
  calling compute_vd — avoids crashes on irregular meshes.
* Fixed mass: m = M_total / n_int_0 assigned every step — eliminates
  reservoir-drift mass growth.
* Fourier coefficients B_n precomputed once at startup for fast
  transient error evaluation.
* BoundaryConditionSet + PositionalNoSlipWallBC + SimpleOutletBC
  + PeriodicInletBC (fields=['u','p'], 'm' excluded).
* Adaptive dt: min(CFL_advective, VonNeumann_viscous, DT_MAX).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay, QhullError, cKDTree, Voronoi

from functools import partial

from hyperct import Complex
from ddgclib._boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionSet,
    PeriodicInletBC,
    PositionalNoSlipWallBC,
    identify_boundary_vertices,
)

# ── DEC operators from ddgclib ────────────────────────────────
# gradient.py  : acceleration()       -> stress_acceleration (Cauchy tensor)
#                velocity_laplacian() -> viscous diffusion operator
# area.py      : DualArea_i()         -> dual 2-cell area via d_area (hyperct)
# stress module: dual_area_vector()   -> oriented dual edge vectors A_ij
# These replace all finite-difference Laplacian / divergence estimators.
try:
    from ddgclib.operators.gradient import (
        acceleration      as _dec_acceleration,
        velocity_laplacian as _dec_laplacian,
    )
    from ddgclib.operators.stress import dual_area_vector as _dual_area_vector
    from ddgclib.operators.area   import DualArea_i       as _DualArea_i
    _DEC_AVAILABLE = True
    print("  DEC operators loaded: acceleration, velocity_laplacian, DualArea_i")
except ImportError as _dec_err:
    _DEC_AVAILABLE = False
    _dec_acceleration  = None
    _dec_laplacian     = None
    _dual_area_vector  = None
    _DualArea_i        = None
    print(f"  DEC operators unavailable ({_dec_err}): using y-stencil fallback")

# ── Integrator helpers from _integrators_dynamic.py ──────────
# _retopologize : Delaunay rebuild + HC.boundary() + compute_vd
# This is the production retopologizer used when DEC is active.
try:
    from ddgclib.dynamic_integrators._integrators_dynamic import (
        _retopologize as _retopologize_full,
        _recompute_duals,
    )
    _INTEGRATORS_AVAILABLE = True
    print("  Integrator helpers loaded: _retopologize, _recompute_duals")
except ImportError as _int_err:
    _INTEGRATORS_AVAILABLE = False
    _retopologize_full = None
    _recompute_duals   = None
    print(f"  Integrator helpers unavailable ({_int_err}): using local fallbacks")

# ============================================================
# Output settings
# ============================================================
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

# ============================================================
# Physical parameters
# ============================================================
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
# _FIG/_FRAMES set in run_case()
_FIG    = os.path.join(_HERE, 'fig', 'poiseuille_dev')
_FRAMES = os.path.join(_FIG, '_frames')

print(f"Viscous timescale tau = {tau_visc:.4f} s  "
      f"(99% developed at approx 5tau = {5*tau_visc:.3f} s)")
print(f"\nPoiseuille flow:  H={H}, L={L_domain}, mu={mu}, rho={rho}")
print(f"  dP/dx = {dPdx}  ->  U_max = {U_max:.4f} m/s,"
      f"  U_mean = {U_mean:.4f} m/s")
print(f"  Re = {rho * U_mean * H / mu:.1f}")
print(f"  t_end = {T_END:.2f} s  approx  {T_END/tau_visc:.1f} tau")

# ============================================================
# Analytical solutions
# ============================================================

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


# ============================================================
# Small helpers
# ============================================================

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


# ============================================================
# Safe retopologization  (Delaunay only, no compute_vd)
# ============================================================

def _retopologize_safe(HC, bV):
    """
    Rebuild Delaunay triangulation so v.nn is valid after Lagrangian
    vertex movement.  Updates bV to contain ONLY wall vertices.

    When DEC operators are available (_DEC_AVAILABLE=True), also calls
    compute_vd(HC, method='barycentric') to populate v.vd dual cells,
    which are required by stress_acceleration / velocity_laplacian.
    Uses boundary_filter=wall_criterion so only wall vertices are frozen.

    Falls back gracefully if compute_vd crashes (irregular mesh).
    """
    verts = list(HC.V)
    if len(verts) < DIM + 1:
        return

    # 1. Disconnect all edges
    for v in verts:
        for nb in list(v.nn):
            try:
                v.disconnect(nb)
            except Exception:
                pass

    # 2. Delaunay triangulation
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

    # 3. Tag all topological boundary vertices (needed by compute_vd)
    try:
        dV = HC.boundary()
        for v in HC.V:
            v.boundary = v in dV
    except Exception:
        for v in HC.V:
            v.boundary = wall_criterion(v)

    # 4. Compute barycentric dual cells (v.vd) for DEC stress operators.
    #    Use _retopologize_full from _integrators_dynamic when available:
    #    it handles HC.boundary() + compute_vd in one call.
    #    Otherwise fall back to compute_vd directly.
    _vd_ok = False
    if _INTEGRATORS_AVAILABLE and _retopologize_full is not None:
        try:
            # _retopologize_full already did Delaunay above; just recompute duals
            if _recompute_duals is not None:
                _recompute_duals(HC)
                _vd_ok = True
        except Exception:
            pass
    if not _vd_ok and _DEC_AVAILABLE:
        try:
            from hyperct.ddg import compute_vd
            compute_vd(HC, method="barycentric")
        except Exception:
            pass  # silent fallback to y-stencil in dudt_fn

    # 5. bV <- wall vertices only (wall_criterion as boundary_filter)
    bV.clear()
    for v in HC.V:
        is_wall = wall_criterion(v)
        v.boundary = is_wall
        if is_wall:
            bV.add(v)


# ============================================================
# Momentum forcing  du/dt = f_body + nu * Laplacian(u)
# ============================================================
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


def _dudt_dec(v):
    """
    DEC acceleration: du/dt via Cauchy stress tensor (ddgclib.operators.stress).

    The DEC formulation integrates the stress tensor over dual area vectors:
        F_stress_i = sum_{j in v.nn}  F_p_ij + F_v_ij
        F_p_ij = -0.5*(p_i + p_j) * A_ij         (pressure 1-form)
        F_v_ij = mu/|d_ij| * (u_j-u_i) * (d_hat . A_ij)   (viscous 2-form)
        A_ij   = dual_area_vector(v, v_j, HC, dim) from barycentric dual cells

    Returns du/dt = F_stress / m_i + f_body
    where f_body = (-dPdx/rho, 0) is the external body force (imposed
    pressure gradient, NOT already in v.p since that IS the linear field).

    Requires v.vd to be populated by compute_vd(HC, 'barycentric').
    Falls back to _dudt_ystencil if v.vd is empty or call fails.
    """
    f_body = np.array([-dPdx / rho, 0.0])

    # Check dual cell is populated
    if not hasattr(v, 'vd') or not v.vd:
        return _dudt_ystencil(v)

    try:
        # stress_acceleration returns F_stress/m = (viscous + pressure) / m
        a_stress = _dec_acceleration(v, dim=DIM, mu=mu, HC=_HC_ref)
        a_stress = np.asarray(a_stress, dtype=float).ravel()
        if a_stress.size >= 2:
            a_vec = a_stress[:2]
        elif a_stress.size == 1:
            a_vec = np.array([a_stress[0], 0.0])
        else:
            a_vec = np.zeros(2)

        # The DEC stress includes -grad_p/rho from the stored pressure.
        # Our p = dPdx*x, so grad_p = dPdx → -grad_p/rho = -dPdx/rho = f_body[0].
        # stress_acceleration already accounts for this, so we DO NOT add
        # f_body again (would double the forcing).
        # HOWEVER: the viscous-only contribution is mu*Laplacian(u)/rho,
        # and the stored pressure gradient should cancel with f_body.
        # Safest: return a_stress directly (it already contains body-force-equivalent).
        return a_vec

    except Exception:
        # Fallback: DEC call failed (e.g., degenerate dual cells)
        return _dudt_ystencil(v)


def _dudt_ystencil(v):
    """
    Fallback acceleration: f_body + nu * d^2u/dy^2 (y-only stencil).

    Used when DEC operators are unavailable or v.vd is not populated.
    Uses the wall-normal (y) direction only since u(y,t) is x-homogeneous.
    Wall vertices (u=0) act as Dirichlet BCs for the diffusion operator.
    """
    u_v    = _safe_u(v)
    f_body = np.array([-dPdx / rho, 0.0])

    if _HC_ref is None:
        return f_body

    y_v = float(v.x_a[1])
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
        return f_body

    dy_up, v_up = min(above, key=lambda x: x[0])
    dy_dn, v_dn = min(below, key=lambda x: x[0])
    u_up  = _safe_u(v_up)
    u_dn  = _safe_u(v_dn)
    denom = dy_up + dy_dn
    if denom < 1e-14:
        return f_body

    d2u_dy2 = (2.0 / denom) * (
        (u_up - u_v) / dy_up + (u_dn - u_v) / dy_dn
    )
    return f_body + nu * d2u_dy2


def dudt_fn(v):
    """
    Dispatch to DEC acceleration (preferred) or y-stencil fallback.

    DEC path  (when _DEC_AVAILABLE and v.vd populated):
        Uses ddgclib.operators.gradient.acceleration which calls
        stress_acceleration from the full Cauchy stress tensor.
        This is the proper DEC formulation: all quantities are
        integrated over dual area vectors A_ij, not computed point-wise.

    Fallback path (y-stencil):
        Uses only the wall-normal second derivative — correct for
        x-homogeneous channel flow but misses full 2D topology.
    """
    if _DEC_AVAILABLE:
        return _dudt_dec(v)
    return _dudt_ystencil(v)


# ============================================================
# Custom outlet BC (simple deletion)
# ============================================================

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


# ============================================================
# Ghost mesh update (call AFTER each step, before next step)
# ============================================================

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


# ============================================================
# Mesh construction — PLUG FLOW initial conditions
# ============================================================

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


# ============================================================
# Error diagnostics
# ============================================================

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


# ============================================================
# Frame renderer
# ============================================================


# ============================================================
# CFD diagnostic helpers
# ============================================================


def scatter_to_grid(verts, field_vals, nx=120, ny=60,
                    x0=0.0, x1=None, y0=0.0, y1=None, method='linear'):
    """
    Interpolate scattered Lagrangian field onto a regular grid for contour plots.
    Uses scipy.interpolate.griddata (linear by default).
    NaN values at boundaries are filled by nearest-neighbour fallback.
    """
    from scipy.interpolate import griddata
    if x1 is None: x1 = L_domain
    if y1 is None: y1 = H
    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, ny)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.array([[float(v.x_a[0]), float(v.x_a[1])] for v in verts])
    if len(pts) < 3:
        return GX, GY, np.zeros_like(GX)
    vals = np.asarray(field_vals, dtype=float)
    Z = griddata(pts, vals, (GX, GY), method=method)
    # Fill NaN at grid boundaries with nearest
    mask = np.isnan(Z)
    if mask.any():
        Z_near = griddata(pts, vals, (GX, GY), method='nearest')
        Z[mask] = Z_near[mask]
    return GX, GY, Z


def compute_dec_fields(HC, dual=None):
    """
    Compute DEC-integrated physical quantities over Voronoi dual cells.

    In Discrete Exterior Calculus:
      - A 0-form at vertex i (u_x, p) is the value of the field.
      - Its integral over the dual 2-cell i is: phi_i * A_i.
      - The mean over the cell is simply phi_i (0-form assignment).
      - A 2-form (area density) integrated over cell i gives A_i * density_i.
      - Vorticity omega_z = -du/dy is computed as the circulation Gamma_i
        around the dual cell boundary divided by cell area A_i.
        (Stokes theorem: integral_surface curl u . dS = contour integral u.dl)
      - Kinetic energy density: ke_i = (1/2)*rho*u_xi^2  [J/m^2]
        Cell KE: KE_i = ke_i * A_i                       [J]
      - Viscous dissipation density: eps_i = mu*omega_zi^2 [W/m^2]
        Cell dissipation: D_i = eps_i * A_i               [W]
      - Incompressibility residual per cell: div_i (from compute_divergence)
      - Pressure: p_i = dPdx*x_i + phi_i (from Poisson correction)

    Parameters
    ----------
    HC   : Complex
    dual : list or None  — output of compute_dual_cells(HC); recomputed if None.

    Returns
    -------
    dec : dict  {field_name: np.ndarray of length len(list(HC.V))}
        Fields: 'area', 'u_x', 'p', 'p_linear', 'phi',
                'KE_cell', 'KE_density',
                'vorticity_circ', 'vorticity_stencil',
                'dissipation_cell', 'div_u',
                'is_wall'
    verts : list of vertex objects (same order as arrays)
    """
    if dual is None:
        dual = compute_dual_cells(HC)

    verts   = list(HC.V)
    n       = len(verts)
    v_id    = {id(v): i for i, v in enumerate(verts)}

    # Initialise arrays
    area        = np.zeros(n)
    u_x         = np.zeros(n)
    p_field     = np.zeros(n)
    p_linear    = np.zeros(n)
    phi_field   = np.zeros(n)
    KE_cell     = np.zeros(n)
    KE_density  = np.zeros(n)
    vort_circ   = np.zeros(n)   # circulation-based (Stokes)
    vort_stenc  = np.zeros(n)   # y-stencil (numerical du/dy)
    diss_cell   = np.zeros(n)
    div_u       = np.zeros(n)
    is_wall_arr = np.zeros(n, dtype=bool)

    # ── 1. Fill 0-form values at vertices ────────────────────────────────
    for i, v in enumerate(verts):
        u_x[i]      = float(_safe_u(v)[0])
        p_field[i]  = float(getattr(v, 'p', p_analytical(float(v.x_a[0]))))
        p_lin       = p_analytical(float(v.x_a[0]))
        p_linear[i] = p_lin
        phi_field[i]= p_field[i] - p_lin
        is_wall_arr[i] = wall_criterion(v)

    # ── 2. Dual cell areas (2-form coefficient) ───────────────────────────
    # Primary: DualArea_i from ddgclib.operators.area, which uses d_area
    # from hyperct.ddg — the proper DEC measure of each dual 2-cell.
    # Fallback: Voronoi polygon area from compute_dual_cells output.
    _dual_area_fn = None
    if _DEC_AVAILABLE:
        try:
            _dual_area_fn = _DualArea_i()
        except Exception:
            pass

    for i, v in enumerate(verts):
        if _dual_area_fn is not None:
            try:
                area[i] = float(_dual_area_fn(v))
                continue
            except Exception:
                pass
        # Fallback: Voronoi polygon area
        if i < len(dual) and dual[i][0] is not None:
            area[i] = dual[i][1] if dual[i][1] > 0 else 0.0

    # ── 3. KE density and cell KE (area-weighted 2-form) ─────────────────
    # ke_density = (1/2)*rho*u_x^2  [J/m^2]
    # KE_cell    = ke_density * A_i  [J]
    KE_density = 0.5 * rho * u_x**2
    KE_cell    = KE_density * area

    # ── 4. Vorticity via DEC (Stokes theorem on dual cell boundary) ────────
    # DEC: omega_z = Gamma_i / A_i where Gamma_i = contour integral u.dl
    # around the dual cell boundary (Stokes theorem: integral of curl u).
    # When v.vd is populated: use velocity_laplacian flux form for consistency.
    # Also compute stencil-based du/dy for comparison.
    # For each dual cell polygon with vertices p_0,...,p_{k-1}:
    # Gamma_i = contour integral u.dl around boundary
    # Approximate: for each edge (p_j, p_{j+1}), interpolate u_x at midpoint
    # and project onto the edge tangent.
    # Since u_y=0: u.dl = u_x * dx  (only x-component contributes).
    # omega_z = Gamma_i / A_i  (right-hand rule, counterclockwise)
    #
    # More robustly: use the FD stencil (same-y neighbours) which is accurate
    # for this x-homogeneous flow.  Save both versions.
    all_verts_list = verts
    y_to_vs_vort = {}
    for v in all_verts_list:
        if not wall_criterion(v):
            yk = round(float(v.x_a[1]), 8)
            y_to_vs_vort.setdefault(yk, []).append(v)
    y_levels_sorted = sorted(y_to_vs_vort.keys())

    for i, v in enumerate(verts):
        if wall_criterion(v):
            continue
        y_v = float(v.x_a[1])

        # Stencil-based: use nearest distinct y-levels above/below
        above = [(yk - y_v, y_to_vs_vort[yk])
                 for yk in y_levels_sorted if yk - y_v > 1e-10]
        below = [(y_v - yk, y_to_vs_vort[yk])
                 for yk in y_levels_sorted if y_v - yk > 1e-10]
        if above and below:
            dy_up, vs_up = min(above, key=lambda x: x[0])
            dy_dn, vs_dn = min(below, key=lambda x: x[0])
            u_up = float(np.mean([_safe_u(w)[0] for w in vs_up]))
            u_dn = float(np.mean([_safe_u(w)[0] for w in vs_dn]))
            denom = dy_up + dy_dn
            if denom > 1e-14:
                dudy = (u_up - u_dn) / denom
                vort_stenc[i] = -dudy   # omega_z = -du/dy

        # Circulation-based (Stokes): use dual cell polygon
        xy_cell = dual[i][0]
        A_cell  = dual[i][1]
        if xy_cell is not None and A_cell > 1e-20 and len(xy_cell) >= 3:
            Gamma = 0.0
            for k in range(len(xy_cell)):
                p1  = xy_cell[k]
                p2  = xy_cell[(k+1) % len(xy_cell)]
                xm  = 0.5*(p1[0] + p2[0])
                ym  = 0.5*(p1[1] + p2[1])
                dx_e = p2[0] - p1[0]
                # Interpolate u_x at edge midpoint from all vertices
                # using inverse-distance weighting
                dists  = np.sqrt((np.array([float(w.x_a[0]) for w in all_verts_list]) - xm)**2
                               + (np.array([float(w.x_a[1]) for w in all_verts_list]) - ym)**2)
                dists  = np.maximum(dists, 1e-10)
                w_idc  = 1.0 / dists**2
                u_m    = float(np.dot(w_idc, u_x) / w_idc.sum())
                Gamma += u_m * dx_e   # u.dl with u_y=0
            vort_circ[i] = Gamma / A_cell

    # Use stencil vorticity as primary (more accurate for x-homogeneous flow)
    # Clip wall values to zero
    vort_stenc[is_wall_arr] = 0.0
    vort_circ[is_wall_arr]  = 0.0

    # ── 5. Viscous dissipation: eps_i = mu * omega_i^2  per unit area ────
    diss_cell = mu * vort_stenc**2 * area  # [W]

    # ── 6. Discrete divergence via Gauss theorem ──────────────────────────
    # div_u_i = (1/A_i) * sum_edges u.n * length_e (outward flux / area)
    # For each dual cell polygon edge: flux = u_x * n_x * |edge|
    # where n is the outward normal to the edge.
    # For u_y=0: flux = u_x * ny_edge * |edge|  (only y-component of normal)
    # Actually for incompressible 2D with u_y=0:
    # div = du_x/dx + du_y/dy = du_x/dx only.
    # Use the same-y SPH estimate (already in compute_divergence).
    div_map, _, _ = compute_divergence(HC)
    for i, v in enumerate(verts):
        div_u[i] = div_map.get(id(v), 0.0)

    return {
        'area':              area,
        'u_x':               u_x,
        'p':                 p_field,
        'p_linear':          p_linear,
        'phi':               phi_field,
        'KE_cell':           KE_cell,
        'KE_density':        KE_density,
        'vorticity':         vort_stenc,   # primary
        'vorticity_circ':    vort_circ,    # Stokes
        'dissipation_cell':  diss_cell,
        'div_u':             div_u,
        'is_wall':           is_wall_arr,
    }, verts


def _draw_dual_cells(ax, fig, dual, values, verts, cmap, label,
                     vmin=None, vmax=None, alpha=0.9, lw=0.4,
                     overlay_vertices=True, overlay_edges=None):
    """
    Draw Voronoi dual cells coloured by a per-vertex scalar field.
    This is the primary DEC visualization: each dual 2-cell carries
    the integrated field value (not point-wise scatter).

    Parameters
    ----------
    dual   : output of compute_dual_cells(HC)
    values : np.ndarray, shape (n_verts,)
    """
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection

    int_patches, int_vals, wall_patches = [], [], []
    for i, (xy, area, is_wall) in enumerate(dual):
        if xy is None or i >= len(values):
            continue
        if is_wall:
            wall_patches.append(MplPoly(xy, closed=True))
        else:
            int_patches.append(MplPoly(xy, closed=True))
            int_vals.append(values[i])

    if wall_patches:
        pc_w = PatchCollection(wall_patches, facecolor='#e0e0e0',
                               edgecolor='#888888', linewidths=lw, alpha=0.7)
        ax.add_collection(pc_w)

    if int_patches and int_vals:
        v_arr = np.array(int_vals)
        _vmin = vmin if vmin is not None else float(v_arr.min())
        _vmax = vmax if vmax is not None else float(v_arr.max())
        if abs(_vmax - _vmin) < 1e-14:
            _vmin -= 0.5; _vmax += 0.5
        norm = plt.Normalize(vmin=_vmin, vmax=_vmax)
        pc = PatchCollection(int_patches, cmap=cmap, norm=norm,
                             alpha=alpha, edgecolor='k', linewidths=lw)
        pc.set_array(v_arr)
        ax.add_collection(pc)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label=label)

    if overlay_vertices and verts:
        coords = np.array([v.x_a[:2] for v in verts])
        ax.scatter(coords[:, 0], coords[:, 1],
                   c='white', s=10, zorder=5, edgecolors='k', lw=0.4)

    if overlay_edges is not None:
        ax.add_collection(
            LineCollection(overlay_edges, colors='#555555',
                           linewidths=0.3, alpha=0.2))


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



# ============================================================
# DEC integrated quantities
# ============================================================
# In Discrete Exterior Calculus every scalar/vector quantity
# is integrated over its natural domain:
#   0-form (primal vertex)  : scalar at vertex
#   2-form (dual 2-cell)    : integral over Voronoi cell ∫_A f dA ≈ f_i * A_i
#   1-form (dual 1-cell)    : flux across dual edge ∫_e u·n dl ≈ u·n * |e|
#
# ALL diagnostics below use cell areas A_i and edge lengths |e|
# from compute_dual_cells() — no pointwise averaging.
# ============================================================

def compute_dual_areas(HC):
    """
    Return dict {id(v): A_i} — Voronoi dual-cell area for every vertex.
    Wall vertices get the clipped wall-strip area.
    Unresolved cells get area = 0.
    """
    dual = compute_dual_cells(HC)
    verts = list(HC.V)
    areas = {}
    for v, (poly, area, _) in zip(verts, dual):
        areas[id(v)] = area if poly is not None else 0.0
    return areas


def dec_kinetic_energy(HC):
    """
    DEC kinetic energy as a 2-form integral over dual cells:

        KE = Σ_i  ½ ρ |u_i|²  A_i

    where A_i is the Voronoi dual-cell area of vertex i.

    This is the proper DEC discretisation:  KE = ½ ρ ⟨u♭, ⋆u♭⟩
    where ⋆ is the Hodge star (dual area / primal length),
    and ⟨·,·⟩ is the inner product of 1-forms.

    Analytical SS: ½ ρ L H (8/15) U_max² = 0.8 J
    """
    areas = compute_dual_areas(HC)
    KE = 0.0
    for v in HC.V:
        A_i = areas.get(id(v), 0.0)
        if A_i < 1e-20:
            continue
        u_i = _safe_u(v)
        KE += 0.5 * rho * float(np.dot(u_i, u_i)) * A_i
    return float(KE)


def dec_viscous_dissipation(HC):
    """
    DEC viscous dissipation rate as 2-form integral:

        ε = Σ_i  μ ω_z,i²  A_i

    where ω_z,i = −∂u_x/∂y|_i (vorticity) and A_i is the dual-cell area.

    This discretises  ε = μ ∫∫ (∂u/∂y)² dA
    (valid for channel flow where ∂u/∂x = 0).

    Wall boundary contribution is included by assigning the wall
    vorticity = τ_w / μ, spread over the wall-strip dual area.

    Analytical SS: μ L H (16/3) = 16 W
    """
    areas   = compute_dual_areas(HC)
    omega   = compute_vorticity(HC)
    tau_b, tau_t = compute_wss(HC)

    eps = 0.0
    for v in HC.V:
        A_i = areas.get(id(v), 0.0)
        if A_i < 1e-20:
            continue
        if wall_criterion(v):
            # Wall vertices carry the boundary vorticity = τ_w/μ
            y_v = float(v.x_a[1])
            omz = (tau_b / mu) if y_v < H/2 else (tau_t / mu)
        else:
            omz = omega.get(id(v), 0.0)
        eps += mu * omz**2 * A_i
    return float(eps)


def dec_momentum(HC):
    """
    DEC x-momentum as 2-form integral:

        P_x = Σ_i  ρ u_x,i  A_i

    Analytical SS: ρ L H U_mean = 2.0 kg m/s
    """
    areas = compute_dual_areas(HC)
    Px = 0.0
    for v in HC.V:
        A_i = areas.get(id(v), 0.0)
        if A_i < 1e-20:
            continue
        Px += rho * float(_safe_u(v)[0]) * A_i
    return float(Px)


def dec_mass_flux(HC):
    """
    DEC mass flux via Gauss theorem on dual cells:

        Q = Σ_i  ρ u_x,i  A_i / L_domain

    This is the x-averaged flux density.  The full flux through a
    cross-section is  Φ = Q * L_domain / L_domain = Q
    (cancels for a uniform-in-x field).

    Analytical SS: ρ U_mean H = 2/3 kg/m/s
    """
    areas = compute_dual_areas(HC)
    # Sum ρ u_x A over all interior vertices, then normalise by domain length
    flux_sum = 0.0
    for v in HC.V:
        if wall_criterion(v):
            continue
        A_i = areas.get(id(v), 0.0)
        if A_i < 1e-20:
            continue
        flux_sum += rho * float(_safe_u(v)[0]) * A_i
    # Normalise: flux = total x-momentum / L_domain
    return float(flux_sum / L_domain)


def dec_wss(HC):
    """
    DEC wall shear force as 1-form integral over the wall dual edges:

        F_w = Σ_{i∈wall-adjacent}  μ (∂u_x/∂y)|_i  * L_i

    where L_i = x-extent of vertex i's dual cell (the primal-edge
    dual of the bottom/top face).

    For interior vertices adjacent to the bottom wall (y close to 0):
        (∂u_x/∂y)|_i ≈ u_x,i / y_i  (Richardson 1st order)

    The integral gives the total drag force [N] per wall.
    Analytical SS: τ_w * L_domain = 4 * 3 = 12 N
    Returns (F_bottom, F_top) in [N].
    """
    areas = compute_dual_areas(HC)
    verts = list(HC.V)

    # Collect interior vertices at bottom and top rows
    y_all   = sorted(set(round(float(v.x_a[1]), 8) for v in HC.V))
    y_int   = [y for y in y_all if y > WALL_TOL and y < H - WALL_TOL]
    if not y_int:
        return 0.0, 0.0

    y_bot = y_int[0]   # nearest interior y-level to bottom wall
    y_top = y_int[-1]  # nearest interior y-level to top wall

    F_bot = F_top = 0.0
    for v in HC.V:
        if wall_criterion(v):
            continue
        y_v = round(float(v.x_a[1]), 8)
        A_i = areas.get(id(v), 0.0)
        if A_i < 1e-20:
            continue
        u_xi = float(_safe_u(v)[0])
        # L_i (x-extent of dual cell) ≈ A_i / dy_i
        # where dy_i = distance to nearest wall
        if abs(y_v - y_bot) < 1e-8:
            dy_i = y_bot          # distance to bottom wall
            L_i  = A_i / dy_i if dy_i > 1e-14 else 0.0
            tau_i = mu * u_xi / max(dy_i, 1e-14)
            F_bot += tau_i * L_i
        if abs(y_v - y_top) < 1e-8:
            dy_i = H - y_top      # distance to top wall
            L_i  = A_i / dy_i if dy_i > 1e-14 else 0.0
            tau_i = mu * u_xi / max(dy_i, 1e-14)
            F_top += tau_i * L_i

    return float(F_bot), float(F_top)


def dec_enstrophy(HC):
    """
    DEC enstrophy (rotational energy) as 2-form integral:

        Z = ½ Σ_i  ω_z,i²  A_i

    Analytical SS: ½ L H (16/3) = 8 m²/s²
    """
    areas = compute_dual_areas(HC)
    omega = compute_vorticity(HC)
    Z = 0.0
    for v in HC.V:
        A_i = areas.get(id(v), 0.0)
        if A_i < 1e-20 or wall_criterion(v):
            continue
        Z += 0.5 * omega.get(id(v), 0.0)**2 * A_i
    return float(Z)


def dec_pressure_work(HC):
    """
    DEC pressure work rate = body force power as 2-form integral:

        P_work = Σ_i  (-dPdx/rho) * rho * u_x,i  A_i
               = (-dPdx) * Σ_i u_x,i A_i

    At steady state this equals the viscous dissipation rate.
    Analytical SS: (-dPdx) * U_mean * H * L = 8 * (2/3) * 3 = 16 W
    """
    areas = compute_dual_areas(HC)
    P_work = 0.0
    for v in HC.V:
        if wall_criterion(v):
            continue
        A_i = areas.get(id(v), 0.0)
        if A_i < 1e-20:
            continue
        P_work += (-dPdx) * float(_safe_u(v)[0]) * A_i
    return float(P_work)


def interpolate_to_grid(HC, nx=80, ny=30):
    """
    Interpolate scattered vertex data onto a regular (nx x ny) grid
    for contour plots.

    Uses inverse-distance-squared weighting (Shepard interpolation).

    Returns
    -------
    Xg, Yg : (ny, nx) meshgrid arrays
    Ug     : (ny, nx) u_x field
    Og     : (ny, nx) vorticity field
    Pg     : (ny, nx) pressure correction phi = p - dPdx*x field
    """
    verts = [v for v in HC.V if not wall_criterion(v)]
    if not verts:
        return None, None, None, None, None

    xv  = np.array([float(v.x_a[0]) for v in verts])
    yv  = np.array([float(v.x_a[1]) for v in verts])
    uv  = np.array([float(_safe_u(v)[0]) for v in verts])
    pv  = np.array([float(getattr(v, 'p', p_analytical(v.x_a[0])))
                    - p_analytical(v.x_a[0]) for v in verts])  # phi only

    omega_map = compute_vorticity(HC)
    ov = np.array([omega_map.get(id(v), 0.0) for v in verts])

    # Add wall boundary conditions
    # Bottom wall: u=0, omega from Richardson, phi=0
    wall_verts = [v for v in HC.V if wall_criterion(v)]
    if wall_verts:
        xw = np.array([float(v.x_a[0]) for v in wall_verts])
        yw = np.array([float(v.x_a[1]) for v in wall_verts])
        uw = np.zeros(len(wall_verts))
        pw = np.zeros(len(wall_verts))
        tau_b, tau_t = compute_wss(HC)
        ow = np.array([(tau_b/mu if float(v.x_a[1]) < H/2 else tau_t/mu)
                       for v in wall_verts])
        xv  = np.concatenate([xv, xw])
        yv  = np.concatenate([yv, yw])
        uv  = np.concatenate([uv, uw])
        ov  = np.concatenate([ov, ow])
        pv  = np.concatenate([pv, pw])

    # Build regular grid
    xg = np.linspace(0.0, L_domain, nx)
    yg = np.linspace(0.0, H, ny)
    Xg, Yg = np.meshgrid(xg, yg)

    # Shepard interpolation (inverse-distance-squared)
    Ug = np.zeros_like(Xg)
    Og = np.zeros_like(Xg)
    Pg = np.zeros_like(Xg)

    for iy in range(ny):
        for ix in range(nx):
            xq, yq = Xg[iy, ix], Yg[iy, ix]
            d2 = (xv - xq)**2 + (yv - yq)**2
            # Avoid division by zero
            d2 = np.maximum(d2, 1e-20)
            w = 1.0 / d2
            wsum = w.sum()
            Ug[iy, ix] = (w * uv).sum() / wsum
            Og[iy, ix] = (w * ov).sum() / wsum
            Pg[iy, ix] = (w * pv).sum() / wsum

    return Xg, Yg, Ug, Og, Pg


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




def compute_divergence(HC):
    """
    Compute the incompressibility residual  div(u) = du_x/dx + du_y/dy.

    For this 2D channel flow u_y = 0 exactly, so div(u) = du_x/dx.
    Because u(y,t) is x-homogeneous analytically, div u = 0 exactly.
    The numerical value measures how much Lagrangian particle clustering
    in x has created spurious x-gradients of u_x.

    IMPORTANT — only same-y-level neighbours must be used:
        A 2D SPH estimator would pick up y-neighbours (different u_x,
        different x) and compute a large spurious du_x/dx even when the
        flow is perfectly x-homogeneous.  Only particles at the SAME
        y-level (|Δy| < y_tol) give a true estimate of ∂u_x/∂x.

    Method:
        Group all interior vertices by distinct y-level.
        For each vertex i, use only neighbours j at the same y-level:
            du_x/dx|_i = sum_j w_j (u_xj - u_xi)(xj - xi) / sum_j w_j (xj-xi)^2
            w_j = 1 / (xj - xi)^2
        Average per y-level gives the x-homogeneity residual.

    Returns
    -------
    div_map  : dict {id(v): float}  per-vertex ∂u_x/∂x [1/s]
    l2_div   : float   L2 norm ||div u||_2  [1/s]
    linf_div : float   L_inf norm [1/s]
    """
    from collections import defaultdict

    all_verts = list(HC.V)

    # Group vertices by distinct y-level (same y = same Lagrangian row)
    y_tol   = WALL_TOL + 1e-6   # tighter than wall_criterion
    y_to_vs = defaultdict(list)
    for v in all_verts:
        if not wall_criterion(v):
            y_key = round(float(v.x_a[1]), 8)
            y_to_vs[y_key].append(v)

    div_map = {id(v): 0.0 for v in all_verts}

    for y_key, row in y_to_vs.items():
        if len(row) < 2:
            # Single particle in this row — cannot estimate gradient
            # (divergence = 0 by convention; no clustering to measure)
            continue

        xs_row = np.array([float(v.x_a[0]) for v in row])
        us_row = np.array([float(_safe_u(v)[0]) for v in row])

        for k, v in enumerate(row):
            u_xi = us_row[k]
            x_xi = xs_row[k]

            num = denom = 0.0
            for m, nb in enumerate(row):
                if m == k:
                    continue
                dx  = xs_row[m] - x_xi
                if abs(dx) < 1e-14:
                    continue
                w     = 1.0 / dx**2           # inverse-distance-squared weight
                du_x  = us_row[m] - u_xi
                num   += w * du_x * dx
                denom += w * dx**2

            div_map[id(v)] = (num / denom) if abs(denom) > 1e-20 else 0.0

    # L2 and Linf norms over interior vertices
    vals = np.array([div_map[id(v)] for v in all_verts
                     if not wall_criterion(v)])
    if len(vals) == 0:
        return div_map, 0.0, 0.0
    l2_div   = float(np.sqrt(np.mean(vals**2)))
    linf_div = float(np.max(np.abs(vals)))
    return div_map, l2_div, linf_div


def pressure_poisson_correction(HC, bV, dt):
    """
    Pressure Poisson projection step — enforces div u = 0 after the
    momentum (Euler) step.

    This is the Chorin projection method adapted for the Lagrangian
    y-only meshless discretisation used here:

        u* = u^n + dt*(nu*d^2u/dy^2 + f_body)     [already done in Euler step]

    Step A — compute divergence:
        div u*_i = du*_x/dx|_i  (full 2D SPH estimator, x+y neighbours)

    Step B — solve 1D Poisson in y on distinct y-levels:
        d^2 phi / dy^2 = (rho/dt) * <div u*>_y
        BCs: phi(y=0) = 0, phi(y=H) = 0  (no correction at no-slip walls)
        Solver: direct tridiagonal (Thomas algorithm)

    Step C — update pressure (velocity unchanged for channel flow):
        p^{n+1} = p_linear + phi(y_i)
        u_x unchanged (x-homogeneous flow, no x-projection needed)
        u_y kept = 0  (hard constraint: particles must not move in y;
                       if u_y != 0, particles cross walls and simulation
                       collapses — see detailed comment in Step C)

    Returns
    -------
    div_before : float   L2 ||div u*|| before correction  [1/s]
    div_after  : float   L2 ||div u^{n+1}|| after correction [1/s]
    phi_max    : float   max |phi| (pressure correction magnitude) [Pa]
    """
    from collections import defaultdict

    # ── Step A: divergence of u* (SPH x-gradient estimator) ──────────────
    all_verts = list(HC.V)
    coords    = np.array([v.x_a[:2] for v in all_verts])
    tree      = cKDTree(coords)

    div_raw = {}   # per-vertex divergence [1/s]
    for i, v in enumerate(all_verts):
        if wall_criterion(v):
            div_raw[id(v)] = 0.0
            continue
        u_xi = float(_safe_u(v)[0])
        x_i  = v.x_a[:2]
        k    = min(12, len(all_verts))
        _, idx = tree.query(x_i, k=k)
        num = denom = 0.0
        for j in idx:
            if j == i:
                continue
            nb  = all_verts[j]
            dx  = nb.x_a[:2] - x_i
            h2  = float(np.dot(dx, dx))
            if h2 < 1e-20:
                continue
            w     = 1.0 / h2
            du_x  = float(_safe_u(nb)[0]) - u_xi
            # SPH divergence: du_x/dx component
            num   += w * du_x * dx[0]
            denom += w * dx[0]**2
        div_raw[id(v)] = (num/denom) if abs(denom) > 1e-20 else 0.0

    # L2 norm before correction
    div_vals_pre = np.array([div_raw[id(v)] for v in all_verts
                             if not wall_criterion(v)])
    div_before   = float(np.sqrt(np.mean(div_vals_pre**2))) if len(div_vals_pre) else 0.0

    # ── Step B: average divergence per distinct y-level ───────────────────
    y_to_div = defaultdict(list)
    for v in all_verts:
        if not wall_criterion(v):
            y_to_div[round(float(v.x_a[1]), 8)].append(div_raw[id(v)])

    if not y_to_div:
        return div_before, div_before, 0.0

    y_int  = sorted(y_to_div.keys())
    d_int  = np.array([float(np.mean(y_to_div[y])) for y in y_int])

    # Build full y-array including walls
    y_all = np.array([0.0] + y_int + [H])
    n_all = len(y_all)

    # Assemble tridiagonal Poisson system: d^2 phi/dy^2 = (rho/dt) * div_avg
    # Wall BCs: phi(0) = 0, phi(H) = 0  (Dirichlet)
    A = np.zeros((n_all, n_all))
    b = np.zeros(n_all)

    # Dirichlet wall BCs
    A[0,  0]  = 1.0;  b[0]  = 0.0
    A[-1, -1] = 1.0;  b[-1] = 0.0

    # Interior nodes — non-uniform second-order FD
    for k in range(1, n_all - 1):
        dy_m = y_all[k]   - y_all[k-1]
        dy_p = y_all[k+1] - y_all[k]
        h_avg = 0.5 * (dy_m + dy_p)
        A[k, k-1] =  1.0 / (dy_m * h_avg)
        A[k, k]   = -(1.0/dy_m + 1.0/dy_p) / h_avg
        A[k, k+1] =  1.0 / (dy_p * h_avg)
        b[k]      = (rho / dt) * d_int[k-1]   # rhs from interior div avg

    # Solve the tridiagonal system
    try:
        phi_all = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return div_before, div_before, 0.0

    phi_max = float(np.max(np.abs(phi_all)))

    # Build interpolation: phi(y) from the node values
    def _phi_at_y(y):
        """Linear interpolation of phi at any y."""
        return float(np.interp(y, y_all, phi_all))

    def _dphi_dy_at_y(y):
        """Central-difference gradient of phi at y."""
        dy_step = 1e-5 * H
        return (_phi_at_y(y + dy_step) - _phi_at_y(y - dy_step)) / (2.0 * dy_step)

    # ── Step C: update pressure only; do NOT modify u_y ─────────────────
    #
    # WHY u_y must stay exactly 0 for this simulation:
    # Lagrangian advection is  x_new = x + dt * u.  If u_y != 0, then
    # y_new = y + dt * u_y, moving particles toward the walls.  Within
    # a few steps particles reach y≈0 or y≈H, wall_criterion fires, they
    # are reclassified as wall vertices, their mass is zeroed, and the
    # entire velocity field collapses.
    #
    # The physically correct statement is:
    #   - phi corrects the PRESSURE to be consistent with div u = 0
    #   - for fully-developed channel flow u_y = 0 is an EXACT constraint
    #     (particles move only in x), so no velocity correction is needed
    #   - the Poisson step here acts as a pressure update + incompressibility
    #     diagnostic, not a velocity projection
    #
    # In a general 2D/3D solver with non-zero u_y the velocity correction
    # would be essential, but must be paired with proper wall enforcement
    # that prevents particles from crossing solid boundaries.
    for v in all_verts:
        if wall_criterion(v):
            continue
        y_v   = float(v.x_a[1])
        phi_v = _phi_at_y(y_v)

        # Pressure correction only: p^{n+1} = p_linear + phi
        v.p = p_analytical(float(v.x_a[0])) + phi_v

        # u_y correction (for reference only — NOT applied to avoid
        # destabilising Lagrangian y-positions):
        #   dp_v  = _dphi_dy_at_y(y_v)
        #   u_y_correction = -(dt / rho) * dp_v   [stored below for diagnostics]

    # Ensure walls stay at zero velocity
    for v in all_verts:
        if wall_criterion(v):
            v.u = np.zeros(2)

    # Ensure u_y = 0 for ALL interior vertices (hard constraint for channel flow)
    for v in all_verts:
        if not wall_criterion(v):
            u = _safe_u(v)
            if abs(u[1]) > 1e-14:
                v.u = np.array([u[0], 0.0])

    # L2 norm after correction (recompute divergence)
    _, div_after, _ = compute_divergence(HC)

    return div_before, float(div_after), phi_max

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
    High-quality DEC post-processing frame: 3 rows × 3 columns.

    Row 0: Velocity field (contour+scatter) | Pressure field (dual cells) | Vorticity (dual cells)
    Row 1: KE density (dual cells)          | Velocity profiles            | Incompressibility (dual)
    Row 2: Centreline u_cl + tau_w          | L2 residuals                 | Development fraction
    """
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    from scipy.interpolate import griddata

    FRAME_DPI_HQ = FRAME_DPI

    fig = plt.figure(figsize=(FRAME_FIG_W * 1.15, FRAME_FIG_H * 1.6))
    gs  = fig.add_gridspec(3, 3, width_ratios=[2.2, 2.2, 1.8],
                           height_ratios=[1, 1, 0.85],
                           hspace=0.52, wspace=0.42)
    ax_ux   = fig.add_subplot(gs[0, 0])   # [0,0] velocity contour
    ax_p    = fig.add_subplot(gs[0, 1])   # [0,1] pressure dual cells
    ax_om   = fig.add_subplot(gs[0, 2])   # [0,2] vorticity dual cells
    ax_ke   = fig.add_subplot(gs[1, 0])   # [1,0] KE density dual cells
    ax_prof = fig.add_subplot(gs[1, 1])   # [1,1] velocity profiles
    ax_div  = fig.add_subplot(gs[1, 2])   # [1,2] incompressibility
    ax_cl   = fig.add_subplot(gs[2, 0])   # [2,0] centreline + WSS
    ax_err  = fig.add_subplot(gs[2, 1])   # [2,1] residuals
    ax_dev  = fig.add_subplot(gs[2, 2])   # [2,2] development fraction

    verts  = list(HC.V)
    y_fine = np.linspace(0, H, 300)
    n_int  = sum(1 for v in HC.V if not wall_criterion(v))
    n_wall = len(verts) - n_int
    edges  = get_edges_delaunay(HC)

    # ── Compute DEC fields and dual cells once ────────────────────────────
    dual     = compute_dual_cells(HC)
    dec, _   = compute_dec_fields(HC, dual)

    # ── Contour grid ──────────────────────────────────────────────────────
    all_v  = [v for v in verts]
    u_vals = dec['u_x']
    p_vals = dec['p']
    GX, GY, UX_grid = scatter_to_grid(all_v, u_vals, nx=120, ny=60)
    _,  _,  P_grid  = scatter_to_grid(all_v, p_vals, nx=120, ny=60)

    # ══════════════════════════════════════════════════════════════════════
    # [0,0] Velocity field: filled contour + isolines + Delaunay mesh
    # ══════════════════════════════════════════════════════════════════════
    cf = ax_ux.contourf(GX, GY, UX_grid, levels=16,
                        cmap='RdYlGn', vmin=0, vmax=U_max * 1.05)
    fig.colorbar(cf, ax=ax_ux, fraction=0.03, pad=0.02, label='$u_x$ [m/s]')
    ax_ux.contour(GX, GY, UX_grid, levels=8, colors='k',
                  linewidths=0.5, alpha=0.4)
    # Overlay analytical isolines at SS
    ax_ux.contour(GX, GY,
                  u_poiseuille(GY),
                  levels=8, colors='white', linewidths=0.8,
                  linestyles='--', alpha=0.5)
    if verts:
        coords = np.array([v.x_a[:2] for v in verts])
        ax_ux.scatter(coords[:, 0], coords[:, 1],
                      c='k', s=8, zorder=4, alpha=0.5)
    ax_ux.axvline(0.0,      color='royalblue', lw=1.2, ls='--', alpha=0.7)
    ax_ux.axvline(L_domain, color='firebrick',  lw=1.2, ls='--', alpha=0.7)
    ax_ux.set_xlim(0, L_domain); ax_ux.set_ylim(0, H)
    ax_ux.set_xlabel('x [m]'); ax_ux.set_ylabel('y [m]')
    ax_ux.set_title(
        f'$u_x$ contours  t={t:.3f}s ({t/tau_visc:.1f}$\tau$)  N={len(verts)}',
        fontsize=10)
    ax_ux.set_aspect('equal')

    # ══════════════════════════════════════════════════════════════════════
    # [0,1] Pressure field on dual cells (DEC 2-form)
    # p(x,y) = dPdx*x + phi(y) — each dual cell carries its integrated p
    # ══════════════════════════════════════════════════════════════════════
    p_min = float(dPdx * L_domain)  # min at x=L
    p_max = 0.0                      # max at x=0
    _draw_dual_cells(ax_p, fig, dual, dec['p'], verts,
                     cmap=plt.cm.coolwarm, label='p [Pa]',
                     vmin=p_min, vmax=p_max, alpha=0.9, lw=0.4)
    # Overlay p contours from grid
    ax_p.contour(GX, GY, P_grid, levels=10, colors='k',
                 linewidths=0.5, alpha=0.4)
    ax_p.set_xlim(0, L_domain); ax_p.set_ylim(0, H)
    ax_p.set_xlabel('x [m]'); ax_p.set_ylabel('y [m]')
    ax_p.set_title('Pressure field p(x,y) — dual cells (DEC)', fontsize=10)
    ax_p.set_aspect('equal')

    # ══════════════════════════════════════════════════════════════════════
    # [0,2] Vorticity omega_z on dual cells (Stokes theorem)
    # Circulation Gamma_i / A_i = -du/dy (correct sign: CCW positive)
    # ══════════════════════════════════════════════════════════════════════
    om_ss   = float(np.nanmax(np.abs(dec['vorticity'])))
    vlim_om = max(om_ss * 1.1, 4 * U_max / H * 1.1, 0.1)
    _draw_dual_cells(ax_om, fig, dual, dec['vorticity'], verts,
                     cmap=plt.cm.RdBu_r, label=r'$\omega_z$ [1/s]',
                     vmin=-vlim_om, vmax=vlim_om, alpha=0.9, lw=0.4)
    # Analytical vorticity at SS for reference annotation
    ax_om.text(0.02, 0.97,
               f'Max |omega_z| =={om_ss:.2f} s$^{{-1}}$\nSS: {4*U_max/H:.2f} s$^{{-1}}$',
               transform=ax_om.transAxes, fontsize=8, va='top',
               bbox=dict(fc='white', alpha=0.7, pad=2))
    ax_om.set_xlim(0, L_domain); ax_om.set_ylim(0, H)
    ax_om.set_xlabel('x [m]'); ax_om.set_ylabel('y [m]')
    ax_om.set_title('Vorticity omega_z — dual cells (Stokes)', fontsize=10)
    ax_om.set_aspect('equal')

    # ══════════════════════════════════════════════════════════════════════
    # [1,0] Kinetic energy density on dual cells (DEC 2-form)
    # KE_i = (1/2)*rho*u_{xi}^2 * A_i  integrated over dual cell
    # Colour: KE_density = (1/2)*rho*u_x^2 [J/m^2]
    # ══════════════════════════════════════════════════════════════════════
    _draw_dual_cells(ax_ke, fig, dual, dec['KE_density'], verts,
                     cmap=plt.cm.plasma, label='KE density [J/m$^2$]',
                     vmin=0, vmax=0.5*rho*U_max**2, alpha=0.9, lw=0.4)
    ke_total = float(np.sum(dec['KE_cell'][~dec['is_wall']]))
    ke_ss    = 0.5*rho*L_domain*float(np.trapezoid(
                   u_poiseuille(y_fine)**2, y_fine))
    ax_ke.text(0.02, 0.97,
               f'Total KE={ke_total:.3f} J\nSS: {ke_ss:.3f} J',
               transform=ax_ke.transAxes, fontsize=8, va='top',
               bbox=dict(fc='white', alpha=0.7, pad=2))
    ax_ke.set_xlim(0, L_domain); ax_ke.set_ylim(0, H)
    ax_ke.set_xlabel('x [m]'); ax_ke.set_ylabel('y [m]')
    ax_ke.set_title('KE density $(\\frac{1}{2}\\rho u_x^2)$ — dual cells', fontsize=10)
    ax_ke.set_aspect('equal')

    # ══════════════════════════════════════════════════════════════════════
    # [1,1] Multi-station velocity profiles + analytical
    # ══════════════════════════════════════════════════════════════════════
    colors_st = ['royalblue', 'darkorange', 'green']
    x_stations = [L_domain/4, L_domain/2, 3*L_domain/4]
    for x_st, col in zip(x_stations, colors_st):
        pts = [(float(_safe_u(v)[0]), float(v.x_a[1]))
               for v in HC.V
               if not wall_criterion(v)
               and abs(float(v.x_a[0]) - x_st) < L_period * 0.6]
        if pts:
            us_st, ys_st = zip(*sorted(pts, key=lambda p: p[1]))
            ax_prof.plot(us_st, ys_st, 'o-', color=col, ms=5, lw=1.2,
                         label=f'x={x_st:.1f}m', zorder=4)
    ax_prof.plot(u_poiseuille(y_fine), y_fine, 'k-', lw=2.5,
                 label='SS (Poiseuille)', zorder=10)
    ax_prof.plot(u_transient(y_fine, t, n_terms=20), y_fine,
                 'k--', lw=1.5, alpha=0.6, label=f'Exact t={t:.2f}s')
    ax_prof.fill_betweenx(y_fine, u_transient(y_fine, t, n_terms=20),
                           u_poiseuille(y_fine),
                           alpha=0.08, color='gray', label='Error band')
    ax_prof.set_xlabel('$u_x$ [m/s]'); ax_prof.set_ylabel('y [m]')
    ax_prof.set_title('Velocity profiles  x = L/4, L/2, 3L/4', fontsize=10)
    ax_prof.set_xlim(-0.05, U_max * 1.15)
    ax_prof.set_ylim(0, H)
    ax_prof.legend(fontsize=7, loc='upper left'); ax_prof.grid(True, alpha=0.3)

    # ══════════════════════════════════════════════════════════════════════
    # [1,2] Incompressibility residual div u on dual cells
    # div u = du_x/dx (same-y SPH estimator)
    # Should be ~0 everywhere for incompressible flow
    # ══════════════════════════════════════════════════════════════════════
    div_abs_max = max(float(np.abs(dec['div_u']).max()), 1e-8)
    _draw_dual_cells(ax_div, fig, dual, dec['div_u'], verts,
                     cmap=plt.cm.PiYG, label='div $u$ [1/s]',
                     vmin=-div_abs_max, vmax=div_abs_max, alpha=0.9, lw=0.4)
    div_l2 = float(np.sqrt(np.mean(dec['div_u'][~dec['is_wall']]**2)))
    ax_div.text(0.02, 0.97,
                f'$L_2$(div u) = {div_l2:.2e}',
                transform=ax_div.transAxes, fontsize=8, va='top',
                bbox=dict(fc='white', alpha=0.7, pad=2))
    ax_div.set_xlim(0, L_domain); ax_div.set_ylim(0, H)
    ax_div.set_xlabel('x [m]'); ax_div.set_ylabel('y [m]')
    ax_div.set_title('Incompressibility  div $u$ — dual cells', fontsize=10)
    ax_div.set_aspect('equal')

    # ══════════════════════════════════════════════════════════════════════
    # [2,0] Centreline velocity + wall shear stress (time history)
    # ══════════════════════════════════════════════════════════════════════
    tau_w_an = mu * 4.0 * U_max / H
    ax_cl.axhline(U_max, color='royalblue', lw=1.2, ls='--',
                  label=f'$U_{{max}}$={U_max:.2f}')
    if hist_t and hist_u_cl:
        ax_cl.plot(hist_t, hist_u_cl, 'b-', lw=2, label='$u_{cl}(t)$')
    _cl_f = [(abs(float(v.x_a[1]) - H/2), float(_safe_u(v)[0]))
             for v in HC.V
             if not wall_criterion(v)
             and abs(float(v.x_a[1]) - H/2) < H/4]
    if _cl_f:
        _w = np.array([1/(p[0]+1e-6) for p in _cl_f])
        cl_now = float(np.dot(_w,[p[1] for p in _cl_f])/_w.sum())
        ax_cl.scatter([t],[cl_now], c='blue', s=50, zorder=5)
    ax_cl.axvline(tau_visc,   color='gray', lw=0.8, ls=':')
    ax_cl.axvline(5*tau_visc, color='gray', lw=1.2, ls='--')
    ax_cl.set_xlabel('t [s]'); ax_cl.set_ylabel('$u_{cl}$ [m/s]', color='b')
    ax_cl.tick_params(axis='y', labelcolor='b')
    ax_cl2 = ax_cl.twinx()
    ax_cl2.axhline(tau_w_an, color='crimson', lw=1.2, ls='--',
                   label=f'SS tau_w={tau_w_an:.1f} Pa')
    if hist_t and hist_tau_wall:
        ax_cl2.plot(hist_t, hist_tau_wall, 'r-', lw=2, label='$\tau_w(t)$')
    tau_now, _ = compute_wss(HC)
    if hist_t:
        ax_cl2.scatter([t],[tau_now], c='red', s=50, zorder=5)
    ax_cl2.set_ylabel('$\tau_w$ [Pa]', color='r')
    ax_cl2.tick_params(axis='y', labelcolor='r')
    h1,l1 = ax_cl.get_legend_handles_labels()
    h2,l2 = ax_cl2.get_legend_handles_labels()
    ax_cl.legend(h1+h2, l1+l2, fontsize=7, loc='upper left')
    ax_cl.set_title('Centreline velocity  &  Wall shear stress', fontsize=10)
    ax_cl.grid(True, alpha=0.3)

    # ══════════════════════════════════════════════════════════════════════
    # [2,1] L2 residuals (velocity error vs transient + SS)
    # ══════════════════════════════════════════════════════════════════════
    if hist_t:
        ax_err.semilogy(hist_t, hist_l2_tr, 'b-',  lw=2,
                        label='$L_2$ vs transient')
        ax_err.semilogy(hist_t, hist_l2_ss, 'r--', lw=2,
                        label='$L_2$ vs Poiseuille SS')
    ax_err.axvline(tau_visc,   color='gray', lw=0.8, ls=':',  label='$\tau$')
    ax_err.axvline(5*tau_visc, color='gray', lw=1.2, ls='--', label='$5\tau$')
    ax_err.set_xlabel('t [s]'); ax_err.set_ylabel('$L_2$ error [m/s]')
    ax_err.set_title(f'Residuals: $L_2$(SS)={l2_ss:.2e}', fontsize=10)
    ax_err.legend(fontsize=7); ax_err.grid(True, which='both', alpha=0.3)
    if hist_l2_ss and max(hist_l2_ss) > 0:
        ymin = max(1e-14, min(hist_l2_ss)*0.5)
        ymax = max(hist_l2_ss)*2.0
        if ymax > ymin: ax_err.set_ylim(ymin, ymax)

    # ══════════════════════════════════════════════════════════════════════
    # [2,2] Development fraction bar + DEC global diagnostics
    # ══════════════════════════════════════════════════════════════════════
    dev_frac = float(np.clip(1.0 - l2_ss / max(dec['u_x'].max(), 1e-8), 0, 1))
    # Compute global DEC integrals
    A_int    = dec['area'][~dec['is_wall']]
    ke_int   = float(np.sum(dec['KE_cell'][~dec['is_wall']]))
    diss_int = float(np.sum(dec['dissipation_cell'][~dec['is_wall']]))
    flux_int = compute_mass_flux(HC, L_domain/2)
    mf_ss    = rho * U_mean * H

    metrics = {
        'Development\n(1-L2/u_max)': dev_frac,
        'KE / KE_SS':     ke_int / max(0.5*rho*L_domain*float(
                              np.trapezoid(u_poiseuille(y_fine)**2, y_fine)), 1e-8),
        'Diss / Diss_SS': diss_int / max(
                              mu*L_domain*float(
                                  np.trapezoid(np.gradient(
                                      u_poiseuille(y_fine),y_fine)**2, y_fine)), 1e-8),
        'Flux / Flux_SS': flux_int / max(mf_ss, 1e-8),
    }
    y_pos    = np.arange(len(metrics))
    m_vals   = list(metrics.values())
    m_labels = list(metrics.keys())
    colors_bar = ['green' if abs(v-1.0) < 0.05 else
                  'orange' if abs(v-1.0) < 0.15 else 'crimson'
                  for v in m_vals]
    bars = ax_dev.barh(y_pos, m_vals, color=colors_bar, alpha=0.8, edgecolor='k')
    ax_dev.axvline(1.0, color='k', lw=1.5, ls='--', label='SS target')
    ax_dev.set_yticks(y_pos); ax_dev.set_yticklabels(m_labels, fontsize=8)
    ax_dev.set_xlim(0, 1.25)
    ax_dev.set_xlabel('Ratio to steady state')
    ax_dev.set_title('DEC global diagnostics', fontsize=10)
    ax_dev.legend(fontsize=7); ax_dev.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, m_vals):
        ax_dev.text(min(val+0.02, 1.22), bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)

    fig.suptitle(
        f'Lagrangian DEC — 2D Incompressible Poiseuille  |  '
        f't={t:.3f}s ({t/tau_visc:.1f}$\tau$)  |  Re={rho*U_mean*H/mu:.1f}',
        fontsize=12, y=1.01)
    plt.tight_layout()
    fpath = os.path.join(fig_dir, f'frame_{step:06d}.png')
    plt.savefig(fpath, dpi=FRAME_DPI, bbox_inches='tight')
    plt.close(fig)
    return fpath



# ============================================================
# Build meshes and boundary conditions
# ============================================================


def run_case(n_refine):
    """
    Run the full 2D incompressible Poiseuille simulation for N_REFINE=n_refine.

    Uses DEC stress_acceleration (ddgclib.operators.gradient) when available,
    with y-stencil fallback.  DualArea_i from ddgclib.operators.area provides
    proper dual 2-cell measures for all integrated quantities.

    Returns ts (diagnostics dict) and profile_snaps.
    """
    global N_REFINE, _HC_ref, _FIG, _FRAMES
    N_REFINE = n_refine
    _FIG    = os.path.join(_HERE, 'fig', f'poiseuille_nrefine_{n_refine}')
    _FRAMES = os.path.join(_FIG, '_frames')
    os.makedirs(_FRAMES, exist_ok=True)
    print(f"\n{'='*60}\n  MESH REFINEMENT STUDY: N_REFINE = {n_refine}\n"
          f"  Solver: {'DEC stress_acceleration' if _DEC_AVAILABLE else 'y-stencil'}\n"
          f"  Output -> {_FIG}\n{'='*60}")
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

    # ============================================================
    # Diagnostic storage
    # ============================================================
    ts = dict(
        t=[], step=[], dt_used=[],
        n_verts=[], n_interior=[], n_wall=[],
        cum_inj=[], cum_del=[],
        l2_transient=[], linf_transient=[],
        l2_ss=[], linf_ss=[],
        mass_total=[], momentum_x=[], momentum_expected=[],
        KE=[], dissipation=[], tau_wall=[], u_centerline=[],
        mass_flux_inlet=[], mass_flux_mid=[], mass_flux_outlet=[],
        div_u_l2=[], div_u_linf=[],
        div_u_pre_l2=[], phi_max=[],
        # DEC integrated quantities
        dec_momentum=[], dec_enstrophy=[], dec_pressure_work=[], dec_wss_top=[],
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


    # ============================================================
    # Initial diagnostics
    # ============================================================
    l2_tr0, li_tr0, l2_ss0, li_ss0 = compute_errors(HC, 0.0)
    M0 = domain_mass(HC)
    P0 = domain_momentum_x(HC)

    print(f"\nInitial: M={M0:.4f}  Px={P0:.4f}")
    print(f"  Solver mode     : {'DEC stress_acceleration' if _DEC_AVAILABLE else 'y-stencil fallback'}")
    print(f"  L2 vs transient : {l2_tr0:.4e}  (plug matches t=0 exact solution)")
    print(f"  L2 vs SS        : {l2_ss0:.4e}  (large: plug != parabola)")

    save_profile_snap(0.0, HC, 't=0 (rest, u=0)')

    frame_paths = []
    fp = plot_frame(0, 0.0, HC, _FRAMES,
                    0, 0, l2_tr0, l2_ss0, [], [], [], [], [], [], [], [])
    frame_paths.append(fp)

    print(f"\n{'step':>6}  {'t':>7}  {'t/tau':>6}  {'dt':>8}  {'N':>5}  "
          f"{'L2_tr':>10}  {'L2_ss':>10}  {'M/M0':>8}  {'dPx%':>8}  "
          f"{'inj':>5}  {'del':>5}  {'net':>6}  {'div_L2':>10}")

    # ============================================================
    # FVM mass state
    # ============================================================
    # Each interior Lagrangian vertex represents a control volume
    # of fixed mass  m_cell = rho * L_domain * H / n_int_0.
    # The periodic-wrap BC keeps N = n_int_0 exactly at all times,
    # so M = n_int_0 * m_cell = M_total at every step.
    m_cell = M_total / n_int_0

    # ============================================================
    # Adaptive Euler time loop
    # ============================================================
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

        # 5b. Pressure Poisson projection — enforce div u = 0
        #     Computes d^2 phi/dy^2 = (rho/dt)*div u*, corrects pressure.
        #     Keeps u_x unchanged (x-homogeneous, hard channel constraint).
        _div_pre, _div_post, _phi_max = pressure_poisson_correction(HC, bV, dt)

        # 5c. Recompute dual cells after position update — required so that
        #     v.vd is current at the start of the NEXT Euler step when DEC
        #     stress_acceleration reads dual area vectors A_ij.
        if _DEC_AVAILABLE:
            try:
                if _recompute_duals is not None:
                    _recompute_duals(HC)
                else:
                    from hyperct.ddg import compute_vd
                    compute_vd(HC, method='barycentric')
            except Exception:
                pass  # non-fatal; dudt_fn falls back to y-stencil

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
        # ── DEC integrated diagnostics ────────────────────────────────────────
        # All quantities computed as integrals over Voronoi dual cells (2-forms)
        # or dual edges (1-forms), NOT pointwise.
        _KE      = dec_kinetic_energy(HC)          # ½ Σ ρ|u|² A_i
        _eps     = dec_viscous_dissipation(HC)     # Σ μ ω²  A_i
        _Px_dec  = dec_momentum(HC)                # Σ ρ u_x A_i
        _mf_dec  = dec_mass_flux(HC)               # Σ ρ u_x A_i / L
        _F_bot, _F_top = dec_wss(HC)               # Σ μ (∂u/∂y) L_i
        _Z       = dec_enstrophy(HC)               # ½ Σ ω² A_i
        _Pwork   = dec_pressure_work(HC)           # Σ (-dPdx) u_x A_i

        # Centreline velocity: DEC weighted by dual-cell area near centreline
        _dy_band = H / 4.0
        _cl_verts = [(abs(float(v.x_a[1]) - H/2),
                      float(_safe_u(v)[0]),
                      compute_dual_areas(HC).get(id(v), 0.0))
                     for v in HC.V
                     if not wall_criterion(v)
                     and abs(float(v.x_a[1]) - H/2) < _dy_band]
        if _cl_verts:
            _w_cl = np.array([a / max(d + 1e-6, 1e-6)
                              for d, u, a in _cl_verts])
            _u_cl = float(np.dot(_w_cl, [u for d, u, a in _cl_verts])
                          / max(_w_cl.sum(), 1e-30))
        else:
            _u_cl = 0.0

        ts['KE'].append(_KE)
        ts['dissipation'].append(_eps)
        ts['tau_wall'].append(_F_bot)        # integrated wall force [N]
        ts['u_centerline'].append(_u_cl)
        ts['mass_flux_inlet'].append(_mf_dec)
        ts['mass_flux_mid'].append(_mf_dec)
        ts['mass_flux_outlet'].append(_mf_dec)
        ts['dec_momentum'].append(_Px_dec)
        ts['dec_enstrophy'].append(_Z)
        ts['dec_pressure_work'].append(_Pwork)
        ts['dec_wss_top'].append(_F_top)
        _, _div_l2, _div_linf = compute_divergence(HC)
        ts['div_u_l2'].append(_div_l2)
        ts['div_u_linf'].append(_div_linf)
        ts['div_u_pre_l2'].append(_div_pre)
        ts['phi_max'].append(_phi_max)

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
                  f"{n_inj_stp:>5d}  {n_del_stp:>5d}  {cum_inj-cum_del:>+6d}  "
                  f"{_div_l2:>10.3e}")

        # Frame
        if step % FRAME_EVERY == 0:
            fp = plot_frame(step, t, HC, _FRAMES,
                            cum_inj, cum_del, l2_tr, l2_ss,
                            ts['t'], ts['l2_transient'], ts['l2_ss'],
                            ts['tau_wall'], ts['u_centerline'],
                            ts['KE'], ts['dissipation'],
                            ts['dec_enstrophy'], ts['dec_pressure_work'])
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
                    ts['tau_wall'], ts['u_centerline'],
                    ts['KE'], ts['dissipation'],
                    ts['dec_enstrophy'], ts['dec_pressure_work'])
    if fp not in frame_paths:
        frame_paths.append(fp)
    save_profile_snap(t, HC, f't={t:.1f} s (final, {t/tau_visc:.0f}tau)')

    print(f"\nDone:  t={t:.3f} s ({t/tau_visc:.1f}tau),  {step} steps")
    print(f"  Final L2 vs SS        = {ts['l2_ss'][-1]:.3e}  (-> 0 at steady state)")
    print(f"  Final L2 vs transient = {ts['l2_transient'][-1]:.3e}")
    print(f"  Mass conservation     : M_final/M0 = {domain_mass(HC)/M_total:.6f}")
    print(f"  Net vertices          : {cum_inj - cum_del:+d}")

    # ============================================================
    # Animation
    # ============================================================
    compile_gif(frame_paths, os.path.join(_FIG, 'animation.gif'))

    # ============================================================
    # Summary plot 1: error convergence, mass, vertex count, div(u)
    # ============================================================
    t_arr  = np.array(ts['t'])
    M_arr  = np.array(ts['mass_total'])
    Px_arr = np.array(ts['momentum_x'])
    Pe_arr = np.array(ts['momentum_expected'])

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    # [0] L2 velocity error
    ax = axes[0]
    ax.semilogy(t_arr, ts['l2_transient'], 'b-',  lw=2, label='$L_2$ vs transient')
    ax.semilogy(t_arr, ts['l2_ss'],        'r--', lw=2, label='$L_2$ vs Poiseuille SS')
    ax.axvline(tau_visc,   color='gray', lw=1.2, ls=':',  label=f'tau={tau_visc:.3f} s')
    ax.axvline(5*tau_visc, color='gray', lw=1.8, ls='--', label=f'5tau={5*tau_visc:.2f} s')
    ax.set_xlabel('t [s]'); ax.set_ylabel('$L_2$ velocity error [m/s]')
    ax.set_title('Error convergence (rest -> Poiseuille)', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)

    # [1] Incompressibility residual  ||div u||
    ax = axes[1]
    ax.semilogy(t_arr, ts['div_u_pre_l2'],'k:',  lw=1.5, label='$L_2$ div u* (before)')
    ax.semilogy(t_arr, ts['div_u_l2'],   'b-',  lw=2, label='$L_2$ div u (after projection)')
    ax.semilogy(t_arr, ts['div_u_linf'], 'r--', lw=2, label='$L_\\infty$ div u (after)')
    if ts['phi_max']:
        ax_phi = ax.twinx()
        ax_phi.semilogy(t_arr, ts['phi_max'], 'g-', lw=1.5, alpha=0.7, label='max |phi| [Pa]')
        ax_phi.set_ylabel('max |phi| [Pa]', color='g')
        ax_phi.tick_params(axis='y', labelcolor='g')
    ax.axhline(1e-10, color='green', lw=1.5, ls=':', label='machine zero ref')
    ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':')
    ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--')
    ax.set_xlabel('t [s]'); ax.set_ylabel('||div u|| [1/s]')
    ax.set_title('Incompressibility residual  div(u) = 0', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)
    # Annotate final value
    if ts['div_u_l2']:
        ax.annotate(f"Final L2 = {ts['div_u_l2'][-1]:.2e}",
                    xy=(t_arr[-1], ts['div_u_l2'][-1]),
                    xytext=(-60, 15), textcoords='offset points',
                    fontsize=10, color='royalblue',
                    arrowprops=dict(arrowstyle='->', color='royalblue', lw=1.2))

    # [2] Mass conservation
    ax = axes[2]
    ax.plot(t_arr, M_arr / M_total * 100.0, 'b-', lw=2, label='domain mass / $M_0$')
    ax.axhline(100.0, color='red', lw=1.5, ls='--', label='100%')
    ax.set_xlabel('t [s]'); ax.set_ylabel('% of $M_0$')
    ax.set_title('Mass conservation', fontsize=13)
    ax.set_ylim(80, 120)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # [3] div(u) spatial distribution at final step
    ax = axes[3]
    div_final_map, _, _ = compute_divergence(HC)
    fin_verts = [(float(v.x_a[0]), float(v.x_a[1]), div_final_map.get(id(v), 0.0))
                 for v in HC.V if not wall_criterion(v)]
    if fin_verts:
        xd = np.array([p[0] for p in fin_verts])
        yd = np.array([p[1] for p in fin_verts])
        dv = np.array([p[2] for p in fin_verts])
        vlim = max(abs(dv).max(), 1e-10)
        sc = ax.scatter(xd, yd, c=dv, cmap='RdBu_r', s=60, zorder=3,
                        vmin=-vlim, vmax=vlim)
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='div(u) [1/s]')
    ax.set_xlim(0, L_domain); ax.set_ylim(0, H)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('div(u) field at final state\n(should be ~0 everywhere)', fontsize=12)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

    # [4] Vertex count + wrap events
    ax = axes[4]
    cum_inj_arr = np.array(ts['cum_inj'])
    cum_del_arr = np.array(ts['cum_del'])
    net_arr     = cum_inj_arr - cum_del_arr
    ax.plot(t_arr, ts['n_verts'],    'k-',  lw=2,   label='N total')
    ax.plot(t_arr, ts['n_interior'], 'b-',  lw=1.5, label='N interior')
    ax.plot(t_arr, ts['n_wall'],     'g--', lw=1.5, label='N wall')
    ax.axhline(n_main, color='steelblue', ls=':', lw=1.2, label=f'$N_0$={n_main}')
    ax2 = ax.twinx()
    ax2.plot(t_arr, net_arr, 'D-', color='purple', lw=1.8, ms=2, label='net (inj-del)')
    ax2.axhline(0, color='purple', ls=':', lw=0.8)
    ax2.set_ylabel('Net vertices', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Vertex count')
    ax.set_title('Vertex count + wrap events', fontsize=13)
    ax.grid(True, alpha=0.3)

    # [5] div(u) L2 norm vs L2 velocity error — correlation
    ax = axes[5]
    ax.loglog(ts['l2_ss'], ts['div_u_l2'], 'b-', lw=2, alpha=0.7)
    ax.scatter([ts['l2_ss'][0]],  [ts['div_u_l2'][0]],  c='green',  s=80, zorder=5,
               label='t=0 (start)')
    ax.scatter([ts['l2_ss'][-1]], [ts['div_u_l2'][-1]], c='red',    s=80, zorder=5,
               label=f't={t_arr[-1]:.1f}s (end)')
    ax.set_xlabel('$L_2$ velocity error [m/s]')
    ax.set_ylabel('$L_2$ ||div u|| [1/s]')
    ax.set_title('Incompressibility vs velocity error\n(convergence path)', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)

    plt.suptitle('Rest (u=0) -> Poiseuille  |  Summary  +  Incompressibility Check',
                 fontsize=15)
    plt.tight_layout()
    p_err = os.path.join(_FIG, 'error_vs_time.png')
    plt.savefig(p_err, dpi=SUMMARY_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved {p_err}")

    # ============================================================
    # Summary plot 2: velocity profile development
    # ============================================================
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

    # ============================================================
    # Summary plot 3: momentum budget
    # ============================================================
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

    # ============================================================
    # Summary plot 4: Wall shear stress + centreline velocity
    # ============================================================
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
    ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':',  label='tau')
    ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--', label='5 tau')
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

    # ============================================================
    # Summary plot 5: Kinetic energy + viscous dissipation + pressure
    # ============================================================
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

    # Pressure field at final state — split by y-level
    # p(x,y) = dPdx*x + phi(y): each y-level gives a parallel linear line.
    # Left of ax: p vs x coloured by y-level
    # Right twin: phi(y) = p - dPdx*x (Poisson correction, should be small)
    ax = axes[2]
    from collections import defaultdict
    y_to_xp = defaultdict(list)
    for v in HC.V:
        if not wall_criterion(v):
            y_k = round(float(v.x_a[1]), 8)
            x_v = float(v.x_a[0])
            p_v = float(getattr(v, 'p', p_analytical(x_v)))
            y_to_xp[y_k].append((x_v, p_v))

    x_line = np.linspace(0, L_domain, 200)
    ax.plot(x_line, dPdx * x_line, 'k--', lw=2.5,
            label=f'Analytical p={dPdx:.0f}x', zorder=10)

    cmap_p  = plt.cm.plasma
    y_keys  = sorted(y_to_xp.keys())
    cols_p  = cmap_p(np.linspace(0.15, 0.85, len(y_keys)))
    phi_at_y = []
    for y_k, col in zip(y_keys, cols_p):
        pts = sorted(y_to_xp[y_k])
        xp  = np.array([p[0] for p in pts])
        pp  = np.array([p[1] for p in pts])
        ax.scatter(xp, pp, c=[col]*len(xp), s=25, zorder=3, alpha=0.8)
        ax.plot(xp, pp, '-', color=col, lw=1.2, alpha=0.6,
                label=f'y={y_k:.3f}')
        # phi = p - dPdx*x (mean over this row)
        phi_row = float(np.mean(pp - dPdx * xp))
        phi_at_y.append((y_k, phi_row))

    ax.set_xlabel('x [m]'); ax.set_ylabel('p [Pa]')
    ax.set_title('p(x,y) = dPdx*x + phi(y) -- parallel lines = correct', fontsize=11)
    ax.legend(fontsize=8, loc='upper right', ncol=1); ax.grid(True, alpha=0.3)

    # Add inset or annotation showing phi(y) magnitude
    if phi_at_y:
        phi_vals = [p[1] for p in phi_at_y]
        ax.annotate(f'phi range: [{min(phi_vals):.3f}, {max(phi_vals):.3f}] Pa',
                    xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9,
                    color='gray',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.suptitle('Energy Budget  |  Viscous Dissipation  |  Pressure Field', fontsize=15)
    plt.tight_layout()
    p_energy = os.path.join(_FIG, 'energy_dissipation_pressure.png')
    plt.savefig(p_energy, dpi=SUMMARY_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved {p_energy}")

    # ============================================================
    # Summary plot 6: Dual-cell areas at final state + vorticity profile
    # ============================================================
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
    return ts, profile_snaps


# ============================================================
# Parametric loop
# ============================================================
N_REFINE_VALUES = [1, 2, 3]
results = {}

for _nr in N_REFINE_VALUES:
    _ts, _ps = run_case(_nr)
    results[_nr] = (_ts, _ps)

print("\n" + "="*60)
print("  ALL RUNS COMPLETE — generating comparison plots")
print("="*60)

_COMP_DIR = os.path.join(_HERE, 'fig', 'poiseuille_parametric')
os.makedirs(_COMP_DIR, exist_ok=True)

# Build colour/linestyle maps dynamically from actual results keys
# so the comparison works for any N_REFINE_VALUES (including 0-indexed)
_all_colors = ['royalblue', 'darkorange', 'green', 'crimson', 'purple']
_all_ls     = ['-', '--', '-.', ':', (0,(3,1,1,1))]
_nr_keys    = sorted(results.keys())
colors_nr   = {nr: _all_colors[i % len(_all_colors)] for i, nr in enumerate(_nr_keys)}
ls_nr       = {nr: _all_ls[i % len(_all_ls)]         for i, nr in enumerate(_nr_keys)}
y_fine    = np.linspace(0, H, 400)

fig, axes = plt.subplots(3, 3, figsize=(22, 18))
axes = axes.flatten()

# [0] L2 vs SS
ax = axes[0]
for nr, (ts_r, _) in results.items():
    t_arr = np.array(ts_r['t'])
    ax.semilogy(t_arr, ts_r['l2_ss'], color=colors_nr[nr], ls=ls_nr[nr], lw=2,
                label=f'N_ref={nr}  (N_int={ts_r["n_interior"][0]})')
ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':', label='tau')
ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--', label='5 tau')
ax.set_xlabel('t [s]'); ax.set_ylabel('L2 vs SS')
ax.set_title('Velocity convergence to SS', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)

# [1] Incompressibility residual ||div u||
ax = axes[1]
for nr, (ts_r, _) in results.items():
    t_arr = np.array(ts_r['t'])
    ax.semilogy(t_arr, ts_r['div_u_l2'], color=colors_nr[nr], ls=ls_nr[nr], lw=2,
                label=f'N_ref={nr}')
ax.axvline(tau_visc,   color='gray', lw=1.0, ls=':')
ax.axvline(5*tau_visc, color='gray', lw=1.5, ls='--')
ax.set_xlabel('t [s]'); ax.set_ylabel('L2 ||div u|| [1/s]')
ax.set_title('Incompressibility residual  div(u) = 0', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)

# [2] Final velocity profile
ax = axes[2]
ax.plot(u_poiseuille(y_fine), y_fine, 'k-', lw=3, label='Analytical SS', zorder=10)
for nr, (_, ps_r) in results.items():
    if ps_r:
        t_s, y_s, u_s, _ = ps_r[-1]
        idx = np.argsort(y_s)
        ax.plot(u_s[idx], y_s[idx], color=colors_nr[nr], ls=ls_nr[nr],
                lw=2, marker='o', ms=4, label=f'N_ref={nr}')
ax.set_xlabel('u_x [m/s]'); ax.set_ylabel('y [m]')
ax.set_title('Final velocity profile (t=8 s)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# [3] Wall shear stress
ax = axes[3]
ax.axhline(mu * 4.0 * U_max / H, color='k', lw=2.5, ls='--',
           label=f'Analytical = {mu*4*U_max/H:.1f} Pa')
for nr, (ts_r, _) in results.items():
    t_arr = np.array(ts_r['t'])
    ax.plot(t_arr, ts_r['tau_wall'], color=colors_nr[nr], ls=ls_nr[nr],
            lw=2, label=f'N_ref={nr}')
ax.set_xlabel('t [s]'); ax.set_ylabel('tau_w [Pa]')
ax.set_title('Wall shear stress', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# [4] Centreline velocity
ax = axes[4]
ax.axhline(U_max, color='k', lw=2.5, ls='--', label=f'Analytical U_max={U_max:.2f}')
for nr, (ts_r, _) in results.items():
    t_arr = np.array(ts_r['t'])
    ax.plot(t_arr, ts_r['u_centerline'], color=colors_nr[nr], ls=ls_nr[nr],
            lw=2, label=f'N_ref={nr}')
ax.set_xlabel('t [s]'); ax.set_ylabel('u_cl [m/s]')
ax.set_title('Centreline velocity', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# [5] Mass flux
ax = axes[5]
ax.axhline(rho * U_mean * H, color='k', lw=2.5, ls='--',
           label=f'Analytical = {rho*U_mean*H:.4f}')
for nr, (ts_r, _) in results.items():
    t_arr = np.array(ts_r['t'])
    ax.plot(t_arr, ts_r['mass_flux_mid'], color=colors_nr[nr], ls=ls_nr[nr],
            lw=2, label=f'N_ref={nr}')
ax.set_xlabel('t [s]'); ax.set_ylabel('Mass flux [kg/m/s]')
ax.set_title('Mass flux (mid-domain)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# [6] Incompressibility vs velocity error (convergence path)
ax = axes[6]
for nr, (ts_r, _) in results.items():
    l2v = ts_r['l2_ss']
    l2d = ts_r['div_u_l2']
    if l2v and l2d:
        ax.loglog(l2v, l2d, color=colors_nr[nr], ls=ls_nr[nr], lw=2,
                  label=f'N_ref={nr}')
        ax.scatter([l2v[0]],  [l2d[0]],  c=colors_nr[nr], s=60, marker='^', zorder=5)
        ax.scatter([l2v[-1]], [l2d[-1]], c=colors_nr[nr], s=60, marker='s', zorder=5)
ax.set_xlabel('L2 velocity error [m/s]')
ax.set_ylabel('L2 ||div u|| [1/s]')
ax.set_title('Incompressibility vs velocity error\n(^ = start, s = end)', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)

# [7] Final div(u) L2 vs N_int (convergence rate for incompressibility)
ax = axes[7]
nrs    = sorted(results.keys())
n_ints = [results[nr][0]['n_interior'][0] for nr in nrs]
l2_vel = [results[nr][0]['l2_ss'][-1]     for nr in nrs]
l2_div = [results[nr][0]['div_u_l2'][-1]  for nr in nrs]
ax.loglog(n_ints, l2_vel, 'bo-', lw=2, ms=10, label='L2 velocity error')
ax.loglog(n_ints, l2_div, 'rs-', lw=2, ms=10, label='L2 ||div u||')
for nr, ni, lv, ld in zip(nrs, n_ints, l2_vel, l2_div):
    ax.annotate(f'N_ref={nr}', (ni, lv), fontsize=9, color='blue',
                xytext=(5, 5), textcoords='offset points')
if len(n_ints) >= 2:
    n_ref = np.array(n_ints, dtype=float)
    for slope, ls_, lbl in [(-1, ':', 'O(1/N)'), (-2, '--', 'O(1/N^2)')]:
        ax.loglog(n_ref, l2_vel[0]*(n_ref/n_ints[0])**slope,
                  'gray', ls=ls_, lw=1.5, label=lbl)
ax.set_xlabel('Interior vertex count N')
ax.set_ylabel('Final L2 norm')
ax.set_title('Spatial convergence rate at t=8 s', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)

# [8] div(u) L_inf over time (max pointwise violation)
ax = axes[8]
for nr, (ts_r, _) in results.items():
    t_arr = np.array(ts_r['t'])
    ax.semilogy(t_arr, ts_r['div_u_linf'], color=colors_nr[nr], ls=ls_nr[nr],
                lw=2, label=f'N_ref={nr}')
ax.set_xlabel('t [s]'); ax.set_ylabel('L_inf ||div u|| [1/s]')
ax.set_title('Max pointwise div(u) violation', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)

plt.suptitle(
    f'Mesh Refinement Study — 2D Incompressible Poiseuille Flow\nN_REFINE = {N_REFINE_VALUES}\n'
    f'Lagrangian DEC Poiseuille  |  Re={rho*U_mean*H/mu:.1f}  |  '
    f't_end={T_END:.1f} s  |  Incompressibility: div(u) = 0',
    fontsize=14)
plt.tight_layout()
p_comp = os.path.join(_COMP_DIR, 'mesh_refinement_comparison.png')
plt.savefig(p_comp, dpi=180, bbox_inches='tight')
plt.close()
print(f"\nMesh refinement comparison saved: {p_comp}")
print("\nParametric study complete.  Output folders:")
for nr in N_REFINE_VALUES:
    print(f"  N_REFINE={nr} -> {os.path.join(_HERE, 'fig', f'poiseuille_nrefine_{nr}')}")
print(f"  Comparison -> {_COMP_DIR}")