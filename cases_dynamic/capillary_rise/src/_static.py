"""Static capillary rise on the library integrators (laneI, 2026-10-06).

Shared by the runners ``capillary_rise_2D.py`` / ``capillary_rise_3D.py``,
the tests in ``ddgclib/tests/test_case_capillary_rise_static.py`` and
``diagnose_static_rise.py``.  There is no time loop here: every run goes
through ``SolverMethods.integrate`` with a preset of
``ddgclib.methods.PRESETS`` (``capillary_rise_static_2D`` / ``_3D``),
gravity through ``SolverMethods.dudt_fn(body_force=...)``, the EOS
through ``pressure_model=``, the surface tension and the contact angle
through ``free_surface=`` (method axis ``contact_line``).

The problem
-----------
A slit of width ``2 r`` (2D) or a round tube of radius ``r`` (3D) stands
in a reservoir whose free surface is at ``y = 0``.  The mesh is the
liquid in the tube from ``y = -D`` (a reservoir band, ``D`` about one to
two tube widths) up to the meniscus.  Walls are no-slip (frozen by
membership), the band below ``y = 0`` is held on the compressible
hydrostatic profile ``P(y) = K expm1(-rho0 g y / K)`` by
``HydrostaticReservoirBC`` (it supplies or absorbs the mass the column
needs), the top facets carry the capillary energy
``gamma A_free - gamma cos(theta) A_wet`` (``FreeSurface``) and the
contact-line vertices slide along the wall (``AxialSlideBC``).

The static answer is the Young-Laplace meniscus of the same pressure
profile (``ddgclib.analytical.young_laplace_meniscus``): its
volume-averaged height is Jurin's height (``gamma cos(theta) / (rho g r)``
in 2D, twice that in 3D) up to the compressibility of the model (about
``rho0 g h / (2 K)`` higher).  The run is scored by integrated
comparisons: the volume-averaged height of the free surface against the
reference, the L2 distance of the surface from the reference profile,
and the integrated pressure error of ``ddgclib.analytical``.

Two initial conditions: ``'flat'`` (a flat meniscus at Jurin's height
with the hydrostatic masses: the surface has to curve and the column to
take in the missing volume through the band) and ``'young_laplace'``
(the column above ``y = 0`` is stretched vertically onto the reference
profile: the start is within the discretisation error of the discrete
equilibrium).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib._boundary_conditions import (
    AxialSlideBC, BoundaryConditionSet, HydrostaticReservoirBC,
)
from ddgclib.analytical import (
    hydrostatic_pressure_tait, jurin_height, young_laplace_meniscus,
)
from ddgclib.eos import TaitMurnaghan
from ddgclib.geometry import ensure_simplex_cache, extrude
from ddgclib.geometry.domains import cylinder_volume
from ddgclib.initial_conditions import HydrostaticEOSMass, ZeroVelocity
from ddgclib.operators.free_surface import FreeSurface, facet_area
from ddgclib.operators.stress import cache_dual_volumes

from cases_dynamic.capillary_rise.src._params import FLUIDS

__all__ = ['CASES', 'StaticColumn', 'build_static_column', 'run_static',
           'refresh_pressure', 'mean_height', 'static_errors',
           'reference_meniscus', 'remap_arm', 'residual_acceleration',
           'run_case']

# Shipped parameters of the two runners: build kwargs and the default
# horizon in acoustic times L_domain / c0.
# ``ic``: the 2D runner starts flat (the pre-shaped start creeps and, at
# refinement 3, lets the fixed connectivity drift into a collapsed cell at
# 177 t_ac; the flat start settles to round-off); the coarse 3D tube
# (an octagon at refinement 1) is started pre-shaped (its flat start
# settles into a second, 9 % lower discrete equilibrium).  laneI section 4.
CASES: dict[str, dict[str, Any]] = {
    'capillary_rise_static_2D': dict(dim=2, n_refine=2, r=2e-3, n_cells=3,
                                     fluid='water', n_tac=100.0, ic='flat',
                                     alpha_art=0.05),
    'capillary_rise_static_3D': dict(dim=3, n_refine=1, r=2e-3, n_cells=3,
                                     fluid='water', n_tac=40.0,
                                     ic='young_laplace', alpha_art=0.05),
}


@dataclass
class StaticColumn:
    """A built column: mesh, vertex sets, BCs, EOS, the capillary surface
    and the numbers a run needs.  ``walls`` never move, so the frozenset
    is safe as a membership filter; ``free`` and ``contact`` move and
    are held as lists (identity)."""

    dim: int
    HC: Any
    bV: set
    walls: frozenset
    free: list
    contact: list
    bc_set: BoundaryConditionSet
    eos: TaitMurnaghan
    ic: HydrostaticEOSMass
    reservoir: HydrostaticReservoirBC
    surface: FreeSurface
    params: dict[str, Any] = field(default_factory=dict)

    def is_wall(self, v) -> bool:
        """``boundary_filter`` for the integrators."""
        return v in self.walls

    @property
    def mobile(self) -> list:
        return [v for v in self.HC.V if v not in self.walls]


def _scaled(HC, dim: int, scale: float, shift: float, axis: int) -> None:
    """``x -> scale x``, then ``x[axis] += shift``, in one move_all."""
    moves = []
    for v in list(HC.V):
        x = np.asarray(v.x_a[:dim], dtype=float) * scale
        x[axis] += shift
        moves.append((v, tuple(float(c) for c in x)))
    HC.V.move_all(moves)


def _build_mesh(dim: int, n_refine: int, r: float, n_cells: int):
    """``n_cells`` unit cells of width ``2 r`` stacked along the gravity
    axis: the structured square (2D) or the ``cylinder_volume``
    cross-section (3D), extruded, with the simplex cache of the
    extrusion (``ensure_simplex_cache``)."""
    w = 2.0 * r
    axis = dim - 1
    if dim == 2:
        unit = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        unit.triangulate()
        for _ in range(n_refine):
            unit.refine_all()
        HC = extrude(unit, n_cells, axis=axis, cdist=1e-10)
        ensure_simplex_cache(HC, dim)
        # the unit is [0, 1]^2: scale both axes by the width
        _scaled(HC, dim, w, 0.0, axis)
    else:
        unit = cylinder_volume(R=r, L=1.0, refinement=n_refine,
                               flow_axis=axis).HC
        HC = extrude(unit, n_cells, axis=axis, cdist=1e-10)
        ensure_simplex_cache(HC, dim)
        # the cross-section is already at radius r: scale the axis only
        moves = [(v, (v.x_a[0], v.x_a[1], float(v.x_a[2]) * w))
                 for v in list(HC.V)]
        HC.V.move_all(moves)
    return HC


def build_static_column(dim: int, n_refine: int, r: float = 2e-3,
                        fluid: str = 'water', n_cells: int = 3,
                        ic: str = 'flat', c0_factor: float = 10.0,
                        theta_deg: float | None = None,
                        g: float = 9.81) -> StaticColumn:
    """Build the column (see the module docstring).

    ``c0_factor``: ``c0 = c0_factor sqrt(g h_J)`` (10: 1 % compression
    over Jurin's height, as the hydrostatic column).  ``theta_deg``
    overrides the fluid's static contact angle.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    if ic not in ('flat', 'young_laplace'):
        raise ValueError(f"ic must be 'flat' or 'young_laplace', got {ic!r}")
    fp = FLUIDS[fluid]
    rho, gamma = fp['rho'], fp['gamma']
    theta = fp['theta_s_deg'] if theta_deg is None else float(theta_deg)
    axis = dim - 1
    w = 2.0 * r
    h_J = jurin_height(r, gamma, theta, rho, g, dim)
    L_dom = n_cells * w
    D = L_dom - h_J
    if D <= 0.0:
        raise ValueError(f"n_cells={n_cells} cells of width {w} do not hold "
                         f"Jurin's height {h_J:.4e} plus a reservoir band")
    K = rho * (c0_factor * math.sqrt(g * h_J))**2
    eos = TaitMurnaghan(rho0=rho, P0=0.0, K=K, n=1.0, rho_clip=(0.5, 2.0))
    c0 = float(eos.sound_speed(rho))
    P_hydro = hydrostatic_pressure_tait(rho, g, K)
    reference = young_laplace_meniscus(r, gamma, theta, P_hydro, dim)

    HC = _build_mesh(dim, n_refine, r, n_cells)
    # reservoir level y = 0: the flat meniscus sits at Jurin's height
    _scaled(HC, dim, 1.0, -D, axis)
    tol = 1e-9 * w
    hull = HC.boundary()
    top = max(v.x_a[axis] for v in HC.V)
    free = [v for v in HC.V if v in hull and abs(v.x_a[axis] - top) < tol]
    if dim == 2:
        contact = [v for v in free
                   if abs(v.x_a[0]) < tol or abs(v.x_a[0] - w) < tol]
    else:
        contact = [v for v in free
                   if np.hypot(v.x_a[0], v.x_a[1]) > r - 1e-6 * r]
    free_ids = {id(v) for v in free}
    wall_list = [v for v in hull if id(v) not in free_ids]

    if ic == 'young_laplace':
        # stretch the column above the reservoir level onto the profile
        moves = []
        for v in list(HC.V):
            y = float(v.x_a[axis])
            if y <= 0.0:
                continue
            lateral = (v.x_a[0] if dim == 2
                       else float(np.hypot(v.x_a[0], v.x_a[1])))
            y_s = float(reference.height(lateral))
            x = np.asarray(v.x_a[:dim], dtype=float)
            x[axis] = y * y_s / h_J
            moves.append((v, tuple(float(c) for c in x)))
        HC.V.move_all(moves)

    # Vertex sets are built AFTER the last move: a vertex hashes by its
    # coordinates, so a set made before a move does not contain it.
    walls = frozenset(wall_list)
    hull = set(hull)
    for v in HC.V:
        v.boundary = v in hull
    compute_vd(HC, method='barycentric')
    cache_dual_volumes(HC, dim)

    bV = set(walls)
    ZeroVelocity(dim=dim).apply(HC, bV)
    y_max = max(v.x_a[axis] for v in HC.V)
    hydro_ic = HydrostaticEOSMass(eos=eos, rho0=rho, g=g, gravity_axis=axis,
                                  h_ref=y_max, P_ref=P_hydro(y_max))
    hydro_ic.apply(HC, bV)

    surface = FreeSurface(HC, dim, gamma=gamma, theta_deg=theta,
                          walls=walls, free=free, contact=contact)
    reservoir = HydrostaticReservoirBC(hydro_ic, level=0.0)
    bc_set = BoundaryConditionSet()
    bc_set.add(AxialSlideBC(axis, contact), contact)
    bc_set.add(reservoir, HC.V)

    edges = [float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
             for v in HC.V for nb in v.nn]
    # Force balance on the DISCRETE cross-section: the 3D tube is the
    # polygon inscribed in the circle, so the exact static mean height of
    # the model is gamma cos(theta) (perimeter / area) / (rho g) up to the
    # compressibility factor of the round-tube reference.
    cross = cross_section(HC, dim, axis, free)
    h_jurin_poly = (gamma * math.cos(math.radians(theta)) * cross['perimeter']
                    / (cross['area'] * rho * g))
    params = dict(
        h_jurin_poly=h_jurin_poly, h_ref_poly=h_jurin_poly * reference.mean / h_J,
        perimeter=cross['perimeter'], area=cross['area'],
        dim=dim, n_refine=n_refine, r=r, w=w, fluid=fluid, rho=rho,
        gamma=gamma, theta_deg=theta, g=g, K=K, c0=c0, c0_factor=c0_factor,
        gravity_axis=axis, n_cells=n_cells, L_dom=L_dom, D=D, ic=ic,
        h_jurin=h_J, h_ref=reference.mean, h_apex_ref=reference.apex,
        h_contact_ref=reference.contact, t_ac=L_dom / c0,
        n_vertices=sum(1 for _ in HC.V), n_frozen=len(bV),
        n_free=len(free), n_contact=len(contact),
        dx_mean=float(np.mean(edges)), dx_min=min(edges),
        mass=sum(v.m for v in HC.V), P_hydro=P_hydro, reference=reference,
    )
    return StaticColumn(dim=dim, HC=HC, bV=bV, walls=walls, free=free,
                        contact=contact, bc_set=bc_set, eos=eos, ic=hydro_ic,
                        reservoir=reservoir, surface=surface, params=params)


def remap_arm(methods):
    """The reconnecting arm of a static preset: Delaunay rebuild that
    keeps the fluid domain + single-phase conservative remap."""
    return methods.replace(
        connectivity='delaunay_material', remap='conservative',
        redistribute_mass=True, label=methods.label + ' [remap arm]',
        notes='Reconnecting arm of the preset (laneP delaunay_material + '
              'conservative remap). Not the shipped configuration.')


def residual_acceleration(col: StaticColumn, dudt_fn: Callable) -> float:
    """max ``|a|`` over the mobile vertices; on a contact vertex the
    lateral components are the wall reaction and are left out."""
    contact = {id(v) for v in col.contact}
    worst = 0.0
    for v in col.mobile:
        a = np.array(dudt_fn(v)[:col.dim], dtype=float)
        if id(v) in contact:
            for ax in range(col.dim):
                if ax != col.params['gravity_axis']:
                    a[ax] = 0.0
        worst = max(worst, float(np.linalg.norm(a)))
    return worst


def cross_section(HC, dim: int, axis: int, free) -> dict[str, float]:
    """Perimeter and area of the tube cross-section as the mesh has it:
    the free-surface vertices projected along *axis* (2D: the width
    and 2 contact points; 3D: the inscribed polygon of the rim)."""
    pts = np.array([np.delete(np.asarray(v.x_a[:dim], dtype=float), axis)
                    for v in free])
    if dim == 2:
        width = float(pts[:, 0].max() - pts[:, 0].min())
        return dict(perimeter=2.0, area=width)
    centre = pts.mean(axis=0)
    rim = pts[np.hypot(*(pts - centre).T) > 0.999 * np.hypot(*(pts - centre).T).max()]
    ang = np.arctan2(rim[:, 1] - centre[1], rim[:, 0] - centre[0])
    rim = rim[np.argsort(ang)]
    nxt = np.roll(rim, -1, axis=0)
    perimeter = float(np.sum(np.hypot(*(nxt - rim).T)))
    area = 0.5 * abs(float(np.sum(rim[:, 0] * nxt[:, 1] - nxt[:, 0] * rim[:, 1])))
    return dict(perimeter=perimeter, area=area)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product (numpy 2 has no 2-vector ``np.cross``)."""
    return float(a[0] * b[1] - a[1] * b[0])


def mean_height(col: StaticColumn) -> float:
    """Volume-averaged height of the free surface: the integral of the
    piecewise-linear surface over the cross-section divided by its area
    (2D: the polyline over the width; 3D: the triangles over the polygon
    of the tube)."""
    dim, ax = col.dim, col.params['gravity_axis']
    num = 0.0
    den = 0.0
    for f in col.surface.free_facets:
        pts = np.array([v.x_a[:dim] for v in f], dtype=float)
        lateral = np.delete(pts, ax, axis=1)
        if dim == 2:
            a = abs(float(lateral[1, 0] - lateral[0, 0]))
        else:
            a = 0.5 * abs(_cross2(lateral[1] - lateral[0],
                                  lateral[2] - lateral[0]))
        num += a * float(pts[:, ax].mean())
        den += a
    return num / den


def shape_error(col: StaticColumn) -> float:
    """Area-weighted RMS distance (along the gravity axis) of the free
    surface from the reference profile, sampled on the facets."""
    dim, ax = col.dim, col.params['gravity_axis']
    ref = col.params['reference']
    num = 0.0
    den = 0.0
    if dim == 2:
        t = np.linspace(0.0, 1.0, 21)
        for f in col.surface.free_facets:
            p0, p1 = (np.asarray(v.x_a[:2], dtype=float) for v in f)
            pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
            d2 = (pts[:, ax] - ref.height(pts[:, 0]))**2
            a = abs(float(p1[0] - p0[0]))
            num += a * float(np.trapezoid(d2, t))
            den += a
    else:
        # degree-2 triangle rule: the three edge midpoints
        for f in col.surface.free_facets:
            pts = np.array([v.x_a[:3] for v in f], dtype=float)
            mids = 0.5 * (pts + np.roll(pts, -1, axis=0))
            rr = np.hypot(mids[:, 0], mids[:, 1])
            d2 = (mids[:, 2] - ref.height(rr))**2
            a = 0.5 * abs(_cross2(pts[1, :2] - pts[0, :2],
                                  pts[2, :2] - pts[0, :2]))
            num += a * float(d2.mean())
            den += a
    return math.sqrt(num / den)


def refresh_pressure(col: StaticColumn) -> None:
    """Duals, dual volumes and EOS pressure at the CURRENT positions."""
    compute_vd(col.HC, method='barycentric')
    cache_dual_volumes(col.HC, col.dim)
    for v in col.HC.V:
        if v.dual_vol > 1e-30:
            v.rho = v.m / v.dual_vol
            v.p = float(col.eos.pressure(v.rho))


def reference_meniscus(col: StaticColumn):
    return col.params['reference']


def static_errors(col: StaticColumn) -> dict[str, float]:
    """Integrated comparison of the current state with the Young-Laplace
    reference: the volume-averaged surface height (``h_mean``) against
    ``h_ref`` (and Jurin's incompressible ``h_jurin``), the RMS distance
    of the surface from the profile, the apex and contact heights, and
    the integrated L2 pressure error of ``ddgclib.analytical`` over the
    mobile vertices above the band against ``P(y)``."""
    from ddgclib.analytical import integrated_l2_norm, integrated_pressure_error
    refresh_pressure(col)
    p = col.params
    ax = p['gravity_axis']
    P_hydro = p['P_hydro']

    def P(x):
        return P_hydro(float(x[ax]))

    above = [v for v in col.mobile if v.x_a[ax] >= 0.0]
    errs = integrated_pressure_error(col.HC, above, P, dim=col.dim)
    h = mean_height(col)
    contact_h = float(np.mean([v.x_a[ax] for v in col.contact]))
    free_ids = {id(v) for v in col.free}
    contact_ids = {id(v) for v in col.contact}
    inner = [v for v in col.free if id(v) not in contact_ids]
    apex = (min(inner, key=lambda v: (v.x_a[0] - p['r'])**2 if col.dim == 2
                else v.x_a[0]**2 + v.x_a[1]**2)
            if inner else None)
    vols = np.array([v.dual_vol for v in col.mobile])
    edges = np.array([np.linalg.norm(v.x_a[:col.dim] - nb.x_a[:col.dim])
                      for v in col.mobile for nb in v.nn])
    return dict(
        # mesh quality: the smallest mobile cell and edge against the mean
        # (a fixed connectivity cannot repair a cell that a slow drift
        # squeezes; laneI section 5)
        vol_min_rel=float(vols.min() / vols.mean()),
        edge_min_rel=float(edges.min() / p['dx_mean']),
        h_mean=h, h_ref=p['h_ref'], h_jurin=p['h_jurin'],
        h_ref_poly=p['h_ref_poly'], h_jurin_poly=p['h_jurin_poly'],
        h_error=h - p['h_ref'], h_error_rel=(h - p['h_ref']) / p['h_ref'],
        h_error_poly_rel=(h - p['h_ref_poly']) / p['h_ref_poly'],
        h_contact=contact_h, h_contact_ref=p['h_contact_ref'],
        h_apex=float(apex.x_a[ax]) if apex is not None else float('nan'),
        h_apex_ref=p['h_apex_ref'],
        shape_rms=shape_error(col),
        p_l2=integrated_l2_norm(col.HC, above, P, dim=col.dim),
        p_max_int=max(errs) if errs else 0.0,
        P_cap=-P_hydro(p['h_ref']),
        energy=col.surface.energy(),
        injected=col.reservoir.injected,
        mass_drift=(sum(v.m for v in col.HC.V) - p['mass']) / p['mass'],
        n_free=len(free_ids),
    )


def run_static(col: StaticColumn, methods, *, n_tac: float,
               alpha_art: float = 0.1, mu: float | None = None,
               cfl: float = 0.25, callback: Callable | None = None,
               custom: Callable | None = None) -> dict[str, Any]:
    """Integrate *col* for *n_tac* acoustic times ``L_dom / c0`` with
    *methods*.  The viscosity is *mu* if given, else ``alpha_art rho c0
    dx_mean``.  Returns the per-step series ``t``, ``ke``, ``umax`` (mobile
    vertices), ``h`` (volume-averaged surface height) and the run numbers.
    """
    p = col.params
    dim = col.dim
    if mu is None:
        mu = alpha_art * p['rho'] * p['c0'] * p['dx_mean']
    g_vec = np.zeros(dim)
    g_vec[p['gravity_axis']] = -p['g']
    dudt_fn = methods.dudt_fn(col.HC, mu=mu, pressure_model=col.eos,
                              body_force=g_vec, free_surface=col.surface)
    dt_ac = cfl * p['dx_min'] / p['c0']
    dt_cap = 0.5 * math.sqrt(p['rho'] * p['dx_min']**3
                             / (2.0 * math.pi * p['gamma']))
    dt = min(dt_ac, dt_cap)
    n_steps = int(round(n_tac * p['t_ac'] / dt))
    t_h: list[float] = []
    ke_h: list[float] = []
    u_h: list[float] = []
    h_h: list[float] = []

    def cb(step, t, HC, bV=None, diagnostics=None):
        ke = 0.0
        umax = 0.0
        for v in HC.V:
            if v in col.walls:
                continue
            u2 = float(np.dot(v.u[:dim], v.u[:dim]))
            ke += 0.5 * v.m * u2
            umax = max(umax, u2)
        t_h.append(t)
        ke_h.append(ke)
        u_h.append(math.sqrt(umax))
        h_h.append(mean_height(col))
        if callback is not None:
            callback(step, t, HC, bV, diagnostics)

    methods.integrate(col.HC, col.bV, dudt_fn, dt=dt, n_steps=n_steps,
                      bc_set=col.bc_set, callback=cb, custom=custom,
                      pressure_model=col.eos, boundary_filter=col.is_wall)
    return dict(t=np.array(t_h), ke=np.array(ke_h), umax=np.array(u_h),
                h=np.array(h_h), dt=dt, dt_acoustic=dt_ac,
                dt_capillary=dt_cap, n_steps=n_steps, mu=mu, cfl=cfl,
                n_tac=n_tac, alpha_art=alpha_art, dudt_fn=dudt_fn)


# ----------------------------------------------------------------------
# runner body shared by the two scripts
# ----------------------------------------------------------------------

def run_case(name: str, argv: list[str] | None = None) -> dict[str, Any]:
    """Run one shipped static case through its preset, write
    ``results/<name>/`` (snapshots, ``methods.json``, ``summary.json``)
    and ``fig/`` under ``--out`` (default: the case directory).  Returns
    the summary dict."""
    import argparse
    import json

    from ddgclib.data import StateHistory
    from ddgclib.methods import PRESETS, record_methods

    defaults = CASES[name]
    ap = argparse.ArgumentParser(description=f"{name} (preset "
                                             f"PRESETS[{name!r}])")
    ap.add_argument('--n-refine', type=int, default=defaults['n_refine'])
    ap.add_argument('--n-tac', type=float, default=defaults['n_tac'],
                    help='horizon in acoustic times L_dom / c0')
    ap.add_argument('--ic', choices=('flat', 'young_laplace'),
                    default=defaults['ic'])
    ap.add_argument('--arm', choices=('preset', 'remap'), default='preset',
                    help="'remap': delaunay_material + conservative remap")
    ap.add_argument('--alpha-art', type=float, default=defaults['alpha_art'],
                    help='artificial viscosity mu = alpha rho c0 dx '
                         '(0.05: enough to settle, 0.02 flutters)')
    ap.add_argument('--r', type=float, default=defaults['r'],
                    help='tube half-width / radius [m]')
    ap.add_argument('--n-cells', type=int, default=defaults['n_cells'])
    ap.add_argument('--out', default=None,
                    help='output root (default: the case directory)')
    ap.add_argument('--no-anim', action='store_true')
    args = ap.parse_args(argv)

    methods = PRESETS[name]
    if args.arm == 'remap':
        methods = remap_arm(methods)

    here = (args.out if args.out is not None else
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tag = name if args.arm == 'preset' else f'{name}_remap'
    tag += f'_{args.ic}'
    fig_dir = os.path.join(here, 'fig')
    res_dir = os.path.join(here, 'results', tag)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    print("=" * 64)
    print(methods.describe())
    print("=" * 64)

    col = build_static_column(defaults['dim'], args.n_refine, r=args.r,
                              fluid=defaults['fluid'], n_cells=args.n_cells,
                              ic=args.ic)
    p = col.params
    err0 = static_errors(col)
    print(f"Mesh: {p['n_vertices']} vertices, {p['n_frozen']} frozen, "
          f"{p['n_free']} free-surface, {p['n_contact']} contact; "
          f"c0 = {p['c0']:.3f} m/s, t_ac = {p['t_ac']:.4e} s")
    print(f"Jurin height {p['h_jurin']:.6e} m, reference (compressible "
          f"Young-Laplace) mean {p['h_ref']:.6e}, apex {p['h_apex_ref']:.6e}, "
          f"contact {p['h_contact_ref']:.6e}; discrete cross-section "
          f"(perimeter / area {p['perimeter'] / p['area']:.4f} against "
          f"{(defaults['dim'] - 1) / p['r']:.4f}): mean {p['h_ref_poly']:.6e}")
    print(f"Start: h_mean {err0['h_mean']:.6e} ({err0['h_error_rel']:+.3e}), "
          f"shape rms {err0['shape_rms']:.3e}")

    n_rec = 200
    cfl = 0.25
    history = StateHistory(fields=['u', 'p'], record_every=1,
                           save_dir=os.path.join(res_dir, 'snapshots'))
    dt_probe = min(cfl * p['dx_min'] / p['c0'],
                   0.5 * math.sqrt(p['rho'] * p['dx_min']**3
                                   / (2.0 * math.pi * p['gamma'])))
    n_steps = int(round(args.n_tac * p['t_ac'] / dt_probe))
    history.record_every = max(1, n_steps // n_rec)
    res = run_static(col, methods, n_tac=args.n_tac, cfl=cfl,
                     alpha_art=args.alpha_art, callback=history.callback)
    err = static_errors(col)
    summary = dict(
        case=name, arm=args.arm, ic=args.ic, n_refine=args.n_refine,
        r=args.r, n_cells=args.n_cells, n_vertices=p['n_vertices'],
        n_steps=res['n_steps'], dt=res['dt'], n_tac=args.n_tac,
        mu=res['mu'], alpha_art=args.alpha_art, c0=p['c0'],
        umax_peak=float(res['umax'].max()), umax_end=float(res['umax'][-1]),
        ke_peak=float(res['ke'].max()), ke_end=float(res['ke'][-1]),
        settled_max_a=residual_acceleration(col, res['dudt_fn']),
        **{f'start_{k}': v for k, v in err0.items()},
        **err,
    )
    print(f"Done: {res['n_steps']} steps, dt {res['dt']:.3e} s "
          f"(acoustic {res['dt_acoustic']:.3e}, capillary "
          f"{res['dt_capillary']:.3e}), mu = {res['mu']:.4f} Pa s")
    print(f"  max|u| peak {summary['umax_peak']:.4e} -> end "
          f"{summary['umax_end']:.4e} m/s, KE end {summary['ke_end']:.4e} J, "
          f"settled max|a| {summary['settled_max_a']:.4e} m/s^2")
    print(f"  h_mean {err['h_mean']:.6e} m against reference {err['h_ref']:.6e}"
          f" ({err['h_error_rel']:+.3e}; Jurin {err['h_jurin']:.6e}; discrete "
          f"cross-section {err['h_ref_poly']:.6e}, {err['h_error_poly_rel']:+.3e}), "
          f"contact {err['h_contact']:.6e} / {err['h_contact_ref']:.6e}, "
          f"apex {err['h_apex']:.6e} / {err['h_apex_ref']:.6e}")
    print(f"  shape rms {err['shape_rms']:.4e} m, pressure L2 {err['p_l2']:.4e}"
          f" Pa (P_cap {err['P_cap']:.2f}), max|p V - int P dV| "
          f"{err['p_max_int']:.3e}, injected mass {err['injected']:+.3e} "
          f"({err['mass_drift']:+.3e} of the mesh); smallest mobile cell "
          f"{err['vol_min_rel']:.3f} of the mean, shortest edge "
          f"{err['edge_min_rel']:.3f} dx")

    record_methods(os.path.join(res_dir, 'methods.json'), methods, col.HC,
                   extra={k: summary[k] for k in (
                       'case', 'arm', 'ic', 'n_refine', 'r', 'n_cells',
                       'n_steps', 'dt', 'n_tac', 'mu', 'alpha_art', 'c0')}
                   | {'eos': 'TaitMurnaghan(n=1, P0=0, K=rho (10 sqrt(g '
                             'h_J))^2)', 'fluid': defaults['fluid'],
                      'theta_deg': p['theta_deg'], 'gamma': p['gamma'],
                      'cfl': res['cfl']})
    np.savez(os.path.join(res_dir, 'series.npz'), t=res['t'], ke=res['ke'],
             umax=res['umax'], h=res['h'])
    with open(os.path.join(res_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
        fh.write('\n')

    _plots(col, res, tag, fig_dir)
    if not args.no_anim:
        _animate(col, history, tag, fig_dir)
    print(f"Outputs: {res_dir}, {fig_dir}")
    return summary


def _plots(col: StaticColumn, res: dict[str, Any], tag: str,
           fig_dir: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = col.params
    ax_g = p['gravity_axis']
    t = res['t'] / p['t_ac']
    ref = p['reference']

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4))
    if np.any(res['ke'] > 0):
        a1.semilogy(t, res['ke'], lw=1)
    a1.set_xlabel('$t / t_{ac}$')
    a1.set_ylabel('kinetic energy [J]')
    a1.grid(True, alpha=0.3)
    a2.semilogy(t, np.maximum(res['umax'], 1e-300), lw=1, color='C3')
    a2.set_xlabel('$t / t_{ac}$')
    a2.set_ylabel('max $|u|$ [m/s]')
    a2.grid(True, alpha=0.3)
    a3.plot(t, res['h'] * 1e3, lw=1, label='volume-averaged height')
    a3.axhline(p['h_ref'] * 1e3, color='k', ls='--', lw=1,
               label='Young-Laplace reference')
    a3.axhline(p['h_jurin'] * 1e3, color='k', ls=':', lw=1,
               label='Jurin (incompressible)')
    a3.set_xlabel('$t / t_{ac}$')
    a3.set_ylabel('h [mm]')
    a3.legend(fontsize=8)
    a3.grid(True, alpha=0.3)
    fig.suptitle(f"{tag}: settling ($\\mu$ = {res['mu']:.3f} Pa s, "
                 f"{p['n_vertices']} vertices)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'{tag}_settling.png'), dpi=150)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 5))
    if col.dim == 2:
        xs = np.linspace(0.0, p['w'], 200)
        a.plot(xs * 1e3, ref.height(xs) * 1e3, 'k--', lw=1.5,
               label='Young-Laplace')
        pts = sorted((float(v.x_a[0]), float(v.x_a[1])) for v in col.free)
        a.plot([q[0] * 1e3 for q in pts], [q[1] * 1e3 for q in pts], 'o-',
               ms=4, label='DDG free surface')
        a.set_xlabel('x [mm]')
    else:
        rs = np.linspace(0.0, p['r'], 200)
        a.plot(rs * 1e3, ref.height(rs) * 1e3, 'k--', lw=1.5,
               label='Young-Laplace')
        a.plot([np.hypot(v.x_a[0], v.x_a[1]) * 1e3 for v in col.free],
               [v.x_a[2] * 1e3 for v in col.free], 'o', ms=4,
               label='DDG free surface')
        a.set_xlabel('radius [mm]')
    a.set_ylabel('height [mm]')
    a.legend()
    a.grid(True, alpha=0.3)
    fig.suptitle(f'{tag}: meniscus against the Young-Laplace profile')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'{tag}_meniscus.png'), dpi=150)
    plt.close(fig)

    from ddgclib.visualization.unified import plot_primal
    fig, _ = plot_primal(col.HC, bV=col.bV, scalar_field='p',
                         title=f'{tag}: pressure', save_path=None,
                         cmap='coolwarm')
    fig.savefig(os.path.join(fig_dir, f'{tag}_mesh_pressure.png'), dpi=150)
    plt.close(fig)
    print(f"  -> fig/{tag}_settling.png, fig/{tag}_meniscus.png, "
          f"fig/{tag}_mesh_pressure.png")


def _animate(col: StaticColumn, history, tag: str, fig_dir: str) -> None:
    from ddgclib.visualization import dynamic_plot_fluid
    path = os.path.join(fig_dir, f'{tag}.mp4')
    try:
        dynamic_plot_fluid(history, col.HC, bV=col.bV, save_path=path)
        print(f"  -> fig/{tag}.mp4")
    except Exception as e:  # noqa: BLE001 - the animation is optional
        print(f"  animation skipped ({type(e).__name__}: {e})")
