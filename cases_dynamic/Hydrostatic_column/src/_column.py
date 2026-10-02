"""Hydrostatic column on the library integrators (laneP, 2026-10-01).

Shared by the four runners ``Hydrostatic_{1D,2D,3D,2D_periodic}.py``, the
tests in ``ddgclib/tests/test_case_hydrostatic.py`` and
``diagnose_column.py``.  There is no time loop here: every run goes
through ``SolverMethods.integrate`` with a preset from
``ddgclib.methods.PRESETS`` (``hydrostatic_1D`` ...), gravity enters
through ``SolverMethods.dudt_fn(body_force=...)``.

The column
----------
Height ``H`` along the last axis, unit width, frozen (no-slip) bottom,
free surface on top, gauge pressure (``P0 = 0``), linear Tait EOS with
``c0 = 10 sqrt(g H)`` (1 % compression at the bottom).  Side walls are
no-slip (frozen vertices) or free-slip (``FreeSlipWallBC``: vertices slide
along the wall, which makes the solution one-dimensional).

Two initial conditions:

``'drop'``
    Uniform density ``rho0`` (zero pressure everywhere).  The column falls,
    compresses and rings at its fundamental acoustic mode (period
    ``4 H / c0``) until viscosity has damped it.  The free surface settles
    at ``settled_height`` (mass conservation).
``'equilibrium'``
    Masses of the compressible hydrostatic profile at the initial
    geometry (``HydrostaticEOSMass``).  The column starts within the
    discretisation error of its discrete equilibrium.

Reference: ``P(y) = K (exp(rho0 g (h - y) / K) - 1)`` with ``h`` the
surface height (``H`` for ``'equilibrium'``, ``settled_height`` for
``'drop'``); compared with the run through the integrated comparisons of
``ddgclib.analytical``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ddgclib._boundary_conditions import BoundaryConditionSet, FreeSlipWallBC
from ddgclib.eos import TaitMurnaghan
from ddgclib.initial_conditions import HydrostaticEOSMass

from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic_column

__all__ = ['Column', 'build_column', 'run_column', 'refresh_pressure',
           'column_errors', 'reference_pressure', 'settled_height',
           'remap_arm', 'residual_acceleration', 'static_residuals',
           'run_case']

# Shipped parameters of the four runners: build_column kwargs and the
# default horizon in acoustic times (the 1D column rings longest: the
# damping time of the fundamental mode grows like H / dx).
CASES: dict[str, dict[str, Any]] = {
    'hydrostatic_1D': dict(dim=1, n_refine=4, H=10.0, side_walls='noslip',
                           n_tac=200.0),
    'hydrostatic_2D': dict(dim=2, n_refine=3, H=1.0, side_walls='noslip',
                           n_tac=100.0),
    'hydrostatic_3D': dict(dim=3, n_refine=2, H=1.0, side_walls='noslip',
                           n_tac=40.0),
    'hydrostatic_2D_periodic': dict(dim=2, n_refine=3, H=1.0,
                                    side_walls='freeslip', n_tac=100.0),
}


@dataclass
class Column:
    """A built column: mesh, frozen set, BCs, EOS and the numbers a run
    needs.  ``walls`` is the frozen vertex set as it was at setup; those
    vertices never move, so it is safe as a membership filter."""

    dim: int
    HC: Any
    bV: set
    walls: frozenset
    bc_set: BoundaryConditionSet
    eos: TaitMurnaghan
    params: dict[str, Any] = field(default_factory=dict)
    # id(v) -> wall-normal axes of a free-slip vertex (constrained there)
    slip_axes: dict[int, list[int]] = field(default_factory=dict)

    def is_wall(self, v) -> bool:
        """``boundary_filter`` for the integrators: only the frozen walls
        stay frozen, free-surface vertices are integrated."""
        return v in self.walls

    @property
    def free(self) -> list:
        return [v for v in self.HC.V if v not in self.walls]


def settled_height(H: float, rho: float, g: float, K: float) -> float:
    """Free-surface height of the compressible column that holds the mass
    ``rho H`` per unit area (linear Tait EOS)."""
    return K / (rho * g) * np.log1p(rho * g * H / K)


def reference_pressure(col: Column) -> Callable[[np.ndarray], float]:
    """Analytical compressible hydrostatic pressure ``P(x)`` of *col*."""
    p = col.params
    rho, g, K, h, ax = p['rho'], p['g'], p['K'], p['h_surface'], p['gravity_axis']

    def P(x):
        depth = h - x[ax]
        return K * np.expm1(rho * g * depth / K) if depth > 0.0 else 0.0

    return P


def build_column(dim: int, n_refine: int, H: float = 1.0, rho: float = 1000.0,
                 g: float = 9.81, side_walls: str = 'noslip',
                 ic: str = 'drop', free_surface: bool = True) -> Column:
    """Build the column (see the module docstring).  ``free_surface=False``
    freezes the top as well (closed box; the control experiment of
    ``diagnose_column.py free_surface``)."""
    if side_walls not in ('noslip', 'freeslip'):
        raise ValueError(f"side_walls must be 'noslip' or 'freeslip', "
                         f"got {side_walls!r}")
    if ic not in ('drop', 'equilibrium'):
        raise ValueError(f"ic must be 'drop' or 'equilibrium', got {ic!r}")
    if not free_surface and side_walls != 'noslip':
        raise ValueError("free_surface=False is the closed no-slip box")
    gax = dim - 1
    K = rho * (10.0 * np.sqrt(g * H))**2
    eos = TaitMurnaghan(rho0=rho, P0=0.0, K=K, n=1.0, rho_clip=(0.5, 2.0))
    c0 = float(eos.sound_speed(rho))

    # zero velocity, uniform density rho (the 'drop' state), duals cached
    HC, bV, bc_set, _, _ = setup_hydrostatic_column(
        dim=dim, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=0.0, mu=0.0,
        gravity_axis=gax, free_surface=free_surface,
        freeze_walls='bottom_only' if side_walls == 'freeslip' else 'all')
    if ic == 'equilibrium':
        HydrostaticEOSMass(eos=eos, rho0=rho, g=g, gravity_axis=gax,
                           h_ref=H, P_ref=0.0).apply(HC, bV)

    slip_axes: dict[int, list[int]] = {}
    if side_walls == 'freeslip':
        tol = 1e-12
        for ax in range(dim - 1):
            for coord in (0.0, 1.0):
                side = [v for v in HC.V if v not in bV
                        and abs(v.x_a[ax] - coord) < tol]
                bc_set.add(FreeSlipWallBC(wall_axis=ax, wall_coord=coord),
                           side)
                for v in side:
                    slip_axes.setdefault(id(v), []).append(ax)

    edges = [float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
             for v in HC.V for nb in v.nn]
    params = dict(
        dim=dim, n_refine=n_refine, H=H, rho=rho, g=g, K=K, c0=c0,
        gravity_axis=gax, side_walls=side_walls, ic=ic, t_ac=H / c0,
        free_surface=free_surface,
        h_surface=H if ic == 'equilibrium' or not free_surface
        else settled_height(H, rho, g, K),
        n_vertices=sum(1 for _ in HC.V), n_frozen=len(bV),
        dx_mean=float(np.mean(edges)), dx_min=min(edges),
        mass=sum(v.m for v in HC.V),
    )
    return Column(dim=dim, HC=HC, bV=bV, walls=frozenset(bV), bc_set=bc_set,
                  eos=eos, params=params, slip_axes=slip_axes)


def residual_acceleration(col: Column, dudt_fn: Callable) -> float:
    """max ``|a|`` over the free vertices at the current state.  On a
    free-slip vertex the wall-normal component is the wall reaction (the
    BC removes it every step), so it is left out."""
    worst = 0.0
    for v in col.free:
        a = np.array(dudt_fn(v)[:col.dim], dtype=float)
        for ax in col.slip_axes.get(id(v), ()):
            a[ax] = 0.0
        worst = max(worst, float(np.linalg.norm(a)))
    return worst


def remap_arm(methods):
    """The reconnecting arm of a hydrostatic preset: Delaunay rebuild that
    keeps the fluid domain + single-phase conservative remap (2D / 3D)."""
    return methods.replace(
        connectivity='delaunay_material', remap='conservative',
        redistribute_mass=True, label=methods.label + ' [remap arm]',
        notes='Reconnecting arm of the preset: Delaunay rebuild that keeps '
              'the fluid domain + single-phase conservative remap. Not the '
              'shipped configuration (the preset keeps the connectivity).')


def run_column(col: Column, methods, *, n_tac: float, alpha_art: float = 0.5,
               mu: float | None = None, cfl: float = 0.25,
               callback: Callable | None = None,
               custom: Callable | None = None) -> dict[str, Any]:
    """Integrate *col* for *n_tac* acoustic times ``H / c0`` with *methods*.

    The viscosity is ``mu`` if given, else the artificial value
    ``alpha_art rho c0 dx_mean`` (the shipped runners use 0.5): it only
    damps the acoustic ringing of the ``'drop'`` start, amplitude e-fold
    time ``8 H^2 rho / (pi^2 mu)`` for the fundamental mode.  *custom* is
    the retopology callable of a ``connectivity='custom'`` *methods*
    (diagnostics only).  Returns the per-step series ``t``, ``ke``,
    ``umax`` (free vertices) and the run numbers.
    """
    p = col.params
    dim = col.dim
    if mu is None:
        mu = alpha_art * p['rho'] * p['c0'] * p['dx_mean']
    g_vec = np.zeros(dim)
    g_vec[p['gravity_axis']] = -p['g']
    dudt_fn = methods.dudt_fn(col.HC, mu=mu, pressure_model=col.eos,
                              body_force=g_vec)
    dt = cfl * p['dx_min'] / p['c0']
    n_steps = int(round(n_tac * p['t_ac'] / dt))
    t_h: list[float] = []
    ke_h: list[float] = []
    u_h: list[float] = []

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
        u_h.append(np.sqrt(umax))
        if callback is not None:
            callback(step, t, HC, bV, diagnostics)

    methods.integrate(col.HC, col.bV, dudt_fn, dt=dt, n_steps=n_steps,
                      bc_set=col.bc_set, callback=cb, custom=custom,
                      pressure_model=col.eos, boundary_filter=col.is_wall)
    return dict(t=np.array(t_h), ke=np.array(ke_h), umax=np.array(u_h),
                dt=dt, n_steps=n_steps, mu=mu, cfl=cfl, n_tac=n_tac,
                dudt_fn=dudt_fn)


def refresh_pressure(col: Column) -> None:
    """Duals, dual volumes and EOS pressure at the CURRENT positions (the
    ``v.p`` left by a run is the one of the last force evaluation, taken
    before the last move)."""
    from hyperct.ddg import compute_vd
    from ddgclib.operators.stress import cache_dual_volumes
    compute_vd(col.HC, method='barycentric')
    cache_dual_volumes(col.HC, col.dim)
    for v in col.HC.V:
        if v.dual_vol > 1e-30:
            v.rho = v.m / v.dual_vol
            v.p = float(col.eos.pressure(v.rho))


def column_errors(col: Column) -> dict[str, float]:
    """Integrated comparison of the current state with the analytical
    compressible profile (``ddgclib.analytical``).

    ``l2`` is the volume-weighted L2 norm over all free vertices,
    ``l2_interior`` over the free vertices that are not on the hull (the
    pressure of a boundary cell is tied to the nodal value by the
    centred flux, not to the cell average, so the boundary cells carry an
    O(dx) offset).  ``max_int`` is ``max |p_i V_i - int P dV|``.  In 3D
    the cell integral is the point value times the volume (the 3D
    quadrature of ``ddgclib.analytical`` is still deferred).
    """
    from ddgclib.analytical import (integrated_l2_norm,
                                    integrated_pressure_error)
    refresh_pressure(col)
    p = col.params
    P = reference_pressure(col)
    free = col.free
    if getattr(col.HC, '_simplices', None) is not None:
        from hyperct.ddg import boundary_from_simplices
        hull = boundary_from_simplices(col.HC, col.dim)
    else:
        hull = col.HC.boundary()
    interior = [v for v in free if v not in hull]
    errs = integrated_pressure_error(col.HC, free, P, dim=col.dim)
    top = max(v.x_a[p['gravity_axis']] for v in col.HC.V)
    return dict(
        l2=integrated_l2_norm(col.HC, free, P, dim=col.dim),
        l2_interior=integrated_l2_norm(col.HC, interior, P, dim=col.dim),
        max_int=max(errs),
        rho_g_H=p['rho'] * p['g'] * p['H'],
        surface_max=float(top),
        mass_drift=(sum(v.m for v in col.HC.V) - p['mass']) / p['mass'],
    )


def decay_rate(res: dict[str, Any], col: Column, t_from: float = 2.0) -> float:
    """Fitted decay rate of the KE envelope [1 / t_ac] after *t_from*
    acoustic times (KE maxima over windows of one fundamental period)."""
    t_ac = col.params['t_ac']
    t = res['t'] / t_ac
    window = 4.0
    tm, km = [], []
    start = t_from
    while start + window <= t[-1] + 1e-9:
        sel = (t >= start) & (t < start + window)
        if sel.any() and res['ke'][sel].max() > 0.0:
            i = int(np.argmax(res['ke'][sel]))
            tm.append(t[sel][i])
            km.append(res['ke'][sel][i])
        start += window
    if len(tm) < 2:
        return float('nan')
    return float(-np.polyfit(tm, np.log(km), 1)[0])


def static_residuals(dim: int, n_refine: int, methods, H: float = 1.0,
                     rho: float = 1000.0, g: float = 9.81,
                     mu: float = 1e-3) -> dict[str, float]:
    """Force balance of the prescribed incompressible profile
    ``P = rho g (H - y)`` in the closed box (all walls frozen, no EOS),
    through ``methods.dudt_fn(body_force=...)``: max ``|a|`` over the
    interior vertices for

    ``nodal``         ``v.p = P(x_vertex)``: the centred flux is exact for
                      nodal values of a linear field (round-off).
    ``cell_average``  ``v.p`` = dual-cell average (``HydrostaticPressure``).
                      The average over a boundary half cell is not the
                      nodal value, so the neighbours of the walls see an
                      O(1) error that does not converge (1D, 2D; in 3D the
                      IC falls back to the nodal value).
    """
    gax = dim - 1
    HC, bV, _, _, _ = setup_hydrostatic_column(
        dim=dim, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=0.0, mu=mu,
        gravity_axis=gax)
    g_vec = np.zeros(dim)
    g_vec[gax] = -g
    dudt_fn = methods.dudt_fn(HC, mu=mu, body_force=g_vec)
    interior = [v for v in HC.V if v not in bV]

    def residual() -> float:
        return max(float(np.linalg.norm(dudt_fn(v))) for v in interior)

    out = {'cell_average': residual()}
    for v in HC.V:
        v.p = rho * g * (H - v.x_a[gax])
    out['nodal'] = residual()
    return out


# ----------------------------------------------------------------------
# runner body shared by the four scripts
# ----------------------------------------------------------------------

def run_case(name: str, argv: list[str] | None = None) -> dict[str, Any]:
    """Run one shipped hydrostatic case through its preset, write
    ``results/<name>/`` (snapshots, ``methods.json``, ``summary.json``)
    and ``fig/``.  Returns the summary dict."""
    import argparse
    import json

    from ddgclib.data import StateHistory
    from ddgclib.methods import PRESETS, record_methods

    defaults = CASES[name]
    ap = argparse.ArgumentParser(description=f"{name} (preset "
                                             f"PRESETS[{name!r}])")
    ap.add_argument('--n-refine', type=int, default=defaults['n_refine'])
    ap.add_argument('--n-tac', type=float, default=defaults['n_tac'],
                    help='horizon in acoustic times H / c0')
    ap.add_argument('--ic', choices=('drop', 'equilibrium'), default='drop')
    ap.add_argument('--arm', choices=('preset', 'remap'), default='preset',
                    help="'remap': delaunay_material + conservative remap")
    ap.add_argument('--alpha-art', type=float, default=0.5,
                    help='artificial viscosity mu = alpha rho c0 dx')
    ap.add_argument('--no-anim', action='store_true')
    args = ap.parse_args(argv)

    methods = PRESETS[name]
    if args.arm == 'remap':
        if defaults['dim'] == 1:
            ap.error("--arm remap needs 2D or 3D (in 1D the chain cannot "
                     "reconnect)")
        methods = remap_arm(methods)

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = name if args.arm == 'preset' else f'{name}_remap'
    fig_dir = os.path.join(here, 'fig')
    res_dir = os.path.join(here, 'results', tag)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    print("=" * 64)
    print(methods.describe())
    print("=" * 64)

    static = static_residuals(defaults['dim'], args.n_refine, methods,
                              H=defaults['H'])
    print(f"Static check (prescribed rho g (H - y), closed box): max|a| = "
          f"{static['nodal']:.3e} m/s^2 with nodal pressures, "
          f"{static['cell_average']:.3e} with cell averages")

    col = build_column(defaults['dim'], args.n_refine, H=defaults['H'],
                       side_walls=defaults['side_walls'], ic=args.ic)
    p = col.params
    print(f"Mesh: {p['n_vertices']} vertices, {p['n_frozen']} frozen, "
          f"c0 = {p['c0']:.2f} m/s, t_ac = {p['t_ac']:.4f} s, "
          f"reference surface height {p['h_surface']:.6f}")

    n_rec = 200
    cfl = 0.25
    n_steps = int(round(args.n_tac * p['t_ac'] / (cfl * p['dx_min'] / p['c0'])))
    history = StateHistory(fields=['u', 'p'],
                           record_every=max(1, n_steps // n_rec),
                           save_dir=os.path.join(res_dir, 'snapshots'))
    res = run_column(col, methods, n_tac=args.n_tac, cfl=cfl,
                     alpha_art=args.alpha_art, callback=history.callback)
    err = column_errors(col)
    summary = dict(
        case=name, arm=args.arm, ic=args.ic, n_refine=args.n_refine,
        n_vertices=p['n_vertices'], n_steps=res['n_steps'], dt=res['dt'],
        n_tac=args.n_tac, mu=res['mu'], alpha_art=args.alpha_art,
        umax_peak=float(res['umax'].max()), umax_end=float(res['umax'][-1]),
        ke_peak=float(res['ke'].max()), ke_end=float(res['ke'][-1]),
        ke_decay_rate_per_tac=decay_rate(res, col),
        settled_max_a=residual_acceleration(col, res['dudt_fn']),
        static_max_a_nodal=static['nodal'],
        static_max_a_cell_average=static['cell_average'],
        **err,
    )
    print(f"Done: {res['n_steps']} steps, mu = {res['mu']:.1f} Pa s")
    print(f"  max|u| peak {summary['umax_peak']:.4e} -> end "
          f"{summary['umax_end']:.4e} m/s, KE end {summary['ke_end']:.4e} J")
    print(f"  settled force balance max|a| = {summary['settled_max_a']:.4e} "
          "m/s^2")
    print(f"  integrated L2 = {err['l2']:.4e} Pa "
          f"({err['l2'] / err['rho_g_H']:.3e} rho g H), interior "
          f"{err['l2_interior']:.4e} Pa, max|p V - int P dV| = "
          f"{err['max_int']:.4e}")
    print(f"  surface max {err['surface_max']:.6f}, mass drift "
          f"{err['mass_drift']:+.2e}")

    record_methods(os.path.join(res_dir, 'methods.json'), methods, col.HC,
                   extra={k: summary[k] for k in (
                       'case', 'arm', 'ic', 'n_refine', 'n_steps', 'dt',
                       'n_tac', 'mu', 'alpha_art')}
                   | {'eos': 'TaitMurnaghan(n=1, P0=0, K=rho (10 sqrt(g H))^2)',
                      'side_walls': p['side_walls'], 'cfl': res['cfl']})
    with open(os.path.join(res_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
        fh.write('\n')

    _plots(col, res, tag, fig_dir)
    if not args.no_anim:
        _animate(col, history, tag, fig_dir)
    print(f"Outputs: {res_dir}, {fig_dir}")
    return summary


def _plots(col: Column, res: dict[str, Any], tag: str, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ddgclib.analytical import integrated_pressure_error

    p = col.params
    ax_g = p['gravity_axis']
    t = res['t'] / p['t_ac']
    P = reference_pressure(col)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    if np.any(res['ke'] > 0):
        a1.semilogy(t, res['ke'], lw=1)
    a1.set_xlabel('$t / t_{ac}$')
    a1.set_ylabel('kinetic energy [J]')
    a1.grid(True, alpha=0.3)
    a2.semilogy(t, np.maximum(res['umax'], 1e-300), lw=1, color='C3')
    a2.set_xlabel('$t / t_{ac}$')
    a2.set_ylabel('max $|u|$ [m/s]')
    a2.grid(True, alpha=0.3)
    fig.suptitle(f"{tag}: settling ($\\mu$ = {res['mu']:.0f} Pa s, "
                 f"{p['n_vertices']} vertices)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'{tag}_settling.png'), dpi=150)
    plt.close(fig)

    free = col.free
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    y_fine = np.linspace(0.0, p['H'], 200)
    x_ref = np.zeros(col.dim)
    ref = []
    for y in y_fine:
        x_ref[ax_g] = y
        ref.append(P(x_ref))
    a1.plot(ref, y_fine, 'k--', lw=1.5, label='analytical (compressible)')
    a1.plot([v.p for v in col.HC.V], [v.x_a[ax_g] for v in col.HC.V], 'o',
            ms=3, alpha=0.6, label='DDG')
    a1.set_xlabel('p [Pa]')
    a1.set_ylabel('height [m]')
    a1.legend()
    a1.grid(True, alpha=0.3)
    a2.scatter([v.x_a[ax_g] for v in free],
               integrated_pressure_error(col.HC, free, P, dim=col.dim), s=10)
    a2.set_xlabel('height [m]')
    a2.set_ylabel(r'$|p_i V_i - \int P\,dV|$')
    a2.set_yscale('log')
    a2.grid(True, alpha=0.3)
    fig.suptitle(f'{tag}: final state against the analytical profile')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'{tag}_final_profile.png'), dpi=150)
    plt.close(fig)

    if col.dim >= 2:
        from ddgclib.visualization.unified import plot_primal
        fig, _ = plot_primal(col.HC, bV=col.bV, scalar_field='p',
                             title=f'{tag}: pressure', save_path=None,
                             cmap='coolwarm')
        fig.savefig(os.path.join(fig_dir, f'{tag}_mesh_pressure.png'), dpi=150)
        plt.close(fig)
    print(f"  -> fig/{tag}_settling.png, fig/{tag}_final_profile.png")


def _animate(col: Column, history, tag: str, fig_dir: str) -> None:
    from ddgclib.visualization import dynamic_plot_fluid
    path = os.path.join(fig_dir, f'{tag}.mp4')
    try:
        dynamic_plot_fluid(history, col.HC, bV=col.bV, save_path=path)
        print(f"  -> fig/{tag}.mp4")
    except Exception as e:  # noqa: BLE001 - the animation is optional
        print(f"  animation skipped ({type(e).__name__}: {e})")
