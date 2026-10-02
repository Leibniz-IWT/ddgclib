"""Developing Lagrangian Poiseuille flow on the library integrators
(laneH, 2026-10-01).

Shared by ``Hagen_Poiseuile_2D.py``, ``Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py``,
``diagnose_poiseuille.py`` and ``ddgclib/tests/test_case_hagen_poiseuille.py``.
There is no time loop here: every run goes through
``SolverMethods.integrate`` with the preset ``hagen_poiseuille_2D`` /
``hagen_poiseuille_3D`` (or a ``.replace(...)`` arm of it) on the mesh, BCs
and ICs of ``src/_setup.py:setup_poiseuille_developing``.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from cases_dynamic.Hagen_Poiseuile.src._metrics import (
    PlaneFlux, census, fluxes, profile_error,
)
from cases_dynamic.Hagen_Poiseuile.src._setup import (
    setup_poiseuille_developing, wall_report, wall_snapshot,
)

__all__ = ['CASES', 'run_developing', 'run_case']

# Shipped parameters of the two runners (setup kwargs, time step, horizon).
# Development is along the path line (prescribed pressure, no pressure
# solve): time constant t_dev = rho D^2 / (pi^2 mu) in 2D and
# rho R^2 / (2.405^2 mu) in 3D, development length about 5 t_dev U_max.
CASES: dict[str, dict[str, Any]] = {
    # Re_D = 10: t_dev 10.1 s, developed from x = 8
    'hagen_poiseuille_2D': dict(
        dim=2, L=12.0, D=1.0, U_avg=0.1, rho=1.0, mu=0.01, n_refine=2,
        inlet_buffer=1.0, outlet_buffer=1.0, cdist=1e-10,
        dt=0.05, t_end=120.0),
    # Re_D = 2: t_dev 0.86 s, developed from x = 1
    'hagen_poiseuille_3D': dict(
        dim=3, L=4.0, D=1.0, U_avg=0.1, rho=1.0, mu=0.05, n_refine=2,
        inlet_buffer=1.0, outlet_buffer=1.0, cdist=1e-10,
        dt=0.01, t_end=10.0),
}


def run_developing(methods, *, dim: int, L: float, mu: float, n_refine: int,
                   dt: float, n_steps: int, window: tuple | None = None,
                   sample_every: int = 10, callback: Callable | None = None,
                   custom: Callable | None = None,
                   **setup_kw: Any) -> dict[str, Any]:
    """Run the developing flow with *methods* and measure it.

    *window* ``(x0, x1)`` is where the velocity is compared with the
    developed profile (default: the downstream half of the channel).
    Every *sample_every* steps the profile error, the volume flux in the
    window and the vertex census are recorded (``u_cross``, the largest
    transverse velocity, is taken over the free vertices of the window
    like the profile error, not over the whole mesh).  The mass carried through
    the inlet plane 0 and the outlet plane ``L`` is accumulated over the
    last whole inlet periods (``period / U_avg`` each: the inlet feeds
    one unit cell per period, so a shorter count depends on which columns
    happen to cross) that fit into the second half of the run, else into
    the whole run; with less than one period the fluxes are NaN.
    *callback* ``(step, t, HC, bV, diagnostics)`` is called every step
    (e.g. ``StateHistory.callback``).  *custom* is the retopology function
    of a ``connectivity='custom'`` arm.

    Returns a dict with the mesh (``HC``, ``bV``, ``params``,
    ``wall_criterion``), the time series (``t``, ``l2``, ``u_max``,
    ``u_cross``, ``q_window``, ``n_total``, ``n_channel``, ``n_outside``)
    and the end values (``profile``, ``mass_flux_in``, ``mass_flux_out``
    in units of ``rho U_avg A``, ``flux_periods``, ``census``,
    ``walls``).
    """
    HC, bV, bc_set, wall, params = setup_poiseuille_developing(
        dim=dim, L=L, mu=mu, n_refine=n_refine, **setup_kw)
    if window is None:
        window = (0.5 * L, L)
    axis = params['flow_axis']
    walls_start = wall_snapshot(HC, wall)
    flux = PlaneFlux(axis, [0.0, L])
    steps_per_period = params['period'] / (params['U_avg'] * dt)
    n_periods = (int((n_steps // 2) / steps_per_period)
                 or int(n_steps / steps_per_period))
    flux_start = n_steps - int(round(n_periods * steps_per_period))
    series: dict[str, list] = {k: [] for k in (
        't', 'l2', 'u_max', 'u_cross', 'q_window', 'n_total', 'n_channel',
        'n_outside')}

    def _callback(step, t, HC, bV=None, diagnostics=None):
        if step == flux_start:
            flux.reset()
        flux.update(HC)
        if (step + 1) % sample_every == 0:
            err = profile_error(HC, bV, params, *window)
            n = census(HC, bV, params)
            series['t'].append(t)
            series['l2'].append(err['l2'])
            series['u_max'].append(err['u_max'])
            series['u_cross'].append(err['u_cross'])
            series['q_window'].append(fluxes(HC, params, *window)['volume'])
            series['n_total'].append(n['total'])
            series['n_channel'].append(n['channel'])
            series['n_outside'].append(n['outside'])
        if callback is not None:
            callback(step, t, HC, bV, diagnostics)

    dudt_fn = methods.dudt_fn(HC, mu=mu)
    t_end = methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                              bc_set=bc_set, boundary_filter=wall,
                              callback=_callback, custom=custom)

    ref = (params['rho'] * params['area'] * n_periods * params['period']
           or float('nan'))
    out: dict[str, Any] = {k: np.array(v) for k, v in series.items()}
    out.update(
        HC=HC, bV=bV, params=params, wall_criterion=wall, dudt_fn=dudt_fn,
        t_end=t_end, dt=dt, n_steps=n_steps, window=window,
        profile=profile_error(HC, bV, params, *window),
        mass_flux_in=flux.mass[0.0] / ref,
        mass_flux_out=flux.mass[float(L)] / ref, flux_periods=n_periods,
        census=census(HC, bV, params),
        walls=wall_report(HC, bV, walls_start),
    )
    return out


# ----------------------------------------------------------------------
# runner body shared by the two scripts
# ----------------------------------------------------------------------

def run_case(name: str, argv: list[str] | None = None) -> dict[str, Any]:
    """Run one shipped case through its preset; write
    ``results/<name>[_<tag>]/`` (snapshots, ``methods.json``,
    ``summary.json``, final state) and ``fig/`` next to the runner.
    Returns the summary dict."""
    import argparse
    import json
    import pickle
    import time

    from ddgclib.data import StateHistory, save_state
    from ddgclib.methods import AXES, PRESETS, record_methods

    defaults = CASES[name]
    dim = defaults['dim']
    ap = argparse.ArgumentParser(
        description=f"{name} (preset PRESETS[{name!r}])")
    ap.add_argument('--n-refine', type=int, default=defaults['n_refine'])
    ap.add_argument('--L', type=float, default=defaults['L'],
                    help='channel length in unit cells')
    ap.add_argument('--mu', type=float, default=defaults['mu'])
    ap.add_argument('--dt', type=float, default=defaults['dt'])
    ap.add_argument('--t-end', type=float, default=defaults['t_end'])
    ap.add_argument('--steps', type=int, default=None,
                    help='number of steps (overrides --t-end)')
    arms = ('viscous_flux', 'pressure_flux', 'frozen_set', 'backend')
    for axis in arms:
        ap.add_argument('--' + axis.replace('_', '-'), default=None,
                        choices=[o.key for o in AXES[axis].options
                                 if o.key is not None],
                        help=f'A/B arm: the preset with {axis} replaced')
    ap.add_argument('--workers', type=int, default=None,
                    help='dudt worker processes (preset: serial)')
    ap.add_argument('--tag', default='',
                    help='write to results/<name>_<tag>/ (for an arm)')
    ap.add_argument('--headless', action='store_true',
                    help='accepted for scripted runs (nothing here blocks)')
    ap.add_argument('--no-anim', action='store_true')
    args = ap.parse_args(argv)

    methods = PRESETS[name]
    changes = {axis: getattr(args, axis) for axis in arms
               if getattr(args, axis) is not None}
    if args.workers is not None:
        changes['workers'] = args.workers if args.workers > 1 else None
    if changes:
        methods = methods.replace(**changes)

    case_dir = 'Hagen_Poiseuile' if dim == 2 else 'Hagen_Poiseuile_3D'
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), case_dir)
    tag = name if not args.tag else f'{name}_{args.tag}'
    fig_dir = os.path.join(here, 'fig')
    res_dir = os.path.join(here, 'results', tag)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    print("=" * 64)
    print(methods.describe())
    print("=" * 64)

    n_steps = (args.steps if args.steps is not None
               else int(round(args.t_end / args.dt)))
    n_rec = 100
    history = StateHistory(fields=['u', 'p'],
                           record_every=max(1, n_steps // n_rec),
                           save_dir=os.path.join(res_dir, 'snapshots'))
    setup_kw = {k: defaults[k] for k in (
        'D', 'U_avg', 'rho', 'inlet_buffer', 'outlet_buffer', 'cdist')}
    t_wall = time.perf_counter()

    def callback(step, t, HC, bV=None, diagnostics=None):
        history.callback(step, t, HC, bV, diagnostics)
        if (step + 1) % max(1, n_steps // 20) == 0:
            print(f"  step {step + 1:>6d}/{n_steps}  t = {t:8.3f}  "
                  f"{sum(1 for _ in HC.V)} vertices  "
                  f"{time.perf_counter() - t_wall:6.0f} s", flush=True)

    res = run_developing(methods, dim=dim, L=args.L, mu=args.mu,
                         n_refine=args.n_refine, dt=args.dt, n_steps=n_steps,
                         sample_every=max(1, n_steps // 400),
                         callback=callback, **setup_kw)
    p = res['params']
    HC, bV = res['HC'], res['bV']

    tail = slice(len(res['t']) * 3 // 4, None)     # last quarter of the run
    summary = dict(
        case=name, changes=changes, dim=dim, n_refine=args.n_refine,
        L=args.L, mu=args.mu, Re_D=p['rho'] * p['U_avg'] * p['D'] / args.mu,
        G=p['G'], U_avg=p['U_avg'], U_max_analytical=p['U_max'],
        t_dev=p['t_dev'], dt=args.dt, n_steps=n_steps, t_end=res['t_end'],
        window=list(res['window']),
        l2_end=res['profile']['l2'],
        l2_tail_mean=float(np.mean(res['l2'][tail])),
        l2_tail_max=float(np.max(res['l2'][tail])),
        u_max_end=res['profile']['u_max'],
        u_cross_max=float(np.max(res['u_cross'])),
        volume_flux_window_tail_mean=float(np.mean(res['q_window'][tail])),
        # NaN (run shorter than one inlet period) is stored as null
        mass_flux_in=res['mass_flux_in'] if res['flux_periods'] else None,
        mass_flux_out=res['mass_flux_out'] if res['flux_periods'] else None,
        flux_periods=res['flux_periods'], fluid_fraction=p['fluid_fraction'],
        n_vertices_start=int(res['n_total'][0]),
        n_vertices_end=int(res['n_total'][-1]),
        n_channel_min=int(np.min(res['n_channel'])),
        n_channel_max=int(np.max(res['n_channel'])),
        n_channel_end=int(res['n_channel'][-1]),
        n_outside_max=int(np.max(res['n_outside'])),
        census_end=res['census'], walls=res['walls'],
    )
    print(f"Done: {n_steps} steps to t = {res['t_end']:.2f} "
          f"({res['t_end'] / p['t_dev']:.1f} t_dev)")
    print(f"  profile on x in {summary['window']}: l2 = "
          f"{summary['l2_end']:.4e} (last quarter: mean "
          f"{summary['l2_tail_mean']:.4e}, max {summary['l2_tail_max']:.4e}); "
          f"u_max {summary['u_max_end']:.5f} (analytical {p['U_max']:.5f}); "
          f"largest transverse velocity in the window "
          f"{summary['u_cross_max']:.2e}")
    print(f"  mass flux through x = 0 / x = L over the last "
          f"{summary['flux_periods']} inlet periods, in rho U_avg A: "
          f"{res['mass_flux_in']:.4f} / {res['mass_flux_out']:.4f} "
          f"(share of the mass that is not in wall cells: "
          f"{p['fluid_fraction']:.4f}); volume flux in the window "
          f"{summary['volume_flux_window_tail_mean']:.4f}")
    print(f"  vertices {summary['n_vertices_start']} -> "
          f"{summary['n_vertices_end']}, in the channel "
          f"{summary['n_channel_min']} to {summary['n_channel_max']}, "
          f"outside the walls at most {summary['n_outside_max']}; walls "
          f"{res['walls']['n_frozen']} of {res['walls']['n_wall_start']} "
          f"frozen, {res['walls']['n_moved']} moved")

    record_methods(os.path.join(res_dir, 'methods.json'), methods, HC,
                   extra={k: summary[k] for k in (
                       'case', 'changes', 'n_refine', 'L', 'mu', 'Re_D', 'G',
                       'dt', 'n_steps')}
                   | {'inlet_buffer': p['inlet_buffer'],
                      'outlet_buffer': p['outlet_buffer'],
                      'pressure': 'prescribed G (L - x), nodal, re-imposed '
                                  'every step (DirichletPressureBC)',
                      'mass': 'rho * dual volume of the periodic tiling',
                      # the reported edge_area_source is what the mesh
                      # carries; the simplex-gradient fluxes do not read it
                      'forces_read_dual_face_areas': not (
                          methods.pressure_flux == methods.viscous_flux
                          == 'simplex_gradient')})
    with open(os.path.join(res_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
        fh.write('\n')
    stem = f'hp{dim}d'       # names the visualize_hp*.py scripts read
    save_state(HC, bV, t=res['t_end'], fields=['u', 'p', 'm'],
               path=os.path.join(res_dir, f'{stem}_final_state.json'),
               extra_meta={'case': name, 'mu': args.mu, 'G': p['G'],
                           'L': args.L, 'n_refine': args.n_refine})
    with open(os.path.join(res_dir, f'{stem}_history.pkl'), 'wb') as fh:
        pickle.dump(history, fh)

    _plots(res, tag, fig_dir)
    if dim == 2 and not args.no_anim:
        _animate(res, history, tag, fig_dir)
    print(f"Outputs: {res_dir}, {fig_dir}")
    return summary


def _plots(res: dict[str, Any], tag: str, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = res['params']
    HC, bV = res['HC'], res['bV']
    axis, dim = p['flow_axis'], p['dim']
    x0, x1 = res['window']
    free = [v for v in HC.V if v not in bV]

    def cross(v):       # wall-normal coordinate: y in 2D, r in 3D
        return (v.x_a[1] if dim == 2
                else float(np.hypot(v.x_a[0], v.x_a[1])))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    s = np.linspace(0.0, p['D'] if dim == 2 else p['R'], 200)
    probe = np.zeros(dim)
    ref = []
    for c in s:
        probe[1 if dim == 2 else 0] = c
        ref.append(p['poiseuille_ic'].analytical_velocity(probe))
    a1.plot(ref, s, 'k-', lw=1.5, label='analytical (developed)')
    win = [v for v in free if x0 <= v.x_a[axis] <= x1]
    a1.plot([v.u[axis] for v in win], [cross(v) for v in win], 'o', ms=4,
            alpha=0.7, label=f'DDG, {x0:g} <= x <= {x1:g}')
    a1.set_xlabel('axial velocity [m/s]')
    a1.set_ylabel('y [m]' if dim == 2 else 'r [m]')
    a1.legend()
    a1.grid(True, alpha=0.3)
    chan = [v for v in free if -p['inlet_buffer'] <= v.x_a[axis] <= p['L']]
    sc = a2.scatter([v.x_a[axis] for v in chan], [v.u[axis] for v in chan],
                    c=[cross(v) for v in chan], s=10, cmap='viridis')
    fig.colorbar(sc, ax=a2, label='y [m]' if dim == 2 else 'r [m]')
    a2.axhline(p['U_max'], color='k', ls='--', lw=1)
    a2.axvline(0.0, color='0.5', lw=1)
    a2.set_xlabel('x along the flow [m] (inlet buffer: x < 0)')
    a2.set_ylabel('axial velocity [m/s]')
    a2.grid(True, alpha=0.3)
    fig.suptitle(f"{tag}: t = {res['t_end']:.1f} s "
                 f"({res['t_end'] / p['t_dev']:.1f} t_dev), l2 = "
                 f"{res['profile']['l2']:.2e}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'{tag}_profile.png'), dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.semilogy(res['t'], res['l2'], lw=1)
    a1.set_xlabel('t [s]')
    a1.set_ylabel('dual-volume weighted l2 error of u in the window')
    a1.grid(True, alpha=0.3)
    a2.plot(res['t'], res['n_total'], lw=1, label='all')
    a2.plot(res['t'], res['n_channel'], lw=1, label='free, in the channel')
    a2.set_xlabel('t [s]')
    a2.set_ylabel('vertices')
    a2.legend()
    a2.grid(True, alpha=0.3)
    fig.suptitle(f'{tag}: development and vertex count')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'{tag}_development.png'), dpi=150)
    plt.close(fig)

    if dim == 2:
        from ddgclib.visualization.unified import plot_primal
        fig, ax = plot_primal(HC, bV=bV, scalar_field='p', vector_field='u',
                              save_path=None,
                              title=f'{tag}: mesh, pressure and velocity')
        xlim, ylim = _limits(p)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        fig.set_size_inches(1.2 * (xlim[1] - xlim[0]) + 2.0, 3.2)
        fig.savefig(os.path.join(fig_dir, f'{tag}_mesh.png'), dpi=150)
        plt.close(fig)
    print(f"  -> fig/{tag}_profile.png, fig/{tag}_development.png")


def _limits(p: dict[str, Any]) -> tuple[tuple, tuple]:
    """Plot window of the 2D channel with its two buffers."""
    return ((-p['inlet_buffer'] - 0.1, p['L'] + p['outlet_buffer'] + 0.1),
            (-0.1 * p['D'], 1.1 * p['D']))


def _animate(res: dict[str, Any], history, tag: str, fig_dir: str) -> None:
    from ddgclib.visualization import dynamic_plot_fluid
    path = os.path.join(fig_dir, f'{tag}.mp4')
    xlim, ylim = _limits(res['params'])
    try:
        dynamic_plot_fluid(history, res['HC'], bV=res['bV'], save_path=path,
                           xlim=xlim, ylim=ylim)
        print(f"  -> fig/{tag}.mp4")
    except Exception as e:  # noqa: BLE001 - the animation is optional
        print(f"  animation skipped ({type(e).__name__}: {e})")
