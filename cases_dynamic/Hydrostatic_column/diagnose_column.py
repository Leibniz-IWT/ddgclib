"""Measurements behind the hydrostatic presets (laneP, 2026-10-01).

Every arm is a ``SolverMethods`` (a preset or ``preset.replace(...)``) run
through ``src/_column.py``; nothing here has its own time loop.

    python cases_dynamic/Hydrostatic_column/diagnose_column.py arms
    python cases_dynamic/Hydrostatic_column/diagnose_column.py convergence
    python cases_dynamic/Hydrostatic_column/diagnose_column.py viscosity
    python cases_dynamic/Hydrostatic_column/diagnose_column.py free_surface
    python cases_dynamic/Hydrostatic_column/diagnose_column.py growth
    python cases_dynamic/Hydrostatic_column/diagnose_column.py periodic
    python cases_dynamic/Hydrostatic_column/diagnose_column.py remap

``arms``         both connectivity arms of every case (the preset and the
                 reconnecting ``delaunay_material`` + remap arm), plus the
                 measured-bad ones (convex-hull Delaunay + remap, 3D
                 ``dual_only``), for both initial conditions.
``convergence``  integrated pressure error of the settled column against
                 refinement (equilibrium start).
``viscosity``    how much artificial viscosity the ``'drop'`` start needs
                 and what happens without it (also with
                 ``density_diffusion``).
``free_surface`` the settled 2D column linearised (no-slip, free-slip and,
                 as a control, a closed lid): force against the energy
                 gradient, symmetry of the stiffness matrix on closed and
                 open fans, saddle and flutter growth rates, the largest
                 rate against viscosity, the fastest mode followed in a
                 real run, and a sloshing seed in both connectivity arms
                 (``--n-refine`` selects the mesh, default 3).

``growth``       saddle and flutter growth rates of the linearised column
                 against refinement (2, 3, 4).
``periodic``     why the column is not run on ``connectivity='periodic'``:
                 seam dual volumes, dual-face closure and simplex count of
                 the library periodic retopology, and a short run.
``remap``        why convex-hull Delaunay + remap fails on the free surface:
                 each rebuild (convex hull, material boundary) with and
                 without the global mass rescale of the remap, the offset
                 ``K (s - 1)`` of that rescale, two controls (closed lid;
                 no gravity with a velocity seed), and how well the
                 material rebuild keeps the domain volume in 2D and 3D.

Results are printed and written as JSON to ``--out`` (default
``results/diagnose``).  See
docs_temp/debug_session/laneP-hydrostatic-library-integrators.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from cases_dynamic.Hydrostatic_column.src._column import (  # noqa: E402
    CASES, build_column, column_errors, decay_rate, refresh_pressure,
    remap_arm, residual_acceleration, run_column)
from ddgclib.methods import PRESETS  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


class _Abort(Exception):
    pass


def _methods(case: str, arm: str):
    m = PRESETS[case]
    if arm == 'preset':
        return m
    if arm == 'remap':
        return remap_arm(m)
    if arm == 'convex_remap':
        return m.replace(connectivity='delaunay', remap='conservative',
                         redistribute_mass=True)
    if arm == 'dual_only':
        return m.replace(connectivity='dual_only')
    raise ValueError(arm)


def _envelope(res, t_ac, t_at, window=4.0):
    """max |u| over the last *window* acoustic times before *t_at* (one
    period of the fundamental mode, so the value is not a phase sample)."""
    t = res['t'] / t_ac
    sel = (t > t_at - window) & (t <= t_at + 1e-9)
    return float(res['umax'][sel].max()) if sel.any() else float('nan')


def run_arm(job: dict) -> dict:
    """One (case, arm, ic, refinement, viscosity) run -> summary row."""
    case, arm = job['case'], job['arm']
    kw = CASES[case]
    n_refine = job.get('n_refine', kw['n_refine'])
    col = build_column(kw['dim'], n_refine, H=kw['H'],
                       side_walls=kw['side_walls'], ic=job.get('ic', 'drop'))
    methods = _methods(case, arm)
    if job.get('density_diffusion') is not None:
        methods = methods.replace(density_diffusion=job['density_diffusion'])
    c0 = col.params['c0']
    state = {'t': 0.0}

    def guard(step, t, HC, bV=None, diagnostics=None):
        state['t'] = t
        if step % 20 == 0:
            if max(float(np.linalg.norm(v.u)) for v in HC.V) > c0:
                raise _Abort

    row = dict(job, n_refine=n_refine, n_vertices=col.params['n_vertices'],
               config={k: v for k, v in methods.to_dict().items()
                       if k in ('dim', 'integrator', 'connectivity', 'remap',
                                'redistribute_mass')})
    t0 = time.time()
    res = None
    try:
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            res = run_column(col, methods, n_tac=job['n_tac'],
                             alpha_art=job.get('alpha_art', 0.5),
                             mu=job.get('mu'), callback=guard)
        row['status'] = 'ok'
    except _Abort:
        row['status'] = (f"BLOW-UP |u| > c0 at "
                         f"{state['t'] / col.params['t_ac']:.2f} t_ac")
    except Exception as e:  # noqa: BLE001 - reported in the table
        row['status'] = f"ERROR {type(e).__name__}: {str(e)[:80]}"
    row['wall_s'] = round(time.time() - t0, 1)
    if res is not None:
        t_ac = col.params['t_ac']
        row.update(
            n_steps=res['n_steps'], mu=res['mu'],
            umax_peak=float(res['umax'].max()),
            umax_env={str(int(tt)): _envelope(res, t_ac, tt)
                      for tt in (10, 20, 40, 100, 200) if tt <= job['n_tac']},
            umax_end_env=_envelope(res, t_ac, job['n_tac']),
            ke_end=float(res['ke'][-1]),
            ke_decay_per_tac=decay_rate(res, col),
            settled_max_a=residual_acceleration(col, res['dudt_fn']),
        )
        row.update(column_errors(col))
    return row


def _pool(jobs, procs):
    with get_context('fork').Pool(min(procs, len(jobs))) as pool:
        return pool.map(run_arm, jobs, chunksize=1)


def _fmt(x, p=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '-'
    return f"{x:.{p}e}" if isinstance(x, float) else str(x)


def _table(rows, cols):
    print(' | '.join(c for c, _ in cols))
    for r in rows:
        print(' | '.join(_fmt(f(r)) for _, f in cols))


def _save(out, name, rows):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'{name}.json'), 'w') as fh:
        json.dump(rows, fh, indent=2, default=float)
        fh.write('\n')


def main_arms(out, n_tac, procs, only=None):
    jobs = []
    for case, kw in CASES.items():
        if only and case not in only:
            continue
        arms = ['preset']
        if kw['dim'] >= 2:
            arms += ['remap', 'convex_remap']
        if kw['dim'] == 3:
            arms += ['dual_only']
        for arm in arms:
            for ic in ('drop', 'equilibrium'):
                jobs.append(dict(case=case, arm=arm, ic=ic, n_tac=n_tac))
    rows = _pool(jobs, procs)
    _save(out, f'arms_{int(n_tac)}', rows)
    _table(rows, [
        ('case', lambda r: r['case']), ('arm', lambda r: r['arm']),
        ('ic', lambda r: r['ic']), ('n_v', lambda r: r['n_vertices']),
        ('connectivity/remap', lambda r: f"{r['config']['connectivity']}/"
                                         f"{r['config']['remap']}"),
        ('status', lambda r: r['status']),
        ('umax_peak', lambda r: r.get('umax_peak')),
        ('env(10)', lambda r: r.get('umax_env', {}).get('10')),
        ('env(end)', lambda r: r.get('umax_end_env')),
        ('KE decay/t_ac', lambda r: r.get('ke_decay_per_tac')),
        ('max|a| end', lambda r: r.get('settled_max_a')),
        ('L2 [Pa]', lambda r: r.get('l2')),
        ('L2 interior', lambda r: r.get('l2_interior')),
        ('max|pV-int|', lambda r: r.get('max_int')),
        ('mass drift', lambda r: r.get('mass_drift')),
        ('wall s', lambda r: r['wall_s']),
    ])


def main_convergence(out, n_tac, procs):
    levels = {'hydrostatic_1D': (3, 4, 5, 6), 'hydrostatic_2D': (2, 3, 4),
              'hydrostatic_2D_periodic': (2, 3, 4), 'hydrostatic_3D': (1, 2)}
    jobs = [dict(case=case, arm='preset', ic='equilibrium', n_tac=n_tac,
                 n_refine=n) for case, ns in levels.items() for n in ns]
    rows = _pool(jobs, procs)
    _save(out, 'convergence', rows)
    _table(rows, [
        ('case', lambda r: r['case']), ('n_refine', lambda r: r['n_refine']),
        ('n_v', lambda r: r['n_vertices']), ('status', lambda r: r['status']),
        ('umax_peak', lambda r: r.get('umax_peak')),
        ('env(end)', lambda r: r.get('umax_end_env')),
        ('L2 [Pa]', lambda r: r.get('l2')),
        ('L2/rho g H', lambda r: r.get('l2') / r['rho_g_H']
         if r.get('l2') is not None else None),
        ('L2 interior', lambda r: r.get('l2_interior')),
        ('max|pV-int|', lambda r: r.get('max_int')),
        ('wall s', lambda r: r['wall_s']),
    ])
    for case in levels:
        sub = [r for r in rows if r['case'] == case and r['status'] == 'ok']
        for a, b in zip(sub, sub[1:]):
            print(f"  {case} refine {a['n_refine']} -> {b['n_refine']}: "
                  f"L2 ratio {a['l2'] / b['l2']:.2f}, interior "
                  f"{a['l2_interior'] / b['l2_interior']:.2f}, max|pV-int| "
                  f"{a['max_int'] / b['max_int']:.2f}")


def main_viscosity(out, n_tac, procs):
    jobs = [dict(case=case, arm='preset', ic=ic, n_tac=n_tac, alpha_art=a)
            for case in ('hydrostatic_1D', 'hydrostatic_2D',
                         'hydrostatic_2D_periodic')
            for ic in ('drop', 'equilibrium')
            for a in (0.0, 0.05, 0.5)]
    for j in jobs:                       # physical viscosity of water
        if j['alpha_art'] == 0.0:
            j['mu'] = 1e-3
    # density diffusion instead of viscosity: does it hold the no-slip drop?
    jobs += [dict(case='hydrostatic_2D', arm='preset', ic='drop', n_tac=n_tac,
                  alpha_art=0.0, mu=1e-3, density_diffusion=d)
             for d in (0.05, 0.1)]
    rows = _pool(jobs, procs)
    _save(out, 'viscosity', rows)
    for r in rows:
        kw = CASES[r['case']]
        # amplitude decay of the fundamental mode k = pi / (2 H) under the
        # diffusion-form viscous term: nu k^2 / 2; KE decays twice as fast
        col_c0 = 10.0 * np.sqrt(9.81 * kw['H'])
        nu = r.get('mu', 0.0) / 1000.0
        r['ke_decay_theory'] = nu * (np.pi / (2 * kw['H']))**2 * kw['H'] / col_c0
    _table(rows, [
        ('case', lambda r: r['case']), ('ic', lambda r: r['ic']),
        ('alpha_art', lambda r: r['alpha_art']), ('mu', lambda r: r.get('mu')),
        ('density_diffusion', lambda r: r.get('density_diffusion')),
        ('status', lambda r: r['status']),
        ('umax_peak', lambda r: r.get('umax_peak')),
        ('env(10)', lambda r: r.get('umax_env', {}).get('10')),
        ('env(end)', lambda r: r.get('umax_end_env')),
        ('KE decay/t_ac', lambda r: r.get('ke_decay_per_tac')),
        ('theory (1D fundamental)', lambda r: r['ke_decay_theory']),
        ('wall s', lambda r: r['wall_s']),
    ])


# ----------------------------------------------------------------------
# free surface
# ----------------------------------------------------------------------

def _dofs(col):
    """(vertex, axis) of every unconstrained degree of freedom."""
    return [(v, d) for v in col.free for d in range(col.dim)
            if d not in col.slip_axes.get(id(v), ())]


def _accelerations(col, dudt_fn, dofs):
    refresh_pressure(col)
    acc = {}
    out = np.empty(len(dofs))
    for i, (v, d) in enumerate(dofs):
        if id(v) not in acc:
            acc[id(v)] = dudt_fn(v)
        out[i] = acc[id(v)][d]
    return out


def _jacobian(col, dudt_fn, dofs, h=1e-7):
    """J = d a / d x of the unconstrained degrees of freedom by central
    differences (the column is left as it was)."""
    n = len(dofs)
    J = np.zeros((n, n))
    for j, (v, d) in enumerate(dofs):
        x0 = np.array(v.x_a[:col.dim])
        cols = []
        for sgn in (+1, -1):
            x = x0.copy()
            x[d] += sgn * h
            col.HC.V.move(v, tuple(x))
            cols.append(_accelerations(col, dudt_fn, dofs))
        J[:, j] = (cols[0] - cols[1]) / (2 * h)
        col.HC.V.move(v, tuple(x0))
    refresh_pressure(col)
    return J


def _growth_rates(lam):
    """Growth rates [1/s] of x'' = J x from the eigenvalues of J.

    A real positive eigenvalue is a saddle direction (monotone growth
    sqrt(lambda)); a complex pair is flutter (oscillation with growth
    Re sqrt(lambda)); real negative eigenvalues are neutral oscillations.
    """
    lam = np.asarray(lam, dtype=complex)
    root = np.sqrt(lam)
    root = np.where(root.real < 0, -root, root)
    is_complex = np.abs(lam.imag) > 1e-9 * np.abs(lam).max()
    k = int(np.argmax(np.where(is_complex, root.real, -1.0)))
    return dict(
        saddle_per_s=float(np.sqrt(max(lam.real[~is_complex].max(), 0.0))),
        n_saddle=int(((lam.real > 1e-6) & ~is_complex).sum()),
        flutter_per_s=float(root.real[is_complex].max())
        if is_complex.any() else 0.0,
        flutter_omega_rad_s=float(abs(root.imag[k])) if is_complex.any() else 0.0,
        n_complex=int(is_complex.sum()),
        max_imag=float(np.abs(lam.imag).max()),
    )


def _energy(col):
    """EOS internal energy + gravitational potential energy [J]."""
    p = col.params
    K, rho0, g, ax = p['K'], p['rho'], p['g'], p['gravity_axis']
    e = 0.0
    for v in col.HC.V:
        r = v.m / v.dual_vol
        e += v.m * K * (np.log(r / rho0) / rho0 + 1.0 / r - 1.0 / rho0)
        e += v.m * g * v.x_a[ax]
    return e


_KINDS = {
    # name -> build_column kwargs of the settled 2D column
    'noslip': dict(side_walls='noslip'),
    'freeslip': dict(side_walls='freeslip'),
    'closed_lid': dict(side_walls='noslip', free_surface=False),
}
_MUS = (0.0, 1e-3, 1e-1, 1.0, 10.0)


def _linearise(job):
    """Settle the 2D column, then: (1) force against the energy gradient,
    (2) symmetry of the stiffness matrix by block, (3) spectrum of the
    inviscid linearisation and its growth rate against viscosity, (4) the
    most unstable mode followed in a real run."""
    from ddgclib.methods import SolverMethods
    kind, n_tac = job['kind'], job['n_tac']
    methods = SolverMethods(dim=2, connectivity='dual_only')
    col = build_column(2, job.get('n_refine', 3), ic='equilibrium',
                       **_KINDS[kind])
    dim, p = col.dim, col.params
    g_vec = np.array([0.0, -p['g']])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_column(col, methods, n_tac=80.0)          # settle on mu_art
        mu_art = 0.5 * p['rho'] * p['c0'] * p['dx_mean']
        dudt0 = methods.dudt_fn(col.HC, mu=0.0, pressure_model=col.eos,
                                body_force=g_vec)
        dudt1 = methods.dudt_fn(col.HC, mu=1.0, pressure_model=col.eos,
                                body_force=g_vec)
    for v in col.HC.V:
        v.u[:] = 0.0
    refresh_pressure(col)
    free = col.free
    top = max(v.x_a[1] for v in col.HC.V)
    surf_ids = ({id(v) for v in free if v.x_a[1] > top - 0.5 * p['dx_min']}
                if p['free_surface'] else set())
    out = dict(kind=kind, n_refine=p['n_refine'],
               settled_max_a=residual_acceleration(col, dudt0),
               g_over_c0_per_s=p['g'] / p['c0'], t_ac=p['t_ac'])

    # (1) F against -dE/dx by central differences (unconstrained components)
    h = 1e-7
    mism = {'surface': [], 'interior': []}
    probe = ([v for v in free if id(v) in surf_ids]
             + [v for v in free if id(v) not in surf_ids][:9])
    for v in probe:
        comps = [d for d in range(dim)
                 if d not in col.slip_axes.get(id(v), ())]
        F = dudt0(v)[:dim] * v.m
        x0 = np.array(v.x_a[:dim])
        F_e = np.zeros(dim)
        for d in comps:
            es = []
            for sgn in (+1, -1):
                x = x0.copy()
                x[d] += sgn * h
                col.HC.V.move(v, tuple(x))
                refresh_pressure(col)
                es.append(_energy(col))
            F_e[d] = -(es[0] - es[1]) / (2 * h)
        col.HC.V.move(v, tuple(x0))
        refresh_pressure(col)
        mism['surface' if id(v) in surf_ids else 'interior'].append(
            float(np.linalg.norm((F - F_e)[comps])) / (v.m * p['g']))
    out['force_minus_energy_gradient_over_weight'] = {
        k: (max(x) if x else None) for k, x in mism.items()}
    out['surface_pressure_Pa'] = ([min(v.p for v in free if id(v) in surf_ids),
                                   max(v.p for v in free if id(v) in surf_ids)]
                                  if surf_ids else None)

    # (2) + (3) Jacobian J = d a / d x of the inviscid column, x'' = J x
    dofs = _dofs(col)
    n = len(dofs)
    J = _jacobian(col, dudt0, dofs, h)
    C1 = np.zeros((n, n))                  # d a / d u per unit viscosity
    base = _accelerations(col, dudt1, dofs)
    for j, (v, d) in enumerate(dofs):
        v.u[d] = 1.0
        C1[:, j] = _accelerations(col, dudt1, dofs) - base
        v.u[d] = 0.0
    M = np.array([v.m for v, _ in dofs])
    K = -(J * M[:, None])
    A = K - K.T
    open_fan = np.array([id(v) in surf_ids or id(v) in col.slip_axes
                         for v, _ in dofs])
    normK = float(np.linalg.norm(K))
    lam, vec = np.linalg.eig(J)
    root = np.sqrt(lam.astype(complex))
    root = np.where(root.real < 0, -root, root)
    i = int(np.argmax(root.real))          # fastest mode: saddle or flutter
    rate = complex(root[i])
    sigma = float(np.sqrt(max(lam.real.max(), 0.0)))
    out.update(
        n_dof=n, n_open_fan_dof=int(open_fan.sum()),
        stiffness_asymmetry=float(np.linalg.norm(A)) / normK,
        asymmetry_closed_closed=float(np.linalg.norm(
            A[np.ix_(~open_fan, ~open_fan)])) / normK,
        asymmetry_open_rows=float(np.sqrt(
            np.linalg.norm(A[np.ix_(open_fan, ~open_fan)])**2 * 2
            + np.linalg.norm(A[np.ix_(open_fan, open_fan)])**2)) / normK,
        lambda_max=float(lam.real.max()), lambda_min=float(lam.real.min()),
        lambda_max_imag=float(np.abs(lam.imag).max()),
        n_positive=int((lam.real > 1e-6).sum()),
        growth_per_s=sigma, growth_per_tac=sigma * p['t_ac'],
        growth_over_g_c0=sigma * p['c0'] / p['g'],
        **_growth_rates(lam),
    )
    out['growth_per_s_vs_mu'] = {}
    for mu in _MUS + (0.1 * mu_art, mu_art):
        ev = np.linalg.eigvals(np.block([[np.zeros((n, n)), np.eye(n)],
                                         [J, mu * C1]]))
        out['growth_per_s_vs_mu'][f"{mu:.4g}"] = float(ev.real.max())

    # (4) the fastest-growing mode in a real run, no viscosity.  With zero
    #     initial velocity x - x_eq = Re(z(t) w), z(t) = eps cosh(rate t);
    #     for a complex pair (flutter) |z| is read from the plane spanned
    #     by Re w and Im w.
    w = vec[:, i] / np.abs(vec[:, i].real).max()
    is_flutter = abs(rate.imag) > 0.0
    basis = (np.column_stack([w.real, w.imag]) if is_flutter
             else w.real[:, None])
    eps = 1e-7
    x_eq = np.array([v.x_a[d] for v, d in dofs])
    for (v, d), dx in zip(dofs, eps * w.real):
        x = np.array(v.x_a[:dim])
        x[d] += dx
        col.HC.V.move(v, tuple(x))
    q = []

    def cb(step, t, HC, bV=None, diagnostics=None):
        if step % 50 == 0:
            x = np.array([v.x_a[d] for v, d in dofs])
            z = np.linalg.lstsq(basis, x - x_eq, rcond=None)[0]
            q.append((t, float(np.linalg.norm(z))))

    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        run_column(col, methods, n_tac=n_tac, mu=0.0, callback=cb)
    q = np.array(q)
    out.update(mode_seed_m=eps, mode_kind='flutter' if is_flutter else 'saddle',
               mode_rate_per_s=rate.real, mode_omega_rad_s=abs(rate.imag),
               mode_amplitude_over_seed=[])
    for tt in np.arange(0.0, n_tac + 1e-9, 50.0):
        k = int(np.argmin(np.abs(q[:, 0] / p['t_ac'] - tt)))
        out['mode_amplitude_over_seed'].append(dict(
            t_tac=float(q[k, 0] / p['t_ac']), run=float(q[k, 1] / eps),
            linear_theory=float(abs(np.cosh(rate * q[k, 0])))))
    return out


def _growth_job(job):
    from ddgclib.methods import SolverMethods
    kind, n_refine = job['kind'], job['n_refine']
    methods = SolverMethods(dim=2, connectivity='dual_only')
    col = build_column(2, n_refine, ic='equilibrium', **_KINDS[kind])
    p = col.params
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_column(col, methods, n_tac=80.0)
        dudt0 = methods.dudt_fn(col.HC, mu=0.0, pressure_model=col.eos,
                                body_force=np.array([0.0, -p['g']]))
    for v in col.HC.V:
        v.u[:] = 0.0
    dofs = _dofs(col)
    lam = np.linalg.eigvals(_jacobian(col, dudt0, dofs))
    rates = _growth_rates(lam)
    return dict(job, n_dof=len(dofs), g_over_c0_per_s=p['g'] / p['c0'],
                saddle_over_g_c0=rates['saddle_per_s'] * p['c0'] / p['g'],
                **rates)


def main_growth(out, n_tac, procs):
    """Saddle and flutter growth rates of the settled inviscid 2D column
    against refinement (eigenvalues of the linearisation only)."""
    jobs = [dict(kind=k, n_refine=n) for n in (2, 3, 4) for k in _KINDS]
    with get_context('fork').Pool(min(procs, len(jobs))) as pool:
        rows = pool.map(_growth_job, jobs, chunksize=1)
    _save(out, 'growth', rows)
    _table(rows, [
        ('kind', lambda r: r['kind']), ('n_refine', lambda r: r['n_refine']),
        ('n_dof', lambda r: r['n_dof']),
        ('saddle [1/s]', lambda r: r['saddle_per_s']),
        ('x g/c0', lambda r: r['saddle_over_g_c0']),
        ('n_saddle', lambda r: r['n_saddle']),
        ('flutter [1/s]', lambda r: r['flutter_per_s']),
        ('flutter omega', lambda r: r['flutter_omega_rad_s']),
        ('n_complex', lambda r: r['n_complex']),
        ('max |Im lambda|', lambda r: r['max_imag']),
    ])


def main_free_surface(out, n_tac, procs, n_refine=3):
    jobs = [dict(kind=k, n_tac=n_tac, n_refine=n_refine) for k in _KINDS]
    slosh_jobs = [dict(case=case, arm=arm, n_tac=min(n_tac, 100.0), mu=mu)
                  for case in ('hydrostatic_2D', 'hydrostatic_2D_periodic')
                  for arm in ('preset', 'remap') for mu in (1e-3, None)]
    with get_context('fork').Pool(min(procs, len(jobs) + len(slosh_jobs))) as pool:
        lin = pool.map_async(_linearise, jobs, chunksize=1)
        slosh = pool.map_async(_slosh, slosh_jobs, chunksize=1)
        lin, slosh = lin.get(), slosh.get()
    results = {'linearised': lin, 'sloshing': slosh}
    _save(out, 'free_surface' if n_refine == 3
          else f'free_surface_r{n_refine}', results)
    for r in lin:
        print(json.dumps(r, indent=1))
    for r in slosh:
        print(json.dumps(r))


def _slosh(job):
    """Settle on mu_art, then seed the first sloshing mode
    (u = a exp(k (y - H)) (-sin kx, cos kx), k = pi, a = 1e-3 c0) and run
    with ``job['mu']`` (``None`` = mu_art): the seed must not grow."""
    case, arm = job['case'], job['arm']
    kw = CASES[case]
    methods = _methods(case, arm)
    col = build_column(kw['dim'], kw['n_refine'], H=kw['H'],
                       side_walls=kw['side_walls'], ic='equilibrium')
    p = col.params
    a = 1e-3 * p['c0']
    k = np.pi
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        run_column(col, methods, n_tac=60.0)
        u_settled = max(float(np.linalg.norm(v.u)) for v in col.free)
        for v in col.free:
            x, y = v.x_a[0], v.x_a[1]
            v.u[:2] = a * np.exp(k * (y - p['H'])) * np.array(
                [-np.sin(k * x), np.cos(k * x)])
            for ax in col.slip_axes.get(id(v), ()):
                v.u[ax] = 0.0
        res = run_column(col, methods, n_tac=job['n_tac'], mu=job['mu'])
    t = res['t'] / p['t_ac']
    ke = [float(res['ke'][(t > lo) & (t <= lo + 20)].max())
          for lo in range(0, int(job['n_tac']), 20)]
    return dict(case=case, arm=arm, mu=res['mu'], seed_u=a,
                u_settled_before=u_settled, ke0=float(res['ke'][0]),
                ke_window_max_over_ke0=[x / float(res['ke'][0]) for x in ke],
                umax_peak=float(res['umax'].max()),
                umax_end_env=_envelope(res, p['t_ac'], job['n_tac'], 20.0),
                gravity_wave_period_tac=float(
                    2 * np.pi / np.sqrt(p['g'] * k * np.tanh(k * p['H']))
                    / p['t_ac']))


def main_periodic(out, n_tac, procs):
    """The x-periodic column on the library periodic path
    (``SolverMethods(connectivity='periodic', periodic_axes=(0,))``)."""
    from hyperct.ddg import compute_vd

    from ddgclib._boundary_conditions import (BoundaryConditionSet,
                                              NoSlipWallBC)
    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
    from ddgclib.eos import TaitMurnaghan
    from ddgclib.geometry.domains import periodic_rectangle
    from ddgclib.initial_conditions import DualVolumeMass, ZeroVelocity
    from ddgclib.methods import SolverMethods
    from ddgclib.operators.stress import cache_dual_volumes, dual_area_vector

    H, rho, g, n_refine = 1.0, 1000.0, 9.81, 3
    K = rho * (10.0 * np.sqrt(g * H))**2
    eos = TaitMurnaghan(rho0=rho, P0=0.0, K=K, n=1.0, rho_clip=(0.5, 2.0))
    c0 = float(eos.sound_speed(rho))
    res = periodic_rectangle(L=1.0, h=H, refinement=n_refine,
                             periodic_axes=[0])
    HC, bounds = res.HC, res.metadata['domain_bounds']

    def bottom(v):
        return abs(v.x_a[1]) < 1e-12

    bV = {v for v in HC.V if bottom(v)}
    for v in HC.V:
        v.boundary = v in res.bV
    compute_vd(HC, method='barycentric')
    cache_dual_volumes(HC, 2)
    v_builder = sum(v.dual_vol for v in HC.V)
    # what the integrator does at the start of every step
    _retopologize(HC, bV, 2, periodic_axes=[0], domain_bounds=bounds,
                  boundary_filter=bottom)
    n_cells = 2**n_refine
    open_fans = [v for v in HC.V if float(np.linalg.norm(
        sum(dual_area_vector(v, nb, HC, 2) for nb in v.nn))) > 1e-12]
    row = dict(
        n_vertices=sum(1 for _ in HC.V),
        volume_builder=v_builder,
        volume_after_periodic_retopology=sum(v.dual_vol for v in HC.V),
        volume_exact=1.0 * H,
        n_simplices=len(HC._simplices),
        n_simplices_expected=4 * n_cells * n_cells,
        n_vertices_not_closing=len(open_fans),
        n_interior_not_closing=sum(1 for v in open_fans
                                   if 1e-9 < v.x_a[1] < H - 1e-9),
        n_interior_tagged_boundary=sum(1 for v in HC.V if v.boundary
                                       and 1e-9 < v.x_a[1] < H - 1e-9),
    )

    ZeroVelocity(dim=2).apply(HC, bV)
    DualVolumeMass(rho=rho).apply(HC, bV)
    edges = [float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
             for v in HC.V for nb in v.nn]
    methods = SolverMethods(dim=2, connectivity='periodic',
                            periodic_axes=(0,))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dudt_fn = methods.dudt_fn(HC, mu=0.5 * rho * c0 * 2.0**-n_refine,
                                  pressure_model=eos, body_force=[0.0, -g])
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=2), bV)
    dt = 0.25 * min(edges) / c0
    t_ac = H / c0
    hist = []

    def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
        hist.append((t / t_ac, max(float(np.linalg.norm(v.u[:2]))
                                   for v in HC_cb.V), len(bV_cb)))

    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        methods.integrate(HC, bV, dudt_fn, dt=dt,
                          n_steps=int(round(n_tac * t_ac / dt)), bc_set=bc_set,
                          callback=cb, pressure_model=eos,
                          domain_bounds=bounds, boundary_filter=bottom)
    h = np.array(hist)
    first = int(np.argmax(h[:, 1] > c0)) if (h[:, 1] > c0).any() else None
    row.update(
        umax_peak=float(h[:, 1].max()), c0=c0,
        t_ac_first_above_c0=float(h[first, 0]) if first is not None else None,
        n_frozen_start=int(h[0, 2]), n_frozen_end=int(h[-1, 2]),
        config=methods.to_dict())
    print(json.dumps(row, indent=1))
    _save(out, 'periodic', row)


# ----------------------------------------------------------------------
# remap: which ingredient breaks convex-hull Delaunay on a free surface
# ----------------------------------------------------------------------

def _remap_job(job):
    """One arm: the rebuild ``job['rebuild']`` (``'convex'`` = library
    ``_retopologize``, ``'material'`` = ``retopologize_material_delaunay``)
    inside the single-phase remap, with the global mass rescale
    (``redistribute_mass_single_phase``, the library remap) or without it
    (every vertex re-targeted to the snapshot pressure, total mass free).
    ``rebuild=None`` runs the preset.  Diagnostic closure through
    ``SolverMethods(connectivity='custom')``; the no-rescale variant is
    measured worse and is deliberately not a library option."""
    from hyperct.ddg import simplex_dual_volumes

    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
    from ddgclib.methods._retopo import retopologize_material_delaunay
    from ddgclib.operators.mass_redistribution import (
        redistribute_mass_single_phase, snapshot_pressure_fresh)

    kw = CASES[job['case']]
    col = build_column(kw['dim'], job.get('n_refine', kw['n_refine']),
                       H=kw['H'], side_walls=kw['side_walls'],
                       ic=job.get('ic', 'drop'),
                       free_surface=job.get('free_surface', True))
    p, dim = col.params, col.dim
    if 'g' in job:
        p['g'] = job['g']                  # body force only; the EOS keeps K
    if job.get('seed'):
        rng = np.random.default_rng(0)
        for v in sorted(col.free, key=lambda w: w.x):
            v.u[:dim] = job['seed'] * rng.standard_normal(dim)
    methods = PRESETS[job['case']]
    st = dict(offset=0.0, changes=[])
    custom = None
    if job['rebuild'] is not None:
        methods = methods.replace(connectivity='custom')
        rebuild = {'convex': _retopologize,
                   'material': retopologize_material_delaunay}[job['rebuild']]

        def custom(HC, bV, d, boundary_filter=None, **_kw):
            snap = snapshot_pressure_fresh(HC, d, col.eos)
            change = rebuild(HC, bV, d, boundary_filter=boundary_filter)
            if change is not None:
                st['changes'].append(change)
            if job['rescale']:
                info = redistribute_mass_single_phase(
                    HC, d, col.eos, bV=bV, pressure_snapshot=snap,
                    include_frozen=True)
                st['offset'] = max(st['offset'],
                                   abs(p['K'] * (info['scale_factor'] - 1.0)))
            else:
                for v in HC.V:
                    if id(v) in snap and v.dual_vol > 1e-30:
                        v.m = float(col.eos.density(snap[id(v)])) * v.dual_vol

    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        res = run_column(col, methods, n_tac=job['n_tac'], custom=custom)
    t = res['t'] / p['t_ac']
    ch = np.array(st['changes'])
    row = dict(job, n_vertices=p['n_vertices'], n_steps=res['n_steps'],
               umax_peak=float(res['umax'].max()),
               t_peak_tac=float(t[int(np.argmax(res['umax']))]),
               umax_end=float(res['umax'][-1]),
               volume_end=float(sum(simplex_dual_volumes(col.HC,
                                                         dim).values())),
               mass_drift=(sum(v.m for v in col.HC.V) - p['mass']) / p['mass'],
               max_offset_Pa=st['offset'] if job.get('rescale') else None)
    if len(ch):
        row.update(domain_change_max=float(np.abs(ch).max()),
                   domain_change_sum=float(ch.sum()),
                   domain_calls_changed=int((np.abs(ch) > 1e-12).sum()))
    return row


def main_remap(out, n_tac, procs):
    c2 = 'hydrostatic_2D'
    jobs = [dict(case=c2, tag=tag, rebuild=rb, rescale=rs, n_tac=n_tac)
            for tag, rb, rs in (
                ('convex + rescale (library remap)', 'convex', True),
                ('convex, no rescale', 'convex', False),
                ('material + rescale (library remap)', 'material', True),
                ('material, no rescale', 'material', False),
                ('preset dual_only', None, None))]
    jobs += [
        dict(case=c2, tag='control closed lid: convex + rescale',
             rebuild='convex', rescale=True, free_surface=False, n_tac=n_tac),
        dict(case=c2, tag='control g = 0, seed 1e-6: convex + rescale',
             rebuild='convex', rescale=True, g=0.0, seed=1e-6, n_tac=n_tac),
        dict(case=c2, tag='control g = 0, seed 1e-6: material + rescale',
             rebuild='material', rescale=True, g=0.0, seed=1e-6, n_tac=n_tac),
        # how well the material rebuild keeps the domain
        dict(case='hydrostatic_2D_periodic', tag='domain: 2D free-slip',
             rebuild='material', rescale=True, n_tac=20.0),
        dict(case=c2, tag='domain: 2D no-slip', rebuild='material',
             rescale=True, n_tac=20.0),
        dict(case='hydrostatic_3D', tag='domain: 3D refinement 1',
             rebuild='material', rescale=True, n_refine=1, n_tac=20.0),
        dict(case='hydrostatic_3D', tag='domain: 3D refinement 2',
             rebuild='material', rescale=True, n_tac=10.0),
        dict(case='hydrostatic_3D', tag='domain: 3D refinement 2, eq',
             rebuild='material', rescale=True, ic='equilibrium', n_tac=10.0),
    ]
    with get_context('fork').Pool(min(procs, len(jobs))) as pool:
        rows = pool.map(_remap_job, jobs, chunksize=1)
    _save(out, 'remap', rows)
    _table(rows, [
        ('arm', lambda r: r['tag']), ('n_tac', lambda r: r['n_tac']),
        ('umax_peak', lambda r: r['umax_peak']),
        ('at t_ac', lambda r: r['t_peak_tac']),
        ('umax_end', lambda r: r['umax_end']),
        ('volume', lambda r: f"{r['volume_end']:.9f}"),
        ('mass drift', lambda r: r['mass_drift']),
        ('max K(s-1) [Pa]', lambda r: r['max_offset_Pa']),
        ('max |dV/V| per call', lambda r: r.get('domain_change_max')),
        ('sum dV/V', lambda r: r.get('domain_change_sum')),
        ('calls changed', lambda r: r.get('domain_calls_changed')),
    ])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=('arms', 'convergence', 'viscosity',
                                     'free_surface', 'growth', 'periodic',
                                     'remap'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'results', 'diagnose'))
    ap.add_argument('--n-tac', type=float, default=None)
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--only', nargs='*', default=None,
                    help='arms: restrict to these cases')
    ap.add_argument('--n-refine', type=int, default=3,
                    help='free_surface: refinement of the linearised column')
    a = ap.parse_args()
    default_tac = dict(arms=40.0, convergence=60.0, viscosity=100.0,
                       free_surface=200.0, growth=0.0, periodic=4.0,
                       remap=8.0)[a.mode]
    kw = {'only': a.only} if a.mode == 'arms' else {}
    if a.mode == 'free_surface':
        kw = {'n_refine': a.n_refine}
    globals()[f'main_{a.mode}'](a.out, a.n_tac or default_tac, a.procs, **kw)
