"""Lane K diagnosis: why single-phase Lagrangian runs with an EOS blow up.

Measurement script only (no library changes).  Reproduces the template
box probe (rectangle L=h=1, refinement 2, rho0 1000, divergence-free swirl
u0 = 0.1 m/s, mu = 50) and runs a matrix of ``SolverMethods`` arms with a
per-step energy ledger that splits every step into

    dE_retopo = PE(new duals, new masses) - PE(old connectivity, same x)
    dE_dyn    = the force + move part of the step

so reconnection-induced energy injection (debugging_plan D1) is separated
from the explicit pressure-velocity coupling.  PE is the exact Tait
internal energy sum_i m_i e(m_i / V_i).

Usage (from the repo root)::

    python cases_dynamic/template/diagnose_single_phase_eos.py matrix  [--out DIR] [--only NAME ...]
    python cases_dynamic/template/diagnose_single_phase_eos.py flipjump
    python cases_dynamic/template/diagnose_single_phase_eos.py hydro   [--out DIR]

Outputs (per arm): ``<out>/<arm>.methods.json`` (record_methods),
``<out>/<arm>.npz`` (per-step series) and a summary table on stdout /
``<out>/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from functools import partial
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
warnings.simplefilter('ignore')

from ddgclib.geometry.domains import rectangle  # noqa: E402
from hyperct.ddg import compute_vd, simplex_dual_volumes  # noqa: E402
from ddgclib.operators.stress import cache_dual_volumes  # noqa: E402
from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC  # noqa: E402
from ddgclib.eos import TaitMurnaghan  # noqa: E402
from ddgclib.initial_conditions import CompositeIC, DualVolumeMass, CustomFieldIC  # noqa: E402
from ddgclib.methods import SolverMethods, record_methods  # noqa: E402
from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize  # noqa: E402

RHO0 = 1000.0
U0 = 0.1
MU = 50.0
T_END = 200 * 0.25 * 0.1767766952966369 / 10.0   # the probe horizon (200 steps at CFL 0.25, c_s 10)


# ---------------------------------------------------------------------------
# setup (identical to scratchpad/probe_template.py)
# ---------------------------------------------------------------------------
def build(refinement: int = 2, setup_volumes: str = 'builder',
          walls_mode: str = 'probe'):
    """Probe setup.  *setup_volumes*: 'builder' (probe: compute_vd +
    cache_dual_volumes on the builder mesh, no simplex cache -> 2D
    fallback dual_cell_area_2d) or 'exact' (rebuild the simplex cache
    first so setup volumes match what every retopology computes)."""
    d = 2
    result = rectangle(L=1.0, h=1.0, refinement=refinement, flow_axis=0)
    HC, bV = result.HC, result.bV
    # probe: NoSlipWallBC on the top/bottom 'walls' group only; the six
    # side (inlet/outlet) vertices are frozen by bV but keep their initial
    # tangential swirl velocity (they act as moving lids).  'all': every
    # hull vertex is a no-slip wall with u = 0.
    walls = (result.boundary_groups['walls'] if walls_mode == 'probe'
             else set(bV))
    for v in HC.V:
        v.boundary = v in bV
    if setup_volumes == 'exact':
        from hyperct.ddg import rebuild_simplex_cache_2d
        rebuild_simplex_cache_2d(HC)
    if setup_volumes == 'delaunay_circum':
        circumcentric_retopo(HC, bV, d)
    elif setup_volumes == 'delaunay':
        # settle the connectivity first: the per-step Delaunay rebuild
        # then starts from the mesh the ICs were computed on (no step-0
        # flips, same volume source)
        _retopologize(HC, bV, d)
    else:
        compute_vd(HC, method='barycentric')
        cache_dual_volumes(HC, d)

    def perturbation(x):
        return U0 * np.array([np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
                              -np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])])

    CompositeIC(CustomFieldIC(perturbation, field_name='u'),
                DualVolumeMass(rho=RHO0)).apply(HC, bV)
    for v in walls:
        v.u = np.zeros(d)
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=d), walls)
    dx_min = min(float(np.linalg.norm(v.x_a[:d] - nb.x_a[:d]))
                 for v in HC.V for nb in v.nn)
    return HC, bV, walls, bc_set, dx_min


def tait_energy(r: np.ndarray, K: float, n: float) -> np.ndarray:
    """Specific internal energy e(rho)/(K/rho0) for P0=0 Tait, r=rho/rho0."""
    r = np.clip(r, 0.5, 2.0)          # the EOS band (saturates outside)
    if abs(n - 1.0) < 1e-12:
        return (K / RHO0) * (np.log(r) + 1.0 / r - 1.0)
    return (K / (n * RHO0)) * ((r ** (n - 1) - 1.0) / (n - 1) + 1.0 / r - 1.0)


# ---------------------------------------------------------------------------
# single-phase conservative remap prototype (diagnose-only, NOT library)
# ---------------------------------------------------------------------------
def remap_retopo(HC, bV, dim, eos=None, snapshot='fresh', include_walls=True,
                 gauge='scale', dual='barycentric', **_kw):
    """Single-phase analogue of the laneD multiphase conservative remap.

    stage 1  snapshot p on the OLD connectivity at the NEW positions
             ('fresh', = what the physics says the pressure now is) or the
             stale v.p of the last force evaluation ('stale', = what the
             library's redistribute_mass_single_phase uses);
    stage 2  library Delaunay rebuild (no redistribution);
    stage 3  m_i = rho(p_snap_i) * V_new_i on interior (+ walls if
             include_walls), then a single mass-conserving scale (gauge).
    """
    verts = list(HC.V)
    if snapshot == 'fresh':
        if getattr(HC, '_simplices', None) is not None:
            vols = simplex_dual_volumes(HC, dim)
        else:
            cache_dual_volumes(HC, dim)
            vols = {v: v.dual_vol for v in verts}
        p_snap = {id(v): float(eos.pressure(v.m / vols[v])) if vols[v] > 1e-30
                  else 0.0 for v in verts}
    else:
        p_snap = {id(v): float(v.p) for v in verts}
    walls = set(bV)
    _retopologize(HC, bV, dim)
    if dual == 'circumcentric':
        compute_vd(HC, method='circumcentric')
        HC._vd_method = 'circumcentric'
        cache_dual_volumes(HC, dim)
    targets = [v for v in HC.V
               if v.dual_vol > 1e-30 and (include_walls or v not in walls)]
    M_before = sum(v.m for v in targets)
    new_m = {v: float(eos.density(p_snap[id(v)])) * v.dual_vol for v in targets}
    if gauge == 'scale':
        s = M_before / sum(new_m.values())
        for v in targets:
            v.m = new_m[v] * s
    else:
        for v in targets:
            v.m = new_m[v]


def circumcentric_retopo(HC, bV, dim, **_kw):
    """Plain Delaunay rebuild with circumcentric (Voronoi) duals: dual
    volumes are continuous across a co-circular Delaunay flip."""
    _retopologize(HC, bV, dim)
    compute_vd(HC, method='circumcentric')
    HC._vd_method = 'circumcentric'
    cache_dual_volumes(HC, dim)


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------
def edge_set(HC) -> set:
    return {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}


class Recorder:
    def __init__(self, HC, bV, walls, eos, K, n, abort_u, refresh_ok=True):
        self.refresh_ok = refresh_ok
        self.verts = list(HC.V)
        self.idx = {id(v): i for i, v in enumerate(self.verts)}
        self.walls = np.array([v in walls for v in self.verts])
        xs = np.array([v.x_a[:2] for v in self.verts])
        self.corner = np.array([(abs(x[0] - 0.5) > 0.49) and (abs(x[1] - 0.5) > 0.49)
                                for x in xs])
        self.eos, self.K, self.n = eos, K, n
        self.abort_u = abort_u
        self.prev_edges = edge_set(HC)
        self.prev_pe_old = None   # PE at old connectivity, x^{k+1}, masses m^k
        self.prev_ke = None
        self.prev_vold = None
        self.rows = []
        self.fields = {k: [] for k in ('p', 'm', 'vol', 'vol_old', 'umag', 'x')}
        self.flip_verts = []
        self.aborted = None
        # setup-state energy (duals as built)
        m = np.array([v.m for v in self.verts])
        vol = np.array([v.dual_vol for v in self.verts])
        u = np.array([v.u[:2] for v in self.verts])
        self.E0 = float(0.5 * np.sum(m * np.sum(u * u, 1))
                        + np.sum(m * tait_energy(m / vol / RHO0, K, n)))
        self.KE0 = float(0.5 * np.sum(m * np.sum(u * u, 1)))

    def __call__(self, step, t, HC, bV=None, diagnostics=None):
        verts = self.verts
        m = np.array([v.m for v in verts])
        vol = np.array([v.dual_vol for v in verts])           # used this step
        u = np.array([v.u[:2] for v in verts])
        p = np.array([float(np.ravel(v.p)[0]) if hasattr(v, 'p') else 0.0
                      for v in verts])
        umag = np.linalg.norm(u, axis=1)
        ke = float(0.5 * np.sum(m * umag ** 2))
        pe_new = float(np.sum(m * tait_energy(m / vol / RHO0, self.K, self.n)))
        # old-connectivity volumes at the new positions
        if getattr(HC, '_simplices', None) is not None:
            vo = simplex_dual_volumes(HC, 2)
            vol_old = np.array([vo[v] for v in verts])
            inverted = 0
            for s in HC._simplices:
                a, b, c = (np.asarray(w.x_a[:2]) for w in s)
                det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if det <= 0:
                    inverted += 1
            # orientation sign convention of the cache may be either; count
            # the minority sign as "inverted"
            inverted = min(inverted, len(HC._simplices) - inverted)
        elif self.refresh_ok:
            # frozen builder connectivity (dual_only): refresh the duals at
            # the new positions exactly as the next retopology will, read
            # them, then restore the attribute the solver used this step
            compute_vd(HC, method='barycentric')
            cache_dual_volumes(HC, 2)
            vol_old = np.array([v.dual_vol for v in verts])
            for v, vv in zip(verts, vol):
                v.dual_vol = vv
            inverted = -1
        else:
            vol_old = vol.copy()
            inverted = -1
        pe_old = float(np.sum(m * tait_energy(m / vol_old / RHO0, self.K, self.n)))
        edges = edge_set(HC)
        flipped = edges ^ self.prev_edges
        n_flips = len(flipped) // 2
        fv = set()
        for e in flipped:
            fv |= set(e)
        self.flip_verts.append(sorted(self.idx[i] for i in fv if i in self.idx))
        self.prev_edges = edges
        # retopology energy jump (masses already redistributed at this step)
        if self.prev_pe_old is None:
            dE_retopo = pe_new - (self.E0 - self.KE0)
            dvol_retopo = vol - np.array([v for v in self._setup_vol])
        else:
            dE_retopo = pe_new - self.prev_pe_old
            dvol_retopo = vol - self.prev_vold
        interior = ~self.walls
        rho = m / np.where(vol > 0, vol, np.nan)
        rel_retopo = np.abs(dvol_retopo) / np.where(vol > 0, vol, np.nan)
        rel_motion = np.abs(vol_old - vol) / np.where(vol > 0, vol, np.nan)
        row = dict(
            step=step, t=t, KE=ke, PE=pe_old, E=ke + pe_old,
            dE_retopo=dE_retopo, n_flips=n_flips, inverted=inverted,
            umax=float(umag.max()),
            p_min_int=float(p[interior].min()), p_max_int=float(p[interior].max()),
            p_min_wall=float(p[self.walls].min()), p_max_wall=float(p[self.walls].max()),
            p_corner_mean=float(p[self.corner].mean()),
            rho_min=float(np.nanmin(rho)), rho_max=float(np.nanmax(rho)),
            vol_total=float(vol.sum()), vol_wall=float(vol[self.walls].sum()),
            mass_total=float(m.sum()),
            max_rel_dvol_retopo=float(np.nanmax(rel_retopo)),
            max_rel_dvol_motion=float(np.nanmax(rel_motion)),
            argmax_u=int(np.argmax(umag)),
        )
        self.rows.append(row)
        self.fields['p'].append(p)
        self.fields['m'].append(m)
        self.fields['vol'].append(vol)
        self.fields['vol_old'].append(vol_old)
        self.fields['umag'].append(umag)
        self.fields['x'].append(np.array([v.x_a[:2] for v in verts]))
        self.prev_pe_old = pe_old
        self.prev_vold = vol_old
        self.prev_ke = ke
        if not np.isfinite(row['umax']) or row['umax'] > self.abort_u:
            self.aborted = step
            raise _Abort(step)


class _Abort(Exception):
    pass


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
BASE = dict(connectivity='delaunay', eos=True, redistribute=False,
            integrator='symplectic_euler', cfl=0.25, c_s=10.0, n=1.0,
            setup_volumes='builder', custom=None, custom_kw=None)


def arm(name, **kw):
    a = dict(BASE)
    a.update(kw)
    a['name'] = name
    return a


MATRIX = [
    # connectivity x EOS x redistribution (symplectic, CFL 0.25, c_s 10, n 1)
    arm('A1_delaunay_eos_redist', redistribute=True),
    arm('A2_delaunay_eos'),
    arm('A3_delaunay_noeos', eos=False),
    arm('A4_dualonly_eos_redist', connectivity='dual_only', redistribute=True),
    arm('A5_dualonly_eos', connectivity='dual_only'),
    arm('A6_dualonly_noeos', connectivity='dual_only', eos=False),
    arm('A7_frozen_eos', connectivity='frozen'),
    arm('A8_frozen_noeos', connectivity='frozen', eos=False),
    arm('A9_eulerian_eos', integrator='euler_velocity_only'),
    arm('A10_eulerian_noeos', integrator='euler_velocity_only', eos=False),
    # integrator
    arm('B1_delaunay_eos_euler', integrator='euler'),
    arm('B2_dualonly_eos_euler', connectivity='dual_only', integrator='euler'),
    # dt (same physical horizon)
    arm('C1_delaunay_eos_cfl0.05', cfl=0.05),
    arm('C2_delaunay_eos_cfl0.01', cfl=0.01),
    arm('C3_dualonly_eos_cfl0.05', connectivity='dual_only', cfl=0.05),
    arm('C4_dualonly_eos_cfl0.01', connectivity='dual_only', cfl=0.01),
    arm('C5_delaunay_eos_redist_cfl0.05', redistribute=True, cfl=0.05),
    # stiffness
    arm('D1_delaunay_eos_cs100', c_s=100.0),
    arm('D2_dualonly_eos_cs100', connectivity='dual_only', c_s=100.0),
    arm('D3_delaunay_eos_n7', n=7.15),
    arm('D4_dualonly_eos_n7', connectivity='dual_only', n=7.15),
    # setup-volume consistency (corner cells, see flipjump)
    arm('E1_delaunay_eos_exactsetup', setup_volumes='exact'),
    arm('E2_delaunay_eos_redist_exactsetup', setup_volumes='exact', redistribute=True),
    arm('E3_dualonly_eos_exactsetup', connectivity='dual_only', setup_volumes='exact'),
    # prototypes (custom retopologize_fn, diagnose-only)
    arm('P1_remap_fresh_walls', connectivity='custom', setup_volumes='exact',
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=True)),
    arm('P2_remap_fresh_interior', connectivity='custom', setup_volumes='exact',
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=False)),
    arm('P3_remap_stale_walls', connectivity='custom', setup_volumes='exact',
        custom='remap', custom_kw=dict(snapshot='stale', include_walls=True)),
    arm('P4_circumcentric', connectivity='custom', setup_volumes='exact',
        custom='circum'),
    arm('P5_remap_fresh_walls_cs100', connectivity='custom', setup_volumes='exact',
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=True),
        c_s=100.0),
    arm('P6_remap_fresh_walls_n7', connectivity='custom', setup_volumes='exact',
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=True),
        n=7.15),
    arm('P7_remap_fresh_walls_long', connectivity='custom', setup_volumes='exact',
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=True),
        t_mult=5.0),
    arm('P8_dualonly_eos_exactsetup_long', connectivity='dual_only',
        setup_volumes='exact', t_mult=5.0),
    # settled (Delaunay) setup: no step-0 flips / volume-source jump, so
    # only the flips produced by the physical motion remain
    arm('F1_delaunay_eos_settled', setup_volumes='delaunay'),
    arm('F2_delaunay_eos_redist_settled', setup_volumes='delaunay', redistribute=True),
    arm('F3_delaunay_eos_settled_cfl0.01', setup_volumes='delaunay', cfl=0.01),
    arm('F4_delaunay_eos_settled_cs100', setup_volumes='delaunay', c_s=100.0),
    arm('F5_delaunay_eos_settled_wallsall', setup_volumes='delaunay', walls='all'),
    arm('F6_delaunay_noeos_settled', setup_volumes='delaunay', eos=False),
    arm('F7_dualonly_eos_settled', connectivity='dual_only', setup_volumes='delaunay'),
    arm('F8_dualonly_eos_settled_long', connectivity='dual_only',
        setup_volumes='delaunay', t_mult=5.0),
    arm('F9_dualonly_eos_wallsall_long', connectivity='dual_only',
        setup_volumes='delaunay', walls='all', t_mult=5.0),
    arm('F10_delaunay_eos_settled_wallsall_long', setup_volumes='delaunay',
        walls='all', t_mult=5.0),
    arm('P9_remap_fresh_walls_settled', connectivity='custom', setup_volumes='delaunay',
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=True)),
    arm('P10_remap_fresh_walls_settled_wallsall_long', connectivity='custom',
        setup_volumes='delaunay', walls='all', t_mult=5.0,
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=True)),
    arm('P11_remap_fresh_nogauge_settled', connectivity='custom', setup_volumes='delaunay',
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=True,
                                       gauge='none')),
    arm('P12_remap_stale_walls_settled', connectivity='custom', setup_volumes='delaunay',
        custom='remap', custom_kw=dict(snapshot='stale', include_walls=True)),
    arm('P13_remap_fresh_walls_settled_cfl0.01', connectivity='custom',
        setup_volumes='delaunay', cfl=0.01,
        custom='remap', custom_kw=dict(snapshot='fresh', include_walls=True)),
    arm('P14_circumcentric_consistent_setup', connectivity='custom',
        setup_volumes='delaunay_circum', walls='all', custom='circum'),
    arm('P15_circumcentric_dualonly', connectivity='dual_only',
        setup_volumes='delaunay_circum', walls='all'),
    # acoustic stability boundary on fixed connectivity / with the remap
    *[arm(f'G_dualonly_cfl{c}', connectivity='dual_only', setup_volumes='delaunay',
          walls='all', cfl=c) for c in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)],
    *[arm(f'G_remap_cfl{c}', connectivity='custom', setup_volumes='delaunay',
          walls='all', cfl=c, custom='remap',
          custom_kw=dict(snapshot='fresh', include_walls=True))
      for c in (0.5, 1.0, 1.5)],
    *[arm(f'G_dualonly_euler_cfl{c}', connectivity='dual_only', setup_volumes='delaunay',
          walls='all', cfl=c, integrator='euler') for c in (0.05, 0.25)],
]


def run_arm(a: dict, out: str) -> dict:
    HC, bV, walls, bc_set, dx_min = build(setup_volumes=a['setup_volumes'],
                                          walls_mode=a.get('walls', 'probe'))
    c_s, n = a['c_s'], a['n']
    K = RHO0 * c_s ** 2
    eos = TaitMurnaghan(rho0=RHO0, P0=0.0, K=K, n=n, rho_clip=(0.5, 2.0))
    for v in HC.V:
        v.p = float(eos.pressure(v.m / v.dual_vol)) if a['eos'] else 0.0
    custom = None
    if a['custom'] == 'remap':
        custom = partial(remap_retopo, eos=eos, **(a['custom_kw'] or {}))
    elif a['custom'] == 'circum':
        custom = circumcentric_retopo
    methods = SolverMethods(dim=2, integrator=a['integrator'],
                            connectivity=a['connectivity'],
                            redistribute_mass=a['redistribute'],
                            label=f"laneK {a['name']}",
                            notes=json.dumps({k: a[k] for k in (
                                'eos', 'cfl', 'c_s', 'n', 'setup_volumes',
                                'custom', 'custom_kw')}))
    pm = eos if a['eos'] else None
    dudt = methods.dudt_fn(HC, mu=MU, pressure_model=pm)
    dt = a['cfl'] * dx_min / c_s
    t_end = T_END * a.get('t_mult', 1.0)
    n_steps = int(round(t_end / dt))
    rec = Recorder(HC, bV, set(bV), eos, K, n, abort_u=c_s,
                   refresh_ok=a['connectivity'] != 'frozen')
    rec._setup_vol = [v.dual_vol for v in rec.verts]
    t0 = time.time()
    err = None
    try:
        methods.integrate(HC, bV, dudt, dt=dt, n_steps=n_steps, bc_set=bc_set,
                          callback=rec, custom=custom,
                          pressure_model=pm if a['redistribute'] else None)
    except _Abort:
        err = f'abort |u|>c_s at step {rec.aborted}'
    except Exception as e:  # noqa: BLE001
        err = f'{type(e).__name__}: {e}'
    wall = time.time() - t0
    record_methods(os.path.join(out, f"{a['name']}.methods.json"), methods, HC,
                   extra={'dt': dt, 'n_steps': n_steps, 'mu': MU, 'u0': U0,
                          'c_s': c_s, 'n': n, 'cfl': a['cfl'],
                          'eos_in_dudt': a['eos'],
                          'setup_volumes': a['setup_volumes'],
                          'custom': a['custom'], 'custom_kw': a['custom_kw']})
    rows = rec.rows
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    np.savez_compressed(os.path.join(out, f"{a['name']}.npz"),
                        walls=rec.walls, corner=rec.corner,
                        **{f'f_{k}': np.array(v) for k, v in rec.fields.items()},
                        **{f's_{k}': v for k, v in arr.items()})
    s = summarize(a, rec, dt, n_steps, err, wall)
    s['methods'] = methods.to_dict()
    with open(os.path.join(out, f"{a['name']}.summary.json"), 'w') as fh:
        json.dump(s, fh, indent=1, default=float)
    return s


def summarize(a, rec, dt, n_steps, err, wall):
    rows = rec.rows
    E0, KE0 = rec.E0, rec.KE0
    E = np.array([r['E'] for r in rows])
    dEr = np.array([r['dE_retopo'] for r in rows])
    nfl = np.array([r['n_flips'] for r in rows])
    first_e2 = next((r['step'] for r in rows if r['E'] > 2 * E0), None)
    first_u2 = next((r['step'] for r in rows if r['umax'] > 2 * U0), None)
    # energy budget: retopology vs dynamics
    E_prev = np.concatenate([[E0], E[:-1]])
    dE_tot = E - E_prev
    dE_dyn = dE_tot - dEr
    last = rows[-1]
    fs = first_u2 if first_u2 is not None else None
    culprit = None
    if fs is not None:
        i = rows[fs]['argmax_u']
        cls = 'corner' if rec.corner[i] else ('wall' if rec.walls[i] else 'interior')
        # interior: is it adjacent to a corner / wall?
        v = rec.verts[i]
        nb_walls = sum(1 for w in v.nn if rec.walls[rec.idx[id(w)]])
        nb_corner = sum(1 for w in v.nn if rec.corner[rec.idx[id(w)]])
        culprit = dict(vertex=i, x=[float(c) for c in rec.fields['x'][fs][i]],
                       cls=cls, wall_nbrs=nb_walls, corner_nbrs=nb_corner)
    return dict(
        name=a['name'], dt=dt, n_steps=n_steps, steps_run=len(rows),
        err=err, wall_s=round(wall, 1),
        dtc_dx=a['cfl'], KE0=KE0, E0=E0,
        KE_end=last['KE'], KE_max=max(r['KE'] for r in rows),
        umax_end=last['umax'], umax_max=max(r['umax'] for r in rows),
        E_end=last['E'],
        sum_dE_retopo=float(dEr.sum()), sum_dE_dyn=float(dE_dyn.sum()),
        sum_dE_retopo_pos=float(dEr[dEr > 0].sum()),
        dE_retopo_step0=float(dEr[0]),
        sum_dE_dyn_pos=float(dE_dyn[dE_dyn > 0].sum()),
        flips_total=int(nfl[1:].sum()), flips_step0=int(nfl[0]),
        steps_with_flips=int((nfl[1:] > 0).sum()),
        first_E_gt_2E0=first_e2, first_umax_gt_2u0=first_u2,
        culprit=culprit,
        p_int=(min(r['p_min_int'] for r in rows), max(r['p_max_int'] for r in rows)),
        p_wall=(min(r['p_min_wall'] for r in rows), max(r['p_max_wall'] for r in rows)),
        p_corner_step0=rows[0]['p_corner_mean'],
        rho=(min(r['rho_min'] for r in rows), max(r['rho_max'] for r in rows)),
        vol_total=(rows[0]['vol_total'], last['vol_total']),
        vol_total_setup=float(np.sum(rec._setup_vol)),
        max_rel_dvol_retopo=float(np.nanmax([r['max_rel_dvol_retopo'] for r in rows[1:]] or [0])),
        max_rel_dvol_motion=float(np.nanmax([r['max_rel_dvol_motion'] for r in rows])),
        inverted_max=max(r['inverted'] for r in rows),
        mass=(rows[0]['mass_total'], last['mass_total']),
        clip_count=dict(rec.eos.clip_count),
    )


def _worker(args):
    a, out = args
    try:
        return run_arm(a, out)
    except Exception as e:  # noqa: BLE001
        return dict(name=a['name'], err=f'SETUP {type(e).__name__}: {e}')


def fmt(x, p=3):
    if x is None:
        return '-'
    if isinstance(x, (tuple, list)):
        return '[' + ', '.join(fmt(y, p) for y in x) + ']'
    if isinstance(x, float):
        return f'{x:.{p}g}'
    return str(x)


def main_matrix(out, only=None, procs=16):
    os.makedirs(out, exist_ok=True)
    arms = [a for a in MATRIX if not only or any(o in a['name'] for o in only)]
    with get_context('fork').Pool(min(procs, len(arms))) as pool:
        res = pool.map(_worker, [(a, out) for a in arms])
    prev = {}
    sp = os.path.join(out, 'summary.json')
    if os.path.exists(sp):
        prev = {r['name']: r for r in json.load(open(sp))}
    for r in res:
        prev[r['name']] = r
    json.dump(list(prev.values()), open(sp, 'w'), indent=1, default=float)
    for r in res:
        print(f"{r['name']:36s} steps {r.get('steps_run')}/{r.get('n_steps')} "
              f"KE {fmt(r.get('KE0'))}->{fmt(r.get('KE_end'))} max|u| {fmt(r.get('umax_max'))} "
              f"u2 {fmt(r.get('first_umax_gt_2u0'))} flips {r.get('flips_step0')}+{r.get('flips_total')} "
              f"dEre {fmt(r.get('sum_dE_retopo'))} (0:{fmt(r.get('dE_retopo_step0'))}) "
              f"dEdyn {fmt(r.get('sum_dE_dyn'))} p_int {fmt(r.get('p_int'))} "
              f"p_wall {fmt(r.get('p_wall'))} rho {fmt(r.get('rho'))} "
              f"V {fmt(r.get('vol_total_setup'))}/{fmt(r.get('vol_total'))} "
              f"cul {r.get('culprit')} {r.get('err') or ''}")


# ---------------------------------------------------------------------------
# static flip-jump probe: dual-volume discontinuity of a Delaunay flip
# ---------------------------------------------------------------------------
def main_flipjump():
    """Barycentric vs circumcentric dual volume of vertex a in a
    co-circular quad a,b,c,d on either side of the flip."""
    ang = np.deg2rad([200.0, 290.0, 20.0, 110.0])
    for eps in (1e-3, -1e-3):
        pts = np.c_[np.cos(ang), np.sin(ang)]
        pts[0] *= 1.0 + eps          # push a in/out of the circumcircle
        from scipy.spatial import Delaunay
        tri = Delaunay(pts).simplices
        area = lambda s: 0.5 * abs(np.cross(pts[s[1]] - pts[s[0]], pts[s[2]] - pts[s[0]]))
        Va = sum(area(s) for s in tri if 0 in s) / 3.0
        print(f"eps={eps:+.0e}  triangles {tri.tolist()}  barycentric V_a={Va:.5f}")
    # setup vs retopo volume source on the probe mesh
    HC, bV, walls, bc, dx = build()
    fb = {v: v.dual_vol for v in HC.V}
    from hyperct.ddg import rebuild_simplex_cache_2d
    rebuild_simplex_cache_2d(HC)
    ex = simplex_dual_volumes(HC, 2)
    bad = [(tuple(np.round(v.x_a[:2], 3)), fb[v], ex[v]) for v in HC.V
           if abs(fb[v] - ex[v]) > 1e-12]
    print('builder-mesh fallback vs exact volumes (differing vertices):', bad)
    print('totals', sum(fb.values()), sum(ex.values()))


# ---------------------------------------------------------------------------
# Hydrostatic_column 2D cross-check (hand-rolled loop, NO reconnection)
# ---------------------------------------------------------------------------
def main_hydro(out, n_tac=8.0, variant='case'):
    """Replicates cases_dynamic/Hydrostatic_column/Hydrostatic_2D.py
    Section 3 (the loop body is copied verbatim) with instrumentation.
    variant 'case' = as shipped; 'exact' = rebuild the simplex cache so the
    per-step volumes are the exact barycentric ones from the start;
    'nogravity' = case setup, g=0 (pure box with a free top).
    Since laneS the builder mesh carries the simplex cache, so 'case' IS
    'exact'.  A variant with 'fallback' in its name ('fallback',
    'nogravity_fallback_seed') drops the cache after setup, so the loop
    runs on the 2D fallback hyperct dual_cell_area_2d (what 'case' did
    before laneS; the fallback itself was fixed in laneS)."""
    os.makedirs(out, exist_ok=True)
    from cases_dynamic.Hydrostatic_column.src._setup import (
        setup_hydrostatic_column, make_gravity_dudt)
    from ddgclib.initial_conditions import ZeroVelocity
    from ddgclib.dynamic_integrators._integrators_dynamic import (
        _recompute_duals, _interior_verts, _move)
    H, rho, g, mu_phys, n_refine = 1.0, 1000.0, 9.81, 1e-3, 3
    K_eos = rho * (10 * np.sqrt(g * H)) ** 2
    eos = TaitMurnaghan(rho0=rho, P0=0.0, K=K_eos, n=1.0, rho_clip=(0.5, 2.0))
    c0 = float(eos.sound_speed(rho))
    HC, bV, bc_set, _, _ = setup_hydrostatic_column(
        dim=2, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=0.0, mu=mu_phys,
        gravity_axis=1, free_surface=True)
    # exact barycentric volumes on the (never changing) builder
    # connectivity, computed on the side for the fallback-vs-exact monitor
    from hyperct.ddg import rebuild_simplex_cache_2d as _rb
    builder_cache = HC._simplices
    _rb(HC)
    tris = list(HC._simplices)
    HC._simplices = None if 'fallback' in variant else builder_cache

    def exact_vols():
        out = {v: 0.0 for v in HC.V}
        for s3 in tris:
            a_, b_, c_ = (np.asarray(w.x_a[:2]) for w in s3)
            ar = 0.5 * abs((b_[0] - a_[0]) * (c_[1] - a_[1]) - (b_[1] - a_[1]) * (c_[0] - a_[0]))
            for w in s3:
                out[w] += ar / 3.0
        return out

    if variant in ('exact', 'nogravity_exact', 'nogravity_exact_seed'):
        from hyperct.ddg import rebuild_simplex_cache_2d
        rebuild_simplex_cache_2d(HC)
        compute_vd(HC, method='barycentric')
        cache_dual_volumes(HC, 2)
    ZeroVelocity(dim=2).apply(HC, bV)
    DualVolumeMass(rho=rho).apply(HC, bV)
    if variant.endswith('_seed'):
        # 1e-6 m/s random seed on the moving vertices (fixed RNG)
        rng = np.random.default_rng(0)
        for v in sorted(HC.V, key=lambda w: tuple(w.x_a)):
            if v not in bV:
                v.u[:2] = 1e-6 * rng.standard_normal(2)
    edges = [np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) for v in HC.V for nb in v.nn]
    dx_mean, dx_min_init = np.mean(edges), min(edges)
    mu_art = 0.5 * rho * c0 * dx_mean
    gg = 0.0 if variant.startswith('nogravity') else g
    dudt_dyn = make_gravity_dudt(dim=2, mu=mu_art, HC=HC, g=gg, gravity_axis=1,
                                 pressure_model=eos)
    verts = list(HC.V)
    idx = {id(v): i for i, v in enumerate(verts)}
    frozen = np.array([v in bV for v in verts])
    xs0 = np.array([v.x_a[:2] for v in verts])
    top = np.isclose(xs0[:, 1], H)
    corner = np.array([(x[0] in (0.0, 1.0)) and (x[1] in (0.0, H)) for x in xs0])
    vtag = np.array([v.boundary for v in verts])
    t_ac = H / c0
    t_end = n_tac * t_ac
    t, step = 0.0, 0
    e0 = edge_set(HC)
    rows, P, U, VOL = [], [], [], []
    CFL = 0.25
    while t < t_end:
        _recompute_duals(HC)
        cache_dual_volumes(HC, dim=2)
        iv = _interior_verts(HC, bV)
        u_max = max((np.linalg.norm(v.u[:2]) for v in iv), default=0.0)
        dx_min = min((np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) for v in iv for nb in v.nn
                      if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 0), default=dx_min_init)
        dt = min(CFL * dx_min / (c0 + u_max), t_end - t)
        acc = {v: dudt_dyn(v) for v in iv}
        for v in iv:
            v.u[:2] += dt * acc[v][:2]
            _move(v, v.x_a[:2] + dt * v.u[:2], HC, bV)
        if bc_set:
            bc_set.apply_all(HC, bV, dt)
        t += dt
        step += 1
        p = np.array([float(v.p) for v in verts])
        um = np.array([np.linalg.norm(v.u[:2]) for v in verts])
        vol = np.array([v.dual_vol for v in verts])
        m = np.array([v.m for v in verts])
        ke = float(0.5 * np.sum(m * um ** 2))
        pe = float(np.sum(m * tait_energy(m / vol / rho, K_eos, 1.0)))
        pot = float(np.sum(m * gg * np.array([v.x_a[1] for v in verts])))
        i = int(np.argmax(um))
        ev = exact_vols()
        ex = np.array([ev[v] for v in verts])
        rel = np.abs(vol - ex) / ex
        rows.append(dict(fb_top=float(rel[top].max()), fb_int=float(rel[~frozen & ~top].max()),
                         fb_frozen=float(rel[frozen].max()), step=step, t_tac=t / t_ac, dt=dt, KE=ke, PE=pe, Epot=pot,
                         umax=float(um.max()), argmax=i,
                         p_top=(float(p[top].min()), float(p[top].max())),
                         p_corner=float(p[corner].mean()),
                         p_int=(float(p[~frozen & ~top].min()), float(p[~frozen & ~top].max())),
                         vol_total=float(vol.sum()),
                         flips=len(edge_set(HC) ^ e0) // 2))
        P.append(p); U.append(um); VOL.append(vol)
        if um.max() > 10 * c0:
            break
    np.savez_compressed(os.path.join(out, f'hydro_{variant}.npz'), P=np.array(P),
                        U=np.array(U), VOL=np.array(VOL), frozen=frozen, top=top,
                        corner=corner, x0=xs0, vtag=vtag,
                        **{k: np.array([r[k] for r in rows]) for k in
                           ('t_tac', 'KE', 'PE', 'Epot', 'umax', 'argmax', 'vol_total', 'flips')})
    print(f"hydro[{variant}] c0={c0:.2f} K={K_eos:.3g} mu_art={mu_art:.1f} "
          f"n_verts={len(verts)} frozen={frozen.sum()} top={top.sum()} "
          f"v.boundary tagged={vtag.sum()}")
    stride = max(1, len(rows) // 25)
    for r in rows[::stride] + [rows[-1]]:
        i = r['argmax']
        cls = ('corner' if corner[i] else 'top' if top[i] else
               'frozen' if frozen[i] else 'interior')
        print(f"  t/tac {r['t_tac']:6.2f} KE {r['KE']:.3e} PE {r['PE']:.3e} "
              f"Epot {r['Epot']:.4e} umax {r['umax']:.3e} @{i}({cls},x={np.round(xs0[i], 3)}) "
              f"p_top {fmt(r['p_top'])} p_int {fmt(r['p_int'])} "
              f"V {r['vol_total']:.5f} flips {r['flips']} "
              f"|Vfb-Vex|/Vex top {r['fb_top']:.2e} int {r['fb_int']:.2e} frozen {r['fb_frozen']:.2e}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('matrix', 'flipjump', 'hydro'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'results', 'laneK'))
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--variant', default='case')
    ap.add_argument('--n-tac', type=float, default=8.0)
    args = ap.parse_args()
    if args.mode == 'matrix':
        main_matrix(args.out, args.only, args.procs)
    elif args.mode == 'flipjump':
        main_flipjump()
    else:
        main_hydro(args.out, args.n_tac, args.variant)
