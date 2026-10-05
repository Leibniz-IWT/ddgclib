#!/usr/bin/env python3
"""Census of the orientation of the dual area vectors A_ij (lane O, 2026-10-02).

``ddgclib.operators.stress.dual_area_vector`` returns, in 2D, the normal
of the segment between the two dual vertices that the endpoints of an edge
share.  Its sign must be "outward from i": ``A_ij . d_ij > 0`` with
``d_ij = x_j - x_i``.  Until lane O the sign was taken from the midpoint of
the dual segment (method axis ``area_orientation``, value
``'dual_midpoint'``), which points the vector AGAINST its edge when the two
triangles at the edge have angles at ``x_i`` that sum to more than 180
degrees and ``x_j`` is close to the line through the two opposite vertices.

This script counts, on a mesh or along a run, how many directed edges the
two rules disagree on, and measures what an orientation error breaks:
closure of the dual cell (``sum_j A_ij = 0``), antisymmetry
(``A_ij = -A_ji``), agreement with the exact vectors of the simplex cache
(``ddgclib.operators.stress.simplex_area_vectors``) and the centred force
of a linear pressure.  The two rules are evaluated here from the shared
dual vertices, independently of what the library is set to, so the counts
are the same on a library before and after the fix.

Usage (repo root)::

    python cases_dynamic/diagnose_area_orientation.py static
    python cases_dynamic/diagnose_area_orientation.py three_d
    python cases_dynamic/diagnose_area_orientation.py list
    python cases_dynamic/diagnose_area_orientation.py census droplet2d laneR_bare
    python cases_dynamic/diagnose_area_orientation.py census all2d --out FILE.json
    python cases_dynamic/diagnose_area_orientation.py census laneR_bare \
        --orientation dual_midpoint           # the run itself on the legacy rule

``static``
    Builder, sheared, jittered and reconnected 2D meshes and the setup
    meshes of the 2D cases: every directed edge of every vertex.
``three_d``
    The 3D sources (the ``batch_e_star`` cache and the ``p_ij`` ring) on a
    jittered box and on the 3D droplet setup: same quantities, and how
    many fan triangles the cache orients one by one against the walk.
``census CASE...``
    Runs the case through its integrator and scans, before EVERY force
    evaluation, the area vectors of the vertices that are about to be
    integrated (the integrators are wrapped, nothing else changes).
    ``--orientation`` replaces ``area_orientation`` on every 2D preset and
    on the arms built here, i.e. the run is ``preset.replace(...)``, and
    the scan reads the vectors with that rule.  It reaches every run whose
    force is built by ``methods.dudt_fn``: the laneL / R / P arms, hp2d*,
    hydro2d*, and since laneW (2026-10-05) the multiphase cases as well
    (the droplet, dam-break, electrolysis and shearing-plate setups take
    ``methods=`` and build their force from it; until laneW they built
    their own ``partial(multiphase_dudt_i, ...)`` and ran the library
    default whatever the preset said, laneO known limit 1).

As a pytest plugin it counts per test (no behaviour change)::

    python -m pytest ddgclib/tests -q -p no:cacheprovider \
        -p cases_dynamic.diagnose_area_orientation
    AREA_ORIENTATION_JSON=FILE.json python -m pytest ...   # also as JSON

Nothing is written unless ``--out`` or ``AREA_ORIENTATION_JSON`` names a
file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_LEGACY, _FIXED = 'dual_midpoint', 'primal_edge'


# ----------------------------------------------------------------------
# the two orientation rules, from the shared dual vertices
# ----------------------------------------------------------------------
def rule_vectors(v_i, v_j):
    """``(A_dual_midpoint, A_primal_edge)`` of the directed edge
    ``(v_i, v_j)`` from the two dual vertices the endpoints share (the
    non-periodic 2D branch of ``dual_area_vector``); ``None`` when they
    share fewer than two."""
    vd = list(v_i.vd.intersection(v_j.vd))
    if len(vd) < 2:
        return None
    e = vd[1].x_a[:2] - vd[0].x_a[:2]
    A = np.array([-e[1], e[0]])
    mid = 0.5 * (vd[0].x_a[:2] + vd[1].x_a[:2])
    legacy = -A if np.dot(A, v_i.x_a[:2] - mid) > 0 else A
    edge = -A if np.dot(A, v_j.x_a[:2] - v_i.x_a[:2]) < 0 else A
    return legacy, edge


def _edge(v, nb, HC, dim):
    """``x_nb - x_v``, minimum image under periodic axes."""
    d = np.array(nb.x_a[:dim] - v.x_a[:dim], dtype=float)
    axes = getattr(HC, '_periodic_axes', None)
    if axes:
        for ax in axes:
            lo, hi = HC._periodic_bounds[ax]
            d[ax] -= round(d[ax] / (hi - lo)) * (hi - lo)
    return d


def _new_counts() -> dict[str, Any]:
    return dict(n_vertices=0, n_edges=0, n_rules_differ=0, n_against=0,
                n_antisym=0, n_zero=0, n_closed_cells=0, closure_max=0.0,
                closure_rel_max=0.0, ref_err_max=None, ref_err_rel_max=None,
                affected=[])


def scan(HC, verts, dim: int = 2, out: dict | None = None,
         reference: bool = True, tag: Any = None,
         source: Callable | None = None,
         orientation: str | None = None) -> dict[str, Any]:
    """Area-vector census of the directed edges at *verts* on the duals
    as they are.  Accumulates into *out* (returned).  *source*
    ``(v, nb) -> A or None`` replaces ``dual_area_vector`` (the 3D cache);
    *orientation* is handed to ``dual_area_vector`` (default: the library
    default rule).

    ``n_rules_differ``  the two orientation rules give different vectors
                        (2D, non-periodic)
    ``n_against``       the LIBRARY vector (with *orientation*) points
                        against its edge
    ``n_antisym``       ``|A_ij + A_ji| > 1e-12 |A_ij|``
    ``n_zero``          the library vector is zero
    ``closure_max``     largest ``|sum_j A_ij|`` of a cell that is not on
                        the hull (``closure_rel_max``: over ``sum_j |A_ij|``)
    ``ref_err_max``     largest ``|A_ij - A_ij(simplex cache)|`` over all
                        edges, hull cells included (needs ``HC._simplices``;
                        not under periodic axes); ``ref_err_rel_max``: over
                        the norm of the exact vector
    ``affected``        ``(tag, x_i, on hull)`` of the vertices with an
                        edge on which the rules differ (first 200)
    """
    from ddgclib.operators import stress

    out = _new_counts() if out is None else out
    periodic = bool(getattr(HC, '_periodic_axes', None))
    exact = getattr(stress, 'simplex_area_vectors', None)
    use_ref = (reference and exact is not None and not periodic
               and getattr(HC, '_simplices', None) is not None)
    if source is None:
        kw = {} if orientation is None else {'orientation': orientation}

        def source(a, b):
            return stress.dual_area_vector(a, b, HC, dim, **kw)
    for v in verts:
        out['n_vertices'] += 1
        ref = None
        if use_ref:
            ref = {id(w): a for w, a in zip(*exact(v, HC, dim))}
        total = np.zeros(dim)
        norm = 0.0
        hit = False
        for nb in v.nn:
            A = source(v, nb)
            B = source(nb, v)
            An = float(np.linalg.norm(A))
            out['n_edges'] += 1
            out['n_zero'] += An == 0.0
            out['n_against'] += float(A @ _edge(v, nb, HC, dim)) < 0.0
            if B is not None:
                out['n_antisym'] += float(np.linalg.norm(A + B)) > 1e-12 * An
            if dim == 2 and not periodic:
                rules = rule_vectors(v, nb)
                if rules is not None and not np.array_equal(*rules):
                    out['n_rules_differ'] += 1
                    hit = True
            if ref is not None:
                R = ref.get(id(nb), np.zeros(dim))
                err = float(np.linalg.norm(A - R))
                out['ref_err_max'] = max(out['ref_err_max'] or 0.0, err)
                Rn = float(np.linalg.norm(R))
                if Rn > 0.0:
                    out['ref_err_rel_max'] = max(
                        out['ref_err_rel_max'] or 0.0, err / Rn)
            total += A
            norm += An
        if hit and len(out['affected']) < 200:
            out['affected'].append(
                (tag, [float(c) for c in v.x_a[:dim]],
                 bool(getattr(v, 'boundary', False))))
        if not getattr(v, 'boundary', False) and norm > 0.0:
            res = float(np.linalg.norm(total))
            out['n_closed_cells'] += 1
            out['closure_max'] = max(out['closure_max'], res)
            out['closure_rel_max'] = max(out['closure_rel_max'], res / norm)
    return out


# ----------------------------------------------------------------------
# census along a run: scan before every force evaluation
# ----------------------------------------------------------------------
class Census:
    """Context manager.  Wraps ``_do_retopologize`` (to learn the mesh)
    and ``_compute_accel`` (to scan the vertices it is about to evaluate)
    of the dynamic integrators; both call the originals unchanged."""

    def __init__(self, every: int = 1, reference: bool = True,
                 orientation: str | None = None):
        self.every = max(1, every)
        self.reference = reference
        self.orientation = orientation
        self.counts = _new_counts()
        self.n_evals = 0
        self.n_scanned = 0
        self.first_eval_differ = None
        self.evals_with_differ = 0
        self.HC = None
        self.dim = None

    def __enter__(self):
        from ddgclib.dynamic_integrators import _integrators_dynamic as mod
        self._mod = mod
        self._orig = (mod._do_retopologize, mod._compute_accel)
        census, (retopo, accel) = self, self._orig

        def do_retopologize(HC, bV, dim, *args, **kwargs):
            census.HC, census.dim = HC, dim
            return retopo(HC, bV, dim, *args, **kwargs)

        def compute_accel(dudt_fn, verts, workers=None, **dudt_kwargs):
            census._evaluate(verts)
            return accel(dudt_fn, verts, workers, **dudt_kwargs)

        mod._do_retopologize, mod._compute_accel = do_retopologize, compute_accel
        return self

    def __exit__(self, *exc):
        self._mod._do_retopologize, self._mod._compute_accel = self._orig
        return False

    def _evaluate(self, verts) -> None:
        k = self.n_evals
        self.n_evals += 1
        if self.HC is None or self.dim not in (2, 3) or k % self.every:
            return
        before = self.counts['n_rules_differ']
        scan(self.HC, verts, self.dim, self.counts, self.reference, tag=k,
             orientation=self.orientation if self.dim == 2 else None)
        self.n_scanned += 1
        if self.counts['n_rules_differ'] > before:
            self.evals_with_differ += 1
            if self.first_eval_differ is None:
                self.first_eval_differ = k

    def report(self) -> dict[str, Any]:
        c = dict(self.counts)
        aff = c.pop('affected')
        c.update(n_force_evals=self.n_evals, n_scanned=self.n_scanned,
                 first_eval_rules_differ=self.first_eval_differ,
                 evals_with_rules_differ=self.evals_with_differ)
        if aff:
            xs = np.array([a[1] for a in aff])
            c['affected_first'] = aff[:5]
            c['affected_x_range'] = [xs.min(axis=0).tolist(),
                                     xs.max(axis=0).tolist()]
            c['affected_on_hull'] = int(sum(a[2] for a in aff))
            c['affected_listed'] = len(aff)
        return c


# ----------------------------------------------------------------------
# runs: name -> (default steps, fn(steps, orientation) -> scalars)
# ----------------------------------------------------------------------
def _oriented(methods, orientation):
    return (methods if orientation is None
            else methods.replace(area_orientation=orientation))


def _dd(name: str):
    """A case of ``diagnose_determinism.py`` (it reads ``PRESETS``, which
    ``_cmd_census`` replaces for ``--orientation``)."""
    def run(steps, orientation):
        from cases_dynamic import diagnose_determinism as dd
        _HC, scalars = dd.CASES[name][1](steps, lambda HC, bV: None, None)
        return scalars
    return run


def _lane_l(frozen_set: str):
    """The wall-collapse reproducer of lane L (test_frozen_set.py)."""
    def run(steps, orientation):
        from ddgclib.tests.test_frozen_set import TestHagenPoiseuille2D as T
        T.N_STEPS = steps
        report, n_on_wall = T._run(frozen_set)
        drop = next((i for i in range(1, len(n_on_wall))
                     if n_on_wall[i] < n_on_wall[i - 1]), None)
        return {'n_frozen': report['n_frozen'], 'n_moved': report['n_moved'],
                'max_displacement': report['max_displacement'],
                'first_wall_drop': drop, 'n_on_wall_end': n_on_wall[-1]}
    return run


def _lane_r(arm: str):
    """The closed box with an EOS of lanes K / R
    (test_single_phase_remap.py::TestBoxDecayWithEOS)."""
    def run(steps, orientation):
        from ddgclib.tests import test_single_phase_remap as t
        methods = _oriented(getattr(t, arm), orientation)
        _HC, ke, umax = t._run(methods, n_steps=steps)
        fin = np.isfinite(ke)
        return {'ke0': float(ke[0]), 'ke_end': float(ke[-1]),
                'ke_max_over_ke0': float(np.nanmax(ke) / ke[0]),
                'umax_max_over_U0': float(np.nanmax(umax) / t.U0),
                'first_step_ke_above_2ke0': next(
                    (int(i) for i in range(len(ke))
                     if not fin[i] or ke[i] > 2.0 * ke[0]), None),
                'all_finite': bool(fin.all())}
    return run


def _lane_p(arm: str):
    """The free-surface column of lane P
    (test_material_delaunay.py::TestColumnThroughTheIntegrator)."""
    def run(steps, orientation):
        from ddgclib.methods import SolverMethods
        from ddgclib.tests.test_material_delaunay import (
            TestColumnThroughTheIntegrator as T,
        )
        material = SolverMethods(dim=2, connectivity='delaunay_material',
                                 remap='conservative', redistribute_mass=True)
        methods = {'material': material,
                   'convex': material.replace(connectivity='delaunay'),
                   'dual_only': SolverMethods(dim=2, connectivity='dual_only'),
                   }[arm]
        umax, volume, p_bottom, n_frozen = T()._run(
            _oriented(methods, orientation))
        return {'umax_max': float(umax.max()), 'volume': float(volume),
                'p_bottom': float(p_bottom), 'n_frozen': n_frozen}
    return run


def _lane_p_column(arm: str):
    """The shipped 2D column (refinement 3, 145 vertices, uniform-density
    start) on the arms of laneP's diagnose_column.py; ``steps`` is the
    number of acoustic times here (laneP quoted the convex-hull + remap
    arm at 42.10 m/s after 2.98 t_ac)."""
    def run(n_tac, orientation):
        from cases_dynamic.Hydrostatic_column.src._column import (
            CASES, build_column, remap_arm, run_column,
        )
        from ddgclib.methods import PRESETS
        kw = CASES['hydrostatic_2D']
        col = build_column(2, kw['n_refine'], H=kw['H'],
                           side_walls=kw['side_walls'], ic='drop')
        m = PRESETS['hydrostatic_2D']
        methods = {'convex_remap': m.replace(connectivity='delaunay',
                                             remap='conservative',
                                             redistribute_mass=True),
                   'remap': remap_arm(m), 'preset': m}[arm]
        res = run_column(col, _oriented(methods, orientation), n_tac=n_tac)
        i = int(np.argmax(res['umax']))
        return {'umax_peak': float(res['umax'][i]),
                't_peak_tac': float(res['t'][i] / col.params['t_ac']),
                'umax_end': float(res['umax'][-1]),
                'volume': float(sum(v.dual_vol for v in col.HC.V)),
                'n_steps': int(res['n_steps'])}
    return run


RUNS: dict[str, tuple[int, Callable]] = {
    # --- oscillating droplet (2D): presets, refinement 2 / 2 ---
    'droplet2d': (300, _dd('droplet2d')),
    'droplet2d_bare': (300, _dd('droplet2d_bare')),
    'droplet2d_dual_only': (300, _dd('droplet2d_dual_only')),
    # --- other 2D presets ---
    'dam_break2d': (400, _dd('dam_break2d')),
    'electrolysis2d': (100, _dd('electrolysis2d')),
    'shearing2d': (10, _dd('shearing2d')),               # periodic branch
    'hp2d': (500, _dd('hp2d')),
    'hp2d_centred_twopoint': (500, _dd('hp2d_centred_twopoint')),
    'hydro2d_remap': (300, _dd('hydro2d_remap')),
    'hydro2d_density_diffusion': (300, _dd('hydro2d_density_diffusion')),
    # --- the arms of the pinned tests of lanes L, R and P ---
    'laneL_hull': (300, _lane_l('hull')),
    'laneL_membership': (300, _lane_l('membership')),
    'laneR_bare': (200, _lane_r('BARE')),
    'laneR_remap': (200, _lane_r('REMAP')),
    'laneR_dual_only': (200, _lane_r('DUAL_ONLY')),
    'laneP_material': (0, _lane_p('material')),
    'laneP_convex': (0, _lane_p('convex')),
    'laneP_dual_only': (0, _lane_p('dual_only')),
    # the shipped column (refinement 3); "steps" = acoustic times
    'laneP_column_convex_remap': (3, _lane_p_column('convex_remap')),
    'laneP_column_remap': (3, _lane_p_column('remap')),
    # --- 3D (the cache or the p_ij ring, whichever the run reads) ---
    'droplet3d': (10, _dd('droplet3d')),
    'hp3d_centred': (30, _dd('hp3d_centred')),
    'hydro3d': (30, _dd('hydro3d')),
}
_ALL2D = [n for n in RUNS if '3d' not in n]


def run_census(name: str, steps: int | None = None,
               orientation: str | None = None, every: int = 1) -> dict:
    default, fn = RUNS[name]
    steps = default if steps is None else steps
    t0 = time.perf_counter()
    scalars: dict[str, Any] = {}
    error = None
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        with Census(every=every, orientation=orientation) as census:
            try:
                scalars = fn(steps, orientation)
            except Exception as e:  # noqa: BLE001 - a run may blow up
                error = f"{type(e).__name__}: {str(e)[:200]}"
    row = {'case': name, 'steps': steps,
           'orientation': orientation or 'library default',
           'census': census.report(), 'scalars': scalars,
           'seconds': round(time.perf_counter() - t0, 1)}
    if error:
        row['error'] = error
    return row


# ----------------------------------------------------------------------
# static meshes
# ----------------------------------------------------------------------
def _linear_pressure_error(HC, dim: int) -> float:
    """Largest ``|F + V g| / (V |g|)`` of the centred force of a linear
    NODAL pressure over the cells that are not on the hull."""
    from ddgclib.operators.stress import stress_force
    g = np.array([0.7, -1.3, 0.4][:dim])
    saved = {}
    for v in HC.V:
        saved[v] = (getattr(v, 'p', None), getattr(v, 'u', None))
        v.p = float(g @ v.x_a[:dim])
        v.u = np.zeros(dim)
    worst = 0.0
    for v in HC.V:
        if getattr(v, 'boundary', False) or not getattr(v, 'dual_vol', 0.0):
            continue
        F = stress_force(v, dim=dim, mu=0.0, HC=HC)
        worst = max(worst, float(np.linalg.norm(F + v.dual_vol * g))
                    / (v.dual_vol * float(np.linalg.norm(g))))
    for v, (p, u) in saved.items():
        v.p, v.u = p, u
    return worst


def sheared_jittered(refinement: int = 3, shear: float = 0.3,
                     jitter: float = 0.2, seed: int = 0, L: float = 2.0):
    """The mesh of the lane H reproducer (and of the test that was its
    strict xfail): a channel whose interior vertices are sheared by
    ``shear * 4 y (1 - y)`` in x and jittered by ``jitter`` mean spacings,
    reconnected by the library Delaunay retopology."""
    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
    from ddgclib.geometry.domains import rectangle
    rng = np.random.default_rng(seed)
    res = rectangle(L=L, h=1.0, refinement=refinement, flow_axis=0)
    HC, bV = res.HC, res.bV
    h = (L / sum(1 for _ in HC.V)) ** 0.5
    HC.V.move_all([
        (v, (v.x_a[0] + shear * 4 * v.x_a[1] * (1 - v.x_a[1])
             + jitter * h * rng.uniform(-1, 1),
             v.x_a[1] + jitter * h * rng.uniform(-1, 1)))
        for v in list(HC.V) if v not in bV])
    _retopologize(HC, bV, 2)
    return HC, bV


def _static_meshes():
    from hyperct.ddg import compute_vd

    from ddgclib.geometry.domains import disk, rectangle
    from ddgclib.operators.stress import cache_dual_volumes

    def built(res):
        compute_vd(res.HC, method='barycentric')
        cache_dual_volumes(res.HC, 2)
        return res.HC

    yield 'rectangle L 2 refinement 3 (builder)', built(
        rectangle(L=2.0, h=1.0, refinement=3, flow_axis=0))
    yield 'disk R 1 refinement 3 (builder)', built(disk(R=1.0, refinement=3))
    yield 'sheared 0.3 + jitter 0.2, refinement 3 (lane H mesh)', \
        sheared_jittered(3)[0]
    yield 'sheared 0.3 + jitter 0.2, refinement 4', sheared_jittered(4)[0]
    yield 'jitter 0.2 only, refinement 3', sheared_jittered(3, shear=0.0)[0]
    yield 'sheared 0.6 + jitter 0.3, refinement 3, seed 1', \
        sheared_jittered(3, shear=0.6, jitter=0.3, seed=1)[0]
    # setup meshes of the 2D cases
    from cases_dynamic.oscillating_droplet.src import _params as P
    from cases_dynamic.oscillating_droplet.src._setup import (
        setup_oscillating_droplet,
    )
    for refine in ((2, 2), (3, 3)):
        HC = setup_oscillating_droplet(
            dim=2, R0=P.R0, epsilon=P.epsilon, l=P.l, rho_d=P.rho_d,
            rho_o=P.rho_o, mu_d=P.mu_d, mu_o=P.mu_o, gamma=P.gamma,
            K_d=P.K_d, K_o=P.K_o, L_domain=P.L_domain,
            refinement_outer=refine[0], refinement_droplet=refine[1])[0]
        yield f'oscillating droplet 2D setup, refinement {refine}', HC
    from cases_dynamic.dam_break.src import _params as D
    from cases_dynamic.dam_break.src._setup import setup_dam_break_multiphase
    yield 'dam break 2D setup, refinement 3', setup_dam_break_multiphase(
        dim=2, a=D.a, L=D.L, H=D.H, W=D.W, col_w=D.col_w, col_h=D.col_h,
        col_d=D.col_d, rho_l=D.rho_l, rho_g=D.rho_g, mu_l=D.mu_l, mu_g=D.mu_g,
        gamma=D.gamma, K_l=D.K_l, K_g=D.K_g, g=D.g,
        gravity_axis=D.gravity_axis, P_atm=D.P_atm, n_refine=3,
        alpha_art=D.alpha_art)[0]
    from cases_dynamic.Hydrostatic_column.src._column import (
        CASES, build_column,
    )
    kw = CASES['hydrostatic_2D']
    yield 'hydrostatic column 2D, refinement 3', build_column(
        2, 3, H=kw['H'], side_walls=kw['side_walls'], ic='drop').HC
    from ddgclib.tests.test_single_phase_remap import _box
    yield 'closed box of lanes K / R, refinement 2', _box()[0]


def _cmd_static(args) -> int:
    from hyperct.ddg import compute_vd

    from ddgclib.operators.stress import cache_dual_volumes
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for label, HC in _static_meshes():
            # the droplet setup deforms the interface after its duals
            # were computed (the first retopology of a run refreshes
            # them): scan every mesh on fresh duals
            compute_vd(HC, method='barycentric')
            cache_dual_volumes(HC, 2)
            c = scan(HC, list(HC.V), 2)
            c.pop('affected')
            c['linear_pressure_err_max'] = _linear_pressure_error(HC, 2)
            rows.append({'mesh': label, **c})
    print(f"{'mesh':<52} {'edges':>6} {'differ':>6} {'against':>7} "
          f"{'antisym':>7} {'zero':>4} {'closure':>9} {'vs simplex':>10} "
          f"{'lin. p':>9}")
    for r in rows:
        ref = r['ref_err_max']
        print(f"{r['mesh']:<52} {r['n_edges']:>6} {r['n_rules_differ']:>6} "
              f"{r['n_against']:>7} {r['n_antisym']:>7} {r['n_zero']:>4} "
              f"{r['closure_max']:>9.2e} "
              f"{'n/a' if ref is None else format(ref, '.2e'):>10} "
              f"{r['linear_pressure_err_max']:>9.2e}")
    if args.out:
        _dump(args.out, rows)
    return 0


# ----------------------------------------------------------------------
# 3D sources
# ----------------------------------------------------------------------
def _fan_census(HC, verts) -> dict[str, Any]:
    """The ``batch_e_star`` cache orients every triangle of the fan
    (edge midpoint, two consecutive shared dual vertices) by itself.
    Count the fans in which the triangles do not all turn the same way
    along the walk: there the forced sum is not the area of the polygon."""
    from hyperct.ddg._operators import _walk_fan_3d
    n_fans = n_mixed = 0
    worst = 0.0
    for v in verts:
        for nb in v.nn:
            try:
                tris = _walk_fan_3d(v, nb, HC)
            except (KeyError, IndexError):
                continue
            if not tris:
                continue
            a = np.array([0.5 * np.cross(m - p, q - p) for m, p, q in tris])
            d = nb.x_a[:3] - v.x_a[:3]
            s = a @ d
            n_fans += 1
            if (s > 0).any() and (s < 0).any():
                n_mixed += 1
                forced = np.abs(s).sum()
                worst = max(worst, float(abs(forced - abs(s.sum())) / forced))
    return {'n_fans': n_fans, 'n_fans_mixed_orientation': n_mixed,
            'forced_minus_signed_rel_max': worst}


def _scan_3d(HC, label: str) -> dict[str, Any]:
    """Both 3D sources at the vertices off the hull: the vectors
    ``stress_force`` reads when the ``batch_e_star`` cache is present, and
    the ``p_ij`` ring it reads without it."""
    from ddgclib.operators.stress import dual_area_vector
    interior = [v for v in HC.V if not getattr(v, 'boundary', False)]
    row: dict[str, Any] = {'mesh': label, 'n_interior': len(interior)}
    cache = getattr(HC, '_edge_area_cache', None)
    keys = ('n_edges', 'n_against', 'n_antisym', 'n_zero', 'closure_rel_max',
            'ref_err_max', 'ref_err_rel_max')
    if cache is not None:
        c = scan(HC, interior, 3,
                 source=lambda a, b: cache.get(id(a), {}).get(id(b)))
        row['cache'] = {k: c[k] for k in keys}
    HC._edge_area_cache = None           # dual_area_vector = the p_ij ring
    try:
        c = scan(HC, interior, 3,
                 source=lambda a, b: dual_area_vector(a, b, HC, 3))
    finally:
        HC._edge_area_cache = cache
    row['p_ij ring'] = {k: c[k] for k in keys}
    row['fan'] = _fan_census(HC, interior)
    return row


def _cmd_three_d(args) -> int:
    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
    from ddgclib.geometry.domains import box
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for amplitude, shear in ((0.0, 0.0), (0.03, 0.0), (0.08, 0.0),
                                 (0.05, 0.3)):
            rng = np.random.default_rng(1)
            res = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=2)
            HC, bV = res.HC, res.bV
            HC.V.move_all([
                (v, tuple(v.x_a + amplitude * rng.uniform(-1, 1, 3)
                          + [shear * 4 * v.x_a[1] * (1 - v.x_a[1]), 0, 0]))
                for v in list(HC.V) if v not in bV])
            _retopologize(HC, bV, 3)
            rows.append(_scan_3d(HC, f'box refinement 2, jitter {amplitude}, '
                                     f'shear {shear}'))
        from cases_dynamic.oscillating_droplet.src import _params as P
        from cases_dynamic.oscillating_droplet.src._setup import (
            setup_oscillating_droplet,
        )
        HC, bV = setup_oscillating_droplet(
            dim=3, R0=P.R0, epsilon=P.epsilon, l=P.l, rho_d=P.rho_d,
            rho_o=P.rho_o, mu_d=P.mu_d, mu_o=P.mu_o, gamma=P.gamma,
            K_d=P.K_d, K_o=P.K_o, L_domain=P.L_domain, refinement_outer=2,
            refinement_droplet=2)[:2]
        _retopologize(HC, bV, 3)
        rows.append(_scan_3d(HC, 'oscillating droplet 3D setup (2, 2), '
                                 'after one Delaunay retopology'))
    for r in rows:
        print(json.dumps(r))
    if args.out:
        _dump(args.out, rows)
    return 0


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def _dump(path: str, obj) -> None:
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1, default=float)
        fh.write('\n')
    print(f"-> {path}")


def _cmd_census(args) -> int:
    names = []
    for n in args.case:
        names += _ALL2D if n == 'all2d' else [n]
    if args.orientation:
        from ddgclib.methods import PRESETS
        for key, m in list(PRESETS.items()):
            if m.dim == 2:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    PRESETS[key] = m.replace(area_orientation=args.orientation)
    rows = []
    for name in names:
        row = run_census(name, args.steps, args.orientation, args.every)
        rows.append(row)
        c = row['census']
        print(f"\n== {name}: {row['steps']} steps, orientation "
              f"{row['orientation']}, {row['seconds']} s"
              + (f", ERROR {row['error']}" if 'error' in row else ''))
        print(f"  force evaluations {c['n_force_evals']} (scanned "
              f"{c['n_scanned']}), directed edges {c['n_edges']}")
        print(f"  rules differ on {c['n_rules_differ']} edges in "
              f"{c['evals_with_rules_differ']} evaluations (first: "
              f"{c['first_eval_rules_differ']}); library vector against its "
              f"edge {c['n_against']}, not antisymmetric {c['n_antisym']}, "
              f"zero {c['n_zero']}")
        print(f"  closure of the cells off the hull: max {c['closure_max']:.3e}"
              f" (relative {c['closure_rel_max']:.3e}); against the simplex "
              f"cache: {c['ref_err_max']}")
        if 'affected_x_range' in c:
            print(f"  affected vertices (first {c['affected_listed']}): x "
                  f"range {c['affected_x_range']}, on the hull "
                  f"{c['affected_on_hull']}")
        print('  ' + ', '.join(f"{k}={v!r}" for k, v in row['scalars'].items()))
    if args.out:
        _dump(args.out, rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    s = sub.add_parser('static')
    s.add_argument('--out', default='')
    t = sub.add_parser('three_d')
    t.add_argument('--out', default='')
    c = sub.add_parser('census')
    c.add_argument('case', nargs='+', choices=sorted(RUNS) + ['all2d'])
    c.add_argument('--steps', type=int)
    c.add_argument('--every', type=int, default=1,
                   help='scan every N-th force evaluation')
    c.add_argument('--orientation', choices=[_LEGACY, _FIXED])
    c.add_argument('--out', default='')
    args = ap.parse_args(argv)
    if args.cmd == 'list':
        for name, (steps, _fn) in RUNS.items():
            print(f"{name:<28} {steps} steps")
        return 0
    return {'static': _cmd_static, 'three_d': _cmd_three_d,
            'census': _cmd_census}[args.cmd](args)


# ----------------------------------------------------------------------
# pytest plugin: per-test count of the calls on which the rules differ
# ----------------------------------------------------------------------
_PLUGIN: dict[str, Any] = {'current': None, 'tests': {}}


def _probe(original):
    def dual_area_vector(v_i, v_j, HC, dim=3, *args, **kwargs):
        if (dim == 2 and _PLUGIN['current'] is not None
                and not getattr(HC, '_periodic_axes', None)):
            row = _PLUGIN['tests'].setdefault(
                _PLUGIN['current'], {'calls_2d': 0, 'rules_differ': 0,
                                     'where': []})
            row['calls_2d'] += 1
            rules = rule_vectors(v_i, v_j)
            if rules is not None and not np.array_equal(*rules):
                row['rules_differ'] += 1
                if len(row['where']) < 2000:
                    row['where'].append(
                        [float(v_i.x_a[0]), float(v_i.x_a[1]),
                         bool(getattr(v_i, 'boundary', False))])
        return original(v_i, v_j, HC, dim, *args, **kwargs)
    dual_area_vector.__wrapped__ = original
    return dual_area_vector


def pytest_configure(config):
    from ddgclib.operators import (
        gradient, multiphase_stress, stabilisation, stress,
    )
    import ddgclib.operators as ops
    probe = _probe(stress.dual_area_vector)
    for mod in (stress, multiphase_stress, stabilisation, gradient, ops):
        if hasattr(mod, 'dual_area_vector'):
            mod.dual_area_vector = probe


def pytest_runtest_logstart(nodeid, location):
    _PLUGIN['current'] = nodeid


def pytest_runtest_logfinish(nodeid, location):
    _PLUGIN['current'] = None


def pytest_terminal_summary(terminalreporter):
    tests = _PLUGIN['tests']
    calls = sum(r['calls_2d'] for r in tests.values())
    differ = sum(r['rules_differ'] for r in tests.values())
    tr = terminalreporter
    tr.write_sep('=', 'dual area vector orientation census (2D, non-periodic)')
    tr.write_line(f"{differ} calls on which dual_midpoint and primal_edge "
                  f"differ, of {calls} calls in {len(tests)} tests")
    for nodeid, r in sorted(tests.items(), key=lambda kv: -kv[1]['rules_differ']):
        if r['rules_differ']:
            hull = sum(w[2] for w in r['where'])
            tr.write_line(f"  {r['rules_differ']:>6} of {r['calls_2d']:>8}  "
                          f"{nodeid}  (v_i on the hull: {hull} of "
                          f"{len(r['where'])} listed)")
    path = os.environ.get('AREA_ORIENTATION_JSON')
    if path:
        _dump(path, {k: r for k, r in tests.items() if r['rules_differ']}
              | {'_total': {'calls_2d': calls, 'rules_differ': differ,
                            'tests': len(tests)}})


if __name__ == '__main__':
    sys.exit(main())
