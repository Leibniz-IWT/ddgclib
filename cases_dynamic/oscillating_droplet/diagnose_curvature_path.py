#!/usr/bin/env python3
"""Curvature-path A/B for the oscillating droplet: the method axis
``curvature_path`` (laneM, 2026-10-05).

MEASUREMENT ONLY.  Every arm is ``PRESETS[...]`` or
``preset.replace(curvature_path=...)``; the arm names are the registered
values of the axis (``AXES['curvature_path'].keys()``), so an arm that
leaves the registry leaves this driver with it.  The force of every run
is bound by ``methods.dudt_fn`` through
``setup_oscillating_droplet(methods=)``.

The value ``'stokes'`` (the conormal boundary integral of the interface
over the barycentric dual cell, Probe 2 of 2026-05-27) was removed from
the library in laneM: on a piecewise-linear surface the integral of the
conormal along the two dual segments inside a triangle is
``n_T x (x_k - x_j) / 2`` whatever the interior point, i.e. the gradient
of the triangle area with respect to ``x_i``, which IS the cotangent
form, so the two stencils agree to round-off on every mesh, static or
moving.  :func:`stokes_reference` keeps that integral here, without the
coordinate-keyed cache the library version had, so the equality can be
re-measured against ``hndA_i_interface`` at any time (``static`` and
``moving``).

Sub-commands (run from the repo root with the ddg env python)::

    python cases_dynamic/oscillating_droplet/diagnose_curvature_path.py static [--dim 2|3|all]
    python cases_dynamic/oscillating_droplet/diagnose_curvature_path.py moving [--dim 2|3|all] [--n-steps N]
    python cases_dynamic/oscillating_droplet/diagnose_curvature_path.py nonmanifold
    python cases_dynamic/oscillating_droplet/diagnose_curvature_path.py floors [--dim 2|3|all] [--arm NAME|all] [--n-steps N]
    python cases_dynamic/oscillating_droplet/diagnose_curvature_path.py dynamic --dim 2|3 --arm NAME [--perturb 1e-15 --perturb-seed K]

``static``: the frozen static droplet (eps 0; 2D refinement 3/3, 3D 2/2)
and the perturbed one (eps 0.05) at step 0: per interface vertex the
surface-tension force of every registered stencil and of the Stokes
reference against the default, the full-force floor (A.5.a) of every
stencil, and the cost of one evaluation over the interface.
``moving``: the same comparison at every force evaluation of a short run
through the presets (3D ``oscillating_droplet_3D`` and ``_3D_delaunay``,
2D ``oscillating_droplet_2D`` and ``_2D_bare_delaunay``; the preset's
``dudt_fn`` is wrapped, the run itself is the preset's), and after
laneI's jitter + Delaunay rebuild (the interface triangles change, the
apex cache is rebuilt).
``nonmanifold``: the same after the jitter, per vertex, split by whether
the vertex touches an interface edge with three or more interface
triangles (where the pre-laneM ``hndA_i_interface`` dropped the third).
``floors``: the A.5.b floors through
``static_droplet_floor_{2D,3D}.replace(curvature_path=arm)``
(``diagnose_a5_bisection.run_a5b``).
``dynamic``: the full run through
``oscillating_droplet_{2D,3D}.replace(curvature_path=arm)``; writes
``score_laneM_{dim}d_{arm}.json`` (with the ``methods`` block),
``methods_laneM_{dim}d_{arm}.json`` and ``diags_laneM_{dim}d_{arm}.json``
and prints l2 / tail / mass / wall next to the pinned baseline.

``--out DIR`` redirects every output (default ``results/laneM/``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (  # noqa: E402
    R0, epsilon as EPS_CASE, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet, t_end_2d, t_end_3d,
)
from cases_dynamic.oscillating_droplet.src._setup import (  # noqa: E402
    setup_oscillating_droplet,
)
from ddgclib._curvatures_heron import _pad3  # noqa: E402
from ddgclib.methods import AXES, PRESETS, record_methods  # noqa: E402
from ddgclib.operators.multiphase_stress import (  # noqa: E402
    _interface_surface_tension,
)

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_CASE_DIR, 'results', 'laneM')
_BASELINES = os.path.join(_CASE_DIR, 'baselines')

AXIS = AXES['curvature_path']
DEFAULT = AXIS.default
ARMS: tuple[str, ...] = tuple(AXIS.keys())
REFINE = {2: (n_refine_outer, n_refine_droplet), 3: (2, 2)}
FLOOR_PRESET = {2: 'static_droplet_floor_2D', 3: 'static_droplet_floor_3D'}
RUN_PRESET = {2: 'oscillating_droplet_2D', 3: 'oscillating_droplet_3D'}
MOVING_PRESETS = {
    2: ('oscillating_droplet_2D', 'oscillating_droplet_2D_bare_delaunay'),
    3: ('oscillating_droplet_3D', 'oscillating_droplet_3D_delaunay'),
}
BASELINE = {2: 'baseline_oscillation.json', 3: 'baseline_oscillation_3d.json'}


def arm_methods(preset: str, arm: str):
    methods = PRESETS[preset]
    if arm == DEFAULT:
        return methods
    return methods.replace(curvature_path=arm,
                           label=f"{methods.label} [laneM arm {arm}]",
                           notes=f"laneM arm {arm!r} of preset {preset!r}: "
                                 f"curvature_path={arm!r}")


def _setup(methods, dim: int, eps: float):
    ro, rd = REFINE[dim]
    return setup_oscillating_droplet(
        dim=dim, R0=R0, epsilon=eps, l=l, rho_d=rho_d, rho_o=rho_o,
        mu_d=mu_d, mu_o=mu_o, gamma=gamma, K_d=K_d, K_o=K_o,
        L_domain=L_domain, refinement_outer=ro, refinement_droplet=rd,
        methods=methods,
    )


def _iface(HC) -> list:
    return [v for v in HC.V if getattr(v, 'is_interface', False)]


def _tri_ids(HC) -> frozenset:
    x2v = {v.x: v for v in HC.V}
    return frozenset(frozenset(id(x2v[k]) for k in t)
                     for t in getattr(HC, 'interface_triangles', ()))


# ---------------------------------------------------------------------------
# The removed 'stokes' stencil, as a measurement reference (no cache)
# ---------------------------------------------------------------------------
def stokes_reference(v, interface_set, HC, gamma_: float, x_to_v: dict) -> np.ndarray:
    """``F_st_i = gamma * oint_{d Gamma_i} nu dl`` over the boundary of the
    barycentric dual cell of ``v`` inside the interface triangles
    (``HC.interface_triangles``): per triangle the two segments edge
    midpoint -> centroid -> edge midpoint, conormal ``nu = n_T x t``
    oriented away from ``v``.  The library function
    ``integrated_hndA_i_interface`` of 2026-05-27 to 2026-10-05, with the
    coordinate map passed in instead of cached on ``HC``."""
    F = np.zeros(3)
    if gamma_ == 0.0:
        return F
    v_key = v.x
    x_i = _pad3(v.x_a)
    for tri_key in HC.interface_triangles:
        if v_key not in tri_key:
            continue
        other = [k for k in tri_key if k != v_key]
        if len(other) != 2:
            continue
        v_j = x_to_v.get(other[0])
        v_k = x_to_v.get(other[1])
        if v_j is None or v_k is None:
            continue
        if v_j not in interface_set or v_k not in interface_set:
            continue
        x_j = _pad3(v_j.x_a)
        x_k = _pad3(v_k.x_a)
        normal2A = np.cross(x_j - x_i, x_k - x_i)
        twoA = np.linalg.norm(normal2A)
        if twoA < 1e-30:
            continue
        n_tri = normal2A / twoA
        c = (x_i + x_j + x_k) / 3.0
        m_ij = 0.5 * (x_i + x_j)
        m_ik = 0.5 * (x_i + x_k)
        for p_start, p_end in ((m_ij, c), (c, m_ik)):
            seg = p_end - p_start
            L = float(np.linalg.norm(seg))
            if L < 1e-30:
                continue
            t = seg / L
            nu = np.cross(n_tri, t)
            seg_mid = 0.5 * (p_start + p_end)
            if np.dot(nu, seg_mid - x_i) < 0.0:
                nu = -nu
            F += gamma_ * L * nu
    return F


def _fst(v, dim: int, mps, HC, path: str) -> np.ndarray:
    return np.asarray(_interface_surface_tension(
        v, dim, mps, HC=HC, curvature_path=path), dtype=float)


def _ref_forces(HC, dim: int, mps) -> dict[int, np.ndarray]:
    iface = _iface(HC)
    iset = set(iface)
    x_to_v = {vv.x: vv for vv in HC.V}
    gam = mps.get_gamma_pair(0, 1)
    out = {}
    for v in iface:
        F = stokes_reference(v, iset | {v}, HC, gam, x_to_v)
        out[id(v)] = F[:dim]
    return out


def _stats(a) -> dict:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {'n': 0}
    return {'n': int(a.size), 'median': float(np.median(a)),
            'max': float(np.max(a)), 'mean': float(np.mean(a))}


def compare_stencils(HC, dim: int, mps, full_force: bool = True) -> dict:
    """Every stencil against the default on the current mesh state."""
    from cases_dynamic.oscillating_droplet.diagnose_a5_bisection import (
        _max_interface_force,
    )
    iface = _iface(HC)
    F_int = {id(v): _fst(v, dim, mps, HC, DEFAULT) for v in iface}
    scale = max(float(np.linalg.norm(f)) for f in F_int.values())
    arms: dict[str, dict[int, np.ndarray]] = {}
    for arm in ARMS:
        if arm == DEFAULT:
            continue
        arms[arm] = {id(v): _fst(v, dim, mps, HC, arm) for v in iface}
    if dim == 3:
        arms['stokes_reference'] = _ref_forces(HC, 3, mps)
    out: dict = {'n_interface': len(iface), 'max_abs_F_st_default': scale,
                 'arms': {}}
    for name, Fa in arms.items():
        diffs, angles, ratios = [], [], []
        for k, fi in F_int.items():
            fa = Fa[k]
            diffs.append(float(np.linalg.norm(fa - fi)))
            ni, na = float(np.linalg.norm(fi)), float(np.linalg.norm(fa))
            if ni > 1e-30 * scale and na > 1e-30 * scale:
                cosang = float(np.dot(fa, fi) / (na * ni))
                angles.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
                ratios.append(na / ni)
        out['arms'][name] = {
            'max_abs_diff': float(max(diffs)),
            'max_rel_diff': float(max(diffs) / scale) if scale > 0 else 0.0,
            'angle_deg': _stats(angles),
            'magnitude_ratio': _stats(ratios),
        }
    if full_force:
        out['max_abs_F_full'] = {
            arm: _max_interface_force(HC, dim, mps, curvature_path=arm)[0]
            for arm in ARMS
        }
    return out


def time_stencils(HC, dim: int, mps, repeats: int = 5) -> dict:
    """Best-of-``repeats`` wall time of one evaluation of each stencil
    over every interface vertex (caches warm)."""
    iface = _iface(HC)
    iset = set(iface)
    gam = mps.get_gamma_pair(0, 1)
    out = {}
    for arm in ARMS:
        _ = [_fst(v, dim, mps, HC, arm) for v in iface]  # warm the caches
        best = float('inf')
        for _r in range(repeats):
            t0 = time.perf_counter()
            for v in iface:
                _fst(v, dim, mps, HC, arm)
            best = min(best, time.perf_counter() - t0)
        out[arm] = best
    if dim == 3:
        x_to_v = {vv.x: vv for vv in HC.V}
        best = float('inf')
        for _r in range(repeats):
            t0 = time.perf_counter()
            for v in iface:
                stokes_reference(v, iset | {v}, HC, gam, x_to_v)
            best = min(best, time.perf_counter() - t0)
        out['stokes_reference'] = best
    return {k: {'s_per_interface_sweep': t,
                'ms_per_vertex': 1e3 * t / max(len(iface), 1)}
            for k, t in out.items()}


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
def run_static(dims) -> dict:
    out = {'arms': list(ARMS), 'meshes': {}}
    for dim in dims:
        for tag, eps, preset in (('eps0', 0.0, FLOOR_PRESET[dim]),
                                 ('eps0.05', EPS_CASE, RUN_PRESET[dim])):
            methods = PRESETS[preset]
            HC, bV, mps, *_ = _setup(methods, dim, eps)
            key = f'{dim}d_{tag}'
            rec = compare_stencils(HC, dim, mps)
            rec['cost'] = time_stencils(HC, dim, mps)
            rec['preset'] = preset
            rec['refinement'] = REFINE[dim]
            out['meshes'][key] = rec
            print(json.dumps({key: {
                'max_abs_F_st_default': rec['max_abs_F_st_default'],
                'arms': {a: {'max_rel_diff': r['max_rel_diff'],
                             'angle_max_deg': r['angle_deg'].get('max')}
                         for a, r in rec['arms'].items()},
                'max_abs_F_full': rec['max_abs_F_full'],
                'cost_ms_per_vertex': {a: c['ms_per_vertex']
                                       for a, c in rec['cost'].items()},
            }}), flush=True)
    return out


# ---------------------------------------------------------------------------
# moving
# ---------------------------------------------------------------------------
def _jitter(HC, bV, amp: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for v in list(HC.V):
        if v in bV:
            continue
        x = v.x_a.copy()
        x[:3] += amp * rng.standard_normal(3)
        HC.V.move(v, tuple(x))


class _ForceProbe:
    """Wraps the preset's ``dudt_fn``: at every force evaluation of an
    interface vertex (right after the step's retopology, when the
    interface sub-complex and the positions agree) it evaluates every
    stencil on that vertex and records the largest deviation from the
    default per step.  The integrator still integrates the preset force.
    (The integrator's callback runs after the move, when the
    coordinate-keyed ``HC.interface_triangles`` are already stale, so
    the comparison cannot be made there.)"""

    def __init__(self, dudt_fn, dim: int, mps, HC):
        self.dudt_fn, self.dim, self.mps, self.HC = dudt_fn, dim, mps, HC
        self.names = [a for a in ARMS if a != DEFAULT]
        if dim == 3:
            self.names.append('stokes_reference')
        self.records: list[dict] = []
        self._cur: dict | None = None
        self._x_to_v: dict = {}
        self._last_tri = None
        self.gam = mps.get_gamma_pair(0, 1)

    def end_step(self) -> None:
        if self._cur is not None:
            s = self._cur['scale']
            for r in self._cur['arms'].values():
                r['max_rel_diff'] = r['max_abs_diff'] / s if s > 0 else 0.0
            self.records.append(self._cur)
        self._cur = None

    def __call__(self, v):
        if getattr(v, 'is_interface', False):
            if self._cur is None:
                self._x_to_v = {vv.x: vv for vv in self.HC.V}
                tri = _tri_ids(self.HC) if self.dim == 3 else None
                self._cur = {
                    'arms': {n: {'max_abs_diff': 0.0, 'max_angle_deg': 0.0}
                             for n in self.names},
                    'scale': 0.0,
                    'interface_triangles_changed': (
                        self._last_tri is not None and tri != self._last_tri),
                }
                self._last_tri = tri
            F_int = _fst(v, self.dim, self.mps, self.HC, DEFAULT)
            ni = float(np.linalg.norm(F_int))
            self._cur['scale'] = max(self._cur['scale'], ni)
            for name in self.names:
                if name == 'stokes_reference':
                    iset = {nb for nb in v.nn if getattr(nb, 'is_interface', False)}
                    Fa = stokes_reference(v, iset | {v}, self.HC, self.gam,
                                          self._x_to_v)[:3]
                else:
                    Fa = _fst(v, self.dim, self.mps, self.HC, name)
                r = self._cur['arms'][name]
                r['max_abs_diff'] = max(r['max_abs_diff'],
                                        float(np.linalg.norm(Fa - F_int)))
                na = float(np.linalg.norm(Fa))
                if ni > 0 and na > 0:
                    cosang = float(np.dot(Fa, F_int) / (na * ni))
                    r['max_angle_deg'] = max(r['max_angle_deg'], float(
                        np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
        return self.dudt_fn(v)


def run_moving(dims, n_steps: int) -> dict:
    out: dict = {'n_steps': n_steps, 'runs': {}}
    for dim in dims:
        for preset in MOVING_PRESETS[dim]:
            methods = PRESETS[preset]
            HC, bV, mps, bc_set, dudt_fn, _r, params = _setup(methods, dim, EPS_CASE)
            c_s = np.sqrt(K_d / rho_d)
            dx_min = min(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
                         for v in HC.V for nb in v.nn
                         if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15)
            dt = min(0.25 * dx_min / c_s, 0.5 * np.sqrt(rho_d * dx_min**3 / gamma))
            probe = _ForceProbe(dudt_fn, dim, mps, HC)
            moved = [0.0]
            x0 = {id(v): v.x_a.copy() for v in HC.V}

            def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
                probe.end_step()
                moved[0] = max(moved[0], max(
                    float(np.linalg.norm(v.x_a - x0[id(v)]))
                    for v in HC_cb.V if id(v) in x0))

            t0 = time.perf_counter()
            methods.integrate(HC, bV, probe, dt=dt, n_steps=n_steps,
                              bc_set=bc_set, callback=cb, mps=mps)
            wall = time.perf_counter() - t0
            per_step = probe.records
            summary = {
                name: {
                    'max_rel_diff_over_steps': max(
                        r['arms'][name]['max_rel_diff'] for r in per_step),
                    'angle_max_deg_over_steps': max(
                        r['arms'][name]['max_angle_deg'] for r in per_step),
                } for name in probe.names
            }
            row = {'preset': preset, 'config': methods.to_dict(), 'dt': dt,
                   'n_force_steps': len(per_step),
                   'wall_s': wall, 'max_displacement_over_R0': moved[0] / R0,
                   'summary': summary, 'per_step': per_step}
            if dim == 3:
                row['interface_triangle_changes'] = int(sum(
                    r['interface_triangles_changed'] for r in per_step))
                # laneI fixture: a 1e-3 jitter (10 % R0 at 1/1, the audit
                # probe) changes the interface triangles under a full
                # Delaunay rebuild; here on the 2/2 mesh.
                if methods.connectivity == 'delaunay':
                    before = probe._last_tri   # at the last force evaluation
                    _jitter(HC, bV, 1e-3)
                    retopo = methods.retopologize_fn(mps=mps)
                    retopo(HC, bV, 3)
                    apex_dropped = not hasattr(HC, '_interface_edge_to_apex')
                    rec = compare_stencils(HC, 3, mps, full_force=False)
                    rec['interface_triangles_changed'] = _tri_ids(HC) != before
                    rec['apex_cache_dropped_by_retopo'] = apex_dropped
                    row['after_jitter_delaunay'] = rec
            out['runs'][f'{dim}d_{preset}'] = row
            print(json.dumps({f'{dim}d_{preset}': {
                'summary': summary, 'wall_s': wall,
                'max_displacement_over_R0': row['max_displacement_over_R0'],
                'interface_triangle_changes': row.get('interface_triangle_changes'),
                'after_jitter_delaunay': {
                    k: v for k, v in row.get('after_jitter_delaunay', {}).items()
                    if k != 'arms'} | {
                    'arms': {a: r['max_rel_diff'] for a, r in
                             row.get('after_jitter_delaunay', {}).get('arms', {}).items()}},
            }}, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
# nonmanifold: where (and whether) the two 3D stencils differ
# ---------------------------------------------------------------------------
def run_nonmanifold(amps=(1e-3, 2e-4)) -> dict:
    """The 3D delaunay preset at 2/2 after a jitter of ``amp`` and a
    Delaunay rebuild: per interface vertex the Stokes reference against
    ``integrated``, split by whether the vertex touches an interface edge
    with more than two interface triangles (a non-manifold edge) or with
    one (a boundary edge).  laneM: with 1e-3 (10 % R0) the relabelled
    interface has 138 vertices and ONE edge with three triangles; before
    laneM ``hndA_i_interface`` summed the first two apexes of an edge and
    the two vertices of that edge were off by 22 %; since laneM every
    triangle counts and the two stencils agree to 1e-15 everywhere."""
    from collections import Counter

    from ddgclib._curvatures_heron import _apex_via_interface_triangles

    out: dict = {}
    for amp in amps:
        methods = PRESETS['oscillating_droplet_3D_delaunay']
        HC, bV, mps, *_ = _setup(methods, 3, EPS_CASE)
        _jitter(HC, bV, amp)
        methods.retopologize_fn(mps=mps)(HC, bV, 3)
        iface = _iface(HC)
        iset = set(iface)
        x2v = {v.x: v for v in HC.V}
        gam = mps.get_gamma_pair(0, 1)
        edge_count: Counter = Counter()
        for t in HC.interface_triangles:
            ks = list(t)
            for i in range(3):
                for j in range(i + 1, 3):
                    edge_count[frozenset((ks[i], ks[j]))] += 1
        rows = []
        scale = 0.0
        for v in iface:
            F_int = _fst(v, 3, mps, HC, DEFAULT)
            F_ref = stokes_reference(v, iset | {v}, HC, gam, x2v)[:3]
            scale = max(scale, float(np.linalg.norm(F_int)))
            nm = bd = n_apex = 0
            for nb in v.nn:
                if nb not in iset:
                    continue
                c = edge_count.get(frozenset((v.x, nb.x)), 0)
                nm += c > 2
                bd += c == 1
                apex = _apex_via_interface_triangles(HC, v, nb)
                n_apex = max(n_apex, len(apex) if apex else 0)
            rows.append((float(np.linalg.norm(F_ref - F_int)), nm, bd, n_apex))
        arr = np.array(rows)
        rel = arr[:, 0] / scale
        manifold = (arr[:, 1] == 0) & (arr[:, 2] == 0)
        rec = {
            'jitter': amp, 'n_interface': len(iface),
            'n_interface_triangles': len(HC.interface_triangles),
            'edges_with_3_or_more_triangles': int(sum(
                1 for c in edge_count.values() if c > 2)),
            'edges_with_1_triangle': int(sum(
                1 for c in edge_count.values() if c == 1)),
            'n_vertices_manifold': int(manifold.sum()),
            'n_vertices_nonmanifold': int((~manifold).sum()),
            'max_apexes_at_one_edge': int(arr[:, 3].max()),
            'max_rel_diff_manifold': (float(rel[manifold].max())
                                      if manifold.any() else None),
            'max_rel_diff_nonmanifold': (float(rel[~manifold].max())
                                         if (~manifold).any() else None),
        }
        out[f'jitter_{amp:g}'] = rec
        print(json.dumps(rec), flush=True)
    return out


# ---------------------------------------------------------------------------
# floors (A.5.b)
# ---------------------------------------------------------------------------
def run_floors(dims, arms, n_steps: int | None) -> dict:
    from cases_dynamic.oscillating_droplet.diagnose_a5_bisection import run_a5b
    out: dict = {}
    for dim in dims:
        ro, rd = REFINE[dim]
        n = n_steps if n_steps is not None else (30 if dim == 2 else 20)
        for arm in arms:
            methods = arm_methods(FLOOR_PRESET[dim], arm)
            t0 = time.perf_counter()
            h = run_a5b(dim=dim, refinement_outer=ro, refinement_droplet=rd,
                        n_steps=n, methods=methods)
            wall = time.perf_counter() - t0
            row = {k: h[k] for k in ('max_abs_F_peak', 'max_abs_F_end',
                                     'max_abs_F_history', 'mass_rel_drift',
                                     'dt', 'n_steps')}
            row['max_abs_F_step0'] = h['max_abs_F_history'][0]
            row['config'] = methods.to_dict()
            row['wall_s_incl_setup'] = wall
            out[f'{dim}d_{arm}'] = row
            print(json.dumps({f'{dim}d_{arm}': {
                k: row[k] for k in ('max_abs_F_step0', 'max_abs_F_peak',
                                    'max_abs_F_end', 'mass_rel_drift',
                                    'wall_s_incl_setup')}}), flush=True)
    return out


# ---------------------------------------------------------------------------
# dynamic (full run, mirror of oscillating_droplet_{2D,3D}.py main)
# ---------------------------------------------------------------------------
def run_dynamic(dim: int, arm: str, out_dir: str, perturb: float = 0.0,
                perturb_seed: int = 0) -> dict:
    from cases_dynamic.oscillating_droplet.src._analytical import (
        lamb_damping_rate, lamb_damping_rate_two_fluid, rayleigh_frequency,
        two_fluid_omega_beta_2d,
    )
    from cases_dynamic.oscillating_droplet.src._metrics import (
        add_two_fluid_reference, oscillation_score, oscillation_score_3d,
        save_score,
    )
    from cases_dynamic.oscillating_droplet.src._plot_helpers import (
        compute_diagnostics,
    )

    preset = RUN_PRESET[dim]
    suffix = f'_laneM_{dim}d_{arm}'
    if perturb:
        suffix += f'_perturb{perturb:g}_s{perturb_seed}'
    methods = arm_methods(preset, arm)
    print(methods.describe(), flush=True)
    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim, rho_outer=rho_o)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)

    HC, bV, mps, bc_set, dudt_fn, _r, params = _setup(methods, dim, EPS_CASE)
    bound = dudt_fn.keywords.get('curvature_path', DEFAULT)
    assert bound == arm, (bound, arm)
    n_perturbed = 0
    if perturb:
        # the round-off yardstick of protocol rule 8 (laneT): shift the
        # free vertices by perturb * scale * r before the run
        from cases_dynamic.diagnose_determinism import _perturb
        n_perturbed = _perturb(HC, bV, perturb, perturb_seed)
        print(f"[{dim}d {arm}] perturbed {n_perturbed} vertices by "
              f"{perturb:g} (seed {perturb_seed})", flush=True)

    c_s = np.sqrt(K_d / rho_d)
    dx_min = min(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
                 for v in HC.V for nb in v.nn
                 if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15)
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min**3 / gamma) if gamma > 0 else 1.0)
    t_end_case = t_end_2d if dim == 2 else t_end_3d
    t_end = min(t_end_case, 5.0 / beta if beta > 0 else 0.01)
    n_steps = int(t_end / dt) + 1
    record_every = max(1, n_steps // (200 if dim == 2 else 100))
    print(f"[{dim}d {arm}] dt={dt:.2e}, n_steps={n_steps}, t_end={t_end:.4f}",
          flush=True)

    diag_list: list[dict] = []

    def record(t):
        d = (compute_diagnostics(HC, dim=2) if dim == 2
             else compute_diagnostics(HC, dim=3, polar_axis='z'))
        d['t'] = float(t)
        if dim == 3:
            d['total_dual_vol'] = float(sum(
                float(getattr(v, 'dual_vol', 0.0) or 0.0) for v in HC.V))
        diag_list.append(d)

    record(0.0)
    t_wall0 = time.perf_counter()
    step_times: list[float] = []
    t_last = [time.perf_counter()]

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        now = time.perf_counter()
        step_times.append(now - t_last[0])
        if step % record_every == 0:
            record(t)
            if step % (record_every * 10) == 0:
                d = diag_list[-1]
                print(f"  [{dim}d {arm}] step {step}/{n_steps} t={t:.4e} "
                      f"R_max={d['R_max']:.6f} KE={d['KE']:.4e} "
                      f"mass={d['total_mass']:.6e} wall={now - t_wall0:.0f}s",
                      flush=True)
        t_last[0] = time.perf_counter()

    t_final = methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                                bc_set=bc_set, callback=callback, mps=mps)
    wall = time.perf_counter() - t_wall0
    record(t_final)

    if dim == 2:
        score = oscillation_score(diag_list, R0=R0, epsilon=EPS_CASE, l=l,
                                  omega=omega, beta=beta)
        omega_tf, beta_tf = two_fluid_omega_beta_2d(
            l, gamma, mu_d, rho_d, R0, mu_outer=mu_o, rho_outer=rho_o)
        beta_tf_energy = lamb_damping_rate_two_fluid(
            l, mu_d, rho_d, R0, mu_outer=mu_o, rho_outer=rho_o)
        score = add_two_fluid_reference(
            score, diag_list, R0=R0, epsilon=EPS_CASE, l=l,
            omega_two_fluid=omega_tf, beta_two_fluid=beta_tf,
            beta_energy=beta_tf_energy)
    else:
        score = oscillation_score_3d(diag_list, R0=R0, epsilon=EPS_CASE, l=l,
                                     omega=omega, beta=beta,
                                     r_boundary=L_domain)
    ro, rd = REFINE[dim]
    score['refinement_outer'] = ro
    score['refinement_droplet'] = rd
    score['retopo_policy'] = 'delaunay_remap' if dim == 2 else 'dual_only'
    score['box_shift'] = params['box_shift']
    score['laneM'] = {
        'arm': arm, 'curvature_path_bound': bound, 'base_preset': preset,
        'perturb': perturb, 'perturb_seed': perturb_seed,
        'n_perturbed': n_perturbed,
        'KE_max': float(max(d['KE'] for d in diag_list)),
        'R_max_end': float(diag_list[-1]['R_max']),
        'wall_s': wall, 'wall_per_step_s': wall / n_steps,
        'median_step_s': float(np.median(step_times)),
    }
    save_score(os.path.join(out_dir, f'score{suffix}.json'), score,
               methods=methods)
    record_methods(
        os.path.join(out_dir, f'methods{suffix}.json'), methods, HC,
        extra={'arm': arm, 'base_preset': preset,
               'perturb': perturb, 'perturb_seed': perturb_seed,
               'retopo_policy': score['retopo_policy'], 'dt': dt,
               'n_steps': n_steps, 't_end': t_end,
               'refinement_outer': ro, 'refinement_droplet': rd,
               'box_shift': params['box_shift'], 'K_d': K_d, 'K_o': K_o,
               'wall_s': wall},
    )
    diag_scalars = [{k: float(v) for k, v in dd.items() if np.ndim(v) == 0}
                    for dd in diag_list]
    with open(os.path.join(out_dir, f'diags{suffix}.json'), 'w') as f:
        json.dump({'kind': 'diag_series', 'diags': diag_scalars}, f)

    keys = ['l2_error_normalized', 'tail_growth', 'mass_drift', 'summary']
    if dim == 3:
        keys += ['R_max_peak', 'apex_l2_error_normalized']
    print(json.dumps({k: score[k] for k in keys}), flush=True)
    print(json.dumps(score['laneM']), flush=True)
    try:
        with open(os.path.join(_BASELINES, BASELINE[dim])) as f:
            base = json.load(f)
        print(json.dumps({'pinned_baseline': {
            k: base[k] for k in ('l2_error_normalized', 'tail_growth')}}),
            flush=True)
    except OSError:
        pass
    return score


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('task', choices=('static', 'moving', 'nonmanifold',
                                     'floors', 'dynamic'))
    ap.add_argument('--dim', choices=('2', '3', 'all'), default='all')
    ap.add_argument('--arm', choices=ARMS + ('all',), default='all')
    ap.add_argument('--n-steps', type=int, default=None,
                    help='moving: steps per run (default 20); floors: '
                         'A.5.b steps (default 30 in 2D, 20 in 3D)')
    ap.add_argument('--out', default=_RESULTS,
                    help='output directory (default results/laneM/)')
    ap.add_argument('--perturb', type=float, default=0.0,
                    help='dynamic: shift the free vertices by EPS * scale '
                         'before the run (diagnose_determinism._perturb); '
                         'the spread under 1e-15 is the round-off yardstick')
    ap.add_argument('--perturb-seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dims = (2, 3) if a.dim == 'all' else (int(a.dim),)
    if a.task == 'static':
        out = run_static(dims)
        path = os.path.join(a.out, 'laneM_static.json')
    elif a.task == 'moving':
        out = run_moving(dims, a.n_steps if a.n_steps is not None else 20)
        path = os.path.join(a.out, 'laneM_moving.json')
    elif a.task == 'nonmanifold':
        out = run_nonmanifold()
        path = os.path.join(a.out, 'laneM_nonmanifold.json')
    elif a.task == 'floors':
        arms = ARMS if a.arm == 'all' else (a.arm,)
        out = run_floors(dims, arms, a.n_steps)
        tag = '' if a.arm == 'all' else f'_{a.arm}'
        tag += '' if a.dim == 'all' else f'_{a.dim}d'
        path = os.path.join(a.out, f'laneM_floors{tag}.json')
    else:
        if a.dim == 'all':
            ap.error('dynamic needs --dim 2 or 3')
        run_dynamic(int(a.dim), DEFAULT if a.arm == 'all' else a.arm, a.out,
                    perturb=a.perturb, perturb_seed=a.perturb_seed)
        sys.exit(0)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'wrote {path}')
