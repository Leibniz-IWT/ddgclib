"""Pinned smoke tests and physical checks of the shearing-plate droplet
case (laneG, 2026-10-06).

The case runs through ``PRESETS['shearing_plate_droplet_2D']`` /
``['_3D']`` (``connectivity='periodic'``) on library paths:

* the anisotropic rescale of the outer box is
  ``ddgclib.geometry.domains.rescale_droplet_box`` (one ``move_all``, the
  droplet and its shell untouched: until laneG the uniform scale put the
  outer vertices at ``(0, +-L/2)`` on the droplet poles and the setup's
  evict loop deleted both interface poles);
* ``retopologize_periodic`` keeps only the simplices whose centroid lies
  in the fundamental domain (a consistent periodic triangulation) and
  measures the seam simplices with minimum-image coordinates (the total
  dual volume was 1.94 x the box);
* the setup resets the outer-phase masses on the periodic duals before
  the Young-Laplace preload (the outer phase sat at -100 Pa).

Physical checks (integrated, ``ddgclib.analytical``): the quiescent
droplet (``U_wall = 0``) holds ``gamma / R`` to the discretisation error
over the short window with the sum of forces at round-off (no spurious
seam force), and under the shipped weak shear it keeps its volume and
its interface for the full short window.  Pins: 2D short window
(droplet volume, interface vertex count, deformation parameter, digest)
and the 3D setup plus a few steps.
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ddgclib.analytical import integrated_phase_pressure_jump  # noqa: E402
from ddgclib.methods import PRESETS  # noqa: E402

from cases_dynamic.shearing_plate_droplet.src import _params as sp  # noqa: E402
from cases_dynamic.shearing_plate_droplet.src._plot_helpers import (  # noqa: E402
    compute_diagnostics,
)
from cases_dynamic.shearing_plate_droplet.src._setup import (  # noqa: E402
    setup_shearing_plate_droplet,
)

TOL = 1e-9


def _build(dim, ro, rd, U=sp.U_wall, methods=None):
    kw = dict(dim=dim, R0=sp.R0, L_x=sp.L_x, L_y=sp.L_y, U_wall=U,
              rho_d=sp.rho_d, rho_o=sp.rho_o, mu_d=sp.mu_d, mu_o=sp.mu_o,
              gamma=sp.gamma, K_d=sp.K_d, K_o=sp.K_o,
              refinement_outer=ro, refinement_droplet=rd,
              methods=methods or PRESETS[f'shearing_plate_droplet_{dim}D'])
    if dim == 3:
        kw['L_z'] = sp.L_z
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return setup_shearing_plate_droplet(**kw)


def _dt(HC, dim):
    c_s = float(np.sqrt(sp.K_o / sp.rho_o))
    dx_min = min(float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
                 for v in HC.V for nb in v.nn
                 if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15)
    return 0.1 * dx_min / c_s


def _digest(HC, dim):
    state = sorted((tuple(v.x_a[:dim]), tuple(v.u[:dim]), float(v.m),
                    tuple(float(p) for p in v.p_phase)) for v in HC.V)
    return hashlib.sha256(repr(state).encode()).hexdigest()[:16]


def _droplet_volume(HC):
    return sum(float(v.dual_vol_phase[1]) for v in HC.V)


def _is_plate(v):
    return abs(abs(v.x_a[1]) - sp.L_y) < TOL


def _net_force(HC, dudt_fn, dim):
    """Sum of m a over the free vertices (the plates are driven by the
    BC), over every vertex, and the largest |m a|, from the force the
    integrator reads."""
    F = np.zeros(dim)
    F_total = np.zeros(dim)
    f_max = 0.0
    for v in HC.V:
        f = float(v.m) * dudt_fn(v)[:dim]
        F_total += f
        if _is_plate(v):
            continue
        F += f
        f_max = max(f_max, float(np.linalg.norm(f)))
    return F, F_total, f_max


def _run(HC, bV, mps, bc_set, dudt_fn, params, methods, n_steps, dim,
         every=None, record=None):
    dt = _dt(HC, dim)

    def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if record is not None and every and (step + 1) % every == 0:
            record(step + 1, t)

    methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                      bc_set=bc_set, callback=cb, mps=mps,
                      domain_bounds=params['domain_bounds'])
    return dt


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

class TestSetup2D:
    def test_mesh_and_rescale(self):
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params = \
            _build(2, 3, 3)
        assert len(list(HC.V)) == 304
        assert sum(1 for v in HC.V if v.is_interface) == 32
        assert len(bV) == 16 and len(groups['top_wall']) == 8
        assert params['n_rescaled'] == 88
        # the droplet and its shell are untouched, no outer vertex inside
        assert all(np.linalg.norm(v.x_a[:2]) > sp.R0 + 1e-6
                   for v in HC.V if v.phase == 0)
        # the box faces are exactly on the channel extents
        assert max(abs(v.x_a[0]) for v in HC.V) == pytest.approx(sp.L_x)
        assert max(abs(v.x_a[1]) for v in HC.V) == pytest.approx(sp.L_y)

    def test_periodic_duals_tile_the_box(self):
        """Seam simplices measured with minimum-image coordinates and one
        image per periodic simplex: the dual volumes sum to the box and
        no facet has three owners (before laneG: 1.94 x the box, 8
        facets with 3 owners on this mesh)."""
        HC, bV, *_ = _build(2, 3, 3)
        box = 4.0 * sp.L_x * sp.L_y
        assert sum(float(v.dual_vol) for v in HC.V) == pytest.approx(
            box, rel=1e-12)
        owners = Counter()
        for s in HC._simplices:
            for f in combinations(s, 2):
                owners[frozenset(id(v) for v in f)] += 1
        counts = Counter(owners.values())
        assert set(counts) == {1, 2} and counts[1] == 16   # the plates
        # no interior vertex tagged boundary
        assert all(not v.boundary or v in bV for v in HC.V)

    def test_young_laplace_preload_and_reentrant(self):
        HC, *_ = _build(2, 3, 3)
        assert integrated_phase_pressure_jump(HC, 1, 0) == pytest.approx(
            sp.gamma / sp.R0, abs=1e-9)
        d1 = _digest(HC, 2)
        HC2, *_ = _build(2, 3, 3)      # a second call in one process
        assert _digest(HC2, 2) == d1

    def test_setup_requires_periodic(self):
        with pytest.raises(ValueError, match='periodic'):
            _build(2, 2, 2, methods=PRESETS['oscillating_droplet_2D'])


# ---------------------------------------------------------------------------
# physical checks
# ---------------------------------------------------------------------------

class TestQuiescentDroplet2D:
    """U_wall = 0: the preloaded droplet is an equilibrium of the
    discrete operator up to the discretisation error of the circle."""

    def test_no_spurious_seam_force_and_laplace(self):
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params = \
            _build(2, 3, 3, U=0.0)
        F, F_total, f_max = _net_force(HC, dudt_fn, 2)
        assert np.linalg.norm(F) < 1e-12 * f_max
        assert np.linalg.norm(F_total) < 1e-12 * f_max
        V0 = _droplet_volume(HC)
        n_steps = 200
        _run(HC, bV, mps, bc_set, dudt_fn, params,
             PRESETS['shearing_plate_droplet_2D'], n_steps, 2)
        F, F_total, f_max = _net_force(HC, dudt_fn, 2)
        # the pairwise fluxes across the seam are antisymmetric: the sum
        # over every cell (plates included) is round-off
        assert np.linalg.norm(F_total) < 1e-12 * f_max
        # the free vertices feel the plates' reaction only
        assert np.linalg.norm(F) < 1e-7 * f_max
        dp = integrated_phase_pressure_jump(HC, 1, 0)
        assert abs(dp / (sp.gamma / sp.R0) - 1.0) < 0.05
        assert abs(_droplet_volume(HC) / V0 - 1.0) < 1e-3
        assert sum(1 for v in HC.V if v.is_interface) == 32
        # the interface relaxes toward the discrete equilibrium at a few
        # mm/s (0.06 U_wall); no vertex moves at the plate speed
        assert max(float(np.linalg.norm(v.u[:2])) for v in HC.V) < 0.1 * sp.U_wall


# ---------------------------------------------------------------------------
# pins
# ---------------------------------------------------------------------------

PIN_2D_SHORT = {        # laneG 2026-10-06, refinement 3/3, dt 3.0330e-5
    'n_steps': 1649,
    'V_over_V_exact': 0.9839915243700096,
    'n_interface': 32,
    'D': 0.0014132936582887858,
    'digest': '3c66efcb929b9e1c',
}


@pytest.mark.slow
class TestShortWindow2D:
    def test_pinned_short_window(self):
        """The shipped short window (refinement 3/3, t = 0.05 s, 1649
        steps): the droplet keeps its volume and its 32 interface
        vertices (before laneG: first interface loss at step 183, |u|
        295 U_wall at t = 0.05 s)."""
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params = \
            _build(2, 3, 3)
        V0 = _droplet_volume(HC)
        _run(HC, bV, mps, bc_set, dudt_fn, params,
             PRESETS['shearing_plate_droplet_2D'], PIN_2D_SHORT['n_steps'], 2)
        d = compute_diagnostics(HC, dim=2)
        assert d['n_interface'] == PIN_2D_SHORT['n_interface']
        assert abs(_droplet_volume(HC) / V0 - 1.0) < 1e-4   # volume kept
        assert _droplet_volume(HC) / (np.pi * sp.R0 ** 2) == pytest.approx(
            PIN_2D_SHORT['V_over_V_exact'], rel=1e-9)
        assert d['D'] == pytest.approx(PIN_2D_SHORT['D'], rel=1e-9)
        assert _digest(HC, 2) == PIN_2D_SHORT['digest']


PIN_3D_SETUP = {        # laneG 2026-10-06, refinement 1/2, dt 4.5045e-5
    'n_vertices': 306,
    'n_interface': 98,
    'n_plates': 8,
    'n_steps': 5,
    'digest': '303c19a7f707bb2a',
}


@pytest.mark.slow
class TestSetupAndSteps3D:
    def test_3d_setup_and_five_steps(self):
        """The 3D setup (refinement 1/2, the short runner's) no longer
        collides in the rescale; five steps through the preset."""
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params = \
            _build(3, 1, 2)
        assert len(list(HC.V)) == PIN_3D_SETUP['n_vertices']
        assert sum(1 for v in HC.V if v.is_interface) == \
            PIN_3D_SETUP['n_interface']
        assert len(bV) == PIN_3D_SETUP['n_plates']
        assert integrated_phase_pressure_jump(HC, 1, 0) == pytest.approx(
            2.0 * sp.gamma / sp.R0, abs=1e-9)
        _run(HC, bV, mps, bc_set, dudt_fn, params,
             PRESETS['shearing_plate_droplet_3D'], PIN_3D_SETUP['n_steps'], 3)
        assert sum(1 for v in HC.V if v.is_interface) == \
            PIN_3D_SETUP['n_interface']
        assert _digest(HC, 3) == PIN_3D_SETUP['digest']
