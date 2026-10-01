"""Tests for ``ddgclib.methods`` (method registry + SolverMethods wrapper).

The wrapper is a naming / plumbing / recording layer: it must build
exactly the partials and integrator kwargs the case runners built by
hand, so every pinned baseline stays bit-identical.  The tests here
prove that, lock the validation rules, and lock the two implicit
(dimension-gated) choices the audit of 2026-09-25 surfaced.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from functools import partial

import numpy as np
import pytest

from ddgclib.methods import (
    AXES, PRESETS, STATUSES, SolverMethods, axes_markdown, config_markdown,
    effective_methods, presets_markdown, record_methods,
)
from ddgclib.methods._config import _MULTI_ONLY


# ---------------------------------------------------------------------------
# registry self-consistency
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_every_axis_default_is_an_option(self):
        for ax in AXES.values():
            if ax.kind == 'choice':
                assert ax.default in ax.keys(), ax.name
            else:
                # numeric axes: default is the first (literal) option
                assert ax.options[0].key == ax.default, ax.name

    def test_explicit_axes_are_solver_methods_fields(self):
        names = {f for f in SolverMethods.__dataclass_fields__}
        for ax in AXES.values():
            if ax.explicit:
                assert ax.name in names, ax.name

    def test_reported_axes_are_resolved_by_effective_methods(self):
        from hyperct import Complex
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        eff = effective_methods(HC, 2)
        for ax in AXES.values():
            if not ax.explicit:
                assert ax.name in eff, ax.name
                val = eff[ax.name]
                # A value outside the registry is allowed only as an honest
                # "cannot resolve" string (fresh mesh, custom connectivity).
                assert val in ax.keys() or any(
                    tag in str(val) for tag in ('not yet called', 'n/a', 'custom')
                ), (ax.name, val)

    def test_statuses_are_vocabulary(self):
        for ax in AXES.values():
            for o in ax.options:
                assert o.status in STATUSES

    def test_code_anchors_point_at_existing_modules(self):
        import importlib
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for ax in AXES.values():
            for o in ax.options:
                first = o.where.split(',')[0].split(':')[0].split(' ')[0].strip()
                assert (root / first).exists(), f"{ax.name}/{o.key}: {first}"
        importlib.import_module('ddgclib.methods')

    def test_markdown_renders(self):
        md = axes_markdown()
        for ax in AXES.values():
            assert f"`{ax.name}`" in md
        pm = presets_markdown(PRESETS)
        for name in PRESETS:
            assert f"`{name}`" in pm
        assert '| axis | value | status |' in config_markdown(PRESETS['dam_break_2D'])

    def test_methods_md_matches_registry(self):
        """METHODS.md embeds the generated tables verbatim; regenerate with
        ``python -m ddgclib.methods --markdown`` after editing _axes.py or
        _presets.py."""
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        doc = (root / 'METHODS.md').read_text()
        assert axes_markdown() in doc, 'METHODS.md axis tables are stale'
        assert presets_markdown(PRESETS) in doc, 'METHODS.md preset table is stale'

    def test_cli_runs(self):
        out = subprocess.run(
            [sys.executable, '-m', 'ddgclib.methods', '--markdown'],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout
        assert '## presets' in out and '### `connectivity`' in out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_defaults_are_single_phase_symplectic_delaunay(self):
        m = SolverMethods(dim=2)
        assert (m.phases, m.integrator, m.connectivity) == (
            'single', 'symplectic_euler', 'delaunay')
        assert m.redistribute_mass is False and m.remap is None

    @pytest.mark.parametrize('kw', [
        dict(dim=4),
        dict(dim=2, phases='two'),
        dict(dim=2, integrator='leapfrog'),
        dict(dim=2, connectivity='delauny'),
        dict(dim=2, phases='multi', split_method='neighbor_count'),
        dict(dim=2, phases='multi', curvature_path='heron'),
        dict(dim=2, projection_every=0),
        dict(dim=2, projection_every=True),
        dict(dim=2, displacement_eps=0.0),
        dict(dim=2, merge_cdist=-1.0),
        dict(dim=2, workers=0),
        dict(dim=2, remesh_kwargs={'L_min': 0.1}),          # needs adaptive
        dict(dim=3, phases='multi', connectivity='adaptive'),
        dict(dim=2, connectivity='periodic'),               # needs periodic_axes
        dict(dim=2, connectivity='periodic', periodic_axes=(2,)),  # out of range
        dict(dim=2, periodic_axes=(0,)),                    # needs periodic
        # the bare refresh never redistributes or re-splits with 'exact'
        dict(dim=2, phases='multi', connectivity='dual_only_bare',
             redistribute_mass=True),
        dict(dim=2, phases='multi', connectivity='dual_only_bare',
             split_method='exact'),
        dict(dim=2, phases='multi', connectivity='dual_only_bare',
             remap='conservative', redistribute_mass=True),
        # multi-only fields on a single-phase config
        *[dict(dim=2, **{f: v}) for f, v in (
            ('remap', 'conservative'), ('projection_every', 2),
            ('split_method', 'exact'), ('curvature_path', 'csf_dual'))],
        # silent no-ops in the code become errors here
        dict(dim=2, phases='multi', connectivity='dual_only',
             remap='conservative', redistribute_mass=True),
        dict(dim=2, phases='multi', connectivity='delaunay',
             remap='conservative', redistribute_mass=False),
        dict(dim=2, phases='multi', connectivity='delaunay',
             redistribute_mass=True, projection_every=2),
        dict(dim=2, phases='multi', connectivity='dual_only',
             redistribute_mass=False, projection_every=2),
        dict(dim=2, phases='multi', connectivity='frozen',
             redistribute_mass=True, projection_every=2),
    ])
    def test_invalid_combinations_raise(self, kw):
        with pytest.raises(ValueError):
            SolverMethods(**kw)

    def test_multi_only_fields_match_registry(self):
        for f in _MULTI_ONLY:
            assert AXES[f].applies_to == 'multi'

    def test_valid_opt_in_combinations(self):
        SolverMethods(dim=2, phases='multi', connectivity='dual_only',
                      redistribute_mass=True, projection_every=5)
        SolverMethods(dim=2, phases='multi', connectivity='delaunay',
                      remap='conservative', redistribute_mass=True,
                      projection_every=2)
        SolverMethods(dim=2, phases='multi', connectivity='adaptive',
                      remap='conservative', redistribute_mass=True,
                      remesh_kwargs={'L_min': 1e-3})
        SolverMethods(dim=2, connectivity='periodic', periodic_axes=(0,))
        SolverMethods(dim=3, phases='multi', connectivity='periodic',
                      periodic_axes=(0, 2), redistribute_mass=True)
        SolverMethods(dim=2, phases='multi', connectivity='dual_only_bare')
        SolverMethods(dim=3, phases='multi', connectivity='frozen')

    def test_broken_status_warns(self):
        """Constructing a config with a 'broken' option warns.  No field
        option carries that status any more (stokes was fixed in laneI),
        so the mechanism is exercised through a patched option."""
        from unittest import mock
        from ddgclib.methods._axes import MethodOption
        broken = MethodOption('stokes', 'x', 'broken', 'ddgclib/methods/_axes.py',
                              'synthetic')
        real = SolverMethods._option

        def patched(axis, value):
            if axis == 'curvature_path' and value == 'stokes':
                return broken
            return real(axis, value)

        with mock.patch.object(SolverMethods, '_option', staticmethod(patched)):
            with pytest.warns(UserWarning, match="'broken'"):
                SolverMethods(dim=3, phases='multi', curvature_path='stokes')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            SolverMethods(dim=3, phases='multi', curvature_path='stokes')  # experimental now
            SolverMethods(dim=2, workers=4)   # experimental, not broken

    def test_status_lookup(self):
        m = PRESETS['oscillating_droplet_2D_projection2']
        assert m.status_of('projection_every') == 'opt-in'
        assert m.status_of('remap') == 'validated'
        assert PRESETS['oscillating_droplet_2D_bare_delaunay'].status_of(
            'connectivity') == 'validated'
        assert SolverMethods(dim=2, displacement_eps=1e-6).status_of(
            'displacement_eps') == 'measured-worse'

    def test_round_trip(self, tmp_path):
        for name, m in PRESETS.items():
            assert SolverMethods.from_dict(m.to_dict()) == m, name
            p = tmp_path / f'{name}.json'
            m.to_json(p)
            assert SolverMethods.from_json(p) == m
        m = SolverMethods(dim=2, connectivity='periodic', periodic_axes=(0,))
        assert SolverMethods.from_dict(json.loads(json.dumps(m.to_dict()))) == m

    def test_replace_revalidates(self):
        m = PRESETS['oscillating_droplet_2D']
        assert m.replace(projection_every=2).projection_every == 2
        with pytest.raises(ValueError):
            m.replace(connectivity='dual_only')   # remap under dual_only

    def test_describe_lists_every_explicit_axis(self):
        text = PRESETS['oscillating_droplet_2D'].describe()
        for name, _ in PRESETS['oscillating_droplet_2D'].explicit_items():
            assert name in text
        single = SolverMethods(dim=2).describe()
        assert 'split_method' not in single


# ---------------------------------------------------------------------------
# builders: single-phase
# ---------------------------------------------------------------------------

class TestSinglePhaseBuilders:
    def test_dudt_matches_canonical_partial(self):
        from ddgclib.operators.stress import dudt_i
        HC = object()
        m = SolverMethods(dim=2)
        fn = m.dudt_fn(HC, mu=0.1)
        ref = partial(dudt_i, dim=2, mu=0.1, HC=HC, pressure_model=None)
        assert fn.func is ref.func and fn.keywords == ref.keywords

    def test_dudt_requires_mu(self):
        with pytest.raises(ValueError, match='mu'):
            SolverMethods(dim=2).dudt_fn(object())

    def test_single_phase_eos_warnings(self):
        """laneK: single-phase reconnection + EOS and euler + EOS are
        measured unstable; the builder warns (never for dual_only or for
        pressure_model=None)."""
        from ddgclib.eos import TaitMurnaghan
        eos = TaitMurnaghan(rho0=1000.0, P0=0.0, K=1e5, n=1)
        with pytest.warns(UserWarning, match='UNSTABLE'):
            SolverMethods(dim=2).dudt_fn(object(), mu=1.0, pressure_model=eos)
        with pytest.warns(UserWarning, match="integrator='euler'"):
            SolverMethods(dim=2, integrator='euler', connectivity='dual_only'
                          ).dudt_fn(object(), mu=1.0, pressure_model=eos)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            SolverMethods(dim=2, connectivity='dual_only').dudt_fn(
                object(), mu=1.0, pressure_model=eos)
            SolverMethods(dim=2).dudt_fn(object(), mu=1.0)
            # laneR: the conservative remap is the supported way to run
            # single-phase reconnection + EOS
            SolverMethods(dim=2, remap='conservative', redistribute_mass=True
                          ).dudt_fn(object(), mu=1.0, pressure_model=eos)

    def test_single_phase_remap_builder(self):
        """laneR: single-phase remap = the library _retopologize with
        retopo_remap bound; the integrator forwards pressure_model /
        redistribute_mass to it by name."""
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize,
        )
        from ddgclib.eos import TaitMurnaghan
        eos = TaitMurnaghan(rho0=1000.0, P0=0.0, K=1e5, n=1)
        m = SolverMethods(dim=2, remap='conservative', redistribute_mass=True)
        fn = m.retopologize_fn()
        assert fn.func is _retopologize
        assert fn.keywords == {'retopo_remap': 'conservative'}
        kw = m.integrator_kwargs(pressure_model=eos)
        assert kw['pressure_model'] is eos and kw['redistribute_mass'] is True
        with pytest.raises(ValueError, match='pressure_model'):
            m.integrator_kwargs()
        for bad in (dict(redistribute_mass=False),
                    dict(redistribute_mass=True, connectivity='dual_only'),
                    dict(redistribute_mass=True, connectivity='frozen')):
            with pytest.raises(ValueError):
                SolverMethods(dim=2, remap='conservative', **bad)
        with pytest.raises(ValueError, match='not available in 1D'):
            SolverMethods(dim=1, remap='conservative', redistribute_mass=True)

    def test_retopologize_fn_values(self):
        assert SolverMethods(dim=2).retopologize_fn() is None
        assert SolverMethods(dim=2, connectivity='frozen').retopologize_fn() is False
        custom = lambda HC, bV, dim: None  # noqa: E731
        assert SolverMethods(dim=2, connectivity='custom').retopologize_fn(
            custom=custom) is custom
        with pytest.raises(ValueError):
            SolverMethods(dim=2, connectivity='custom').retopologize_fn()
        with pytest.raises(ValueError):
            SolverMethods(dim=2).retopologize_fn(custom=custom)

    def test_integrator_kwargs_single_phase(self):
        m = SolverMethods(dim=2, connectivity='dual_only', merge_cdist=1e-3,
                          displacement_eps=1e-6)
        kw = m.integrator_kwargs()
        assert kw['retopologize_fn'] is None
        assert kw['skip_triangulation'] is True
        assert kw['remesh_mode'] == 'delaunay' and kw['remesh_kwargs'] is None
        assert kw['merge_cdist'] == 1e-3 and kw['displacement_eps'] == 1e-6
        assert kw['redistribute_mass'] is False and kw['pressure_model'] is None
        assert 'periodic_axes' not in kw

    def test_single_phase_redistribution_needs_eos(self):
        m = SolverMethods(dim=2, redistribute_mass=True)
        with pytest.raises(ValueError, match='pressure_model'):
            m.integrator_kwargs()
        eos = object()
        assert m.integrator_kwargs(pressure_model=eos)['pressure_model'] is eos

    def test_periodic_kwargs(self):
        m = SolverMethods(dim=2, connectivity='periodic', periodic_axes=(0,))
        with pytest.raises(ValueError, match='domain_bounds'):
            m.integrator_kwargs()
        kw = m.integrator_kwargs(domain_bounds=[[0.0, 1.0], [0.0, 1.0]])
        assert kw['retopologize_fn'] is None
        assert kw['periodic_axes'] == [0]
        assert kw['domain_bounds'] == [(0.0, 1.0), (0.0, 1.0)]

    def test_dual_only_bare_partial(self):
        from ddgclib.methods._retopo import bare_dual_refresh
        fn = SolverMethods(dim=2, connectivity='dual_only_bare').retopologize_fn()
        assert fn.func is bare_dual_refresh and fn.keywords == {'mps': None}

    def test_integrate_runs_every_integrator(self):
        """Every registered integrator is callable through integrate()."""
        from hyperct import Complex
        for name in AXES['integrator'].keys():
            HC = Complex(1, domain=[(0.0, 1.0)])
            HC.triangulate()
            HC.refine_all()
            bV = set()
            for v in HC.V:
                v.u = np.zeros(1)
                v.p = 0.0
                v.m = 1.0
                if abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14:
                    bV.add(v)
            m = SolverMethods(dim=1, integrator=name)
            zero = lambda v: np.zeros(1)  # noqa: E731
            if name == 'euler_adaptive':
                t = m.integrate(HC, bV, zero, dt=1e-3, t_end=3e-3)
                assert t == pytest.approx(3e-3)
            else:
                t = m.integrate(HC, bV, zero, dt=1e-3, n_steps=3)
                assert t == pytest.approx(3e-3)
        with pytest.raises(ValueError, match='n_steps'):
            SolverMethods(dim=1).integrate(HC, bV, zero, dt=1e-3)


# ---------------------------------------------------------------------------
# builders: multiphase (bit-identity with the hand-written case code)
# ---------------------------------------------------------------------------

def _kw_without_mps(fn):
    return {k: v for k, v in fn.keywords.items() if k != 'mps'}


@pytest.fixture(scope='module')
def droplet_2d():
    from cases_dynamic.oscillating_droplet.src._setup import (
        setup_oscillating_droplet,
    )

    def build(**over):
        kw = dict(dim=2, R0=0.01, epsilon=0.05, l=2, rho_d=800.0,
                  rho_o=1000.0, mu_d=0.5, mu_o=0.1, gamma=0.05,
                  L_domain=0.05, refinement_outer=1, refinement_droplet=2)
        kw.update(over)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return setup_oscillating_droplet(**kw)
    return build


class TestMultiphaseBuilders:
    def test_multiphase_requires_mps(self):
        m = PRESETS['oscillating_droplet_2D']
        with pytest.raises(ValueError, match='mps'):
            m.retopologize_fn()
        with pytest.raises(ValueError, match='mps'):
            m.dudt_fn(object())

    @pytest.mark.parametrize('preset, runner_extra', [
        ('oscillating_droplet_2D', {'retopo_remap': 'conservative'}),
        ('oscillating_droplet_2D_dual_only', {'skip_triangulation': True}),
        ('oscillating_droplet_2D_bare_delaunay', {}),
        ('oscillating_droplet_2D_projection2',
         {'retopo_remap': 'conservative', 'projection_every': 2}),
        ('static_droplet_floor_2D', {}),
    ])
    def test_retopo_partial_equals_runner_dispatch(self, droplet_2d, preset,
                                                   runner_extra):
        """The runner pattern: setup binds (mps, split_method,
        redistribute_mass); the policy dispatch adds one more partial
        (oscillating_droplet_2D.py:96-99).  The wrapper must produce the
        same function and the same flattened keyword set."""
        m = PRESETS[preset]
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = droplet_2d(
            split_method=m.split_method,
            redistribute_mass=m.redistribute_mass)
        case_fn = partial(retopo_fn, **runner_extra)
        wrap_fn = m.retopologize_fn(mps=mps)
        assert wrap_fn.func is case_fn.func
        assert wrap_fn.keywords == case_fn.keywords
        assert wrap_fn.keywords['mps'] is mps
        assert wrap_fn.args == case_fn.args == ()

    def test_3d_dual_only_partial(self, ):
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize_multiphase,
        )
        m = PRESETS['oscillating_droplet_3D']
        mps = object()
        fn = m.retopologize_fn(mps=mps)
        assert fn.func is _retopologize_multiphase
        assert fn.keywords == dict(mps=mps, split_method='neighbour_count',
                                   redistribute_mass=True,
                                   skip_triangulation=True)

    def test_dam_break_bindings(self):
        """dam_break_2D.py: partial(_retopologize_multiphase, mps,
        redistribute_mass=True) + retopo_remap='conservative'; split_method
        left at its default.  The wrapper binds split_method explicitly
        with the same value, which is the only difference."""
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize_multiphase,
        )
        mps = object()
        case_fn = partial(partial(_retopologize_multiphase, mps=mps,
                                  redistribute_mass=True),
                          retopo_remap='conservative')
        wrap_fn = PRESETS['dam_break_2D'].retopologize_fn(mps=mps)
        assert wrap_fn.func is case_fn.func
        assert {**case_fn.keywords, 'split_method': 'neighbour_count'} == wrap_fn.keywords
        # 3D: the runner passed skip_triangulation=True at INTEGRATOR level
        # (forwarded by name); the wrapper binds it in the partial.
        wrap3 = PRESETS['dam_break_3D'].retopologize_fn(mps=mps)
        assert wrap3.keywords['skip_triangulation'] is True
        assert PRESETS['dam_break_3D'].integrator_kwargs(mps=mps)[
            'skip_triangulation'] is True

    def test_dudt_partial_and_body_force(self, droplet_2d):
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = droplet_2d()
        m = PRESETS['oscillating_droplet_2D']
        meos = dudt_fn.keywords['pressure_model']
        fn = m.dudt_fn(HC, mps=mps, pressure_model=meos)
        assert fn.func is dudt_fn.func and fn.keywords == dudt_fn.keywords
        # non-default curvature_path is bound explicitly
        fn2 = m.replace(curvature_path='csf_dual').dudt_fn(
            HC, mps=mps, pressure_model=meos)
        assert fn2.keywords['curvature_path'] == 'csf_dual'
        # body force wraps exactly like dam_break/src/_setup.py:209-213
        g = m.dudt_fn(HC, mps=mps, pressure_model=meos, body_force=[0.0, -9.81])
        v = next(v for v in HC.V if not v.boundary)
        np.testing.assert_array_equal(g(v), fn(v) + np.array([0.0, -9.81]))
        with pytest.raises(ValueError):
            m.dudt_fn(HC, mps=mps, body_force=[0.0, 0.0, -9.81, 1.0])

    def test_integrate_is_bit_identical_to_runner_path(self, droplet_2d):
        """8 symplectic-Euler steps of the delaunay_remap policy through
        (a) the hand-written runner call and (b) SolverMethods.integrate
        give the same state to the bit (the TestProjectionCadence2D
        state-tuple comparison)."""
        from ddgclib.dynamic_integrators import symplectic_euler

        def state(HC):
            return sorted(
                (tuple(v.x_a[:2]), tuple(v.u[:2]), float(v.m),
                 tuple(float(p) for p in v.p_phase))
                for v in HC.V)

        m = PRESETS['oscillating_droplet_2D']
        states = []
        for use_wrapper in (False, True):
            HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = droplet_2d(
                split_method=m.split_method,
                redistribute_mass=m.redistribute_mass)
            c_s = float(np.sqrt(params['K_d'] / 800.0))
            dx_min = min(
                float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
                for v in HC.V for nb in v.nn
                if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15)
            dt = min(0.25 * dx_min / c_s,
                     0.5 * float(np.sqrt(800.0 * dx_min ** 3 / 0.05)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if use_wrapper:
                    m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=8,
                                bc_set=bc_set, mps=mps)
                else:
                    symplectic_euler(
                        HC, bV, dudt_fn, dt=dt, n_steps=8, dim=2,
                        bc_set=bc_set,
                        retopologize_fn=partial(retopo_fn,
                                                retopo_remap='conservative'),
                        remesh_mode=params['remesh_mode'],
                        remesh_kwargs=params['remesh_kwargs'],
                    )
            states.append(state(HC))
        assert states[0] == states[1]

    def test_dual_only_bare_is_bit_identical_to_static_droplet_closure(
            self, droplet_2d):
        """static_droplet_2D.py used to carry this closure; the library
        version selected by connectivity='dual_only_bare' must give the
        same state to the bit."""
        from hyperct.ddg import compute_vd
        from ddgclib.operators.stress import cache_dual_volumes
        from ddgclib.dynamic_integrators import symplectic_euler

        def _dual_only_retopo(HC, bV, dim, _mps=None, **_kw):
            dV = HC.boundary()
            for v in HC.V:
                v.boundary = v in dV
            compute_vd(HC, method='barycentric')
            cache_dual_volumes(HC, dim)
            if _mps is not None:
                _mps.split_dual_volumes(HC, dim)
            bV.clear()
            bV.update(dV)

        def state(HC):
            return sorted(
                (tuple(v.x_a[:2]), tuple(v.u[:2]), float(v.m),
                 tuple(float(p) for p in v.p_phase), float(v.dual_vol))
                for v in HC.V)

        m = PRESETS['static_droplet_2D']
        assert m.connectivity == 'dual_only_bare'
        states = []
        for use_wrapper in (False, True):
            HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = droplet_2d(
                epsilon=0.0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if use_wrapper:
                    m.integrate(HC, bV, dudt_fn, dt=1e-5, n_steps=6,
                                bc_set=bc_set, mps=mps)
                else:
                    symplectic_euler(
                        HC, bV, dudt_fn, dt=1e-5, n_steps=6, dim=2,
                        bc_set=bc_set,
                        retopologize_fn=partial(_dual_only_retopo, _mps=mps))
            states.append(state(HC))
        assert states[0] == states[1]

    _SHEARING_PROBE = r'''
import hashlib, sys, warnings
sys.path.insert(0, {root!r})
warnings.simplefilter('ignore')
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.methods import PRESETS
from cases_dynamic.shearing_plate_droplet.src._setup import setup_shearing_plate_droplet
from cases_dynamic.shearing_plate_droplet.src import _params as sp
m = PRESETS['shearing_plate_droplet_2D']
HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params = setup_shearing_plate_droplet(
    dim=2, R0=sp.R0, L_x=sp.L_x, L_y=sp.L_y, U_wall=sp.U_wall, rho_d=sp.rho_d,
    rho_o=sp.rho_o, mu_d=sp.mu_d, mu_o=sp.mu_o, gamma=sp.gamma, K_d=sp.K_d, K_o=sp.K_o,
    refinement_outer=3, refinement_droplet=3, redistribute_mass=m.redistribute_mass)
if {use_wrapper}:
    fn = m.retopologize_fn(mps=mps, domain_bounds=params['domain_bounds'])
    assert fn.keywords['periodic_axes'] == list(params['periodic_axes'])
    m.integrate(HC, bV, dudt_fn, dt=1e-5, n_steps=3, bc_set=bc_set, mps=mps,
                domain_bounds=params['domain_bounds'])
else:
    symplectic_euler(HC, bV, dudt_fn, dt=1e-5, n_steps=3, dim=2, bc_set=bc_set,
                     retopologize_fn=retopo_fn, remesh_mode=params['remesh_mode'],
                     remesh_kwargs=params['remesh_kwargs'])
state = sorted((tuple(v.x_a[:2]), tuple(v.u[:2]), float(v.m),
                tuple(float(p) for p in v.p_phase)) for v in HC.V)
print('STATE', hashlib.sha256(repr(state).encode()).hexdigest(), len(state))
'''

    def test_periodic_multiphase_is_bit_identical_to_shearing_wrapper(self):
        """shearing_plate_droplet/src/_setup.py built a periodic multiphase
        closure; connectivity='periodic' + phases='multi' selects the
        library copy.  Three steps of the shipped 2D configuration must
        agree to the bit.  Each variant runs in its own interpreter: the
        case setup is not re-entrant (a second call in one process crashes
        on the outer-vertex rescale key collision, audit 2026-09-25)."""
        from pathlib import Path
        root = str(Path(__file__).resolve().parents[2])
        digests = []
        for use_wrapper in (False, True):
            code = self._SHEARING_PROBE.format(root=root, use_wrapper=use_wrapper)
            out = subprocess.run([sys.executable, '-c', code], cwd=root,
                                 capture_output=True, text=True, timeout=600)
            assert out.returncode == 0, out.stderr[-2000:]
            line = [ln for ln in out.stdout.splitlines() if ln.startswith('STATE')]
            assert line, out.stdout[-500:]
            digests.append(line[-1])
        assert digests[0] == digests[1]

    def test_electrolysis_presets_match_setup_partials(self):
        from cases_dynamic.electrolysis_bubble.src._setup import (
            setup_electrolysis_bubble,
        )
        m = PRESETS['electrolysis_bubble_2D']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
                setup_electrolysis_bubble(
                    dim=2, refinement_outer=1, refinement_droplet=2,
                    redistribute_mass=m.redistribute_mass)
        wrap_fn = m.retopologize_fn(mps=mps)
        assert wrap_fn.func is retopo_fn.func
        assert wrap_fn.keywords == retopo_fn.keywords
        # fritz: redistribute_mass unbound in the case -> integrator default
        # False, which the preset states explicitly
        assert PRESETS['electrolysis_bubble_fritz_2D'].redistribute_mass is False


# ---------------------------------------------------------------------------
# effective (implicit) methods + recording
# ---------------------------------------------------------------------------

class TestEffectiveMethods:
    def _run_one_retopo(self, dim):
        from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
        if dim == 2:
            from ddgclib.geometry.domains import rectangle
            res = rectangle(L=1.0, h=1.0, refinement=2)
        else:
            from ddgclib.geometry.domains import box
            res = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=1)
        HC, bV = res.HC, res.bV
        for v in HC.V:
            v.u = np.zeros(dim)
            v.p = 0.0
            v.m = 1.0
        _retopologize(HC, bV, dim)
        return HC, bV

    def test_2d_has_no_edge_area_cache_and_half_cell_boundary(self):
        """Audit 2026-09-25 §0.1: batch_e_star raises for dim != 3, so the
        2D retopology keeps boundary dual volumes and builds A_ij from
        v.vd.  Lock the implicit choice."""
        HC, bV = self._run_one_retopo(2)
        eff = effective_methods(HC, 2)
        assert eff['edge_area_source'] == 'shared_vd_2d'
        assert eff['edge_area_cache_present'] is False
        assert eff['boundary_dual_vol'] == 'half_cell'
        assert eff['dual_volume'] == 'simplex_exact'
        assert eff['dual_path'] == 'simplex_aware'
        assert max(v.dual_vol for v in bV) > 0.0
        assert sum(v.dual_vol for v in HC.V) == pytest.approx(1.0)

    def test_3d_uses_edge_area_cache_and_zeroed_boundary(self):
        """Audit §0.2: after a 3D retopology the force reads the
        batch_e_star cache and boundary dual volumes are zeroed."""
        HC, bV = self._run_one_retopo(3)
        eff = effective_methods(HC, 3)
        assert eff['edge_area_source'] == 'batch_e_star_cache'
        assert eff['edge_area_cache_present'] is True
        assert eff['boundary_dual_vol'] == 'zeroed'
        assert eff['dual_volume'] == 'simplex_exact'
        assert all(v.dual_vol == 0.0 for v in bV)

    def test_dual_only_reports_carried_bv(self):
        HC, bV = self._run_one_retopo(2)
        m = SolverMethods(dim=2, connectivity='dual_only')
        assert effective_methods(HC, 2, m)['boundary_rule'] == 'carried_bV'

    def test_record_methods_writes_json(self, tmp_path):
        HC, bV = self._run_one_retopo(2)
        m = SolverMethods(dim=2, label='test')
        path = tmp_path / 'results' / 'methods.json'
        doc = record_methods(path, m, HC, extra={'dt': 1e-3, 'n': np.int64(3),
                                                 'arr': np.zeros(2)})
        on_disk = json.loads(path.read_text())
        assert on_disk == doc
        assert on_disk['schema'] == 'ddgclib.methods/1'
        assert SolverMethods.from_dict(on_disk['config']) == m
        assert on_disk['effective']['edge_area_source'] == 'shared_vd_2d'
        assert on_disk['status']['connectivity'] == 'validated'
        assert on_disk['extra'] == {'dt': 1e-3, 'n': 3, 'arr': [0.0, 0.0]}
        assert 'ddgclib' in on_disk['git'] and 'hyperct' in on_disk['git']
