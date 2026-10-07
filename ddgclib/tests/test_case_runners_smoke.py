"""Smoke tests for the runners laneX revived (2026-10-06): every one starts
headlessly in its shortest mode from a scratch copy of its case directory,
exits 0 and writes ``methods.json`` with the preset it is bound to, so an
import path or a hand-bound force cannot break silently again.

Runners covered: ``cube2droplet`` (2D, 3D, adaptive, mass redistribution,
BC comparison), ``bc_demo/bc_demo.py``, ``liquid_bridge_cfd_dem`` and
``liquid_bridge_equilibrium`` Cases 1 and 5.  The in-process checks at the
end pin the film force builder and the integrated axial force of the exact
catenoid (a minimal surface, reference zero) against the neck force scale
``2 pi gamma a``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ddgclib.methods import PRESETS, SolverMethods  # noqa: E402

_COPY_IGNORE = shutil.ignore_patterns(
    '__pycache__', 'out', 'fig', 'results*', 'snapshots', '*.png', '*.gif',
    '*.mp4', '*.npz', '*.pptx', '*.ipynb', '*.msh')


def _scratch_copy(tmp_path, case: str) -> str:
    """Copy ``cases_dynamic/<case>`` (code only) under *tmp_path* so a run
    writes its ``fig/`` / ``results/`` / ``out/`` there, never in the tree."""
    src = os.path.join(REPO, 'cases_dynamic', case)
    dst = os.path.join(str(tmp_path), 'cases_dynamic', case)
    shutil.copytree(src, dst, ignore=_COPY_IGNORE)
    return dst


def _run(case_dir: str, script: str, *args: str, timeout: float = 600.0):
    env = dict(os.environ, PYTHONPATH=REPO, MPLBACKEND='Agg',
               DDGCLIB_CASE5_WORKERS='1')
    proc = subprocess.run(
        [sys.executable, os.path.join(case_dir, script), *args],
        cwd=case_dir, env=env, capture_output=True, text=True,
        timeout=timeout)
    assert proc.returncode == 0, (
        f"{script} {' '.join(args)} exited {proc.returncode}\n"
        f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr (tail) ---\n{proc.stderr[-3000:]}")
    return proc


def _methods_json(path: str, preset: str | SolverMethods) -> dict:
    with open(path) as fh:
        doc = json.load(fh)
    want = PRESETS[preset] if isinstance(preset, str) else preset
    assert doc['schema'] == 'ddgclib.methods/1'
    assert SolverMethods.from_dict(doc['config']) == want
    assert doc['effective']['n_vertices'] > 0
    return doc


# ----------------------------------------------------------------------
# cube2droplet
# ----------------------------------------------------------------------
class TestCubeToDroplet:
    def test_2d(self, tmp_path):
        # refinement 4 = the shipped resolution (545 vertices, 41 bulk
        # droplet vertices): at refinement 3 the droplet is 5 bulk vertices
        # and the primal-subcomplex relabelling erodes it within two
        # Delaunay rebuilds (laneX log 4.1)
        d = _scratch_copy(tmp_path, 'cube2droplet')
        _run(d, 'cube_to_droplet_2D.py', '--n-steps', '20', '--n-refine', '4',
             '--no-anim')
        doc = _methods_json(os.path.join(d, 'results', 'methods.json'),
                            'cube_to_droplet_2D')
        extra = doc['extra']
        assert extra['n_steps'] == 20 and extra['arm'] == 'base'
        assert 0.0 < extra['circularity_final'] <= 1.0
        assert extra['dp_laplace'] == pytest.approx(0.01 / (0.01 * 2 / np.sqrt(np.pi)))
        assert np.isfinite(extra['dp_integrated_final'])

    def test_2d_arms(self, tmp_path):
        d = _scratch_copy(tmp_path, 'cube2droplet')
        # the historic setup configuration (no remap), kept reachable
        _run(d, 'cube_to_droplet_2D.py', '--n-steps', '5', '--n-refine', '4',
             '--no-anim', '--arm', 'bare')
        _methods_json(os.path.join(d, 'results', 'methods.json'),
                      PRESETS['cube_to_droplet_2D'].replace(remap=None))
        assert PRESETS['cube_to_droplet_2D'].remap == 'conservative'

    def test_3d(self, tmp_path):
        d = _scratch_copy(tmp_path, 'cube2droplet')
        _run(d, 'cube_to_droplet_3D.py', '--n-steps', '5', '--n-refine', '2',
             '--no-anim')
        doc = _methods_json(os.path.join(d, 'results', 'methods.json'),
                            'cube_to_droplet_3D')
        assert 0.0 < doc['extra']['sphericity_final'] <= 1.0

    def test_adaptive(self, tmp_path):
        d = _scratch_copy(tmp_path, 'cube2droplet')
        _run(d, 'cube_to_droplet_2D_adaptive.py', '--n-steps', '3',
             '--n-refine', '3')
        with open(os.path.join(d, 'results', 'methods_adaptive.json')) as fh:
            doc = json.load(fh)
        m = SolverMethods.from_dict(doc['config'])
        assert m.connectivity == 'adaptive' and m.remesh_kwargs
        assert m.replace(connectivity='delaunay', remesh_kwargs=None) == \
            PRESETS['cube_to_droplet_2D']

    @pytest.mark.slow
    def test_mass_redist_and_bc_comparison(self, tmp_path):
        d = _scratch_copy(tmp_path, 'cube2droplet')
        _run(d, 'cube_to_droplet_2D_mass_redist.py', '--n-steps', '5',
             '--n-refine', '3', '--no-anim')
        r = os.path.join(d, 'results')
        _methods_json(os.path.join(r, 'methods_redist.json'), 'cube_to_droplet_2D')
        _methods_json(os.path.join(r, 'methods_redist_bare.json'),
                      PRESETS['cube_to_droplet_2D'].replace(remap=None))
        _methods_json(os.path.join(r, 'methods_no_redist.json'),
                      PRESETS['cube_to_droplet_2D'].replace(
                          remap=None, redistribute_mass=False))
        _methods_json(os.path.join(r, 'methods_no_retopo.json'),
                      'cube_to_droplet_2D_dual_only')
        _run(d, 'cube_to_droplet_2D_bc_comparison.py', '--n-steps', '5',
             '--n-refine', '3', '--no-anim')
        for mode in ('atmos', 'reservoir', 'absorb', 'expanding'):
            doc = _methods_json(os.path.join(r, f'methods_bc_{mode}.json'),
                                'cube_to_droplet_2D_dual_only')
            assert doc['extra']['bc_mode'] == mode


# ----------------------------------------------------------------------
# bc_demo, liquid bridges
# ----------------------------------------------------------------------
def test_bc_demo(tmp_path):
    d = _scratch_copy(tmp_path, 'bc_demo')
    _run(d, 'bc_demo.py', '--n-steps', '10')
    doc = _methods_json(os.path.join(d, 'results', 'methods.json'), 'bc_demo_2D')
    assert doc['extra']['n_vertices_final'] > 0


def test_liquid_bridge_cfd_dem(tmp_path):
    d = _scratch_copy(tmp_path, 'liquid_bridge_cfd_dem')
    _run(d, 'liquid_bridge_cfd_dem_case.py', '--n-steps', '3')
    doc = _methods_json(os.path.join(d, 'results', 'methods.json'),
                        'liquid_bridge_cfd_dem_3D')
    assert doc['extra']['n_steps'] == 3
    with open(os.path.join(d, 'results', 'history.json')) as fh:
        hist = json.load(fh)
    assert hist[-1]['step'] == 3
    assert np.isfinite(hist[-1]['capillary_force_mag'])


@pytest.mark.slow
def test_liquid_bridge_equilibrium_case_1(tmp_path):
    d = _scratch_copy(tmp_path, 'liquid_bridge_equilibrium')
    _run(d, 'Case_1_equilibrium_particle_particle_bridge_benchmark.py',
         '--refinements', '2')
    doc = _methods_json(os.path.join(d, 'out', 'Case_1', 'methods.json'),
                        'liquid_bridge_film_3D')
    assert doc['extra'] == {'refinement': 2, 'dt': 2e-6, 'n_steps': 100,
                            'damping': 20.0}


@pytest.mark.slow
def test_liquid_bridge_equilibrium_case_5(tmp_path):
    d = _scratch_copy(tmp_path, 'liquid_bridge_equilibrium')
    _run(d, 'Case_5_volumetric_stress_equilibrium_particle_particle_bridge_benchmark.py',
         '--refinements', '0')
    _methods_json(os.path.join(d, 'out', 'Case_5', 'methods.json'),
                  'liquid_bridge_volume_3D')


# ----------------------------------------------------------------------
# the film force against analytical references (in process)
# ----------------------------------------------------------------------
class TestFilmForce:
    def test_builder_is_the_surface_tension_partial(self):
        from ddgclib.operators.surface_tension import (
            surface_tension_acceleration,
        )
        m = PRESETS['liquid_bridge_film_3D']
        fn = m.dudt_fn(HC='mesh', gamma=0.0728, damping=20.0)
        assert fn.func is surface_tension_acceleration
        assert fn.keywords == dict(gamma=0.0728, damping=20.0, dim=3, HC='mesh')
        assert m.retopologize_fn() is False
        with pytest.raises(ValueError, match='gamma'):
            m.dudt_fn(HC=None)
        with pytest.raises(ValueError, match='film'):
            m.dudt_fn(HC=None, gamma=1.0, mu=0.0)
        with pytest.raises(ValueError, match='film'):
            PRESETS['hydrostatic_2D'].dudt_fn(HC=None, mu=1.0, gamma=1.0)

    @pytest.mark.parametrize('refinement', [2, 3])
    def test_catenoid_axial_force_is_zero(self, refinement):
        """Minimal surface: the integrated axial film force over the
        interior vertices is zero analytically; on the z-symmetric
        catenoid mesh of Case 1 it cancels to round-off (the benchmark's
        own metric).  The LOCAL Heron curvature on that mesh does not
        vanish (laneX: mean |2H| a = 0.556 at refinements 3 to 5, see the
        lane log), so the local force is not asserted here."""
        from cases_dynamic.liquid_bridge_equilibrium import (
            Case_1_equilibrium_particle_particle_bridge_benchmark as c1,
        )
        HC, bV = c1._build_live_endres_catenoid(refinement)
        c1._prepare_surface_benchmark_state(HC, bV)
        fn = c1.METHODS.dudt_fn(HC, gamma=c1.GAMMA)
        F_z = sum(float((fn(v) * v.m)[2]) for v in HC.V if v not in bV)
        scale = 2.0 * np.pi * c1.GAMMA * c1.ABC[0]
        assert abs(F_z) / scale < 1e-12, abs(F_z) / scale
        assert F_z == pytest.approx(c1._stress_capillary_force_error(HC, bV),
                                    abs=1e-15)
        # local content: the film force on every interior vertex is the
        # discrete area gradient -gamma dA/dx_i (central differences of
        # the one-ring triangle areas, eps 1e-6), measured 3.5e-10 /
        # 1.0e-9 of the largest force at refinements 2 / 3 (laneX fix 1)
        verts, tris = c1._extract_triangles(HC)
        coords = np.array([v.x_a for v in verts], dtype=float)
        index = {id(v): i for i, v in enumerate(verts)}
        incident = {i: [] for i in range(len(verts))}
        for tri in tris:
            tri = tuple(int(i) for i in tri)
            for i in tri:
                incident[i].append(tri)
        eps = 1e-6
        d_max, g_max = 0.0, 0.0
        for v in verts:
            if v in bV:
                continue
            i = index[id(v)]
            grad = np.zeros(3)
            for k in range(3):
                a_plus = a_minus = 0.0
                for tri in incident[i]:
                    c_plus = coords[list(tri)].copy()
                    c_minus = c_plus.copy()
                    local = tri.index(i)
                    c_plus[local, k] += eps
                    c_minus[local, k] -= eps
                    a_plus += c1._triangle_area(*c_plus)
                    a_minus += c1._triangle_area(*c_minus)
                grad[k] = -c1.GAMMA * (a_plus - a_minus) / (2.0 * eps)
            d_max = max(d_max, float(np.linalg.norm(fn(v) * v.m - grad)))
            g_max = max(g_max, float(np.linalg.norm(grad)))
        assert g_max > 0.0
        assert d_max / g_max < 1e-7, (d_max, g_max)
