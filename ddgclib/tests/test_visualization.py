"""Tests for ddgclib.visualization package."""

import numpy as np
import pytest

from hyperct import Complex

# Use Agg backend to avoid display issues in CI/headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_1d():
    HC = Complex(1, domain=[(0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    for v in HC.V:
        v.u = np.array([v.x_a[0]])
        v.P = v.x_a[0] * 100
    return HC


@pytest.fixture
def mesh_2d():
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14):
            bV.add(v)
        v.u = np.array([v.x_a[0], -v.x_a[1]])
        v.P = v.x_a[0] + v.x_a[1]
    return HC, bV


@pytest.fixture
def mesh_3d():
    HC = Complex(3, domain=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    for v in HC.V:
        v.u = np.array([0.0, 0.0, v.x_a[2]])
        v.P = v.x_a[2] * 100
    return HC


@pytest.fixture(autouse=True)
def close_figs():
    """Close all figures after each test."""
    yield
    plt.close('all')


# ---------------------------------------------------------------------------
# 1D plot tests
# ---------------------------------------------------------------------------

class TestPlot1D:
    def test_scalar_field_1d(self, mesh_1d):
        from ddgclib.visualization import plot_scalar_field_1d
        fig, ax = plot_scalar_field_1d(mesh_1d, field='P')
        assert fig is not None
        assert ax is not None

    def test_scalar_field_1d_with_analytical(self, mesh_1d):
        from ddgclib.visualization import plot_scalar_field_1d
        fig, ax = plot_scalar_field_1d(
            mesh_1d, field='P',
            analytical_fn=lambda x: x * 100,
            title='Test Pressure',
        )
        assert len(ax.lines) >= 2  # data + analytical

    def test_velocity_profile_1d(self, mesh_1d):
        from ddgclib.visualization import plot_velocity_profile_1d
        fig, ax = plot_velocity_profile_1d(mesh_1d, component=0)
        assert fig is not None

    def test_1d_with_existing_axes(self, mesh_1d):
        from ddgclib.visualization import plot_scalar_field_1d
        fig0, ax0 = plt.subplots()
        fig, ax = plot_scalar_field_1d(mesh_1d, field='P', ax=ax0)
        assert ax is ax0


# ---------------------------------------------------------------------------
# 2D plot tests
# ---------------------------------------------------------------------------

class TestPlot2D:
    def test_scalar_field_2d(self, mesh_2d):
        from ddgclib.visualization import plot_scalar_field_2d
        HC, _ = mesh_2d
        fig, ax = plot_scalar_field_2d(HC, field='P', title='Pressure 2D')
        assert fig is not None

    def test_vector_field_2d(self, mesh_2d):
        from ddgclib.visualization import plot_vector_field_2d
        HC, bV = mesh_2d
        fig, ax = plot_vector_field_2d(HC, bV=bV, title='Velocity 2D')
        assert fig is not None

    def test_mesh_2d(self, mesh_2d):
        from ddgclib.visualization import plot_mesh_2d
        HC, bV = mesh_2d
        fig, ax = plot_mesh_2d(HC, bV=bV, title='Mesh 2D')
        assert fig is not None

    def test_mesh_2d_no_boundary(self, mesh_2d):
        from ddgclib.visualization import plot_mesh_2d
        HC, _ = mesh_2d
        fig, ax = plot_mesh_2d(HC, title='Mesh no boundary')
        assert fig is not None


# ---------------------------------------------------------------------------
# 3D plot tests
# ---------------------------------------------------------------------------

class TestPlot3D:
    def test_scalar_field_3d(self, mesh_3d):
        from ddgclib.visualization import plot_scalar_field_3d
        fig, ax = plot_scalar_field_3d(mesh_3d, field='P', title='Pressure 3D')
        assert fig is not None

    def test_extract_slice_profile(self, mesh_3d):
        from ddgclib.visualization import extract_slice_profile
        verts = extract_slice_profile(mesh_3d, axis=2, position=0.5, tol=0.6)
        assert len(verts) > 0
        for v in verts:
            assert abs(v.x_a[2] - 0.5) < 0.6

    def test_extract_slice_empty(self, mesh_3d):
        from ddgclib.visualization import extract_slice_profile
        verts = extract_slice_profile(mesh_3d, axis=2, position=99.0, tol=0.01)
        assert len(verts) == 0

    def test_extract_slice_sorted(self, mesh_3d):
        from ddgclib.visualization import extract_slice_profile
        verts = extract_slice_profile(mesh_3d, axis=2, position=0.0, tol=0.1,
                                      sort_by=0)
        if len(verts) > 1:
            xs = [v.x_a[0] for v in verts]
            assert xs == sorted(xs)


# ---------------------------------------------------------------------------
# Polyscope tests (skipped if not installed)
# ---------------------------------------------------------------------------

class TestPolyscope:
    def test_import_guard(self):
        """Polyscope functions should raise ImportError if not installed."""
        try:
            import polyscope
            pytest.skip("polyscope is installed; skip import guard test")
        except ImportError:
            from ddgclib.visualization.polyscope_3d import register_point_cloud
            with pytest.raises(ImportError, match="polyscope is required"):
                register_point_cloud(None)


# ---------------------------------------------------------------------------
# Animation tests
# ---------------------------------------------------------------------------

class TestAnimation:
    def test_animate_scalar_1d(self, mesh_1d):
        from ddgclib.data import StateHistory
        from ddgclib.visualization.animation import animate_scalar_1d

        history = StateHistory(fields=['P'])
        # Create a few manual snapshots
        for t_val in [0.0, 0.1, 0.2]:
            for v in mesh_1d.V:
                v.P = v.x_a[0] * 100 + t_val * 10
            history.append(t_val, mesh_1d)

        anim = animate_scalar_1d(history, field='P')
        assert anim is not None

    def test_animate_scalar_2d(self, mesh_2d):
        from ddgclib.data import StateHistory
        from ddgclib.visualization.animation import animate_scalar_2d

        HC, _ = mesh_2d
        history = StateHistory(fields=['P'])
        for t_val in [0.0, 0.1]:
            for v in HC.V:
                v.P = v.x_a[0] + t_val
            history.append(t_val, HC)

        anim = animate_scalar_2d(history, field='P')
        assert anim is not None


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_all_exports(self):
        from ddgclib.visualization import (
            plot_scalar_field_1d,
            plot_velocity_profile_1d,
            plot_scalar_field_2d,
            plot_vector_field_2d,
            plot_mesh_2d,
            plot_scalar_field_3d,
            extract_slice_profile,
        )
        assert callable(plot_scalar_field_1d)
        assert callable(extract_slice_profile)
