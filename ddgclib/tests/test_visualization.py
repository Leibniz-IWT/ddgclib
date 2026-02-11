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

# ---------------------------------------------------------------------------
# Unified wrapper tests
# ---------------------------------------------------------------------------

class TestPlotPrimal:
    """Tests for plot_primal() dimension-dispatching wrapper."""

    def test_primal_1d_basic(self, mesh_1d):
        from ddgclib.visualization import plot_primal
        fig, ax = plot_primal(mesh_1d, title='Primal 1D', save_path=None)
        assert fig is not None
        assert ax is not None

    def test_primal_1d_scalar(self, mesh_1d):
        from ddgclib.visualization import plot_primal
        fig, ax = plot_primal(mesh_1d, scalar_field='P', title='P on 1D',
                              save_path=None)
        assert fig is not None

    def test_primal_1d_vector(self, mesh_1d):
        from ddgclib.visualization import plot_primal
        fig, ax = plot_primal(mesh_1d, vector_field='u', title='u on 1D',
                              save_path=None)
        assert fig is not None

    def test_primal_1d_both_fields(self, mesh_1d):
        from ddgclib.visualization import plot_primal
        fig, ax = plot_primal(mesh_1d, scalar_field='P', vector_field='u',
                              save_path=None)
        assert fig is not None

    def test_primal_2d_basic(self, mesh_2d):
        from ddgclib.visualization import plot_primal
        HC, bV = mesh_2d
        fig, ax = plot_primal(HC, bV=bV, title='Primal 2D', save_path=None)
        assert fig is not None

    def test_primal_2d_scalar(self, mesh_2d):
        from ddgclib.visualization import plot_primal
        HC, bV = mesh_2d
        fig, ax = plot_primal(HC, bV=bV, scalar_field='P', title='P on 2D',
                              save_path=None)
        assert fig is not None

    def test_primal_2d_vector(self, mesh_2d):
        from ddgclib.visualization import plot_primal
        HC, bV = mesh_2d
        fig, ax = plot_primal(HC, bV=bV, vector_field='u', title='u on 2D',
                              save_path=None)
        assert fig is not None

    def test_primal_2d_both_fields(self, mesh_2d):
        from ddgclib.visualization import plot_primal
        HC, bV = mesh_2d
        fig, ax = plot_primal(HC, scalar_field='P', vector_field='u',
                              save_path=None)
        assert fig is not None

    def test_primal_2d_no_edges(self, mesh_2d):
        from ddgclib.visualization import plot_primal
        HC, _ = mesh_2d
        fig, ax = plot_primal(HC, show_edges=False, title='No edges',
                              save_path=None)
        assert fig is not None

    def test_primal_3d_basic(self, mesh_3d):
        from ddgclib.visualization import plot_primal
        fig, ax = plot_primal(mesh_3d, title='Primal 3D', save_path=None)
        assert fig is not None

    def test_primal_3d_scalar(self, mesh_3d):
        from ddgclib.visualization import plot_primal
        fig, ax = plot_primal(mesh_3d, scalar_field='P', title='P on 3D',
                              save_path=None)
        assert fig is not None

    def test_primal_3d_vector(self, mesh_3d):
        from ddgclib.visualization import plot_primal
        fig, ax = plot_primal(mesh_3d, vector_field='u', title='u on 3D',
                              save_path=None)
        assert fig is not None

    def test_primal_existing_axes(self, mesh_2d):
        from ddgclib.visualization import plot_primal
        HC, _ = mesh_2d
        fig0, ax0 = plt.subplots()
        fig, ax = plot_primal(HC, ax=ax0, save_path=None)
        assert ax is ax0

    def test_primal_custom_cmap(self, mesh_2d):
        from ddgclib.visualization import plot_primal
        HC, _ = mesh_2d
        fig, ax = plot_primal(HC, scalar_field='P', cmap='plasma',
                              save_path=None)
        assert fig is not None

    def test_primal_save_path(self, mesh_2d, tmp_path):
        from ddgclib.visualization import plot_primal
        HC, _ = mesh_2d
        out = tmp_path / 'test_primal.png'
        fig, ax = plot_primal(HC, scalar_field='P', save_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotDual:
    """Tests for plot_dual() dimension-dispatching wrapper."""

    def test_dual_1d(self, mesh_1d):
        from ddgclib.visualization import plot_dual
        fig, ax = plot_dual(mesh_1d, title='Dual 1D', save_path=None)
        assert fig is not None

    def test_dual_1d_scalar(self, mesh_1d):
        from ddgclib.visualization import plot_dual
        fig, ax = plot_dual(mesh_1d, scalar_field='P', title='P dual 1D',
                            save_path=None)
        assert fig is not None

    def test_dual_2d_no_vd(self, mesh_2d):
        """Without compute_vd, plot_dual should warn but not crash."""
        from ddgclib.visualization import plot_dual
        HC, bV = mesh_2d
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig, ax = plot_dual(HC, bV=bV, title='Dual 2D (no vd)',
                                save_path=None)
        assert fig is not None

    def test_dual_3d_requires_vertex(self, mesh_3d):
        from ddgclib.visualization import plot_dual
        with pytest.raises(ValueError, match="vertex.*must be provided"):
            plot_dual(mesh_3d, save_path=None)

    def test_dual_save_path(self, mesh_1d, tmp_path):
        from ddgclib.visualization import plot_dual
        out = tmp_path / 'test_dual.png'
        fig, ax = plot_dual(mesh_1d, save_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotPrimalDispatch:
    """Verify dimension detection dispatches correctly."""

    def test_dim_1_detected(self, mesh_1d):
        assert mesh_1d.dim == 1

    def test_dim_2_detected(self, mesh_2d):
        HC, _ = mesh_2d
        assert HC.dim == 2

    def test_dim_3_detected(self, mesh_3d):
        assert mesh_3d.dim == 3


# ---------------------------------------------------------------------------
# plot_fluid tests
# ---------------------------------------------------------------------------

class TestPlotFluid:
    """Tests for plot_fluid() high-level wrapper."""

    def test_fluid_2d_defaults(self, mesh_2d):
        from ddgclib.visualization import plot_fluid
        HC, bV = mesh_2d
        fig, axes = plot_fluid(HC, bV=bV, save_path=None)
        assert fig is not None
        assert len(axes) == 2  # pressure + velocity panels

    def test_fluid_2d_custom_time(self, mesh_2d):
        from ddgclib.visualization import plot_fluid
        HC, bV = mesh_2d
        fig, axes = plot_fluid(HC, bV=bV, t=0.42, save_path=None)
        assert '0.42' in fig._suptitle.get_text()

    def test_fluid_2d_scalar_only(self, mesh_2d):
        from ddgclib.visualization import plot_fluid
        HC, _ = mesh_2d
        fig, axes = plot_fluid(HC, vector_field=None, save_path=None)
        assert len(axes) == 1

    def test_fluid_2d_vector_only(self, mesh_2d):
        from ddgclib.visualization import plot_fluid
        HC, _ = mesh_2d
        fig, axes = plot_fluid(HC, scalar_field=None, save_path=None)
        assert len(axes) == 1

    def test_fluid_1d(self, mesh_1d):
        from ddgclib.visualization import plot_fluid
        fig, axes = plot_fluid(mesh_1d, save_path=None)
        assert fig is not None

    def test_fluid_3d(self, mesh_3d):
        from ddgclib.visualization import plot_fluid
        fig, axes = plot_fluid(mesh_3d, save_path=None)
        assert fig is not None

    def test_fluid_no_mesh(self, mesh_2d):
        from ddgclib.visualization import plot_fluid
        HC, _ = mesh_2d
        fig, axes = plot_fluid(HC, show_mesh=False, save_path=None)
        assert fig is not None

    def test_fluid_save_path(self, mesh_2d, tmp_path):
        from ddgclib.visualization import plot_fluid
        HC, bV = mesh_2d
        out = tmp_path / 'test_fluid.png'
        fig, _ = plot_fluid(HC, bV=bV, save_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_fluid_default_time_label(self, mesh_2d):
        from ddgclib.visualization import plot_fluid
        HC, _ = mesh_2d
        fig, _ = plot_fluid(HC, save_path=None)
        assert '0.0000 s' in fig._suptitle.get_text()


# ---------------------------------------------------------------------------
# dynamic_plot_fluid tests
# ---------------------------------------------------------------------------

class TestDynamicPlotFluid:
    """Tests for dynamic_plot_fluid() animation wrapper."""

    def _make_history(self, HC, n_frames=3):
        from ddgclib.data import StateHistory
        history = StateHistory(fields=['u', 'P'])
        for i in range(n_frames):
            t = i * 0.1
            for v in HC.V:
                v.P = v.x_a[0] * 100 + t * 50
            history.append(t, HC)
        return history

    def test_dynamic_2d_returns_anim(self, mesh_2d):
        from ddgclib.visualization import dynamic_plot_fluid
        HC, bV = mesh_2d
        history = self._make_history(HC)
        anim = dynamic_plot_fluid(history, HC, bV=bV, save_path=None)
        assert anim is not None

    def test_dynamic_1d_returns_anim(self, mesh_1d):
        from ddgclib.visualization import dynamic_plot_fluid
        history = self._make_history(mesh_1d)
        anim = dynamic_plot_fluid(history, mesh_1d, save_path=None)
        assert anim is not None

    def test_dynamic_empty_history(self, mesh_2d):
        from ddgclib.data import StateHistory
        from ddgclib.visualization import dynamic_plot_fluid
        HC, _ = mesh_2d
        history = StateHistory(fields=['P'])
        anim = dynamic_plot_fluid(history, HC, save_path=None)
        assert anim is None

    def test_dynamic_frame_dir(self, mesh_2d, tmp_path):
        from ddgclib.visualization import dynamic_plot_fluid
        HC, bV = mesh_2d
        history = self._make_history(HC, n_frames=2)
        frame_dir = tmp_path / 'frames'
        anim = dynamic_plot_fluid(
            history, HC, bV=bV,
            save_path=None, frame_dir=str(frame_dir), name='test',
        )
        # Trigger rendering by drawing first frame
        anim._init_draw()
        anim._step(0)
        anim._step(1)
        png_files = list(frame_dir.glob('test_*.png'))
        assert len(png_files) == 2

    def test_dynamic_save_gif(self, mesh_2d, tmp_path):
        from ddgclib.visualization import dynamic_plot_fluid
        HC, _ = mesh_2d
        history = self._make_history(HC, n_frames=2)
        out = tmp_path / 'test.gif'
        anim = dynamic_plot_fluid(
            history, HC, save_path=str(out), writer='pillow', fps=5,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_dynamic_scalar_only(self, mesh_2d):
        from ddgclib.visualization import dynamic_plot_fluid
        HC, _ = mesh_2d
        history = self._make_history(HC, n_frames=2)
        anim = dynamic_plot_fluid(
            history, HC, vector_field=None, save_path=None,
        )
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
            plot_primal,
            plot_dual,
            plot_fluid,
            dynamic_plot_fluid,
        )
        assert callable(plot_scalar_field_1d)
        assert callable(extract_slice_profile)
        assert callable(plot_primal)
        assert callable(plot_dual)
        assert callable(plot_fluid)
        assert callable(dynamic_plot_fluid)
