"""Tests for GPU/backend support in dual mesh computation.

Verifies that compute_vd works correctly with different backends
(numpy, multiprocessing, torch) and that results are consistent.
"""
import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex
from hyperct.ddg import compute_vd
from hyperct._backend import get_backend


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def simple_2d():
    """2D mesh on [0,1]^2 with boundary vertices marked."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    for v in HC.V:
        v.boundary = any(
            abs(v.x_a[i] - b) < 1e-14
            for i in range(2) for b in (0.0, 1.0)
        )
    return HC


@pytest.fixture
def simple_3d():
    """3D mesh on [0,1]^3 with boundary vertices marked."""
    HC = Complex(3, domain=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    for v in HC.V:
        v.boundary = any(
            abs(v.x_a[i] - b) < 1e-14
            for i in range(3) for b in (0.0, 1.0)
        )
    return HC


class TestNumpyBackend:
    def test_compute_vd_barycentric_2d(self, simple_2d):
        backend = get_backend("numpy")
        compute_vd(simple_2d, method="barycentric", backend=backend)
        assert hasattr(simple_2d, 'Vd')
        assert len(simple_2d.Vd) > 0

    def test_compute_vd_circumcentric_2d(self, simple_2d):
        backend = get_backend("numpy")
        compute_vd(simple_2d, method="circumcentric", backend=backend)
        assert hasattr(simple_2d, 'Vd')
        assert len(simple_2d.Vd) > 0

    def test_compute_vd_barycentric_3d(self, simple_3d):
        backend = get_backend("numpy")
        compute_vd(simple_3d, method="barycentric", backend=backend)
        assert hasattr(simple_3d, 'Vd')
        assert len(simple_3d.Vd) > 0

    def test_dual_vertices_populated(self, simple_2d):
        compute_vd(simple_2d, method="barycentric")
        for v in simple_2d.V:
            if not getattr(v, 'boundary', False):
                assert hasattr(v, 'vd'), f"Vertex at {v.x} missing vd"
                assert len(v.vd) > 0, f"Vertex at {v.x} has empty vd"


class TestMultiprocessingBackend:
    def test_compute_vd_2d(self, simple_2d):
        backend = get_backend("multiprocessing")
        compute_vd(simple_2d, method="barycentric", backend=backend)
        assert len(simple_2d.Vd) > 0


class TestBarycVsCircum:
    def test_different_dual_positions_2d(self, simple_2d):
        """Barycentric and circumcentric should produce different dual positions."""
        HC_b = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC_b.triangulate()
        HC_b.refine_all()
        for v in HC_b.V:
            v.boundary = any(
                abs(v.x_a[i] - b) < 1e-14
                for i in range(2) for b in (0.0, 1.0)
            )

        HC_c = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC_c.triangulate()
        HC_c.refine_all()
        for v in HC_c.V:
            v.boundary = any(
                abs(v.x_a[i] - b) < 1e-14
                for i in range(2) for b in (0.0, 1.0)
            )

        compute_vd(HC_b, method="barycentric")
        compute_vd(HC_c, method="circumcentric")

        # Both should produce dual vertices but at different positions
        assert len(HC_b.Vd) > 0
        assert len(HC_c.Vd) > 0

        # Collect dual positions
        pos_b = sorted([tuple(vd.x_a) for vd in HC_b.Vd])
        pos_c = sorted([tuple(vd.x_a) for vd in HC_c.Vd])

        # They should differ (unless the mesh is perfectly symmetric)
        if len(pos_b) == len(pos_c):
            differs = any(
                not np.allclose(pb, pc, atol=1e-10)
                for pb, pc in zip(pos_b, pos_c)
            )
            assert differs, "Barycentric and circumcentric duals should differ"


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
class TestTorchBackend:
    def test_compute_vd_2d(self, simple_2d):
        backend = get_backend("torch")
        compute_vd(simple_2d, method="barycentric", backend=backend)
        assert len(simple_2d.Vd) > 0

    def test_compute_vd_3d(self, simple_3d):
        backend = get_backend("torch")
        compute_vd(simple_3d, method="barycentric", backend=backend)
        assert len(simple_3d.Vd) > 0

    def test_consistency_with_numpy(self):
        """Torch backend should produce same dual count as numpy backend."""
        HC_np = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC_np.triangulate()
        HC_np.refine_all()
        for v in HC_np.V:
            v.boundary = any(
                abs(v.x_a[i] - b) < 1e-14
                for i in range(2) for b in (0.0, 1.0)
            )

        HC_t = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC_t.triangulate()
        HC_t.refine_all()
        for v in HC_t.V:
            v.boundary = any(
                abs(v.x_a[i] - b) < 1e-14
                for i in range(2) for b in (0.0, 1.0)
            )

        compute_vd(HC_np, method="barycentric", backend=get_backend("numpy"))
        compute_vd(HC_t, method="barycentric", backend=get_backend("torch"))

        assert len(HC_np.Vd) == len(HC_t.Vd), \
            f"Numpy produced {len(HC_np.Vd)} duals, torch produced {len(HC_t.Vd)}"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
class TestCudaBackend:
    def test_cuda_detected(self):
        backend = get_backend("torch")
        assert backend.has_cuda, "CUDA should be available"

    def test_compute_vd_cuda_2d(self, simple_2d):
        backend = get_backend("torch")
        compute_vd(simple_2d, method="barycentric", backend=backend)
        assert len(simple_2d.Vd) > 0
