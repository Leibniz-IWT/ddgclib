"""
Tests for vectorized Heron curvature kernels.

Validates that:
- Numpy batch kernel matches scalar kernel to machine precision
- Torch batch kernel matches scalar kernel to machine precision
- Mesh-level assembler produces correct curvature on sphere meshes
"""
import numpy as np
import pytest

from ddgclib._curvatures_heron import (
    HNdC_ijk,
    HNdC_ijk_batch,
    heron_mean_curvature_vectors,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Analytical test cases (from PR #29)
# ---------------------------------------------------------------------------
_SQRT3 = np.sqrt(3.0)

ANALYTICAL_CASES = [
    {
        'name': 'equilateral_side_1',
        'e_ij': np.array([1.0, 0.0, 0.0]),
        'l_ij': 1.0,
        'l_jk': 1.0,
        'l_ik': 1.0,
        'theory_hnda': np.array([1.0 / (2.0 * _SQRT3), 0.0, 0.0]),
        'theory_c': 1.0 / (8.0 * _SQRT3),
    },
    {
        'name': 'right_triangle_zero_weight',
        'e_ij': np.array([-1.0, 1.0, 0.0]),
        'l_ij': np.sqrt(2.0),
        'l_jk': 1.0,
        'l_ik': 1.0,
        'theory_hnda': np.array([0.0, 0.0, 0.0]),
        'theory_c': 0.0,
    },
    {
        'name': 'obtuse_120deg',
        'e_ij': np.array([-1.5, np.sqrt(3.0) / 2.0, 0.0]),
        'l_ij': np.sqrt(3.0),
        'l_jk': 1.0,
        'l_ik': 1.0,
        'theory_hnda': np.array([np.sqrt(3.0) / 4.0, -0.25, 0.0]),
        'theory_c': np.sqrt(3.0) / 8.0,
    },
]


class TestHNdCIjkBatchNumpy:
    """Test that HNdC_ijk_batch matches scalar HNdC_ijk."""

    @pytest.mark.parametrize('case', ANALYTICAL_CASES, ids=lambda c: c['name'])
    def test_analytical_single(self, case):
        """Single-triangle batch should match scalar exactly."""
        e_ij = case['e_ij'][np.newaxis, :]
        l_ij = np.array([case['l_ij']])
        l_jk = np.array([case['l_jk']])
        l_ik = np.array([case['l_ik']])

        hnda_batch, c_batch = HNdC_ijk_batch(e_ij, l_ij, l_jk, l_ik)
        hnda_scalar, c_scalar = HNdC_ijk(case['e_ij'], case['l_ij'],
                                          case['l_jk'], case['l_ik'])

        np.testing.assert_allclose(hnda_batch[0], hnda_scalar, atol=1e-15)
        np.testing.assert_allclose(c_batch[0], c_scalar, atol=1e-15)

    @pytest.mark.parametrize('case', ANALYTICAL_CASES, ids=lambda c: c['name'])
    def test_analytical_vs_theory(self, case):
        """Batch result should match analytical theory."""
        e_ij = case['e_ij'][np.newaxis, :]
        l_ij = np.array([case['l_ij']])
        l_jk = np.array([case['l_jk']])
        l_ik = np.array([case['l_ik']])

        hnda_batch, c_batch = HNdC_ijk_batch(e_ij, l_ij, l_jk, l_ik)

        np.testing.assert_allclose(hnda_batch[0], case['theory_hnda'], atol=1e-15)
        np.testing.assert_allclose(c_batch[0], case['theory_c'], atol=1e-15)

    def test_random_batch_matches_scalar(self):
        """Batch of random triangles should match element-wise scalar calls."""
        rng = np.random.default_rng(42)
        n = 500
        p_i = rng.standard_normal((n, 3))
        p_j = rng.standard_normal((n, 3))
        p_k = rng.standard_normal((n, 3))

        e_ij = p_j - p_i
        e_jk = p_k - p_j
        e_ik = p_k - p_i
        l_ij = np.linalg.norm(e_ij, axis=1)
        l_jk = np.linalg.norm(e_jk, axis=1)
        l_ik = np.linalg.norm(e_ik, axis=1)

        hnda_batch, c_batch = HNdC_ijk_batch(e_ij, l_ij, l_jk, l_ik)

        for idx in range(n):
            hnda_s, c_s = HNdC_ijk(e_ij[idx], l_ij[idx], l_jk[idx], l_ik[idx])
            np.testing.assert_allclose(hnda_batch[idx], hnda_s, atol=1e-14,
                                       err_msg=f"hnda mismatch at idx={idx}")
            np.testing.assert_allclose(c_batch[idx], c_s, atol=1e-14,
                                       err_msg=f"c mismatch at idx={idx}")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestHNdCIjkBatchTorch:
    """Test that HNdC_ijk_batch_torch matches scalar HNdC_ijk."""

    @pytest.fixture(autouse=True)
    def _import_torch_kernel(self):
        from ddgclib._curvatures_heron import HNdC_ijk_batch_torch
        self.kernel = HNdC_ijk_batch_torch

    @pytest.mark.parametrize('case', ANALYTICAL_CASES, ids=lambda c: c['name'])
    def test_analytical_single(self, case):
        e_ij = case['e_ij'][np.newaxis, :]
        l_ij = np.array([case['l_ij']])
        l_jk = np.array([case['l_jk']])
        l_ik = np.array([case['l_ik']])

        hnda_t, c_t = self.kernel(e_ij, l_ij, l_jk, l_ik)
        hnda_np = hnda_t.detach().cpu().numpy()
        c_np = c_t.detach().cpu().numpy()

        hnda_scalar, c_scalar = HNdC_ijk(case['e_ij'], case['l_ij'],
                                          case['l_jk'], case['l_ik'])

        np.testing.assert_allclose(hnda_np[0], hnda_scalar, atol=1e-14)
        np.testing.assert_allclose(c_np[0], c_scalar, atol=1e-14)

    def test_random_batch_matches_numpy(self):
        rng = np.random.default_rng(42)
        n = 500
        p_i = rng.standard_normal((n, 3))
        p_j = rng.standard_normal((n, 3))
        p_k = rng.standard_normal((n, 3))

        e_ij = p_j - p_i
        e_jk = p_k - p_j
        e_ik = p_k - p_i
        l_ij = np.linalg.norm(e_ij, axis=1)
        l_jk = np.linalg.norm(e_jk, axis=1)
        l_ik = np.linalg.norm(e_ik, axis=1)

        hnda_np, c_np = HNdC_ijk_batch(e_ij, l_ij, l_jk, l_ik)

        hnda_t, c_t = self.kernel(e_ij, l_ij, l_jk, l_ik)
        hnda_torch = hnda_t.detach().cpu().numpy()
        c_torch = c_t.detach().cpu().numpy()

        np.testing.assert_allclose(hnda_torch, hnda_np, atol=1e-14)
        np.testing.assert_allclose(c_torch, c_np, atol=1e-14)


# ---------------------------------------------------------------------------
# Sphere mesh helpers (from PR #29, simplified)
# ---------------------------------------------------------------------------

def _build_regular_icosahedron(radius=1.0):
    """Regular icosahedron inscribed in a sphere of given radius."""
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    verts = np.array([
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ], dtype=np.float64)
    verts = radius * verts / np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)

    # Orient faces outward
    for idx, tri in enumerate(faces):
        a, b, c = verts[tri]
        normal = np.cross(b - a, c - a)
        if np.dot(normal, (a + b + c) / 3.0) < 0:
            faces[idx] = [tri[0], tri[2], tri[1]]
    return verts, faces


class TestHeronMeanCurvatureVectors:
    """Test mesh-level assembler on sphere meshes."""

    def test_icosahedron_curvature_direction(self):
        """On a sphere, HNdA should point radially outward."""
        R = 1.0
        verts, faces = _build_regular_icosahedron(R)
        H = heron_mean_curvature_vectors(verts, faces, backend='numpy')

        rhat = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        H_norm = np.linalg.norm(H, axis=1, keepdims=True)
        H_hat = H / np.maximum(H_norm, 1e-30)

        alignment = np.sum(H_hat * rhat, axis=1)
        # All vertices should have curvature aligned with radial direction
        np.testing.assert_allclose(np.abs(alignment), 1.0, atol=1e-10)

    def test_icosahedron_curvature_magnitude(self):
        """On a regular icosahedron (equal edges), H/R ~ 2/R by symmetry."""
        R = 2.0
        verts, faces = _build_regular_icosahedron(R)
        H = heron_mean_curvature_vectors(verts, faces, backend='numpy')

        # Compute dual area per vertex (sum of c_ijk)
        n_verts = verts.shape[0]
        C = np.zeros(n_verts)
        i_arr, j_arr, k_arr = faces[:, 0], faces[:, 1], faces[:, 2]
        vi, vj, vk = verts[i_arr], verts[j_arr], verts[k_arr]
        e_ij = vj - vi; e_ik = vk - vi; e_jk = vk - vj
        l_ij = np.linalg.norm(e_ij, axis=1)
        l_ik = np.linalg.norm(e_ik, axis=1)
        l_jk = np.linalg.norm(e_jk, axis=1)
        _, c_ij = HNdC_ijk_batch(e_ij, l_ij, l_jk, l_ik)
        _, c_ik = HNdC_ijk_batch(e_ik, l_ik, l_jk, l_ij)
        _, c_ji = HNdC_ijk_batch(-e_ij, l_ij, l_ik, l_jk)
        _, c_jk = HNdC_ijk_batch(e_jk, l_jk, l_ik, l_ij)
        _, c_ki = HNdC_ijk_batch(-e_ik, l_ik, l_ij, l_jk)
        _, c_kj = HNdC_ijk_batch(-e_jk, l_jk, l_ij, l_ik)
        np.add.at(C, i_arr, c_ij); np.add.at(C, i_arr, c_ik)
        np.add.at(C, j_arr, c_ji); np.add.at(C, j_arr, c_jk)
        np.add.at(C, k_arr, c_ki); np.add.at(C, k_arr, c_kj)

        H_mag = np.linalg.norm(H, axis=1)
        abs_curvature = H_mag / np.maximum(C, 1e-30)
        # Mean curvature on a sphere of radius R is 2/R = 1.0
        expected = 2.0 / R
        np.testing.assert_allclose(abs_curvature, expected, rtol=1e-10)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_matches_numpy(self):
        """Torch backend should match numpy backend on icosahedron."""
        R = 1.0
        verts, faces = _build_regular_icosahedron(R)
        H_np = heron_mean_curvature_vectors(verts, faces, backend='numpy')
        H_torch = heron_mean_curvature_vectors(verts, faces, backend='torch')
        np.testing.assert_allclose(H_torch, H_np, atol=1e-14)

    def test_hyperct_numpy_backend(self):
        """Passing a hyperct NumpyBackend object should match string 'numpy'."""
        from hyperct._backend import NumpyBackend
        R = 1.0
        verts, faces = _build_regular_icosahedron(R)
        H_str = heron_mean_curvature_vectors(verts, faces, backend='numpy')
        H_obj = heron_mean_curvature_vectors(verts, faces, backend=NumpyBackend())
        np.testing.assert_allclose(H_obj, H_str, atol=1e-15)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_hyperct_torch_backend(self):
        """Passing a hyperct TorchBackend object should match string 'torch'."""
        from hyperct._backend import TorchBackend
        R = 1.0
        verts, faces = _build_regular_icosahedron(R)
        H_str = heron_mean_curvature_vectors(verts, faces, backend='numpy')
        H_obj = heron_mean_curvature_vectors(verts, faces, backend=TorchBackend())
        np.testing.assert_allclose(H_obj, H_str, atol=1e-14)

    def test_registry_access(self):
        """'heron-vectorized' should be registered in curvature_i_methods."""
        from ddgclib.operators.curvature import curvature_i_methods, _ensure_benchmarks_registered
        _ensure_benchmarks_registered()
        assert "heron-vectorized" in curvature_i_methods
