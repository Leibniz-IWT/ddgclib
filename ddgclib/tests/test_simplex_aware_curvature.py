"""Regression tests for the simplex-aware curvature stencil refactor.

Covers the migration of the legacy ``vi.nn.intersection(vj.nn)`` apex
enumeration in :mod:`ddgclib._curvatures_heron` and
:mod:`ddgclib._curvatures_heron_torch_vectorized` to use the explicit
``HC._edge_to_apex`` cache (built lazily by
:func:`hyperct.ddg.get_edge_apex_map` from ``HC._simplices``) and the
explicit interface-triangle list (``HC.interface_triangles``).

The legacy flag-complex apex enumeration is unreliable on
Delaunay-derived meshes with skinny / sliver simplices because the
1-skeleton can contain spurious K_{dim+1} cliques near boundaries — see
``grok_1-skeleton-comment.pdf`` and
:func:`hyperct.ddg.boundary_from_simplices`.  The simplex-aware path
walks the explicit top-dim simplex list (or interface triangle list)
and is exact.

Mirrors the structure of :mod:`test_simplex_aware_duals`.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex
from hyperct.ddg import (
    connect_and_cache_simplices,
    get_edge_apex_map,
    invalidate_simplex_cache,
)

from ddgclib._curvatures_heron import (
    _apex_via_interface_triangles,
    _apex_via_simplex_cache,
    hndA_i,
    hndA_i_interface,
    integrated_hndA_i_interface,
)


# ---------------------------------------------------------------------------
# Apex-cache primitives
# ---------------------------------------------------------------------------


def _two_tets_sharing_face():
    """Hand-built 3D fixture: two tets sharing triangle (v1, v2, v3)."""
    HC = Complex(3, domain=[(-0.1, 1.1)] * 3)
    v0 = HC.V[(0.0, 0.0, 0.0)]
    v1 = HC.V[(1.0, 0.0, 0.0)]
    v2 = HC.V[(0.0, 1.0, 0.0)]
    v3 = HC.V[(0.0, 0.0, 1.0)]
    v4 = HC.V[(1.0, 1.0, 1.0)]
    verts = [v0, v1, v2, v3, v4]
    simplices = [[0, 1, 2, 3], [4, 1, 2, 3]]
    connect_and_cache_simplices(HC, verts, 3, simplices=simplices)
    return HC, verts


class TestEdgeApexMap:
    """Cache primitive: ``HC._edge_to_apex`` from ``HC._simplices``."""

    def test_returns_none_when_simplices_absent(self):
        HC = Complex(2, domain=[(0.0, 1.0)] * 2)
        HC.triangulate()
        assert getattr(HC, '_simplices', None) is None
        assert get_edge_apex_map(HC) is None

    def test_apex_for_shared_face(self):
        """For 3D tets, each tet edge has the OTHER two vertices of the
        tet as apex.  Edges (v1,v2), (v1,v3), (v2,v3) lie on the shared
        face — each belongs to both tets, so each contributes the
        union of the two ``other`` pairs."""
        HC, verts = _two_tets_sharing_face()
        v0, v1, v2, v3, v4 = verts
        apex_map = get_edge_apex_map(HC)
        assert apex_map is not None
        # Edge (v1, v2): tet 0 → apex = {v0, v3}; tet 1 → apex = {v3, v4}
        # Set union over both tets = {v0, v3, v4}
        assert set(apex_map[frozenset((id(v1), id(v2)))]) == {v0, v3, v4}
        assert set(apex_map[frozenset((id(v1), id(v3)))]) == {v0, v2, v4}
        assert set(apex_map[frozenset((id(v2), id(v3)))]) == {v0, v1, v4}

    def test_apex_for_unshared_edge(self):
        """Edge (v0, v1) belongs only to tet 0 → apex = {v2, v3} (the
        other two vertices of tet 0)."""
        HC, verts = _two_tets_sharing_face()
        v0, v1, v2, v3, _v4 = verts
        apex_map = get_edge_apex_map(HC)
        assert set(apex_map[frozenset((id(v0), id(v1)))]) == {v2, v3}

    def test_cache_is_reused(self):
        HC, _ = _two_tets_sharing_face()
        m1 = get_edge_apex_map(HC)
        m2 = get_edge_apex_map(HC)
        assert m1 is m2  # cached on HC._edge_to_apex

    def test_cache_invalidation(self):
        HC, _ = _two_tets_sharing_face()
        get_edge_apex_map(HC)
        assert HC._edge_to_apex is not None
        invalidate_simplex_cache(HC)
        # Both caches must be cleared.
        assert HC._simplices is None
        assert HC._edge_to_apex is None


class TestApexHelpersInCurvature:
    """The ``_apex_via_*`` helpers used inside the curvature loops."""

    def test_simplex_cache_returns_none_without_HC(self):
        assert _apex_via_simplex_cache(None, None, None) is None

    def test_simplex_cache_returns_none_when_unpopulated(self):
        HC = Complex(2, domain=[(0.0, 1.0)] * 2)
        HC.triangulate()
        # Need real vertices; a quick stand-in for the lookup signature.
        v_iter = iter(HC.V)
        vi = next(v_iter)
        vj = next(v_iter)
        assert _apex_via_simplex_cache(HC, vi, vj) is None

    def test_interface_triangles_returns_none_when_absent(self):
        HC = Complex(2, domain=[(0.0, 1.0)] * 2)
        HC.triangulate()
        v_iter = iter(HC.V)
        vi = next(v_iter)
        vj = next(v_iter)
        assert _apex_via_interface_triangles(HC, vi, vj) is None


# ---------------------------------------------------------------------------
# 3D fixture with embedded sphere — droplet_in_box_3d
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def droplet_3d():
    """3D box with embedded sphere; tets in HC._simplices, interface
    triangles in HC.interface_triangles."""
    from ddgclib.geometry.domains._multiphase_droplet import droplet_in_box_3d
    r = droplet_in_box_3d(
        R=0.01, L=0.05,
        refinement_outer=1, refinement_droplet=1,
    )
    return r


class TestHndAIInterfaceEquivalence3D:
    """Equivalence: legacy ``vi.nn ∩ vj.nn ∩ interface_set`` ≡
    interface-triangle-aware path on a clean droplet mesh.

    On the droplet_in_box_3d fixture the interface is a closed
    triangulated sphere (validated by ``validate_closure``).  Both apex
    paths must agree to machine precision; the simplex-aware path then
    becomes the primary, and the legacy path remains as fallback for
    surface meshes without an interface_triangles cache (``HC=None``).
    """

    def test_per_vertex_curvature_matches(self, droplet_3d):
        HC = droplet_3d.HC
        interface = droplet_3d.boundary_groups['interface']
        assert hasattr(HC, 'interface_triangles')
        assert len(HC.interface_triangles) > 0

        max_diff_HNdA = 0.0
        max_diff_C = 0.0
        for v in interface:
            interface_nbs = {nb for nb in v.nn if nb in interface} | {v}
            HNdA_legacy, C_legacy = hndA_i_interface(
                v, interface_nbs, HC=None,
            )
            HNdA_simplex, C_simplex = hndA_i_interface(
                v, interface_nbs, HC=HC,
            )
            diff = float(np.max(np.abs(HNdA_legacy - HNdA_simplex)))
            max_diff_HNdA = max(max_diff_HNdA, diff)
            max_diff_C = max(max_diff_C, abs(C_legacy - C_simplex))

        # Machine precision: the two paths enumerate the same triangle
        # set (clean mesh, no flag-clique ghosts).
        assert max_diff_HNdA < 1e-12, (
            f"hndA_i_interface legacy vs simplex-aware diverged: "
            f"max |ΔHNdA| = {max_diff_HNdA:.3e}"
        )
        assert max_diff_C < 1e-14, (
            f"Dual area diverged: max |ΔC| = {max_diff_C:.3e}"
        )

    def test_simplex_aware_uses_interface_cache(self, droplet_3d):
        """After one call with HC, ``HC._interface_edge_to_apex`` is
        populated and reused on subsequent calls."""
        HC = droplet_3d.HC
        # Reset any prior cache, then invoke simplex-aware path.
        if hasattr(HC, '_interface_edge_to_apex'):
            del HC._interface_edge_to_apex
        interface = droplet_3d.boundary_groups['interface']
        v = next(iter(interface))
        interface_nbs = {nb for nb in v.nn if nb in interface} | {v}
        hndA_i_interface(v, interface_nbs, HC=HC)
        cache = getattr(HC, '_interface_edge_to_apex', None)
        assert cache is not None
        assert len(cache) > 0


# ---------------------------------------------------------------------------
# Vectorized backend (torch wrapper) — same legacy/simplex-aware contract
# ---------------------------------------------------------------------------


class TestHndAIVectorizedEquivalence:
    """The vectorized scalar wrappers in
    :mod:`_curvatures_heron_torch_vectorized` must accept ``HC=...`` and
    agree with the non-vectorized scalar path on a closed-surface mesh.
    """

    def test_vectorized_hndA_i_falls_back_on_tet_mesh(self, droplet_3d):
        """``_apex_via_simplex_cache`` requires triangle simplices
        (3-tuples).  On a volumetric tet mesh (4-tuples) it must return
        ``None`` so the curvature loop falls back to the legacy
        ``vi.nn ∩ vj.nn`` path — passing ``HC`` then yields *bitwise*
        the same result as ``HC=None``."""
        from ddgclib._curvatures_heron_torch_vectorized import (
            hndA_i as hndA_i_vec,
        )
        HC = droplet_3d.HC
        # Confirm fixture is the volumetric tet case.
        assert HC._simplices is not None
        assert len(HC._simplices[0]) == 4

        interface = droplet_3d.boundary_groups['interface']
        sample = list(interface)[:5]
        for v in sample:
            HNdA_legacy, C_legacy = hndA_i_vec(v, HC=None)
            HNdA_simplex, C_simplex = hndA_i_vec(v, HC=HC)
            npt.assert_array_equal(HNdA_legacy, HNdA_simplex)
            assert C_legacy == C_simplex


# ---------------------------------------------------------------------------
# Integrated Stokes-based surface tension on 3D interface
# ---------------------------------------------------------------------------


class TestIntegratedHndAIInterface:
    """``integrated_hndA_i_interface`` — Stokes-theorem F_st on a 3D
    interface (barycentric dual sub-edge boundary integral)."""

    def test_returns_zero_when_HC_none(self):
        """No HC, no ``interface_triangles`` -> returns zeros (caller's
        responsibility to fall back to the cotangent form)."""
        F = integrated_hndA_i_interface(
            v=type('V', (), {'x': (0.0,), 'x_a': np.zeros(3)})(),
            interface_set=set(), HC=None, gamma=0.05,
        )
        npt.assert_array_equal(F, np.zeros(3))

    def test_returns_zero_when_gamma_zero(self, droplet_3d):
        """Early return on ``gamma == 0`` (no work, no allocation)."""
        HC = droplet_3d.HC
        interface = droplet_3d.boundary_groups['interface']
        v = next(iter(interface))
        F = integrated_hndA_i_interface(
            v, interface | {v}, HC=HC, gamma=0.0,
        )
        npt.assert_array_equal(F, np.zeros(3))

    def test_flat_interface_zero_to_machine_precision(self, capsys):
        """On a Kuhn-decomposed cube with a planar z=0 interface, the
        Stokes boundary integral around every interior interface vertex
        cancels by pairwise opposite-conormal contributions — F_st = 0
        to machine precision, NOT relying on a kappa = 0 sample."""
        from ddgclib.tests.test_multiphase_flat_interface import (
            _build_flat_interface_3d, _configure_flat_two_phase,
        )
        HC, bV = _build_flat_interface_3d(n_xy=3, n_half=2)
        mps = _configure_flat_two_phase(HC, dim=3, gamma=0.05)
        assert hasattr(HC, 'interface_triangles')
        assert len(HC.interface_triangles) > 0

        interface = {v for v in HC.V if getattr(v, 'is_interface', False)}
        max_F = 0.0
        for v in interface:
            if v in bV:
                continue  # truncated dual cell -> not interior
            F = integrated_hndA_i_interface(
                v, interface | {v}, HC=HC, gamma=0.05,
            )
            max_F = max(max_F, float(np.linalg.norm(F)))
        with capsys.disabled():
            print(f"\n[flat z=0] integrated_hndA_i_interface: "
                  f"max|F_st| over interior interface = {max_F:.3e}")
        assert max_F < 1e-12, (
            f"Stokes integrated F_st should be machine zero on flat "
            f"interface; got max|F| = {max_F:.3e}"
        )

    def test_spherical_interface_points_inward(self, droplet_3d, capsys):
        """On a closed spherical interface, F_st on each interface vertex
        must point inward (toward the sphere centre) — anti-parallel to
        the outward normal at v.  This is the Young-Laplace inward pull.

        Also checks the *sign convention* matches the existing cotangent
        path: both forms point inward on a convex droplet, so the dot
        product of the two F_st vectors is positive on every vertex.
        """
        from ddgclib.operators.multiphase_stress import _interface_surface_tension  # noqa
        HC = droplet_3d.HC
        interface = droplet_3d.boundary_groups['interface']
        # Sphere centre = mean of interface vertex positions (droplet_in_box_3d
        # centres the droplet at the box centroid).
        centre = np.mean(np.array([v.x_a[:3] for v in interface]), axis=0)
        gamma = 0.05

        n_iface = 0
        n_inward = 0
        n_sign_match = 0
        for v in interface:
            x = np.asarray(v.x_a[:3], dtype=float)
            radial_out = x - centre
            r = float(np.linalg.norm(radial_out))
            if r < 1e-12:
                continue
            radial_out /= r

            F_new = integrated_hndA_i_interface(
                v, interface | {v}, HC=HC, gamma=gamma,
            )
            # Existing cotangent form (negative HNdA -> F_st)
            HNdA_cot, _ = hndA_i_interface(v, interface | {v}, HC=HC)
            F_cot = -gamma * HNdA_cot[:3]

            n_iface += 1
            if float(np.dot(F_new, radial_out)) < 0:
                n_inward += 1
            if float(np.dot(F_new, F_cot)) > 0:
                n_sign_match += 1

        with capsys.disabled():
            print(f"\n[sphere] integrated_hndA_i_interface: "
                  f"{n_inward}/{n_iface} inward, "
                  f"{n_sign_match}/{n_iface} sign-agree with cotangent")
        # All interface vertices must have F_st pointing inward and the
        # sign must agree with the existing cotangent form.
        assert n_inward == n_iface, (
            f"Stokes F_st should point inward on every interface vertex; "
            f"got {n_inward}/{n_iface}"
        )
        assert n_sign_match == n_iface, (
            f"Stokes F_st sign disagrees with cotangent F_st on "
            f"{n_iface - n_sign_match}/{n_iface} vertices"
        )

    def test_spherical_interface_total_force_balances(self, droplet_3d, capsys):
        """Symmetry sanity: on a closed sphere, sum of F_st_i over every
        interface vertex must be ~zero (Newton's third law for the
        whole interface).  The bound is generous because the sphere is
        not perfectly symmetric on the icosphere-style mesh."""
        HC = droplet_3d.HC
        interface = droplet_3d.boundary_groups['interface']
        gamma = 0.05
        total = np.zeros(3)
        max_F = 0.0
        for v in interface:
            F = integrated_hndA_i_interface(
                v, interface | {v}, HC=HC, gamma=gamma,
            )
            total += F
            max_F = max(max_F, float(np.linalg.norm(F)))
        with capsys.disabled():
            print(f"\n[sphere] integrated_hndA_i_interface: "
                  f"sum |F_st| = {np.linalg.norm(total):.3e}  "
                  f"max per-vertex |F_st| = {max_F:.3e}")
        # Sum should be small compared to per-vertex magnitude; on a
        # closed icosphere the imbalance is O(1e-3) relative.
        assert np.linalg.norm(total) < 1e-2 * max_F * len(interface) ** 0.5, (
            f"Total F_st on closed sphere should ~cancel by symmetry; "
            f"got |sum| = {np.linalg.norm(total):.3e}, "
            f"max per-vertex = {max_F:.3e}"
        )
