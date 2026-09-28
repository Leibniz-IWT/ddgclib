"""Tests for the integrated 2D curvature operator.

Validates the core identity

    integral_{Gamma_i} kappa * N ds = t_next - t_prev

on synthetic and real interface meshes.  The constant-curvature case
(the circle) is reconstructed to machine precision.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ddgclib.operators.curvature_2d import (
    integrated_curvature_normal_2d,
    surface_tension_force_2d,
    reconstruct_arc_length_and_bulge_area,
)


class _FakeVertex:
    """Minimal duck-typed vertex for unit tests."""

    def __init__(self, x, is_interface=True):
        self.x_a = np.asarray(x, dtype=float)
        self.is_interface = is_interface
        self.nn = set()


def _build_ring(N: int, R: float = 1.0):
    """Return a list of ``N`` interface vertices evenly spaced on a
    circle of radius ``R``, with 1-ring neighbours wired up as a ring."""
    verts = []
    for k in range(N):
        theta = 2 * np.pi * k / N
        verts.append(_FakeVertex([R * np.cos(theta), R * np.sin(theta)]))
    for k in range(N):
        verts[k].nn = {verts[(k - 1) % N], verts[(k + 1) % N]}
    return verts


class TestIntegratedCurvatureCircle:
    """The integrated curvature normal on a regular polygon on a circle
    should equal ``t_next - t_prev`` with magnitude ``2 sin(pi/N)``."""

    @pytest.mark.parametrize("N", [4, 6, 12, 32, 128])
    def test_magnitude_matches_analytical(self, N):
        R = 1.0
        verts = _build_ring(N, R=R)
        expected_mag = 2.0 * math.sin(math.pi / N)
        for v in verts:
            I = integrated_curvature_normal_2d(v)
            assert I.shape == (2,)
            assert abs(np.linalg.norm(I) - expected_mag) < 1e-12

    @pytest.mark.parametrize("N", [4, 8, 16, 64])
    def test_points_inward(self, N):
        R = 1.5
        verts = _build_ring(N, R=R)
        for v in verts:
            I = integrated_curvature_normal_2d(v)
            # For a convex closed curve the curvature normal points
            # toward the interior (toward the origin here).
            radial_outward = v.x_a / np.linalg.norm(v.x_a)
            assert np.dot(I, radial_outward) < -1e-12

    @pytest.mark.parametrize("N", [5, 16, 64])
    def test_sum_around_closed_curve_is_zero(self, N):
        """Total integrated curvature over a closed curve is zero."""
        verts = _build_ring(N, R=1.3)
        total = sum(
            (integrated_curvature_normal_2d(v) for v in verts),
            np.zeros(2),
        )
        assert np.linalg.norm(total) < 1e-12

    def test_surface_tension_force_scales_with_gamma(self):
        verts = _build_ring(16, R=1.0)
        v = verts[0]
        F1 = surface_tension_force_2d(v, gamma=0.05)
        F2 = surface_tension_force_2d(v, gamma=0.10)
        assert np.allclose(F2, 2.0 * F1)

    def test_zero_gamma_returns_zero(self):
        verts = _build_ring(8, R=1.0)
        F = surface_tension_force_2d(verts[0], gamma=0.0)
        assert np.allclose(F, 0.0)

    def test_isolated_vertex_returns_zero(self):
        """A vertex with <2 interface neighbours has zero contribution."""
        v = _FakeVertex([0.0, 0.0])
        v.nn = set()
        F = surface_tension_force_2d(v, gamma=0.05)
        assert np.allclose(F, 0.0)


class TestArcLengthReconstruction:
    """Closed-form arc length and bulge area from the notebook should be
    exact (machine precision) for constant-curvature edges."""

    def test_quarter_circle_arc_length(self):
        v_i = np.array([1.0, 0.0])
        v_j = np.array([0.0, 1.0])
        t_i = np.array([0.0, 1.0])
        t_j = np.array([-1.0, 0.0])
        Delta_T = t_j - t_i
        L, A_bulge, r, c = reconstruct_arc_length_and_bulge_area(
            v_i, v_j, Delta_T,
        )
        assert abs(L - math.pi / 2) < 1e-14
        assert abs(r - 1.0) < 1e-14
        assert abs(c - math.sqrt(2)) < 1e-14
        expected_bulge = 0.5 * (math.pi / 2 - 1.0)  # sector - triangle
        assert abs(A_bulge - expected_bulge) < 1e-14

    def test_full_circle_from_N_segments(self):
        """Summing the closed-form edge arc lengths around a regular
        N-gon on the unit circle recovers 2*pi to machine precision."""
        for N in (4, 6, 12, 60, 360):
            verts = _build_ring(N, R=1.0)
            theta_per_edge = 2 * math.pi / N
            dT_mag = 2 * math.sin(theta_per_edge / 2)
            total_L = 0.0
            for k in range(N):
                v = verts[k]
                v_next = verts[(k + 1) % N]
                L, _, _, _ = reconstruct_arc_length_and_bulge_area(
                    v.x_a, v_next.x_a, np.array([dT_mag, 0.0]),
                )
                total_L += L
            assert abs(total_L - 2 * math.pi) < 1e-12


class TestIntegratedCurvatureOnDroplet:
    """End-to-end sanity check against a real oscillating droplet mesh."""

    def test_unperturbed_droplet_forces_point_inward(self):
        pytest.importorskip("cases_dynamic.oscillating_droplet.src._setup",
                            reason="oscillating_droplet case required")
        from cases_dynamic.oscillating_droplet.src._setup import (
            setup_oscillating_droplet,
        )
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
            setup_oscillating_droplet(
                dim=2, R0=0.01, epsilon=0.0, l=2,
                rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
                gamma=0.05, L_domain=0.05,
                refinement_outer=1, refinement_droplet=2,
            )
        n_checked = 0
        for v in HC.V:
            if not getattr(v, 'is_interface', False):
                continue
            I = integrated_curvature_normal_2d(v)
            if np.linalg.norm(I) < 1e-14:
                continue
            radial = v.x_a[:2] / np.linalg.norm(v.x_a[:2])
            assert np.dot(I, radial) < 0.0
            n_checked += 1
        assert n_checked > 4
