"""Tests for DEM contact force models."""

import numpy as np
import numpy.testing as npt
import pytest

from ddgclib.dem._particle import Particle, ParticleSystem
from ddgclib.dem._contact import Contact, _sphere_sphere_test
from ddgclib.dem._force_models import (
    HertzContact,
    LinearSpringDashpot,
    ContactForceResult,
    contact_force_registry,
)


def _make_overlapping_contact(
    delta_n: float = 0.1,
    dim: int = 3,
    u_i=None,
    u_j=None,
) -> Contact:
    """Helper: create a contact with specified overlap along x-axis."""
    R = 1.0
    sep = 2 * R - delta_n  # center-to-center distance
    p_i = Particle.sphere(
        x=np.zeros(dim), radius=R, dim=dim,
        u=u_i if u_i is not None else np.zeros(dim),
    )
    p_j = Particle.sphere(
        x=np.array([sep] + [0] * (dim - 1)), radius=R, dim=dim,
        u=u_j if u_j is not None else np.zeros(dim),
    )
    c = _sphere_sphere_test(p_i, p_j, dim)
    assert c is not None, f"Expected contact with delta_n={delta_n}"
    return c


class TestHertzContact:
    """Tests for Hertzian contact force model."""

    def test_repulsive_normal_force(self):
        """Normal force is repulsive (pushes i away from j)."""
        model = HertzContact(E=1e6, nu=0.3, gamma_n=0.0)
        c = _make_overlapping_contact(delta_n=0.1)
        result = model.compute(c)
        # F_n should point in -x direction (away from j)
        assert result.F_n[0] < 0

    def test_hertz_force_scales_with_overlap_3_2(self):
        """Hertz force scales as delta_n^(3/2) when damping=0."""
        model = HertzContact(E=1e6, nu=0.3, gamma_n=0.0, gamma_t=0.0)
        c1 = _make_overlapping_contact(delta_n=0.1)
        c2 = _make_overlapping_contact(delta_n=0.2)
        r1 = model.compute(c1)
        r2 = model.compute(c2)
        F1 = np.linalg.norm(r1.F_n)
        F2 = np.linalg.norm(r2.F_n)
        ratio = F2 / F1
        expected_ratio = (0.2 / 0.1) ** 1.5
        npt.assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_newton_third_law(self):
        """Force on j = -force on i."""
        model = HertzContact()
        c = _make_overlapping_contact(delta_n=0.1)
        result = model.compute(c)
        npt.assert_allclose(
            result.total_force_on_j, -result.total_force_on_i, atol=1e-15
        )

    def test_zero_tangential_no_sliding(self):
        """No tangential force when particles have identical velocity."""
        model = HertzContact(gamma_n=0.0)
        c = _make_overlapping_contact(delta_n=0.1)
        result = model.compute(c)
        npt.assert_allclose(result.F_t, np.zeros(3), atol=1e-15)

    def test_tangential_force_opposes_sliding(self):
        """Tangential force opposes relative tangential velocity."""
        model = HertzContact(gamma_t=1.0, mu_friction=10.0)
        # Give j a tangential velocity relative to i
        c = _make_overlapping_contact(
            delta_n=0.1,
            u_j=np.array([0.0, 1.0, 0.0]),
        )
        result = model.compute(c)
        # F_t should be in -y direction (opposing j's tangential motion)
        assert result.F_t[1] < 0

    def test_damping_reduces_force(self):
        """Damping term changes force magnitude when particles approach."""
        model_no_damp = HertzContact(E=1e6, nu=0.3, gamma_n=0.0)
        model_damp = HertzContact(E=1e6, nu=0.3, gamma_n=1e3)
        c = _make_overlapping_contact(
            delta_n=0.1,
            u_j=np.array([-1.0, 0, 0]),  # approaching
        )
        r_no = model_no_damp.compute(c)
        r_yes = model_damp.compute(c)
        # Forces should differ due to damping
        assert not np.allclose(r_no.F_n, r_yes.F_n)

    def test_name(self):
        """Model name is 'hertz'."""
        assert HertzContact().name() == "hertz"

    def test_2d_contact(self):
        """Hertz model works in 2D."""
        model = HertzContact(E=1e6, nu=0.3, gamma_n=0.0)
        c = _make_overlapping_contact(delta_n=0.1, dim=2)
        result = model.compute(c)
        assert result.F_n[0] < 0
        assert len(result.torque_i) == 1  # 2D torque is scalar


class TestLinearSpringDashpot:
    """Tests for linear spring-dashpot contact force model."""

    def test_repulsive_normal_force(self):
        """Normal force is repulsive."""
        model = LinearSpringDashpot(k_n=1e5, c_n=0.0)
        c = _make_overlapping_contact(delta_n=0.1)
        result = model.compute(c)
        assert result.F_n[0] < 0

    def test_linear_scaling(self):
        """LSD force is linear in overlap."""
        model = LinearSpringDashpot(k_n=1e5, c_n=0.0, c_t=0.0)
        c1 = _make_overlapping_contact(delta_n=0.1)
        c2 = _make_overlapping_contact(delta_n=0.2)
        r1 = model.compute(c1)
        r2 = model.compute(c2)
        F1 = np.linalg.norm(r1.F_n)
        F2 = np.linalg.norm(r2.F_n)
        npt.assert_allclose(F2 / F1, 2.0, rtol=1e-10)

    def test_newton_third_law(self):
        """Force on j = -force on i."""
        model = LinearSpringDashpot()
        c = _make_overlapping_contact(delta_n=0.1)
        result = model.compute(c)
        npt.assert_allclose(
            result.total_force_on_j, -result.total_force_on_i, atol=1e-15
        )

    def test_name(self):
        """Model name is 'linear_spring_dashpot'."""
        assert LinearSpringDashpot().name() == "linear_spring_dashpot"


class TestForceRegistry:
    """Tests for the contact force model registry."""

    def test_hertz_registered(self):
        """Hertz model is registered."""
        assert "hertz" in contact_force_registry

    def test_lsd_registered(self):
        """Linear spring-dashpot model is registered."""
        assert "linear_spring_dashpot" in contact_force_registry

    def test_lookup(self):
        """Registry lookup returns the correct class."""
        cls = contact_force_registry["hertz"]
        assert cls is HertzContact

    def test_available(self):
        """available() lists both models."""
        avail = contact_force_registry.available()
        assert "hertz" in avail
        assert "linear_spring_dashpot" in avail

    def test_unknown_raises(self):
        """Unknown model key raises KeyError."""
        with pytest.raises(KeyError):
            contact_force_registry["nonexistent"]
