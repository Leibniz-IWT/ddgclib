"""Tests for sintered particle bonds and aggregate tracking."""

import numpy as np
import numpy.testing as npt
import pytest

from ddgclib.dem._particle import Particle, ParticleSystem
from ddgclib.dem._bonds import SinterBond, BondManager


class TestSinterBond:
    """Unit tests for SinterBond."""

    def test_form_at_current_positions(self):
        """SinterBond.form() records correct rest length."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2.5, 0, 0], radius=1.0, dim=3)
        bond = SinterBond.form(p_i, p_j, t=1.0)
        npt.assert_allclose(bond.rest_length, 2.5, atol=1e-14)
        assert bond.active is True
        npt.assert_allclose(bond.formation_time, 1.0)

    def test_initial_neck_radius(self):
        """Initial neck radius is fraction of min(R_i, R_j)."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=2.0, dim=3)
        p_j = Particle.sphere(x=[3, 0, 0], radius=1.0, dim=3)
        bond = SinterBond.form(p_i, p_j, initial_neck_ratio=0.2)
        npt.assert_allclose(bond.neck_radius, 0.2 * 1.0)

    def test_zero_stretch_zero_force(self):
        """No force when particles are at rest length."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3)
        bond = SinterBond.form(p_i, p_j)
        F_i, F_j, _, _ = bond.force_and_torque(dim=3)
        npt.assert_allclose(F_i, np.zeros(3), atol=1e-10)
        npt.assert_allclose(F_j, np.zeros(3), atol=1e-10)

    def test_tensile_force(self):
        """Bond under tension: pulls particles toward each other."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3)
        bond = SinterBond.form(p_i, p_j, k_bond=1e4, c_bond=0.0)
        # Move j further away (stretch > 0)
        p_j.x_a[0] = 2.5
        F_i, F_j, _, _ = bond.force_and_torque(dim=3)
        # F_i should pull i toward j (positive x)
        assert F_i[0] > 0
        # F_j should pull j toward i (negative x)
        assert F_j[0] < 0

    def test_compressive_force(self):
        """Bond under compression: pushes particles apart."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3)
        bond = SinterBond.form(p_i, p_j, k_bond=1e4, c_bond=0.0)
        # Move j closer (stretch < 0)
        p_j.x_a[0] = 1.5
        F_i, F_j, _, _ = bond.force_and_torque(dim=3)
        assert F_i[0] < 0
        assert F_j[0] > 0

    def test_newton_third_law(self):
        """F_i = -F_j."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3)
        bond = SinterBond.form(p_i, p_j, k_bond=1e4, c_bond=0.0)
        p_j.x_a[0] = 2.5
        F_i, F_j, _, _ = bond.force_and_torque(dim=3)
        npt.assert_allclose(F_i, -F_j, atol=1e-14)

    def test_fracture(self):
        """Bond breaks when strain exceeds max_strain."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3)
        bond = SinterBond.form(p_i, p_j, max_strain=0.1)
        # Move j to exceed 10% strain
        p_j.x_a[0] = 2.0 + 0.25  # 12.5% strain
        bond.force_and_torque(dim=3)
        assert bond.active is False

    def test_no_fracture_within_threshold(self):
        """Bond stays active when strain is below max_strain."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3)
        bond = SinterBond.form(p_i, p_j, max_strain=0.1)
        p_j.x_a[0] = 2.0 + 0.15  # 7.5% strain
        bond.force_and_torque(dim=3)
        assert bond.active is True

    def test_neck_growth_monotonic(self):
        """Neck radius increases over time."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1e-6, dim=3)
        p_j = Particle.sphere(x=[2e-6, 0, 0], radius=1e-6, dim=3)
        bond = SinterBond.form(p_i, p_j, initial_neck_ratio=0.1)
        r0 = bond.neck_radius
        bond.grow_neck(dt=1e-3, D=1e-15)
        assert bond.neck_radius > r0

    def test_neck_capped_at_radius(self):
        """Neck radius cannot exceed min particle radius."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2, 0, 0], radius=0.5, dim=3)
        bond = SinterBond.form(p_i, p_j, initial_neck_ratio=0.9)
        # Grow with very large D to force overshoot
        bond.grow_neck(dt=1e10, D=1e10)
        assert bond.neck_radius <= 0.5


class TestBondManager:
    """Tests for BondManager container."""

    def test_add_bond(self):
        """Bonds can be added to manager."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[2, 0, 0], radius=1.0, dim=3))
        bond = SinterBond.form(p1, p2)
        mgr = BondManager()
        mgr.add(bond, ps)
        assert len(mgr.bonds) == 1

    def test_cluster_id_assignment(self):
        """add() assigns shared cluster_id to bonded particles."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[2, 0, 0], radius=1.0, dim=3))
        assert p1.cluster_id is None
        assert p2.cluster_id is None
        bond = SinterBond.form(p1, p2)
        mgr = BondManager()
        mgr.add(bond, ps)
        assert p1.cluster_id is not None
        assert p1.cluster_id == p2.cluster_id

    def test_apply_forces(self):
        """apply_forces() accumulates bond forces on particles."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3))
        bond = SinterBond.form(p1, p2, k_bond=1e4, c_bond=0.0)
        mgr = BondManager()
        mgr.add(bond, ps)
        # Stretch bond
        p2.x_a[0] = 2.5
        ps.reset_all_forces()
        mgr.apply_forces(ps)
        assert p1.force[0] > 0  # pulled toward j
        assert p2.force[0] < 0  # pulled toward i

    def test_remove_broken(self):
        """remove_broken() removes fractured bonds."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3))
        bond = SinterBond.form(p1, p2, max_strain=0.05)
        mgr = BondManager()
        mgr.add(bond, ps)
        # Force fracture
        bond.active = False
        n_removed = mgr.remove_broken()
        assert n_removed == 1
        assert len(mgr.bonds) == 0

    def test_grow_all_necks(self):
        """grow_all_necks() advances sintering on all active bonds."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1e-6, dim=3))
        p2 = ps.add(Particle.sphere(x=[2e-6, 0, 0], radius=1e-6, dim=3))
        bond = SinterBond.form(p1, p2, initial_neck_ratio=0.1)
        mgr = BondManager()
        mgr.add(bond, ps)
        r_before = bond.neck_radius
        mgr.grow_all_necks(dt=1e-3, D=1e-15)
        assert bond.neck_radius > r_before

    def test_bonds_for_particle(self):
        """bonds_for_particle() returns correct bonds."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[2, 0, 0], radius=1.0, dim=3))
        p3 = ps.add(Particle.sphere(x=[4, 0, 0], radius=1.0, dim=3))
        b12 = SinterBond.form(p1, p2)
        b23 = SinterBond.form(p2, p3)
        mgr = BondManager()
        mgr.add(b12, ps)
        mgr.add(b23, ps)
        assert len(mgr.bonds_for_particle(p1)) == 1
        assert len(mgr.bonds_for_particle(p2)) == 2
        assert len(mgr.bonds_for_particle(p3)) == 1

    def test_active_count(self):
        """active_count property."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[2, 0, 0], radius=1.0, dim=3))
        bond = SinterBond.form(p1, p2)
        mgr = BondManager()
        mgr.add(bond, ps)
        assert mgr.active_count == 1
        bond.active = False
        assert mgr.active_count == 0
