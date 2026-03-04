"""Tests for DEM Particle and ParticleSystem data structures."""

import numpy as np
import numpy.testing as npt
import pytest

from ddgclib.dem._particle import Particle, ParticleSystem


class TestParticle:
    """Unit tests for Particle dataclass."""

    def test_sphere_3d_mass(self):
        """Particle.sphere() computes correct 3D sphere mass."""
        R = 1e-3
        rho = 2500.0
        p = Particle.sphere(x=[0, 0, 0], radius=R, rho_s=rho, dim=3)
        expected = rho * (4.0 / 3.0) * np.pi * R**3
        npt.assert_allclose(p.m, expected, rtol=1e-12)

    def test_sphere_3d_inertia(self):
        """Particle.sphere() computes correct moment of inertia (2/5 mR^2)."""
        R = 1e-3
        rho = 2500.0
        p = Particle.sphere(x=[0, 0, 0], radius=R, rho_s=rho, dim=3)
        expected_I = 0.4 * p.m * R**2
        npt.assert_allclose(p.I, expected_I, rtol=1e-12)

    def test_sphere_2d_mass(self):
        """Particle.sphere() computes correct 2D disk mass."""
        R = 1e-3
        rho = 2500.0
        p = Particle.sphere(x=[0, 0], radius=R, rho_s=rho, dim=2)
        expected = rho * np.pi * R**2
        npt.assert_allclose(p.m, expected, rtol=1e-12)

    def test_sphere_2d_inertia(self):
        """2D disk moment of inertia: 1/2 mR^2."""
        R = 1e-3
        rho = 2500.0
        p = Particle.sphere(x=[0, 0], radius=R, rho_s=rho, dim=2)
        expected_I = 0.5 * p.m * R**2
        npt.assert_allclose(p.I, expected_I, rtol=1e-12)

    def test_sphere_initial_velocity(self):
        """Particle.sphere() with custom initial velocity."""
        u0 = np.array([1.0, 2.0, 3.0])
        p = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3, u=u0)
        npt.assert_array_equal(p.u, u0)

    def test_sphere_default_zero_velocity(self):
        """Default velocity is zero."""
        p = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        npt.assert_array_equal(p.u, np.zeros(3))

    def test_reset_forces(self):
        """reset_forces() zeros force and torque accumulators."""
        p = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p.force[:] = [1.0, 2.0, 3.0]
        p.torque[:] = [4.0, 5.0, 6.0]
        p.reset_forces()
        npt.assert_array_equal(p.force, np.zeros(3))
        npt.assert_array_equal(p.torque, np.zeros(3))

    def test_kinetic_energy(self):
        """Kinetic energy = 0.5*m*|u|^2 + 0.5*I*|omega|^2."""
        p = Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3)
        p.u = np.array([1.0, 0.0, 0.0])
        p.omega = np.array([0.0, 0.0, 2.0])
        expected = 0.5 * p.m * 1.0 + 0.5 * p.I * 4.0
        npt.assert_allclose(p.kinetic_energy, expected, rtol=1e-12)

    def test_dim_property(self):
        """dim property matches position vector length."""
        p2 = Particle.sphere(x=[0, 0], radius=1.0, dim=2)
        p3 = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        assert p2.dim == 2
        assert p3.dim == 3

    def test_wetted_attributes(self):
        """Particle.sphere() passes through wetted kwargs."""
        p = Particle.sphere(
            x=[0, 0, 0], radius=1e-3, dim=3,
            wetted=True, wetting_angle=0.5, liquid_volume=1e-12,
        )
        assert p.wetted is True
        npt.assert_allclose(p.wetting_angle, 0.5)
        npt.assert_allclose(p.liquid_volume, 1e-12)


class TestParticleSystem:
    """Unit tests for ParticleSystem container."""

    def test_add_assigns_unique_ids(self):
        """Each added particle gets a unique sequential id."""
        ps = ParticleSystem(dim=3)
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[1, 0, 0], radius=1.0, dim=3))
        assert p1.id == 0
        assert p2.id == 1
        assert len(ps) == 2

    def test_remove(self):
        """Particle can be removed from system."""
        ps = ParticleSystem(dim=3)
        p = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        assert len(ps) == 1
        ps.remove(p)
        assert len(ps) == 0

    def test_getitem_by_id(self):
        """Particles can be looked up by id."""
        ps = ParticleSystem(dim=3)
        p = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        assert ps[p.id] is p

    def test_iter(self):
        """ParticleSystem is iterable."""
        ps = ParticleSystem(dim=3)
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[1, 0, 0], radius=1.0, dim=3))
        result = list(ps)
        assert result == [p1, p2]

    def test_positions(self):
        """positions() returns (N, dim) array."""
        ps = ParticleSystem(dim=3)
        ps.add(Particle.sphere(x=[1, 2, 3], radius=1.0, dim=3))
        ps.add(Particle.sphere(x=[4, 5, 6], radius=1.0, dim=3))
        pos = ps.positions()
        assert pos.shape == (2, 3)
        npt.assert_array_equal(pos[0], [1, 2, 3])
        npt.assert_array_equal(pos[1], [4, 5, 6])

    def test_velocities(self):
        """velocities() returns (N, dim) array."""
        ps = ParticleSystem(dim=3)
        p = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3,
                                    u=np.array([1.0, 2.0, 3.0])))
        vel = ps.velocities()
        npt.assert_array_equal(vel[0], [1.0, 2.0, 3.0])

    def test_radii(self):
        """radii() returns (N,) array."""
        ps = ParticleSystem(dim=3)
        ps.add(Particle.sphere(x=[0, 0, 0], radius=0.5, dim=3))
        ps.add(Particle.sphere(x=[1, 0, 0], radius=1.5, dim=3))
        r = ps.radii()
        npt.assert_array_equal(r, [0.5, 1.5])

    def test_default_gravity_3d(self):
        """Default gravity is [0, 0, -9.81] in 3D."""
        ps = ParticleSystem(dim=3)
        npt.assert_array_equal(ps.gravity, [0, 0, -9.81])

    def test_default_gravity_2d(self):
        """Default gravity is [0, -9.81] in 2D."""
        ps = ParticleSystem(dim=2)
        npt.assert_array_equal(ps.gravity, [0, -9.81])

    def test_custom_gravity(self):
        """Custom gravity vector is set correctly."""
        ps = ParticleSystem(dim=3, gravity=np.array([0, 0, 0]))
        npt.assert_array_equal(ps.gravity, [0, 0, 0])

    def test_apply_gravity(self):
        """apply_gravity() adds m*g to non-boundary particles."""
        ps = ParticleSystem(dim=3, gravity=np.array([0, 0, -10.0]))
        p = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        ps.reset_all_forces()
        ps.apply_gravity()
        expected_Fz = p.m * (-10.0)
        npt.assert_allclose(p.force[2], expected_Fz)

    def test_apply_gravity_skips_boundary(self):
        """Boundary particles are not affected by gravity."""
        ps = ParticleSystem(dim=3, gravity=np.array([0, 0, -10.0]))
        p = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p.boundary = True
        ps.reset_all_forces()
        ps.apply_gravity()
        npt.assert_array_equal(p.force, np.zeros(3))

    def test_cluster_management(self):
        """new_cluster_id() returns unique ids; cluster_particles() filters."""
        ps = ParticleSystem(dim=3)
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[1, 0, 0], radius=1.0, dim=3))
        p3 = ps.add(Particle.sphere(x=[2, 0, 0], radius=1.0, dim=3))
        cid = ps.new_cluster_id()
        p1.cluster_id = cid
        p2.cluster_id = cid
        result = ps.cluster_particles(cid)
        assert p1 in result
        assert p2 in result
        assert p3 not in result

    def test_total_kinetic_energy(self):
        """total_kinetic_energy() sums all particles."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[1, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        p1.u = np.array([1.0, 0, 0])
        p2.u = np.array([0, 1.0, 0])
        ke = ps.total_kinetic_energy()
        expected = 0.5 * p1.m * 1.0 + 0.5 * p2.m * 1.0
        npt.assert_allclose(ke, expected, rtol=1e-12)

    def test_total_momentum(self):
        """total_momentum() returns sum of m*u."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[1, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        p1.u = np.array([1.0, 0, 0])
        p2.u = np.array([-1.0, 0, 0])
        mom = ps.total_momentum()
        npt.assert_allclose(mom, np.zeros(3), atol=1e-15)

    def test_center_of_mass(self):
        """center_of_mass() returns mass-weighted average position."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        p2 = ps.add(Particle.sphere(x=[2, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        com = ps.center_of_mass()
        npt.assert_allclose(com, [1.0, 0, 0], atol=1e-15)

    def test_empty_system(self):
        """Empty system returns correct defaults for batch accessors."""
        ps = ParticleSystem(dim=3)
        assert ps.positions().shape == (0, 3)
        assert ps.velocities().shape == (0, 3)
        npt.assert_array_equal(ps.total_momentum(), np.zeros(3))
        npt.assert_array_equal(ps.center_of_mass(), np.zeros(3))
        assert ps.total_kinetic_energy() == 0.0
