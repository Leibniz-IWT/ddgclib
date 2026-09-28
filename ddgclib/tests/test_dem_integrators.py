"""Tests for DEM time integrators."""

import numpy as np
import numpy.testing as npt
import pytest

from ddgclib.dem._particle import Particle, ParticleSystem
from ddgclib.dem._contact import ContactDetector
from ddgclib.dem._force_models import HertzContact, LinearSpringDashpot
from ddgclib.dem._integrators import (
    dem_velocity_verlet,
    dem_symplectic_euler,
    dem_step,
)


def _make_free_particle_system(dim=3):
    """Single particle system with no contacts, zero gravity."""
    ps = ParticleSystem(dim=dim, gravity=np.zeros(dim))
    p = ps.add(Particle.sphere(x=np.zeros(dim), radius=1.0, rho_s=1.0, dim=dim))
    return ps, p


class TestFreeFall:
    """Test particle under constant gravity (no contacts)."""

    def test_free_fall_velocity_verlet(self):
        """Free-fall position: x(t) = 0.5 * g * t^2."""
        g = -10.0
        ps = ParticleSystem(dim=3, gravity=np.array([0, 0, g]))
        p = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))

        model = HertzContact()
        detector = ContactDetector(ps)

        dt = 1e-4
        n_steps = 1000
        t = 0.0

        # Need initial force computation for Velocity Verlet
        ps.reset_all_forces()
        ps.apply_gravity()

        for _ in range(n_steps):
            dem_velocity_verlet(ps, detector, model, dt, dim=3)
            t += dt

        expected_z = 0.5 * g * t**2
        npt.assert_allclose(p.x_a[2], expected_z, rtol=1e-3)

    def test_free_fall_symplectic_euler(self):
        """Free-fall with symplectic Euler, slightly less accurate."""
        g = -10.0
        ps = ParticleSystem(dim=3, gravity=np.array([0, 0, g]))
        p = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))

        model = HertzContact()
        detector = ContactDetector(ps)

        dt = 1e-4
        n_steps = 1000
        t = n_steps * dt

        for _ in range(n_steps):
            dem_symplectic_euler(ps, detector, model, dt, dim=3)

        expected_z = 0.5 * g * t**2
        npt.assert_allclose(p.x_a[2], expected_z, rtol=5e-2)

    def test_free_fall_velocity_linear(self):
        """Under constant gravity, velocity increases linearly."""
        g = -10.0
        ps = ParticleSystem(dim=3, gravity=np.array([0, 0, g]))
        p = ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))

        model = HertzContact()
        detector = ContactDetector(ps)

        ps.reset_all_forces()
        ps.apply_gravity()

        dt = 1e-4
        n_steps = 1000
        for _ in range(n_steps):
            dem_velocity_verlet(ps, detector, model, dt, dim=3)

        expected_vz = g * n_steps * dt
        npt.assert_allclose(p.u[2], expected_vz, rtol=1e-6)


class TestElasticCollision:
    """Two equal particles in head-on elastic collision."""

    def test_momentum_conservation(self):
        """Total momentum is conserved during elastic collision."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(
            x=[-0.5, 0, 0], radius=0.6, rho_s=1.0, dim=3,
            u=np.array([1.0, 0, 0]),
        ))
        p2 = ps.add(Particle.sphere(
            x=[0.5, 0, 0], radius=0.6, rho_s=1.0, dim=3,
            u=np.array([-1.0, 0, 0]),
        ))
        mom_initial = ps.total_momentum().copy()

        model = LinearSpringDashpot(k_n=1e5, c_n=0.0, c_t=0.0)
        detector = ContactDetector(ps)

        ps.reset_all_forces()
        ps.apply_gravity()

        dt = 1e-5
        for _ in range(5000):
            dem_velocity_verlet(ps, detector, model, dt, dim=3)

        mom_final = ps.total_momentum()
        npt.assert_allclose(mom_final, mom_initial, atol=1e-10)

    def test_energy_conservation_undamped(self):
        """Total KE is approximately conserved for undamped elastic collision."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        # Start separated (distance > R_i + R_j) so no initial overlap energy
        ps.add(Particle.sphere(
            x=[-2.0, 0, 0], radius=0.6, rho_s=1.0, dim=3,
            u=np.array([1.0, 0, 0]),
        ))
        ps.add(Particle.sphere(
            x=[2.0, 0, 0], radius=0.6, rho_s=1.0, dim=3,
            u=np.array([-1.0, 0, 0]),
        ))
        ke_initial = ps.total_kinetic_energy()

        model = LinearSpringDashpot(k_n=1e5, c_n=0.0, c_t=0.0)
        detector = ContactDetector(ps)

        ps.reset_all_forces()
        ps.apply_gravity()

        dt = 1e-5
        # Run long enough for approach + collision + separation
        for _ in range(30000):
            dem_velocity_verlet(ps, detector, model, dt, dim=3)

        ke_final = ps.total_kinetic_energy()
        # After collision, KE should be close to initial (elastic, no damping)
        npt.assert_allclose(ke_final, ke_initial, rtol=0.05)

    def test_com_stays_at_origin(self):
        """Center of mass stays at origin for symmetric collision."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(
            x=[-1.0, 0, 0], radius=0.6, rho_s=1.0, dim=3,
            u=np.array([1.0, 0, 0]),
        ))
        ps.add(Particle.sphere(
            x=[1.0, 0, 0], radius=0.6, rho_s=1.0, dim=3,
            u=np.array([-1.0, 0, 0]),
        ))

        model = LinearSpringDashpot(k_n=1e5, c_n=0.0, c_t=0.0)
        detector = ContactDetector(ps)

        ps.reset_all_forces()
        ps.apply_gravity()

        dt = 1e-5
        for _ in range(5000):
            dem_velocity_verlet(ps, detector, model, dt, dim=3)

        com = ps.center_of_mass()
        npt.assert_allclose(com, np.zeros(3), atol=1e-10)


class TestDemStep:
    """Tests for the dem_step() entry point."""

    def test_dem_step_velocity_verlet(self):
        """dem_step with velocity_verlet method runs without error."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        model = HertzContact()
        ps.reset_all_forces()
        dem_step(ps, detector, model, dt=1e-4, dim=3, method="velocity_verlet")

    def test_dem_step_symplectic_euler(self):
        """dem_step with symplectic_euler method runs without error."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        model = HertzContact()
        dem_step(ps, detector, model, dt=1e-4, dim=3, method="symplectic_euler")

    def test_substepping(self):
        """Sub-stepping produces similar result to full stepping."""
        # Single step of dt
        ps1 = ParticleSystem(dim=3, gravity=np.array([0, 0, -10.0]))
        p1 = ps1.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        det1 = ContactDetector(ps1)
        m = HertzContact()
        ps1.reset_all_forces()
        ps1.apply_gravity()
        dem_step(ps1, det1, m, dt=1e-3, dim=3, n_sub=1, method="velocity_verlet")

        # 10 sub-steps of dt/10
        ps2 = ParticleSystem(dim=3, gravity=np.array([0, 0, -10.0]))
        p2 = ps2.add(Particle.sphere(x=[0, 0, 0], radius=1.0, rho_s=1.0, dim=3))
        det2 = ContactDetector(ps2)
        ps2.reset_all_forces()
        ps2.apply_gravity()
        dem_step(ps2, det2, m, dt=1e-3, dim=3, n_sub=10, method="velocity_verlet")

        # Both should give same result for free fall (no contacts)
        npt.assert_allclose(p1.x_a, p2.x_a, rtol=1e-3)

    def test_callback(self):
        """Callback is invoked at each sub-step."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        model = HertzContact()

        calls = []
        dem_step(ps, detector, model, dt=1e-3, dim=3, n_sub=5,
                 callback=lambda step, t, ps: calls.append(step))
        assert calls == [0, 1, 2, 3, 4]

    def test_unknown_method_raises(self):
        """Unknown integration method raises ValueError."""
        ps = ParticleSystem(dim=3)
        detector = ContactDetector(ps)
        model = HertzContact()
        with pytest.raises(ValueError, match="Unknown DEM method"):
            dem_step(ps, detector, model, dt=1e-4, method="runge_kutta_9000")

    def test_boundary_particle_frozen(self):
        """Boundary particles don't move or accelerate."""
        ps = ParticleSystem(dim=3, gravity=np.array([0, 0, -10.0]))
        p = ps.add(Particle.sphere(x=[0, 0, 5.0], radius=1.0, dim=3))
        p.boundary = True
        detector = ContactDetector(ps)
        model = HertzContact()
        x_initial = p.x_a.copy()

        dem_step(ps, detector, model, dt=1e-3, dim=3, n_sub=10)
        npt.assert_array_equal(p.x_a, x_initial)
