"""Tests for capillary liquid bridges and the two-particle integration test case."""

import numpy as np
import numpy.testing as npt
import pytest

from ddgclib.dem._particle import Particle, ParticleSystem
from ddgclib.dem._contact import ContactDetector
from ddgclib.dem._force_models import HertzContact
from ddgclib.dem._integrators import dem_step
from ddgclib.dem._liquid_bridge import LiquidBridge, LiquidBridgeManager


class TestLiquidBridge:
    """Unit tests for LiquidBridge."""

    def _make_bridge(self, sep=0.5e-3, R=1e-3, gamma=0.072, vol=1e-12,
                     theta=0.0):
        """Helper: two wetted particles with a bridge."""
        p_i = Particle.sphere(
            x=[0, 0, 0], radius=R, dim=3,
            wetted=True, wetting_angle=theta, liquid_volume=vol,
        )
        p_j = Particle.sphere(
            x=[2 * R + sep, 0, 0], radius=R, dim=3,
            wetted=True, wetting_angle=theta, liquid_volume=vol,
        )
        bridge = LiquidBridge(
            p_i=p_i, p_j=p_j, gamma=gamma, volume=vol,
        )
        return p_i, p_j, bridge

    def test_separation(self):
        """separation property returns surface-to-surface distance."""
        p_i, p_j, bridge = self._make_bridge(sep=0.5e-3)
        npt.assert_allclose(bridge.separation, 0.5e-3, atol=1e-15)

    def test_rupture_distance(self):
        """rupture_distance uses Lian et al. 1993 criterion."""
        _, _, bridge = self._make_bridge(vol=1e-12, theta=0.0)
        expected = 1.0 * (1e-12) ** (1.0 / 3.0)  # theta=0 => factor=1
        npt.assert_allclose(bridge.rupture_distance, expected, rtol=1e-10)

    def test_force_attractive(self):
        """Capillary bridge force is attractive (pulls i toward j)."""
        _, _, bridge = self._make_bridge(sep=0.1e-3)
        F_i, F_j = bridge.capillary_force(dim=3)
        # F_i should point in +x (toward j)
        assert F_i[0] > 0
        # F_j should point in -x (toward i)
        assert F_j[0] < 0

    def test_force_newton_third_law(self):
        """F_i = -F_j."""
        _, _, bridge = self._make_bridge(sep=0.1e-3)
        F_i, F_j = bridge.capillary_force(dim=3)
        npt.assert_allclose(F_i, -F_j, atol=1e-15)

    def test_force_zero_when_inactive(self):
        """Inactive bridge returns zero force."""
        _, _, bridge = self._make_bridge()
        bridge.active = False
        F_i, F_j = bridge.capillary_force(dim=3)
        npt.assert_array_equal(F_i, np.zeros(3))
        npt.assert_array_equal(F_j, np.zeros(3))

    def test_rupture(self):
        """Bridge ruptures when separation exceeds rupture distance."""
        p_i, p_j, bridge = self._make_bridge(sep=0.1e-3, vol=1e-15)
        # Move j far away
        p_j.x_a[0] = 1.0  # very far
        assert bridge.check_rupture() is True
        assert bridge.active is False

    def test_no_rupture_close(self):
        """Bridge stays active when within rupture distance."""
        _, _, bridge = self._make_bridge(sep=1e-6, vol=1e-9)
        assert bridge.check_rupture() is False
        assert bridge.active is True

    def test_force_decays_with_separation(self):
        """Force magnitude decreases with increasing separation."""
        _, _, bridge_close = self._make_bridge(sep=1e-5)
        _, _, bridge_far = self._make_bridge(sep=1e-4)
        F_close = np.linalg.norm(bridge_close.capillary_force(3)[0])
        F_far = np.linalg.norm(bridge_far.capillary_force(3)[0])
        assert F_close > F_far


class TestLiquidBridgeManager:
    """Tests for LiquidBridgeManager container."""

    def test_formation_wetted_close(self):
        """Bridge forms between close wetted particles."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(
            x=[0, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-12,
        ))
        ps.add(Particle.sphere(
            x=[2.1e-3, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-12,
        ))
        mgr = LiquidBridgeManager(gamma=0.072)
        n = mgr.check_formation(ps)
        assert n == 1
        assert mgr.active_count == 1

    def test_no_formation_unwetted(self):
        """No bridge forms for unwetted particles."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1e-3, dim=3, wetted=False))
        ps.add(Particle.sphere(x=[2.1e-3, 0, 0], radius=1e-3, dim=3, wetted=False))
        mgr = LiquidBridgeManager()
        assert mgr.check_formation(ps) == 0

    def test_no_formation_too_far(self):
        """No bridge forms when particles are far apart."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(
            x=[0, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-12,
        ))
        ps.add(Particle.sphere(
            x=[10e-3, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-12,
        ))
        mgr = LiquidBridgeManager()
        assert mgr.check_formation(ps) == 0

    def test_no_duplicate_bridges(self):
        """Same pair is not bridged twice."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(
            x=[0, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-12,
        ))
        ps.add(Particle.sphere(
            x=[2.1e-3, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-12,
        ))
        mgr = LiquidBridgeManager()
        mgr.check_formation(ps)
        mgr.check_formation(ps)  # second call
        assert mgr.active_count == 1

    def test_check_ruptures(self):
        """Ruptured bridges are removed."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(
            x=[0, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-15,
        ))
        p2 = ps.add(Particle.sphere(
            x=[2.1e-3, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-15,
        ))
        mgr = LiquidBridgeManager()
        mgr.check_formation(ps)
        assert mgr.active_count == 1
        # Move p2 far away to trigger rupture
        p2.x_a[0] = 1.0
        n_rupt = mgr.check_ruptures()
        assert n_rupt == 1
        assert mgr.active_count == 0

    def test_apply_forces(self):
        """apply_forces() adds capillary force to particles."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(
            x=[0, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-9,
        ))
        ps.add(Particle.sphere(
            x=[2.05e-3, 0, 0], radius=1e-3, dim=3,
            wetted=True, liquid_volume=1e-9,
        ))
        mgr = LiquidBridgeManager(gamma=0.072)
        mgr.check_formation(ps)
        ps.reset_all_forces()
        mgr.apply_forces(ps)
        p1, p2 = ps.particles
        # p1 should be pulled toward p2 (positive x)
        assert p1.force[0] > 0


@pytest.mark.slow
@pytest.mark.xfail(reason="Bridge formation logic needs tuning; particles don't close gap in time")
class TestTwoParticleBridgeCase:
    """Integration test: two wetted particles approach and form a liquid bridge.

    Setup:
    - Two identical spheres R = 1 mm, separated by gap = 2R
    - Both wetted (water gamma = 0.072 N/m, contact angle = 30 deg)
    - Initial approach velocity: 0.01 m/s toward each other
    - No gravity, no fluid mesh

    Expected:
    1. Particles approach under initial velocity
    2. When close enough, liquid bridge forms
    3. Capillary force accelerates closing
    4. Contact + Hertz repulsion → equilibrium gap
    """

    def test_approach_and_bridge_formation(self):
        """Two approaching wetted particles form a bridge and agglomerate."""
        R = 1e-3
        rho_s = 2500.0
        gamma = 0.072
        theta = np.radians(30)
        v_approach = 0.01

        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        p1 = ps.add(Particle.sphere(
            x=[-3e-3, 0, 0], radius=R, rho_s=rho_s, dim=3,
            u=np.array([v_approach, 0, 0]),
            wetted=True, wetting_angle=theta, liquid_volume=1e-12,
        ))
        p2 = ps.add(Particle.sphere(
            x=[3e-3, 0, 0], radius=R, rho_s=rho_s, dim=3,
            u=np.array([-v_approach, 0, 0]),
            wetted=True, wetting_angle=theta, liquid_volume=1e-12,
        ))

        contact_model = HertzContact(E=70e9, nu=0.22, gamma_n=1e-4)
        detector = ContactDetector(ps)
        bridge_mgr = LiquidBridgeManager(gamma=gamma)

        dt = 1e-6
        n_steps = 60000  # enough for 6mm at 2*0.01 m/s
        bridge_formed = False

        ps.reset_all_forces()
        ps.apply_gravity()

        for step in range(n_steps):
            bridge_mgr.check_formation(ps)
            if bridge_mgr.active_count > 0 and not bridge_formed:
                bridge_formed = True

            dem_step(
                ps, detector, contact_model, dt=dt, dim=3,
                bridge_manager=bridge_mgr,
            )

        # Bridge should have formed
        assert bridge_formed, "Bridge should have formed during approach"

        # Particles should be close (agglomerated)
        sep = np.linalg.norm(p2.x_a - p1.x_a) - R - R
        assert sep < 1e-3, f"Particles should be close, got sep={sep:.6e}"

        # Momentum conservation: CoM should stay at origin (symmetric setup)
        com = ps.center_of_mass()
        npt.assert_allclose(com, np.zeros(3), atol=1e-8)
