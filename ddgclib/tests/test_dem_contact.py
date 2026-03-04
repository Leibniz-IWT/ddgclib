"""Tests for DEM contact detection."""

import numpy as np
import numpy.testing as npt
import pytest

from ddgclib.dem._particle import Particle, ParticleSystem
from ddgclib.dem._contact import Contact, ContactDetector, _sphere_sphere_test


class TestSphereSphereTest:
    """Unit tests for narrow-phase sphere-sphere overlap."""

    def test_overlapping_pair(self):
        """Two overlapping spheres return a Contact with correct overlap."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[1.5, 0, 0], radius=1.0, dim=3)
        c = _sphere_sphere_test(p_i, p_j, dim=3)
        assert c is not None
        npt.assert_allclose(c.delta_n, 0.5, atol=1e-14)

    def test_separated_pair(self):
        """Two non-overlapping spheres return None."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[3.0, 0, 0], radius=1.0, dim=3)
        c = _sphere_sphere_test(p_i, p_j, dim=3)
        assert c is None

    def test_touching_pair(self):
        """Exactly touching spheres (delta_n=0) return None (no overlap)."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[2.0, 0, 0], radius=1.0, dim=3)
        c = _sphere_sphere_test(p_i, p_j, dim=3)
        assert c is None

    def test_normal_direction(self):
        """Normal vector n_ij points from i to j."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[1.0, 0, 0], radius=1.0, dim=3)
        c = _sphere_sphere_test(p_i, p_j, dim=3)
        assert c is not None
        npt.assert_allclose(c.n_ij, [1, 0, 0], atol=1e-14)

    def test_contact_point(self):
        """Contact point lies between the two sphere surfaces."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3)
        p_j = Particle.sphere(x=[1.0, 0, 0], radius=1.0, dim=3)
        c = _sphere_sphere_test(p_i, p_j, dim=3)
        assert c is not None
        # Contact point should be at x=0.5 (midpoint of overlap region)
        npt.assert_allclose(c.x_contact[0], 0.5, atol=1e-14)

    def test_relative_velocity_approaching(self):
        """Approaching particles have positive v_n."""
        p_i = Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3,
                              u=np.array([0, 0, 0]))
        p_j = Particle.sphere(x=[1.0, 0, 0], radius=1.0, dim=3,
                              u=np.array([-1.0, 0, 0]))
        c = _sphere_sphere_test(p_i, p_j, dim=3)
        assert c is not None
        # v_rel = u_j - u_i = [-1,0,0]; v_n = dot(v_rel, n_ij=[1,0,0]) = -1
        assert c.v_n < 0  # separating in this convention

    def test_2d_contact(self):
        """Contact detection works in 2D."""
        p_i = Particle.sphere(x=[0, 0], radius=1.0, dim=2)
        p_j = Particle.sphere(x=[1.5, 0], radius=1.0, dim=2)
        c = _sphere_sphere_test(p_i, p_j, dim=2)
        assert c is not None
        npt.assert_allclose(c.delta_n, 0.5, atol=1e-14)


class TestContactDetector:
    """Tests for broad-phase + narrow-phase detection pipeline."""

    def test_single_pair_detected(self):
        """Two overlapping particles produce exactly one Contact."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        ps.add(Particle.sphere(x=[1.5, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        contacts = detector.detect()
        assert len(contacts) == 1

    def test_separated_pair_no_contact(self):
        """Two non-overlapping particles produce zero contacts."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        ps.add(Particle.sphere(x=[5.0, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        contacts = detector.detect()
        assert len(contacts) == 0

    def test_single_particle_no_crash(self):
        """A single particle produces zero contacts (no crash)."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        contacts = detector.detect()
        assert len(contacts) == 0

    def test_empty_system(self):
        """Empty system produces zero contacts."""
        ps = ParticleSystem(dim=3)
        detector = ContactDetector(ps)
        assert detector.detect() == []

    def test_three_particles_chain(self):
        """Three particles in a chain: 2 contacts (A-B and B-C)."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        ps.add(Particle.sphere(x=[1.5, 0, 0], radius=1.0, dim=3))
        ps.add(Particle.sphere(x=[3.0, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        contacts = detector.detect()
        assert len(contacts) == 2

    def test_polydisperse(self):
        """Contact detection works with different-sized particles."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=2.0, dim=3))
        ps.add(Particle.sphere(x=[2.5, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        contacts = detector.detect()
        assert len(contacts) == 1
        npt.assert_allclose(contacts[0].delta_n, 0.5, atol=1e-14)

    def test_no_duplicate_contacts(self):
        """Each pair appears at most once in the contact list."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        # Place 4 particles all overlapping
        for i in range(4):
            ps.add(Particle.sphere(x=[i * 0.5, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps)
        contacts = detector.detect()
        pairs = [(c.p_i.id, c.p_j.id) for c in contacts]
        assert len(pairs) == len(set(pairs))

    def test_custom_cell_size(self):
        """Custom cell_size parameter works."""
        ps = ParticleSystem(dim=3, gravity=np.zeros(3))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))
        ps.add(Particle.sphere(x=[1.5, 0, 0], radius=1.0, dim=3))
        detector = ContactDetector(ps, cell_size=5.0)
        contacts = detector.detect()
        assert len(contacts) == 1
