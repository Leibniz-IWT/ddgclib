"""Tests for DEM fluid-particle coupling and I/O."""

import json
import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

from ddgclib.dem._particle import Particle, ParticleSystem
from ddgclib.dem._coupling import (
    interpolate_fluid_at_particle,
    apply_particle_feedback,
    _stokes_drag,
    _schiller_naumann_drag,
    FluidParticleCoupler,
)
from ddgclib.dem._io import (
    save_particles,
    load_particles,
    import_particle_cloud,
)


# ---------------------------------------------------------------------------
# Mock vertex for fluid coupling tests (no hyperct dependency needed)
# ---------------------------------------------------------------------------

class MockVertex:
    """Minimal vertex stand-in with x_a, u, p attributes."""

    def __init__(self, x, u=None, p=0.0, m=1.0):
        self.x_a = np.asarray(x, dtype=float)
        self.u = np.asarray(u, dtype=float) if u is not None else np.zeros_like(self.x_a)
        self.p = np.array([p])
        self.m = m


class MockComplex:
    """Minimal Complex stand-in with iterable V."""

    def __init__(self, vertices):
        self.V = vertices


# ---------------------------------------------------------------------------
# Interpolation tests
# ---------------------------------------------------------------------------

class TestInterpolation:
    """Tests for IDW fluid velocity interpolation."""

    def _make_uniform_field(self, dim=3, n=10, u_val=None):
        """Create mock vertices with a uniform velocity field."""
        if u_val is None:
            u_val = np.ones(dim)
        rng = np.random.default_rng(42)
        verts = []
        for _ in range(n):
            x = rng.uniform(-1, 1, dim)
            verts.append(MockVertex(x=x, u=u_val, p=5.0))
        positions = np.array([v.x_a[:dim] for v in verts])
        return verts, positions

    def test_uniform_field_exact(self):
        """IDW interpolation recovers a uniform field exactly."""
        dim = 3
        u_val = np.array([2.0, -1.0, 0.5])
        verts, positions = self._make_uniform_field(dim=dim, u_val=u_val)
        p = Particle.sphere(x=[0, 0, 0], radius=0.1, dim=dim)

        u_interp, p_interp = interpolate_fluid_at_particle(
            p, verts, positions, dim, n_nearest=6,
        )
        npt.assert_allclose(u_interp, u_val, atol=1e-12)
        npt.assert_allclose(p_interp, 5.0, atol=1e-12)

    def test_uniform_field_exact_with_kdtree(self):
        """IDW interpolation with KDTree also recovers uniform field."""
        pytest.importorskip("scipy")
        from scipy.spatial import KDTree

        dim = 3
        u_val = np.array([2.0, -1.0, 0.5])
        verts, positions = self._make_uniform_field(dim=dim, u_val=u_val)
        kdtree = KDTree(positions)
        p = Particle.sphere(x=[0, 0, 0], radius=0.1, dim=dim)

        u_interp, p_interp = interpolate_fluid_at_particle(
            p, verts, positions, dim, n_nearest=6, kdtree=kdtree,
        )
        npt.assert_allclose(u_interp, u_val, atol=1e-12)

    def test_nearest_dominates(self):
        """Vertex at particle position dominates interpolation."""
        dim = 3
        verts = [
            MockVertex(x=[0, 0, 0], u=[10, 0, 0]),
            MockVertex(x=[1, 0, 0], u=[0, 0, 0]),
            MockVertex(x=[0, 1, 0], u=[0, 0, 0]),
            MockVertex(x=[0, 0, 1], u=[0, 0, 0]),
        ]
        positions = np.array([v.x_a[:dim] for v in verts])
        # Particle very close to first vertex
        p = Particle.sphere(x=[1e-10, 0, 0], radius=0.1, dim=dim)

        u_interp, _ = interpolate_fluid_at_particle(
            p, verts, positions, dim, n_nearest=4,
        )
        # Should be dominated by the [10,0,0] vertex
        assert u_interp[0] > 5.0

    def test_2d_interpolation(self):
        """IDW works in 2D."""
        dim = 2
        u_val = np.array([3.0, -2.0])
        verts, positions = self._make_uniform_field(dim=dim, n=8, u_val=u_val)
        p = Particle.sphere(x=[0, 0], radius=0.1, dim=dim)

        u_interp, _ = interpolate_fluid_at_particle(
            p, verts, positions, dim, n_nearest=4,
        )
        npt.assert_allclose(u_interp, u_val, atol=1e-12)


# ---------------------------------------------------------------------------
# Drag model tests
# ---------------------------------------------------------------------------

class TestDragModels:
    """Tests for Stokes and Schiller-Naumann drag."""

    def test_stokes_direction(self):
        """Stokes drag opposes relative velocity."""
        u_rel = np.array([1.0, 0, 0])
        F = _stokes_drag(u_rel, R=1e-3, mu=1e-3)
        # Drag should be in the same direction as u_rel (fluid pushing particle)
        assert F[0] > 0

    def test_stokes_magnitude(self):
        """Stokes drag magnitude: 6*pi*mu*R*|u_rel|."""
        u_rel = np.array([1.0, 0, 0])
        R, mu = 1e-3, 1e-3
        F = _stokes_drag(u_rel, R=R, mu=mu)
        expected = 6.0 * np.pi * mu * R * 1.0
        npt.assert_allclose(np.linalg.norm(F), expected, rtol=1e-12)

    def test_stokes_zero_for_zero_urel(self):
        """Zero relative velocity gives zero drag."""
        F = _stokes_drag(np.zeros(3), R=1e-3, mu=1e-3)
        npt.assert_array_equal(F, np.zeros(3))

    def test_schiller_naumann_greater_than_stokes(self):
        """Schiller-Naumann drag >= Stokes for Re > 0."""
        u_rel = np.array([0.1, 0, 0])
        R, mu, rho_f = 1e-3, 1e-3, 1000.0
        F_stokes = _stokes_drag(u_rel, R, mu)
        F_sn = _schiller_naumann_drag(u_rel, R, mu, rho_f)
        assert np.linalg.norm(F_sn) >= np.linalg.norm(F_stokes)

    def test_schiller_naumann_reduces_to_stokes_at_low_re(self):
        """At very low Re, S-N correction ≈ 1."""
        u_rel = np.array([1e-10, 0, 0])
        R, mu, rho_f = 1e-3, 1e-3, 1000.0
        F_stokes = _stokes_drag(u_rel, R, mu)
        F_sn = _schiller_naumann_drag(u_rel, R, mu, rho_f)
        npt.assert_allclose(F_sn, F_stokes, rtol=1e-3)


# ---------------------------------------------------------------------------
# Feedback momentum conservation tests
# ---------------------------------------------------------------------------

class TestFeedback:
    """Tests for particle-to-fluid reaction force distribution."""

    def test_total_reaction_conserved(self):
        """Total feedback force sums to the input reaction."""
        dim = 3
        verts = [
            MockVertex(x=[0, 0, 0]),
            MockVertex(x=[1, 0, 0]),
            MockVertex(x=[0, 1, 0]),
            MockVertex(x=[0, 0, 1]),
        ]
        positions = np.array([v.x_a[:dim] for v in verts])
        p = Particle.sphere(x=[0.3, 0.3, 0.3], radius=0.1, dim=dim)
        reaction = np.array([1.0, -2.0, 0.5])

        apply_particle_feedback(
            reaction, p, verts, positions, dim, n_nearest=4,
        )

        total = np.zeros(dim)
        for v in verts:
            if hasattr(v, "dem_feedback_force"):
                total += v.dem_feedback_force[:dim]
        npt.assert_allclose(total, reaction, atol=1e-12)


# ---------------------------------------------------------------------------
# FluidParticleCoupler tests
# ---------------------------------------------------------------------------

class TestFluidParticleCoupler:
    """Tests for the FluidParticleCoupler class."""

    def test_unknown_drag_model_raises(self):
        """Unknown drag model name raises ValueError."""
        ps = ParticleSystem(dim=3)
        HC = MockComplex([])
        with pytest.raises(ValueError, match="Unknown drag model"):
            FluidParticleCoupler(HC, ps, drag_model="invalid")

    def test_get_external_forces_fn(self):
        """External forces function applies stored drag."""
        dim = 3
        verts = [MockVertex(x=[0, 0, 0], u=[1, 0, 0], p=0)]
        HC = MockComplex(verts)
        ps = ParticleSystem(dim=dim, gravity=np.zeros(dim))
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1e-3, dim=dim))

        coupler = FluidParticleCoupler(HC, ps, dim=dim, mu=1e-3)
        coupler.fluid_to_particle(dt=1e-4)

        # External forces fn should add drag to particle
        fn = coupler.get_external_forces_fn()
        ps.reset_all_forces()
        fn(ps)
        p = ps.particles[0]
        # Should have nonzero force from drag
        assert np.linalg.norm(p.force[:dim]) > 0


# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------

class TestIO:
    """Tests for save/load/import of particle data."""

    def test_save_load_round_trip(self, tmp_path):
        """Save then load recovers identical particle data."""
        ps = ParticleSystem(dim=3, gravity=np.array([0, 0, -9.81]))
        ps.add(Particle.sphere(x=[1, 2, 3], radius=0.5, rho_s=2000.0, dim=3,
                               u=np.array([0.1, -0.2, 0.3])))
        ps.add(Particle.sphere(x=[-1, 0, 0], radius=0.3, rho_s=3000.0, dim=3,
                               wetted=True, wetting_angle=0.5))

        path = tmp_path / "test_state.json"
        save_particles(ps, t=1.5, path=path)

        ps_loaded, t_loaded = load_particles(path)

        assert t_loaded == 1.5
        assert len(ps_loaded) == 2
        assert ps_loaded.dim == 3
        npt.assert_allclose(ps_loaded.gravity, np.array([0, 0, -9.81]))

        for p_orig, p_load in zip(ps.particles, ps_loaded.particles):
            npt.assert_allclose(p_load.x_a, p_orig.x_a)
            npt.assert_allclose(p_load.u, p_orig.u)
            assert p_load.radius == p_orig.radius
            npt.assert_allclose(p_load.m, p_orig.m, rtol=1e-10)
            assert p_load.wetted == p_orig.wetted

    def test_save_creates_valid_json(self, tmp_path):
        """Output file is valid JSON with expected format."""
        ps = ParticleSystem(dim=3)
        ps.add(Particle.sphere(x=[0, 0, 0], radius=1.0, dim=3))

        path = tmp_path / "state.json"
        save_particles(ps, path=path)

        data = json.loads(path.read_text())
        assert data["format"] == "ddgclib_dem_state_v1"
        assert data["n_particles"] == 1
        assert len(data["particles"]) == 1

    def test_load_bad_format_raises(self, tmp_path):
        """Loading wrong format raises ValueError."""
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"format": "wrong"}))

        with pytest.raises(ValueError, match="Unknown DEM state format"):
            load_particles(path)

    def test_import_particle_cloud_monodisperse(self):
        """Import with scalar radius creates monodisperse system."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)
        ps = import_particle_cloud(positions, radii=0.5, rho_s=1000.0, dim=3)

        assert len(ps) == 3
        for p in ps.particles:
            assert p.radius == 0.5
            assert p.rho_s == 1000.0

    def test_import_particle_cloud_polydisperse(self):
        """Import with per-particle radii."""
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        radii = np.array([0.3, 0.7])
        ps = import_particle_cloud(positions, radii=radii, dim=3)

        assert ps.particles[0].radius == 0.3
        assert ps.particles[1].radius == 0.7

    def test_import_with_velocities(self):
        """Import with initial velocities."""
        positions = np.array([[0, 0, 0]], dtype=float)
        velocities = np.array([[1.0, 2.0, 3.0]])
        ps = import_particle_cloud(
            positions, radii=1.0, dim=3, velocities=velocities,
        )
        npt.assert_allclose(ps.particles[0].u[:3], [1, 2, 3])

    def test_import_with_wetting(self):
        """Import with wetting parameters."""
        positions = np.array([[0, 0, 0]], dtype=float)
        ps = import_particle_cloud(
            positions, radii=1.0, dim=3,
            wetted=True, wetting_angle=0.5, liquid_volume=1e-9,
        )
        p = ps.particles[0]
        assert p.wetted is True
        assert p.wetting_angle == 0.5
        assert p.liquid_volume == 1e-9

    def test_import_bad_shape_raises(self):
        """Wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="positions must have shape"):
            import_particle_cloud(np.zeros((3, 2)), dim=3)

    def test_import_bad_radii_shape_raises(self):
        """Wrong radii shape raises ValueError."""
        with pytest.raises(ValueError, match="radii must be scalar"):
            import_particle_cloud(
                np.zeros((3, 3)), radii=np.array([1, 2]), dim=3,
            )

    def test_import_2d(self):
        """Import 2D particle cloud."""
        positions = np.array([[0, 0], [1, 1]], dtype=float)
        ps = import_particle_cloud(positions, radii=0.5, dim=2)
        assert ps.dim == 2
        assert len(ps) == 2
