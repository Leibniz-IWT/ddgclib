"""Tests for ddgclib._boundary_conditions module."""

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    DirichletPressureBC,
    DirichletVelocityBC,
    FreeSlipWallBC,
    NeumannBC,
    NoSlipWallBC,
    OutletDeleteBC,
    OutletBufferedDeleteBC,
    PeriodicInletBC,
    PeriodicInletBufferedBC,
    PositionalNoSlipWallBC,
    identify_boundary_vertices,
    identify_cube_boundaries,
)


# Fixtures

@pytest.fixture
def mesh_2d():
    """2D mesh on [0, 1]^2 with fields initialized."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14):
            bV.add(v)
        v.u = np.array([1.0, 0.5])
        v.p = 100.0
        v.m = 1.0
    return HC, bV


# Boundary identification tests

class TestIdentifyBoundaryVertices:
    def test_identifies_left_wall(self, mesh_2d):
        HC, _ = mesh_2d
        left = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0]) < 1e-14)
        assert len(left) > 0
        for v in left:
            assert abs(v.x_a[0]) < 1e-14

    def test_empty_for_impossible_criterion(self, mesh_2d):
        HC, _ = mesh_2d
        result = identify_boundary_vertices(HC, lambda v: v.x_a[0] > 100)
        assert len(result) == 0


class TestIdentifyCubeBoundaries:
    def test_finds_all_boundary_verts(self, mesh_2d):
        HC, bV_expected = mesh_2d
        bV = identify_cube_boundaries(HC, 0.0, 1.0, dim=2)
        assert bV == bV_expected

    def test_no_interior_verts(self, mesh_2d):
        HC, _ = mesh_2d
        bV = identify_cube_boundaries(HC, 0.0, 1.0, dim=2)
        for v in bV:
            on_boundary = any(
                abs(v.x_a[i]) < 1e-13 or abs(v.x_a[i] - 1.0) < 1e-13
                for i in range(2)
            )
            assert on_boundary


# Concrete BC tests

class TestNoSlipWallBC:
    def test_zeros_velocity(self, mesh_2d):
        HC, bV = mesh_2d
        bc = NoSlipWallBC(dim=2)
        count = bc.apply(HC, dt=0.01, target_vertices=bV)
        assert count == len(bV)
        for v in bV:
            npt.assert_array_equal(v.u, np.zeros(2))

    def test_leaves_interior_alone(self, mesh_2d):
        HC, bV = mesh_2d
        bc = NoSlipWallBC(dim=2)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in HC.V:
            if v not in bV:
                assert v.u[0] == 1.0  # Unchanged


class TestFreeSlipWallBC:
    def test_zeroes_normal_velocity_and_keeps_tangential(self, mesh_2d):
        HC, _ = mesh_2d
        left = [v for v in HC.V if abs(v.x_a[0]) < 1e-14]
        n = FreeSlipWallBC(wall_axis=0, wall_coord=0.0).apply(
            HC, dt=0.01, target_vertices=left)
        assert n == len(left) > 0
        for v in left:
            npt.assert_array_equal(v.u, [0.0, 0.5])
        for v in HC.V:
            if v not in left:
                npt.assert_array_equal(v.u, [1.0, 0.5])

    def test_puts_a_drifted_vertex_back_on_the_wall(self, mesh_2d):
        HC, _ = mesh_2d
        left = [v for v in HC.V if abs(v.x_a[0]) < 1e-14]
        y_new = {}
        for v in left:                 # the integrator moved them
            y_new[id(v)] = v.x_a[1] + 0.01
            HC.V.move(v, (0.003, y_new[id(v)]))
        FreeSlipWallBC(wall_axis=0, wall_coord=0.0).apply(
            HC, dt=0.01, target_vertices=left)
        for v in left:
            assert v.x_a[0] == 0.0
            assert v.x_a[1] == y_new[id(v)]      # tangential motion kept
            assert v.x == (0.0, y_new[id(v)])    # cache key updated

    def test_needs_its_wall_vertices(self, mesh_2d):
        HC, _ = mesh_2d
        with pytest.raises(ValueError, match='wall vertices'):
            FreeSlipWallBC(wall_axis=0, wall_coord=0.0).apply(HC, dt=0.01)


class TestDirichletVelocityBC:
    def test_constant_value(self, mesh_2d):
        HC, bV = mesh_2d
        bc = DirichletVelocityBC(np.array([0.0, 0.1]), dim=2)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in bV:
            npt.assert_array_equal(v.u, np.array([0.0, 0.1]))

    def test_callable_value(self, mesh_2d):
        HC, bV = mesh_2d
        bc = DirichletVelocityBC(lambda v: np.array([v.x_a[1], 0.0]), dim=2)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in bV:
            npt.assert_allclose(v.u[0], v.x_a[1])
            assert v.u[1] == 0.0


class TestDirichletPressureBC:
    def test_constant_pressure(self, mesh_2d):
        HC, bV = mesh_2d
        bc = DirichletPressureBC(value=200.0)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in bV:
            assert v.p == 200.0

    def test_callable_pressure(self, mesh_2d):
        HC, bV = mesh_2d
        bc = DirichletPressureBC(value=lambda v: v.x_a[0] * 100)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in bV:
            npt.assert_allclose(v.p, v.x_a[0] * 100)


class TestNeumannBC:
    def test_zero_gradient_copies_neighbor(self, mesh_2d):
        """Zero Neumann should copy the nearest interior neighbor's value."""
        HC, bV = mesh_2d
        # Set a known field pattern
        for v in HC.V:
            v.p = v.x_a[0] * 10  # Linear in x

        bc = NeumannBC(field_name='p', flux_value=0.0)
        bc.apply(HC, dt=0.01, target_vertices=bV)

        # Boundary vertices should now have their nearest interior neighbor's P
        for v in bV:
            interior_nbs = [nb for nb in v.nn if nb not in bV]
            if interior_nbs:
                # Should be close to nearest interior neighbor value
                nb = min(interior_nbs, key=lambda nb: np.linalg.norm(v.x_a - nb.x_a))
                npt.assert_allclose(v.p, nb.p, atol=1e-10)


class TestOutletDeleteBC:
    def test_deletes_past_outlet(self):
        HC = Complex(1, domain=[(0.0, 10.0)])
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        initial_count = sum(1 for _ in HC.V)
        bc = OutletDeleteBC(outlet_pos=8.0, axis=0)
        deleted = bc.apply(HC, dt=0.01)
        final_count = sum(1 for _ in HC.V)
        assert deleted > 0
        assert final_count < initial_count
        # No vertex should be past outlet
        for v in HC.V:
            assert v.x_a[0] < 8.0


class TestOutletBufferedDeleteBC:
    """Tests for the buffered outlet BC with ghost zone."""

    def _make_1d_mesh(self, domain_end=10.0):
        """1D mesh on [0, domain_end] with velocity and fields."""
        HC = Complex(1, domain=[(0.0, domain_end)])
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        for v in HC.V:
            v.u = np.array([1.0])
            v.p = 0.0
            v.m = 1.0
        return HC

    def test_buffer_entry_detection(self):
        HC = self._make_1d_mesh(10.0)
        bV = set()
        bc = OutletBufferedDeleteBC(outlet_pos=8.0, buffer_width=3.0,
                                    axis=0, bV=bV)
        bc.apply(HC, dt=0.01)
        # Vertices past 8.0 should be in buffer
        buffer_verts = bc.buffer_vertices
        mesh_past_outlet = {v for v in HC.V if v.x_a[0] > 8.0}
        assert len(buffer_verts) == len(mesh_past_outlet)
        assert len(buffer_verts) > 0

    def test_velocity_freeze(self):
        HC = self._make_1d_mesh(10.0)
        bc = OutletBufferedDeleteBC(outlet_pos=8.0, buffer_width=3.0, axis=0)
        bc.apply(HC, dt=0.01)

        # Contaminate buffer vertex velocities (simulate integrator update)
        for v in bc.buffer_vertices:
            v.u[0] = -99.0

        # Apply again — velocities should be reset to frozen values
        bc.apply(HC, dt=0.01)
        for vid, (v, frozen_u, _) in bc._buffer.items():
            npt.assert_array_equal(v.u, frozen_u)

    def test_position_correction(self):
        HC = self._make_1d_mesh(10.0)
        bc = OutletBufferedDeleteBC(outlet_pos=8.0, buffer_width=5.0, axis=0)
        dt = 0.1

        # First apply to populate buffer
        bc.apply(HC, dt=dt)

        # Pick a buffer vertex and record its expected trajectory
        first_vid = next(iter(bc._buffer))
        buf_v, frozen_u, correct_pos = bc._buffer[first_vid]
        expected_pos = correct_pos.copy()

        # Simulate 10 steps: each step, scramble the position (as an
        # integrator would), then let the BC correct it.
        for _ in range(10):
            expected_pos[:1] += frozen_u * dt
            # Simulate integrator moving vertex to wrong position
            wrong_pos = buf_v.x_a.copy()
            wrong_pos[0] += 0.5  # arbitrary wrong displacement
            HC.V.move(buf_v, tuple(wrong_pos))
            bc.apply(HC, dt=dt)

        npt.assert_allclose(buf_v.x_a[:1], expected_pos[:1], atol=1e-12)

    def test_deletion_at_buffer_end(self):
        HC = self._make_1d_mesh(10.0)
        bc = OutletBufferedDeleteBC(outlet_pos=6.0, buffer_width=2.0, axis=0)
        # buffer_end = 8.0, so vertices at 8.0+ should be deleted
        initial = sum(1 for _ in HC.V)
        deleted = bc.apply(HC, dt=0.01)
        assert deleted > 0
        for v in HC.V:
            assert v.x_a[0] < 8.0

    def test_bV_cleanup(self):
        HC = self._make_1d_mesh(10.0)
        bV = set()
        # Add some vertices to bV that are past buffer_end
        for v in HC.V:
            if v.x_a[0] >= 8.0:
                bV.add(v)
        bc = OutletBufferedDeleteBC(outlet_pos=6.0, buffer_width=2.0,
                                    axis=0, bV=bV)
        bc.apply(HC, dt=0.01)
        # Deleted vertices must not remain in bV
        for v in bV:
            assert v.x_a[0] < 8.0

    def test_domain_vertices_untouched(self):
        HC = self._make_1d_mesh(10.0)
        bc = OutletBufferedDeleteBC(outlet_pos=8.0, buffer_width=3.0, axis=0)
        # Record domain vertex velocities before apply
        domain_vels = {id(v): v.u.copy() for v in HC.V
                       if v.x_a[0] <= 8.0}
        bc.apply(HC, dt=0.01)
        # Domain vertices should be unchanged
        for v in HC.V:
            if id(v) in domain_vels:
                npt.assert_array_equal(v.u, domain_vels[id(v)])


# BoundaryConditionSet tests

class TestBoundaryConditionSet:
    def test_applies_all_in_order(self, mesh_2d):
        HC, bV = mesh_2d
        left = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0]) < 1e-14)
        right = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0] - 1.0) < 1e-14)

        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=2), left)
        bc_set.add(DirichletVelocityBC(np.array([2.0, 0.0]), dim=2), right)

        diagnostics = bc_set.apply_all(HC, bV, dt=0.01)

        assert len(diagnostics) == 2
        for v in left:
            npt.assert_array_equal(v.u, np.zeros(2))
        for v in right:
            npt.assert_array_equal(v.u, np.array([2.0, 0.0]))

    def test_method_chaining(self, mesh_2d):
        HC, bV = mesh_2d
        bc_set = (BoundaryConditionSet()
                  .add(NoSlipWallBC(dim=2), bV)
                  .add(DirichletPressureBC(0.0), bV))
        assert len(bc_set._bcs) == 2

    def test_default_applies_to_all_bV(self, mesh_2d):
        HC, bV = mesh_2d
        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=2))  # No specific vertices

        bc_set.apply_all(HC, bV, dt=0.01)
        for v in bV:
            npt.assert_array_equal(v.u, np.zeros(2))


# Periodic inlet: coordinate-key collisions (laneL, audit F10 C2 / C3)

class TestPeriodicInletBC:
    @staticmethod
    def _unit(u_inlet=0.25):
        """Unit cell [0, 1]^2: 13 vertices, columns at x = 0, 1/4, 1/2,
        3/4, 1 with 3, 2, 3, 2, 3 vertices."""
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        HC.refine_all()
        for v in HC.V:
            v.u = np.array([u_inlet, 0.0])
            v.p = 7.0
            v.m = 1.0
        return HC

    def test_ghost_reset_keeps_every_vertex(self):
        """The reset shifts the ghost by one period, i.e. its x = 1 face
        onto the keys of its x = 0 face.  A loop of single moves lost a
        vertex there (13 -> 12)."""
        unit = self._unit()
        bc = PeriodicInletBC(unit, velocity=0.25, axis=0, inlet_pos=0.0,
                             cdist=1e-10, period=1.0)
        xs = sorted(v.x for v in bc.ghost.V)
        assert len(xs) == 13
        assert xs == sorted((v.x[0] - 1.0, v.x[1]) for v in unit.V)
        # edges survive the shift
        assert (sum(len(v.nn) for v in bc.ghost.V)
                == sum(len(v.nn) for v in unit.V))

    def test_advance_by_exactly_one_column_spacing(self):
        """velocity * dt equal to the column spacing: every ghost vertex
        moves onto the key its downstream neighbour holds."""
        mesh = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        bc = PeriodicInletBC(self._unit(), velocity=0.25, axis=0,
                             inlet_pos=0.0, cdist=1e-10, period=1.0)
        entered = []
        for _ in range(4):
            # the flow carries the columns that are already in the mesh on
            mesh.V.move_all([(v, (v.x[0] + 0.25, v.x[1])) for v in mesh.V])
            entered.append(bc.apply(mesh, dt=1.0))
        # x = 0, -1/4, -1/2, -3/4 columns enter one per step; the x = -1
        # column reaches the inlet plane and waits (strict >)
        assert entered == [3, 2, 3, 2]
        assert len(mesh.V) == 10
        assert sorted(v.x for v in bc.ghost.V) == [(0.0, 0.0), (0.0, 0.5),
                                                  (0.0, 1.0)]

    def test_injection_on_an_occupied_key_keeps_the_resident_state(self):
        """Every ghost column enters at inlet_pos + velocity * dt.  A
        frozen wall vertex left there by an earlier column must not be
        reset to the inlet velocity by the next one."""
        mesh = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        bc = PeriodicInletBC(self._unit(), velocity=0.25, axis=0,
                             inlet_pos=0.0, cdist=1e-10, period=1.0)
        dt = 0.5                       # dx = 1/8: one column every 2 steps
        assert bc.apply(mesh, dt) == 3
        wall = mesh.V[(0.125, 0.0)]    # no-slip: stays where it entered
        wall.u = np.zeros(2)
        wall.p = -1.0
        fluid = mesh.V[(0.125, 0.5)]   # advected away by the integrator
        mesh.V.move(fluid, (0.25, 0.5))
        n = len(mesh.V)
        for _ in range(4):             # x = -1/4 (2 vertices), then -1/2 (3)
            bc.apply(mesh, dt)
        assert mesh.V[(0.125, 0.0)] is wall
        npt.assert_array_equal(wall.u, np.zeros(2))
        assert wall.p == -1.0
        # the x = -1/2 column: (0.125, 0) and (0.125, 1) were occupied,
        # only (0.125, 0.5) is new
        assert len(mesh.V) == n + 2 + 1
        npt.assert_array_equal(mesh.V[(0.125, 0.5)].u, np.array([0.25, 0.0]))

    def test_mesh_advancer_step_of_one_column_spacing(self):
        """MeshAdvancer advects the whole mesh; a step of exactly one
        column spacing puts every vertex on the key of the next column."""
        from ddgclib._boundary_conditions import MeshAdvancer
        mesh = self._unit()
        inlet = PeriodicInletBC(self._unit(), velocity=0.25, axis=0,
                                inlet_pos=0.0, cdist=1e-10, period=1.0)
        adv = MeshAdvancer(mesh, inlet, OutletDeleteBC(outlet_pos=10.0, axis=0),
                           velocity=0.25)
        assert adv.step(dt=1.0) == 0
        # the 13 vertices are all still there, one spacing downstream (a
        # loop of single moves raises on the first one).  The ghost's
        # leading column enters on the keys of the mesh's former x = 0
        # column (the seam is in both), so it adds nothing.
        assert len(mesh.V) == 13
        assert sorted({v.x[0] for v in mesh.V}) == [0.25, 0.5, 0.75, 1.0, 1.25]
        assert sum(len(v.nn) for v in mesh.V) == sum(
            len(v.nn) for v in self._unit().V)


# Periodic inlet with an upstream buffer of prescribed motion (laneH)

class TestPeriodicInletBufferedBC:
    U = 0.25

    @classmethod
    def _unit(cls):
        """Unit cell [0, 1]^2: columns at x = 0, 1/4, 1/2, 3/4, 1 with
        3, 2, 3, 2, 3 vertices; walls y = 0 and y = 1."""
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        HC.refine_all()
        for v in HC.V:
            v.u = np.array([cls.U, 0.0])
            v.m = 1.0
        return HC

    @staticmethod
    def _wall(v):
        return v.x[1] == 0.0 or v.x[1] == 1.0

    @classmethod
    def _channel(cls):
        """Mesh on [-1, 2] x [0, 1] (buffer [-1, 0]), frozen walls, inlet
        BC then wall BC."""
        from ddgclib.geometry._complex_operations import extrude
        mesh = extrude(cls._unit(), 3.0, axis=0, cdist=1e-10)
        mesh.V.move_all([(v, (v.x[0] - 1.0, v.x[1])) for v in list(mesh.V)])
        for v in mesh.V:
            v.u = np.array([cls.U, 0.0])
            v.m = 1.0
        bV = {v for v in mesh.V if cls._wall(v)}
        inlet = PeriodicInletBufferedBC(
            cls._unit(), velocity=cls.U, buffer_width=1.0, axis=0,
            inlet_pos=0.0, cdist=1e-10, fields=['u', 'm'], period=1.0, bV=bV)
        walls = PositionalNoSlipWallBC(cls._wall, dim=2, bV=bV)
        inlet.apply(mesh, 0.0)
        walls.apply(mesh, 0.0)
        return mesh, bV, inlet, walls

    def test_ghost_has_no_leading_face(self):
        """The downstream face of the ghost is the periodic image of the
        upstream face of the copy before it: 13 - 3 vertices, the last
        column one spacing upstream of the injection plane."""
        _, _, inlet, _ = self._channel()
        xs = sorted({v.x[0] for v in inlet.ghost.V})
        assert len(inlet.ghost.V) == 10
        assert xs == [-2.0, -1.75, -1.5, -1.25]

    def test_the_initial_buffer_is_registered(self):
        mesh, bV, inlet, _ = self._channel()
        free_upstream = {v for v in mesh.V if v.x[0] <= 0.0 and v not in bV}
        assert inlet.buffer_vertices == free_upstream
        assert len(free_upstream) == 7      # 1 + 2 + 1 + 2 + 1 fluid vertices
        assert not inlet.buffer_vertices & bV

    def test_buffer_motion_is_prescribed(self):
        """Whatever the integrator did to a buffer vertex, the BC puts it
        on its kinematic position with the inlet velocity."""
        mesh, _, inlet, _ = self._channel()
        v = mesh.V[(-0.5, 0.5)]
        mesh.V.move(v, (-0.43, 0.61))          # a contaminated step
        v.u = np.array([-3.0, 2.0])
        inlet.apply(mesh, dt=0.1)
        npt.assert_allclose(v.x, (-0.5 + self.U * 0.1, 0.5), atol=1e-15)
        npt.assert_array_equal(v.u, np.array([self.U, 0.0]))
        assert v in inlet.buffer_vertices

    def test_release_at_the_inlet_plane(self):
        """A buffer vertex that crosses inlet_pos is released on its
        kinematic position and is left to the integrator from then on."""
        mesh, _, inlet, _ = self._channel()
        v = mesh.V[(0.0, 0.5)]
        assert v in inlet.buffer_vertices
        inlet.apply(mesh, dt=0.1)
        assert v not in inlet.buffer_vertices
        npt.assert_allclose(v.x, (self.U * 0.1, 0.5), atol=1e-15)
        mesh.V.move(v, (0.3, 0.52))
        v.u = np.array([0.4, 0.01])
        inlet.apply(mesh, dt=0.1)
        assert v.x == (0.3, 0.52)
        npt.assert_array_equal(v.u, np.array([0.4, 0.01]))

    def test_frozen_vertices_are_left_alone(self):
        mesh, bV, inlet, _ = self._channel()
        before = {v: v.x for v in bV}
        for _ in range(5):
            inlet.apply(mesh, dt=0.1)
        assert all(v.x == x for v, x in before.items())
        assert all(np.all(v.u == 0.0) for v in bV)

    def test_plug_flow_keeps_the_column_spacing(self):
        """Two periods of plug flow (released vertices carried on at the
        inlet velocity): one column per spacing, none injected twice, the
        buffer holds the same number of vertices throughout, and each
        wall gets exactly one extra vertex, one advection step from the
        upstream corner."""
        mesh, bV, inlet, walls = self._channel()
        n_walls = len(bV)
        dt = 0.5                               # dx = 1/8: half a spacing
        dx = self.U * dt
        n_buffer = []
        for _ in range(16):                    # two periods
            free = [v for v in mesh.V if v not in bV
                    and v not in inlet.buffer_vertices]
            mesh.V.move_all([(v, (v.x[0] + dx, v.x[1])) for v in free])
            inlet.apply(mesh, dt)
            walls.apply(mesh, dt)
            n_buffer.append(len(inlet.buffer_vertices))
        columns = sorted({round(v.x[0], 9) for v in mesh.V if v not in bV})
        npt.assert_allclose(np.diff(columns), 0.25, atol=1e-9)
        assert columns[0] <= -1.0 + 0.25 and columns[-1] == 4.0
        # 7 fluid vertices per period plus the column on the release plane
        assert set(n_buffer) <= {6, 7, 8}
        extra = sorted(v.x for v in bV if v.x[0] == -1.0 + dx)
        assert extra == [(-1.0 + dx, 0.0), (-1.0 + dx, 1.0)]
        assert len(bV) == n_walls + 2

    def test_one_extra_wall_vertex_whatever_the_step(self):
        """``U dt`` does not divide the column spacing (0.0925 against
        0.25).  The wall rows are injected once and then dropped from the
        ghost, so each wall still gets exactly one extra vertex, and the
        fluid columns keep their spacing.  Before the fix round of laneH
        every injected wall-row column left one more frozen vertex within
        one advection step of the upstream corner (two per period and
        wall here), without bound."""
        mesh, bV, inlet, walls = self._channel()
        n_walls = len(bV)
        dt = 0.37
        dx = self.U * dt
        for _ in range(44):                    # four periods
            free = [v for v in mesh.V if v not in bV
                    and v not in inlet.buffer_vertices]
            mesh.V.move_all([(v, (v.x[0] + dx, v.x[1])) for v in free])
            inlet.apply(mesh, dt)
            walls.apply(mesh, dt)
        assert len(bV) == n_walls + 2
        extra = sorted(v.x for v in bV if -1.0 < v.x[0] <= -1.0 + dx)
        assert [x[1] for x in extra] == [0.0, 1.0]
        assert not any(self._wall(gv) for gv in inlet.ghost.V)
        columns = sorted({round(v.x[0], 9) for v in mesh.V if v not in bV})
        npt.assert_allclose(np.diff(columns), 0.25, atol=1e-9)
