"""Unit tests for the primal-subcomplex interface model.

Covers:
- :func:`ddgclib.geometry.extract_interface` (populates HC attributes)
- :func:`ddgclib.geometry.validate_closure`
- :func:`ddgclib.geometry.curve_neighbours`
- :meth:`MultiphaseSystem.assign_simplex_phases`
- :meth:`MultiphaseSystem.assign_vertex_phases_from_simplices`
- :meth:`MultiphaseSystem.identify_interface_from_subcomplex`
"""
import unittest

import numpy as np

from hyperct import Complex

from ddgclib.multiphase import (
    INTERFACE_PHASE,
    MultiphaseSystem,
    PhaseProperties,
    _simplex_key,
    iter_top_simplices,
)
from ddgclib.eos import TaitMurnaghan
from ddgclib.geometry import (
    connect_and_cache_simplices,
    extract_interface,
    validate_closure,
    curve_neighbours,
)


def _make_mps(n_phases: int = 2) -> MultiphaseSystem:
    phases = [
        PhaseProperties(eos=TaitMurnaghan(rho0=1000.0), mu=0.1,
                        rho0=1000.0, name=f"ph{i}")
        for i in range(n_phases)
    ]
    return MultiphaseSystem(phases=phases, gamma={(0, 1): 0.05})


# ---------------------------------------------------------------------------
# Hand-built 2D fixtures
# ---------------------------------------------------------------------------


def _square_2d_mesh():
    """4 vertices, 2 triangles sharing diagonal (0,0)-(1,1)."""
    HC = Complex(2, domain=[(-0.1, 1.1), (-0.1, 1.1)])
    v00 = HC.V[(0.0, 0.0)]
    v10 = HC.V[(1.0, 0.0)]
    v11 = HC.V[(1.0, 1.0)]
    v01 = HC.V[(0.0, 1.0)]
    verts = [v00, v10, v11, v01]
    simplices = [[0, 2, 3], [0, 1, 2]]
    connect_and_cache_simplices(HC, verts, 2, simplices=simplices)
    return HC, (v00, v10, v11, v01)


class TestExtract2DMinimal(unittest.TestCase):
    def test_two_triangles_one_interface_edge(self):
        HC, (v00, v10, v11, v01) = _square_2d_mesh()
        mps = _make_mps()
        mps.assign_simplex_phases(
            HC, 2, criterion_fn=lambda c: 0 if c[0] < 0.5 else 1,
        )
        extract_interface(HC, mps.simplex_phase, 2)

        self.assertEqual(len(HC.interface_edges), 1)
        self.assertIn(frozenset({v00.x, v11.x}), HC.interface_edges)
        self.assertEqual(HC.interface_vertices, {v00, v11})

        # Not closed (1 edge) — should fail validation.
        with self.assertRaises(ValueError):
            validate_closure(HC, 2)

    def test_uniform_phase_no_interface(self):
        HC, _ = _square_2d_mesh()
        mps = _make_mps()
        mps.assign_simplex_phases(HC, 2, criterion_fn=lambda c: 1)
        extract_interface(HC, mps.simplex_phase, 2)
        self.assertEqual(len(HC.interface_edges), 0)
        self.assertEqual(len(HC.interface_vertices), 0)


def _grid_2d_mesh(nx: int = 4, ny: int = 4):
    """(nx+1) x (ny+1) grid triangulated diagonally."""
    HC = Complex(2, domain=[(-0.5, nx + 0.5), (-0.5, ny + 0.5)])
    grid = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            v = HC.V[(float(i), float(j))]
            grid[(i, j)] = v
    verts = [grid[(i, j)] for j in range(ny + 1) for i in range(nx + 1)]

    def idx(i, j):
        return j * (nx + 1) + i

    simplices = []
    for j in range(ny):
        for i in range(nx):
            simplices.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            simplices.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    connect_and_cache_simplices(HC, verts, 2, simplices=simplices)
    return HC, grid


class TestExtract2DGrid(unittest.TestCase):
    def test_closed_rectangular_interface(self):
        HC, grid = _grid_2d_mesh(nx=4, ny=4)
        mps = _make_mps()
        def crit(c):
            return 1 if (1.0 < c[0] < 3.0 and 1.0 < c[1] < 3.0) else 0
        mps.assign_simplex_phases(HC, 2, criterion_fn=crit)
        extract_interface(HC, mps.simplex_phase, 2)

        self.assertGreater(len(HC.interface_edges), 0)
        validate_closure(HC, 2)

        # Every interface vertex has exactly 2 interface-edge neighbours.
        for v in HC.interface_vertices:
            degree = sum(
                1 for nb in v.nn
                if nb in HC.interface_vertices
                and frozenset({v.x, nb.x}) in HC.interface_edges
            )
            self.assertEqual(degree, 2)

    def test_curve_neighbours_round_trip(self):
        HC, grid = _grid_2d_mesh(nx=4, ny=4)
        mps = _make_mps()
        def crit(c):
            return 1 if (1.0 < c[0] < 3.0 and 1.0 < c[1] < 3.0) else 0
        mps.assign_simplex_phases(HC, 2, criterion_fn=crit)
        extract_interface(HC, mps.simplex_phase, 2)

        # Walk the polyline from an arbitrary start.
        start = next(iter(HC.interface_vertices))
        walked = [start]
        prev = None
        curr = start
        for _ in range(len(HC.interface_vertices) + 1):
            a, b = curve_neighbours(curr, HC)
            nxt = b if a is prev else a
            if nxt is None or nxt is start:
                break
            walked.append(nxt)
            prev, curr = curr, nxt
        self.assertEqual(set(walked), HC.interface_vertices)


class TestVertexPhaseDerivation(unittest.TestCase):
    def test_bulk_and_interface_tagging(self):
        HC, grid = _grid_2d_mesh(nx=4, ny=4)
        mps = _make_mps()
        def crit(c):
            return 1 if (1.0 < c[0] < 3.0 and 1.0 < c[1] < 3.0) else 0
        mps.assign_simplex_phases(HC, 2, criterion_fn=crit)
        mps.assign_vertex_phases_from_simplices(HC, 2)

        n_iface = sum(1 for v in HC.V if v.phase == INTERFACE_PHASE)
        n_bulk0 = sum(1 for v in HC.V if v.phase == 0)
        n_bulk1 = sum(1 for v in HC.V if v.phase == 1)
        self.assertGreater(n_iface, 0)
        self.assertGreater(n_bulk0, 0)
        self.assertGreater(n_bulk1, 0)

        for v in HC.V:
            self.assertTrue(hasattr(v, 'interface_phases'))
            self.assertGreaterEqual(len(v.interface_phases), 1)


class TestIdentifyInterfaceFromSubcomplex(unittest.TestCase):
    def test_end_to_end_tags(self):
        HC, grid = _grid_2d_mesh(nx=4, ny=4)
        mps = _make_mps()
        def crit(c):
            return 1 if (1.0 < c[0] < 3.0 and 1.0 < c[1] < 3.0) else 0
        mps.assign_simplex_phases(HC, 2, criterion_fn=crit)
        interface = mps.identify_interface_from_subcomplex(HC, 2)

        self.assertTrue(hasattr(HC, 'interface_vertices'))
        self.assertGreater(len(interface), 0)

        for v in HC.V:
            if v in interface:
                self.assertTrue(v.is_interface)
                self.assertEqual(v.phase, INTERFACE_PHASE)
            else:
                self.assertFalse(v.is_interface)
                self.assertIn(v.phase, (0, 1))

    def test_requires_simplex_phase(self):
        HC, _ = _grid_2d_mesh(nx=2, ny=2)
        mps = _make_mps()
        with self.assertRaises(ValueError):
            mps.identify_interface_from_subcomplex(HC, 2)


# ---------------------------------------------------------------------------
# Hand-built 3D fixtures
# ---------------------------------------------------------------------------


def _two_tets_sharing_face():
    """Two tets sharing triangle (v1, v2, v3)."""
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


class TestExtract3DMinimal(unittest.TestCase):
    def test_two_tets_one_interface_triangle(self):
        HC, verts = _two_tets_sharing_face()
        v0, v1, v2, v3, v4 = verts
        mps = _make_mps()
        mps.assign_simplex_phases(
            HC, 3, criterion_fn=lambda c: 0 if sum(c) < 1.0 else 1,
        )
        extract_interface(HC, mps.simplex_phase, 3)

        self.assertEqual(len(HC.interface_triangles), 1)
        self.assertEqual(HC.interface_vertices, {v1, v2, v3})
        # Not closed — single triangle.
        with self.assertRaises(ValueError):
            validate_closure(HC, 3)


def _octahedron_envelope_3d():
    """Inner tet (phase 1) surrounded by 4 outer tets (phase 0)."""
    a = 1.0
    inner = [(0.0, 0.0, 0.0), (a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)]

    def reflect(opposite_idx):
        face = [np.array(inner[k]) for k in range(4) if k != opposite_idx]
        centroid = np.mean(face, axis=0)
        opp = np.array(inner[opposite_idx])
        return tuple(2 * centroid - opp)

    all_pos = list(inner) + [reflect(i) for i in range(4)]
    mins = np.min(np.array(all_pos), axis=0) - 0.1
    maxs = np.max(np.array(all_pos), axis=0) + 0.1
    HC = Complex(3, domain=[(float(mins[i]), float(maxs[i])) for i in range(3)])
    verts = [HC.V[pos] for pos in all_pos]

    simplices = [
        [0, 1, 2, 3],
        [4, 1, 2, 3],
        [5, 0, 2, 3],
        [6, 0, 1, 3],
        [7, 0, 1, 2],
    ]
    connect_and_cache_simplices(HC, verts, 3, simplices=simplices)
    return HC, verts


class TestExtract3DClosed(unittest.TestCase):
    def test_closed_tetrahedral_surface(self):
        HC, verts = _octahedron_envelope_3d()
        v0, v1, v2, v3 = verts[:4]
        mps = _make_mps()
        def crit(c):
            return 1 if np.linalg.norm(c) < 0.5 else 0
        mps.assign_simplex_phases(HC, 3, criterion_fn=crit)
        extract_interface(HC, mps.simplex_phase, 3)

        self.assertEqual(len(HC.interface_triangles), 4)
        validate_closure(HC, 3)
        self.assertEqual(HC.interface_vertices, {v0, v1, v2, v3})


# ---------------------------------------------------------------------------
# Closure-failure diagnostics
# ---------------------------------------------------------------------------


class TestClosureFailures(unittest.TestCase):
    def test_2d_dangling_edge_message(self):
        """Fabricate HC with a single dangling interface edge."""

        class _V:
            def __init__(self, x):
                self.x = tuple(x)
                self.x_a = np.array(x, dtype=float)
                self.nn = set()
                self.boundary = False

        v0 = _V((0.0, 0.0))
        v1 = _V((1.0, 0.0))
        v0.nn = {v1}
        v1.nn = {v0}

        # Mock HC with interface sets
        class _HC:
            interface_vertices = {v0, v1}
            interface_edges = {frozenset({v0.x, v1.x})}

        with self.assertRaises(ValueError) as cm:
            validate_closure(_HC, 2)
        self.assertIn("degree 1", str(cm.exception))

    def test_3d_open_surface_message(self):
        """Single triangle — 3 edges each shared by only 1 triangle."""

        class _V:
            def __init__(self, x):
                self.x = tuple(x)
                self.x_a = np.array(x, dtype=float)

        v0 = _V((0.0, 0.0, 0.0))
        v1 = _V((1.0, 0.0, 0.0))
        v2 = _V((0.0, 1.0, 0.0))

        class _HC:
            interface_vertices = {v0, v1, v2}
            interface_edges = set()
            interface_triangles = {frozenset({v0.x, v1.x, v2.x})}

        with self.assertRaises(ValueError) as cm:
            validate_closure(_HC, 3)
        self.assertIn("expected 2", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
