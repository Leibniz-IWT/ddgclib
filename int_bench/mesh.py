"""
Minimal n-dimensional simplicial mesh.

This is a deliberately light wrapper around ``scipy.spatial.Delaunay``
for n >= 2 (with a hand-rolled construction for n = 1) that exposes
exactly the connectivity needed by the integrated-operator pipeline:

* ``vertices``   — (N, n) array of vertex positions.
* ``simplices``  — (S, n+1) array of simplex vertex indices.
* ``neighbors``  — list of vertex 1-rings (sets of indices).
* ``vertex_simplices`` — for each vertex, the simplices that contain it.
* ``boundary``   — boolean mask of boundary vertices.

A regular grid factory (``regular_grid``) produces uniform meshes for
1D / 2D / 3D used by the validation tests. A jitter helper perturbs
interior vertices and re-Delaunay-triangulates so non-symmetric
meshes can be tested too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SimplexMesh:
    """Simplicial complex with vertex-centric connectivity."""

    vertices: np.ndarray                       # (N, dim)
    simplices: np.ndarray                      # (S, dim+1)
    domain: list[tuple[float, float]]          # bounding box
    neighbors: list[set[int]] = field(default_factory=list)
    vertex_simplices: list[list[int]] = field(default_factory=list)
    boundary: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def dim(self) -> int:
        return self.vertices.shape[1]

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_simplices(self) -> int:
        return self.simplices.shape[0]

    @property
    def interior_vertices(self) -> np.ndarray:
        return np.where(~self.boundary)[0]

    @property
    def boundary_vertices(self) -> np.ndarray:
        return np.where(self.boundary)[0]

    # --- factories ---------------------------------------------------------

    @classmethod
    def from_points(
        cls,
        vertices: np.ndarray,
        simplices: np.ndarray,
        domain: list[tuple[float, float]] | None = None,
    ) -> "SimplexMesh":
        verts = np.asarray(vertices, dtype=float)
        cells = np.asarray(simplices, dtype=int)
        dim = verts.shape[1]
        if domain is None:
            domain = [(float(verts[:, d].min()), float(verts[:, d].max()))
                      for d in range(dim)]
        m = cls(vertices=verts, simplices=cells, domain=list(domain))
        m._build_connectivity()
        m._tag_boundary()
        return m

    @classmethod
    def regular_grid(
        cls,
        dim: int,
        n: int = 5,
        domain: list[tuple[float, float]] | None = None,
    ) -> "SimplexMesh":
        """Uniform Cartesian grid simplicially decomposed via Kuhn cubes.

        * 1D: ``n`` equispaced points on [0, 1].
        * 2D / 3D: an ``n x n`` (x ``n``) Cartesian grid, with each
          unit cube decomposed into ``d!`` Kuhn simplices (2 triangles
          per square, 6 tets per cube). This avoids the degenerate
          cospherical Delaunay triangulation that ``scipy.spatial``
          produces on regular grids and yields a globally conforming
          simplicial complex.

        Parameters
        ----------
        dim : int
        n : int
            Number of vertices per axis.
        domain : list of tuple, optional
            Per-axis bounds. Defaults to unit hypercube.
        """
        if domain is None:
            domain = [(0.0, 1.0)] * dim

        axes = [np.linspace(lo, hi, n) for (lo, hi) in domain]
        if dim == 1:
            verts = axes[0].reshape(-1, 1)
            simplices = np.column_stack([
                np.arange(n - 1), np.arange(1, n)
            ])
            return cls.from_points(verts, simplices, domain)

        # Lattice vertices indexed by (i_0, ..., i_{d-1}).
        grids = np.meshgrid(*axes, indexing="ij")
        verts = np.stack([g.ravel() for g in grids], axis=-1)

        from itertools import permutations, product
        n_per_axis = [n] * dim
        strides = np.array(
            [np.prod(n_per_axis[a + 1:], dtype=int) for a in range(dim)],
            dtype=int,
        )

        def lattice_index(coord: tuple[int, ...]) -> int:
            return int(sum(c * s for c, s in zip(coord, strides)))

        simplices = []
        # Each unit cube has its lower corner at (i_0, ..., i_{d-1}); the
        # 2^d vertices are addressed by binary masks over the d axes.
        cube_axes = list(range(dim))
        cube_corners = list(product([0, 1], repeat=dim))
        for cube_origin in product(*[range(n - 1) for _ in range(dim)]):
            origin = np.array(cube_origin, dtype=int)
            corner_idx = {
                mask: lattice_index(tuple(origin + np.array(mask)))
                for mask in cube_corners
            }
            # d! Kuhn simplices per cube (paths from 0...0 to 1...1).
            for perm in permutations(cube_axes):
                bits = [0] * dim
                path = [tuple(bits)]
                for axis in perm:
                    bits[axis] = 1
                    path.append(tuple(bits))
                simplices.append([corner_idx[c] for c in path])

        simplices = np.array(simplices, dtype=int)
        return cls.from_points(verts, simplices, domain)

    # --- mutation ----------------------------------------------------------

    def jittered(
        self,
        seed: int = 42,
        amplitude: float = 0.1,
    ) -> "SimplexMesh":
        """Return a copy with interior vertices perturbed.

        Boundary vertices stay fixed so the domain shape is preserved.
        The mesh is re-Delaunay-triangulated after jittering.
        """
        rng = np.random.default_rng(seed)
        verts = self.vertices.copy()

        # Estimate a per-vertex edge scale from the current 1-rings.
        edge_lens = np.full(self.n_vertices, 0.1)
        for i in range(self.n_vertices):
            if not self.neighbors[i]:
                continue
            edge_lens[i] = min(
                np.linalg.norm(self.vertices[i] - self.vertices[j])
                for j in self.neighbors[i]
            )

        for i in range(self.n_vertices):
            if self.boundary[i]:
                continue
            offset = rng.uniform(
                -amplitude * edge_lens[i],
                +amplitude * edge_lens[i],
                size=self.dim,
            )
            verts[i] += offset

        if self.dim == 1:
            order = np.argsort(verts[:, 0])
            verts = verts[order]
            simplices = np.column_stack([
                np.arange(len(verts) - 1), np.arange(1, len(verts))
            ])
        else:
            from scipy.spatial import Delaunay
            tri = Delaunay(verts)
            simplices = tri.simplices

        return SimplexMesh.from_points(verts, simplices, self.domain)

    # --- connectivity ------------------------------------------------------

    def _build_connectivity(self) -> None:
        N = self.n_vertices
        self.neighbors = [set() for _ in range(N)]
        self.vertex_simplices = [[] for _ in range(N)]

        for s_idx, simplex in enumerate(self.simplices):
            for vi in simplex:
                self.vertex_simplices[vi].append(s_idx)
                for vj in simplex:
                    if vi != vj:
                        self.neighbors[vi].add(int(vj))

    def _tag_boundary(self, tol: float = 1e-12) -> None:
        N = self.n_vertices
        bd = np.zeros(N, dtype=bool)
        for i in range(N):
            for d in range(self.dim):
                lo, hi = self.domain[d]
                if (
                    abs(self.vertices[i, d] - lo) < tol
                    or abs(self.vertices[i, d] - hi) < tol
                ):
                    bd[i] = True
                    break
        self.boundary = bd

    # --- field sampling ----------------------------------------------------

    def sample_scalar(self, f: Callable[[np.ndarray], float]) -> np.ndarray:
        """Sample a scalar field at all vertex positions."""
        return np.array([f(x) for x in self.vertices])

    def sample_vector(
        self,
        u: Callable[[np.ndarray], np.ndarray],
        m: int | None = None,
    ) -> np.ndarray:
        """Sample a vector field of dim ``m`` at all vertex positions."""
        sample = u(self.vertices[0])
        m = m if m is not None else len(sample)
        out = np.empty((self.n_vertices, m))
        out[0] = sample
        for i in range(1, self.n_vertices):
            out[i] = u(self.vertices[i])
        return out

    # --- queries -----------------------------------------------------------

    def primal_star(self, v_idx: int) -> list[int]:
        """Indices of simplices containing vertex ``v_idx``."""
        return list(self.vertex_simplices[v_idx])

    def primal_star_volume(self, v_idx: int) -> float:
        """Total n-volume of all simplices in the star of ``v_idx``."""
        from .geometry import simplex_volume
        return float(sum(
            simplex_volume(self.vertices[self.simplices[s]])
            for s in self.primal_star(v_idx)
        ))

    def dual_volume_barycentric(self, v_idx: int) -> float:
        """Barycentric dual cell volume = (1/(n+1)) * star volume.

        For a barycentric subdivision, exactly one of the (n+1) pieces
        of each simplex containing ``v_idx`` belongs to its dual cell,
        so the volume is the simplex volume divided by ``n+1``.
        """
        return self.primal_star_volume(v_idx) / (self.dim + 1)


__all__ = ["SimplexMesh"]
