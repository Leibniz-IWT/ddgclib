"""
Integrated gradient and Hessian operators on a simplicial mesh.

Two control volumes are supported per interior vertex ``v_i``:

* ``"primal"``: the closed star of ``v_i`` (union of n-simplices
  containing ``v_i``). Its (n-1)-boundary consists of exactly one face
  per simplex (the face opposite ``v_i``).

* ``"dual"``: the barycentric dual cell of ``v_i``. Its boundary is the
  union over 1-ring neighbors ``j`` of the dual face ``A_ij``, where
  ``A_ij`` is the sum of per-simplex contributions from
  :func:`int_bench.geometry.barycentric_dual_face_vector`.

The discrete operators are evaluated as boundary integrals via the
divergence theorem::

    ∫_V ∇f dV = ∮_∂V f n dA
    ∫_V ∇⊗∇f dV = ∮_∂V (∇f) ⊗ n dA = ∇( ∫_V ∇f dV ) recursively

For the gradient, ``f`` on a boundary face is interpolated linearly
from the vertex values of the face. For the Hessian we run a global
gradient pass first (cell-averaged ``g_i = G_i / V_i``) and then apply
the gradient operator a second time to each component of ``g``.

These operators are exact (machine precision) for affine fields on any
control volume, and convergent for higher-order fields. The dual form
is the classical DDG identity ``G_i = 0.5 Σ_j (f_j - f_i) A_ij``; the
primal form is the analogous boundary sum on the closed star.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .geometry import (
    area_vector,
    barycentric_dual_face_vector,
    simplex_volume,
)
from .mesh import SimplexMesh


ControlVolume = Literal["primal", "dual"]


# ---------------------------------------------------------------------------
# Boundary face enumeration
# ---------------------------------------------------------------------------

def primal_star_boundary(
    mesh: SimplexMesh,
    v_idx: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Boundary faces of the closed star of ``v_idx``.

    For each n-simplex T containing ``v_idx``, the unique (n-1)-face
    of T not containing ``v_idx`` lies on the boundary of the star.

    Returns
    -------
    list of (face_vertex_indices, area_vector_outward)
        ``face_vertex_indices`` is an array of n vertex indices.
        ``area_vector_outward`` points away from ``v_idx``.
    """
    out = []
    n = mesh.dim
    for s_idx in mesh.primal_star(v_idx):
        simplex = mesh.simplices[s_idx]
        face_idx = np.array([v for v in simplex if v != v_idx])
        face_pts = mesh.vertices[face_idx]
        # Orient: ``area_vector`` returns ±n_face_vol; choose the sign
        # that points away from v_idx.
        A = area_vector(face_pts)
        v_pos = mesh.vertices[v_idx]
        # Vector from any face vertex to v_idx; outward from v means
        # opposite to this direction.
        face_centroid = face_pts.mean(axis=0)
        outward_dir = face_centroid - v_pos
        if np.dot(A, outward_dir) < 0:
            A = -A
        out.append((face_idx, A))
    return out


def dual_cell_boundary(
    mesh: SimplexMesh,
    v_idx: int,
) -> dict[int, np.ndarray]:
    """Dual cell boundary: per-neighbor area vector ``A_ij``.

    Returns
    -------
    dict[int, ndarray]
        ``j -> A_ij`` where ``A_ij`` points from ``v_idx`` to ``j``
        and is the sum of per-simplex barycentric dual face
        contributions.
    """
    A_ij: dict[int, np.ndarray] = {
        j: np.zeros(mesh.dim) for j in mesh.neighbors[v_idx]
    }
    for s_idx in mesh.primal_star(v_idx):
        simplex = mesh.simplices[s_idx]
        # Local index of v_idx in this simplex
        i_local = int(np.where(simplex == v_idx)[0][0])
        for j_local, j_global in enumerate(simplex):
            if j_local == i_local:
                continue
            j_global = int(j_global)
            simplex_pts = mesh.vertices[simplex]
            A_ij[j_global] += barycentric_dual_face_vector(
                simplex_pts, i_local, j_local,
            )
    return A_ij


# ---------------------------------------------------------------------------
# Integrated gradient — scalar field
# ---------------------------------------------------------------------------

def integrated_gradient(
    mesh: SimplexMesh,
    f_vals: np.ndarray,
    v_idx: int,
    control: ControlVolume = "dual",
) -> np.ndarray:
    """Integrated gradient ``∫_V ∇f dV`` at vertex ``v_idx``.

    Parameters
    ----------
    mesh : SimplexMesh
    f_vals : ndarray, shape (n_vertices,)
        Vertex-sampled scalar field.
    v_idx : int
        Interior vertex index.
    control : {"primal", "dual"}
        Choice of control volume.

    Returns
    -------
    ndarray, shape (dim,)
        Integrated gradient.
    """
    n = mesh.dim
    G = np.zeros(n)

    if control == "dual":
        # DDG identity: 0.5 Σ_j (f_j - f_i) A_ij
        f_i = f_vals[v_idx]
        for j, A_ij in dual_cell_boundary(mesh, v_idx).items():
            G += 0.5 * (f_vals[j] - f_i) * A_ij
        return G

    if control == "primal":
        # Closed star: ∮_∂star f n dA, with f on each face replaced by
        # the centroid value (= mean of vertex values for an n-simplex
        # face — exact for linear f).
        for face_idx, A in primal_star_boundary(mesh, v_idx):
            f_face = f_vals[face_idx].mean()
            G += f_face * A
        return G

    raise ValueError(f"unknown control volume: {control!r}")


# ---------------------------------------------------------------------------
# Integrated gradient — vector field (gradient tensor)
# ---------------------------------------------------------------------------

def integrated_gradient_tensor(
    mesh: SimplexMesh,
    u_vals: np.ndarray,
    v_idx: int,
    control: ControlVolume = "dual",
) -> np.ndarray:
    """Integrated gradient tensor ``∫_V ∇u dV`` for a vector field u.

    Parameters
    ----------
    u_vals : ndarray, shape (n_vertices, m)
        Vertex-sampled vector field (m components).

    Returns
    -------
    ndarray, shape (m, dim)
        Integrated gradient tensor; row ``a`` is ``∫ ∇u_a dV``.
    """
    m = u_vals.shape[1]
    G = np.zeros((m, mesh.dim))

    if control == "dual":
        u_i = u_vals[v_idx]
        for j, A_ij in dual_cell_boundary(mesh, v_idx).items():
            G += 0.5 * np.outer(u_vals[j] - u_i, A_ij)
        return G

    if control == "primal":
        for face_idx, A in primal_star_boundary(mesh, v_idx):
            u_face = u_vals[face_idx].mean(axis=0)
            G += np.outer(u_face, A)
        return G

    raise ValueError(f"unknown control volume: {control!r}")


# ---------------------------------------------------------------------------
# Integrated Hessian — scalar field
# ---------------------------------------------------------------------------

def vertex_gradient_field(
    mesh: SimplexMesh,
    f_vals: np.ndarray,
    control: ControlVolume = "dual",
    boundary_strategy: Literal["nan", "zero", "skip"] = "zero",
) -> np.ndarray:
    """Cell-averaged gradient at every vertex.

    For each interior vertex ``i``:

        g_i = ∫_{V_i} ∇f dV / vol(V_i)

    Boundary vertices use the chosen strategy (default: zero) since
    the integrated operator is not well-defined there without ghost
    cells. The Hessian step needs ``g_j`` for every neighbor ``j`` of
    an interior vertex; if ``j`` is on the boundary, those values
    are taken from this same cell-averaged construction (still
    convergent in the bulk).

    Parameters
    ----------
    boundary_strategy : str
        How to handle boundary vertices. ``"zero"`` is a reasonable
        default for smooth fields with the boundary of the mesh
        chosen well outside the region of interest.
    """
    n_v = mesh.n_vertices
    g = np.zeros((n_v, mesh.dim))
    for i in range(n_v):
        if mesh.boundary[i]:
            if boundary_strategy == "nan":
                g[i] = np.nan
            elif boundary_strategy == "zero":
                g[i] = 0.0
            else:  # skip — leave as zero
                g[i] = 0.0
            continue
        G = integrated_gradient(mesh, f_vals, i, control=control)
        if control == "dual":
            V = mesh.dual_volume_barycentric(i)
        else:
            V = mesh.primal_star_volume(i)
        g[i] = G / V if V > 1e-30 else 0.0
    return g


def integrated_hessian(
    mesh: SimplexMesh,
    f_vals: np.ndarray,
    v_idx: int,
    control: ControlVolume = "dual",
    boundary_strategy: Literal["nan", "zero", "skip"] = "zero",
) -> np.ndarray:
    """Integrated Hessian ``∫_V ∇⊗∇f dV`` at vertex ``v_idx``.

    Implemented as the integrated gradient of the cell-averaged
    gradient field (a "vector field" from the operator's point of
    view). This converges to the true Hessian integral for smooth
    fields and is exact for affine ``∇f`` (i.e. quadratic ``f``)
    when both passes use the same control volume on a uniform mesh.

    Boundary vertices are handled via ``boundary_strategy`` since
    ``v_idx`` is required to be interior but its 1-ring may include
    boundary vertices.

    Returns
    -------
    ndarray, shape (dim, dim)
        Integrated Hessian; row ``a`` is ``∫ ∇(∂_a f) dV``.
    """
    g = vertex_gradient_field(
        mesh, f_vals, control=control, boundary_strategy=boundary_strategy
    )
    return integrated_gradient_tensor(mesh, g, v_idx, control=control)


# ---------------------------------------------------------------------------
# Volume helpers (exposed for benchmark callers)
# ---------------------------------------------------------------------------

def control_volume(
    mesh: SimplexMesh,
    v_idx: int,
    control: ControlVolume = "dual",
) -> float:
    """n-volume of the chosen control volume around ``v_idx``."""
    if control == "dual":
        return mesh.dual_volume_barycentric(v_idx)
    if control == "primal":
        return mesh.primal_star_volume(v_idx)
    raise ValueError(f"unknown control volume: {control!r}")


__all__ = [
    "ControlVolume",
    "control_volume",
    "primal_star_boundary",
    "dual_cell_boundary",
    "integrated_gradient",
    "integrated_gradient_tensor",
    "integrated_hessian",
    "vertex_gradient_field",
]
