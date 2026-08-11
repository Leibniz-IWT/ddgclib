"""Mesh-dynamic operators for ddgclib tetra/free-surface calculations.

This module is the reusable mesh side of the PR33/PR35/PR37 workflow.  It
keeps the state as vertices and tetrahedra, computes force-driven vertex
velocities, applies a sparse tetra-volume pressure projection, and supplies
the geometric constraints needed by sphere/film contact-line cases.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

try:  # Optional but substantially improves large tetra solves.
    import pyamg
except ImportError:  # pragma: no cover - exercised by minimal installations.
    pyamg = None


def _krylov_preconditioner(matrix: sparse.csr_matrix) -> spla.LinearOperator:
    """Return an AMG preconditioner, with a diagonal fallback."""

    if pyamg is not None and matrix.shape[0] >= 500:
        hierarchy = pyamg.smoothed_aggregation_solver(
            matrix,
            symmetry="symmetric",
            max_coarse=200,
        )
        return hierarchy.aspreconditioner(cycle="V")
    diagonal = matrix.diagonal()
    diagonal = np.where(np.abs(diagonal) > 1.0e-300, diagonal, 1.0)
    return spla.LinearOperator(matrix.shape, matvec=lambda x: x / diagonal)


@dataclass(frozen=True)
class PressureProjectionResult:
    """Diagnostics returned by the sparse tetra pressure projection."""

    pressure_pa: np.ndarray
    residual_before_m3_s: float
    residual_after_m3_s: float
    pressure_l2_pa: float
    pressure_linf_pa: float
    solver_info: int
    mini_bubble_velocity_m_s: np.ndarray | None = None
    p2_edges: np.ndarray | None = None
    p2_edge_velocity_m_s: np.ndarray | None = None
    # PR33 pressure contribution to the vertex momentum equation.  This is
    # stored explicitly so callers do not hide it inside a recovered PR35
    # Cauchy residual.
    pressure_force_n: np.ndarray | None = None
    # Optional force-space reaction for a velocity upper-bound active set.
    # This is deliberately separate from ``pressure_force_n=B.T*p``.
    constraint_force_n: np.ndarray | None = None
    constraint_multiplier_n: np.ndarray | None = None
    upper_velocity_constraint_active: bool = False
    unconstrained_upper_velocity_value_m_s: np.ndarray | None = None
    upper_velocity_constraint_residual_m_s: np.ndarray | None = None


def tetra_mini_bubble_velocity(
    points: np.ndarray,
    tets: np.ndarray,
    pressure_pa: np.ndarray,
    *,
    viscosity_pa_s: float,
    density_kg_m3: float,
    dt_s: float,
    viscous_form: str = "vector_laplacian",
) -> np.ndarray:
    """Reconstruct the condensed MINI velocity bubble in every tetrahedron.

    The pressure stabilization used by the MINI element is obtained by
    eliminating the element-interior velocity ``b*u_b``.  The same ``u_b``
    must be restored when evaluating volume fluxes; otherwise the algebraic
    pressure solve is stable but the physical velocity field is missing its
    through-element mode.  The normalized bubble is
    ``b=256*lambda0*lambda1*lambda2*lambda3``.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    pressure = np.asarray(pressure_pa, dtype=float)
    if pressure.shape != (len(pts),):
        raise ValueError("pressure_pa must contain one P1 value per vertex")
    valid, volumes, gradients = _tet_shape_gradients(pts, tet_arr)
    result = np.zeros((len(tet_arr), 3), dtype=float)
    if not np.any(valid):
        return result
    nodes = tet_arr[valid]
    volume = volumes[valid]
    grad = gradients[valid]
    mu = max(float(viscosity_pa_s), 1.0e-300)
    density = max(float(density_kg_m3), 0.0)
    dt = max(float(dt_s), 1.0e-300)
    bubble_gradient_tensor = (
        (65536.0 / 15120.0)
        * volume[:, None, None]
        * np.einsum("tia,tib->tab", grad, grad)
    )
    grad_bubble_sq = np.trace(
        bubble_gradient_tensor, axis1=1, axis2=2
    )
    bubble_mass = (65536.0 / 415800.0) * volume
    scalar_momentum = mu * grad_bubble_sq + density * bubble_mass / dt
    bubble_divergence = (-(32.0 / 105.0) * volume)[:, None, None] * grad
    pressure_load = np.einsum(
        "tia,ti->ta", bubble_divergence, pressure[nodes]
    )
    if _normalize_viscous_form(viscous_form) == "vector_laplacian":
        result[valid] = -pressure_load / np.maximum(
            scalar_momentum[:, None], 1.0e-300
        )
    else:
        bubble_momentum = (
            scalar_momentum[:, None, None] * np.eye(3, dtype=float)[None]
            + mu * bubble_gradient_tensor
        )
        result[valid] = -np.linalg.solve(
            bubble_momentum, pressure_load[:, :, None]
        )[:, :, 0]
    return result


@dataclass(frozen=True)
class _MiniBubbleMomentumData:
    """Element data for an exact transient MINI-bubble condensation.

    The historical MINI path condensed only ``B_b A_b^-1 B_b.T`` into the
    pressure block.  That is sufficient for a steady, unloaded bubble, but a
    transient body-force problem also has the exact element terms

    ``A_vb = rho/dt * integral(lambda_i*b) I``,
    ``f_b  = integral(b*f_body) + rho/dt*integral(b*u_old)``,

    where ``b=256*lambda0*lambda1*lambda2*lambda3``.  Keeping these values in
    one object makes the condensed solve and the reconstructed bubble use the
    identical quadrature-free coefficients.
    """

    valid_tet_indices: np.ndarray
    nodes: np.ndarray
    bubble_momentum_n_s_m: np.ndarray
    inverse_bubble_momentum_m_s_n: np.ndarray
    bubble_divergence_m2: np.ndarray
    vertex_bubble_inertia_n_s_m: np.ndarray
    bubble_rhs_n: np.ndarray
    previous_bubble_velocity_m_s: np.ndarray


def _mini_bubble_momentum_data(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    previous_vertex_velocity_m_s: np.ndarray,
    previous_bubble_velocity_m_s: np.ndarray | None,
    body_force_density_n_m3: np.ndarray | None,
    viscosity_pa_s: float,
    inertia_density_kg_m3: float,
    dt_s: float,
    viscous_form: str,
) -> _MiniBubbleMomentumData:
    """Return exact local MINI momentum, body-load, and history terms."""

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    old_vertex = np.asarray(previous_vertex_velocity_m_s, dtype=float)
    if old_vertex.shape != pts.shape:
        raise ValueError("previous_vertex_velocity_m_s must match points")
    valid, volumes, gradients = _tet_shape_gradients(pts, tet_arr)
    valid_indices = np.flatnonzero(valid)
    nodes = tet_arr[valid]
    volume = volumes[valid]
    grad = gradients[valid]
    tet_count = len(tet_arr)

    if previous_bubble_velocity_m_s is None:
        old_bubble_all = np.zeros((tet_count, 3), dtype=float)
    else:
        old_bubble_all = np.asarray(previous_bubble_velocity_m_s, dtype=float)
        if old_bubble_all.shape != (tet_count, 3):
            raise ValueError(
                "mini_bubble_previous_velocity_m_s must have shape "
                "(n_tets, 3)"
            )
    old_bubble = old_bubble_all[valid]

    if body_force_density_n_m3 is None:
        body_density = np.zeros((len(nodes), 3), dtype=float)
    else:
        supplied_body = np.asarray(body_force_density_n_m3, dtype=float)
        if supplied_body.shape == (3,):
            body_density = np.broadcast_to(supplied_body, (len(nodes), 3)).copy()
        elif supplied_body.shape == (tet_count, 3):
            body_density = supplied_body[valid].copy()
        else:
            raise ValueError(
                "mini_bubble_body_force_density_n_m3 must have shape (3,) "
                "or (n_tets, 3)"
            )
    if not np.all(np.isfinite(body_density)):
        raise ValueError("MINI bubble body-force density must be finite")

    mu = max(float(viscosity_pa_s), 1.0e-300)
    density = max(float(inertia_density_kg_m3), 0.0)
    dt = max(float(dt_s), 1.0e-300)
    bubble_gradient_tensor = (
        (65536.0 / 15120.0)
        * volume[:, None, None]
        * np.einsum("tia,tib->tab", grad, grad)
    )
    grad_bubble_sq = np.trace(
        bubble_gradient_tensor, axis1=1, axis2=2
    )
    bubble_mass = (65536.0 / 415800.0) * volume
    scalar_momentum = mu * grad_bubble_sq + density * bubble_mass / dt
    bubble_momentum = (
        scalar_momentum[:, None, None] * np.eye(3, dtype=float)[None]
    )
    if _normalize_viscous_form(viscous_form) == "symmetric_gradient":
        bubble_momentum = bubble_momentum + mu * bubble_gradient_tensor
    inverse_bubble_momentum = np.linalg.inv(bubble_momentum)

    bubble_integral = (32.0 / 105.0) * volume
    vertex_bubble_mass = (8.0 / 105.0) * volume
    bubble_divergence = -bubble_integral[:, None, None] * grad
    vertex_bubble_inertia = density * vertex_bubble_mass / dt

    # Exact backward-Euler history tested with b: the old P1 field contributes
    # four equal int(lambda_i*b) terms, and the old bubble contributes int(b^2).
    old_vertex_sum = np.sum(old_vertex[nodes], axis=1)
    bubble_rhs = bubble_integral[:, None] * body_density
    bubble_rhs += (density / dt) * (
        vertex_bubble_mass[:, None] * old_vertex_sum
        + bubble_mass[:, None] * old_bubble
    )
    return _MiniBubbleMomentumData(
        valid_tet_indices=valid_indices,
        nodes=nodes,
        bubble_momentum_n_s_m=bubble_momentum,
        inverse_bubble_momentum_m_s_n=inverse_bubble_momentum,
        bubble_divergence_m2=bubble_divergence,
        vertex_bubble_inertia_n_s_m=vertex_bubble_inertia,
        bubble_rhs_n=bubble_rhs,
        previous_bubble_velocity_m_s=old_bubble,
    )


def _apply_exact_mini_bubble_condensation(
    momentum: sparse.csr_matrix,
    divergence: sparse.csr_matrix,
    full_momentum_rhs: np.ndarray,
    data: _MiniBubbleMomentumData,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Condense loaded transient MINI bubbles into the P1 mixed system."""

    nodes = np.asarray(data.nodes, dtype=int)
    n_vertices = int(divergence.shape[0])
    n_dofs = 3 * n_vertices
    if len(nodes) == 0:
        return (
            momentum,
            divergence,
            np.asarray(full_momentum_rhs, dtype=float).copy(),
            np.zeros(n_vertices, dtype=float),
        )
    inverse = np.asarray(data.inverse_bubble_momentum_m_s_n, dtype=float)
    cross = np.asarray(data.vertex_bubble_inertia_n_s_m, dtype=float)
    bubble_divergence = np.asarray(data.bubble_divergence_m2, dtype=float)
    inverse_rhs = np.einsum("tab,tb->ta", inverse, data.bubble_rhs_n)

    # A_hat = A_vv - A_vb A_bb^-1 A_bv.  Assemble nine component
    # blocks separately to avoid a (n_tet,4,3,4,3) temporary on large meshes.
    node_rows = np.repeat(nodes, 4, axis=1).reshape(-1)
    node_columns = np.tile(nodes, (1, 4)).reshape(-1)
    momentum_correction = sparse.csr_matrix((n_dofs, n_dofs), dtype=float)
    for component_a in range(3):
        for component_b in range(3):
            local_scalar = (
                cross * cross * inverse[:, component_a, component_b]
            )
            local = np.broadcast_to(
                local_scalar[:, None, None], (len(nodes), 4, 4)
            )
            block = sparse.coo_matrix(
                (
                    local.reshape(-1),
                    (
                        3 * node_rows + component_a,
                        3 * node_columns + component_b,
                    ),
                ),
                shape=(n_dofs, n_dofs),
            ).tocsr()
            momentum_correction = momentum_correction + block
    condensed_momentum = (momentum - momentum_correction).tocsr()

    # B_hat = B_v - B_b A_bb^-1 A_bv.  Every one of the four local P1
    # velocity nodes has the same scalar vertex-bubble mass coupling.
    divergence_factor = np.einsum(
        "tia,tab,t->tib", bubble_divergence, inverse, cross
    )
    divergence_correction = sparse.csr_matrix(
        divergence.shape, dtype=float
    )
    pressure_rows = np.repeat(nodes, 4, axis=1).reshape(-1)
    for component in range(3):
        local = np.broadcast_to(
            divergence_factor[:, :, component, None],
            (len(nodes), 4, 4),
        )
        block = sparse.coo_matrix(
            (
                local.reshape(-1),
                (
                    pressure_rows,
                    3 * node_columns + component,
                ),
            ),
            shape=divergence.shape,
        ).tocsr()
        divergence_correction = divergence_correction + block
    condensed_divergence = (divergence - divergence_correction).tocsr()

    # The vertex equation has the old-bubble cross-mass load before
    # condensation, while eliminating the new bubble subtracts
    # A_vb*A_bb^-1*f_b.  Both are exact element contributions.
    rhs = np.asarray(full_momentum_rhs, dtype=float).copy()
    vertex_local_rhs = cross[:, None] * (
        data.previous_bubble_velocity_m_s - inverse_rhs
    )
    for local_vertex in range(4):
        for component in range(3):
            np.add.at(
                rhs,
                3 * nodes[:, local_vertex] + component,
                vertex_local_rhs[:, component],
            )

    # B_hat*u - S*p = -B_b*A_bb^-1*f_b.
    continuity_rhs = np.zeros(n_vertices, dtype=float)
    local_continuity_rhs = -np.einsum(
        "tia,ta->ti", bubble_divergence, inverse_rhs
    )
    for local_pressure in range(4):
        np.add.at(
            continuity_rhs,
            nodes[:, local_pressure],
            local_continuity_rhs[:, local_pressure],
        )
    return condensed_momentum, condensed_divergence, rhs, continuity_rhs


def _reconstruct_exact_mini_bubble_velocity(
    data: _MiniBubbleMomentumData,
    tets: np.ndarray,
    vertex_velocity_m_s: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Reconstruct bubbles from the same loaded transient local equation."""

    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    velocity = np.asarray(vertex_velocity_m_s, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    result = np.zeros((len(tet_arr), 3), dtype=float)
    if len(data.nodes) == 0:
        return result
    current_vertex_sum = np.sum(velocity[data.nodes], axis=1)
    vertex_coupling = (
        data.vertex_bubble_inertia_n_s_m[:, None] * current_vertex_sum
    )
    pressure_load = np.einsum(
        "tia,ti->ta", data.bubble_divergence_m2, pressure[data.nodes]
    )
    local_rhs = data.bubble_rhs_n - vertex_coupling - pressure_load
    result[data.valid_tet_indices] = np.einsum(
        "tab,tb->ta", data.inverse_bubble_momentum_m_s_n, local_rhs
    )
    return result


@dataclass(frozen=True)
class ViscousVelocityResult:
    """Forces and convergence data returned by the tetra-viscous solve."""

    residual_l2_n: float
    velocity_linf_m_s: float
    solver_info: int
    operator_diagonal_n_s_m: np.ndarray
    # Direct PR35 Case-12 Newtonian Cauchy force used by the momentum solver:
    #
    #   sigma = mu * (grad(u) + grad(u).T),
    #   F_i = -sum_T |T| * sigma_T @ grad(N_i) = -K_mu @ u.
    #
    # This is a solver force, not a residual-based diagnostic.
    cauchy_force_n: np.ndarray | None = None
    # Physical right-hand-side force associated with any explicit linear
    # velocity-constraint multiplier in the saddle system.
    constraint_force_n: np.ndarray | None = None


def _axisymmetric_ring_indices(
    points: np.ndarray,
    *,
    tolerance_m: float = 1.0e-11,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return cylindrical radii and one ring index per spatial point."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")
    tolerance = max(float(tolerance_m), 64.0 * np.finfo(float).eps)
    radius = np.hypot(pts[:, 0], pts[:, 1])
    meridian = np.column_stack((radius, pts[:, 2]))
    keys = np.rint(meridian / tolerance).astype(np.int64)
    _unique, ring = np.unique(keys, axis=0, return_inverse=True)
    ring_count = int(np.max(ring)) + 1 if len(ring) else 0
    return radius, ring, ring_count


def _exact_axisymmetric_pressure_transform(
    points: np.ndarray,
    *,
    tolerance_m: float = 1.0e-11,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Map one reduced P1 pressure coefficient to each meridian ring."""

    pts = np.asarray(points, dtype=float)
    _radius, ring, ring_count = _axisymmetric_ring_indices(
        pts, tolerance_m=tolerance_m
    )
    transform = sparse.coo_matrix(
        (
            np.ones(len(pts), dtype=float),
            (np.arange(len(pts), dtype=int), ring),
        ),
        shape=(len(pts), ring_count),
    ).tocsr()
    return transform, ring


def _exact_axisymmetric_velocity_transform(
    points: np.ndarray,
    constrained_dofs: np.ndarray,
    *,
    tangent_vertex_directions: np.ndarray | None = None,
    tolerance_m: float = 1.0e-11,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return a Cartesian velocity basis for an axisymmetric node set.

    Vertices of a revolved tetra mesh are grouped by their common ``(r, z)``
    meridian coordinate.  Each unconstrained off-axis ring receives one
    radial and one axial velocity degree of freedom.  Axis vertices retain
    only axial velocity, which is the regular axisymmetric limit.  A nonzero
    tangent direction replaces the two meridional velocity degrees of freedom
    by one sliding degree of freedom, allowing a contact-line no-penetration
    condition to be imposed in the saddle-point system rather than projected
    afterward.

    ``points`` may contain either the P1 mesh vertices or the complete P2
    velocity-node set (vertices followed by tetra-edge midpoints).  Thus the
    same construction constrains every Taylor--Hood velocity degree of
    freedom, not merely the returned vertex velocities.
    """

    pts = np.asarray(points, dtype=float)
    constrained = np.asarray(constrained_dofs, dtype=bool)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n_vertices, 3)")
    if constrained.shape != (3 * len(pts),):
        raise ValueError("constrained_dofs must contain three entries per vertex")
    tolerance = max(float(tolerance_m), 64.0 * np.finfo(float).eps)
    radius, geometric_ring, _geometric_ring_count = _axisymmetric_ring_indices(
        pts, tolerance_m=tolerance
    )

    vertex_constrained = constrained.reshape((-1, 3))
    if np.any(vertex_constrained != vertex_constrained[:, :1]):
        raise ValueError(
            "Exact axisymmetric reduction currently requires whole-vertex constraints"
        )
    vertex_fixed = vertex_constrained[:, 0]

    tangent = None
    tangent_active = np.zeros(len(pts), dtype=bool)
    if tangent_vertex_directions is not None:
        tangent = np.asarray(tangent_vertex_directions, dtype=float)
        if tangent.shape != pts.shape:
            raise ValueError("tangent_vertex_directions must match points")
        tangent_norm = np.linalg.norm(tangent, axis=1)
        tangent_active = tangent_norm > 1.0e-14
        tangent = tangent.copy()
        tangent[tangent_active] /= tangent_norm[tangent_active, None]

    # Coincident geometric nodes may legitimately belong to different
    # boundary spaces (for example a fixed solid seed node and a free PR37
    # contact-line node).  Pressure remains one coefficient per geometric
    # (r,z) ring, but velocity groups must also respect fixed/free and
    # tangent/free classifications.
    velocity_key = np.column_stack(
        (
            geometric_ring,
            vertex_fixed.astype(np.int8),
            tangent_active.astype(np.int8),
        )
    )
    _unique_velocity_key, ring = np.unique(
        velocity_key, axis=0, return_inverse=True
    )
    ring_count = int(np.max(ring)) + 1 if len(ring) else 0

    radial_column = np.full(ring_count, -1, dtype=int)
    axial_column = np.full(ring_count, -1, dtype=int)
    tangent_column = np.full(ring_count, -1, dtype=int)
    velocity_dofs = 0
    for group in range(ring_count):
        members = np.flatnonzero(ring == group)
        if members.size == 0 or vertex_fixed[members[0]]:
            continue
        if tangent_active[members[0]]:
            tangent_column[group] = velocity_dofs
            velocity_dofs += 1
            continue
        if float(np.mean(radius[members])) > 2.0 * tolerance:
            radial_column[group] = velocity_dofs
            velocity_dofs += 1
        axial_column[group] = velocity_dofs
        velocity_dofs += 1

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for vertex in range(len(pts)):
        group = int(ring[vertex])
        if vertex_fixed[vertex]:
            continue
        if tangent_column[group] >= 0:
            assert tangent is not None
            column = int(tangent_column[group])
            for component in range(3):
                value = float(tangent[vertex, component])
                if abs(value) > 1.0e-15:
                    rows.append(3 * vertex + component)
                    columns.append(column)
                    values.append(value)
            continue
        if radial_column[group] >= 0:
            inverse_radius = 1.0 / max(float(radius[vertex]), tolerance)
            column = int(radial_column[group])
            rows.extend((3 * vertex, 3 * vertex + 1))
            columns.extend((column, column))
            values.extend(
                (
                    float(pts[vertex, 0]) * inverse_radius,
                    float(pts[vertex, 1]) * inverse_radius,
                )
            )
        if axial_column[group] >= 0:
            rows.append(3 * vertex + 2)
            columns.append(int(axial_column[group]))
            values.append(1.0)

    velocity_transform = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(3 * len(pts), velocity_dofs),
    ).tocsr()
    return velocity_transform, ring


def _exact_axisymmetric_transforms(
    points: np.ndarray,
    constrained_dofs: np.ndarray,
    *,
    tangent_vertex_directions: np.ndarray | None = None,
    tolerance_m: float = 1.0e-11,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Return velocity and pressure transforms for a P1 axisymmetric solve.

    Galerkin reduction with both transforms constrains velocity *and* pressure
    during the solve; it is not an a-posteriori azimuthal average.
    """

    velocity_transform, velocity_ring = _exact_axisymmetric_velocity_transform(
        points,
        constrained_dofs,
        tangent_vertex_directions=tangent_vertex_directions,
        tolerance_m=tolerance_m,
    )
    pressure_transform, pressure_ring = _exact_axisymmetric_pressure_transform(
        points, tolerance_m=tolerance_m
    )
    # Velocity rings may be split by boundary-condition class while pressure
    # retains one coefficient per geometric meridian ring.  The mixed
    # Galerkin products below support these distinct bases directly.
    return velocity_transform, pressure_transform, velocity_ring


def implicit_tetra_stokes_velocity_pressure(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    velocities_m_s: np.ndarray,
    external_forces_n: np.ndarray,
    masses_kg: np.ndarray,
    viscosity_pa_s: float | np.ndarray,
    viscous_form: str = "vector_laplacian",
    dt_s: float,
    inertia_weight: float = 1.0,
    continuity_source_m3_s: np.ndarray | None = None,
    continuity_velocity_coupling: sparse.spmatrix | None = None,
    mini_bubble_consistent_momentum: bool = False,
    mini_bubble_body_force_density_n_m3: np.ndarray | None = None,
    mini_bubble_previous_velocity_m_s: np.ndarray | None = None,
    constrained_vertices: np.ndarray | None = None,
    prescribed_vertex_velocities_m_s: np.ndarray | None = None,
    enforce_exact_axisymmetry: bool = False,
    axisymmetry_tolerance_m: float = 1.0e-11,
    tangent_vertex_directions: np.ndarray | None = None,
    additional_stiffness_n_s_m: sparse.spmatrix | None = None,
    linear_velocity_constraints: sparse.spmatrix | None = None,
    linear_velocity_constraint_rhs_m_s: np.ndarray | None = None,
    pressure_stabilization_coefficient: float = 1.0 / 12.0,
    pressure_stabilization_method: str = "mini",
    pressure_relative_regularization: float = 1.0e-10,
    rtol: float = 1.0e-8,
    maxiter: int = 500,
    linear_system_cache: dict | None = None,
) -> tuple[np.ndarray, PressureProjectionResult, ViscousVelocityResult]:
    """Solve the coupled PR33/PR35 velocity-pressure system on P1 tetrahedra.

    The earlier split projection replaced the inverse viscous operator in the
    pressure Schur complement by its diagonal.  That approximation is poor in
    a thin film and can leave alternating radial fluxes.  This routine solves
    the stabilized saddle-point system directly:

    ``[A  B.T; B  -S] [u, p] = [M u_old / dt + F, 0]``.

    ``A`` contains inertia, PR35 viscosity, and optional implicit capillarity;
    ``B`` is the PR33 weak divergence, and ``S`` is the standard equal-order
    pressure stabilization.  ``continuity_source_m3_s`` supplies the weak
    conservative transfer row in ``B u-S p=q``; omitting it gives ``q=0``.
    No case-specific profile or validation datum is used by this operator.

    ``mini_bubble_consistent_momentum`` is deliberately opt-in so historical
    cases retain their original algebra.  When enabled with MINI pressure
    stabilization, the solver also condenses the exact bubble body-force
    load, vertex--bubble inertial cross block, and previous P1/bubble history.
    The caller supplies a volumetric force density (for gravity,
    ``rho*[0,0,-g]``) in addition to its ordinary assembled P1 vertex loads.
    """

    points_arr = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    velocity = np.asarray(velocities_m_s, dtype=float)
    force = np.asarray(external_forces_n, dtype=float)
    masses = np.asarray(masses_kg, dtype=float)
    if velocity.shape != points_arr.shape or force.shape != points_arr.shape:
        raise ValueError("velocities_m_s and external_forces_n must match points.")
    if masses.shape != (len(points_arr),):
        raise ValueError("masses_kg must have one entry per vertex.")

    dt = max(float(dt_s), 1.0e-30)
    inertia = max(float(inertia_weight), 0.0)
    mass_diagonal = np.repeat(inertia * np.maximum(masses, 0.0) / dt, 3)
    viscosity_scalar = float(
        np.asarray(viscosity_pa_s, dtype=float).reshape(-1)[0]
    )
    assembly_cache = (
        linear_system_cache.setdefault("operator_assembly", {})
        if linear_system_cache is not None
        else None
    )
    assembly_key = (
        int(points_arr.__array_interface__["data"][0]),
        points_arr.shape,
        int(tet_arr.__array_interface__["data"][0]),
        tet_arr.shape,
        float(viscosity_scalar),
        str(viscous_form),
        float(dt),
        float(inertia),
        str(pressure_stabilization_method).lower(),
        float(pressure_stabilization_coefficient),
        bool(mini_bubble_consistent_momentum),
    )
    reuse_assembly = bool(
        assembly_cache is not None
        and assembly_cache.get("key") == assembly_key
    )
    if reuse_assembly:
        viscous_stiffness = assembly_cache["viscous_stiffness"]
    else:
        viscous_stiffness = tetra_viscous_stiffness_matrix(
            points_arr,
            tet_arr,
            viscosity_pa_s,
            viscous_form=viscous_form,
        )
    momentum = viscous_stiffness + sparse.diags(
        mass_diagonal, format="csr"
    )
    if additional_stiffness_n_s_m is not None:
        extra = sparse.csr_matrix(additional_stiffness_n_s_m)
        if extra.shape != momentum.shape:
            raise ValueError(
                "additional_stiffness_n_s_m must have shape (3*n_vertices, 3*n_vertices)."
            )
        momentum = momentum + extra

    stabilization_method = str(pressure_stabilization_method).lower()
    if bool(mini_bubble_consistent_momentum) and stabilization_method != "mini":
        raise ValueError(
            "mini_bubble_consistent_momentum requires MINI stabilization"
        )
    if (
        not bool(mini_bubble_consistent_momentum)
        and (
            mini_bubble_body_force_density_n_m3 is not None
            or mini_bubble_previous_velocity_m_s is not None
        )
    ):
        raise ValueError(
            "MINI bubble body/history data require "
            "mini_bubble_consistent_momentum=True"
        )

    constrained = constrained_dof_mask(len(points_arr), constrained_vertices)
    prescribed_velocity = np.zeros_like(points_arr)
    if prescribed_vertex_velocities_m_s is not None:
        prescribed_velocity = np.asarray(
            prescribed_vertex_velocities_m_s, dtype=float
        ).copy()
        if prescribed_velocity.shape != points_arr.shape:
            raise ValueError(
                "prescribed_vertex_velocities_m_s must match points"
            )
        prescribed_dof_mask = np.linalg.norm(
            prescribed_velocity, axis=1
        ) > 0.0
        constrained_vertex_mask = constrained.reshape((-1, 3))[:, 0]
        if np.any(prescribed_dof_mask & ~constrained_vertex_mask):
            raise ValueError(
                "nonzero prescribed velocity requires a constrained vertex"
            )
    prescribed_vector = prescribed_velocity.reshape(-1)
    free = ~constrained
    divergence = (
        assembly_cache["divergence"]
        if reuse_assembly
        else tet_vertex_divergence_matrix(points_arr, tet_arr)
    )
    continuity_source = (
        np.zeros(len(points_arr), dtype=float)
        if continuity_source_m3_s is None
        else np.asarray(continuity_source_m3_s, dtype=float)
    )
    if continuity_source.shape != (len(points_arr),):
        raise ValueError(
            "continuity_source_m3_s must have one value per pressure vertex"
        )
    mini_momentum_data = None
    cached_stabilization = (
        assembly_cache.get("stabilization")
        if reuse_assembly
        else None
    )
    cached_density = (
        assembly_cache.get("density")
        if reuse_assembly
        else None
    )
    if cached_stabilization is not None:
        stabilization = cached_stabilization
        density = float(cached_density)
    elif stabilization_method == "mini":
        valid_tet, tet_volume, _tet_gradient = _tet_shape_gradients(
            points_arr, tet_arr
        )
        liquid_volume = float(np.sum(tet_volume[valid_tet]))
        density = float(np.sum(masses)) / max(liquid_volume, 1.0e-300)
        stabilization = tetra_mini_pressure_stabilization_matrix(
            points_arr,
            tet_arr,
            viscosity_pa_s=viscosity_scalar,
            density_kg_m3=inertia * density,
            dt_s=dt,
            viscous_form=viscous_form,
        )
    else:
        stabilization = tetra_pressure_stabilization_matrix(
            points_arr,
            tet_arr,
            viscosity_pa_s=viscosity_scalar,
            coefficient=float(pressure_stabilization_coefficient),
        )
        density = 0.0
    if assembly_cache is not None and not reuse_assembly:
        assembly_cache.clear()
        assembly_cache.update(
            {
                "key": assembly_key,
                "viscous_stiffness": viscous_stiffness,
                "divergence": divergence,
                "stabilization": stabilization,
                "density": float(density),
            }
        )
        linear_system_cache["operator_assembly_builds"] = int(
            linear_system_cache.get("operator_assembly_builds", 0)
        ) + 1
    elif reuse_assembly and linear_system_cache is not None:
        linear_system_cache["operator_assembly_reuses"] = int(
            linear_system_cache.get("operator_assembly_reuses", 0)
        ) + 1
    full_momentum_rhs = (
        mass_diagonal * velocity.reshape(-1) + force.reshape(-1)
    )
    bubble_continuity_rhs = np.zeros(len(points_arr), dtype=float)
    if bool(mini_bubble_consistent_momentum):
        mini_momentum_data = _mini_bubble_momentum_data(
            points_arr,
            tet_arr,
            previous_vertex_velocity_m_s=velocity,
            previous_bubble_velocity_m_s=mini_bubble_previous_velocity_m_s,
            body_force_density_n_m3=mini_bubble_body_force_density_n_m3,
            viscosity_pa_s=viscosity_scalar,
            inertia_density_kg_m3=inertia * density,
            dt_s=dt,
            viscous_form=viscous_form,
        )
        (
            momentum,
            divergence,
            full_momentum_rhs,
            bubble_continuity_rhs,
        ) = _apply_exact_mini_bubble_condensation(
            momentum,
            divergence,
            full_momentum_rhs,
            mini_momentum_data,
        )
    if continuity_velocity_coupling is None:
        continuity_operator = divergence
    else:
        velocity_coupling = sparse.csr_matrix(
            continuity_velocity_coupling
        )
        if velocity_coupling.shape != divergence.shape:
            raise ValueError(
                "continuity_velocity_coupling must have shape "
                "(n_pressure_vertices, 3*n_vertices)"
            )
        continuity_operator = (divergence - velocity_coupling).tocsr()
    divergence_free = divergence[:, free].tocsr()
    continuity_free = continuity_operator[:, free].tocsr()
    schur_scale = continuity_free @ sparse.diags(
        1.0 / np.maximum(np.abs(momentum.diagonal()[free]), 1.0e-300),
        format="csr",
    ) @ divergence_free.T
    pressure_scale = (
        float(np.mean(np.abs((schur_scale + stabilization).diagonal())))
        if stabilization.shape[0]
        else 0.0
    )
    pressure_regularization = max(
        float(pressure_relative_regularization) * max(pressure_scale, 1.0e-30),
        1.0e-30,
    )
    pressure_block = -(
        stabilization
        + pressure_regularization
        * sparse.eye(stabilization.shape[0], format="csr")
    )
    # Eliminate the nonhomogeneous Dirichlet trace u=g from both mixed
    # equations.  Omitting either term makes the pressure solve compensate a
    # boundary motion that is absent from its algebraic continuity equation:
    #
    #   A_ff u_f + B_f.T p = f_f - A_fc g_c
    #   B_f u_f - S p     =       - B_c g_c.
    momentum_rhs = (
        full_momentum_rhs - np.asarray(momentum @ prescribed_vector).ravel()
    )[free]
    continuity_rhs = (
        bubble_continuity_rhs
        + continuity_source
        - np.asarray(
            continuity_operator @ prescribed_vector, dtype=float
        ).ravel()
    )
    constraint_free = None
    constraint_rhs = np.empty(0, dtype=float)
    if linear_velocity_constraints is not None:
        constraint = sparse.csr_matrix(linear_velocity_constraints)
        if constraint.shape[1] != 3 * len(points_arr):
            raise ValueError(
                "linear_velocity_constraints must have 3*n_vertices columns"
            )
        supplied_rhs = np.asarray(
            linear_velocity_constraint_rhs_m_s, dtype=float
        ).reshape(-1)
        if supplied_rhs.shape != (constraint.shape[0],):
            raise ValueError(
                "linear_velocity_constraint_rhs_m_s must match constraint rows"
            )
        constraint_rhs = supplied_rhs - np.asarray(
            constraint @ prescribed_vector, dtype=float
        ).ravel()
        constraint_free = constraint[:, free].tocsr()
    velocity_transform = None
    pressure_transform = None
    if bool(enforce_exact_axisymmetry):
        velocity_transform, pressure_transform, _ring = (
            _exact_axisymmetric_transforms(
                points_arr,
                constrained,
                tangent_vertex_directions=tangent_vertex_directions,
                tolerance_m=float(axisymmetry_tolerance_m),
            )
        )
        velocity_transform_free = velocity_transform[free, :].tocsr()
        reduced_momentum = (
            velocity_transform_free.T
            @ momentum[free][:, free]
            @ velocity_transform_free
        ).tocsr()
        reduced_pressure_gradient = (
            pressure_transform.T
            @ divergence_free
            @ velocity_transform_free
        ).tocsr()
        reduced_continuity = (
            pressure_transform.T
            @ continuity_free
            @ velocity_transform_free
        ).tocsr()
        reduced_pressure_block = (
            pressure_transform.T @ pressure_block @ pressure_transform
        ).tocsr()
        reduced_constraint = (
            constraint_free @ velocity_transform_free
            if constraint_free is not None
            else None
        )
        if reduced_constraint is None:
            system = sparse.bmat(
                (
                    (reduced_momentum, reduced_pressure_gradient.T),
                    (reduced_continuity, reduced_pressure_block),
                ),
                format="csr",
            )
        else:
            zero_pc = sparse.csr_matrix(
                (reduced_pressure_block.shape[0], reduced_constraint.shape[0])
            )
            zero_cc = sparse.csr_matrix(
                (reduced_constraint.shape[0], reduced_constraint.shape[0])
            )
            system = sparse.bmat(
                (
                    (
                        reduced_momentum,
                        reduced_pressure_gradient.T,
                        reduced_constraint.T,
                    ),
                    (reduced_continuity, reduced_pressure_block, zero_pc),
                    (reduced_constraint, zero_pc.T, zero_cc),
                ),
                format="csr",
            )
        reduced_momentum_rhs = np.asarray(
            velocity_transform_free.T @ momentum_rhs, dtype=float
        ).ravel()
        reduced_continuity_rhs = np.asarray(
            pressure_transform.T @ continuity_rhs, dtype=float
        ).ravel()
        rhs = np.concatenate(
            (reduced_momentum_rhs, reduced_continuity_rhs, constraint_rhs)
        )
        velocity_unknowns = int(velocity_transform.shape[1])
        pressure_unknowns = int(pressure_transform.shape[1])
    else:
        reduced_momentum = momentum[free][:, free].tocsr()
        if constraint_free is None:
            system = sparse.bmat(
                (
                    (reduced_momentum, divergence_free.T),
                    (continuity_free, pressure_block),
                ),
                format="csr",
            )
        else:
            zero_pc = sparse.csr_matrix(
                (pressure_block.shape[0], constraint_free.shape[0])
            )
            zero_cc = sparse.csr_matrix(
                (constraint_free.shape[0], constraint_free.shape[0])
            )
            system = sparse.bmat(
                (
                    (reduced_momentum, divergence_free.T, constraint_free.T),
                    (continuity_free, pressure_block, zero_pc),
                    (constraint_free, zero_pc.T, zero_cc),
                ),
                format="csr",
            )
        rhs = np.concatenate((momentum_rhs, continuity_rhs, constraint_rhs))
        velocity_unknowns = int(np.count_nonzero(free))
        pressure_unknowns = int(pressure_block.shape[0])

    info = 0
    if system.shape[0] <= 100000:
        if linear_system_cache is None:
            solution = spla.spsolve(system.tocsc(), rhs)
        else:
            system_csc = system.tocsc()
            # A nonlinear PR37 solve can alternate between the unconstrained
            # and sphere-active-set matrices while changing only the RHS.
            # Retaining only the most recent LU factorization therefore
            # refactorized both matrices on every Cox iteration.  Keep a small
            # exact matrix cache; equality is still bitwise, so no factor is
            # ever reused for a changed operator.
            factor_caches = linear_system_cache.setdefault(
                "factorizations_cached", []
            )
            matching_factor = None
            for cached in factor_caches:
                cached_system = cached.get("system_csc")
                cached_factor = cached.get("factor")
                if (
                    cached_system is not None
                    and cached_factor is not None
                    and cached_system.shape == system_csc.shape
                    and np.array_equal(
                        cached_system.indptr, system_csc.indptr
                    )
                    and np.array_equal(
                        cached_system.indices, system_csc.indices
                    )
                    and np.array_equal(
                        cached_system.data, system_csc.data
                    )
                ):
                    matching_factor = cached_factor
                    break
            if matching_factor is not None:
                solution = matching_factor.solve(rhs)
                linear_system_cache["hits"] = int(
                    linear_system_cache.get("hits", 0)
                ) + 1
            else:
                factor = spla.splu(system_csc)
                solution = factor.solve(rhs)
                factor_caches.append(
                    {
                        "system_csc": system_csc,
                        "factor": factor,
                    }
                )
                if len(factor_caches) > 16:
                    del factor_caches[0]
                linear_system_cache["factorizations"] = int(
                    linear_system_cache.get("factorizations", 0)
                ) + 1
    else:
        diagonal = np.maximum(np.abs(system.diagonal()), 1.0e-30)
        solution, info = spla.minres(
            system,
            rhs,
            M=sparse.diags(1.0 / diagonal, format="csr"),
            rtol=float(rtol),
            maxiter=int(maxiter),
        )
    if info != 0 or not np.all(np.isfinite(solution)):
        solution = spla.lsmr(
            system,
            rhs,
            atol=float(rtol),
            btol=float(rtol),
            maxiter=int(maxiter),
        )[0]
        info = -abs(int(info)) if int(info) != 0 else 0

    if bool(enforce_exact_axisymmetry):
        assert velocity_transform is not None and pressure_transform is not None
        velocity_vec = prescribed_vector + np.asarray(
            velocity_transform @ solution[:velocity_unknowns], dtype=float
        ).ravel()
        pressure = np.asarray(
            pressure_transform
            @ solution[
                velocity_unknowns : velocity_unknowns + pressure_unknowns
            ],
            dtype=float,
        ).ravel()
    else:
        velocity_vec = prescribed_vector.copy()
        velocity_vec[free] = solution[:velocity_unknowns]
        pressure = np.asarray(
            solution[velocity_unknowns : velocity_unknowns + pressure_unknowns],
            dtype=float,
        )
    constraint_multipliers = np.asarray(
        solution[velocity_unknowns + pressure_unknowns :],
        dtype=float,
    )
    if linear_velocity_constraints is None:
        constraint_force = np.zeros(3 * len(points_arr), dtype=float)
    else:
        constraint_force = -np.asarray(
            sparse.csr_matrix(linear_velocity_constraints).T
            @ constraint_multipliers,
            dtype=float,
        ).reshape(-1)
    projected = velocity_vec.reshape(points_arr.shape)
    continuity = (
        continuity_operator @ velocity_vec
        - stabilization @ pressure
        - bubble_continuity_rhs
        - continuity_source
    )
    momentum_residual = (
        momentum @ velocity_vec
        + divergence.T @ pressure
        - full_momentum_rhs
    )
    mini_bubble_velocity = None
    if stabilization_method == "mini":
        if mini_momentum_data is not None:
            mini_bubble_velocity = _reconstruct_exact_mini_bubble_velocity(
                mini_momentum_data,
                tet_arr,
                projected,
                pressure,
            )
        else:
            mini_bubble_velocity = tetra_mini_bubble_velocity(
                points_arr,
                tet_arr,
                pressure,
                viscosity_pa_s=viscosity_scalar,
                density_kg_m3=inertia * density,
                dt_s=dt,
                viscous_form=viscous_form,
            )
    pressure_result = PressureProjectionResult(
        pressure_pa=pressure,
        residual_before_m3_s=float(
            np.sqrt(
                np.mean(
                    (
                        divergence @ velocity.reshape(-1)
                        - continuity_source
                    )
                    ** 2
                )
            )
        ),
        residual_after_m3_s=float(np.sqrt(np.mean(continuity**2))),
        pressure_l2_pa=float(np.sqrt(np.mean(pressure**2))),
        pressure_linf_pa=float(np.max(np.abs(pressure))),
        solver_info=int(info),
        mini_bubble_velocity_m_s=mini_bubble_velocity,
        pressure_force_n=np.asarray(
            -divergence.T @ pressure, dtype=float
        ).reshape(points_arr.shape),
    )
    # This is exactly the Case-12 tetra-Cauchy force.  ``viscous_stiffness`` is
    # the same K_mu already present in the solved momentum block, so this force
    # is part of the solver equation rather than a post-hoc residual split.
    cauchy_force = -np.asarray(
        viscous_stiffness @ velocity_vec, dtype=float
    ).reshape(points_arr.shape)
    viscous_result = ViscousVelocityResult(
        residual_l2_n=float(np.sqrt(np.mean(momentum_residual[free] ** 2)))
        if np.any(free)
        else 0.0,
        velocity_linf_m_s=float(np.max(np.linalg.norm(projected, axis=1))),
        solver_info=int(info),
        operator_diagonal_n_s_m=np.asarray(momentum.diagonal(), dtype=float),
        cauchy_force_n=cauchy_force,
        constraint_force_n=constraint_force.reshape(points_arr.shape),
    )
    return projected, pressure_result, viscous_result


def _tetra_p2_velocity_topology(
    tets: np.ndarray,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global P2 edges and the ten scalar velocity nodes per tetra."""

    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    edge_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        dtype=int,
    )
    local_edges = np.sort(tet_arr[:, edge_pairs], axis=2)
    edges, inverse = np.unique(
        local_edges.reshape((-1, 2)), axis=0, return_inverse=True
    )
    edge_nodes = int(n_vertices) + inverse.reshape((-1, 6))
    local_nodes = np.concatenate((tet_arr, edge_nodes), axis=1)
    return edges, local_edges, local_nodes


def _tetra_p2_stiffness_and_divergence(
    points: np.ndarray,
    tets: np.ndarray,
    viscosity_pa_s: float,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble the exact degree-two Taylor-Hood tetra operators.

    Four symmetric tetra quadrature points integrate the products of linear
    P2 gradients, and ``P1 * grad(P2)``, exactly.  The returned divergence has
    one continuous P1 pressure row per original mesh vertex.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    valid, volumes, gradients = _tet_shape_gradients(pts, tet_arr)
    if not np.all(valid):
        tet_arr = tet_arr[valid]
        volumes = volumes[valid]
        gradients = gradients[valid]
    else:
        volumes = volumes.copy()
        gradients = gradients.copy()
    edges, _local_edges, local_nodes = _tetra_p2_velocity_topology(
        tet_arr, len(pts)
    )
    n_velocity_nodes = len(pts) + len(edges)
    if len(tet_arr) == 0:
        return (
            sparse.csr_matrix((3 * n_velocity_nodes, 3 * n_velocity_nodes)),
            sparse.csr_matrix((len(pts), 3 * n_velocity_nodes)),
            edges,
            local_nodes,
        )

    large = 0.5854101966249685
    small = 0.1381966011250105
    barycentric = np.full((4, 4), small, dtype=float)
    np.fill_diagonal(barycentric, large)
    edge_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        dtype=int,
    )
    local_stiffness = np.zeros((len(tet_arr), 10, 10), dtype=float)
    local_divergence = np.zeros((len(tet_arr), 4, 10, 3), dtype=float)
    for lam in barycentric:
        grad_basis = np.empty((len(tet_arr), 10, 3), dtype=float)
        grad_basis[:, :4] = (
            (4.0 * lam - 1.0)[None, :, None] * gradients
        )
        for local_edge, (first, second) in enumerate(edge_pairs):
            grad_basis[:, 4 + local_edge] = 4.0 * (
                lam[first] * gradients[:, second]
                + lam[second] * gradients[:, first]
            )
        local_stiffness += 0.25 * np.einsum(
            "tia,tja->tij", grad_basis, grad_basis
        )
        local_divergence += 0.25 * np.einsum(
            "i,tja->tija", lam, grad_basis
        )
    local_stiffness *= (
        float(viscosity_pa_s) * volumes[:, None, None]
    )
    local_divergence *= volumes[:, None, None, None]

    stiffness_rows = np.repeat(local_nodes, 10, axis=1).reshape(-1)
    stiffness_cols = np.tile(local_nodes, (1, 10)).reshape(-1)
    scalar_stiffness = sparse.coo_matrix(
        (
            local_stiffness.reshape(-1),
            (stiffness_rows, stiffness_cols),
        ),
        shape=(n_velocity_nodes, n_velocity_nodes),
    ).tocsr()
    momentum = sparse.kron(
        scalar_stiffness, sparse.eye(3, format="csr"), format="csr"
    )

    pressure_nodes = tet_arr
    divergence_rows = np.broadcast_to(
        pressure_nodes[:, :, None, None], local_divergence.shape
    ).reshape(-1)
    divergence_cols = np.broadcast_to(
        3 * local_nodes[:, None, :, None]
        + np.arange(3)[None, None, None, :],
        local_divergence.shape,
    ).reshape(-1)
    divergence = sparse.coo_matrix(
        (
            local_divergence.reshape(-1),
            (divergence_rows, divergence_cols),
        ),
        shape=(len(pts), 3 * n_velocity_nodes),
    ).tocsr()
    return momentum, divergence, edges, local_nodes


def _tetra_p2_consistent_mass(
    points: np.ndarray,
    tets: np.ndarray,
    density_kg_m3: float | np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return the exact degree-four P2 tetra mass matrix and global edges."""

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    valid, volumes, _gradients = _tet_shape_gradients(pts, tet_arr)
    density = np.asarray(density_kg_m3, dtype=float)
    if density.ndim == 0:
        density_tet = np.full(len(tet_arr), float(density), dtype=float)
    elif density.shape == (len(tet_arr),):
        density_tet = density
    else:
        raise ValueError(
            "density_kg_m3 must be scalar or one value per tetrahedron"
        )
    if not np.all(valid):
        tet_arr = tet_arr[valid]
        volumes = volumes[valid]
        density_tet = density_tet[valid]
    edges, _local_edges, local_nodes = _tetra_p2_velocity_topology(
        tet_arr, len(pts)
    )
    n_velocity_nodes = len(pts) + len(edges)
    if len(tet_arr) == 0:
        return sparse.csr_matrix((0, 0)), edges

    centroid = np.full(4, 0.25, dtype=float)
    first = 0.7857142857142857
    second = 0.07142857142857143
    paired_large = 0.3994035761667992
    paired_small = 0.1005964238332008
    barycentric: list[np.ndarray] = [centroid]
    weights: list[float] = [-0.07893333333333333]
    for index in range(4):
        value = np.full(4, second, dtype=float)
        value[index] = first
        barycentric.append(value)
        weights.append(0.04573333333333333)
    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            value = np.full(4, paired_small, dtype=float)
            value[first_index] = paired_large
            value[second_index] = paired_large
            barycentric.append(value)
            weights.append(0.14933333333333335)
    local_mass = np.zeros((len(tet_arr), 10, 10), dtype=float)
    edge_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        dtype=int,
    )
    for lam, weight in zip(barycentric, weights):
        basis = np.empty(10, dtype=float)
        basis[:4] = lam * (2.0 * lam - 1.0)
        basis[4:] = np.asarray(
            [4.0 * lam[i] * lam[j] for i, j in edge_pairs], dtype=float
        )
        local_mass += float(weight) * basis[None, :, None] * basis[None, None, :]
    local_mass *= density_tet[:, None, None] * volumes[:, None, None]
    rows = np.repeat(local_nodes, 10, axis=1).reshape(-1)
    cols = np.tile(local_nodes, (1, 10)).reshape(-1)
    scalar_mass = sparse.coo_matrix(
        (local_mass.reshape(-1), (rows, cols)),
        shape=(n_velocity_nodes, n_velocity_nodes),
    ).tocsr()
    return sparse.kron(
        scalar_mass, sparse.eye(3, format="csr"), format="csr"
    ), edges


def p2_consistent_surface_force(
    points: np.ndarray,
    tets: np.ndarray,
    surface_faces: np.ndarray,
    vertex_surface_force_n: np.ndarray,
    vertex_surface_area_m2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a DDG nodal surface force into the conforming P2 test space.

    Cotangent force is the surface-energy derivative for the P1 geometry.  A
    Taylor-Hood momentum solve must not apply that load only to its vertex
    basis functions: doing so excites an artificial vertex/edge mode.  This
    routine reconstructs the P1 traction ``force/dual_area`` and integrates
    it exactly against every quadratic triangle basis function.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    faces = np.asarray(surface_faces, dtype=int).reshape((-1, 3))
    force = np.asarray(vertex_surface_force_n, dtype=float)
    area = np.asarray(vertex_surface_area_m2, dtype=float)
    edges, _local_edges, _local_nodes = _tetra_p2_velocity_topology(
        tet_arr, len(pts)
    )
    result = np.zeros((len(pts) + len(edges), 3), dtype=float)
    if len(faces) == 0:
        return result, edges
    traction = np.divide(
        force,
        np.maximum(area, 1.0e-300)[:, None],
        out=np.zeros_like(force),
        where=area[:, None] > 1.0e-300,
    )
    face_edges = np.sort(
        np.vstack(
            (
                faces[:, (0, 1)],
                faces[:, (0, 2)],
                faces[:, (1, 2)],
            )
        ),
        axis=1,
    )
    unique_face_edges, edge_count = np.unique(
        face_edges, axis=0, return_counts=True
    )
    boundary_vertices = np.unique(unique_face_edges[edge_count == 1])
    # At an open free-surface boundary the cotangent derivative contains a
    # line traction.  It is balanced by the solid/contact-line reaction and
    # must remain on that boundary DOF; smearing it into adjacent P2 face
    # modes creates a false capillary jet.
    traction[boundary_vertices] = 0.0
    edge_lookup = {
        (int(first), int(second)): index
        for index, (first, second) in enumerate(edges)
    }
    face_edge_pairs = ((0, 1), (0, 2), (1, 2))
    face_nodes = np.empty((len(faces), 6), dtype=int)
    face_nodes[:, :3] = faces
    for face_index, face in enumerate(faces):
        for local_edge, (first, second) in enumerate(face_edge_pairs):
            key = tuple(sorted((int(face[first]), int(face[second]))))
            face_nodes[face_index, 3 + local_edge] = (
                len(pts) + edge_lookup[key]
            )
    face_points = pts[faces]
    face_area = 0.5 * np.linalg.norm(
        np.cross(
            face_points[:, 1] - face_points[:, 0],
            face_points[:, 2] - face_points[:, 0],
        ),
        axis=1,
    )
    barycentric = [
        np.asarray((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
        np.asarray((0.6, 0.2, 0.2)),
        np.asarray((0.2, 0.6, 0.2)),
        np.asarray((0.2, 0.2, 0.6)),
    ]
    weights = (-27.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0)
    local_force = np.zeros((len(faces), 6, 3), dtype=float)
    for lam, weight in zip(barycentric, weights):
        basis = np.asarray(
            (
                lam[0] * (2.0 * lam[0] - 1.0),
                lam[1] * (2.0 * lam[1] - 1.0),
                lam[2] * (2.0 * lam[2] - 1.0),
                4.0 * lam[0] * lam[1],
                4.0 * lam[0] * lam[2],
                4.0 * lam[1] * lam[2],
            ),
            dtype=float,
        )
        local_traction = np.einsum(
            "i,fia->fa", lam, traction[faces]
        )
        local_force += (
            float(weight)
            * face_area[:, None, None]
            * basis[None, :, None]
            * local_traction[:, None, :]
        )
    for local_node in range(6):
        np.add.at(result, face_nodes[:, local_node], local_force[:, local_node])
    result[boundary_vertices] += force[boundary_vertices]
    return result, edges


def implicit_tetra_taylor_hood_velocity_pressure(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    velocities_m_s: np.ndarray,
    external_forces_n: np.ndarray,
    viscosity_pa_s: float,
    density_kg_m3: float = 0.0,
    dt_s: float = 1.0,
    inertia_weight: float = 0.0,
    p2_external_forces_n: np.ndarray | None = None,
    p2_body_force_density_n_m3: np.ndarray | None = None,
    continuity_source_m3_s: np.ndarray | None = None,
    continuity_velocity_coupling: sparse.spmatrix | None = None,
    constrained_vertices: np.ndarray | None = None,
    prescribed_vertex_velocities_m_s: np.ndarray | None = None,
    tangent_vertex_directions: np.ndarray | None = None,
    p2_edge_constrained_vertices: np.ndarray | None = None,
    p2_constrained_edges: np.ndarray | None = None,
    enforce_exact_axisymmetry: bool = False,
    axisymmetry_tolerance_m: float = 1.0e-11,
    additional_stiffness_n_s_m: sparse.spmatrix | None = None,
    pressure_relative_regularization: float = 1.0e-12,
    rtol: float = 2.0e-7,
    maxiter: int = 1200,
    direct_solve_threshold: int = 90000,
) -> tuple[np.ndarray, PressureProjectionResult, ViscousVelocityResult]:
    """Creeping-flow P2/P1 Taylor-Hood solve on a tetrahedral liquid mesh.

    The quadratic velocity space resolves the parabolic thin-film mode that a
    one-layer P1/MINI mesh cannot represent.  Pressure remains continuous P1,
    so this is a parameter-free inf-sup-stable alternative to pressure
    stabilization; no experimental bridge data enter the operator.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    velocity = np.asarray(velocities_m_s, dtype=float)
    force = np.asarray(external_forces_n, dtype=float)
    if velocity.shape != pts.shape or force.shape != pts.shape:
        raise ValueError("velocities_m_s and external_forces_n must match points")
    viscous_momentum, divergence, edges, _local_nodes = (
        _tetra_p2_stiffness_and_divergence(
            pts, tet_arr, float(viscosity_pa_s)
        )
    )
    n_vertices = len(pts)
    n_velocity_nodes = n_vertices + len(edges)
    inertia = max(float(inertia_weight), 0.0)
    dt = max(float(dt_s), 1.0e-30)
    mass = sparse.csr_matrix(viscous_momentum.shape, dtype=float)
    if inertia > 0.0:
        mass, mass_edges = _tetra_p2_consistent_mass(
            pts, tet_arr, float(density_kg_m3)
        )
        if not np.array_equal(mass_edges, edges):
            raise RuntimeError("P2 mass and stiffness edge topologies differ")
    momentum = viscous_momentum + (inertia / dt) * mass
    if additional_stiffness_n_s_m is not None:
        extra = sparse.coo_matrix(additional_stiffness_n_s_m)
        if extra.shape == momentum.shape:
            momentum = momentum + extra.tocsr()
        elif extra.shape == (3 * n_vertices, 3 * n_vertices):
            expanded = sparse.coo_matrix(
                (extra.data, (extra.row, extra.col)),
                shape=(3 * n_velocity_nodes, 3 * n_velocity_nodes),
            ).tocsr()
            momentum = momentum + expanded
        else:
            raise ValueError(
                "additional_stiffness_n_s_m must act on either the P1 "
                "vertices or the complete P2 velocity space"
            )

    fixed_vertex = np.zeros(n_vertices, dtype=bool)
    if constrained_vertices is not None:
        values = np.asarray(constrained_vertices)
        if values.dtype == bool:
            fixed_vertex[:] = values
        else:
            fixed_vertex[np.asarray(values, dtype=int)] = True
    tangent_direction = np.zeros((n_velocity_nodes, 3), dtype=float)
    tangent_vertex = np.zeros(n_velocity_nodes, dtype=bool)
    if tangent_vertex_directions is not None:
        supplied_tangent = np.asarray(
            tangent_vertex_directions, dtype=float
        )
        if supplied_tangent.shape == (n_vertices, 3):
            tangent_direction[:n_vertices] = supplied_tangent
            vertex_tangent = (
                np.linalg.norm(supplied_tangent, axis=1) > 0.0
            )
            edge_tangent = np.all(vertex_tangent[edges], axis=1)
            if np.any(edge_tangent):
                edge_ids = n_vertices + np.flatnonzero(edge_tangent)
                tangent_direction[edge_ids] = (
                    supplied_tangent[edges[edge_tangent, 0]]
                    + supplied_tangent[edges[edge_tangent, 1]]
                )
        elif supplied_tangent.shape == (n_velocity_nodes, 3):
            tangent_direction[:] = supplied_tangent
        else:
            raise ValueError(
                "tangent_vertex_directions must match either the P1 "
                "vertices or all P2 velocity nodes"
            )
        tangent_norm = np.linalg.norm(tangent_direction, axis=1)
        tangent_vertex = tangent_norm > 0.0
        if np.any(fixed_vertex & tangent_vertex[:n_vertices]):
            raise ValueError(
                "a velocity vertex cannot be both fixed and tangent-constrained"
            )
        tangent_direction[tangent_vertex] /= tangent_norm[tangent_vertex, None]

    fixed_scalar = np.zeros(n_velocity_nodes, dtype=bool)
    fixed_scalar[:n_vertices] = fixed_vertex
    fixed_scalar[n_vertices:] = np.all(fixed_vertex[edges], axis=1)
    if p2_edge_constrained_vertices is not None:
        edge_constraint_vertex = np.asarray(
            p2_edge_constrained_vertices, dtype=bool
        )
        if edge_constraint_vertex.shape != (n_vertices,):
            raise ValueError(
                "p2_edge_constrained_vertices must match the P1 vertices"
            )
        fixed_scalar[n_vertices:] |= np.all(
            edge_constraint_vertex[edges], axis=1
        )
    if p2_constrained_edges is not None:
        edge_constraint = np.asarray(p2_constrained_edges, dtype=bool)
        if edge_constraint.shape != (len(edges),):
            raise ValueError(
                "p2_constrained_edges must contain one value per P2 edge"
            )
        fixed_scalar[n_vertices:] |= edge_constraint
    if np.any(fixed_scalar & tangent_vertex):
        raise ValueError(
            "a P2 velocity node cannot be both fixed and tangent-constrained"
        )
    fixed_dof = np.repeat(fixed_scalar, 3)
    prescribed_nodes = np.zeros((n_velocity_nodes, 3), dtype=float)
    if prescribed_vertex_velocities_m_s is not None:
        prescribed_vertices = np.asarray(
            prescribed_vertex_velocities_m_s, dtype=float
        )
        if prescribed_vertices.shape != (n_vertices, 3):
            raise ValueError(
                "prescribed_vertex_velocities_m_s must match the P1 vertices"
            )
        if np.any(np.linalg.norm(prescribed_vertices[~fixed_vertex], axis=1) > 0.0):
            raise ValueError(
                "nonzero prescribed velocity requires a constrained vertex"
            )
        prescribed_nodes[:n_vertices] = prescribed_vertices
        if len(edges):
            prescribed_nodes[n_vertices:] = 0.5 * (
                prescribed_vertices[edges[:, 0]]
                + prescribed_vertices[edges[:, 1]]
            )
    prescribed_dof = prescribed_nodes.reshape(-1)

    # Build a sparse velocity basis.  In the ordinary 3-D solve each free P2
    # node contributes three Cartesian columns (or one prescribed tangent
    # column at the P1 contact line).  The exact-axisymmetric solve instead
    # groups *all* P2 nodes, including tetra-edge midpoints, by cylindrical
    # (r,z) rings and gives each ring one radial and one axial coefficient.
    if bool(enforce_exact_axisymmetry):
        p2_points = np.vstack(
            (pts, 0.5 * (pts[edges[:, 0]] + pts[edges[:, 1]]))
        )
        velocity_basis, _p2_velocity_ring = (
            _exact_axisymmetric_velocity_transform(
                p2_points,
                fixed_dof,
                tangent_vertex_directions=tangent_direction,
                tolerance_m=float(axisymmetry_tolerance_m),
            )
        )
        reduced_velocity_dofs = int(velocity_basis.shape[1])
    else:
        basis_rows: list[int] = []
        basis_columns: list[int] = []
        basis_values: list[float] = []
        reduced_velocity_dofs = 0
        for node in range(n_velocity_nodes):
            if fixed_scalar[node]:
                continue
            if tangent_vertex[node]:
                for component in range(3):
                    value = float(tangent_direction[node, component])
                    if value != 0.0:
                        basis_rows.append(3 * node + component)
                        basis_columns.append(reduced_velocity_dofs)
                        basis_values.append(value)
                reduced_velocity_dofs += 1
            else:
                for component in range(3):
                    basis_rows.append(3 * node + component)
                    basis_columns.append(reduced_velocity_dofs)
                    basis_values.append(1.0)
                    reduced_velocity_dofs += 1
        velocity_basis = sparse.coo_matrix(
            (basis_values, (basis_rows, basis_columns)),
            shape=(3 * n_velocity_nodes, reduced_velocity_dofs),
        ).tocsr()
    reduced_momentum = (
        velocity_basis.T @ momentum @ velocity_basis
    ).tocsr()

    continuity_source = (
        np.zeros(n_vertices, dtype=float)
        if continuity_source_m3_s is None
        else np.asarray(continuity_source_m3_s, dtype=float)
    )
    if continuity_source.shape != (n_vertices,):
        raise ValueError(
            "continuity_source_m3_s must have one value per P1 pressure node"
        )
    if continuity_velocity_coupling is None:
        continuity_operator = divergence
    else:
        supplied_coupling = sparse.coo_matrix(
            continuity_velocity_coupling
        )
        if supplied_coupling.shape == (
            n_vertices,
            3 * n_vertices,
        ):
            continuity_coupling = sparse.coo_matrix(
                (
                    supplied_coupling.data,
                    (
                        supplied_coupling.row,
                        supplied_coupling.col,
                    ),
                ),
                shape=(n_vertices, 3 * n_velocity_nodes),
            ).tocsr()
        elif supplied_coupling.shape == divergence.shape:
            continuity_coupling = supplied_coupling.tocsr()
        else:
            raise ValueError(
                "continuity_velocity_coupling must act on either the P1 "
                "vertices or the complete P2 velocity space"
            )
        continuity_operator = (
            divergence - continuity_coupling
        ).tocsr()

    # Pin one pressure coefficient to remove only the constant null mode.  In
    # exact-axisymmetric mode the coefficients belong to P1 (r,z) rings, so
    # both pressure trial and test spaces are reduced during the solve.
    if bool(enforce_exact_axisymmetry):
        full_pressure_basis, pressure_ring = (
            _exact_axisymmetric_pressure_transform(
                pts, tolerance_m=float(axisymmetry_tolerance_m)
            )
        )
        pressure_coefficient_free = np.ones(
            full_pressure_basis.shape[1], dtype=bool
        )
        if pressure_coefficient_free.size:
            pressure_coefficient_free[int(pressure_ring[0])] = False
        pressure_basis = full_pressure_basis[
            :, pressure_coefficient_free
        ].tocsr()
        reduced_pressure_gradient = (
            pressure_basis.T @ divergence @ velocity_basis
        ).tocsr()
        reduced_continuity = (
            pressure_basis.T @ continuity_operator @ velocity_basis
        ).tocsr()
    else:
        pressure_free = np.ones(n_vertices, dtype=bool)
        pressure_free[0] = False
        pressure_basis = sparse.eye(n_vertices, format="csr")[
            :, pressure_free
        ].tocsr()
        reduced_pressure_gradient = (
            pressure_basis.T @ divergence @ velocity_basis
        ).tocsr()
        reduced_continuity = (
            pressure_basis.T @ continuity_operator @ velocity_basis
        ).tocsr()
    inverse_momentum_diagonal = 1.0 / np.maximum(
        np.abs(reduced_momentum.diagonal()), 1.0e-300
    )
    schur_diagonal = np.asarray(
        (
            reduced_continuity
            @ sparse.diags(inverse_momentum_diagonal, format="csr")
            @ reduced_pressure_gradient.T
        ).diagonal(),
        dtype=float,
    )
    pressure_scale = (
        float(np.mean(np.abs(schur_diagonal)))
        if schur_diagonal.size
        else 1.0
    )
    # P2/P1 is inf-sup stable and one pressure value is pinned above.  Any
    # diagonal pressure block would turn exact incompressibility into
    # ``B u = epsilon p`` and, in a thin film, that tiny algebraic leakage
    # integrates into a large false bridge volume.  Keep the saddle block
    # identically zero.
    pressure_regularization = 0.0
    pressure_block = sparse.csr_matrix(
        (
            pressure_basis.shape[1],
            pressure_basis.shape[1],
        ),
        dtype=float,
    )
    system = sparse.bmat(
        (
            (reduced_momentum, reduced_pressure_gradient.T),
            (reduced_continuity, pressure_block),
        ),
        format="csr",
    )
    if p2_external_forces_n is None:
        rhs_velocity = np.zeros((n_velocity_nodes, 3), dtype=float)
        rhs_velocity[:n_vertices] = force
    else:
        rhs_velocity = np.asarray(p2_external_forces_n, dtype=float).copy()
        if rhs_velocity.shape != (n_velocity_nodes, 3):
            raise ValueError(
                "p2_external_forces_n must contain one vector per P2 node: "
                f"got {rhs_velocity.shape}, expected {(n_velocity_nodes, 3)}, "
                f"vertices={n_vertices}, edges={len(edges)}"
            )
    if p2_body_force_density_n_m3 is not None:
        body_force_density = np.asarray(
            p2_body_force_density_n_m3,
            dtype=float,
        )
        if body_force_density.shape != (3,):
            raise ValueError(
                "p2_body_force_density_n_m3 must have shape (3,)"
            )
        if not np.all(np.isfinite(body_force_density)):
            raise ValueError(
                "p2_body_force_density_n_m3 must be finite"
            )
        unit_volume_mass, body_force_edges = _tetra_p2_consistent_mass(
            pts,
            tet_arr,
            1.0,
        )
        if not np.array_equal(body_force_edges, edges):
            raise RuntimeError(
                "P2 body-force and stiffness edge topologies differ"
            )
        rhs_velocity += np.asarray(
            unit_volume_mass
            @ np.tile(body_force_density, n_velocity_nodes)
        ).reshape((-1, 3))
    old_velocity = np.zeros((n_velocity_nodes, 3), dtype=float)
    old_velocity[:n_vertices] = velocity
    if len(edges):
        old_velocity[n_vertices:] = 0.5 * (
            velocity[edges[:, 0]] + velocity[edges[:, 1]]
        )
    momentum_rhs = (
        rhs_velocity.reshape(-1)
        + (inertia / dt) * np.asarray(mass @ old_velocity.reshape(-1)).ravel()
    )
    momentum_rhs_after_constraints = (
        momentum_rhs - np.asarray(momentum @ prescribed_dof).ravel()
    )
    reduced_velocity_rhs = np.asarray(
        velocity_basis.T @ momentum_rhs_after_constraints
    ).ravel()
    reduced_continuity_rhs = np.asarray(
        pressure_basis.T @ continuity_source
    ).ravel()
    if np.any(fixed_dof):
        reduced_continuity_rhs -= np.asarray(
            pressure_basis.T
            @ continuity_operator
            @ prescribed_dof
        ).ravel()
    rhs = np.concatenate((reduced_velocity_rhs, reduced_continuity_rhs))

    info = 0
    if system.shape[0] <= max(0, int(direct_solve_threshold)):
        solution = spla.spsolve(system.tocsc(), rhs)
        if not np.all(np.isfinite(solution)):
            solution = spla.lsmr(
                system,
                rhs,
                atol=float(rtol),
                btol=float(rtol),
                maxiter=int(maxiter),
            )[0]
            info = -1
    else:
        preconditioner_diagonal = np.concatenate(
            (
                np.maximum(np.abs(reduced_momentum.diagonal()), 1.0e-30),
                np.maximum(schur_diagonal, 1.0e-30),
            )
        )
        if continuity_velocity_coupling is None:
            solution, info = spla.minres(
                system,
                rhs,
                M=sparse.diags(
                    1.0 / preconditioner_diagonal, format="csr"
                ),
                rtol=float(rtol),
                maxiter=int(maxiter),
            )
        else:
            solution, info = spla.gmres(
                system,
                rhs,
                M=sparse.diags(
                    1.0 / preconditioner_diagonal, format="csr"
                ),
                rtol=float(rtol),
                atol=0.0,
                restart=80,
                maxiter=int(maxiter),
            )
        if info != 0 or not np.all(np.isfinite(solution)):
            solution = spla.lsmr(
                system,
                rhs,
                atol=float(rtol),
                btol=float(rtol),
                maxiter=int(maxiter),
            )[0]
            info = -abs(int(info)) if int(info) else 0
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Taylor-Hood tetra solve returned non-finite values")

    free_count = int(reduced_velocity_dofs)
    velocity_all = prescribed_dof + np.asarray(
        velocity_basis @ solution[:free_count]
    ).ravel()
    pressure = np.asarray(
        pressure_basis @ solution[free_count:], dtype=float
    ).ravel()
    vertex_velocity = velocity_all[: 3 * n_vertices].reshape((-1, 3))
    edge_velocity = velocity_all[3 * n_vertices :].reshape((-1, 3))
    continuity = (
        continuity_operator @ velocity_all - continuity_source
    )
    reduced_continuity = np.asarray(
        pressure_basis.T @ continuity, dtype=float
    ).ravel()
    continuity_diagnostic = (
        reduced_continuity
        if bool(enforce_exact_axisymmetry)
        else continuity
    )
    momentum_residual = (
        momentum @ velocity_all
        + divergence.T @ pressure
        - momentum_rhs
    )
    pressure_result = PressureProjectionResult(
        pressure_pa=pressure,
        residual_before_m3_s=float(
            np.sqrt(
                np.mean(
                    (
                        continuity_operator[:, : 3 * n_vertices]
                        @ velocity.reshape(-1)
                        - continuity_source
                    )
                    ** 2
                )
            )
        ),
        residual_after_m3_s=float(
            np.sqrt(np.mean(continuity_diagnostic**2))
            if continuity_diagnostic.size
            else 0.0
        ),
        pressure_l2_pa=float(np.sqrt(np.mean(pressure**2))),
        pressure_linf_pa=float(np.max(np.abs(pressure))),
        solver_info=int(info),
        p2_edges=edges,
        p2_edge_velocity_m_s=edge_velocity,
    )
    viscous_result = ViscousVelocityResult(
        residual_l2_n=float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        velocity_basis.T @ momentum_residual
                    ).ravel()
                    ** 2
                )
            )
        ),
        velocity_linf_m_s=float(
            max(
                np.max(np.linalg.norm(vertex_velocity, axis=1)),
                np.max(np.linalg.norm(edge_velocity, axis=1))
                if len(edge_velocity)
                else 0.0,
            )
        ),
        solver_info=int(info),
        operator_diagonal_n_s_m=np.asarray(
            momentum.diagonal()[: 3 * n_vertices], dtype=float
        ),
    )
    return vertex_velocity, pressure_result, viscous_result


def implicit_tetra_stokes_p0_velocity_pressure(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    velocities_m_s: np.ndarray,
    external_forces_n: np.ndarray,
    masses_kg: np.ndarray,
    viscosity_pa_s: float | np.ndarray,
    viscous_form: str = "vector_laplacian",
    dt_s: float,
    inertia_weight: float = 1.0,
    target_tet_volumes_m3: np.ndarray | None = None,
    constrained_vertices: np.ndarray | None = None,
    enforce_exact_axisymmetry: bool = False,
    axisymmetry_tolerance_m: float = 1.0e-11,
    tangent_vertex_directions: np.ndarray | None = None,
    additional_stiffness_n_s_m: sparse.spmatrix | None = None,
    pressure_relative_regularization: float = 1.0e-12,
    rtol: float = 1.0e-8,
    maxiter: int = 1000,
) -> tuple[np.ndarray, PressureProjectionResult, ViscousVelocityResult]:
    """Solve PR35 momentum with one PR33 pressure unknown per tetrahedron.

    The continuity row is the exact Lagrangian tetra-volume rate
    ``B u=(V_target-V)/dt``.  Omitting ``target_tet_volumes_m3`` recovers
    ``B u=0``.

    ``inertia_weight=1`` gives the backward-Euler transient equation
    ``(M/dt+K)u+B.T*p=M*u_old/dt+F``.  ``inertia_weight=0`` gives its
    quasi-static Stokes limit without changing the pressure or force
    operators.

    Unlike an equal-order vertex-pressure formulation, this operator does not
    require a mesh-dependent pressure-Laplacian stabilization that can absorb
    capillary suction in a very thin film.  The tiny diagonal term only fixes
    redundant pressure modes; it is scaled from ``B diag(A)^-1 B.T`` and tends
    to zero without changing the velocity constraint.
    """

    points_arr = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    velocity = np.asarray(velocities_m_s, dtype=float)
    force = np.asarray(external_forces_n, dtype=float)
    masses = np.asarray(masses_kg, dtype=float)
    if velocity.shape != points_arr.shape or force.shape != points_arr.shape:
        raise ValueError("velocities_m_s and external_forces_n must match points.")
    if masses.shape != (len(points_arr),):
        raise ValueError("masses_kg must have one entry per vertex.")

    dt = max(float(dt_s), 1.0e-30)
    inertia = max(float(inertia_weight), 0.0)
    mass_diagonal = np.repeat(
        inertia * np.maximum(masses, 0.0) / dt,
        3,
    )
    viscous_stiffness = tetra_viscous_stiffness_matrix(
        points_arr,
        tet_arr,
        viscosity_pa_s,
        viscous_form=viscous_form,
    )
    momentum = viscous_stiffness + sparse.diags(
        mass_diagonal, format="csr"
    )
    if additional_stiffness_n_s_m is not None:
        extra = sparse.csr_matrix(additional_stiffness_n_s_m)
        if extra.shape != momentum.shape:
            raise ValueError(
                "additional_stiffness_n_s_m must have shape (3*n_vertices, 3*n_vertices)."
            )
        momentum = momentum + extra

    constrained = constrained_dof_mask(len(points_arr), constrained_vertices)
    free = ~constrained
    volumes, volume_gradient = tet_volume_matrix_sparse(points_arr, tet_arr)
    target_volumes = (
        volumes
        if target_tet_volumes_m3 is None
        else np.asarray(target_tet_volumes_m3, dtype=float)
    )
    if target_volumes.shape != volumes.shape:
        raise ValueError(
            "target_tet_volumes_m3 must have one value per tetrahedron"
        )
    target_volume_rate = (target_volumes - volumes) / dt
    divergence_free = volume_gradient[:, free].tocsr()
    velocity_transform = None
    if bool(enforce_exact_axisymmetry):
        velocity_transform, _ring = _exact_axisymmetric_velocity_transform(
            points_arr,
            constrained,
            tangent_vertex_directions=tangent_vertex_directions,
            tolerance_m=float(axisymmetry_tolerance_m),
        )
        velocity_transform_free = velocity_transform[free, :].tocsr()
        reduced_momentum = (
            velocity_transform_free.T
            @ momentum[free][:, free]
            @ velocity_transform_free
        ).tocsr()
        reduced_divergence = (
            divergence_free @ velocity_transform_free
        ).tocsr()
    else:
        reduced_momentum = momentum[free][:, free].tocsr()
        reduced_divergence = divergence_free
    inverse_diagonal = sparse.diags(
        1.0 / np.maximum(np.abs(reduced_momentum.diagonal()), 1.0e-300),
        format="csr",
    )
    schur_diagonal = np.asarray(
        (
            reduced_divergence
            @ inverse_diagonal
            @ reduced_divergence.T
        ).diagonal(),
        dtype=float,
    )
    pressure_scale = float(np.mean(np.abs(schur_diagonal))) if schur_diagonal.size else 0.0
    pressure_regularization = max(
        float(pressure_relative_regularization) * max(pressure_scale, 1.0e-30),
        1.0e-30,
    )
    pressure_block = -pressure_regularization * sparse.eye(
        len(tet_arr), format="csr"
    )
    system = sparse.bmat(
        (
            (reduced_momentum, reduced_divergence.T),
            (reduced_divergence, pressure_block),
        ),
        format="csr",
    )
    full_momentum_rhs = (
        mass_diagonal * velocity.reshape(-1) + force.reshape(-1)
    )
    momentum_rhs = full_momentum_rhs[free]
    if velocity_transform is not None:
        momentum_rhs = np.asarray(
            velocity_transform_free.T @ momentum_rhs,
            dtype=float,
        ).reshape(-1)
    rhs = np.concatenate((momentum_rhs, target_volume_rate))

    info = 0
    if system.shape[0] <= 100000:
        solution = spla.spsolve(system.tocsc(), rhs)
    else:
        diagonal = np.maximum(np.abs(system.diagonal()), 1.0e-30)
        solution = spla.lsmr(
            system,
            rhs,
            atol=float(rtol),
            btol=float(rtol),
            maxiter=int(maxiter),
            x0=np.zeros(system.shape[1], dtype=float),
        )[0]
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Monolithic P0 tetra Stokes solve returned non-finite values")

    velocity_unknowns = int(reduced_momentum.shape[0])
    if velocity_transform is not None:
        velocity_vec = np.asarray(
            velocity_transform @ solution[:velocity_unknowns],
            dtype=float,
        ).reshape(-1)
    else:
        velocity_vec = np.zeros(3 * len(points_arr), dtype=float)
        velocity_vec[free] = solution[:velocity_unknowns]
    pressure = np.asarray(solution[velocity_unknowns:], dtype=float)
    projected = velocity_vec.reshape(points_arr.shape)
    continuity_before = (
        volume_gradient @ velocity.reshape(-1) - target_volume_rate
    )
    continuity_after = volume_gradient @ velocity_vec - target_volume_rate
    momentum_residual = (
        momentum @ velocity_vec
        + volume_gradient.T @ pressure
        - full_momentum_rhs
    )
    pressure_force = -np.asarray(
        volume_gradient.T @ pressure,
        dtype=float,
    ).reshape(points_arr.shape)
    cauchy_force = -np.asarray(
        viscous_stiffness @ velocity_vec,
        dtype=float,
    ).reshape(points_arr.shape)
    pressure_result = PressureProjectionResult(
        pressure_pa=pressure,
        residual_before_m3_s=float(np.sqrt(np.mean(continuity_before**2)))
        if continuity_before.size
        else 0.0,
        residual_after_m3_s=float(np.sqrt(np.mean(continuity_after**2)))
        if continuity_after.size
        else 0.0,
        pressure_l2_pa=float(np.sqrt(np.mean(pressure**2))) if pressure.size else 0.0,
        pressure_linf_pa=float(np.max(np.abs(pressure))) if pressure.size else 0.0,
        solver_info=int(info),
        pressure_force_n=pressure_force,
    )
    viscous_result = ViscousVelocityResult(
        residual_l2_n=float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(
                            velocity_transform_free.T
                            @ momentum_residual[free],
                            dtype=float,
                        ).reshape(-1)
                        if velocity_transform is not None
                        else momentum_residual[free]
                    )
                    ** 2
                )
            )
        )
        if np.any(free)
        else 0.0,
        velocity_linf_m_s=float(np.max(np.linalg.norm(projected, axis=1))),
        solver_info=int(info),
        operator_diagonal_n_s_m=np.asarray(momentum.diagonal(), dtype=float),
        cauchy_force_n=cauchy_force,
        constraint_force_n=momentum_residual.reshape(points_arr.shape),
    )
    return projected, pressure_result, viscous_result


def tet_cell_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return absolute tetrahedron volumes for an indexed volume mesh."""

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int)
    if tet_arr.size == 0:
        return np.zeros(0, dtype=float)
    p0 = pts[tet_arr[:, 0]]
    p1 = pts[tet_arr[:, 1]]
    p2 = pts[tet_arr[:, 2]]
    p3 = pts[tet_arr[:, 3]]
    triple = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
    return np.abs(triple) / 6.0


def tet_volume_matrix_sparse(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Return tet volumes and sparse volume-gradient matrix ``B``.

    ``B[t, 3*i:3*i+3]`` stores ``dV_t/dx_i``.  Therefore ``B @ u`` is the
    discrete Lagrangian volume rate of each tetrahedron.  This is the same
    pressure/continuity operator used by PR33, represented sparsely so it can
    be used by large liquid-bridge meshes.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int)
    n_tets = tet_arr.shape[0]
    n_vertices = pts.shape[0]
    if n_tets == 0:
        return np.zeros(0, dtype=float), sparse.csr_matrix((0, 3 * n_vertices), dtype=float)

    tet_pts = pts[tet_arr]
    p0 = tet_pts[:, 0]
    p1 = tet_pts[:, 1]
    p2 = tet_pts[:, 2]
    p3 = tet_pts[:, 3]
    triple = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
    sign = np.where(triple >= 0.0, 1.0, -1.0)
    volumes = np.abs(triple) / 6.0
    g1 = sign[:, None] * np.cross(p2 - p0, p3 - p0) / 6.0
    g2 = sign[:, None] * np.cross(p3 - p0, p1 - p0) / 6.0
    g3 = sign[:, None] * np.cross(p1 - p0, p2 - p0) / 6.0
    g0 = -(g1 + g2 + g3)
    grads = np.stack((g0, g1, g2, g3), axis=1)

    rows = np.repeat(np.arange(n_tets, dtype=int), 12)
    component_offsets = np.arange(3, dtype=int)
    cols = (3 * tet_arr[:, :, None] + component_offsets[None, None, :]).reshape(-1)
    data = grads.reshape(-1)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_tets, 3 * n_vertices)).tocsr()
    return volumes, matrix


def lump_tet_masses(
    n_vertices: int,
    tets: np.ndarray,
    tet_volumes: np.ndarray,
    density_kg_m3: float,
    floor_kg: float = 1.0e-18,
) -> np.ndarray:
    """Barycentric tetra-volume mass lumping onto mesh vertices."""

    masses = np.zeros(int(n_vertices), dtype=float)
    for tet, volume in zip(np.asarray(tets, dtype=int), np.asarray(tet_volumes, dtype=float)):
        share = 0.25 * float(density_kg_m3) * float(volume)
        for vertex_idx in tet:
            masses[int(vertex_idx)] += share
    return np.maximum(masses, float(floor_kg))


def constrained_dof_mask(n_vertices: int, constrained_vertices: np.ndarray | None = None) -> np.ndarray:
    """Return a boolean mask over flattened xyz degrees of freedom."""

    mask = np.zeros(3 * int(n_vertices), dtype=bool)
    if constrained_vertices is None:
        return mask
    vertices = np.asarray(constrained_vertices, dtype=bool)
    if vertices.size != n_vertices:
        raise ValueError("constrained_vertices must have one entry per vertex.")
    for component in range(3):
        mask[component::3] = vertices
    return mask


def sphere_tangent_projectors(
    points: np.ndarray,
    vertices: np.ndarray,
    *,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
) -> np.ndarray:
    """Return per-vertex xyz projectors with sphere-tangent blocks on ``vertices``.

    All unselected vertices receive the identity projector.  The selected
    blocks are ``I - n n^T`` and therefore retain both tangent directions while
    removing penetration into a fixed sphere.  These blocks can be supplied
    directly to :func:`sparse_pressure_projection`.
    """

    pts = np.asarray(points, dtype=float)
    projectors = np.broadcast_to(np.eye(3, dtype=float), (pts.shape[0], 3, 3)).copy()
    idx = np.asarray(vertices, dtype=int)
    if idx.size == 0:
        return projectors
    if np.any(idx < 0) or np.any(idx >= pts.shape[0]):
        raise ValueError("vertices contains an out-of-range vertex index.")
    center = np.asarray(
        [0.0, 0.0, float(sphere_tip_z_m) + float(sphere_radius_m)],
        dtype=float,
    )
    normal = pts[idx] - center[None, :]
    norm = np.linalg.norm(normal, axis=1)
    valid = norm > 1.0e-30
    normal[valid] /= norm[valid, None]
    normal[~valid] = 0.0
    projectors[idx] = np.eye(3, dtype=float)[None, :, :] - np.einsum(
        "ni,nj->nij", normal, normal
    )
    return projectors


def _mobility_matrix(
    masses_kg: np.ndarray,
    constrained_vertices: np.ndarray | None,
    vertex_projectors: np.ndarray | None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build the block vertex mobility used by the PR33 pressure projection."""

    masses = np.asarray(masses_kg, dtype=float)
    n_vertices = masses.shape[0]
    constrained = constrained_dof_mask(n_vertices, constrained_vertices)
    if vertex_projectors is None:
        inv_mass = np.repeat(1.0 / np.maximum(masses, 1.0e-300), 3)
        inv_mass[constrained] = 0.0
        return sparse.diags(inv_mass, format="csr"), constrained

    projectors = np.asarray(vertex_projectors, dtype=float)
    if projectors.shape != (n_vertices, 3, 3):
        raise ValueError("vertex_projectors must have shape (n_vertices, 3, 3).")
    if not np.all(np.isfinite(projectors)):
        raise ValueError("vertex_projectors must contain only finite values.")
    # A symmetric mobility is required for the pressure Schur complement.
    projectors = 0.5 * (projectors + np.swapaxes(projectors, 1, 2))
    blocks = projectors / np.maximum(masses, 1.0e-300)[:, None, None]
    if constrained_vertices is not None:
        fixed = np.asarray(constrained_vertices, dtype=bool)
        if fixed.size != n_vertices:
            raise ValueError("constrained_vertices must have one entry per vertex.")
        blocks[fixed] = 0.0
    rows = np.repeat(np.arange(3 * n_vertices, dtype=int).reshape(n_vertices, 3), 3, axis=1).reshape(-1)
    cols = np.tile(np.arange(3 * n_vertices, dtype=int).reshape(n_vertices, 3), (1, 3)).reshape(-1)
    mobility = sparse.coo_matrix(
        (blocks.reshape(-1), (rows, cols)),
        shape=(3 * n_vertices, 3 * n_vertices),
    ).tocsr()
    mobility.eliminate_zeros()
    return mobility, constrained


def _tet_shape_gradients(
    points: np.ndarray,
    tets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid-tet mask, volumes, and linear shape-function gradients."""

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    if tet_arr.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0), np.zeros((0, 4, 3))
    tet_pts = pts[tet_arr]
    p0 = tet_pts[:, 0]
    p1 = tet_pts[:, 1]
    p2 = tet_pts[:, 2]
    p3 = tet_pts[:, 3]
    triple = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
    sign = np.where(triple >= 0.0, 1.0, -1.0)
    volumes = np.abs(triple) / 6.0
    volume_grads = np.empty((tet_arr.shape[0], 4, 3), dtype=float)
    volume_grads[:, 1] = sign[:, None] * np.cross(p2 - p0, p3 - p0) / 6.0
    volume_grads[:, 2] = sign[:, None] * np.cross(p3 - p0, p1 - p0) / 6.0
    volume_grads[:, 3] = sign[:, None] * np.cross(p1 - p0, p2 - p0) / 6.0
    volume_grads[:, 0] = -np.sum(volume_grads[:, 1:], axis=1)
    valid = np.isfinite(volumes) & (volumes > 1.0e-30)
    gradients = np.zeros_like(volume_grads)
    gradients[valid] = volume_grads[valid] / volumes[valid, None, None]
    return valid, volumes, gradients


def _normalize_viscous_form(viscous_form: str) -> str:
    """Return the canonical tetra viscous bilinear-form name."""

    form = str(viscous_form).strip().lower()
    if form in {
        "vector_laplacian",
        "laplacian",
        "component_gradient",
        "grad_grad",
    }:
        return "vector_laplacian"
    if form in {
        "symmetric_gradient",
        "cauchy",
        "newtonian_cauchy",
        "strain_rate",
        "2d_d",
    }:
        return "symmetric_gradient"
    raise ValueError(
        "viscous_form must be 'vector_laplacian' or 'symmetric_gradient'"
    )


def tetra_viscous_stiffness_matrix(
    points: np.ndarray,
    tets: np.ndarray,
    viscosity_pa_s: float | np.ndarray,
    *,
    viscous_form: str = "vector_laplacian",
) -> sparse.csr_matrix:
    """Assemble a linear-tetra viscous operator.

    ``vector_laplacian`` preserves the historical component-wise form
    ``mu*integral(grad(u):grad(v))``.  ``symmetric_gradient`` assembles the
    Newtonian Cauchy-stress form ``2*mu*integral(D(u):D(v))``.  The two bulk
    operators are equivalent only after using exact incompressibility together
    with boundary conditions that remove the integration-by-parts boundary
    term; they are not generally interchangeable on a traction-free surface.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    valid, volumes, gradients = _tet_shape_gradients(pts, tet_arr)
    n_vertices = pts.shape[0]
    if not np.any(valid):
        return sparse.csr_matrix((3 * n_vertices, 3 * n_vertices), dtype=float)
    tet_nodes = tet_arr[valid]
    tet_volumes = volumes[valid]
    tet_gradients = gradients[valid]
    mu = np.asarray(viscosity_pa_s, dtype=float)
    if mu.ndim == 0:
        mu_tet = np.full(tet_nodes.shape[0], float(mu), dtype=float)
    elif mu.shape == (tet_arr.shape[0],):
        mu_tet = mu[valid]
    else:
        raise ValueError("viscosity_pa_s must be scalar or one value per tetrahedron.")
    form = _normalize_viscous_form(viscous_form)
    gradient_dot = np.einsum("tia,tja->tij", tet_gradients, tet_gradients)
    if form == "vector_laplacian":
        local = (
            mu_tet[:, None, None]
            * tet_volumes[:, None, None]
            * gradient_dot
        )
        rows = np.repeat(tet_nodes, 4, axis=1).reshape(-1)
        cols = np.tile(tet_nodes, (1, 4)).reshape(-1)
        scalar = sparse.coo_matrix(
            (local.reshape(-1), (rows, cols)),
            shape=(n_vertices, n_vertices),
        ).tocsr()
        return sparse.kron(
            scalar, sparse.eye(3, format="csr"), format="csr"
        )

    # With the node-major component ordering (i,a),
    #
    #   2 D(N_i e_a):D(N_j e_b)
    #     = delta_ab grad(N_i).grad(N_j)
    #       + d_b(N_i) d_a(N_j).
    #
    # This directly assembles the full 12-by-12 elemental Cauchy-stress block
    # and therefore retains the component coupling absent from the historical
    # vector Laplacian.
    # Assemble one component block at a time.  Materializing the full
    # ``(n_tet,4,3,4,3)`` tensor plus matching COO indices has a large transient
    # memory cost on the refined bridge meshes, while the nine 4-by-4 blocks
    # below are algebraically identical.
    weighted_volume = mu_tet * tet_volumes
    node_rows = np.repeat(tet_nodes, 4, axis=1).reshape(-1)
    node_columns = np.tile(tet_nodes, (1, 4)).reshape(-1)
    matrix = sparse.csr_matrix(
        (3 * n_vertices, 3 * n_vertices), dtype=float
    )
    for component_a in range(3):
        for component_b in range(3):
            local = (
                tet_gradients[:, :, component_b, None]
                * tet_gradients[:, None, :, component_a]
            )
            if component_a == component_b:
                local = local + gradient_dot
            local *= weighted_volume[:, None, None]
            block = sparse.coo_matrix(
                (
                    local.reshape(-1),
                    (
                        3 * node_rows + component_a,
                        3 * node_columns + component_b,
                    ),
                ),
                shape=(3 * n_vertices, 3 * n_vertices),
            ).tocsr()
            matrix = matrix + block
    matrix.eliminate_zeros()
    return matrix


def tetra_depth_averaged_lubrication_stiffness(
    points: np.ndarray,
    tets: np.ndarray,
    gap_height_m: float | np.ndarray,
    viscosity_pa_s: float,
    *,
    mobility_coefficient: float = 3.0,
    substrate_normal: np.ndarray = np.asarray((0.0, 0.0, 1.0)),
    velocity_order: int = 1,
) -> sparse.csr_matrix:
    """Assemble a gap-dependent depth-averaged lubrication resistance.

    For a no-slip substrate below a shear-free liquid interface,

    ``q = -(h**3/(3*mu))*grad_parallel(p)``

    gives the in-plane Brinkman coefficient ``beta=3*mu/h**2``. This
    routine integrates ``beta*u_parallel.v_parallel`` with a consistent P1
    or P2 tetra mass matrix. Its result has units N s/m and can therefore be
    added directly to a PR35 momentum stiffness matrix as ``K_lub(h)``.

    ``mobility_coefficient`` is exposed so another independently derived
    boundary condition can replace the shear-free value 3. The gap is never
    clipped: non-positive or non-finite values are rejected.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    n_vertices = len(pts)
    shape = (3 * n_vertices, 3 * n_vertices)
    if tet_arr.size == 0:
        return sparse.csr_matrix(shape, dtype=float)
    if np.any(tet_arr < 0) or np.any(tet_arr >= n_vertices):
        raise ValueError("tets contains a vertex outside points")

    viscosity = float(viscosity_pa_s)
    coefficient = float(mobility_coefficient)
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity_pa_s must be positive and finite")
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError(
            "mobility_coefficient must be positive and finite"
        )

    gap = np.asarray(gap_height_m, dtype=float)
    if gap.ndim == 0:
        gap_tet = np.full(len(tet_arr), float(gap), dtype=float)
    elif gap.shape == (len(tet_arr),):
        gap_tet = gap
    else:
        raise ValueError(
            "gap_height_m must be scalar or one value per tetrahedron"
        )
    if np.any(~np.isfinite(gap_tet)) or np.any(gap_tet <= 0.0):
        raise ValueError(
            "gap_height_m must remain positive and finite; no gap cap is "
            "applied by the lubrication operator"
        )

    normal = np.asarray(substrate_normal, dtype=float).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 0.0:
        raise ValueError("substrate_normal must be finite and nonzero")
    normal /= normal_norm
    parallel_projector = np.eye(3, dtype=float) - np.outer(normal, normal)

    valid, volumes, _gradients = _tet_shape_gradients(pts, tet_arr)
    if not np.any(valid):
        return sparse.csr_matrix(shape, dtype=float)
    beta_all = coefficient * viscosity / gap_tet**2
    if int(velocity_order) == 2:
        isotropic_mass, _edges = _tetra_p2_consistent_mass(
            pts,
            tet_arr,
            beta_all,
        )
        n_velocity_nodes = isotropic_mass.shape[0] // 3
        projector = sparse.kron(
            sparse.eye(n_velocity_nodes, format="csr"),
            sparse.csr_matrix(parallel_projector),
            format="csr",
        )
        matrix = (
            projector @ isotropic_mass @ projector
        ).tocsr()
        matrix.eliminate_zeros()
        return matrix
    if int(velocity_order) != 1:
        raise ValueError("velocity_order must be 1 or 2")

    active_tets = tet_arr[valid]
    beta = beta_all[valid]
    local_mass = (
        beta[:, None, None]
        * volumes[valid, None, None]
        * (
            np.ones((4, 4), dtype=float)
            + np.eye(4, dtype=float)[None, :, :]
        )
        / 20.0
    )
    node_rows = np.repeat(active_tets, 4, axis=1).reshape(-1)
    node_columns = np.tile(active_tets, (1, 4)).reshape(-1)
    matrix = sparse.csr_matrix(shape, dtype=float)
    for component_a in range(3):
        for component_b in range(3):
            projector_value = float(
                parallel_projector[component_a, component_b]
            )
            if abs(projector_value) <= 1.0e-15:
                continue
            block = sparse.coo_matrix(
                (
                    projector_value * local_mass.reshape(-1),
                    (
                        3 * node_rows + component_a,
                        3 * node_columns + component_b,
                    ),
                ),
                shape=shape,
            ).tocsr()
            matrix = matrix + block
    matrix.sum_duplicates()
    return matrix


def tet_vertex_divergence_matrix(
    points: np.ndarray,
    tets: np.ndarray,
) -> sparse.csr_matrix:
    """Assemble the weak P1 velocity-to-vertex-pressure divergence matrix.

    The previous one-pressure-per-tetra projection is an unstable P1/P0 pair
    on these unstructured meshes and develops checkerboard pressure modes.
    Vertex pressure gives a continuous P1 projection with one constant null
    mode, removed by the projection regularization.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    valid, volumes, gradients = _tet_shape_gradients(pts, tet_arr)
    n_vertices = len(pts)
    if not np.any(valid):
        return sparse.csr_matrix((n_vertices, 3 * n_vertices), dtype=float)
    nodes = tet_arr[valid]
    local = (
        volumes[valid, None, None, None]
        * 0.25
        * np.broadcast_to(
            gradients[valid, None, :, :],
            (len(nodes), 4, 4, 3),
        )
    )
    rows = np.broadcast_to(nodes[:, :, None, None], local.shape).reshape(-1)
    cols = np.broadcast_to(
        3 * nodes[:, None, :, None] + np.arange(3)[None, None, None, :],
        local.shape,
    ).reshape(-1)
    return sparse.coo_matrix(
        (local.reshape(-1), (rows, cols)),
        shape=(n_vertices, 3 * n_vertices),
    ).tocsr()


def tetra_boundary_surface_pressure_forces(
    points: np.ndarray,
    tets: np.ndarray,
    boundary_faces: np.ndarray,
    pressure_pa: float | np.ndarray,
) -> np.ndarray:
    """Assemble outward P1 pressure forces on tetrahedral boundary faces.

    For face ``f=(i,j,k)``, this is the same triangle integration used by
    Case 12 of PR35:

    ``F_i^f = F_j^f = F_k^f = p_f * A_f / 3``,

    where ``A_f`` is oriented outward from the adjacent liquid tetrahedron.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    faces = np.asarray(boundary_faces, dtype=int).reshape((-1, 3))
    force = np.zeros_like(pts)
    if len(faces) == 0:
        return force
    if len(tet_arr) == 0:
        raise ValueError("Boundary pressure force requires tetrahedra")
    if np.any(faces < 0) or np.any(faces >= len(pts)):
        raise ValueError("boundary_faces contain an invalid vertex index")

    local_faces = np.stack(
        (
            tet_arr[:, (1, 2, 3)],
            tet_arr[:, (0, 2, 3)],
            tet_arr[:, (0, 1, 3)],
            tet_arr[:, (0, 1, 2)],
        ),
        axis=1,
    ).reshape((-1, 3))
    local_keys = np.sort(local_faces, axis=1)
    face_keys = np.sort(faces, axis=1)
    key_dtype = np.dtype((np.void, local_keys.dtype.itemsize * 3))
    local_view = np.ascontiguousarray(local_keys).view(key_dtype).reshape(-1)
    face_view = np.ascontiguousarray(face_keys).view(key_dtype).reshape(-1)
    order = np.argsort(local_view)
    sorted_view = local_view[order]
    location = np.searchsorted(sorted_view, face_view)
    if np.any(location >= len(sorted_view)) or np.any(
        sorted_view[np.minimum(location, len(sorted_view) - 1)] != face_view
    ):
        raise ValueError("A pressure boundary face has no adjacent tetrahedron")
    adjacent_tet = order[location] // 4

    triangle = pts[faces]
    area_vector = 0.5 * np.cross(
        triangle[:, 1] - triangle[:, 0],
        triangle[:, 2] - triangle[:, 0],
    )
    face_centroid = np.mean(triangle, axis=1)
    tet_centroid = np.mean(pts[tet_arr[adjacent_tet]], axis=1)
    reverse = np.sum(
        area_vector * (face_centroid - tet_centroid), axis=1
    ) < 0.0
    area_vector[reverse] *= -1.0

    pressure = np.asarray(pressure_pa, dtype=float)
    if pressure.ndim == 0:
        pressure = np.full(len(faces), float(pressure), dtype=float)
    elif pressure.shape != (len(faces),):
        raise ValueError(
            "pressure_pa must be scalar or one value per boundary face"
        )
    if not np.all(np.isfinite(pressure)):
        raise ValueError("pressure_pa must be finite")
    local_force = pressure[:, None] * area_vector / 3.0
    for local_vertex in range(3):
        np.add.at(force, faces[:, local_vertex], local_force)
    return force


def tetra_pressure_stabilization_matrix(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    viscosity_pa_s: float,
    coefficient: float = 1.0 / 12.0,
) -> sparse.csr_matrix:
    """Brezzi-Pitkaranta pressure stabilization for equal-order tetrahedra."""

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    valid, volumes, gradients = _tet_shape_gradients(pts, tet_arr)
    n_vertices = len(pts)
    if not np.any(valid):
        return sparse.csr_matrix((n_vertices, n_vertices), dtype=float)
    nodes = tet_arr[valid]
    volume = volumes[valid]
    grad = gradients[valid]
    # Use the minimum tetra altitude, not a volume-equivalent isotropic size.
    # The bridge-film mesh is deliberately anisotropic (micron-scale normal
    # layers and much longer tangential edges).  A cube-root volume length
    # over-stabilizes pressure by orders of magnitude in that setting and can
    # suppress the physical capillary-pressure gradient.  The minimum altitude
    # is the directional resolution relevant to the pressure mode.
    tet_points = pts[nodes]
    face_area = np.stack(
        (
            0.5
            * np.linalg.norm(
                np.cross(
                    tet_points[:, 2] - tet_points[:, 1],
                    tet_points[:, 3] - tet_points[:, 1],
                ),
                axis=1,
            ),
            0.5
            * np.linalg.norm(
                np.cross(
                    tet_points[:, 2] - tet_points[:, 0],
                    tet_points[:, 3] - tet_points[:, 0],
                ),
                axis=1,
            ),
            0.5
            * np.linalg.norm(
                np.cross(
                    tet_points[:, 1] - tet_points[:, 0],
                    tet_points[:, 3] - tet_points[:, 0],
                ),
                axis=1,
            ),
            0.5
            * np.linalg.norm(
                np.cross(
                    tet_points[:, 1] - tet_points[:, 0],
                    tet_points[:, 2] - tet_points[:, 0],
                ),
                axis=1,
            ),
        ),
        axis=1,
    )
    element_length = 3.0 * volume / np.maximum(
        np.max(face_area, axis=1), 1.0e-300
    )
    scale = (
        float(coefficient)
        * element_length**2
        / max(float(viscosity_pa_s), 1.0e-30)
    )
    local = (
        scale[:, None, None]
        * volume[:, None, None]
        * np.einsum("tia,tja->tij", grad, grad)
    )
    rows = np.repeat(nodes, 4, axis=1).reshape(-1)
    cols = np.tile(nodes, (1, 4)).reshape(-1)
    return sparse.coo_matrix(
        (local.reshape(-1), (rows, cols)),
        shape=(n_vertices, n_vertices),
    ).tocsr()


def tetra_mini_pressure_stabilization_matrix(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    viscosity_pa_s: float,
    density_kg_m3: float,
    dt_s: float,
    viscous_form: str = "vector_laplacian",
) -> sparse.csr_matrix:
    """Static-condensation pressure block for the tetrahedral MINI element.

    Each tetra receives the normalized interior velocity bubble
    ``b=256*lambda0*lambda1*lambda2*lambda3``.  Eliminating its three velocity
    coefficients gives ``B_b A_b^-1 B_b.T`` on the continuous P1 pressure
    space.  All integrals are analytic barycentric monomial integrals, so the
    stabilization is parameter-free and naturally respects anisotropic
    tetrahedra.
    """

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int).reshape((-1, 4))
    valid, volumes, gradients = _tet_shape_gradients(pts, tet_arr)
    n_vertices = len(pts)
    if not np.any(valid):
        return sparse.csr_matrix((n_vertices, n_vertices), dtype=float)
    nodes = tet_arr[valid]
    volume = volumes[valid]
    grad = gradients[valid]
    mu = max(float(viscosity_pa_s), 1.0e-300)
    density = max(float(density_kg_m3), 0.0)
    dt = max(float(dt_s), 1.0e-300)

    # Exact integrals for the normalized quartic bubble.
    bubble_gradient_tensor = (
        (65536.0 / 15120.0)
        * volume[:, None, None]
        * np.einsum("tia,tib->tab", grad, grad)
    )
    grad_bubble_sq = np.trace(
        bubble_gradient_tensor, axis1=1, axis2=2
    )
    bubble_mass = (65536.0 / 415800.0) * volume
    scalar_momentum = mu * grad_bubble_sq + density * bubble_mass / dt
    coupling_scale = -(32.0 / 105.0) * volume
    bubble_divergence = coupling_scale[:, None, None] * grad
    if _normalize_viscous_form(viscous_form) == "vector_laplacian":
        local = np.einsum(
            "tia,tja->tij", bubble_divergence, bubble_divergence
        ) / np.maximum(scalar_momentum[:, None, None], 1.0e-300)
    else:
        bubble_momentum = (
            scalar_momentum[:, None, None] * np.eye(3, dtype=float)[None]
            + mu * bubble_gradient_tensor
        )
        inverse_bubble_momentum = np.linalg.inv(bubble_momentum)
        local = np.einsum(
            "tia,tab,tjb->tij",
            bubble_divergence,
            inverse_bubble_momentum,
            bubble_divergence,
        )
    rows = np.repeat(nodes, 4, axis=1).reshape(-1)
    cols = np.tile(nodes, (1, 4)).reshape(-1)
    return sparse.coo_matrix(
        (local.reshape(-1), (rows, cols)),
        shape=(n_vertices, n_vertices),
    ).tocsr()


def vertex_pressure_projection(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    velocities_m_s: np.ndarray,
    masses_kg: np.ndarray,
    dt_s: float,
    velocity_operator_diagonal_n_s_m: np.ndarray | None = None,
    viscosity_pa_s: float | None = None,
    pressure_stabilization_coefficient: float = 1.0 / 12.0,
    constrained_vertices: np.ndarray | None = None,
    vertex_projectors: np.ndarray | None = None,
    enforce_exact_axisymmetry: bool = False,
    axisymmetry_tolerance_m: float = 1.0e-11,
    tangent_vertex_directions: np.ndarray | None = None,
    relative_regularization: float = 1.0e-9,
    rtol: float = 1.0e-8,
    maxiter: int = 300,
) -> tuple[np.ndarray, PressureProjectionResult]:
    """Project a tetra velocity field with continuous vertex pressure."""

    velocity = np.asarray(velocities_m_s, dtype=float)
    masses = np.asarray(masses_kg, dtype=float)
    dt = max(float(dt_s), 1.0e-30)
    divergence = tet_vertex_divergence_matrix(points, tets)
    if velocity_operator_diagonal_n_s_m is None:
        mobility, constrained = _mobility_matrix(
            masses, constrained_vertices, vertex_projectors
        )
        mobility = dt * mobility
    else:
        diagonal = np.asarray(velocity_operator_diagonal_n_s_m, dtype=float)
        if diagonal.shape != (3 * len(masses),):
            raise ValueError(
                "velocity_operator_diagonal_n_s_m must have one value per velocity DOF"
            )
        constrained = constrained_dof_mask(len(masses), constrained_vertices)
        if vertex_projectors is None:
            inverse = 1.0 / np.maximum(np.abs(diagonal), 1.0e-300)
            inverse[constrained] = 0.0
            mobility = sparse.diags(inverse, format="csr")
        else:
            projectors = np.asarray(vertex_projectors, dtype=float)
            blocks = np.zeros_like(projectors)
            for vertex in range(len(masses)):
                local_scale = float(np.mean(np.abs(diagonal[3 * vertex : 3 * vertex + 3])))
                blocks[vertex] = projectors[vertex] / max(local_scale, 1.0e-300)
                if np.all(constrained[3 * vertex : 3 * vertex + 3]):
                    blocks[vertex] = 0.0
            row = np.repeat(np.arange(3 * len(masses)), 3)
            column = np.concatenate(
                [
                    np.tile(3 * vertex + np.arange(3), 3)
                    for vertex in range(len(masses))
                ]
            )
            mobility = sparse.coo_matrix(
                (blocks.reshape(-1), (row, column)),
                shape=(3 * len(masses), 3 * len(masses)),
            ).tocsr()
    velocity_vec = velocity.reshape(-1)
    velocity_transform = None
    pressure_transform = None
    if bool(enforce_exact_axisymmetry):
        velocity_transform, pressure_transform, _ring = (
            _exact_axisymmetric_transforms(
                np.asarray(points, dtype=float),
                constrained,
                tangent_vertex_directions=tangent_vertex_directions,
                tolerance_m=float(axisymmetry_tolerance_m),
            )
        )
        transform_norm = np.asarray(
            velocity_transform.power(2).sum(axis=0)
        ).reshape(-1)
        coefficients = np.divide(
            np.asarray(velocity_transform.T @ velocity_vec).reshape(-1),
            transform_norm,
            out=np.zeros_like(transform_norm),
            where=transform_norm > 1.0e-30,
        )
        velocity_vec = np.asarray(
            velocity_transform @ coefficients
        ).reshape(-1)
    weak_divergence = divergence @ velocity_vec
    system = divergence @ mobility @ divergence.T
    if viscosity_pa_s is not None and float(pressure_stabilization_coefficient) > 0.0:
        system = system + tetra_pressure_stabilization_matrix(
            points,
            tets,
            viscosity_pa_s=float(viscosity_pa_s),
            coefficient=float(pressure_stabilization_coefficient),
        )
    diagonal_scale = (
        float(np.mean(np.abs(system.diagonal()))) if system.shape[0] else 0.0
    )
    regularization = max(
        float(relative_regularization) * max(diagonal_scale, 1.0e-30),
        1.0e-30,
    )
    system = system + regularization * sparse.eye(system.shape[0], format="csr")
    rhs = weak_divergence
    solve_system = system
    solve_rhs = rhs
    if pressure_transform is not None:
        solve_system = (
            pressure_transform.T @ system @ pressure_transform
        ).tocsr()
        solve_rhs = np.asarray(
            pressure_transform.T @ rhs
        ).reshape(-1)
    if solve_system.shape[0]:
        if solve_system.shape[0] <= 30000:
            pressure_coefficients = spla.spsolve(
                solve_system.tocsc(), solve_rhs
            )
            info = 0
        else:
            pressure_coefficients, info = spla.cg(
                solve_system,
                solve_rhs,
                M=_krylov_preconditioner(solve_system),
                rtol=float(rtol),
                atol=0.0,
                maxiter=int(maxiter),
            )
        if info != 0 or not np.all(np.isfinite(pressure_coefficients)):
            pressure_coefficients = spla.lsmr(
                solve_system,
                solve_rhs,
                atol=float(rtol),
                btol=float(rtol),
                maxiter=int(maxiter),
            )[0]
            info = -abs(int(info)) if int(info) != 0 else 0
        pressure = (
            np.asarray(
                pressure_transform @ pressure_coefficients
            ).reshape(-1)
            if pressure_transform is not None
            else np.asarray(pressure_coefficients, dtype=float)
        )
    else:
        pressure = np.zeros(0, dtype=float)
        info = 0
    projected_vec = velocity_vec - mobility @ (divergence.T @ pressure)
    projected_vec[constrained] = 0.0
    if velocity_transform is not None:
        transform_norm = np.asarray(
            velocity_transform.power(2).sum(axis=0)
        ).reshape(-1)
        coefficients = np.divide(
            np.asarray(velocity_transform.T @ projected_vec).reshape(-1),
            transform_norm,
            out=np.zeros_like(transform_norm),
            where=transform_norm > 1.0e-30,
        )
        projected_vec = np.asarray(
            velocity_transform @ coefficients
        ).reshape(-1)
    residual_after = divergence @ projected_vec
    projected = projected_vec.reshape(velocity.shape)
    result = PressureProjectionResult(
        pressure_pa=np.asarray(pressure, dtype=float),
        residual_before_m3_s=float(np.sqrt(np.mean(weak_divergence**2)))
        if weak_divergence.size
        else 0.0,
        residual_after_m3_s=float(np.sqrt(np.mean(residual_after**2)))
        if residual_after.size
        else 0.0,
        pressure_l2_pa=float(np.sqrt(np.mean(pressure**2))) if pressure.size else 0.0,
        pressure_linf_pa=float(np.max(np.abs(pressure))) if pressure.size else 0.0,
        solver_info=int(info),
    )
    return projected, result


def implicit_tetra_viscous_velocity(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    velocities_m_s: np.ndarray,
    external_forces_n: np.ndarray,
    masses_kg: np.ndarray,
    viscosity_pa_s: float | np.ndarray,
    viscous_form: str = "vector_laplacian",
    dt_s: float,
    constrained_vertices: np.ndarray | None = None,
    additional_stiffness_n_s_m: sparse.spmatrix | None = None,
    rtol: float = 1.0e-8,
    maxiter: int = 400,
) -> tuple[np.ndarray, ViscousVelocityResult]:
    """Advance velocity with an implicit PR35 viscous-stress solve."""

    points_arr = np.asarray(points, dtype=float)
    velocities = np.asarray(velocities_m_s, dtype=float)
    forces = np.asarray(external_forces_n, dtype=float)
    masses = np.asarray(masses_kg, dtype=float)
    if velocities.shape != points_arr.shape or forces.shape != points_arr.shape:
        raise ValueError("velocities_m_s and external_forces_n must match points.")
    if masses.shape != (points_arr.shape[0],):
        raise ValueError("masses_kg must have one entry per vertex.")
    dt = max(float(dt_s), 1.0e-30)
    mass_diag = np.repeat(np.maximum(masses, 1.0e-300) / dt, 3)
    stiffness = tetra_viscous_stiffness_matrix(
        points_arr,
        tets,
        viscosity_pa_s,
        viscous_form=viscous_form,
    )
    system = stiffness + sparse.diags(mass_diag, format="csr")
    if additional_stiffness_n_s_m is not None:
        extra = sparse.csr_matrix(additional_stiffness_n_s_m)
        if extra.shape != system.shape:
            raise ValueError(
                "additional_stiffness_n_s_m must have shape (3*n_vertices, 3*n_vertices)."
            )
        system = system + extra
    rhs = mass_diag * velocities.reshape(-1) + forces.reshape(-1)
    fixed = constrained_dof_mask(points_arr.shape[0], constrained_vertices)
    free = ~fixed
    solution = np.zeros_like(rhs)
    info = 0
    if np.any(free):
        reduced = system[free][:, free].tocsr()
        reduced_rhs = rhs[free]
        preconditioner = _krylov_preconditioner(reduced)
        reduced_solution, info = spla.cg(
            reduced,
            reduced_rhs,
            M=preconditioner,
            rtol=float(rtol),
            atol=0.0,
            maxiter=int(maxiter),
        )
        if info != 0 or not np.all(np.isfinite(reduced_solution)):
            reduced_solution = spla.lsmr(
                reduced,
                reduced_rhs,
                atol=float(rtol),
                btol=float(rtol),
                maxiter=int(maxiter),
            )[0]
            info = -abs(int(info)) if int(info) != 0 else 0
        solution[free] = reduced_solution
    residual = system @ solution - rhs
    velocity = solution.reshape(velocities.shape)
    result = ViscousVelocityResult(
        residual_l2_n=float(np.sqrt(np.mean(residual[free] ** 2))) if np.any(free) else 0.0,
        velocity_linf_m_s=float(np.max(np.linalg.norm(velocity, axis=1))) if velocity.size else 0.0,
        solver_info=int(info),
        operator_diagonal_n_s_m=np.asarray(system.diagonal(), dtype=float),
    )
    return velocity, result


def sparse_pressure_projection(
    *,
    velocities_m_s: np.ndarray,
    nonpressure_forces_n: np.ndarray,
    masses_kg: np.ndarray,
    tet_volumes_m3: np.ndarray,
    target_tet_volumes_m3: np.ndarray,
    volume_matrix: sparse.csr_matrix,
    dt_s: float,
    constrained_vertices: np.ndarray | None = None,
    vertex_projectors: np.ndarray | None = None,
    axisymmetric_points_m: np.ndarray | None = None,
    axisymmetric_tangent_vertex_directions: np.ndarray | None = None,
    axisymmetry_tolerance_m: float = 1.0e-11,
    upper_velocity_constraints: sparse.spmatrix | None = None,
    upper_velocity_constraint_rhs_m_s: np.ndarray | None = None,
    upper_velocity_constraint_tolerance_m_s: float = 1.0e-12,
    relative_regularization: float = 1.0e-10,
    solver: str = "cg",
    jacobi_iterations: int = 10,
    jacobi_omega: float = 0.72,
    rtol: float = 1.0e-8,
    maxiter: int = 300,
) -> tuple[np.ndarray, PressureProjectionResult]:
    """Project force-driven vertex velocity onto tetra-volume continuity.

    The solve is the sparse counterpart of the PR33 pressure projection:

    ``B u^{n+1} = -(V^n - V_target) / dt``

    with ``u^{n+1} = u^n + dt M^{-1}(F_np + B^T p)``.  Constrained vertices
    are implemented by zeroing their inverse mass in all xyz components.

    When ``axisymmetric_points_m`` is supplied, the PR33 projection is
    assembled directly in the shared meridional ring-velocity basis.  This is
    a Galerkin reduction of the same ``B``, mass, force, and ``B.T*p``
    operators; it does not average independently solved Cartesian velocities.
    """

    velocities = np.asarray(velocities_m_s, dtype=float)
    forces = np.asarray(nonpressure_forces_n, dtype=float)
    masses = np.asarray(masses_kg, dtype=float)
    tet_volumes = np.asarray(tet_volumes_m3, dtype=float)
    target_tet_volumes = np.asarray(target_tet_volumes_m3, dtype=float)
    dt = max(float(dt_s), 1.0e-30)

    if velocities.shape != forces.shape:
        raise ValueError("velocities_m_s and nonpressure_forces_n must have the same shape.")
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError("velocities_m_s must have shape (n_vertices, 3).")
    if masses.shape[0] != velocities.shape[0]:
        raise ValueError("masses_kg must have one entry per vertex.")

    mobility, constrained = _mobility_matrix(
        masses,
        constrained_vertices,
        vertex_projectors,
    )

    upper_constraint = None
    upper_constraint_rhs = None
    if upper_velocity_constraints is not None:
        upper_constraint = sparse.csr_matrix(upper_velocity_constraints)
        if upper_constraint.shape[1] != 3 * len(velocities):
            raise ValueError(
                "upper_velocity_constraints must have 3*n_vertices columns"
            )
        upper_constraint_rhs = np.asarray(
            upper_velocity_constraint_rhs_m_s, dtype=float
        ).reshape(-1)
        if upper_constraint_rhs.shape != (upper_constraint.shape[0],):
            raise ValueError(
                "upper_velocity_constraint_rhs_m_s must match constraint rows"
            )
        if upper_constraint.shape[0] != 1:
            raise ValueError(
                "The sparse PR33 upper-bound active set currently supports "
                "one scalar constraint"
            )
        if not np.all(np.isfinite(upper_constraint_rhs)):
            raise ValueError(
                "upper_velocity_constraint_rhs_m_s must be finite"
            )
        if axisymmetric_points_m is None:
            raise ValueError(
                "The sparse PR33 upper-bound active set requires the exact-"
                "axisymmetric velocity basis"
            )
        if str(solver).lower() not in {
            "lsmr_velocity",
            "velocity_lsmr",
            "minimum_velocity_lsmr",
        }:
            raise ValueError(
                "The sparse PR33 upper-bound active set requires the "
                "lsmr_velocity projection"
            )

    velocity_vec = velocities.reshape(-1)
    force_vec = forces.reshape(-1)
    bmat = volume_matrix.tocsr()
    target_rates = -(tet_volumes - target_tet_volumes) / dt
    if str(solver).lower() in {
        "lsmr_velocity",
        "velocity_lsmr",
        "minimum_velocity_lsmr",
    }:
        # Solve the PR33 projection through its weighted divergence operator
        # C=B M^{-1/2}, not through the squared-condition pressure Schur
        # matrix C C^T.  The two forms are algebraically identical:
        #
        #   C y = target_rate - B u*,
        #   du = M^{-1/2} y,
        #   C^T q = y,  p=q/dt,  F_pressure=B^T p.
        #
        # The second least-squares solve reconstructs the exact named
        # pressure-force array used by the velocity correction.
        axisymmetric_basis = None
        if axisymmetric_points_m is not None:
            axisymmetric_points = np.asarray(
                axisymmetric_points_m, dtype=float
            )
            if axisymmetric_points.shape != velocities.shape:
                raise ValueError(
                    "axisymmetric_points_m must match velocities_m_s"
                )
            axisymmetric_basis, _axisymmetric_ring = (
                _exact_axisymmetric_velocity_transform(
                    axisymmetric_points,
                    constrained,
                    tangent_vertex_directions=(
                        axisymmetric_tangent_vertex_directions
                    ),
                    tolerance_m=float(axisymmetry_tolerance_m),
                )
            )
            if axisymmetric_basis.shape[1] == 0:
                raise ValueError(
                    "axisymmetric PR33 projection has no free meridional "
                    "velocity degrees of freedom"
                )

        if axisymmetric_basis is not None:
            mass_dof = np.repeat(
                np.maximum(masses, 1.0e-300), 3
            )
            reduced_mass = np.asarray(
                axisymmetric_basis.power(2).T @ mass_dof
            ).reshape(-1)
            reduced_mass = np.maximum(reduced_mass, 1.0e-300)
            reduced_old_velocity = np.asarray(
                axisymmetric_basis.T @ (mass_dof * velocity_vec)
            ).reshape(-1) / reduced_mass
            reduced_force = np.asarray(
                axisymmetric_basis.T @ force_vec
            ).reshape(-1)
            reduced_trial = (
                reduced_old_velocity
                + dt * reduced_force / reduced_mass
            )
            trial_vec = np.asarray(
                axisymmetric_basis @ reduced_trial
            ).reshape(-1)
            reduced_divergence = (
                bmat @ axisymmetric_basis
            ).tocsr()
            sqrt_reduced_mobility = sparse.diags(
                1.0 / np.sqrt(reduced_mass), format="csr"
            )
            weighted_divergence = (
                reduced_divergence @ sqrt_reduced_mobility
            ).tocsr()
            velocity_rhs = target_rates - bmat @ trial_vec
            velocity_solution = spla.lsmr(
                weighted_divergence,
                velocity_rhs,
                atol=float(rtol),
                btol=float(rtol),
                maxiter=int(maxiter),
            )
            weighted_correction = np.asarray(
                velocity_solution[0], dtype=float
            )
            pressure_solution = spla.lsmr(
                weighted_divergence.T,
                weighted_correction,
                atol=float(rtol),
                btol=float(rtol),
                maxiter=int(maxiter),
            )
            pressure = (
                np.asarray(pressure_solution[0], dtype=float) / dt
            )
            pressure_force = np.asarray(
                bmat.T @ pressure, dtype=float
            )
            pressure_force_vertices = pressure_force.reshape(
                velocities.shape
            )
            radius = np.hypot(
                axisymmetric_points[:, 0],
                axisymmetric_points[:, 1],
            )
            active_radius = radius > float(axisymmetry_tolerance_m)
            azimuthal = np.zeros_like(axisymmetric_points)
            azimuthal[active_radius, 0] = (
                -axisymmetric_points[active_radius, 1]
                / radius[active_radius]
            )
            azimuthal[active_radius, 1] = (
                axisymmetric_points[active_radius, 0]
                / radius[active_radius]
            )
            angular_pressure_force = np.sum(
                pressure_force_vertices * azimuthal, axis=1
            )
            pressure_force_vertices = (
                pressure_force_vertices
                - angular_pressure_force[:, None] * azimuthal
            )
            pressure_force = pressure_force_vertices.reshape(-1)
            reduced_pressure_force = np.asarray(
                axisymmetric_basis.T @ pressure_force
            ).reshape(-1)
            reduced_projected = (
                reduced_trial
                + dt * reduced_pressure_force / reduced_mass
            )
            unconstrained_reduced_projected = reduced_projected.copy()
            projected_vec = np.asarray(
                axisymmetric_basis @ reduced_projected
            ).reshape(-1)
            projected_vec[constrained] = 0.0
            constraint_force_vertices = np.zeros_like(velocities)
            constraint_multiplier = np.zeros(0, dtype=float)
            unconstrained_constraint_value = np.zeros(0, dtype=float)
            constraint_residual = np.zeros(0, dtype=float)
            upper_constraint_active = False
            if upper_constraint is not None:
                unconstrained_constraint_value = np.asarray(
                    upper_constraint @ projected_vec, dtype=float
                ).reshape(-1)
                tolerance = max(
                    float(upper_velocity_constraint_tolerance_m_s), 0.0
                )
                if (
                    unconstrained_constraint_value[0]
                    > upper_constraint_rhs[0] + tolerance
                ):
                    # Apply a unit generalized force C.T, project its velocity
                    # response through the same PR33 nullspace, then solve the
                    # scalar complementarity equation C*u=U_max.  The final
                    # update is therefore generated by an explicit reaction
                    # force plus a separately retained B.T*p correction.
                    reduced_constraint = sparse.csr_matrix(
                        upper_constraint @ axisymmetric_basis
                    )
                    reduced_unit_force = np.asarray(
                        reduced_constraint.T.toarray(), dtype=float
                    ).reshape(-1)
                    reduced_response_trial = (
                        dt * reduced_unit_force / reduced_mass
                    )
                    response_rhs = -np.asarray(
                        reduced_divergence @ reduced_response_trial,
                        dtype=float,
                    ).reshape(-1)
                    response_solution = spla.lsmr(
                        weighted_divergence,
                        response_rhs,
                        atol=float(rtol),
                        btol=float(rtol),
                        maxiter=int(maxiter),
                    )
                    response_weighted_correction = np.asarray(
                        response_solution[0], dtype=float
                    )
                    response_pressure_solution = spla.lsmr(
                        weighted_divergence.T,
                        response_weighted_correction,
                        atol=float(rtol),
                        btol=float(rtol),
                        maxiter=int(maxiter),
                    )
                    response_pressure = np.asarray(
                        response_pressure_solution[0], dtype=float
                    ) / dt
                    response_pressure_force = np.asarray(
                        bmat.T @ response_pressure, dtype=float
                    ).reshape(velocities.shape)
                    response_angular_force = np.sum(
                        response_pressure_force * azimuthal, axis=1
                    )
                    response_pressure_force = (
                        response_pressure_force
                        - response_angular_force[:, None] * azimuthal
                    )
                    reduced_response_pressure_force = np.asarray(
                        axisymmetric_basis.T
                        @ response_pressure_force.reshape(-1)
                    ).reshape(-1)
                    reduced_response = (
                        reduced_response_trial
                        + dt
                        * reduced_response_pressure_force
                        / reduced_mass
                    )
                    response_vec = np.asarray(
                        axisymmetric_basis @ reduced_response
                    ).reshape(-1)
                    response_vec[constrained] = 0.0
                    sensitivity = float(
                        np.asarray(
                            upper_constraint @ response_vec,
                            dtype=float,
                        ).reshape(-1)[0]
                    )
                    if not math.isfinite(sensitivity) or sensitivity <= 0.0:
                        raise RuntimeError(
                            "The PR33 Cox-speed reaction has no positive "
                            "contact mobility"
                        )
                    multiplier = (
                        float(upper_constraint_rhs[0])
                        - float(unconstrained_constraint_value[0])
                    ) / sensitivity
                    constraint_multiplier = np.asarray(
                        (multiplier,), dtype=float
                    )
                    constraint_force_vec = np.asarray(
                        upper_constraint.T.toarray(), dtype=float
                    ).reshape(-1) * multiplier
                    constraint_force_vertices = constraint_force_vec.reshape(
                        velocities.shape
                    )
                    pressure = pressure + multiplier * response_pressure
                    pressure_force_vertices = (
                        pressure_force_vertices
                        + multiplier * response_pressure_force
                    )
                    pressure_force = pressure_force_vertices.reshape(-1)
                    reduced_pressure_force = np.asarray(
                        axisymmetric_basis.T @ pressure_force
                    ).reshape(-1)
                    # Form the endpoint from the same nullspace response used
                    # to calculate the scalar multiplier.  This avoids losing
                    # the small constrained velocity to cancellation between
                    # the large pressure and contact-reaction forces.
                    reduced_projected = (
                        unconstrained_reduced_projected
                        + multiplier * reduced_response
                    )
                    projected_vec = np.asarray(
                        axisymmetric_basis @ reduced_projected
                    ).reshape(-1)
                    projected_vec[constrained] = 0.0
                    upper_constraint_active = True
                constraint_residual = np.asarray(
                    upper_constraint @ projected_vec, dtype=float
                ).reshape(-1) - upper_constraint_rhs
            residual_before = bmat @ velocity_vec - target_rates
            residual_after = bmat @ projected_vec - target_rates
            velocity_istop = int(velocity_solution[1])
            pressure_istop = int(pressure_solution[1])
            info = (
                0
                if velocity_istop in {1, 2}
                and pressure_istop in {1, 2}
                else -max(velocity_istop, pressure_istop)
            )
            projected = projected_vec.reshape(velocities.shape)
            result = PressureProjectionResult(
                pressure_pa=pressure,
                residual_before_m3_s=float(
                    np.sqrt(np.mean(residual_before * residual_before))
                )
                if residual_before.size
                else 0.0,
                residual_after_m3_s=float(
                    np.sqrt(np.mean(residual_after * residual_after))
                )
                if residual_after.size
                else 0.0,
                pressure_l2_pa=float(
                    np.sqrt(np.mean(pressure * pressure))
                )
                if pressure.size
                else 0.0,
                pressure_linf_pa=float(
                    np.max(np.abs(pressure), initial=0.0)
                ),
                solver_info=info,
                pressure_force_n=pressure_force_vertices,
                constraint_force_n=constraint_force_vertices,
                constraint_multiplier_n=constraint_multiplier,
                upper_velocity_constraint_active=upper_constraint_active,
                unconstrained_upper_velocity_value_m_s=(
                    unconstrained_constraint_value
                ),
                upper_velocity_constraint_residual_m_s=constraint_residual,
            )
            return projected, result

        if vertex_projectors is None:
            sqrt_inv_mass = np.repeat(
                1.0 / np.sqrt(np.maximum(masses, 1.0e-300)), 3
            )
            sqrt_inv_mass[constrained] = 0.0
            sqrt_mobility = sparse.diags(
                sqrt_inv_mass, format="csr"
            )
        else:
            projectors = np.asarray(vertex_projectors, dtype=float)
            projectors = 0.5 * (
                projectors + np.swapaxes(projectors, 1, 2)
            )
            blocks = (
                projectors
                / np.sqrt(np.maximum(masses, 1.0e-300))[:, None, None]
            )
            if constrained_vertices is not None:
                fixed = np.asarray(constrained_vertices, dtype=bool)
                blocks[fixed] = 0.0
            rows = np.repeat(
                np.arange(3 * len(masses), dtype=int).reshape(-1, 3),
                3,
                axis=1,
            ).reshape(-1)
            cols = np.tile(
                np.arange(3 * len(masses), dtype=int).reshape(-1, 3),
                (1, 3),
            ).reshape(-1)
            sqrt_mobility = sparse.coo_matrix(
                (blocks.reshape(-1), (rows, cols)),
                shape=(3 * len(masses), 3 * len(masses)),
            ).tocsr()
            sqrt_mobility.eliminate_zeros()
        trial_vec = velocity_vec + dt * (mobility @ force_vec)
        trial_vec[constrained] = 0.0
        weighted_divergence = (bmat @ sqrt_mobility).tocsr()
        velocity_rhs = target_rates - bmat @ trial_vec
        velocity_solution = spla.lsmr(
            weighted_divergence,
            velocity_rhs,
            atol=float(rtol),
            btol=float(rtol),
            maxiter=int(maxiter),
        )
        weighted_correction = np.asarray(
            velocity_solution[0], dtype=float
        )
        projected_vec = (
            trial_vec + sqrt_mobility @ weighted_correction
        )
        projected_vec[constrained] = 0.0
        pressure_solution = spla.lsmr(
            weighted_divergence.T,
            weighted_correction,
            atol=float(rtol),
            btol=float(rtol),
            maxiter=int(maxiter),
        )
        pressure = np.asarray(pressure_solution[0], dtype=float) / dt
        pressure_force = np.asarray(
            bmat.T @ pressure, dtype=float
        )
        # Reconstruct from the named B^T p force so the reported array and
        # solver-used update are bitwise the same path.
        projected_vec = (
            trial_vec + dt * (mobility @ pressure_force)
        )
        projected_vec[constrained] = 0.0
        residual_before = bmat @ velocity_vec - target_rates
        residual_after = bmat @ projected_vec - target_rates
        velocity_istop = int(velocity_solution[1])
        pressure_istop = int(pressure_solution[1])
        info = (
            0
            if velocity_istop in {1, 2}
            and pressure_istop in {1, 2}
            else -max(velocity_istop, pressure_istop)
        )
        projected = projected_vec.reshape(velocities.shape)
        result = PressureProjectionResult(
            pressure_pa=pressure,
            residual_before_m3_s=float(
                np.sqrt(np.mean(residual_before * residual_before))
            )
            if residual_before.size
            else 0.0,
            residual_after_m3_s=float(
                np.sqrt(np.mean(residual_after * residual_after))
            )
            if residual_after.size
            else 0.0,
            pressure_l2_pa=float(
                np.sqrt(np.mean(pressure * pressure))
            )
            if pressure.size
            else 0.0,
            pressure_linf_pa=float(
                np.max(np.abs(pressure), initial=0.0)
            ),
            solver_info=info,
            pressure_force_n=pressure_force.reshape(velocities.shape),
        )
        return projected, result

    rhs = (target_rates - bmat @ velocity_vec) / dt - bmat @ (mobility @ force_vec)
    stiffness = bmat @ mobility @ bmat.T
    diag_scale = float(np.mean(np.abs(stiffness.diagonal()))) if stiffness.shape[0] else 0.0
    reg = max(float(relative_regularization) * max(diag_scale, 1.0e-30), 1.0e-30)
    system = stiffness + reg * sparse.eye(stiffness.shape[0], format="csr")

    if system.shape[0] == 0:
        pressure = np.zeros(0, dtype=float)
        info = 0
    elif str(solver).lower() == "jacobi":
        diag = system.diagonal()
        diag = np.where(np.abs(diag) > 1.0e-300, diag, 1.0)
        pressure = np.zeros_like(rhs)
        omega = float(np.clip(jacobi_omega, 0.05, 1.0))
        for _ in range(max(int(jacobi_iterations), 1)):
            pressure += omega * (rhs - system @ pressure) / diag
        info = 0
    elif str(solver).lower() == "lsmr":
        pressure = spla.lsmr(system, rhs, atol=float(rtol), btol=float(rtol), maxiter=int(maxiter))[0]
        info = 0
    else:
        if str(solver).lower() == "cg_jacobi":
            diagonal = system.diagonal()
            diagonal = np.where(
                np.abs(diagonal) > 1.0e-300, diagonal, 1.0
            )
            preconditioner = spla.LinearOperator(
                system.shape, matvec=lambda value: value / diagonal
            )
        else:
            preconditioner = _krylov_preconditioner(system)
        pressure, info = spla.cg(
            system,
            rhs,
            M=preconditioner,
            rtol=float(rtol),
            atol=0.0,
            maxiter=int(maxiter),
        )
        if info != 0 or not np.all(np.isfinite(pressure)):
            pressure = spla.lsmr(system, rhs, atol=float(rtol), btol=float(rtol), maxiter=int(maxiter))[0]
            info = -abs(int(info)) if int(info) != 0 else 0

    total_force_vec = force_vec + bmat.T @ pressure
    projected_vec = velocity_vec + dt * (mobility @ total_force_vec)
    projected_vec[constrained] = 0.0
    projected = projected_vec.reshape(velocities.shape)
    residual_before = bmat @ velocity_vec - target_rates
    residual_after = bmat @ projected_vec - target_rates
    result = PressureProjectionResult(
        pressure_pa=np.asarray(pressure, dtype=float),
        residual_before_m3_s=float(np.sqrt(np.mean(residual_before * residual_before))) if residual_before.size else 0.0,
        residual_after_m3_s=float(np.sqrt(np.mean(residual_after * residual_after))) if residual_after.size else 0.0,
        pressure_l2_pa=float(np.sqrt(np.mean(pressure * pressure))) if pressure.size else 0.0,
        pressure_linf_pa=float(np.max(np.abs(pressure))) if pressure.size else 0.0,
        solver_info=int(info),
        pressure_force_n=np.asarray(
            bmat.T @ pressure, dtype=float
        ).reshape(velocities.shape),
    )
    return projected, result


def triangle_area_vectors(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return one oriented area vector per triangular face."""

    pts = np.asarray(points, dtype=float)
    tri = pts[np.asarray(faces, dtype=int)]
    return 0.5 * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def vertex_area_vectors(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Accumulate one third of each triangle area vector onto its vertices."""

    pts = np.asarray(points, dtype=float)
    face_arr = np.asarray(faces, dtype=int)
    area_vec = np.zeros_like(pts, dtype=float)
    if face_arr.size == 0:
        return area_vec
    vectors = triangle_area_vectors(pts, face_arr)
    flip = vectors[:, 2] < 0.0
    vectors[flip] *= -1.0
    share = vectors / 3.0
    np.add.at(area_vec, face_arr[:, 0], share)
    np.add.at(area_vec, face_arr[:, 1], share)
    np.add.at(area_vec, face_arr[:, 2], share)
    return area_vec


def heron_surface_tension_forces_from_faces(
    points: np.ndarray,
    faces: np.ndarray,
    surface_tension_n_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Vectorized ddgclib Heron force on a triangulated free surface.

    This is the indexed-face equivalent of
    ``ddgclib._curvatures_heron.hndA_i``:

    ``F_i^Heron = -gamma * (HN dA)_i``.

    The stable Heron triangle-area expression reduces algebraically to the
    cotangent edge weight used below.  Keeping the array implementation avoids
    rebuilding a HyperCT neighbor graph at every PR35/PR33 nonlinear solve.
    Regression tests compare this function directly with ``hndA_i``.
    """

    pts = np.asarray(points, dtype=float)
    face_arr = np.asarray(faces, dtype=int)
    forces = np.zeros_like(pts, dtype=float)
    areas = np.zeros(pts.shape[0], dtype=float)
    if face_arr.size == 0:
        return forces, areas, 0.0

    ia = face_arr[:, 0]
    ib = face_arr[:, 1]
    ic = face_arr[:, 2]
    a = pts[ia]
    b = pts[ib]
    c = pts[ic]

    def cot_batch(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        cross_norm = np.linalg.norm(np.cross(u, v), axis=1)
        dot = np.einsum("ij,ij->i", u, v)
        return np.divide(dot, cross_norm, out=np.zeros_like(dot), where=cross_norm > 1.0e-30)

    tri_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    valid = tri_area > 1.0e-30
    if not np.any(valid):
        return forces, areas, 0.0
    ia = ia[valid]
    ib = ib[valid]
    ic = ic[valid]
    a = a[valid]
    b = b[valid]
    c = c[valid]
    tri_area = tri_area[valid]

    np.add.at(areas, ia, tri_area / 3.0)
    np.add.at(areas, ib, tri_area / 3.0)
    np.add.at(areas, ic, tri_area / 3.0)

    cot_a = cot_batch(b - a, c - a)
    cot_b = cot_batch(c - b, a - b)
    cot_c = cot_batch(a - c, b - c)
    lap = np.zeros_like(pts, dtype=float)
    np.add.at(lap, ib, cot_a[:, None] * (b - c))
    np.add.at(lap, ic, cot_a[:, None] * (c - b))
    np.add.at(lap, ic, cot_b[:, None] * (c - a))
    np.add.at(lap, ia, cot_b[:, None] * (a - c))
    np.add.at(lap, ia, cot_c[:, None] * (a - b))
    np.add.at(lap, ib, cot_c[:, None] * (b - a))

    forces = -float(surface_tension_n_m) * 0.5 * lap
    area_vec = vertex_area_vectors(pts, face_arr)
    denom = float(np.sum(area_vec * area_vec))
    p_equiv = 0.0 if denom <= 0.0 else -float(np.sum(area_vec * forces) / denom)
    return forces, areas, p_equiv


def cotangent_surface_tension_forces(
    points: np.ndarray,
    faces: np.ndarray,
    surface_tension_n_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Backward-compatible alias for the vectorized Heron force."""

    return heron_surface_tension_forces_from_faces(
        points,
        faces,
        surface_tension_n_m,
    )


def cotangent_surface_tension_stiffness(
    points: np.ndarray,
    faces: np.ndarray,
    surface_tension_n_m: float,
) -> sparse.csr_matrix:
    """Frozen-cotangent Hessian used for implicit capillary motion.

    For one geometry, the DDG force is ``F_gamma = -K_gamma x``.  A backward
    Euler position update ``x_new = x + dt*u`` therefore contributes
    ``dt*K_gamma`` to the velocity-system stiffness.  The returned matrix is
    ``K_gamma``; callers apply their own timestep.
    """

    pts = np.asarray(points, dtype=float)
    face_arr = np.asarray(faces, dtype=int).reshape((-1, 3))
    n_vertices = len(pts)
    if face_arr.size == 0:
        return sparse.csr_matrix((3 * n_vertices, 3 * n_vertices))
    tri = pts[face_arr]
    area2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    valid = area2 > 1.0e-30
    face_arr = face_arr[valid]
    tri = tri[valid]
    area2 = area2[valid]

    def cot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", u, v) / area2

    cot_a = cot(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cot_b = cot(tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 1])
    cot_c = cot(tri[:, 0] - tri[:, 2], tri[:, 1] - tri[:, 2])
    edge_i = np.concatenate((face_arr[:, 1], face_arr[:, 2], face_arr[:, 0]))
    edge_j = np.concatenate((face_arr[:, 2], face_arr[:, 0], face_arr[:, 1]))
    weight = 0.5 * float(surface_tension_n_m) * np.concatenate(
        (cot_a, cot_b, cot_c)
    )
    rows = np.concatenate((edge_i, edge_j, edge_i, edge_j))
    cols = np.concatenate((edge_i, edge_j, edge_j, edge_i))
    data = np.concatenate((weight, weight, -weight, -weight))
    scalar = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_vertices, n_vertices)
    ).tocsr()
    return sparse.kron(scalar, sparse.eye(3, format="csr"), format="csr")


def pressure_area_forces(points: np.ndarray, faces: np.ndarray, pressure_pa: float | np.ndarray) -> np.ndarray:
    """Map scalar face/constant pressure to vertex forces via area vectors."""

    pts = np.asarray(points, dtype=float)
    face_arr = np.asarray(faces, dtype=int)
    forces = np.zeros_like(pts, dtype=float)
    if face_arr.size == 0:
        return forces
    vectors = triangle_area_vectors(pts, face_arr)
    flip = vectors[:, 2] < 0.0
    vectors[flip] *= -1.0
    pressure = np.asarray(pressure_pa, dtype=float)
    if pressure.ndim == 0:
        face_force = float(pressure) * vectors
    elif pressure.shape[0] == face_arr.shape[0]:
        face_force = pressure[:, None] * vectors
    else:
        raise ValueError("pressure_pa must be scalar or one value per face.")
    share = face_force / 3.0
    np.add.at(forces, face_arr[:, 0], share)
    np.add.at(forces, face_arr[:, 1], share)
    np.add.at(forces, face_arr[:, 2], share)
    return forces


def sphere_disjoining_pressure_forces(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
    hamaker_constant_j: float,
    precursor_thickness_m: float,
    interaction_range_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a Hamaker wetting-pressure force for a sphere above a liquid mesh.

    The force is evaluated on the actual triangular free surface and is mapped
    to vertices with :func:`pressure_area_forces`.  A positive Hamaker constant
    pulls the liquid interface upward toward the lower sphere surface,
    regularising the otherwise singular zero-radius complete-wetting event.
    It is a general material model: ``A`` and the precursor thickness are
    explicit inputs rather than a prescribed bridge radius or profile.

    Returns ``(vertex_forces, face_pressures, minimum_positive_gap)``.  Faces
    outside ``interaction_range_m`` receive zero pressure when a range is
    supplied.
    """

    pts = np.asarray(points, dtype=float)
    face_arr = np.asarray(faces, dtype=int)
    forces = np.zeros_like(pts, dtype=float)
    if face_arr.size == 0 or float(hamaker_constant_j) <= 0.0:
        return forces, np.zeros(face_arr.shape[0], dtype=float), float("inf")

    centroid = np.mean(pts[face_arr], axis=1)
    radius = np.hypot(centroid[:, 0], centroid[:, 1])
    sphere_radius = max(float(sphere_radius_m), 1.0e-30)
    clipped_radius = np.minimum(radius, sphere_radius * (1.0 - 1.0e-12))
    sphere_z = float(sphere_tip_z_m) + sphere_radius - np.sqrt(
        np.maximum(sphere_radius * sphere_radius - clipped_radius * clipped_radius, 0.0)
    )
    gap = sphere_z - centroid[:, 2]
    positive_gap = np.maximum(gap, 0.0)
    precursor = max(float(precursor_thickness_m), 1.0e-15)
    pressure = float(hamaker_constant_j) / (
        6.0 * math.pi * (positive_gap + precursor) ** 3
    )
    valid = gap > 0.0
    if interaction_range_m is not None and float(interaction_range_m) > 0.0:
        valid &= gap <= float(interaction_range_m)
    pressure = np.where(valid, pressure, 0.0)
    forces = pressure_area_forces(pts, face_arr, pressure)
    min_gap = float(np.min(gap[gap > 0.0])) if np.any(gap > 0.0) else 0.0
    return forces, pressure, min_gap


def sphere_lower_surface_z(
    radius_m: np.ndarray | float,
    *,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
) -> np.ndarray | float:
    """Lower z-coordinate of a sphere whose bottom tip is at ``sphere_tip_z_m``."""

    r = np.asarray(radius_m, dtype=float)
    radius = float(sphere_radius_m)
    clipped = np.minimum(r, radius * (1.0 - 1.0e-12))
    z = float(sphere_tip_z_m) + radius - np.sqrt(np.maximum(radius * radius - clipped * clipped, 0.0))
    if np.isscalar(radius_m):
        return float(z)
    return z


def project_vertices_to_sphere_lower_surface(
    points: np.ndarray,
    vertices: np.ndarray,
    *,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
) -> np.ndarray:
    """Project selected vertices onto the lower sphere surface at fixed x/y."""

    updated = np.asarray(points, dtype=float).copy()
    idx = np.asarray(vertices, dtype=int)
    if idx.size == 0:
        return updated
    radius = np.hypot(updated[idx, 0], updated[idx, 1])
    updated[idx, 2] = sphere_lower_surface_z(
        radius,
        sphere_radius_m=float(sphere_radius_m),
        sphere_tip_z_m=float(sphere_tip_z_m),
    )
    return updated


def tangent_project_sphere_velocities(
    points: np.ndarray,
    velocities: np.ndarray,
    vertices: np.ndarray,
    *,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
) -> np.ndarray:
    """Remove normal velocity on vertices constrained to slide on a sphere."""

    pts = np.asarray(points, dtype=float)
    vel = np.asarray(velocities, dtype=float).copy()
    idx = np.asarray(vertices, dtype=int)
    if idx.size == 0:
        return vel
    center = np.asarray([0.0, 0.0, float(sphere_tip_z_m) + float(sphere_radius_m)], dtype=float)
    normal = pts[idx] - center[None, :]
    norm = np.linalg.norm(normal, axis=1)
    valid = norm > 1.0e-30
    normal[valid] /= norm[valid, None]
    normal[~valid] = 0.0
    normal_velocity = np.sum(vel[idx] * normal, axis=1)
    vel[idx] -= normal_velocity[:, None] * normal
    return vel


def boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    """Return triangular boundary faces that belong to exactly one tetrahedron."""

    face_map: dict[tuple[int, int, int], int] = {}
    counts: dict[tuple[int, int, int], int] = {}
    for tet in np.asarray(tets, dtype=int):
        a, b, c, d = [int(v) for v in tet]
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(face))
            face_map[key] = key
            counts[key] = counts.get(key, 0) + 1
    return np.asarray([face for face, count in counts.items() if count == 1], dtype=int)


def connected_components_from_tets(tets: np.ndarray, n_vertices: int | None = None) -> np.ndarray:
    """Label vertex connectivity components induced by tetrahedra."""

    tet_arr = np.asarray(tets, dtype=int)
    if n_vertices is None:
        n_vertices = int(tet_arr.max()) + 1 if tet_arr.size else 0
    parent = np.arange(int(n_vertices), dtype=int)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return int(a)

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    for tet in tet_arr:
        a = int(tet[0])
        for vertex in tet[1:]:
            union(a, int(vertex))
    roots = np.asarray([find(i) for i in range(int(n_vertices))], dtype=int)
    unique = {root: idx for idx, root in enumerate(np.unique(roots))}
    return np.asarray([unique[root] for root in roots], dtype=int)


def mesh_quality(points: np.ndarray, tets: np.ndarray) -> dict[str, float]:
    """Return basic tetra quality and volume diagnostics."""

    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int)
    volumes = tet_cell_volumes(pts, tet_arr)
    if tet_arr.size == 0:
        return {
            "tet_count": 0.0,
            "volume_m3": 0.0,
            "min_volume_m3": 0.0,
            "negative_volume_count": 0.0,
            "min_edge_m": 0.0,
            "max_edge_m": 0.0,
        }
    tet_pts = pts[tet_arr]
    raw_arr = np.einsum(
        "ij,ij->i",
        tet_pts[:, 1] - tet_pts[:, 0],
        np.cross(tet_pts[:, 2] - tet_pts[:, 0], tet_pts[:, 3] - tet_pts[:, 0]),
    ) / 6.0
    edge_pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=int)
    edge_vec = tet_pts[:, edge_pairs[:, 1], :] - tet_pts[:, edge_pairs[:, 0], :]
    edge_arr = np.linalg.norm(edge_vec.reshape(-1, 3), axis=1)
    return {
        "tet_count": float(tet_arr.shape[0]),
        "volume_m3": float(np.sum(volumes)),
        "min_volume_m3": float(np.min(volumes)),
        "negative_volume_count": float(np.sum(raw_arr < 0.0)),
        "min_edge_m": float(np.min(edge_arr)) if edge_arr.size else 0.0,
        "max_edge_m": float(np.max(edge_arr)) if edge_arr.size else 0.0,
    }
