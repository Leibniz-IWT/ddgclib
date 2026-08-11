"""PR #33 benchmark-local operators.

This module holds the pressure-reference, tet-volume, ALE-flux, and active
retopology helpers used by the oscillating-droplet benchmark.  They are kept
outside ``ddgclib.operators.stress`` and ``ddgclib.operators.multiphase_stress``
so the core library only carries the existing stress/multiphase APIs while this
case can still reuse ddgclib's Heron curvature routines.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark add-on operators: HC/FHeron droplet volume-pressure closure
# ---------------------------------------------------------------------------

def build_surface_hc_from_faces(points: np.ndarray, faces: np.ndarray):
    """Build an HC surface graph from triangular interface connectivity.

    Reference: the surface curvature force uses the ddgclib/HyperCT Heron
    integrated mean-curvature vector, as described in the ddgclib
    fundamentals and in Heron's discrete curvature construction.
    """
    from hyperct import Complex

    HC = Complex(3, domain=None)
    vertices = [HC.V[tuple(float(x) for x in point)] for point in np.asarray(points, dtype=float)]
    for a, b, c in np.asarray(faces, dtype=int):
        va, vb, vc = vertices[int(a)], vertices[int(b)], vertices[int(c)]
        va.connect(vb)
        vb.connect(vc)
        vc.connect(va)
    for vertex in vertices:
        vertex.u = np.zeros(3, dtype=float)
        vertex.p = 0.0
        vertex.m = 1.0
        vertex.phase = 0
        vertex.boundary = False
        vertex.is_interface = True
    return HC, vertices


def heron_surface_force_from_faces(
    points: np.ndarray,
    faces: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodal FHeron and Heron scalar dual area on a surface mesh.

    Formula used by the benchmark:

        F_i^Heron = -gamma (HN dA)_i

    Reference: ddgclib Heron curvature operator / HyperCT surface graph.
    The function is intentionally separate from ``stress_force`` because it is
    an interface curvature force, not a bulk Cauchy stress flux.
    """
    from ddgclib._curvatures_heron import hndA_i

    _HC, vertices = build_surface_hc_from_faces(points, faces)
    forces = np.zeros((len(vertices), 3), dtype=float)
    areas = np.zeros(len(vertices), dtype=float)
    for i, vertex in enumerate(vertices):
        hnda_i, area_i = hndA_i(vertex)
        forces[i] = -float(gamma) * np.asarray(hnda_i[:3], dtype=float)
        areas[i] = float(area_i)
    return forces, areas


def mesh_volume_from_faces(points: np.ndarray, faces: np.ndarray) -> float:
    """Closed triangular surface volume by signed origin tetrahedra."""
    pts = np.asarray(points, dtype=float)
    volumes = [
        abs(float(np.dot(pts[int(a)], np.cross(pts[int(b)], pts[int(c)])))) / 6.0
        for a, b, c in np.asarray(faces, dtype=int)
    ]
    return float(np.sum(volumes))


def vertex_area_vectors_from_faces(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Lumped oriented boundary area vectors.

    These vectors are used only for pressure-equivalent diagnostics.  The
    dynamic pressure operator below uses tet-volume gradients.
    """
    pts = np.asarray(points, dtype=float)
    area_vectors = np.zeros_like(pts, dtype=float)
    for a, b, c in np.asarray(faces, dtype=int):
        a_i, b_i, c_i = int(a), int(b), int(c)
        pa, pb, pc = pts[a_i], pts[b_i], pts[c_i]
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        centroid = (pa + pb + pc) / 3.0
        if float(np.dot(area_vec, centroid)) < 0.0:
            area_vec = -area_vec
        share = area_vec / 3.0
        area_vectors[a_i] += share
        area_vectors[b_i] += share
        area_vectors[c_i] += share
    return area_vectors


def pressure_equivalent_from_forces(forces: np.ndarray, area_vectors: np.ndarray) -> float:
    """Least-squares scalar pressure balancing a surface force.

    Formula:

        p = -sum_i A_i . F_i / sum_i A_i . A_i
    """
    denom = float(np.sum(np.asarray(area_vectors, dtype=float) * np.asarray(area_vectors, dtype=float)))
    if denom <= 0.0:
        return 0.0
    return -float(np.sum(np.asarray(area_vectors, dtype=float) * np.asarray(forces, dtype=float))) / denom


def heron_forces_for_points(
    points: np.ndarray,
    faces: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return FHeron, area vectors, equivalent capillary pressure, volume."""
    forces, _areas = heron_surface_force_from_faces(points, faces, gamma)
    area_vectors = vertex_area_vectors_from_faces(points, faces)
    pressure = pressure_equivalent_from_forces(forces, area_vectors)
    return forces, area_vectors, pressure, mesh_volume_from_faces(points, faces)


def tet_cell_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Tetrahedron volumes for a simplicial volume mesh."""
    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int)
    p0 = pts[tet_arr[:, 0]]
    p1 = pts[tet_arr[:, 1]]
    p2 = pts[tet_arr[:, 2]]
    p3 = pts[tet_arr[:, 3]]
    triple = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
    return np.abs(triple) / 6.0


def tet_volume_matrix(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tet volumes and volume-gradient matrix B.

    B[t, 3*i:3*i+3] = dV_t / dx_i, so B u is the discrete tet-volume
    continuity operator.  This is the pressure operator used in the corrected
    droplet benchmark.

    References: Chorin (1968) and Temam projection methods for the role of
    a pressure constraint; here the discrete volume gradient replaces a grid
    divergence on the unstructured Lagrangian tet mesh.
    """
    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=int)
    volumes = np.zeros(tet_arr.shape[0], dtype=float)
    matrix = np.zeros((tet_arr.shape[0], pts.shape[0] * 3), dtype=float)
    for tet_idx, tet in enumerate(tet_arr):
        i0, i1, i2, i3 = [int(i) for i in tet]
        p0, p1, p2, p3 = pts[i0], pts[i1], pts[i2], pts[i3]
        triple = float(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0)))
        sign = 1.0 if triple >= 0.0 else -1.0
        volumes[tet_idx] = abs(triple) / 6.0
        g1 = sign * np.cross(p2 - p0, p3 - p0) / 6.0
        g2 = sign * np.cross(p3 - p0, p1 - p0) / 6.0
        g3 = sign * np.cross(p1 - p0, p2 - p0) / 6.0
        g0 = -(g1 + g2 + g3)
        for vertex_idx, gradient in ((i0, g0), (i1, g1), (i2, g2), (i3, g3)):
            start = 3 * vertex_idx
            matrix[tet_idx, start:start + 3] += gradient
    return volumes, matrix


def tet_volume_lumped_nodal_values(n_vertices: int, tets: np.ndarray, tet_values: np.ndarray) -> np.ndarray:
    """Barycentric tet-volume lumping: each tet contributes value/4.

    This is used after retopology to avoid HC-dual-volume jumps.  The sum of
    nodal values is exactly the sum of tet values.
    """
    nodal = np.zeros(int(n_vertices), dtype=float)
    for tet, value in zip(np.asarray(tets, dtype=int), np.asarray(tet_values, dtype=float)):
        share = 0.25 * float(value)
        for vertex_idx in tet:
            nodal[int(vertex_idx)] += share
    return nodal


def lump_tet_masses(n_vertices: int, tets: np.ndarray, volumes: np.ndarray, rho: float) -> np.ndarray:
    """Lump physical tet mass to vertices using barycentric tet-volume weights."""
    return np.maximum(tet_volume_lumped_nodal_values(n_vertices, tets, float(rho) * np.asarray(volumes)), 1.0e-18)


def mass_inverse_dof(masses: np.ndarray) -> np.ndarray:
    """Repeat nodal inverse masses over x/y/z degrees of freedom."""
    return np.repeat(1.0 / np.maximum(np.asarray(masses, dtype=float), 1.0e-300), 3)


def local_volume_stiffness(matrix: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return S = B M^-1 B^T and repeated inverse mass vector."""
    inv_mass_dof = mass_inverse_dof(masses)
    stiffness = (np.asarray(matrix, dtype=float) * inv_mass_dof[None, :]) @ np.asarray(matrix, dtype=float).T
    return stiffness, inv_mass_dof


def solve_regularized(matrix: np.ndarray, rhs: np.ndarray, *, relative: float = 1.0e-12) -> np.ndarray:
    """Solve a possibly singular pressure system with light Tikhonov damping."""
    lhs = np.asarray(matrix, dtype=float)
    rhs_array = np.asarray(rhs, dtype=float)
    scale = max(float(np.mean(np.abs(np.diag(lhs)))) if lhs.size else 0.0, 1.0e-30)
    regularized = lhs + float(relative) * scale * np.eye(lhs.shape[0], dtype=float)
    try:
        return np.linalg.solve(regularized, rhs_array)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(regularized, rhs_array, rcond=1.0e-12)[0]


def volume_gradient_pressure_matrix(volume_matrix: np.ndarray) -> np.ndarray:
    """Map tet pressures to nodal forces with F^p = B^T p."""
    return np.asarray(volume_matrix, dtype=float).T


def compressible_eos_pressure_correction(
    *,
    velocities: np.ndarray,
    nonpressure_forces: np.ndarray,
    masses: np.ndarray,
    tet_volumes: np.ndarray,
    target_tet_volumes: np.ndarray,
    reference_tet_volumes: np.ndarray,
    volume_matrix: np.ndarray,
    dt: float,
    bulk_modulus: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weak-compressible EOS pressure correction for tet volumes.

    Solves:

        (I + dt^2 D_K B M^-1 B^T) dp
          = -D_K (V - Vtar + dt B u*)

    Reference: pressure-based weakly compressible correction analogous to
    coupled pressure-density solvers; the discrete B operator is the local
    tet-volume continuity operator.
    """
    stiffness, inv_mass_dof = local_volume_stiffness(volume_matrix, masses)
    u_star = np.asarray(velocities, dtype=float) + float(dt) * (
        np.asarray(nonpressure_forces, dtype=float) / np.maximum(np.asarray(masses, dtype=float)[:, None], 1.0e-300)
    )
    compressibility = float(bulk_modulus) / np.maximum(np.asarray(reference_tet_volumes, dtype=float), 1.0e-300)
    linear_error = (
        np.asarray(tet_volumes, dtype=float)
        - np.asarray(target_tet_volumes, dtype=float)
        + float(dt) * (np.asarray(volume_matrix, dtype=float) @ u_star.reshape(-1))
    )
    system = np.eye(np.asarray(tet_volumes).shape[0]) + (compressibility[:, None] * float(dt) * float(dt)) * stiffness
    rhs = -compressibility * linear_error
    pressure_delta = solve_regularized(system, rhs)
    velocity_vec = u_star.reshape(-1) + float(dt) * inv_mass_dof * (volume_gradient_pressure_matrix(volume_matrix) @ pressure_delta)
    return velocity_vec.reshape(np.asarray(velocities).shape), pressure_delta


def incompressible_volume_projection(
    *,
    velocities: np.ndarray,
    nonpressure_forces: np.ndarray,
    masses: np.ndarray,
    tet_volumes: np.ndarray,
    target_tet_volumes: np.ndarray,
    volume_matrix: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local incompressible projection using B u^{n+1}=target_rates.

    Reference: Chorin (1968), Temam projection methods.  This implementation
    uses tet volume gradients, so the projected vertex velocity satisfies the
    local discrete volume/continuity constraint.
    """
    inv_mass_dof = mass_inverse_dof(masses)
    pressure_matrix = volume_gradient_pressure_matrix(volume_matrix)
    target_rates = -(np.asarray(tet_volumes, dtype=float) - np.asarray(target_tet_volumes, dtype=float)) / float(dt)
    velocity_vec = np.asarray(velocities, dtype=float).reshape(-1)
    nonpressure_vec = np.asarray(nonpressure_forces, dtype=float).reshape(-1)
    stiffness = (np.asarray(volume_matrix, dtype=float) * inv_mass_dof[None, :]) @ pressure_matrix
    rhs = (
        (target_rates - np.asarray(volume_matrix, dtype=float) @ velocity_vec) / float(dt)
        - np.asarray(volume_matrix, dtype=float) @ (inv_mass_dof * nonpressure_vec)
    )
    pressure_values = solve_regularized(stiffness, rhs)
    total_forces = np.asarray(nonpressure_forces, dtype=float) + (pressure_matrix @ pressure_values).reshape(np.asarray(velocities).shape)
    projected = np.asarray(velocities, dtype=float) + float(dt) * (
        inv_mass_dof * total_forces.reshape(-1)
    ).reshape(np.asarray(velocities).shape)
    return projected, pressure_values


def tet_face_pairs(tets: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    """Return internal tet faces as ``(face_vertices, left_tet, right_tet)``."""
    face_map: dict[tuple[int, int, int], list[int]] = {}
    for tet_idx, tet in enumerate(np.asarray(tets, dtype=int)):
        a, b, c, d = [int(index) for index in tet]
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            face_map.setdefault(tuple(sorted(face)), []).append(tet_idx)
    return [
        (np.asarray(face, dtype=int), int(owners[0]), int(owners[1]))
        for face, owners in face_map.items()
        if len(owners) == 2
    ]


def tet_face_ale_fluxes(
    points: np.ndarray,
    velocities: np.ndarray,
    tets: np.ndarray,
    face_pairs: list[tuple[np.ndarray, int, int]],
    tet_masses: np.ndarray,
    reference_tet_masses: np.ndarray,
    tet_volumes: np.ndarray,
    dt: float,
    *,
    reference_density: float,
    sound_speed: float,
    mass_floor_fraction: float = 0.35,
    mass_update_coupling: float = 0.02,
    momentum_force_coupling: float = 0.0,
    density_change_limit: float | None = None,
    flux_cfl_limit: float = 0.25,
    flux_scheme: str = "upwind",
    face_flux_mode: str = "cell-average",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Internal ALE face mass/momentum flux on a tet mesh.

    References: Hirt, Amsden & Cook (1974) ALE formulation; Toro (2009) for
    upwind/Rusanov numerical fluxes.  For ``face_flux_mode='lagrangian'`` the
    mesh-face velocity equals the material-face velocity, so internal material
    flux is zero; this is the physical setting for #5/#6/#11/#12.
    """
    if flux_scheme not in {"upwind", "rusanov"}:
        raise ValueError("flux_scheme must be 'upwind' or 'rusanov'")
    if face_flux_mode not in {"cell-average", "lagrangian"}:
        raise ValueError("face_flux_mode must be 'cell-average' or 'lagrangian'")

    n_tets = np.asarray(tets, dtype=int).shape[0]
    cell_centers = np.mean(np.asarray(points, dtype=float)[np.asarray(tets, dtype=int)], axis=1)
    cell_velocities = np.mean(np.asarray(velocities, dtype=float)[np.asarray(tets, dtype=int)], axis=1)
    cell_density = np.asarray(tet_masses, dtype=float) / np.maximum(np.asarray(tet_volumes, dtype=float), 1.0e-300)
    dm_dt = np.zeros(n_tets, dtype=float)
    cell_force = np.zeros((n_tets, 3), dtype=float)
    mass_flux_l1 = 0.0
    max_abs_mass_flux = 0.0
    momentum_flux_l1 = 0.0
    volume_flux_l1 = 0.0

    for face_vertices, left, right in face_pairs:
        pa, pb, pc = np.asarray(points, dtype=float)[face_vertices]
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        center_delta = cell_centers[right] - cell_centers[left]
        if float(np.dot(area_vec, center_delta)) < 0.0:
            area_vec = -area_vec
        area = float(np.linalg.norm(area_vec))
        if area <= 0.0:
            continue
        normal = area_vec / area
        mesh_face_velocity = np.mean(np.asarray(velocities, dtype=float)[face_vertices], axis=0)
        left_velocity = cell_velocities[left]
        right_velocity = cell_velocities[right]
        left_density = float(cell_density[left])
        right_density = float(cell_density[right])

        if face_flux_mode == "lagrangian":
            relative_volume_flux = 0.0
            mass_flux = 0.0
            momentum_flux = np.zeros(3, dtype=float)
        elif flux_scheme == "rusanov":
            left_relative_normal = float(np.dot(left_velocity - mesh_face_velocity, normal))
            right_relative_normal = float(np.dot(right_velocity - mesh_face_velocity, normal))
            max_wave_speed = max(
                abs(left_relative_normal) + float(sound_speed),
                abs(right_relative_normal) + float(sound_speed),
                1.0e-30,
            )
            mass_flux = area * (
                0.5 * (left_density * left_relative_normal + right_density * right_relative_normal)
                - 0.5 * max_wave_speed * (right_density - left_density)
            )
            momentum_flux = area * (
                0.5 * (left_density * left_velocity * left_relative_normal + right_density * right_velocity * right_relative_normal)
                - 0.5 * max_wave_speed * (right_density * right_velocity - left_density * left_velocity)
            )
            relative_volume_flux = mass_flux / max(0.5 * (left_density + right_density), 1.0e-300)
        else:
            fluid_face_velocity = 0.5 * (left_velocity + right_velocity)
            relative_volume_flux = float(np.dot(fluid_face_velocity - mesh_face_velocity, area_vec))
            if relative_volume_flux >= 0.0:
                density = left_density
                upwind_velocity = left_velocity
            else:
                density = right_density
                upwind_velocity = right_velocity
            mass_flux = density * relative_volume_flux
            momentum_flux = mass_flux * upwind_velocity

        dm_dt[left] -= mass_flux
        dm_dt[right] += mass_flux
        cell_force[left] -= momentum_flux
        cell_force[right] += momentum_flux
        mass_flux_l1 += abs(float(mass_flux))
        max_abs_mass_flux = max(max_abs_mass_flux, abs(float(mass_flux)))
        momentum_flux_l1 += float(np.linalg.norm(momentum_flux))
        volume_flux_l1 += abs(float(relative_volume_flux))

    effective_dm_dt = float(mass_update_coupling) * dm_dt
    effective_cell_force = float(momentum_force_coupling) * cell_force
    scale = 1.0
    mass_delta = float(dt) * effective_dm_dt
    mass_abs_delta = np.abs(mass_delta)
    if flux_cfl_limit > 0.0 and np.any(mass_abs_delta > 0.0):
        cfl_scale = np.min(
            float(flux_cfl_limit) * np.asarray(tet_masses, dtype=float)[mass_abs_delta > 0.0]
            / mass_abs_delta[mass_abs_delta > 0.0]
        )
        scale = min(scale, max(0.0, float(cfl_scale)))
    floors = float(mass_floor_fraction) * np.asarray(reference_tet_masses, dtype=float)
    proposed = np.asarray(tet_masses, dtype=float) + mass_delta
    floor_bad = proposed < floors
    if np.any(floor_bad):
        denom = -mass_delta[floor_bad]
        valid = denom > 0.0
        if np.any(valid):
            floor_scale = np.min((np.asarray(tet_masses)[floor_bad][valid] - floors[floor_bad][valid]) / denom[valid])
            scale = min(scale, max(0.0, float(floor_scale)))
    if density_change_limit is not None and density_change_limit > 0.0:
        lower_masses = (1.0 - float(density_change_limit)) * float(reference_density) * np.maximum(tet_volumes, 1.0e-300)
        upper_masses = (1.0 + float(density_change_limit)) * float(reference_density) * np.maximum(tet_volumes, 1.0e-300)
        increasing = (mass_delta > 0.0) & (proposed > upper_masses)
        if np.any(increasing):
            density_scale = np.min((upper_masses[increasing] - np.asarray(tet_masses)[increasing]) / mass_delta[increasing])
            scale = min(scale, max(0.0, float(density_scale)))
        decreasing = (mass_delta < 0.0) & (proposed < lower_masses)
        if np.any(decreasing):
            density_scale = np.min((lower_masses[decreasing] - np.asarray(tet_masses)[decreasing]) / mass_delta[decreasing])
            scale = min(scale, max(0.0, float(density_scale)))

    effective_dm_dt *= scale
    effective_cell_force *= scale
    updated_masses = np.maximum(np.asarray(tet_masses, dtype=float) + float(dt) * effective_dm_dt, floors)
    nodal_flux_force = np.zeros_like(np.asarray(points, dtype=float))
    for tet, force in zip(np.asarray(tets, dtype=int), effective_cell_force):
        for vertex_idx in tet:
            nodal_flux_force[int(vertex_idx)] += 0.25 * force
    stats = {
        "mass_flux_l1_kg_s": float(mass_flux_l1),
        "mass_flux_max_kg_s": float(max_abs_mass_flux),
        "momentum_flux_l1_n": float(momentum_flux_l1),
        "volume_flux_l1_m3_s": float(volume_flux_l1),
        "flux_limiter_scale": float(scale),
        "mass_update_coupling": float(mass_update_coupling),
        "momentum_force_coupling": float(momentum_force_coupling),
        "density_change_limit": float(density_change_limit) if density_change_limit is not None else 0.0,
        "flux_cfl_limit": float(flux_cfl_limit),
        "flux_scheme_rusanov": 1.0 if flux_scheme == "rusanov" else 0.0,
        "face_flux_lagrangian": 1.0 if face_flux_mode == "lagrangian" else 0.0,
    }
    return updated_masses, nodal_flux_force, stats


def delaunay_retriangulate_points(
    points: np.ndarray,
    fallback_tets: np.ndarray,
    *,
    qhull_options: str = "Qbb Qc",
    min_volume_fraction: float = 0.1,
) -> np.ndarray:
    """Fresh Delaunay tet list for active retopology.

    Reference: Delaunay retopology changes connectivity.  The benchmark
    recomputes B=dV/dx and remaps tet mass/target volume from the new tet
    volume list to avoid HC-dual-volume density jumps after flips.
    """
    try:
        from scipy.spatial import Delaunay

        triangulation = Delaunay(np.asarray(points, dtype=float), qhull_options=qhull_options)
        raw_tets = np.asarray(triangulation.simplices, dtype=int)
        valid = np.all(raw_tets < np.asarray(points).shape[0], axis=1)
        raw_tets = raw_tets[valid]
        if raw_tets.size == 0:
            return np.asarray(fallback_tets, dtype=int).copy()
        volumes = tet_cell_volumes(points, raw_tets)
        positive = volumes[volumes > 0.0]
        min_volume = float(min_volume_fraction) * float(np.median(positive)) if positive.size else 0.0
        filtered = raw_tets[volumes > min_volume]
        if filtered.size == 0:
            return np.asarray(fallback_tets, dtype=int).copy()
        return np.asarray(filtered, dtype=int)
    except Exception:
        return np.asarray(fallback_tets, dtype=int).copy()


def active_retopology_tet_remap(
    points: np.ndarray,
    current_tets: np.ndarray,
    *,
    density: float | None = None,
    target_total_mass: float | None = None,
    target_total_volume: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Rebuild tets and remap mass/target volumes after retopology.

    This is the controlled benchmark workaround for the HC dual-volume jump:
    after a connectivity flip, use the new tet volumes directly and assign
    m_t = rho V_t, V_t^tar = V_t.  It is not a conservative old-dual/new-dual
    overlap remap.
    """
    old_sorted = {tuple(sorted(int(i) for i in tet)) for tet in np.asarray(current_tets, dtype=int)}
    new_tets = delaunay_retriangulate_points(points, current_tets)
    new_sorted = {tuple(sorted(int(i) for i in tet)) for tet in np.asarray(new_tets, dtype=int)}
    changed = len(old_sorted.symmetric_difference(new_sorted))
    new_volumes = tet_cell_volumes(points, new_tets)
    current_volume = float(np.sum(new_volumes))
    if current_volume <= 1.0e-300:
        return np.asarray(current_tets, dtype=int).copy(), new_volumes, new_volumes, {
            "retriangulation_active": 1.0,
            "display_tet_count": float(np.asarray(current_tets).shape[0]),
            "display_internal_face_count": float(len(tet_face_pairs(current_tets))),
            "display_changed_tets": 0.0,
            "display_changed_fraction": 0.0,
        }
    if target_total_mass is not None:
        target_density = float(target_total_mass) / current_volume
    elif density is not None:
        target_density = float(density)
    else:
        raise ValueError("density or target_total_mass must be provided")
    target_volume_scale = (
        float(target_total_volume) / current_volume
        if target_total_volume is not None
        else 1.0
    )
    reference_tet_volumes = target_volume_scale * new_volumes
    tet_masses = target_density * new_volumes
    stats = {
        "retriangulation_active": 1.0,
        "display_tet_count": float(new_tets.shape[0]),
        "display_internal_face_count": float(len(tet_face_pairs(new_tets))),
        "display_changed_tets": float(changed),
        "display_changed_fraction": float(changed) / max(float(len(old_sorted) + len(new_sorted)), 1.0),
    }
    return new_tets, reference_tet_volumes, tet_masses, stats


# ---------------------------------------------------------------------------
# Benchmark add-on operators: multiphase tet-volume pressure closure
# ---------------------------------------------------------------------------

def multiphase_tet_volume_matrix(points: np.ndarray, tets: np.ndarray):
    """Tet volume-gradient operator for a multiphase simplicial mesh.

    This wrapper intentionally uses tet volumes, not HC local dual volumes, so
    a Delaunay flip does not directly create an HC-dual density/pressure jump.

    References: Chorin projection/continuity pressure solves; ddgclib
    retopology benchmark workaround for HC dual-volume jumps.
    """
    return tet_volume_matrix(points, tets)


def multiphase_tet_volume_lumped_masses(
    n_vertices: int,
    tets: np.ndarray,
    tet_volumes: np.ndarray,
    phases: np.ndarray,
    phase_densities: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return tet masses and barycentric nodal masses for phase-labelled tets.

    Each tet receives ``rho_phase V_t`` and each tet mass is lumped one quarter
    to its four vertices.
    """
    rho = np.asarray([float(phase_densities[int(phase)]) for phase in np.asarray(phases, dtype=int)], dtype=float)
    tet_masses = rho * np.asarray(tet_volumes, dtype=float)
    nodal_masses = tet_volume_lumped_nodal_values(n_vertices, tets, tet_masses)
    return tet_masses, nodal_masses


def multiphase_compressible_eos_pressure_correction(
    *,
    velocities: np.ndarray,
    nonpressure_forces: np.ndarray,
    inv_mass_dof: np.ndarray,
    tet_volumes: np.ndarray,
    target_tet_volumes: np.ndarray,
    reference_tet_volumes: np.ndarray,
    volume_matrix: np.ndarray,
    base_pressures: np.ndarray,
    bulk_by_tet: np.ndarray,
    dt: float,
    pressure_delta_limit: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiphase weak-compressible pressure correction.

    Solves the same EOS/continuity correction as the one-phase benchmark:

        (I + dt^2 D_K B M^-1 B^T) dp
          = -D_K (V - Vtar + dt B u*)

    where ``D_K = diag(K_t/V_t^0)`` may differ by phase.
    """
    pressure_matrix = np.asarray(volume_matrix, dtype=float).T
    u_star = np.asarray(velocities, dtype=float) + float(dt) * (
        np.asarray(inv_mass_dof, dtype=float) * np.asarray(nonpressure_forces, dtype=float).reshape(-1)
    ).reshape(np.asarray(velocities).shape)
    compressibility = np.asarray(bulk_by_tet, dtype=float) / np.maximum(
        np.asarray(reference_tet_volumes, dtype=float),
        1.0e-300,
    )
    stiffness = (np.asarray(volume_matrix, dtype=float) * np.asarray(inv_mass_dof, dtype=float)[None, :]) @ pressure_matrix
    linear_error = (
        np.asarray(tet_volumes, dtype=float)
        - np.asarray(target_tet_volumes, dtype=float)
        + float(dt) * (np.asarray(volume_matrix, dtype=float) @ u_star.reshape(-1))
    )
    system = np.eye(np.asarray(tet_volumes).shape[0]) + (compressibility[:, None] * float(dt) * float(dt)) * stiffness
    rhs = -compressibility * linear_error
    pressure_delta = solve_regularized(system, rhs)
    if pressure_delta_limit is not None and float(pressure_delta_limit) > 0.0:
        limit = float(pressure_delta_limit)
        pressure_delta = np.clip(pressure_delta, -limit, limit)
    velocity_vec = u_star.reshape(-1) + float(dt) * np.asarray(inv_mass_dof, dtype=float) * (pressure_matrix @ pressure_delta)
    return velocity_vec.reshape(np.asarray(velocities).shape), np.asarray(base_pressures, dtype=float) + pressure_delta


def multiphase_sparse_compressible_eos_pressure_correction(
    *,
    velocities: np.ndarray,
    nonpressure_forces: np.ndarray,
    mobility_matrix,
    tet_volumes: np.ndarray,
    target_tet_volumes: np.ndarray,
    reference_tet_volumes: np.ndarray,
    volume_matrix,
    base_pressures: np.ndarray,
    bulk_by_tet: np.ndarray,
    dt: float,
    diagonal_regularization: float = 0.0,
    stiffness_matrix=None,
    pressure_delta_limit: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sparse multiphase weak-compressible PR33 pressure correction.

    This is the sparse counterpart of
    ``multiphase_compressible_eos_pressure_correction`` for production meshes.
    It solves

        (I + dt^2 D_K B M^-1 B^T) dp
          = -D_K (V - Vtar + dt B u*)

    and returns the pressure force vector ``F^p = B^T (p_base + dp)``.
    ``mobility_matrix`` may be a full sparse mobility operator, including
    boundary projectors and fixed-node zero rows.
    """
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve

    volume_op = volume_matrix.tocsr()
    pressure_op = volume_op.T.tocsr()
    mobility = mobility_matrix.tocsr()
    velocities_array = np.asarray(velocities, dtype=float)
    n_tets = int(np.asarray(tet_volumes).shape[0])
    base_pressures_array = np.asarray(base_pressures, dtype=float)
    u_star_vec = velocities_array.reshape(-1) + float(dt) * np.asarray(
        mobility @ np.asarray(nonpressure_forces, dtype=float).reshape(-1),
        dtype=float,
    )
    if stiffness_matrix is None:
        stiffness = (volume_op @ mobility @ pressure_op).tocsr()
    else:
        stiffness = stiffness_matrix.tocsr()
    compressibility = np.asarray(bulk_by_tet, dtype=float) / np.maximum(
        np.asarray(reference_tet_volumes, dtype=float),
        1.0e-300,
    )
    linear_error = (
        np.asarray(tet_volumes, dtype=float)
        - np.asarray(target_tet_volumes, dtype=float)
        + float(dt) * np.asarray(volume_op @ u_star_vec, dtype=float)
    )
    system = (
        eye(n_tets, format="csr")
        + diags(compressibility * float(dt) * float(dt), 0, shape=(n_tets, n_tets)) @ stiffness
    ).tocsr()
    if float(diagonal_regularization) > 0.0:
        system = system + diags(
            np.full(n_tets, float(diagonal_regularization), dtype=float),
            0,
            shape=(n_tets, n_tets),
        )
    rhs = -compressibility * linear_error
    pressure_delta = np.asarray(spsolve(system, rhs), dtype=float)
    if pressure_delta_limit is not None and float(pressure_delta_limit) > 0.0:
        limit = float(pressure_delta_limit)
        pressure_delta = np.clip(pressure_delta, -limit, limit)
    pressure_total = base_pressures_array + pressure_delta
    pressure_force_vec = np.asarray(pressure_op @ pressure_total, dtype=float)
    velocity_vec = u_star_vec + float(dt) * np.asarray(mobility @ pressure_force_vec, dtype=float)
    return (
        velocity_vec.reshape(velocities_array.shape),
        pressure_total,
        pressure_delta,
        pressure_force_vec,
    )


def multiphase_incompressible_volume_projection(
    *,
    velocities: np.ndarray,
    nonpressure_forces: np.ndarray,
    inv_mass_dof: np.ndarray,
    tet_volumes: np.ndarray,
    target_tet_volumes: np.ndarray,
    volume_matrix: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiphase local incompressible projection with B u = target_rates.

    Reference: Chorin/Temam projection method, applied to tet-volume
    continuity on the moving multiphase mesh.
    """
    pressure_matrix = np.asarray(volume_matrix, dtype=float).T
    target_rates = -(np.asarray(tet_volumes, dtype=float) - np.asarray(target_tet_volumes, dtype=float)) / float(dt)
    velocity_vec = np.asarray(velocities, dtype=float).reshape(-1)
    nonpressure_vec = np.asarray(nonpressure_forces, dtype=float).reshape(-1)
    stiffness = (np.asarray(volume_matrix, dtype=float) * np.asarray(inv_mass_dof, dtype=float)[None, :]) @ pressure_matrix
    rhs = (
        (target_rates - np.asarray(volume_matrix, dtype=float) @ velocity_vec) / float(dt)
        - np.asarray(volume_matrix, dtype=float) @ (np.asarray(inv_mass_dof, dtype=float) * nonpressure_vec)
    )
    pressure_values = solve_regularized(stiffness, rhs)
    total_forces = np.asarray(nonpressure_forces, dtype=float) + (pressure_matrix @ pressure_values).reshape(np.asarray(velocities).shape)
    projected = np.asarray(velocities, dtype=float) + float(dt) * (
        np.asarray(inv_mass_dof, dtype=float) * total_forces.reshape(-1)
    ).reshape(np.asarray(velocities).shape)
    return projected, pressure_values


def multiphase_active_retopology_remap(
    points: np.ndarray,
    current_tets: np.ndarray,
    phases: np.ndarray,
    *,
    phase_densities: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Active retopology remap for phase-labelled tet meshes.

    The new Delaunay tet list is rebuilt, each new tet is assigned the phase
    of the nearest old tet center, and mass/target volume are recomputed from
    the new tet volume.  This is a controlled benchmark remap, not a fully
    conservative old/new-cell overlap remap.
    """
    pts = np.asarray(points, dtype=float)
    old_tets = np.asarray(current_tets, dtype=int)
    old_centers = np.mean(pts[old_tets], axis=1)
    new_tets = delaunay_retriangulate_points(pts, old_tets)
    new_centers = np.mean(pts[new_tets], axis=1)
    new_phases = np.zeros(new_tets.shape[0], dtype=int)
    for i, center in enumerate(new_centers):
        nearest = int(np.argmin(np.sum((old_centers - center) ** 2, axis=1)))
        new_phases[i] = int(np.asarray(phases, dtype=int)[nearest])
    new_volumes = tet_cell_volumes(pts, new_tets)
    rho = np.asarray([float(phase_densities[int(phase)]) for phase in new_phases], dtype=float)
    tet_masses = rho * new_volumes
    old_sorted = {tuple(sorted(int(v) for v in tet)) for tet in old_tets}
    new_sorted = {tuple(sorted(int(v) for v in tet)) for tet in new_tets}
    changed = len(old_sorted.symmetric_difference(new_sorted))
    stats = {
        "retriangulation_active": 1.0,
        "display_tet_count": float(new_tets.shape[0]),
        "display_internal_face_count": float(len(tet_face_pairs(new_tets))),
        "display_changed_tets": float(changed),
        "display_changed_fraction": float(changed) / max(float(len(old_sorted) + len(new_sorted)), 1.0),
    }
    return new_tets, new_phases, new_volumes, tet_masses, stats
