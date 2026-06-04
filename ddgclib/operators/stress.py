"""
Cauchy stress tensor operators for discrete fluid dynamics.

Face-centered integrated FVM formulation.  Forces on each Lagrangian
parcel (dual cell) are computed via Stokes' theorem as surface integrals
over dual flux planes:

    F_i = sum_j (F_p_ij + F_v_ij)

Pressure (face-average, conservative):
    F_p_ij = -0.5 * (p_i + p_j) * A_ij

Viscous (face-centered diffusion):
    F_v_ij = mu * (grad u)_f . A_ij
           = (mu / |d_ij|) * du * (d_hat . A_ij)

This is the "diffusion form" (mu * Laplacian u), which equals
div(mu * (grad u + grad u^T)) for incompressible flow (div u = 0).
The symmetric (transpose) term is omitted because the rank-1 face
gradient has spurious discrete compressibility on non-orthogonal edges.

The old pressure_gradient and velocity_laplacian are special cases:
    pressure-only: sigma = -p * I  (mu = 0)
    Laplacian-only: mu * grad(u) . A  (p = 0)

Additional diagnostic operators are provided for analytical comparison:
    velocity_difference_tensor — integrated Du_i (no /Vol)
    velocity_difference_tensor_pointwise — Du_i / Vol_i
    cauchy_stress — pointwise sigma from pointwise du
    integrated_cauchy_stress — volume-integrated sigma from integrated Du

Constitutive relation TODOs
---------------------------
# TODO: Add viscoelastic constitutive relation (Maxwell/Oldroyd-B)
#   sigma = -p*I + tau_elastic + tau_viscous
# TODO: Add non-Newtonian (power-law / Carreau) viscosity
#   mu_eff = K * |strain_rate|^(n-1)
# TODO: Add elastic solid constitutive relation (Hookean)
#   sigma = C : epsilon  (4th-order stiffness tensor)
# TODO: Add surface tension stress (interface parcels)
#   sigma += gamma * (I - n outer n) * kappa
"""

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
# Geometry: dual area vectors and dual volumes
# TODO: move dual_area_vector and dual_volume to hyperct.ddg (pure geometry)
# ---------------------------------------------------------------------------

def dual_area_vector(v_i, v_j, HC, dim: int = 3) -> np.ndarray:
    """Oriented dual area vector for the interface between parcels i and j.

    Computes A_ij, the total outward area vector of the dual face separating
    the dual cells of v_i and v_j.  In the continuum limit this is:

        A_ij = int_{S_ij} n dS

    where n is the outward unit normal from parcel i.

    In 2D the dual face is the line segment between the two shared dual
    vertices; A_ij is the outward-facing normal with magnitude equal to the
    segment length.

    In 3D the dual face is the DEC p_ij polygon: tet barycenters interleaved
    with face barycenters (x_i + x_j + x_k)/3.  This construction guarantees
    linear precision (machine eps) for barycentric duals on any tetrahedral
    mesh.  See :func:`_dual_area_vector_3d_p_ij`.  Falls back to the legacy
    e_star fan-walk (:func:`_dual_area_vector_3d_e_star`) for boundary or
    degenerate edges.

    Parameters
    ----------
    v_i, v_j : vertex objects
        Endpoints of the primal edge.  Must have ``v.vd`` populated.
    HC : Complex
        Simplicial complex with duals computed (``compute_vd``).
    dim : int
        Spatial dimension (1, 2, or 3).

    Returns
    -------
    np.ndarray
        Oriented area vector, shape ``(dim,)``.

    Notes
    -----
    # TODO: move to hyperct.ddg._operators (pure geometry, no physics)
    """
    if dim == 1:
        # In 1D the "area vector" is a signed scalar direction (+/- 1)
        # pointing outward from v_i along the primal edge
        direction = v_j.x_a[0] - v_i.x_a[0]
        return np.array([np.sign(direction)])

    elif dim == 2:
        # When periodic axes are set, always compute dual area from primal
        # geometry using minimum-image coordinates.  compute_vd's dual
        # vertex positions are wrong for any triangle that includes a
        # periodic-face vertex with wrapped neighbors.
        periodic_axes = getattr(HC, '_periodic_axes', None)
        if periodic_axes:
            periodic_bounds = HC._periodic_bounds
            x_i = v_i.x_a[:2]

            def _min_image(x_other):
                result = x_other.copy()
                for ax in periodic_axes:
                    p = periodic_bounds[ax][1] - periodic_bounds[ax][0]
                    delta = result[ax] - x_i[ax]
                    result[ax] -= round(delta / p) * p
                return result

            x_j = _min_image(v_j.x_a[:2])
            common = v_i.nn.intersection(v_j.nn)
            if len(common) < 2:
                # Boundary edge: use single triangle + edge midpoint
                if len(common) < 1:
                    return np.zeros(2)
                v3 = list(common)[0]
                x3 = _min_image(v3.x_a[:2])
                bary = (x_i + x_j + x3) / 3.0
                midpt = 0.5 * (x_i + x_j)
                dual_edge = bary - midpt
                A_ij = np.array([-dual_edge[1], dual_edge[0]])
                vec_to_i = x_i - 0.5 * (bary + midpt)
                if np.dot(A_ij, vec_to_i) > 0:
                    A_ij = -A_ij
                return A_ij
            # Interior edge: pick two triangles (one on each side of edge).
            # With ghost resolution, shared count can be >2. Use cross
            # product sign to find one neighbor on each side.
            edge_vec = x_j - x_i
            left = None
            right = None
            for v3 in common:
                x3 = _min_image(v3.x_a[:2])
                cross = edge_vec[0] * (x3[1] - x_i[1]) - edge_vec[1] * (x3[0] - x_i[0])
                if cross > 0 and left is None:
                    left = x3
                elif cross <= 0 and right is None:
                    right = x3
                if left is not None and right is not None:
                    break
            if left is None or right is None:
                return np.zeros(2)
            bary_l = (x_i + x_j + left) / 3.0
            bary_r = (x_i + x_j + right) / 3.0
            dual_edge = bary_l - bary_r
            A_ij = np.array([-dual_edge[1], dual_edge[0]])
            centroid = 0.5 * (bary_l + bary_r)
            vec_to_i = x_i - centroid
            if np.dot(A_ij, vec_to_i) > 0:
                A_ij = -A_ij
            return A_ij

        # Standard (non-periodic) path
        vdnn = v_i.vd.intersection(v_j.vd)
        vd_list = list(vdnn)
        if len(vd_list) < 2:
            # Degenerate: boundary edge with single dual vertex
            return np.zeros(2)
        vd1, vd2 = vd_list[0], vd_list[1]
        dual_edge = vd2.x_a[:2] - vd1.x_a[:2]
        # Normal to dual edge: rotation of the dual edge direction vector
        A_ij = np.array([-dual_edge[1], dual_edge[0]])
        # Orient outward from v_i
        centroid = 0.5 * (vd1.x_a[:2] + vd2.x_a[:2])
        vec_to_i = v_i.x_a[:2] - centroid
        if np.dot(A_ij, vec_to_i) > 0:
            A_ij = -A_ij
        return A_ij

    elif dim == 3:
        return _dual_area_vector_3d_p_ij(v_i, v_j, HC)

    else:
        raise NotImplementedError(f"dual_area_vector not implemented for dim={dim}")


def _dual_area_vector_3d_p_ij(v_i, v_j, HC) -> np.ndarray:
    """3D dual area vector using the DEC p_ij face construction.

    The dual face polygon interleaves **tet barycenters** with **face
    barycenters** ``(x_i + x_j + x_k) / 3`` of the primal triangular
    faces shared by consecutive tetrahedra around edge ``(i, j)``.

    This gives linear precision (machine epsilon) for barycentric duals
    on any tetrahedral mesh — a property that the simpler polygon of
    tet-barycenters-only does not satisfy because the non-planar face
    produces an incorrect area vector when triangulated without the
    intermediate face-barycenter vertices.

    Falls back to :func:`_dual_area_vector_3d_e_star` when the
    ring-walk or common-neighbor lookup fails (boundary / degenerate
    topologies).
    """
    shared_vd = v_i.vd.intersection(v_j.vd)
    if len(shared_vd) < 3:
        # Boundary or degenerate — fall back
        return _dual_area_vector_3d_e_star(v_i, v_j, HC)

    # --- Build ring order from dual vertex connectivity ---
    shared_list = list(shared_vd)
    ring = [shared_list[0]]
    remaining = set(shared_list[1:])
    while remaining:
        curr = ring[-1]
        nxt = None
        for cand in remaining:
            if cand in curr.nn:
                nxt = cand
                break
        if nxt is None:
            break
        ring.append(nxt)
        remaining.discard(nxt)

    if len(ring) < len(shared_list):
        # Connectivity walk incomplete — fall back to angular sort.
        # This happens for boundary-adjacent edges where dual vertices
        # have truncated connectivity.
        from hyperct.ddg._dual_cell import _angular_sort_3d
        sorted_ring = _angular_sort_3d(shared_list)
        if sorted_ring is not None and len(sorted_ring) >= 3:
            ring = sorted_ring
        elif len(ring) < 3:
            return _dual_area_vector_3d_e_star(v_i, v_j, HC)

    # --- Common neighbors = opposite vertices of shared triangular faces ---
    common_nbs = list(v_i.nn.intersection(v_j.nn))
    if not common_nbs:
        return _dual_area_vector_3d_e_star(v_i, v_j, HC)

    x_i = v_i.x_a[:3]
    x_j = v_j.x_a[:3]

    # --- Build interleaved polygon: tet_bary, face_bary, ... ---
    interleaved = []
    for k in range(len(ring)):
        tet_bary = ring[k].x_a[:3]
        tet_next = ring[(k + 1) % len(ring)].x_a[:3]
        interleaved.append(tet_bary)

        # Find the face barycenter (x_i + x_j + x_k)/3 between these
        # two consecutive tets.  The correct x_k is the common neighbor
        # whose face barycenter lies closest to the midpoint of the two
        # tet barycenters.
        mid = 0.5 * (tet_bary + tet_next)
        best_fb = None
        best_dist = np.inf
        for cn in common_nbs:
            fb = (x_i + x_j + cn.x_a[:3]) / 3.0
            dist = np.linalg.norm(fb - mid)
            if dist < best_dist:
                best_dist = dist
                best_fb = fb
        if best_fb is not None:
            interleaved.append(best_fb)

    polygon = np.array(interleaved)

    # --- Compute area vector by centroid-fan triangulation ---
    centroid = polygon.mean(axis=0)
    A_ij = np.zeros(3)
    n_pts = len(polygon)
    for k in range(n_pts):
        p1 = polygon[k]
        p2 = polygon[(k + 1) % n_pts]
        A_ij += 0.5 * np.cross(p1 - centroid, p2 - centroid)

    # --- Orient outward from v_i ---
    d_ij = x_j - x_i
    if np.dot(A_ij, d_ij) < 0:
        A_ij = -A_ij
    return A_ij


def _dual_area_vector_3d_e_star(v_i, v_j, HC) -> np.ndarray:
    """3D dual area vector via the e_star fan-walk (legacy).

    Uses the ``e_star`` fan-walk triangulation from the primal edge
    midpoint through the shared dual vertices.  This was the original
    implementation.  It does NOT satisfy linear precision for barycentric
    duals on non-symmetric meshes because the non-planar face polygon
    (tet barycenters only, without interleaved face barycenters) produces
    an incorrect area vector.

    Kept as fallback for boundary / degenerate topologies where the
    p_ij ring-walk cannot be performed.
    """
    from hyperct.ddg import e_star as _e_star
    try:
        A_ijk_arr = _e_star(v_i, v_j, HC, dim=3)  # shape (N, 3)
    except (IndexError, KeyError):
        return np.zeros(3)
    if not isinstance(A_ijk_arr, np.ndarray) or A_ijk_arr.size == 0:
        return np.zeros(3)
    vc_12_pos = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a
    vc_12 = HC.Vd[tuple(vc_12_pos)]
    vec_to_i = v_i.x_a - vc_12.x_a
    A_ij = np.zeros(3)
    for A_ijk in A_ijk_arr:
        if np.dot(A_ijk, vec_to_i) > 0:
            A_ijk = -A_ijk
        A_ij += A_ijk
    return A_ij


def dual_volume(v, HC, dim: int = 3) -> float:
    """Volume (area in 2D) of the dual cell around vertex v.

    The dual cell is the Voronoi-like polyhedron (3D) or polygon (2D) whose
    boundary consists of the dual faces separating v from each neighbor.

        Vol_i = dual cell measure of parcel i

    In 2D this is the dual cell area; in 3D the dual cell volume.

    Parameters
    ----------
    v : vertex object
        Must have ``v.nn`` and ``v.vd`` populated.
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension (1, 2, or 3).

    Returns
    -------
    float
        Dual cell volume (2D: area, 3D: volume).

    Notes
    -----
    # TODO: move to hyperct.ddg._operators (pure geometry, no physics)
    """
    if dim == 1:
        # 1D dual cell = interval between the two dual vertices
        vd_list = list(v.vd)
        if len(vd_list) < 2:
            # Boundary vertex with single dual vertex: half-edge
            if len(vd_list) == 1 and v.nn:
                v_j = next(iter(v.nn))
                return 0.5 * abs(v_j.x_a[0] - v.x_a[0])
            return 0.0
        positions = [vd.x_a[0] for vd in vd_list]
        return max(positions) - min(positions)

    elif dim == 2:
        from hyperct.ddg import dual_cell_area_2d
        return dual_cell_area_2d(v, include_edge_midpoints=True)

    elif dim == 3:
        from hyperct.ddg import v_star as _v_star
        total_vol = 0.0
        for v_j in v.nn:
            try:
                result = _v_star(v, v_j, HC, dim=3)
                if isinstance(result, tuple) and len(result) == 2:
                    _, V_ij = result
                    total_vol += np.sum(np.abs(V_ij))
                else:
                    # Scalar return (shouldn't happen in 3D)
                    total_vol += float(result)
            except (KeyError, IndexError, ValueError):
                continue
        return total_vol

    else:
        raise NotImplementedError(f"dual_volume not implemented for dim={dim}")


# ---------------------------------------------------------------------------
# Dual volume caching
# ---------------------------------------------------------------------------

def cache_dual_volumes(HC, dim: int = 3) -> None:
    """Compute and cache dual cell volumes on all vertices.

    Sets ``v.dual_vol = dual_volume(v, HC, dim)`` for every vertex in
    ``HC.V``.  Should be called after ``compute_vd`` (e.g. inside
    ``_retopologize``) so that operators can read ``v.dual_vol``
    instead of recomputing on the fly.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension.
    """
    for v in HC.V:
        try:
            v.dual_vol = dual_volume(v, HC, dim)
        except (ValueError, IndexError):
            # Degenerate vertex (e.g. domain corner with too few neighbors)
            v.dual_vol = 0.0


def _get_dual_vol(v, HC, dim: int = 3) -> float:
    """Return cached dual volume, computing it on demand if missing."""
    try:
        return v.dual_vol
    except AttributeError:
        v.dual_vol = dual_volume(v, HC, dim)
        return v.dual_vol


# ---------------------------------------------------------------------------
# Physics: velocity difference tensor, strain rate, stress
# ---------------------------------------------------------------------------

def velocity_difference_tensor(v, HC, dim: int = 3) -> np.ndarray:
    """Discrete integrated velocity difference tensor Du_i at vertex v.

    Computes the DDG volume-integrated quantity:

        Du_i = 0.5 * sum_j (u_j - u_i) outer A_ij

    This is analogous to int_{V_i} grad(u) dV (NOT divided by Vol_i).
    It is the natural integrated discrete form — not a pointwise gradient
    approximation.

    To get the pointwise gradient approximation, use
    :func:`velocity_difference_tensor_pointwise` or divide by
    ``v.dual_vol``.

    Parameters
    ----------
    v : vertex object
        Must have ``v.u`` (velocity ndarray), ``v.nn``, ``v.vd``.
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Integrated velocity difference tensor, shape ``(dim, dim)``.
        Component ``Du_i[a, b] = 0.5 * sum_j (u_j^a - u_i^a) * A_ij^b``.
    """
    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    Du_i = np.zeros((dim, dim))
    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim)
        delta_u = v_j.u[:dim] - v.u[:dim]
        Du_i += np.outer(delta_u, A_ij)
    Du_i *= 0.5
    return Du_i


def velocity_difference_tensor_pointwise(v, HC, dim: int = 3) -> np.ndarray:
    """Pointwise velocity gradient approximation at vertex v.

    Returns ``Du_i / Vol_i`` — the volume-averaged velocity gradient.
    Useful for comparison with analytical solutions.

    Parameters
    ----------
    v : vertex object
        Must have ``v.u``, ``v.nn``, ``v.vd``.
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Pointwise velocity gradient, shape ``(dim, dim)``.
    """
    Vol_i = _get_dual_vol(v, HC, dim)
    if Vol_i < 1e-30:
        return np.zeros((dim, dim))
    return velocity_difference_tensor(v, HC, dim) / Vol_i


def scalar_gradient_integrated(
    v,
    HC,
    dim: int = 3,
    field_attr: str = 'f',
) -> np.ndarray:
    """Integrated gradient of a scalar field over the dual cell of v.

    Computes the DDG volume-integrated quantity::

        Df_i = 0.5 * sum_j (f_j - f_i) * A_ij

    This is the scalar analog of :func:`velocity_difference_tensor`.
    It approximates ``∫_{V_i} ∇f dV``.

    Parameters
    ----------
    v : vertex object
        Must have the scalar field attribute (default ``v.f``) and
        ``v.nn``, ``v.vd`` populated.
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension.
    field_attr : str
        Name of the scalar field attribute on vertices (default ``'f'``).

    Returns
    -------
    np.ndarray
        Integrated gradient vector, shape ``(dim,)``.
    """
    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    f_i = getattr(v, field_attr)
    Df_i = np.zeros(dim)
    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim)
        delta_f = getattr(v_j, field_attr) - f_i
        Df_i += delta_f * A_ij
    Df_i *= 0.5
    return Df_i


def strain_rate(du: np.ndarray) -> np.ndarray:
    """Symmetric strain rate tensor from velocity difference tensor.

    Computes the symmetric part:

        epsilon = 0.5 * (du + du^T)

    For an incompressible Newtonian fluid, the deviatoric stress is:

        tau = 2 * mu * epsilon

    Parameters
    ----------
    du : np.ndarray
        Velocity difference tensor, shape ``(dim, dim)``.

    Returns
    -------
    np.ndarray
        Symmetric strain rate tensor, shape ``(dim, dim)``.
    """
    return 0.5 * (du + du.T)


def cauchy_stress(
    p: float,
    du: np.ndarray,
    mu: float,
    dim: int = 3,
) -> np.ndarray:
    """Cauchy stress tensor for a Newtonian fluid.

    Constitutive relation:

        sigma = -p * I + tau
        tau = 2 * mu * epsilon
        epsilon = 0.5 * (du + du^T)

    where p is the scalar pressure (positive in compression), mu is the
    dynamic viscosity, and du is the discrete integrated velocity difference
    tensor.

    Parameters
    ----------
    p : float
        Scalar pressure.
    du : np.ndarray
        Velocity difference tensor, shape ``(dim, dim)``.
    mu : float
        Dynamic viscosity [Pa.s].
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Cauchy stress tensor, shape ``(dim, dim)``.
    """
    return -p * np.eye(dim) + 2.0 * mu * strain_rate(du)


def integrated_cauchy_stress(
    p: float,
    Du: np.ndarray,
    mu: float,
    Vol_i: float,
    dim: int = 3,
) -> np.ndarray:
    """Integrated Cauchy stress tensor over the dual cell volume.

    Computes the volume-integrated stress:

        Sigma_int = -p * Vol_i * I + 2 * mu * strain_rate(Du)

    where ``Du`` is the integrated velocity difference tensor (NOT divided
    by volume).  This is the natural discrete quantity — the pointwise
    stress ``cauchy_stress`` is recovered by dividing by ``Vol_i``.

    Parameters
    ----------
    p : float
        Scalar pressure.
    Du : np.ndarray
        Integrated velocity difference tensor, shape ``(dim, dim)``.
        From :func:`velocity_difference_tensor` (without /Vol).
    mu : float
        Dynamic viscosity [Pa.s].
    Vol_i : float
        Dual cell volume.
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Integrated Cauchy stress tensor, shape ``(dim, dim)``.
    """
    return -p * Vol_i * np.eye(dim) + 2.0 * mu * strain_rate(Du)


def _resolve_pressure(v, pressure_model, HC, dim):
    """Resolve the pressure for a single vertex.

    Parameters
    ----------
    v : vertex object
    pressure_model : None, callable, or EquationOfState
        - ``None``: read ``v.p`` as-is (default, incompressible).
        - callable ``fn(v) -> float``: externally defined pressure field.
        - :class:`~ddgclib.eos.EquationOfState`: compute pressure from
          density ``rho = v.m / dual_vol`` via the EOS.  Also updates
          ``v.p`` and ``v.rho`` in-place so downstream code sees fresh
          values.
    HC : Complex
    dim : int

    Returns
    -------
    float
        Pressure at the vertex.
    """
    if pressure_model is None:
        p = v.p
        return float(p) if np.ndim(p) == 0 else float(p[0])

    if callable(pressure_model) and not hasattr(pressure_model, 'pressure'):
        # Plain callable: fn(v) -> float
        return float(pressure_model(v))

    # EquationOfState: P = eos.pressure(m / dual_vol)
    # Uses cached dual_vol (updated by _retopologize at each step).
    vol = _get_dual_vol(v, HC, dim)
    if vol < 1e-30:
        return float(pressure_model.pressure(pressure_model.rho0))
    rho = v.m / vol
    p = float(pressure_model.pressure(rho))
    # Update vertex in-place so other code (callbacks, diagnostics) sees it
    v.rho = rho
    v.p = p
    return p


# ---------------------------------------------------------------------------
# Factored flux primitives (shared by stress_force and multiphase_stress)
# ---------------------------------------------------------------------------

def pressure_flux(p_i: float, p_j: float, A_ij: np.ndarray) -> np.ndarray:
    """Face-average pressure flux: F_p_ij = -0.5 * (p_i + p_j) * A_ij."""
    return -0.5 * (p_i + p_j) * A_ij


def viscous_flux(
    mu: float,
    delta_u: np.ndarray,
    d_ij: np.ndarray,
    A_ij: np.ndarray,
) -> np.ndarray:
    """Face-centered viscous diffusion flux.

    F_v_ij = (mu / |d_ij|) * delta_u * (d_hat . A_ij)
    """
    d_norm = float(np.linalg.norm(d_ij))
    if d_norm < 1e-30:
        return np.zeros_like(A_ij)
    d_hat = d_ij / d_norm
    return (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)


def stress_force(v, dim: int = 3, mu: float = 8.9e-4, HC=None,
                 pressure_model=None) -> np.ndarray:
    """Integrated force on FVM via face-centered fluxes (Stokes' theorem).

    For each dual flux plane between parcels i and j, the force has two
    contributions computed directly from edge data:

    Pressure (face-average, conservative):

        F_p_ij = -0.5 * (p_i + p_j) * A_ij

    Viscous (face-centered diffusion):

        F_v_ij = mu * (grad u)_f . A_ij
               = (mu / |d_ij|) * du * (d_hat . A_ij)

    where du = u_j - u_i, d_hat = (x_j - x_i) / |x_j - x_i|.

    This is the "diffusion form" of the viscous term (mu * Laplacian u),
    which is equivalent to the full symmetric stress divergence
    div(mu * (grad u + grad u^T)) for incompressible flow (div u = 0).
    The symmetric (transpose) term mu * grad(div u) is omitted because
    the rank-1 face gradient has spurious discrete compressibility.

    Total: F_i = sum_j (F_p_ij + F_v_ij)

    Parameters
    ----------
    v : vertex object
        Must have ``v.p``, ``v.u``, ``v.nn``, ``v.vd``.
    dim : int
        Spatial dimension.
    mu : float
        Dynamic viscosity [Pa.s].
    HC : Complex
        Simplicial complex with duals computed.
    pressure_model : None, callable, or EquationOfState
        Controls how vertex pressure is obtained:

        - ``None`` (default): read ``v.p`` as-is (prescribed or constant).
        - callable ``fn(v) -> float``: externally defined pressure field,
          evaluated each time the force is computed.
        - :class:`~ddgclib.eos.EquationOfState`: weakly compressible
          pressure from density ``rho = m / dual_vol``.  Updates ``v.p``
          and ``v.rho`` in-place.

    Returns
    -------
    np.ndarray
        Force vector, shape ``(dim,)``.
    """
    p_i = _resolve_pressure(v, pressure_model, HC, dim)
    u_i = v.u[:dim]
    x_i = v.x_a[:dim]

    # Use cached oriented edge area vectors when available (set by
    # batch_e_star(..., orient=True) during retopologization).
    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    F = np.zeros(dim)
    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim)

        p_j = _resolve_pressure(v_j, pressure_model, HC, dim)
        delta_u = v_j.u[:dim] - u_i
        d_ij = v_j.x_a[:dim] - x_i
        F += pressure_flux(p_i, p_j, A_ij)
        F += viscous_flux(mu, delta_u, d_ij, A_ij)

    return F


def stress_acceleration(
    v,
    dim: int = 3,
    mu: float = 8.9e-4,
    HC=None,
    pressure_model=None,
) -> np.ndarray:
    """Acceleration from Cauchy stress: a_i = F_stress_i / m_i.

    Newton's second law on Lagrangian parcel i:

        m_i * dv_i/dt = F_stress_i + F_body
        a_i = F_stress_i / m_i

    This is a drop-in replacement for the old ``acceleration()`` function
    and can be used directly as ``dudt_fn`` for the dynamic integrators::

        from functools import partial
        dudt_fn = partial(stress_acceleration, dim=3, mu=1e-3, HC=HC)
        t = euler(HC, bV, dudt_fn, dt=1e-4, n_steps=100)

    For weakly compressible flow with an equation of state::

        from ddgclib.eos import TaitMurnaghan
        eos = TaitMurnaghan(rho0=1000.0, P0=101325.0)
        dudt_fn = partial(stress_acceleration, dim=2, mu=1e-3, HC=HC,
                          pressure_model=eos)

    Parameters
    ----------
    v : vertex object
        Must have ``v.p``, ``v.u``, ``v.m``, ``v.nn``, ``v.vd``.
    dim : int
        Spatial dimension.
    mu : float
        Dynamic viscosity [Pa.s].
    HC : Complex
        Simplicial complex with duals computed.
    pressure_model : None, callable, or EquationOfState
        See :func:`stress_force`.

    Returns
    -------
    np.ndarray
        Acceleration vector, shape ``(dim,)``.
    """
    return stress_force(v, dim=dim, mu=mu, HC=HC,
                        pressure_model=pressure_model) / v.m


# Simplified alias for use as dudt_fn in dynamic integrators
dudt_i = stress_acceleration
"""Alias for :func:`stress_acceleration`.

Provides simplified notation for the acceleration function used as
``dudt_fn`` in dynamic integrators::

    from ddgclib.operators.stress import dudt_i
    from ddgclib.dynamic_integrators import euler_velocity_only

    # Pass directly with keyword args forwarded by the integrator:
    euler_velocity_only(HC, bV, dudt_i, dt=1e-4, n_steps=100,
                        dim=2, mu=0.1, HC=HC)

    # Or bind parameters with functools.partial:
    from functools import partial
    dudt_fn = partial(dudt_i, dim=2, mu=0.1, HC=HC)
    euler_velocity_only(HC, bV, dudt_fn, dt=1e-4, n_steps=100)
"""
