"""Dynamic fluid film for the CFD-DEM liquid bridge case.

Manages surface meshes representing the thin wetting film around each
particle.  Unlike the previous static implementation, these films evolve
dynamically under surface tension using ddgclib's dynamic integrators.

Key components:

* **retopologize_surface** — surface mesh maintenance (remesh + boundary)
  that replaces the default Delaunay + compute_vd pipeline.
* **stokes_integral** — computes the capillary force on a particle by
  summing surface tension forces on the particle-attached film boundary.
* **detect_and_form_bridge** — connects rim vertices from opposite films
  when they approach within a threshold distance.
* **sync_film_to_particles** — moves particle-attached vertices with
  the DEM particles (no-slip condition).
* **snapshot** — serializes the film state for visualization.
"""

from __future__ import annotations

import numpy as np

from ddgclib._curvatures_heron import hndA_i
from ddgclib.operators.surface_tension import dual_area_heron


# ── Surface retopologize (replaces Delaunay + compute_vd) ───────────


def retopologize_surface(
    HC, bV, dim,
    *,
    min_edge: float = 1e-5,
    max_edge: float = 5e-4,
    particle_centers: list[np.ndarray] | None = None,
    particle_radii: list[float] | None = None,
    frozen_radius_tol: float = 0.05,
    rho_f: float = 1000.0,
    film_thickness: float = 1e-5,
):
    """Surface mesh maintenance: remesh + boundary update.

    Replaces ``_retopologize`` for surface meshes.  Does NOT use Delaunay
    triangulation or ``compute_vd``.

    Parameters
    ----------
    HC : Complex
        Film surface mesh.
    bV : set
        Boundary vertex set (modified in-place).
    dim : int
        Spatial dimension (3).
    min_edge, max_edge : float
        Edge length bounds for remeshing.
    particle_centers : list of (3,) arrays
        Current particle centers (for frozen vertex detection).
    particle_radii : list of float
        Particle radii.
    frozen_radius_tol : float
        Relative tolerance: vertices within ``R * (1 + frozen_radius_tol)``
        of a particle center are frozen (move with particle).
    rho_f : float
        Fluid density [kg/m^3].
    film_thickness : float
        Film thickness [m] for mass computation.
    """
    from ddgclib._bubble import remesh
    remesh(HC, min_edge, max_edge, bV)

    # Recompute topological boundary
    dV = HC.boundary()
    for v in HC.V:
        v.boundary = v in dV

    # Determine which boundary vertices are frozen (particle-attached)
    frozen = set()
    if particle_centers is not None and particle_radii is not None:
        for v in dV:
            for center, R in zip(particle_centers, particle_radii):
                dist = np.linalg.norm(v.x_a[:3] - center)
                if dist < R * (1.0 + frozen_radius_tol):
                    frozen.add(v)
                    break

    bV.clear()
    bV.update(frozen)

    # Recompute vertex masses from dual area
    for v in HC.V:
        C_i = dual_area_heron(v)
        v.m = rho_f * film_thickness * max(C_i, 1e-20)


# ── Stokes integral (capillary force on particle) ───────────────────


def stokes_integral(
    HC,
    particle_id: int,
    gamma: float,
    dim: int = 3,
) -> np.ndarray:
    """Compute the capillary force on a particle from its film.

    Sums the surface tension forces on all vertices tagged with the
    given ``particle_id``.  By Newton's third law, the film pulls the
    particle with force equal and opposite to the surface tension force
    on the boundary vertices:

        F_particle = -sum_v F_st(v) = sum_v gamma * HNdA_i(v)

    Parameters
    ----------
    HC : Complex
        Film surface mesh.
    particle_id : int
        ID of the particle (vertices must have ``v.particle_id`` set).
    gamma : float
        Surface tension [N/m].
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Force vector on the particle, shape ``(dim,)``.
    """
    F = np.zeros(dim)
    for v in HC.V:
        if getattr(v, 'particle_id', -1) == particle_id:
            HNdA, _ = hndA_i(v)
            # Opposite sign: F_st = -gamma*HNdA, so F_particle = +gamma*HNdA
            F += gamma * HNdA[:dim]
    return F


# ── Bridge formation via proximity detection ────────────────────────


def detect_and_form_bridge(
    HC,
    threshold: float,
    min_connections: int = 3,
) -> int:
    """Connect rim vertices from opposite films when close enough.

    Scans boundary vertices from different particles and connects pairs
    that are within ``threshold`` distance.  This creates bridge topology
    that surface tension will shape into a minimal-area bridge.

    Parameters
    ----------
    HC : Complex
        Film surface mesh (single Complex with both hemispheres).
    threshold : float
        Maximum distance for bridge connection [m].
    min_connections : int
        Minimum number of connections to form before declaring bridge.

    Returns
    -------
    int
        Number of new connections made.
    """
    # Collect rim vertices by particle_id
    rims: dict[int, list] = {}
    for v in HC.V:
        if v.boundary:
            pid = getattr(v, 'particle_id', -1)
            if pid >= 0:
                rims.setdefault(pid, []).append(v)

    particle_ids = sorted(rims.keys())
    if len(particle_ids) < 2:
        return 0

    n_connected = 0
    for i, pid_a in enumerate(particle_ids):
        for pid_b in particle_ids[i + 1:]:
            verts_a = rims[pid_a]
            verts_b = rims[pid_b]

            # Find close pairs
            for va in verts_a:
                best_dist = float('inf')
                best_vb = None
                for vb in verts_b:
                    d = np.linalg.norm(va.x_a[:3] - vb.x_a[:3])
                    if d < best_dist:
                        best_dist = d
                        best_vb = vb

                if best_vb is not None and best_dist < threshold:
                    # Check if already connected
                    if best_vb not in va.nn:
                        va.connect(best_vb)
                        n_connected += 1

    return n_connected


# ── Sync film to particle positions ─────────────────────────────────


def sync_film_to_particles(
    HC,
    particles: list,
    film_radius_factor: float = 1.01,
):
    """Move particle-attached film vertices to track particle motion.

    For each vertex tagged with a ``particle_id``, projects it onto
    the sphere of radius ``R_film = R * film_radius_factor`` centred
    at the current particle position, preserving its angular position.

    Parameters
    ----------
    HC : Complex
        Film surface mesh.
    particles : list
        Particle objects with ``id``, ``x_a``, ``radius``, ``u`` attributes.
    film_radius_factor : float
        Film radius = particle radius * factor.
    """
    # Build lookup
    p_lookup = {p.id: p for p in particles}

    for v in list(HC.V):
        pid = getattr(v, 'particle_id', -1)
        if pid < 0:
            continue
        p = p_lookup.get(pid)
        if p is None:
            continue

        center = p.x_a[:3]
        R_film = p.radius * film_radius_factor

        # Vector from particle center to vertex
        r_vec = v.x_a[:3] - center
        r_norm = np.linalg.norm(r_vec)

        if r_norm < 1e-15:
            continue

        # Project onto sphere of radius R_film
        new_pos = center + R_film * (r_vec / r_norm)

        # Update position
        full_pos = v.x_a.copy()
        full_pos[:3] = new_pos
        HC.V.move(v, tuple(full_pos))

        # No-slip: film velocity matches particle velocity
        v.u[:3] = p.u[:3]


# ── Snapshot for visualization ──────────────────────────────────────


def snapshot(HC, dim: int = 3) -> dict:
    """Capture the film mesh state for serialization / visualization.

    Returns
    -------
    dict with 'vertices', 'edges', 'particle_ids', 'curvature', 'boundary'.
    """
    v_list = list(HC.V)
    v_to_idx = {id(v): i for i, v in enumerate(v_list)}

    verts = [v.x_a[:dim].tolist() for v in v_list]
    velocities = [v.u[:dim].tolist() for v in v_list]
    particle_ids = [getattr(v, 'particle_id', -1) for v in v_list]
    boundary = [bool(v.boundary) for v in v_list]

    # Curvature (compute on the fly)
    curvature = []
    for v in v_list:
        try:
            HNdA, C_i = hndA_i(v)
            H_mag = float(np.linalg.norm(HNdA))
            if C_i > 1e-20:
                H_mag /= C_i
            curvature.append(H_mag)
        except Exception:
            curvature.append(0.0)

    # Edges
    edges = set()
    for v in v_list:
        for nb in v.nn:
            edge = tuple(sorted([v_to_idx[id(v)], v_to_idx[id(nb)]]))
            edges.add(edge)
    edges = [list(e) for e in sorted(edges)]

    return {
        'vertices': verts,
        'velocities': velocities,
        'edges': edges,
        'particle_ids': particle_ids,
        'curvature': curvature,
        'boundary': boundary,
        'n_vertices': len(verts),
        'n_edges': len(edges),
    }
