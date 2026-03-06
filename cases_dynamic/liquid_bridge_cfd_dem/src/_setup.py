"""Setup for the CFD-DEM two-particle liquid bridge case.

Creates the same DEM system as the pure-DEM case, plus surface-conforming
fluid film meshes (spherical shells around each particle) that evolve
dynamically under surface tension.

No volumetric cube mesh — the film IS the fluid domain.
"""

import numpy as np
from hyperct import Complex

from ddgclib.geometry import sphere, rotate_surface, translate_surface
from ddgclib.operators.surface_tension import dual_area_heron
from cases_dynamic.liquid_bridge_dem.src._setup import setup_liquid_bridge
from . import _params as P


def setup_liquid_bridge_cfd_dem(
    film_thickness=None, film_refinement=None,
    rho_f=None,
    **dem_kwargs,
):
    """Set up the coupled CFD-DEM liquid bridge scenario.

    Creates spherical film meshes around each particle within a single
    Complex object.  The films evolve under surface tension via the
    dynamic integrators (no Delaunay / compute_vd needed).

    Returns
    -------
    ps : ParticleSystem
    detector : ContactDetector
    contact_model : HertzContact
    bridge_mgr : LiquidBridgeManager
    HC_film : Complex
        Surface mesh containing both hemispherical films.
    bV_film : set
        Frozen (particle-attached) film vertices.
    """
    film_thickness = film_thickness if film_thickness is not None else P.film_thickness
    film_refinement = film_refinement if film_refinement is not None else P.film_refinement
    rho_f = rho_f if rho_f is not None else P.rho_f

    # ── DEM setup (identical to pure-DEM case) ────────────────────────
    ps, detector, contact_model, bridge_mgr = setup_liquid_bridge(**dem_kwargs)

    p1, p2 = ps.particles[0], ps.particles[1]

    # ── Film mesh: two hemispheres in a single Complex ────────────────
    HC_film, bV_film = _create_film_mesh(
        p1, p2,
        film_thickness=film_thickness,
        refinement=film_refinement,
        rho_f=rho_f,
    )

    return ps, detector, contact_model, bridge_mgr, HC_film, bV_film


def _create_film_mesh(
    p1, p2,
    film_thickness: float = 1e-5,
    refinement: int = 2,
    rho_f: float = 1000.0,
) -> tuple[Complex, set]:
    """Create the film surface mesh for both particles.

    Builds two hemispherical shells (one per particle), rotated to face
    each other, translated to particle positions, and merged into a
    single Complex.  The open rims face the gap between particles.

    Parameters
    ----------
    p1, p2 : Particle
        The two DEM particles.
    film_thickness : float
        Film thickness beyond particle surface [m].
    refinement : int
        Mesh refinement level.
    rho_f : float
        Fluid density [kg/m^3].

    Returns
    -------
    HC_film : Complex
        Single Complex containing both hemispheres.
    bV_film : set
        Frozen vertices (particle-attached, NOT the free rim).
    """
    R_film1 = p1.radius + film_thickness
    R_film2 = p2.radius + film_thickness

    # Particle-particle axis
    dx = p2.x_a[:3] - p1.x_a[:3]
    dist = np.linalg.norm(dx)
    axis = dx / dist if dist > 1e-15 else np.array([1.0, 0.0, 0.0])

    # Create hemisphere for p1 (open side faces p2)
    # phi_range: 0 = north pole (closed side), pi/2 = equator (open rim)
    # Use slightly more than hemisphere for better bridge coverage
    HC1, bV1 = sphere(
        R=R_film1, refinement=refinement,
        phi_range=(0.01, np.pi * 0.55),
    )
    # Rotate so the north pole (closed end) points AWAY from p2
    # The z-axis in the parametric sphere → (-axis) direction
    from ddgclib.geometry._parametric_surfaces import rotation_matrix_align
    R_mat1 = rotation_matrix_align(np.array([0.0, 0.0, 1.0]), -axis)
    rotate_surface(HC1, R_mat1)
    translate_surface(HC1, p1.x_a[:3].tolist())

    # Create hemisphere for p2 (open side faces p1)
    HC2, bV2 = sphere(
        R=R_film2, refinement=refinement,
        phi_range=(0.01, np.pi * 0.55),
    )
    R_mat2 = rotation_matrix_align(np.array([0.0, 0.0, 1.0]), axis)
    rotate_surface(HC2, R_mat2)
    translate_surface(HC2, p2.x_a[:3].tolist())

    # Merge into a single Complex by creating a new Complex and
    # adding all vertices from both hemispheres
    HC_film = Complex(3, domain=[])

    # Add vertices from HC1
    v1_map = {}  # old vertex -> new vertex
    for v in HC1.V:
        v_new = HC_film.V[tuple(v.x_a)]
        v_new.particle_id = p1.id
        v1_map[id(v)] = v_new

    # Copy connectivity from HC1
    for v in HC1.V:
        for nb in v.nn:
            v1_map[id(v)].connect(v1_map[id(nb)])

    # Add vertices from HC2
    v2_map = {}
    for v in HC2.V:
        v_new = HC_film.V[tuple(v.x_a)]
        v_new.particle_id = p2.id
        v2_map[id(v)] = v_new

    # Copy connectivity from HC2
    for v in HC2.V:
        for nb in v.nn:
            v2_map[id(v)].connect(v2_map[id(nb)])

    # Detect topological boundary of the merged mesh
    dV = HC_film.boundary()
    for v in HC_film.V:
        v.boundary = v in dV

    # Initialize fluid attributes
    for v in HC_film.V:
        v.u = np.zeros(3)
        v.p = np.array([0.0])
        # Mass from dual area
        C_i = dual_area_heron(v)
        v.m = rho_f * film_thickness * max(C_i, 1e-20)

    # Frozen vertices: those close to their particle center
    # (the "back" of each hemisphere, away from the gap)
    bV_film = set()
    for v in HC_film.V:
        pid = getattr(v, 'particle_id', -1)
        if pid == p1.id:
            center = p1.x_a[:3]
            R = p1.radius
        elif pid == p2.id:
            center = p2.x_a[:3]
            R = p2.radius
        else:
            continue

        # "Inner" vertices: distance from particle center close to R_film
        # These are NOT on the rim — they're the shell vertices
        # Freeze vertices that are far from the gap (back of hemisphere)
        r_vec = v.x_a[:3] - center
        r_norm = np.linalg.norm(r_vec)
        if r_norm < 1e-15:
            continue
        r_hat = r_vec / r_norm

        # Angle relative to the axis toward the other particle
        cos_angle = np.dot(r_hat, axis if pid == p1.id else -axis)
        # Freeze vertices on the back half (cos_angle < 0 means away from gap)
        if cos_angle < -0.3:
            bV_film.add(v)

    return HC_film, bV_film
