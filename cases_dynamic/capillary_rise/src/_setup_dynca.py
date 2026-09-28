"""Setup for the data-driven (dynCA) capillary rise case.

Modern successor of ``_setup.py``: instead of a static-contact-angle
Washburn body force, the meniscus driving pressure is back-computed
from the experimental dynamic contact angle (see ``_dynamic_ca.py``),
and the reservoir is modelled as an Eulerian *mass-source band* below
the datum (y < 0) instead of a closed stretching column:

  - The tube extends from ``y_bot < 0`` (submerged inlet, reservoir
    band) to the initial meniscus at ``h0 > 0``; y = 0 is the dish
    (reservoir) level where gauge pressure is zero.
  - Each step, vertices inside the band get their mass reset to the
    hydrostatic EOS target ``m = rho_target(y) * dual_vol`` — a
    constant-pressure reservoir that supplies mass without topology
    injection.
  - In 2D, the mass-conserving adaptive remesh (``hyperct.remesh``)
    splits band edges as the column pulls new fluid upward, keeping
    resolution without global retriangulation (the ``dual_only``
    philosophy from the oscillating-droplet campaign).
  - Wall vertices are frozen (no-slip) except the topmost wall vertex
    on each wall line — the contact line — which slides along the wall
    (wall-normal velocity zero, wall-parallel free).

The 2D slit is *matched* to the 3D tube (half-width a = R/2 and
mu_2d = 2/3 mu, see ``_dynamic_ca.matched_slit_params``) so its
reduced-order dynamics are identical to the experimental tube.
"""
from __future__ import annotations

import numpy as np

from hyperct import Complex
from hyperct.ddg import compute_vd, connect_and_cache_simplices

from ddgclib.eos import TaitMurnaghan
from ddgclib.operators.stress import cache_dual_volumes


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------

def build_strip_2d(width: float, y_bot: float, y_top: float, nx: int):
    """Structured near-isotropic triangulated strip with simplex cache.

    ``nx`` cells across the width; the vertical spacing matches the
    horizontal spacing.  Registering the triangles explicitly activates
    the exact barycentric dual-volume path.
    """
    dx = width / nx
    ny = max(1, round((y_top - y_bot) / dx))
    dy = (y_top - y_bot) / ny
    HC = Complex(2, domain=[(0.0, width), (y_bot, y_top)])
    idx = {}
    verts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            v = HC.V[(i * dx, y_bot + j * dy)]
            idx[(i, j)] = len(verts)
            verts.append(v)
    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b = idx[(i, j)], idx[(i + 1, j)]
            c, d = idx[(i + 1, j + 1)], idx[(i, j + 1)]
            if (i + j) % 2 == 0:
                tris.append((a, b, c)); tris.append((a, c, d))
            else:
                tris.append((a, b, d)); tris.append((b, c, d))
    connect_and_cache_simplices(HC, verts, 2, simplices=np.array(tris))
    return HC, {'dx': dx, 'dy': dy, 'nx': nx, 'ny': ny}


def build_tube_3d(R: float, z_bot: float, z_top: float, n_rings: int = 3):
    """3D cylinder from ``z_bot`` to ``z_top`` built as extruded disk layers.

    ``cylinder_volume`` refines its bounding box uniformly, which for a
    long thin tube produces ~15:1 anisotropic tets.  Instead we stack
    near-isotropic structured layers (hex-pattern disk cross-section,
    layer spacing = radial spacing) and Delaunay-triangulate once — the
    domain is convex, so a single global triangulation at setup is safe
    and no per-step retopology is used (``dual_only`` philosophy).
    """
    # Hex-ring disk template
    pts2d = [(0.0, 0.0)]
    for k in range(1, n_rings + 1):
        rk = R * k / n_rings
        n_az = 6 * k
        for j in range(n_az):
            phi = 2.0 * np.pi * (j + 0.5 * (k % 2)) / n_az
            pts2d.append((rk * np.cos(phi), rk * np.sin(phi)))
    dz = R / n_rings
    nz = max(2, round((z_top - z_bot) / dz))
    dz = (z_top - z_bot) / nz

    HC = Complex(3, domain=[(-R, R), (-R, R), (z_bot, z_top)])
    verts = []
    coords = []
    for iz in range(nz + 1):
        z = z_bot + iz * dz
        for (x, y) in pts2d:
            verts.append(HC.V[(x, y, z)])
            coords.append((x, y, z))
    connect_and_cache_simplices(HC, verts, 3, coords=np.array(coords))
    # The legacy 3D dual walk requires the TOPOLOGICAL boundary (hull),
    # not positional tags — tag it exactly from the simplex cache.
    from hyperct.ddg import boundary_from_simplices
    bset = set(boundary_from_simplices(HC, 3))
    for v in HC.V:
        v.boundary = v in bset
    return HC, {'dx': dz, 'n_layer': len(pts2d), 'nz': nz, 'bset': bset}


# ---------------------------------------------------------------------------
# Positional group tagging (rebuilt every step — remesh-safe)
# ---------------------------------------------------------------------------

def tag_groups_2d(HC, width: float, y_bot: float, tol: float):
    """Positionally identify walls / bottom / contact line / free surface.

    Sets ``v.boundary`` on all vertices.  Returns dict of vertex sets:
    ``wall`` (excl. contact line), ``contact`` (topmost wall vertex per
    wall line), ``bottom``, ``frozen`` (wall + bottom), ``surface``
    (vertices within one row of the maximum height).
    """
    left, right, bottom = set(), set(), set()
    y_max = -np.inf
    for v in HC.V:
        x, y = v.x_a[0], v.x_a[1]
        if y > y_max:
            y_max = y
        if x < tol:
            left.add(v)
        elif x > width - tol:
            right.add(v)
        if y < y_bot + tol:
            bottom.add(v)

    contact = set()
    for side in (left, right):
        wall_only = side - bottom
        if wall_only:
            contact.add(max(wall_only, key=lambda v: v.x_a[1]))

    wall = (left | right) - contact
    frozen = wall | bottom

    # Free surface: contact line + everything within ~one row of the top
    row_tol = 0.6 * _top_row_spacing(HC, axis=1)
    surface = {v for v in HC.V if v.x_a[1] > y_max - row_tol} | contact

    all_b = left | right | bottom | surface
    for v in HC.V:
        v.boundary = v in all_b
    return {'wall': wall, 'contact': contact, 'bottom': bottom,
            'frozen': frozen, 'surface': surface, 'y_max': y_max}


def tag_groups_3d(HC, R: float, z_bot: float, tol: float):
    """3D analogue of ``tag_groups_2d`` for a vertical cylinder."""
    wall_all, bottom = set(), set()
    z_max = -np.inf
    for v in HC.V:
        x, y, z = v.x_a[0], v.x_a[1], v.x_a[2]
        if z > z_max:
            z_max = z
        if np.hypot(x, y) > R - tol:
            wall_all.add(v)
        if z < z_bot + tol:
            bottom.add(v)

    row_tol = 0.6 * _top_row_spacing(HC, axis=2)
    contact = {v for v in (wall_all - bottom) if v.x_a[2] > z_max - row_tol}
    wall = wall_all - contact
    frozen = wall | bottom
    surface = {v for v in HC.V if v.x_a[2] > z_max - row_tol} | contact

    all_b = wall_all | bottom | surface
    for v in HC.V:
        v.boundary = v in all_b
    return {'wall': wall, 'contact': contact, 'bottom': bottom,
            'frozen': frozen, 'surface': surface, 'y_max': z_max}


def _top_row_spacing(HC, axis: int) -> float:
    """Median vertical spacing among the topmost two rows of vertices."""
    ys = sorted({round(float(v.x_a[axis]), 12) for v in HC.V}, reverse=True)
    if len(ys) < 2:
        return 1e-12
    return max(ys[0] - ys[1], 1e-12)


# ---------------------------------------------------------------------------
# EOS / ICs / reservoir band
# ---------------------------------------------------------------------------

def make_eos(rho0: float, c0: float,
             rho_clip: tuple[float, float] = (0.3, 3.0)) -> TaitMurnaghan:
    """Gauge-pressure TaitMurnaghan (n=1) with a wide clip window.

    The window is deliberately wider than the droplet default: the
    contact-corner boundary layer stretches strongly while the wall
    shear develops, and the EOS must keep supplying *tension* there —
    saturating the clip zeroes the restoring stiffness and lets the
    stretch run away (observed in the dynCA smoke tests at clip 0.5).
    """
    return TaitMurnaghan(rho0=rho0, P0=0.0, K=rho0 * c0 * c0, n=1.0,
                         rho_clip=rho_clip)


def hydrostatic_density(y, eos: TaitMurnaghan, rho0: float, g: float):
    """EOS-consistent density at height y (P = 0 gauge at datum y = 0)."""
    alpha = rho0 * g / eos.K
    P = eos.K * (np.exp(alpha * np.maximum(-y, 0.0)) - 1.0)
    return eos.density(P), P

def band_mass_reset(HC, eos, rho0: float, g: float, gravity_axis: int,
                    y_datum: float = 0.0) -> float:
    """Reset mass in the reservoir band (y < y_datum) to hydrostatic target.

    Emulates a constant-pressure reservoir feeding the tube.  Returns
    the net mass added this call (for the injected-mass diagnostic).
    """
    added = 0.0
    for v in HC.V:
        y = v.x_a[gravity_axis]
        if y < y_datum:
            vol = getattr(v, 'dual_vol', 0.0)
            if vol <= 0.0:
                continue
            rho_t, P_t = hydrostatic_density(y, eos, rho0, g)
            m_new = rho_t * vol
            added += m_new - getattr(v, 'm', 0.0)
            v.m = m_new
            v.p = P_t
    return added


def boundary_mass_reset(HC, verts, eos, rho0: float, g: float,
                        gravity_axis: int) -> float:
    """Prescribe hydrostatic target density on frozen boundary vertices.

    Frozen (no-slip wall / floor) vertices are boundary-condition
    carriers, not material parcels: their mass is fixed while their dual
    cells stretch with the passing flow, so their EOS density drifts and
    injects spurious wall tension.  Resetting to the hydrostatic target
    each step (the weakly-compressible dummy-wall-particle treatment)
    keeps their pressure physical.  Returns net mass added.
    """
    added = 0.0
    for v in verts:
        vol = getattr(v, 'dual_vol', 0.0)
        if vol <= 0.0:
            continue
        y = v.x_a[gravity_axis]
        rho_t, P_t = hydrostatic_density(y, eos, rho0, g)
        m_new = rho_t * vol
        added += m_new - getattr(v, 'm', 0.0)
        v.m = m_new
        v.p = P_t
    return added


def apply_ics(HC, dim: int, rho0: float, eos, g: float, half_width: float,
              hdot0: float, y_bot: float, wall_tol: float):
    """Initial mass (uniform rho0 above datum, hydrostatic below) and a
    developed laminar velocity profile with mean ``hdot0``.

    2D slit peak factor 1.5, 3D tube peak factor 2.  The profile ramps
    to zero over the bottom 20% of the band (fixed floor).
    """
    gravity_axis = dim - 1
    for v in HC.V:
        y = v.x_a[gravity_axis]
        vol = getattr(v, 'dual_vol', 0.0)
        if y < 0.0:
            rho_t, P_t = hydrostatic_density(y, eos, rho0, g)
        else:
            rho_t, P_t = rho0, 0.0
        v.m = rho_t * max(vol, 0.0)
        v.p = P_t

        if dim == 2:
            xi = (v.x_a[0] - half_width) / half_width      # -1..1
            u_par = 1.5 * hdot0 * (1.0 - xi * xi)
            on_wall = (v.x_a[0] < wall_tol
                       or v.x_a[0] > 2 * half_width - wall_tol)
        else:
            rr = np.hypot(v.x_a[0], v.x_a[1]) / half_width  # 0..1
            u_par = 2.0 * hdot0 * max(1.0 - rr * rr, 0.0)
            on_wall = np.hypot(v.x_a[0], v.x_a[1]) > half_width - wall_tol

        ramp = min(1.0, (y - y_bot) / max(0.2 * (-y_bot), 1e-12)) if y < 0 else 1.0
        v.u = np.zeros(dim)
        if not on_wall:
            v.u[gravity_axis] = u_par * max(ramp, 0.0)


# ---------------------------------------------------------------------------
# Resolved free-surface machinery (2D)
# ---------------------------------------------------------------------------

def extract_surface_chain(HC, width: float, y_bot: float, tol: float):
    """Ordered free-surface polyline (left contact -> right contact).

    Boundary edges are those with exactly one incident triangle (from
    the fresh simplex cache); surface edges are boundary edges that lie
    neither on a wall line nor on the floor.  Returns the ordered vertex
    list, or None if the chain is broken.
    """
    simplices = getattr(HC, '_simplices', None)
    if not simplices:
        return None
    from collections import defaultdict
    edge_count = defaultdict(int)
    for tri in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge_count[frozenset((id(tri[i]), id(tri[j])))] += 1
    by_id = {id(v): v for v in HC.V}

    def on_wall(v):
        return v.x_a[0] < tol or v.x_a[0] > width - tol

    def on_floor(v):
        return v.x_a[1] < y_bot + tol

    surf_ids = set()
    contacts = {}
    for e, cnt in edge_count.items():
        if cnt != 1:
            continue
        va, vb = (by_id[i] for i in e)
        if on_floor(va) or on_floor(vb):
            continue
        if on_wall(va) and on_wall(vb):
            continue
        for v in (va, vb):
            if on_wall(v):
                # candidate contact vertex: keep topmost per side
                side = 0 if v.x_a[0] < 0.5 * width else 1
                cur = contacts.get(side)
                if cur is None or v.x_a[1] > cur.x_a[1]:
                    contacts[side] = v
            else:
                surf_ids.add(id(v))
    if len(contacts) != 2 or not surf_ids:
        return None

    # The pre-breakdown meniscus has no overhangs: order by x.  This is
    # robust to local non-manifold junk near the corners, which breaks
    # a strict edge walk.
    interior = sorted((by_id[i] for i in surf_ids), key=lambda v: v.x_a[0])
    return [contacts[0]] + interior + [contacts[1]]


def surface_tension_forces(chain, gamma: float, cos_theta: float):
    """Discrete line-tension forces on the free-surface polyline [N/m].

    Each surface edge pulls its endpoints together with tension
    ``gamma`` (per unit depth), giving the exact discrete curvature
    force ``gamma * (t_next - t_prev)`` on interior chain vertices.  At
    the two contact vertices the wall-side pull is the *data-driven*
    three-phase contact force ``gamma * cos_theta`` directed up the
    wall — the measured dynamic contact angle imposed as a force, in
    place of a contact-line model.
    """
    F = {}
    n = len(chain)
    for i, v in enumerate(chain):
        f = np.zeros(2)
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                d = chain[j].x_a[:2] - v.x_a[:2]
                L = float(np.linalg.norm(d))
                if L > 0.0:
                    f += gamma * d / L
        if i == 0 or i == n - 1:
            f[1] += gamma * cos_theta      # wall-side contact-line pull
        else:
            # Project onto the local surface normal: the tangential
            # component of polyline tension only bunches marker
            # vertices along the interface (no Marangoni stresses for
            # constant gamma) — standard front-tracking practice.
            tvec = chain[i + 1].x_a[:2] - chain[i - 1].x_a[:2]
            Lt = float(np.linalg.norm(tvec))
            if Lt > 0.0:
                nvec = np.array([-tvec[1], tvec[0]]) / Lt
                f = float(f @ nvec) * nvec
        F[id(v)] = f
    return F


# ---------------------------------------------------------------------------
# dudt wrapper: stress + gravity + data-driven capillary body force
# ---------------------------------------------------------------------------

def make_dudt_dynca(dim: int, mu: float, HC, eos, rho0: float, g: float,
                    p_cap_fn, state: dict, drive_mode: str = 'surface'):
    """dudt = stress/m + g + data-driven capillary drive.

    drive='surface' (resolved): line-tension forces on the free-surface
    chain plus the data-driven contact force are supplied by the driver
    via ``state['F_surf']`` (dict id(v) -> force per unit depth); the
    Young-Laplace driving pressure then *emerges* from the resolved
    meniscus curvature.

    drive='body' (reduced): Washburn-equivalent body force
    ``a_cap = P_cap(t) / (rho0 * h)`` on vertices above the datum, with
    ``P_cap(t)`` back-computed from the measured dynamic contact angle.

    ``state`` is a mutable dict holding driver-updated keys ``'t'``
    (sim time mapped to experiment time), ``'h'`` (current meniscus
    height), ``'p_cap'`` and ``'F_surf'``.
    """
    from functools import partial
    from ddgclib.operators.stress import stress_acceleration

    gravity_axis = dim - 1
    g_vec = np.zeros(dim)
    g_vec[gravity_axis] = -g
    stress_fn = partial(stress_acceleration, dim=dim, mu=mu, HC=HC,
                        pressure_model=eos)

    def dudt_fn(v):
        a = stress_fn(v) + g_vec
        # Surface line-tension forces: full data-driven drive in
        # 'surface' mode; pure cohesion regularizer (slave angle 90
        # deg, zero net pull) in 'body' mode.
        f = state['F_surf'].get(id(v))
        if f is not None and v.m > 0.0:
            a[:2] = a[:2] + f / v.m
        if drive_mode == 'body' and v.x_a[gravity_axis] > 0.0:
            h = max(state['h'], 1e-9)
            a[gravity_axis] += state['p_cap'] / (rho0 * h)
        return a

    def update_state(t_exp: float, h_now: float, F_surf=None):
        state['t'] = t_exp
        state['h'] = h_now
        state['p_cap'] = float(p_cap_fn(t_exp))
        state['F_surf'] = F_surf if F_surf is not None else {}

    return dudt_fn, update_state
