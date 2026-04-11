"""Integrated 2D curvature operators for interface curves.

This module implements the **integrated curvature normal** on a piecewise
linear 1D interface curve embedded in 2D.  The central identity is the
Fundamental Theorem of Calculus applied to the tangent vector:

    integrated_curvature_i  =  integral_{Gamma_i} kappa * N ds
                            =  T(end) - T(start)
                            =  t_next - t_prev

where ``Gamma_i`` is the portion of the interface curve inside the dual
cell of vertex ``v_i`` (the two half-edges that meet at ``v_i``), and
``t_prev``, ``t_next`` are the unit tangents along the previous and
next primal edges of the interface curve.

This is an *exact* identity on a piecewise linear curve: the tangent
is piecewise constant along each edge, so the integral of the curvature
normal over the dual cell collapses to the difference of the two edge
tangents bounding it.  For constant-curvature curves (the circle) this
reconstructs the arc length and enclosed area to machine precision,
see ``dev_notebooks/2D_machine_precision_area_from_curvatures``.

The surface tension force on an interface vertex is then obtained
directly from the integrated curvature:

    F_st_i = gamma * integral_{Gamma_i} kappa * N ds
           = gamma * (t_next - t_prev)

which is *already integrated over the dual edge* — there is no
"area per vertex" prefactor needed and no pointwise curvature
sampling.  The direction points toward the centre of curvature
(inward on a convex interface).

For an unoriented interface (such as the closed interface curve around
a droplet) the two curve neighbours of ``v_i`` are identified by
walking along interface edges via ``v.nn`` intersected with the set of
other interface vertices.
"""
from __future__ import annotations

import numpy as np


def _interface_neighbours(v) -> set:
    """Return the set of 1-ring neighbours of ``v`` that are themselves
    flagged as sharp-interface vertices."""
    return {nb for nb in v.nn if getattr(nb, 'is_interface', False)}


def _select_curve_neighbours(v, interface_nbs: set) -> tuple:
    """Pick the two curve neighbours ``(v_prev, v_next)`` of ``v`` along
    a closed 1D interface curve embedded in 2D.

    Strategy: sort the interface neighbours by polar angle around ``v``
    and pick the pair that brackets the **largest angular gap** — the
    gap that the curve does *not* cross.  This is robust for convex
    droplets and is the standard heuristic for 2D closed curves.

    For more than two interface neighbours this still picks the two
    that are most nearly colinear with ``v`` across the curve.
    """
    if len(interface_nbs) < 2:
        return None, None

    x_v = v.x_a[:2]

    def _angle(nb):
        d = nb.x_a[:2] - x_v
        return np.arctan2(d[1], d[0])

    sorted_nbs = sorted(interface_nbs, key=_angle)
    angles = [_angle(nb) for nb in sorted_nbs]
    n = len(sorted_nbs)

    max_gap = -1.0
    max_gap_idx = 0
    for i in range(n):
        if i < n - 1:
            gap = angles[i + 1] - angles[i]
        else:
            gap = angles[0] + 2 * np.pi - angles[-1]
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i

    v_prev = sorted_nbs[max_gap_idx]
    v_next = sorted_nbs[(max_gap_idx + 1) % n]
    return v_prev, v_next


def integrated_curvature_normal_2d(v, interface_nbs: set | None = None) -> np.ndarray:
    """Integrated curvature normal at a 2D interface vertex.

    Computes::

        integral_{Gamma_i} kappa * N ds = t_next - t_prev

    where ``t_prev`` and ``t_next`` are unit tangent vectors along the
    previous and next primal edges of the interface curve at ``v``.
    The result is a 2-vector pointing toward the centre of curvature
    (inward on a convex interface) with magnitude ``2 sin(theta/2)``,
    where ``theta`` is the exterior angle between the two edges.

    This quantity is the exact dual-integrated curvature for a piecewise
    linear curve — it is *not* a local sample of ``kappa`` at ``v`` nor
    an "average per unit length".  Multiplying by surface tension
    ``gamma`` gives the surface tension force on the vertex dual cell
    directly::

        F_st_i = gamma * integrated_curvature_normal_2d(v)

    Parameters
    ----------
    v : vertex
        Must have attributes ``x_a`` (position) and ``nn`` (1-ring
        neighbours).  The interface-neighbour set can be supplied
        externally via ``interface_nbs`` to avoid recomputing it.
    interface_nbs : set, optional
        Precomputed set of 1-ring interface neighbours of ``v``.  If
        ``None``, it is derived from ``v.nn`` by filtering on
        ``nb.is_interface``.

    Returns
    -------
    np.ndarray, shape (2,)
        The integrated curvature normal ``t_next - t_prev``.  Returns
        a zero vector if the curve neighbours cannot be identified.
    """
    if interface_nbs is None:
        interface_nbs = _interface_neighbours(v)
    if len(interface_nbs) < 2:
        return np.zeros(2)

    v_prev, v_next = _select_curve_neighbours(v, interface_nbs)
    if v_prev is None:
        return np.zeros(2)

    x_v = v.x_a[:2]
    e_prev = x_v - v_prev.x_a[:2]      # prev -> v
    e_next = v_next.x_a[:2] - x_v      # v -> next

    l_prev = float(np.linalg.norm(e_prev))
    l_next = float(np.linalg.norm(e_next))
    if l_prev < 1e-30 or l_next < 1e-30:
        return np.zeros(2)

    t_prev = e_prev / l_prev
    t_next = e_next / l_next
    return t_next - t_prev


def surface_tension_force_2d(
    v,
    gamma: float,
    interface_nbs: set | None = None,
) -> np.ndarray:
    """Surface tension force on a 2D interface vertex from integrated
    curvature.

    ::

        F_st_i = gamma * integral_{Gamma_i} kappa * N ds
               = gamma * (t_next - t_prev)

    Already integrated over the dual edge — no extra area/length factor.

    Parameters
    ----------
    v : vertex
    gamma : float
        Surface tension coefficient [N/m].
    interface_nbs : set, optional
        Precomputed interface neighbours of ``v``.

    Returns
    -------
    np.ndarray, shape (2,)
    """
    if gamma == 0.0:
        return np.zeros(2)
    return gamma * integrated_curvature_normal_2d(v, interface_nbs)


def reconstruct_arc_length_and_bulge_area(
    v_i: np.ndarray,
    v_j: np.ndarray,
    Delta_T: np.ndarray,
) -> tuple:
    """Closed-form arc length and bulge area from integrated curvature.

    Assumes constant curvature along the edge (a circular arc).  For an
    edge with endpoints ``v_i``, ``v_j`` and integrated curvature
    ``Delta_T`` (the change in unit tangent from start to end), this
    reconstructs the arc length ``L``, bulge area ``A_bulge``, radius
    of curvature ``r`` and chord length ``c`` to machine precision.

    Derivation (see ``dev_notebooks/2D_machine_precision_area_from_curvatures``):

        c       = |v_j - v_i|
        d       = |Delta_T|  =  2 sin(theta/2)
        theta   = arccos(1 - d**2 / 2)             # turning angle
        r       = c / d                            # circular radius
        L       = r * theta                        # arc length
        A_bulge = 0.5 * r**2 * (theta - sin(theta))

    Parameters
    ----------
    v_i, v_j : np.ndarray, shape (2,)
        Edge endpoints.
    Delta_T : np.ndarray, shape (2,)
        Integrated curvature vector (change in unit tangent).

    Returns
    -------
    (L, A_bulge, r, c) : tuple of floats
    """
    v_i = np.asarray(v_i, dtype=float)
    v_j = np.asarray(v_j, dtype=float)
    Delta_T = np.asarray(Delta_T, dtype=float)

    c = float(np.linalg.norm(v_j - v_i))
    d = float(np.linalg.norm(Delta_T))
    if d == 0.0:
        return c, 0.0, float('inf'), c

    r = c / d
    cos_theta = 1.0 - (d * d) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    L = r * theta
    A_bulge = 0.5 * r * r * (theta - np.sin(theta))
    return L, A_bulge, r, c
