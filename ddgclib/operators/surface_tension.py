"""
Surface tension operators for discrete surface meshes.

Computes surface tension forces from mean curvature via the cotangent-weight
Heron's formula (``_curvatures_heron.py``).  These operators work directly
on the primal mesh and do NOT require ``compute_vd`` or dual mesh data.

The primary use case is thin-film surface meshes (e.g. spherical shells,
capillary bridges) where the standard volumetric stress operators cannot
be applied because ``compute_vd`` requires a volumetric (non-surface) mesh.

Usage
-----
    from functools import partial
    from ddgclib.operators.surface_tension import surface_tension_acceleration
    from ddgclib.dynamic_integrators import symplectic_euler

    dudt_fn = partial(surface_tension_acceleration, gamma=0.072, dim=3)
    symplectic_euler(HC, bV, dudt_fn, dt=1e-5, n_steps=100,
                     retopologize_fn=False)
"""

import numpy as np

from ddgclib._curvatures_heron import hndA_i


def surface_tension_force(v, gamma: float = 0.072, dim: int = 3) -> np.ndarray:
    """Surface tension force on vertex v from discrete mean curvature.

    Computes:

        F_st = -gamma * HNdA_i

    where ``HNdA_i`` is the integrated mean curvature normal vector from
    Heron's formula cotangent weights.  This is the discrete form of the
    Young-Laplace surface tension force, already integrated over the dual
    area of the vertex.

    Parameters
    ----------
    v : vertex object
        Must have ``v.x_a``, ``v.nn`` (1-ring neighbors).
    gamma : float
        Surface tension coefficient [N/m].
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Force vector, shape ``(dim,)``.
    """
    HNdA, C_i = hndA_i(v)
    return -gamma * HNdA[:dim]


def surface_tension_acceleration(
    v,
    gamma: float = 0.072,
    damping: float = 0.0,
    dim: int = 3,
    **kwargs,
) -> np.ndarray:
    """Acceleration from surface tension: a = F_st / m.

    Drop-in ``dudt_fn`` for dynamic integrators.  Works on surface meshes
    without ``compute_vd``.

    Parameters
    ----------
    v : vertex object
        Must have ``v.x_a``, ``v.nn``, ``v.m`` (mass).
    gamma : float
        Surface tension coefficient [N/m].
    damping : float
        Velocity damping coefficient for numerical stability.
        Adds ``-damping * v.u`` to the force.
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Acceleration vector, shape ``(dim,)``.
    """
    F = surface_tension_force(v, gamma=gamma, dim=dim)
    if damping > 0:
        F -= damping * v.u[:dim]
    return F / v.m


def dual_area_heron(v) -> float:
    """Dual area of vertex v from Heron's formula.

    Returns the second output of ``hndA_i`` — the sum of barycentric
    sub-cell areas around the vertex.

    Parameters
    ----------
    v : vertex object

    Returns
    -------
    float
        Dual area (sum of cotangent-weighted triangle sub-areas).
    """
    _, C_i = hndA_i(v)
    return C_i
