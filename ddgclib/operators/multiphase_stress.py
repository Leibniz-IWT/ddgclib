"""Multiphase stress operators for the integrated Cauchy momentum equation.

Each dual cell belongs to a single phase and uses that phase's viscosity.
Cross-phase edges (between an interface vertex and an outer-phase vertex)
use the viscosity of the dual cell's own phase.  There is no harmonic mean.

Interface vertices additionally receive the surface tension force from
curvature of the sharp interface.

This module provides ``multiphase_dudt_i`` as a drop-in replacement
for ``dudt_i`` in dynamic integrators.

Usage
-----
    from functools import partial
    from ddgclib.operators.multiphase_stress import multiphase_dudt_i

    dudt_fn = partial(multiphase_dudt_i, dim=2, mps=mps, HC=HC,
                      pressure_model=meos)
    symplectic_euler(HC, bV, dudt_fn, dt=1e-4, n_steps=100, ...)
"""
from __future__ import annotations

import numpy as np

from ddgclib.operators.stress import dual_area_vector, _resolve_pressure
from ddgclib.operators.curvature_2d import surface_tension_force_2d


def multiphase_stress_force(
    v,
    dim: int = 3,
    mps=None,
    HC=None,
    pressure_model=None,
) -> np.ndarray:
    """Integrated force on a dual cell with multiphase physics.

    The force on vertex *v* is:

        F_i = sum_j [ F_p_ij + F_v_ij ]  +  F_st_i

    where:

    - **Pressure flux**: ``F_p_ij = -0.5*(p_i + p_j)*A_ij``
      with ``p_i = v.p_phase[v.phase]`` (own-phase pressure).
    - **Viscous flux**: ``F_v_ij = (mu / |d_ij|) * du * (d_hat . A_ij)``
      with ``mu = mps.phases[v.phase].mu`` — the viscosity of the dual
      cell's own phase.  No harmonic mean.
    - **Surface tension**: only on ``is_interface`` vertices.

    Parameters
    ----------
    v : vertex object
    dim : int
    mps : MultiphaseSystem
    HC : Complex
    pressure_model : callable or None
        Typically ``MultiphaseEOS`` — updates ``v.p_phase`` in-place.
    """
    # Resolve own-phase pressure (populates v.p_phase via MultiphaseEOS)
    p_i = _resolve_pressure(v, pressure_model, HC, dim)
    u_i = v.u[:dim]
    x_i = v.x_a[:dim]

    # Own-phase viscosity (no harmonic mean)
    mu = mps.get_mu(v.phase)

    # Edge area cache
    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    F = np.zeros(dim)
    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim)

        # --- Pressure flux (face-average, conservative) ---
        p_j = _resolve_pressure(v_j, pressure_model, HC, dim)
        F -= 0.5 * (p_i + p_j) * A_ij

        # --- Viscous flux (own-phase viscosity) ---
        delta_u = v_j.u[:dim] - u_i
        d_ij = v_j.x_a[:dim] - x_i
        d_norm = np.linalg.norm(d_ij)
        if d_norm < 1e-30:
            continue
        d_hat = d_ij / d_norm
        F += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)

    # --- Surface tension on sharp interface vertices only ---
    if getattr(v, 'is_interface', False):
        F += _interface_surface_tension(v, dim, mps)

    return F


def _interface_surface_tension(v, dim: int, mps) -> np.ndarray:
    """Compute surface tension force on a sharp interface vertex.

    In both 2D and 3D the surface tension force is already integrated
    over the portion of the interface inside the vertex dual cell:

        F_st_i = integral_{Gamma_i} gamma * kappa * N dS

    - **3D**: cotangent-weight Heron curvature restricted to the
      interface sub-mesh (``hndA_i_interface``).  ``F_st = -gamma * HNdA_i``.
    - **2D**: integrated dual curvature from the Fundamental Theorem of
      Calculus applied to the tangent vector of the interface curve
      (``surface_tension_force_2d``).  ``F_st = gamma * (t_next - t_prev)``,
      which reconstructs a constant-curvature arc to machine precision.
    """
    interface_nbs = {nb for nb in v.nn if getattr(nb, 'is_interface', False)}
    if len(interface_nbs) < 2:
        return np.zeros(dim)

    phases = getattr(v, 'interface_phases', frozenset())
    if len(phases) < 2:
        return np.zeros(dim)

    phase_list = sorted(phases)
    gamma = mps.get_gamma_pair(phase_list[0], phase_list[1])
    if gamma == 0.0:
        return np.zeros(dim)

    if dim == 3:
        from ddgclib._curvatures_heron import hndA_i_interface
        HNdA, _C_i = hndA_i_interface(v, interface_nbs | {v})
        return -gamma * HNdA[:dim]
    else:
        F = np.zeros(dim)
        F[:2] = surface_tension_force_2d(v, gamma, interface_nbs)
        return F


def multiphase_stress_acceleration(
    v,
    dim: int = 3,
    mps=None,
    HC=None,
    pressure_model=None,
) -> np.ndarray:
    """Acceleration from multiphase stress: a_i = F_i / m_i."""
    F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC,
                                pressure_model=pressure_model)
    if v.m < 1e-30:
        return np.zeros(dim)
    return F / v.m


# Canonical alias for integrator usage
multiphase_dudt_i = multiphase_stress_acceleration
