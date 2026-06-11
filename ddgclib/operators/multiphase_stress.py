"""Multiphase stress operators.

This is a lightweight local backport of the operator stack Stefan referenced.
For sharp-interface surface benchmarks, it gracefully falls back to the
curvature-based interface force when volumetric dual geometry is absent.
"""

from __future__ import annotations

import numpy as np

from ddgclib._curvatures_heron import hndA_i_interface
from ddgclib.operators.stress import dual_area_vector


def _resolve_pressure(v, pressure_model=None, HC=None, dim: int = 3) -> float:
    """Return a scalar pressure for vertex ``v``.

    ``pressure_model`` may update vertex state in-place.  The fallback is the
    scalar/array-like ``v.p`` field used by the single-phase stress operators.
    """
    if pressure_model is not None:
        return float(pressure_model(v, HC=HC, dim=dim))

    p_val = getattr(v, "p", 0.0)
    if np.ndim(p_val) == 0:
        return float(p_val)
    return float(np.asarray(p_val).ravel()[0])


def _interface_surface_tension(v, dim: int, mps) -> np.ndarray:
    interface_nbs = {nb for nb in v.nn if getattr(nb, "is_interface", False)}
    if len(interface_nbs) < 2:
        return np.zeros(dim)

    phases = getattr(v, "interface_phases", frozenset())
    if len(phases) < 2:
        return np.zeros(dim)

    phase_list = sorted(phases)
    gamma = mps.get_gamma_pair(phase_list[0], phase_list[1])
    if gamma == 0.0:
        return np.zeros(dim)

    HNdA, _ = hndA_i_interface(v, interface_nbs | {v})
    return -gamma * HNdA[:dim]


def multiphase_stress_force(
    v,
    dim: int = 3,
    mps=None,
    HC=None,
    pressure_model=None,
) -> np.ndarray:
    """Integrated force on a multiphase dual cell.

    When volumetric dual geometry is available, this mirrors the face-flux
    structure of ``stress_force``.  On pure surface/interface meshes it skips
    the pressure/viscous flux terms and still returns the sharp-interface
    curvature force, which is the relevant contribution for the catenoid
    equilibrium benchmark.
    """
    F = np.zeros(dim)

    can_use_flux_terms = HC is not None and hasattr(v, "vd")
    if can_use_flux_terms:
        p_i = _resolve_pressure(v, pressure_model, HC, dim)
        u_i = v.u[:dim]
        x_i = v.x_a[:dim]
        mu = mps.get_mu(v.phase) if mps is not None else 0.0

        for v_j in v.nn:
            try:
                A_ij = dual_area_vector(v, v_j, HC, dim)
            except (KeyError, IndexError, ValueError, RuntimeError, ZeroDivisionError):
                continue

            p_j = _resolve_pressure(v_j, pressure_model, HC, dim)
            F -= 0.5 * (p_i + p_j) * A_ij

            delta_u = v_j.u[:dim] - u_i
            d_ij = v_j.x_a[:dim] - x_i
            d_norm = np.linalg.norm(d_ij)
            if d_norm < 1e-30:
                continue
            d_hat = d_ij / d_norm
            F += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)

    if getattr(v, "is_interface", False) and mps is not None:
        F += _interface_surface_tension(v, dim, mps)

    return F


def multiphase_stress_acceleration(
    v,
    dim: int = 3,
    mps=None,
    HC=None,
    pressure_model=None,
) -> np.ndarray:
    F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC, pressure_model=pressure_model)
    if v.m < 1e-30:
        return np.zeros(dim)
    return F / v.m


multiphase_dudt_i = multiphase_stress_acceleration
