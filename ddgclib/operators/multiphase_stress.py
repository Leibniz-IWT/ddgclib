"""Multiphase stress operators for the integrated Cauchy momentum equation.

Per-phase summed formulation
----------------------------

For every vertex ``v`` we compute a force that sums per-phase
contributions over the phases present in the vertex's dual cell::

    F_i = sum_{k in phases_present(v)} F_i^{(k)} + F_st_i

Each phase-k sub-force ``F_i^{(k)}`` uses:

- ``p_phase[k]`` on both sides of each sub-face (never mixing phases
  across a face).  If the neighbour ``v_j`` is bulk in a different
  phase (so it does not store ``p_phase[k]``), we use ``v_i.p_phase[k]``
  on the ``v_j`` end as well — this is consistent with the sub-face
  lying entirely in phase ``k`` (bulk phase of ``v_j``) with no
  phase-k neighbour to provide a reference.  See notes below.
- ``dual area vector`` ``A_ij^{(k)}`` from the phase-k sub-polygon of
  the dual cell (see :mod:`ddgclib.geometry._dual_split_2d`).
- viscosity ``μ_k`` on the phase-k sub-face, or the harmonic mean
  ``2 μ_i μ_j / (μ_i + μ_j)`` at μ-jumps on edges that straddle the
  interface (both endpoints interface, neighbour in the other phase).

Bulk vertices have ``phases_present = {v.phase}``; the sum collapses
to the current single-phase formula, so bulk physics is unchanged
from the previous own-phase-only scheme.

Surface tension remains a separate ``F_st`` force on sharp-interface
vertices — do NOT add ``γκ`` to the pressure field.  It is already
integrated over the dual edge via the Fundamental Theorem of Calculus
for the tangent vector (2D) / cotangent-weight Heron curvature on the
interface sub-mesh (3D).

Pressure-flux sub-case for bulk neighbours
------------------------------------------
For an edge (i, j) where ``v_i`` is interface and ``v_j`` is bulk
in phase ``k``, the dual face lies entirely in phase ``k`` (bulk
side).  ``v_i`` stores ``v_i.p_phase[k]`` (the outer/own phase
pressure) and ``v_j`` stores ``v_j.p = v_j.p_phase[k]`` (its own
pressure).  Both are phase-k pressures at the two ends of the face
— exactly what the face-average flux wants.  The previous
own-phase-only code read ``v_j.p_phase[own_phase]`` with
``own_phase = v_i.phase`` — wrong when ``v_j.phase != v_i.phase``
because ``v_j`` does not store a phase-``v_i.phase`` pressure.

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

from ddgclib.operators.stress import (
    dual_area_vector,
    pressure_flux,
    viscous_flux,
    _resolve_pressure,
)
from ddgclib.operators.curvature_2d import surface_tension_force_2d
from ddgclib.geometry._dual_split_2d import edge_phase_area_fractions


def _phases_present(v, n_phases: int) -> list[int]:
    """Return the sorted list of phases with non-zero presence at v.

    Uses ``v.interface_phases`` when available (populated by
    ``MultiphaseSystem.identify_interface``); otherwise falls back to
    ``{v.phase}``.  Guaranteed to return at least one entry.
    """
    phases = getattr(v, 'interface_phases', None)
    if phases is None or len(phases) == 0:
        return [int(v.phase)]
    return sorted(int(k) for k in phases if 0 <= int(k) < n_phases)


def _phase_pressure(v, k: int, fallback: float = 0.0) -> float:
    """Return v.p_phase[k] if populated, else ``fallback``."""
    p_phase = getattr(v, 'p_phase', None)
    if p_phase is None or k >= len(p_phase):
        return fallback
    val = float(p_phase[k])
    if val == 0.0:
        return fallback
    return val


def _face_viscosity_for_phase(
    mps, _v_i, _v_j, k: int,
) -> float:
    """Return the viscosity on the phase-k sub-face of edge (i,j).

    Each per-phase sub-face of the dual face lies entirely within one
    phase (determined by the interface geometry splitting the dual
    cell).  The viscosity is therefore unambiguously ``μ_k`` — no
    blending or harmonic-mean approximation is needed.
    """
    return float(mps.get_mu(k))


def multiphase_stress_force(
    v,
    dim: int = 3,
    mps=None,
    HC=None,
    pressure_model=None,
    curvature_path: str = 'integrated',
) -> np.ndarray:
    """Integrated force on a dual cell with per-phase summed stress.

    ::

        F_i = sum_k F_i^{(k)} + F_st_i

    See module docstring for the full sub-force formulation.

    Parameters
    ----------
    v : vertex object
    dim : int
    mps : MultiphaseSystem
    HC : Complex
    pressure_model : callable or None
        Typically :class:`MultiphaseEOS` — updates ``v.p_phase`` in-place.
        Not called inside the per-phase loop; ``mps.refresh()`` is
        expected to have already populated ``v.p_phase``.
    """
    n_phases = mps.n_phases
    has_p_phase = hasattr(v, 'p_phase')
    is_interface_i = bool(getattr(v, 'is_interface', False))

    phases_present = _phases_present(v, n_phases)

    # Pressure references (used when a neighbour lacks phase-k pressure)
    if has_p_phase:
        p_i_by_phase = {k: float(v.p_phase[k]) for k in phases_present}
    else:
        p_fallback = _resolve_pressure(v, pressure_model, HC, dim)
        p_i_by_phase = {k: p_fallback for k in phases_present}

    u_i = v.u[:dim]
    x_i = v.x_a[:dim]

    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    F = np.zeros(dim)

    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim)

        delta_u = v_j.u[:dim] - u_i
        d_ij = v_j.x_a[:dim] - x_i

        # Per-edge phase fractions (sums to 1.0).  ``dim`` selects the
        # curve-adjacency rule: 2D uses two-curve-neighbour polyline,
        # 3D treats every interface neighbour as surface-adjacent.
        fractions = edge_phase_area_fractions(
            v, v_j, dim=dim, interface=HC,
        )

        for k, frac in fractions.items():
            if k not in phases_present:
                # Sub-face lies in a phase not present at v.  Skip the
                # contribution rather than add spurious flux — this
                # happens only for bulk-bulk cross-phase edges (a mesh
                # artefact) where v has no mass/pressure in phase k.
                continue

            A_k = frac * A_ij
            p_i_k = p_i_by_phase[k]
            p_j_k = _phase_pressure(v_j, k, fallback=p_i_k)
            mu_k = _face_viscosity_for_phase(mps, v, v_j, k)

            F += pressure_flux(p_i_k, p_j_k, A_k)
            F += viscous_flux(mu_k, delta_u, d_ij, A_k)

    # --- Surface tension on sharp interface vertices only ---
    if is_interface_i:
        F += _interface_surface_tension(
            v, dim, mps, HC=HC, curvature_path=curvature_path,
        )

    return F


def _interface_surface_tension(
    v, dim: int, mps, HC=None, curvature_path: str = 'integrated',
) -> np.ndarray:
    """Compute surface tension force on a sharp interface vertex.

    In both 2D and 3D the surface tension force is already integrated
    over the portion of the interface inside the vertex dual cell::

        F_st_i = integral_{Gamma_i} gamma * kappa * N dS

    Parameters
    ----------
    curvature_path : {'integrated', 'csf_dual'}
        Discrete curvature stencil:

        - ``'integrated'`` (default): the FTC / cotangent-Heron form.
          3D — cotangent-weight Heron curvature restricted to the
          interface sub-mesh (``hndA_i_interface``).
          2D — Fundamental Theorem of Calculus on the tangent vector of
          the piecewise-linear interface (``surface_tension_force_2d``,
          ``F_st = gamma * (t_next - t_prev)``).  Exact for piecewise
          linear curves; on a polygon approximating a smooth interface
          the static-droplet residual converges first-order in mesh
          spacing — see plan note 2026-05-06.
        - ``'csf_dual'``: an experimental Continuum-Surface-Force form
          that aligns the curvature stencil with the dual face-area
          vector S_inner used by the per-phase pressure flux::

              F_st_csf = gamma * |Δt| / |S_inner| * S_inner

          Same magnitude as the FTC form, but the force direction is
          the dual-chord outward-normal direction rather than the
          tangent-difference direction.  Used as an A/B probe to
          quantify how much of the residual is direction-mismatch vs.
          magnitude-mismatch on irregular meshes.  Not a replacement for
          ``'integrated'`` — does not converge to ``γ κ N`` at the same
          order on smooth interfaces.
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

    if curvature_path == 'csf_dual':
        return _csf_dual_surface_tension(v, dim, gamma, HC, interface_nbs)

    if curvature_path == 'stokes':
        # Stokes-theorem boundary integral on the barycentric dual cell
        # of v_i restricted to interface triangles.  3D only; 2D delegates
        # to the existing FTC form which is already an exact Stokes
        # discretisation on piecewise-linear curves.
        if dim == 3:
            from ddgclib._curvatures_heron import integrated_hndA_i_interface
            F3 = integrated_hndA_i_interface(
                v, interface_nbs | {v}, HC=HC, gamma=gamma,
            )
            return F3[:dim]
        F = np.zeros(dim)
        F[:2] = surface_tension_force_2d(v, gamma, interface_nbs)
        return F

    if curvature_path != 'integrated':
        raise ValueError(
            f"Unknown curvature_path={curvature_path!r}; "
            f"expected 'integrated', 'stokes', or 'csf_dual'."
        )

    if dim == 3:
        from ddgclib._curvatures_heron import hndA_i_interface
        HNdA, _C_i = hndA_i_interface(v, interface_nbs | {v}, HC=HC)
        return -gamma * HNdA[:dim]
    else:
        F = np.zeros(dim)
        F[:2] = surface_tension_force_2d(v, gamma, interface_nbs)
        return F


def _csf_dual_surface_tension(
    v, dim: int, gamma: float, HC, interface_nbs,
) -> np.ndarray:
    """Continuum-Surface-Force surface tension aligned with the dual
    face-area vector ``S_inner``.

    Constructs the same |F_st| as the ``'integrated'`` path but in the
    dual-chord outward-normal direction.  Used as an A/B diagnostic to
    quantify how much of the static-droplet residual is direction
    mismatch (S_inner not anti-parallel to t_next - t_prev on irregular
    polygons) vs magnitude mismatch (chord length ≠ R·Δθ).

    For ``γ = 0`` this returns zero exactly.  For a flat interface with
    ``γ > 0`` and ``κ = 0``, ``Δt = 0`` so this returns zero — passes
    the 2A regression guard.
    """
    if gamma == 0.0:
        return np.zeros(dim)

    if dim == 3:
        # Magnitude: |HNdA| from the cotangent-Heron stencil; direction:
        # S_inner unit vector from the dual face-area accumulation.
        from ddgclib._curvatures_heron import hndA_i_interface
        HNdA, _C_i = hndA_i_interface(v, interface_nbs | {v}, HC=HC)
        delta_t_mag = float(np.linalg.norm(HNdA[:dim]))
    else:
        from ddgclib.operators.curvature_2d import (
            integrated_curvature_normal_2d,
        )
        delta_t = integrated_curvature_normal_2d(v, interface_nbs)
        delta_t_mag = float(np.linalg.norm(delta_t))

    if delta_t_mag < 1e-30:
        return np.zeros(dim)

    # S_inner = sum over neighbours j of frac_inner(i,j) * A_ij — the
    # sum of dual-face area vectors weighted by each face's interior-side
    # phase fraction.  For a 2-phase interface vertex with phase set
    # {0, 1}, "interior side" is the higher-phase index by convention.
    from ddgclib.operators.stress import dual_area_vector
    from ddgclib.geometry._dual_split_2d import edge_phase_area_fractions
    phases = sorted(int(k) for k in getattr(v, 'interface_phases', set()))
    if len(phases) < 2:
        return np.zeros(dim)
    inner_phase = phases[-1]

    S_inner = np.zeros(dim)
    for v_j in v.nn:
        A_ij = dual_area_vector(v, v_j, HC, dim)
        fractions = edge_phase_area_fractions(v, v_j, dim=dim, interface=HC)
        if inner_phase in fractions:
            S_inner += fractions[inner_phase] * A_ij[:dim]

    s_mag = float(np.linalg.norm(S_inner))
    if s_mag < 1e-30:
        return np.zeros(dim)

    # F_st_csf = gamma * |Δt| / |S_inner| * S_inner.  This is the
    # FTC magnitude redirected along S_inner — same direction as the
    # outward normal of the dual chord (= the area vector the per-phase
    # pressure flux integrates against).
    return gamma * delta_t_mag / s_mag * S_inner


def multiphase_stress_acceleration(
    v,
    dim: int = 3,
    mps=None,
    HC=None,
    pressure_model=None,
    curvature_path: str = 'integrated',
) -> np.ndarray:
    """Acceleration from multiphase stress: a_i = F_i / m_i."""
    F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC,
                                pressure_model=pressure_model,
                                curvature_path=curvature_path)
    if v.m < 1e-30:
        return np.zeros(dim)
    return F / v.m


# Canonical alias for integrator usage
multiphase_dudt_i = multiphase_stress_acceleration
