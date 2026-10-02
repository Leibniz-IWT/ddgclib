"""Stabilisation operators for the weakly compressible Lagrangian pipeline.

``density_diffusion_step``
    Gradient-corrected density diffusion on the dual mesh (delta-SPH type,
    Antuono, Marrone & Colagrossi 2010).  The vertex-centred scheme with
    the centred pressure flux ``-1/2 (p_i + p_j) A_ij`` cannot see a
    checkerboard density pattern (alternating compressed / expanded
    cells): the face-averaged pressure of such a pattern is uniform, so
    the mode is force-free and, once excited by shear, remeshing or mass
    resets, it persists.  The operator below is an integrated,
    pairwise-antisymmetric mass flux across the dual faces,

        dm_i/dt = sum_j delta c0 |A_ij| [ (rho_j - rho_i)
                                          - 1/2 (grad rho_i + grad rho_j) . d_ij ],

    with ``grad rho`` the linear-precision integrated gradient of
    :func:`ddgclib.operators.stress.scalar_gradient_integrated` divided by
    the dual volume.  Properties:

    * exact mass conservation (antisymmetric flux);
    * vanishes identically for linear density fields when both cells are
      closed (interior), so hydrostatic equilibrium is preserved to
      round-off; for open (boundary) cells the correction is switched off
      and the plain diffusion is used;
    * diffusivity ``delta c0 |d_ij|``; explicit stability needs roughly
      ``delta < 0.3`` at the acoustic CFL number 0.4.

    It changes masses, not momenta, so it adds no shear viscosity.  Used
    through the ``density_diffusion`` integrator kwarg (method axis
    ``density_diffusion``) or directly by a case loop.

Diagnosed and validated on ``cases_dynamic/capillary_rise`` (2026-09-26,
see ``cases_dynamic/capillary_rise_energy_grad/README.md`` Section 5).
"""
from __future__ import annotations

import numpy as np

from ddgclib.operators.stress import dual_area_vector, scalar_gradient_integrated

__all__ = ["density_diffusion_step"]


def density_diffusion_step(HC, verts, delta: float, c0: float, dt: float,
                           dim: int = 2, corrected: bool = True) -> dict:
    """Apply one explicit density-diffusion step to the vertices ``verts``.

    Only vertices in ``verts`` exchange mass (frozen walls and prescribed
    cells are excluded by the caller); every vertex of ``HC`` must have
    ``m`` and a cached ``dual_vol`` (``cache_dual_volumes``).

    Returns ``{'max_rel_dm', 'sum_dm', 'n_pairs', 'n_uncorrected'}``.
    """
    vs = [v for v in verts if getattr(v, "dual_vol", 0.0) > 0.0 and getattr(v, "m", 0.0) > 0.0]
    vset = set(vs)
    # densities on ALL vertices: the integrated gradient of a cell reads
    # the neighbours' field, including frozen / prescribed cells
    for v in HC.V:
        vol = getattr(v, "dual_vol", 0.0)
        v.rho = (getattr(v, "m", 0.0) / vol) if vol > 0.0 else 0.0
    grad: dict = {}
    closed: dict = {}
    if corrected:
        for v in vs:
            sumA = np.zeros(dim)
            absA = 0.0
            for w in v.nn:
                A = dual_area_vector(v, w, HC, dim)
                sumA += A
                absA += float(np.linalg.norm(A))
            closed[id(v)] = absA > 0.0 and float(np.linalg.norm(sumA)) < 1e-8 * absA
            grad[id(v)] = (scalar_gradient_integrated(v, HC, dim, field_attr="rho") / v.dual_vol
                           if closed[id(v)] else np.zeros(dim))
    dm: dict = {id(v): 0.0 for v in vs}
    n_pairs = n_unc = 0
    # Each pair once, from its first endpoint in the order of ``vs`` (by
    # ``id()`` the orientation of A and the order of the sums into ``dm``
    # followed the memory addresses and differed between processes).
    done: set = set()
    for v in vs:
        done.add(id(v))
        for w in v.nn:
            if w not in vset or id(w) in done:
                continue
            A = dual_area_vector(v, w, HC, dim)
            An = float(np.linalg.norm(A))
            if An == 0.0:
                continue
            drho = w.rho - v.rho
            if corrected and closed.get(id(v), False) and closed.get(id(w), False):
                d = w.x_a[:dim] - v.x_a[:dim]
                drho -= 0.5 * float((grad[id(v)] + grad[id(w)]) @ d)
            elif corrected:
                n_unc += 1
            J = delta * c0 * An * drho          # mass flux w -> v per unit depth
            dm[id(v)] += dt * J
            dm[id(w)] -= dt * J
            n_pairs += 1
    max_rel = 0.0
    tot = 0.0
    for v in vs:
        d = dm[id(v)]
        max_rel = max(max_rel, abs(d) / v.m)
        v.m += d
        tot += d
    return {"max_rel_dm": max_rel, "sum_dm": tot, "n_pairs": n_pairs, "n_uncorrected": n_unc}
