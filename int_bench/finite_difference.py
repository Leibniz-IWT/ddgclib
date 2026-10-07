"""
Pointwise finite-difference gradient and Hessian.

The discrete operators implemented in :mod:`int_bench.operators` are
*integrated* over a control volume. To compare against a pointwise
finite-difference reference, we compute the FD gradient/Hessian at the
vertex location and multiply by the control volume measure.

This is meaningful as a validation only for fields where the gradient
is approximately constant over the control volume (linear and
mildly-non-linear fields), or in the limit of small cells. For a
quadratic field on a uniform mesh, the integrated and pointwise
quantities agree because the gradient is linear in ``x`` and its
volume average over a centred cell equals its value at the cell
centre. We exploit this for the validation suite.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .mesh import SimplexMesh
from .operators import ControlVolume, control_volume


def fd_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Central-difference gradient at ``x`` with step ``h``.

    Returns
    -------
    ndarray, shape (dim,)
    """
    dim = len(x)
    g = np.empty(dim)
    for a in range(dim):
        xp = x.copy(); xp[a] += h
        xm = x.copy(); xm[a] -= h
        g[a] = (f(xp) - f(xm)) / (2 * h)
    return g


def fd_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """Central-difference Hessian at ``x`` with step ``h``.

    Diagonal:    H_aa = (f(x + h e_a) - 2 f(x) + f(x - h e_a)) / h^2
    Off-diag:    H_ab = (f(x+he_a+he_b) - f(x+he_a-he_b)
                       - f(x-he_a+he_b) + f(x-he_a-he_b)) / (4 h^2)

    Returns
    -------
    ndarray, shape (dim, dim)
    """
    dim = len(x)
    H = np.empty((dim, dim))
    f0 = f(x)
    for a in range(dim):
        xp = x.copy(); xp[a] += h
        xm = x.copy(); xm[a] -= h
        H[a, a] = (f(xp) - 2 * f0 + f(xm)) / (h * h)
    for a in range(dim):
        for b in range(a + 1, dim):
            xpp = x.copy(); xpp[a] += h; xpp[b] += h
            xpm = x.copy(); xpm[a] += h; xpm[b] -= h
            xmp = x.copy(); xmp[a] -= h; xmp[b] += h
            xmm = x.copy(); xmm[a] -= h; xmm[b] -= h
            H[a, b] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
            H[b, a] = H[a, b]
    return H


# ---------------------------------------------------------------------------
# Integrated comparisons (FD pointwise * control-volume measure)
# ---------------------------------------------------------------------------

def integrated_gradient_fd(
    mesh: SimplexMesh,
    f: Callable[[np.ndarray], float],
    v_idx: int,
    control: ControlVolume = "dual",
    h: float = 1e-5,
) -> np.ndarray:
    """``vol(V) * ∇f(x_i)`` — pointwise FD scaled by the cell measure.

    For an affine ``f`` this matches the integrated quantity exactly;
    for higher-order fields it is the leading-order approximation.
    """
    g = fd_gradient(f, mesh.vertices[v_idx], h=h)
    return control_volume(mesh, v_idx, control=control) * g


def integrated_hessian_fd(
    mesh: SimplexMesh,
    f: Callable[[np.ndarray], float],
    v_idx: int,
    control: ControlVolume = "dual",
    h: float = 1e-4,
) -> np.ndarray:
    """``vol(V) * H(x_i)`` — pointwise FD Hessian scaled by cell measure.

    For a quadratic ``f`` the true Hessian is constant, and on a
    centred mesh ``vol(V) * H(x_i) = ∫_V H dV`` exactly. This is the
    intended validation target for the quadratic suite.
    """
    H = fd_hessian(f, mesh.vertices[v_idx], h=h)
    return control_volume(mesh, v_idx, control=control) * H


__all__ = [
    "fd_gradient",
    "fd_hessian",
    "integrated_gradient_fd",
    "integrated_hessian_fd",
]
