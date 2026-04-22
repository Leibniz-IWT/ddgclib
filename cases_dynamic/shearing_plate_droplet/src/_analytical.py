"""Analytical references for a droplet in simple shear flow.

Background flow (no droplet)
----------------------------
Planar Couette flow between plates moving at ``+U`` and ``-U`` separated
by ``2 L_y``::

    u_x(y) = (U / L_y) * y
    u_y    = 0

The volumetric flow rate across any vertical plane is zero; streamlines
are horizontal lines.

Taylor (1934) small-deformation theory
--------------------------------------
For a Newtonian drop in Stokes shear (``Re << 1``) with capillary number
``Ca = mu_o * gamma_dot * R0 / gamma`` and viscosity ratio
``lambda = mu_d / mu_o``, the steady-state droplet shape is ellipsoidal
with deformation parameter::

    D = (L - B) / (L + B) = Ca * (19*lambda + 16) / (16*lambda + 16)

where ``L`` and ``B`` are the major and minor semi-axes of the deformed
droplet.  In the limit ``lambda -> 0``, ``D = Ca``.

The major axis is tilted from the flow direction by an angle ``theta``
that approaches 45° at low ``Ca`` and decreases with ``Ca``.
"""
from __future__ import annotations

import numpy as np


def couette_profile(y: float | np.ndarray, U_wall: float,
                    L_y: float) -> float | np.ndarray:
    """Background Couette velocity ``u_x(y)`` between plates at ``y = ±L_y``.

    Parameters
    ----------
    y : float or array
        Wall-normal position.
    U_wall : float
        Top-plate speed (bottom plate is ``-U_wall``).
    L_y : float
        Half-height (plate at ``y = +L_y``).
    """
    return U_wall * np.asarray(y, dtype=float) / L_y


def taylor_deformation(Ca: float, visc_ratio: float) -> float:
    """Taylor's small-deformation parameter ``D = (L - B) / (L + B)``.

    Valid for ``Ca << 1`` and Stokes flow (``Re << 1``).

    Parameters
    ----------
    Ca : float
        Capillary number ``mu_o * gamma_dot * R0 / gamma``.
    visc_ratio : float
        Viscosity ratio ``lambda = mu_d / mu_o``.

    References
    ----------
    G.I. Taylor, Proc. R. Soc. A, 146, 501-523 (1934).
    """
    lam = float(visc_ratio)
    return Ca * (19.0 * lam + 16.0) / (16.0 * lam + 16.0)


def compute_deformation_from_interface(
    interface_positions: np.ndarray,
    center: np.ndarray | None = None,
    dim: int = 2,
) -> dict:
    """Estimate deformation parameter from sampled interface vertex positions.

    Fits a bounding ellipse (principal axes of the position covariance)
    and returns ``D = (a - b)/(a + b)`` where ``a >= b`` are the
    semi-axis proxies, plus the tilt angle relative to the x-axis.

    Parameters
    ----------
    interface_positions : ndarray, shape (n, dim)
        Interface vertex coordinates (2D or 3D).
    center : array-like or None
        Droplet centre; defaults to the centroid of the points.
    dim : int
        Spatial dimension.

    Returns
    -------
    dict
        ``{'D': deformation, 'a': major, 'b': minor, 'tilt': radians}``
        For 3D, ``a``, ``b``, ``c`` are returned for the three principal
        semi-axes and ``D = (a - c) / (a + c)``.
    """
    X = np.asarray(interface_positions, dtype=float)
    if center is None:
        c = X.mean(axis=0)
    else:
        c = np.asarray(center, dtype=float)
    Y = X - c
    if Y.shape[0] < dim + 1:
        return {'D': float('nan'), 'a': float('nan'), 'b': float('nan'),
                'tilt': float('nan')}

    # Principal-axis analysis: eigendecomposition of second-moment matrix.
    # Semi-axis estimates from sqrt of eigenvalues times a dimension factor
    # so the result matches the radius of the best-fit ellipse for a
    # uniformly sampled boundary.
    M = (Y.T @ Y) / Y.shape[0]
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    # Uniform boundary samples on ellipse of semi-axes (a, b) give
    # covariance ~diag(a^2/2, b^2/2) in 2D; ~diag(a^2/5, b^2/5, c^2/5) in 3D.
    scale = np.sqrt(2.0) if dim == 2 else np.sqrt(5.0)
    axes = scale * np.sqrt(np.maximum(vals, 0.0))
    if dim == 2:
        a, b = float(axes[0]), float(axes[1])
        tilt = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
        D = (a - b) / (a + b) if (a + b) > 0 else 0.0
        return {'D': D, 'a': a, 'b': b, 'tilt': tilt}
    else:
        a, b, c_ = float(axes[0]), float(axes[1]), float(axes[2])
        # Tilt of the major axis in the x-y plane
        tilt = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
        D = (a - c_) / (a + c_) if (a + c_) > 0 else 0.0
        return {'D': D, 'a': a, 'b': b, 'c': c_, 'tilt': tilt}
