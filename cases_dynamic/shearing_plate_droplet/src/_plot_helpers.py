"""Diagnostics and plotting utilities for the shearing-plate droplet case.

Most plotting delegates to ``ddgclib.visualization`` (``plot_fluid``,
``plot_primal``, ``dynamic_plot_fluid``).  This module adds:

- ``compute_diagnostics``: per-frame state summary (KE, mass, droplet
  deformation parameter D, tilt angle).
- ``plot_velocity_profile``: mean ``u_x(y)`` averaged over streamwise
  bands, compared to the Couette ``u_x = (U/L_y) * y`` reference.
- ``plot_deformation_history``: D(t) and tilt(t) time-series plots.
"""
from __future__ import annotations

import numpy as np

from ._analytical import (
    couette_profile, compute_deformation_from_interface,
)


def _interface_positions(HC, dim: int) -> np.ndarray:
    coords = [v.x_a[:dim].copy() for v in HC.V
              if getattr(v, 'is_interface', False)]
    if not coords:
        return np.zeros((0, dim))
    return np.asarray(coords, dtype=float)


def compute_diagnostics(HC, dim: int = 2) -> dict:
    """Per-frame diagnostics for a shearing-plate droplet simulation.

    Returns keys: ``KE``, ``total_mass``, ``com``, ``R_max``, ``R_min``,
    ``n_interface``, ``D`` (deformation parameter), ``tilt`` (radians),
    ``semi_major``, ``semi_minor`` (estimated ellipse axes),
    ``u_max`` (maximum fluid velocity magnitude).
    """
    total_mass = 0.0
    com = np.zeros(dim)
    KE = 0.0
    u_max = 0.0
    for v in HC.V:
        m = float(v.m)
        total_mass += m
        com += m * v.x_a[:dim]
        uv = v.u[:dim]
        KE += 0.5 * m * float(np.dot(uv, uv))
        umag = float(np.linalg.norm(uv))
        if umag > u_max:
            u_max = umag
    if total_mass > 0:
        com /= total_mass

    iface = _interface_positions(HC, dim)
    n_interface = int(iface.shape[0])
    if n_interface >= dim + 1:
        fit = compute_deformation_from_interface(iface, center=com, dim=dim)
        D = float(fit['D'])
        tilt = float(fit['tilt'])
        a = float(fit['a'])
        b = float(fit['b'])
        radii = np.linalg.norm(iface - com, axis=1)
        R_max = float(np.max(radii))
        R_min = float(np.min(radii))
    else:
        D = tilt = a = b = float('nan')
        R_max = R_min = 0.0

    return {
        'KE': KE,
        'total_mass': total_mass,
        'com': com,
        'R_max': R_max,
        'R_min': R_min,
        'n_interface': n_interface,
        'D': D,
        'tilt': tilt,
        'semi_major': a,
        'semi_minor': b,
        'u_max': u_max,
    }


def plot_velocity_profile(
    HC, U_wall: float, L_y: float, dim: int = 2, n_bins: int = 20,
    ax=None, save_path: str | None = None, title: str = "",
):
    """Plot mean ``u_x(y)`` averaged over x (and z, in 3D) vs Couette.

    Interior vertices are binned by ``y``; the mean ``u_x`` per bin
    is plotted alongside the analytical ``u_x = (U_wall / L_y) * y``
    reference line.
    """
    import matplotlib.pyplot as plt

    ys, ux = [], []
    for v in HC.V:
        ys.append(float(v.x_a[1]))
        ux.append(float(v.u[0]))
    ys = np.asarray(ys); ux = np.asarray(ux)
    if ys.size == 0:
        return None, None

    bins = np.linspace(-L_y, L_y, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    mean_ux = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (ys >= bins[i]) & (ys < bins[i + 1])
        if mask.any():
            mean_ux[i] = np.mean(ux[mask])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.get_figure()

    ax.plot(mean_ux, centres, 'bo-', label='Simulation (mean)')
    yf = np.linspace(-L_y, L_y, 100)
    ax.plot(couette_profile(yf, U_wall, L_y), yf, 'k--',
            label=r'Couette $u_x = U y / L_y$')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.set_xlabel('$u_x$ [m/s]')
    ax.set_ylabel('$y$ [m]')
    ax.set_title(title or "Streamwise velocity profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_deformation_history(
    t_arr, D_arr, tilt_arr, Ca: float, D_taylor: float | None = None,
    ax=None, save_path: str | None = None, title: str = "",
):
    """Plot the deformation parameter D(t) and tilt(t)."""
    import matplotlib.pyplot as plt

    t_arr = np.asarray(t_arr, dtype=float)
    D_arr = np.asarray(D_arr, dtype=float)
    tilt_arr = np.asarray(tilt_arr, dtype=float)

    if ax is None:
        fig, (ax_D, ax_th) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    else:
        fig = ax.get_figure()
        ax_D = ax
        ax_th = None

    ax_D.plot(t_arr, D_arr, 'b-', label='Simulation')
    if D_taylor is not None:
        ax_D.axhline(D_taylor, color='r', ls='--',
                      label=f'Taylor small-$D$: {D_taylor:.3f}')
    ax_D.set_ylabel(r'Deformation $D = (L-B)/(L+B)$')
    ax_D.set_title(title or f'Droplet deformation (Ca = {Ca:.3f})')
    ax_D.legend()
    ax_D.grid(True, alpha=0.3)

    if ax_th is not None:
        ax_th.plot(t_arr, np.degrees(tilt_arr), 'g-')
        ax_th.axhline(45.0, color='k', ls=':', alpha=0.5,
                       label='Low-$Ca$ limit: 45°')
        ax_th.set_xlabel('t [s]')
        ax_th.set_ylabel('Tilt angle [deg]')
        ax_th.legend()
        ax_th.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, (ax_D, ax_th)
