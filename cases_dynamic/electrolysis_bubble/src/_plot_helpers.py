"""Diagnostics for the electrolysis bubble case.

Tracks bubble centroid, equivalent radius, kinetic energy, and total
(gas, liquid) mass.  Detachment is flagged when the bubble centroid
clears a height threshold above the electrode.
"""
from __future__ import annotations

import numpy as np


def compute_diagnostics(
    HC,
    dim: int,
    gas_phase: int = 1,
    electrode_level: float | None = None,
) -> dict:
    """Return bubble diagnostics at the current time step."""
    total_mass = 0.0
    gas_mass = 0.0
    gas_volume = 0.0
    KE = 0.0
    com = np.zeros(dim)
    gas_com = np.zeros(dim)

    iface_positions = []
    iface_n = 0

    for v in HC.V:
        total_mass += float(v.m)
        com += float(v.m) * v.x_a[:dim]
        u = v.u[:dim]
        KE += 0.5 * float(v.m) * float(u @ u)

        vol_g = float(v.dual_vol_phase[gas_phase])
        m_g = float(v.m_phase[gas_phase])
        if vol_g > 1e-30 and m_g > 1e-30:
            gas_mass += m_g
            gas_volume += vol_g
            gas_com += m_g * v.x_a[:dim]

        if bool(getattr(v, 'is_interface', False)):
            iface_positions.append(v.x_a[:dim].copy())
            iface_n += 1

    if total_mass > 0:
        com /= total_mass
    if gas_mass > 0:
        gas_com /= gas_mass

    # Equivalent bubble radius: area^(1/2)/sqrt(pi) in 2D, volume^(1/3)
    # with 4/3 pi scaling in 3D.  Uses the gas-phase dual volume sum.
    if dim == 2:
        R_eq = float(np.sqrt(max(gas_volume, 0.0) / np.pi))
    else:
        R_eq = float((3.0 * max(gas_volume, 0.0) / (4.0 * np.pi)) ** (1.0 / 3.0))

    # Lowest interface vertex height (distance above electrode).
    min_iface_z = np.inf
    max_iface_z = -np.inf
    if iface_positions:
        zs = np.array([p[dim - 1] for p in iface_positions])
        min_iface_z = float(zs.min())
        max_iface_z = float(zs.max())

    detached = False
    detach_gap = 0.0
    if electrode_level is not None and iface_positions:
        # Detached means the lowest interface vertex has clearly lifted
        # off the electrode by at least 0.3 * R_eq.  (The outer caller
        # layers on an additional rise-of-centroid check.)
        detach_gap = float(min_iface_z - electrode_level)
        detached = bool(detach_gap > 0.3 * R_eq)

    return {
        'total_mass': total_mass,
        'gas_mass': gas_mass,
        'gas_volume': gas_volume,
        'KE': KE,
        'com': com,
        'gas_com': gas_com,
        'R_eq': R_eq,
        'n_interface': iface_n,
        'min_iface_z': min_iface_z,
        'max_iface_z': max_iface_z,
        'detached': detached,
        'detach_gap': detach_gap,
    }


def plot_radius_history(t_arr, R_arr, R0, R_detach=None, ax=None, title=""):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()
    ax.plot(t_arr, R_arr, 'bo-', markersize=3, label='$R_\\mathrm{eq}$')
    ax.axhline(R0, color='k', ls=':', label=f'$R_0={R0:.2e}$')
    if R_detach is not None:
        ax.axhline(R_detach, color='r', ls='--',
                   label=f'Fritz $R_\\mathrm{{det}}={R_detach:.2e}$')
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Equivalent bubble radius [m]")
    ax.set_title(title or "Bubble growth")
    ax.legend()
    return fig, ax


def plot_centroid_history(t_arr, z_arr, electrode_level, R_eq_arr=None,
                           ax=None, title=""):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()
    ax.plot(t_arr, z_arr, 'b-', label='Gas-phase centroid')
    ax.axhline(electrode_level, color='k', ls=':', label='Electrode')
    if R_eq_arr is not None:
        ax.plot(t_arr, np.asarray(electrode_level) + np.asarray(R_eq_arr),
                'g--', label='Electrode + $R_\\mathrm{eq}$')
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Bubble centroid height [m]")
    ax.set_title(title or "Centroid rise")
    ax.legend()
    return fig, ax
