"""Plotting helpers for capillary rise simulations."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import numpy as np


def compute_diagnostics(
    HC, dim: int, gravity_axis: int, free_surface_verts: set,
) -> dict:
    """Compute current simulation diagnostics.

    Returns dict with:
        'h_mean': mean height of free surface vertices [m]
        'h_max': max height of free surface vertices [m]
        'h_min': min height of free surface vertices [m]
        'KE': total kinetic energy [J]
        'total_mass': total mass [kg]
        'u_max': maximum velocity magnitude [m/s]
    """
    heights = []
    for v in free_surface_verts:
        try:
            heights.append(v.x_a[gravity_axis])
        except Exception:
            pass

    heights = np.array(heights) if heights else np.array([0.0])

    KE = 0.0
    total_mass = 0.0
    u_max = 0.0
    for v in HC.V:
        m = getattr(v, 'm', 0.0)
        total_mass += m
        u = getattr(v, 'u', np.zeros(dim))
        u_norm = np.linalg.norm(u[:dim])
        KE += 0.5 * m * u_norm**2
        if u_norm > u_max:
            u_max = u_norm

    return {
        'h_mean': float(np.mean(heights)),
        'h_max': float(np.max(heights)),
        'h_min': float(np.min(heights)),
        'KE': float(KE),
        'total_mass': float(total_mass),
        'u_max': float(u_max),
    }


def plot_height_vs_time(
    t_arr, h_arr, h_washburn=None, h_jurin=None,
    h_data_t=None, h_data_h=None,
    ax=None, title=None, save_path=None,
):
    """Plot meniscus height h(t) vs analytical Washburn solution.

    Parameters
    ----------
    t_arr : array-like
        Simulation time array [s].
    h_arr : array-like
        Simulation height array [m].
    h_washburn : tuple of (t_wash, h_wash), optional
        Washburn ODE solution arrays.
    h_jurin : float, optional
        Equilibrium Jurin height [m].
    h_data_t, h_data_h : array-like, optional
        Experimental / fitted reference data.
    ax : matplotlib Axes, optional
        Axes to plot on; created if *None*.
    title : str, optional
        Plot title.
    save_path : str, optional
        Save figure to this path.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    Notes
    -----
    Heights are plotted in cm for readability.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # Convert to cm
    ax.plot(t_arr, np.array(h_arr) * 100, 'b-', lw=2, label='DDG simulation')

    if h_washburn is not None:
        t_w, h_w = h_washburn
        ax.plot(t_w, h_w * 100, 'r--', lw=1.5, label='Washburn ODE')

    if h_jurin is not None:
        ax.axhline(
            h_jurin * 100, color='k', ls=':', lw=1, alpha=0.5,
            label=f'Jurin h={h_jurin * 100:.2f} cm',
        )

    if h_data_t is not None and h_data_h is not None:
        ax.plot(
            h_data_t, np.array(h_data_h) * 100,
            'go', ms=4, alpha=0.6, label='Lunowa fitted data',
        )

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Meniscus height [cm]')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)

    return fig, ax


def plot_fluid_snapshot(HC, bV, t, dim, save_path=None):
    """Pressure + velocity snapshot using ddgclib.visualization.plot_fluid."""
    from ddgclib.visualization import plot_fluid

    fig, ax = plot_fluid(
        HC, bV=bV, scalar_field='p', vector_field='u',
        dim=dim, title=f'Capillary rise t={t:.4f} s',
        save_path=save_path,
    )
    return fig, ax


def plot_multi_radius(
    results: dict,
    fluid_name: str = "water",
    save_path: str | None = None,
):
    """Plot h(t) for multiple tube radii on one figure.

    Parameters
    ----------
    results : dict
        Mapping ``{R_mm: {'t': array, 'h': array, 'h_wash': (t, h),
        'h_jurin': float, 'data': (t, h) or None}}``.
    fluid_name : str
        Fluid name used in the super-title.
    save_path : str, optional
        Save figure to this path.

    Returns
    -------
    fig
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    for i, (R_mm, res) in enumerate(sorted(results.items())):
        ax = axes[i]
        ax.plot(res['t'], np.array(res['h']) * 100, 'b-', lw=2, label='DDG')

        if 'h_wash' in res and res['h_wash'] is not None:
            t_w, h_w = res['h_wash']
            ax.plot(t_w, h_w * 100, 'r--', lw=1.5, label='Washburn')

        if 'h_jurin' in res:
            ax.axhline(
                res['h_jurin'] * 100, color='k', ls=':', lw=1, alpha=0.5,
            )

        if 'data' in res and res['data'] is not None:
            dt, dh = res['data']
            ax.plot(
                dt, np.array(dh) * 100, 'go', ms=3, alpha=0.5, label='Lunowa',
            )

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Height [cm]')
        ax.set_title(f'R = {R_mm} mm')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Capillary Rise \u2014 {fluid_name.capitalize()}', fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def save_animation_frame(HC, bV, t, dim, frame_dir, step, gravity_axis=None):
    """Save a single animation frame.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    bV : set
        Boundary vertices.
    t : float
        Current simulation time [s].
    dim : int
        Spatial dimension.
    frame_dir : str
        Directory to write frame images into (created if needed).
    step : int
        Time-step index used for the filename.
    gravity_axis : int, optional
        Unused; reserved for future orientation labelling.

    Returns
    -------
    str
        Path to the saved frame image.
    """
    import matplotlib.pyplot as plt

    os.makedirs(frame_dir, exist_ok=True)
    path = os.path.join(frame_dir, f'frame_{step:06d}.png')
    try:
        plot_fluid_snapshot(HC, bV, t, dim, save_path=path)
        plt.close('all')
    except Exception:
        pass
    return path
