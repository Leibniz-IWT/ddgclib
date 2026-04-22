"""Plotting utilities for the oscillating droplet case study.

Uses the standard ``ddgclib.visualization`` wrappers (``plot_fluid``,
``plot_primal``) for pressure/velocity fields, and adds phase-specific
overlays.

For animations, provides ``record_frame`` (captures per-vertex data
for one frame) and ``make_multiphase_animation`` (builds a matplotlib
FuncAnimation from a list of frames, with interface markers).

For polyscope, use the standard ``interactive_history_viewer`` from
``ddgclib.visualization.polyscope_3d`` with a ``StateHistory`` that
records ``['u', 'p', 'phase']``.
"""
from __future__ import annotations

import numpy as np


def plot_droplet_fluid(HC, bV=None, t: float = 0.0, dim: int = 2,
                       save_path: str = None, **kwargs):
    """Standard pressure + velocity snapshot using ``plot_fluid``.

    Delegates to ``ddgclib.visualization.plot_fluid`` which creates
    a two-panel figure (pressure colormap + velocity quiver).
    """
    from ddgclib.visualization import plot_fluid
    return plot_fluid(HC, bV=bV, t=t, save_path=save_path, **kwargs)


def plot_droplet_phases(HC, bV=None, dim: int = 2, ax=None, title: str = "",
                        save_path: str = None, dpi: int = 150):
    """Plot mesh colored by phase using ``plot_primal``.

    Uses ``scalar_field='phase'`` with a discrete colormap so that
    phase 0 and phase 1 are visually distinct.  Interface vertices
    are overlaid as red markers.
    """
    from ddgclib.visualization import plot_primal

    fig, ax = plot_primal(
        HC, bV=bV, scalar_field='phase', ax=ax,
        save_path=None, cmap='coolwarm',
        title=title or "Phase field",
    )

    # Overlay interface vertices
    xs_iface, ys_iface = [], []
    for v in HC.V:
        if getattr(v, 'is_interface', False):
            xs_iface.append(v.x_a[0])
            ys_iface.append(v.x_a[1])
    if xs_iface:
        ax.scatter(xs_iface, ys_iface, c='red', s=12, zorder=5,
                   label='Interface')
        ax.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_interface_shape(HC, R0: float, dim: int = 2, ax=None,
                         title: str = ""):
    """Polar plot of interface radius vs angle."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'),
                                figsize=(6, 6))
    else:
        fig = ax.get_figure()

    thetas, radii = [], []
    for v in HC.V:
        if not getattr(v, 'is_interface', False):
            continue
        x = v.x_a[:dim]
        r = np.linalg.norm(x)
        if r < 1e-30:
            continue
        if dim == 2:
            theta = np.arctan2(x[1], x[0])
        else:
            theta = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])
        thetas.append(theta)
        radii.append(r)

    order = np.argsort(thetas)
    thetas = np.array(thetas)[order]
    radii = np.array(radii)[order]

    ax.plot(thetas, radii, 'ro-', markersize=3, label='Numerical')
    ax.plot(np.linspace(-np.pi, np.pi, 200),
            np.full(200, R0), 'k--', alpha=0.5, label=f'R₀={R0}')
    ax.set_title(title or "Interface shape")
    ax.legend(loc='lower right', fontsize=8)
    return fig, ax


def plot_radius_envelope(t_arr, R_max_sim, R_max_analytical=None, R0=None,
                          ax=None, title: str = ""):
    """Max interface radius vs time with analytical overlay."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(t_arr, R_max_sim, 'bo-', markersize=3, label='Simulation')
    if R_max_analytical is not None:
        ax.plot(t_arr, R_max_analytical, 'r--', label='Analytical')
    if R0 is not None:
        ax.axhline(R0, color='k', linestyle=':', alpha=0.5, label=f'R₀={R0}')

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("R_max [m]")
    ax.set_title(title or "Droplet radius envelope")
    ax.legend()
    return fig, ax


def plot_apex_trajectory(
    t_arr, r_apex_arr, theta_apex_arr, R0, epsilon, l, omega, beta,
    ax=None, title: str = "",
):
    """Plot r_apex(t) with analytical Rayleigh-Lamb overlay evaluated
    at the (time-varying) apex angle theta_apex(t).

    Using the actual theta_apex each step (rather than theta=0) accounts
    for tangential drift of the tracked interface vertex.
    """
    import matplotlib.pyplot as plt
    from ._analytical import radius_perturbation

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    t_arr = np.asarray(t_arr)
    r_apex_arr = np.asarray(r_apex_arr)
    theta_apex_arr = np.asarray(theta_apex_arr)

    r_analytical = np.array([
        float(radius_perturbation(t, theta, R0, epsilon, l, omega, beta))
        for t, theta in zip(t_arr, theta_apex_arr)
    ])

    ax.plot(t_arr, r_apex_arr, 'b-', label=r'Numerical $r_{\mathrm{apex}}(t)$')
    ax.plot(t_arr, r_analytical, 'r--', label='Rayleigh–Lamb (analytical)')
    ax.axhline(R0, color='k', linestyle=':', alpha=0.5, label=f'$R_0={R0}$')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Interface radius at apex [m]")
    ax.set_title(title or r"Interface apex trajectory ($\theta\approx 0$)")
    ax.legend()
    return fig, ax


def plot_energy_history(t_arr, KE_arr, ax=None, title: str = ""):
    """Kinetic energy vs time."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.semilogy(t_arr, np.maximum(KE_arr, 1e-30), 'b-', label='KE')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Kinetic energy [J]")
    ax.set_title(title or "Kinetic energy decay")
    ax.legend()
    return fig, ax


def compute_diagnostics(HC, dim: int = 2, theta_ref: float = 0.0):
    """Compute diagnostic quantities from current mesh state.

    Parameters
    ----------
    HC : Complex
    dim : int
    theta_ref : float
        Reference polar angle used to locate the "apex" tracer: the
        interface vertex whose angle is closest to ``theta_ref`` is
        returned as ``r_apex``. Re-searched every call because
        retopology destroys per-vertex tag attributes.

    Returns
    -------
    dict with keys:
        R_max, R_min : extreme interface radii
        KE : total kinetic energy
        total_mass : sum of all vertex masses
        com : centre of mass position
        r_apex : interface radius at the vertex nearest theta_ref
        theta_apex : actual angle of that vertex (drifts from theta_ref
                     as tangential flow advects it)
        n_interface : number of interface vertices
    """
    R_max = 0.0
    R_min = np.inf
    KE = 0.0
    total_mass = 0.0
    com = np.zeros(dim)

    best_dtheta = np.inf
    r_apex = 0.0
    theta_apex = 0.0
    n_interface = 0

    for v in HC.V:
        total_mass += v.m
        com += v.m * v.x_a[:dim]
        u = v.u[:dim]
        KE += 0.5 * v.m * np.dot(u, u)

        if getattr(v, 'is_interface', False):
            x = v.x_a[:dim]
            r = float(np.linalg.norm(x))
            R_max = max(R_max, r)
            R_min = min(R_min, r)
            n_interface += 1
            if dim == 2:
                theta = float(np.arctan2(x[1], x[0]))
            else:
                # 3D: polar angle from +x (apex of the +x axis)
                theta = float(np.arctan2(np.sqrt(x[1]**2 + x[2]**2), x[0]))
            # Circular distance to theta_ref
            dtheta = abs((theta - theta_ref + np.pi) % (2 * np.pi) - np.pi)
            if dtheta < best_dtheta:
                best_dtheta = dtheta
                r_apex = r
                theta_apex = theta

    if total_mass > 0:
        com /= total_mass

    return {
        'R_max': R_max,
        'R_min': R_min if R_min < np.inf else 0.0,
        'KE': KE,
        'total_mass': total_mass,
        'com': com,
        'r_apex': r_apex,
        'theta_apex': theta_apex,
        'n_interface': n_interface,
    }


# -----------------------------------------------------------------------
# Animation frame capture (for custom FuncAnimation with interface markers)
# -----------------------------------------------------------------------

def record_frame(HC, t: float, dim: int = 2) -> dict:
    """Capture per-vertex data for one animation frame.

    Returns a dict with arrays for positions, fields, phase, and
    interface membership — everything needed to render a single frame.
    """
    xs, ps, us, phases, is_iface = [], [], [], [], []
    for v in HC.V:
        xs.append(v.x_a[:dim].copy())
        ps.append(float(v.p) if np.ndim(v.p) == 0 else float(v.p[0]))
        us.append(v.u[:dim].copy())
        phases.append(int(v.phase))
        is_iface.append(bool(getattr(v, 'is_interface', False)))
    return {
        't': t,
        'x': np.array(xs),
        'p': np.array(ps),
        'u': np.array(us),
        'phase': np.array(phases),
        'is_interface': np.array(is_iface),
    }


def make_multiphase_animation(
    frames: list[dict],
    R0: float,
    zoom: float = None,
    save_path: str = 'fig/droplet_2D.mp4',
    fps: int = 20,
    dpi: int = 120,
):
    """Build a matplotlib FuncAnimation from recorded frames.

    Two-panel figure: pressure colormap + velocity quiver, with
    interface vertices highlighted as red markers and the equilibrium
    circle drawn as a dashed line.

    Parameters
    ----------
    frames : list of dict
        Each from ``record_frame()``.
    R0 : float
        Equilibrium radius (for reference circle).
    zoom : float or None
        Half-extent of the view window.  If None, uses 2.5*R0.
    save_path : str
        Output file path (.mp4, .gif, .avi).
    fps : int
        Playback frames per second.
    dpi : int
        Frame resolution.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import os

    if not frames:
        print("No frames to animate.")
        return None

    if zoom is None:
        zoom = 2.5 * R0

    # Global pressure bounds for stable colorbar
    all_p = np.concatenate([f['p'] for f in frames])
    p_min, p_max = np.nanmin(all_p), np.nanmax(all_p)
    if abs(p_max - p_min) < 1e-30:
        p_min, p_max = p_min - 1, p_max + 1

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=0.88)
    suptitle = fig.suptitle('t = 0.0000 s', fontsize=13)

    def update(idx):
        f = frames[idx]
        x, p, u = f['x'], f['p'], f['u']
        phase, is_if = f['phase'], f['is_interface']

        for ax in (ax_p, ax_v):
            ax.clear()
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(-zoom, zoom)
            ax.set_aspect('equal')
            ax.add_patch(plt.Circle((0, 0), R0, fill=False,
                                     color='k', ls='--', lw=1, alpha=0.5))

        suptitle.set_text(f't = {f["t"]:.4f} s')

        # Pressure panel
        sc = ax_p.scatter(x[:, 0], x[:, 1], c=p, s=8, cmap='viridis',
                          vmin=p_min, vmax=p_max, edgecolors='none')
        if is_if.any():
            ax_p.scatter(x[is_if, 0], x[is_if, 1], c='red', s=15,
                         zorder=5, edgecolors='none')
        ax_p.set_title('Pressure [Pa]')

        # Velocity panel
        u_mag = np.linalg.norm(u, axis=1)
        ax_v.scatter(x[:, 0], x[:, 1], c=u_mag, s=8, cmap='coolwarm',
                     edgecolors='none')
        mask = u_mag > 1e-15
        if mask.any():
            ax_v.quiver(x[mask, 0], x[mask, 1], u[mask, 0], u[mask, 1],
                        scale=max(u_mag.max() * 15, 1e-10), alpha=0.7,
                        width=0.003)
        if is_if.any():
            ax_v.scatter(x[is_if, 0], x[is_if, 1], c='red', s=15,
                         zorder=5, edgecolors='none')
        ax_v.set_title('Velocity [m/s]')

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=False)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    ext = os.path.splitext(save_path)[1].lower()
    writer = 'pillow' if ext == '.gif' else 'ffmpeg'
    try:
        anim.save(save_path, writer=writer, fps=fps, dpi=dpi)
        print(f"Animation saved: {save_path} ({len(frames)} frames)")
    except Exception as e:
        # Fallback: save individual frames
        frame_dir = save_path.rsplit('.', 1)[0] + '_frames'
        os.makedirs(frame_dir, exist_ok=True)
        for i in range(len(frames)):
            update(i)
            fig.savefig(os.path.join(frame_dir, f'frame_{i:04d}.png'), dpi=dpi)
        print(f"Frames saved to {frame_dir}/ ({len(frames)} frames)")
        print(f"(animation writer '{writer}' failed: {e})")

    plt.close(fig)
    return anim
