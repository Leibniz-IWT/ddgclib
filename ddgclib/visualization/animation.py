"""Animation utilities for time-series visualization from StateHistory.

Usage
-----
    from ddgclib.data import StateHistory
    from ddgclib.visualization.animation import animate_scalar_1d

    history = StateHistory(fields=['P'])
    # ... run simulation with history.callback ...
    anim = animate_scalar_1d(history, field='P')
    anim.save('pressure_evolution.gif')
"""

import numpy as np


def animate_scalar_1d(
    history,
    field: str = 'P',
    interval: int = 100,
    xlabel: str = 'x',
    ylabel: str = None,
    title_fmt: str = 't = {t:.4f}',
    figsize: tuple = (8, 5),
):
    """Create a matplotlib animation of a 1D scalar field over time.

    Parameters
    ----------
    history : StateHistory
        Recorded simulation history.
    field : str
        Field name to animate.
    interval : int
        Milliseconds between frames.
    xlabel, ylabel : str
        Axis labels.
    title_fmt : str
        Format string for title (receives ``t`` as keyword).
    figsize : tuple
        Figure size.

    Returns
    -------
    anim : FuncAnimation
        Matplotlib animation object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=figsize)

    # Get data range from all snapshots
    all_x = []
    all_f = []
    for t, snapshot, _ in history._snapshots:
        for key, vdata in snapshot.items():
            all_x.append(key[0])  # 1D: first coordinate
            if field in vdata:
                val = vdata[field]
                all_f.append(float(val) if np.ndim(val) == 0 else float(val[0]))

    if not all_x:
        return None

    x_min, x_max = min(all_x), max(all_x)
    f_min, f_max = min(all_f), max(all_f)
    margin = max(abs(f_max - f_min) * 0.1, 1e-10)

    line, = ax.plot([], [], 'o-')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(f_min - margin, f_max + margin)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or field)
    ax.grid(True, alpha=0.3)
    title = ax.set_title('')

    def init():
        line.set_data([], [])
        return line, title

    def update(frame):
        t, snapshot, _ = history._snapshots[frame]
        x_vals = []
        f_vals = []
        for key, vdata in snapshot.items():
            if field in vdata:
                x_vals.append(key[0])
                val = vdata[field]
                f_vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))

        idx = np.argsort(x_vals)
        line.set_data(np.array(x_vals)[idx], np.array(f_vals)[idx])
        title.set_text(title_fmt.format(t=t))
        return line, title

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=history.n_snapshots,
        interval=interval, blit=True,
    )
    return anim


def animate_scalar_2d(
    history,
    field: str = 'P',
    interval: int = 100,
    cmap: str = 'viridis',
    s: float = 30,
    title_fmt: str = 't = {t:.4f}',
    figsize: tuple = (8, 7),
):
    """Create a matplotlib animation of a 2D scalar field over time.

    Parameters
    ----------
    history : StateHistory
        Recorded simulation history.
    field : str
        Field name to animate.
    interval : int
        Milliseconds between frames.
    cmap : str
        Colormap name.
    s : float
        Marker size.
    title_fmt : str
        Format string for title.
    figsize : tuple
        Figure size.

    Returns
    -------
    anim : FuncAnimation
        Matplotlib animation object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=figsize)

    # Determine global data range
    all_vals = []
    for _, snapshot, _ in history._snapshots:
        for key, vdata in snapshot.items():
            if field in vdata:
                val = vdata[field]
                all_vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))

    if not all_vals:
        return None

    vmin, vmax = min(all_vals), max(all_vals)

    # Initial empty scatter
    sc = ax.scatter([], [], c=[], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=ax, label=field)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    title = ax.set_title('')

    def update(frame):
        t, snapshot, _ = history._snapshots[frame]
        x = []
        y = []
        c = []
        for key, vdata in snapshot.items():
            if field in vdata:
                x.append(key[0])
                y.append(key[1])
                val = vdata[field]
                c.append(float(val) if np.ndim(val) == 0 else float(val[0]))

        offsets = np.column_stack([x, y]) if x else np.empty((0, 2))
        sc.set_offsets(offsets)
        sc.set_array(np.array(c))
        title.set_text(title_fmt.format(t=t))
        return sc, title

    anim = FuncAnimation(
        fig, update,
        frames=history.n_snapshots,
        interval=interval, blit=False,
    )
    return anim
