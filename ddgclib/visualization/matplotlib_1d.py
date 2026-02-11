"""1D visualization: scalar fields and velocity profiles along a line."""

import numpy as np


def plot_scalar_field_1d(
    HC,
    field: str = 'P',
    ax=None,
    label: str = None,
    analytical_fn=None,
    analytical_label: str = 'Analytical',
    xlabel: str = 'x',
    ylabel: str = None,
    title: str = None,
    **plot_kwargs,
):
    """Plot a scalar field vs position for a 1D mesh.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (1D).
    field : str
        Vertex attribute name to plot (default 'P').
    ax : matplotlib Axes or None
        If None, a new figure is created.
    label : str or None
        Plot label.
    analytical_fn : callable or None
        If provided, ``fn(x) -> value`` plotted as a dashed reference line.
    analytical_label : str
        Label for analytical curve.
    xlabel, ylabel, title : str
        Axis labels and title.
    **plot_kwargs
        Forwarded to ``ax.plot()``.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    x_vals = []
    f_vals = []
    for v in HC.V:
        x_vals.append(float(v.x_a[0]))
        val = getattr(v, field)
        f_vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))

    idx = np.argsort(x_vals)
    x_arr = np.array(x_vals)[idx]
    f_arr = np.array(f_vals)[idx]

    ax.plot(x_arr, f_arr, 'o-', label=label or field, **plot_kwargs)

    if analytical_fn is not None:
        x_fine = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_anal = np.array([analytical_fn(xi) for xi in x_fine])
        ax.plot(x_fine, y_anal, 'k--', lw=1.5, label=analytical_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or field)
    if title:
        ax.set_title(title)
    if label or analytical_fn:
        ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_velocity_profile_1d(
    HC,
    component: int = 0,
    ax=None,
    label: str = 'u',
    analytical_fn=None,
    analytical_label: str = 'Analytical',
    xlabel: str = 'x',
    ylabel: str = 'u',
    title: str = None,
    **plot_kwargs,
):
    """Plot a velocity component vs position for a 1D mesh.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    component : int
        Velocity component index (default 0).
    ax : matplotlib Axes or None
        If None, a new figure is created.
    analytical_fn : callable or None
        If provided, ``fn(x) -> value`` plotted as a reference.
    **plot_kwargs
        Forwarded to ``ax.plot()``.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    x_vals = []
    u_vals = []
    for v in HC.V:
        x_vals.append(float(v.x_a[0]))
        u_vals.append(float(v.u[component]))

    idx = np.argsort(x_vals)
    x_arr = np.array(x_vals)[idx]
    u_arr = np.array(u_vals)[idx]

    ax.plot(x_arr, u_arr, 'o-', label=label, **plot_kwargs)

    if analytical_fn is not None:
        x_fine = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_anal = np.array([analytical_fn(xi) for xi in x_fine])
        ax.plot(x_fine, y_anal, 'k--', lw=1.5, label=analytical_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label or analytical_fn:
        ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
