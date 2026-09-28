"""2D visualization: scalar fields, vector fields (quiver), and mesh plots."""

import numpy as np


def plot_scalar_field_2d(
    HC,
    field: str = 'p',
    ax=None,
    cmap: str = 'viridis',
    s: float = 30,
    colorbar: bool = True,
    title: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
    **scatter_kwargs,
):
    """Scatter plot of a scalar field on a 2D mesh.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (2D).
    field : str
        Vertex attribute name (default 'p').
    ax : matplotlib Axes or None
        If None, a new figure is created.
    cmap : str
        Colormap name.
    s : float
        Marker size.
    colorbar : bool
        Whether to add a colorbar.
    title : str or None
        Plot title.
    xlim : tuple of (float, float) or None
        Restrict x-axis range for zooming.
    ylim : tuple of (float, float) or None
        Restrict y-axis range for zooming.
    **scatter_kwargs
        Forwarded to ``ax.scatter()``.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    x = np.array([float(v.x_a[0]) for v in HC.V])
    y = np.array([float(v.x_a[1]) for v in HC.V])
    vals = []
    for v in HC.V:
        val = getattr(v, field)
        vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))
    c = np.array(vals)

    sc = ax.scatter(x, y, c=c, cmap=cmap, s=s, **scatter_kwargs)
    if colorbar:
        fig.colorbar(sc, ax=ax, label=field)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    return fig, ax


def plot_vector_field_2d(
    HC,
    ax=None,
    scale: float = 1.0,
    bV: set = None,
    title: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
    **quiver_kwargs,
):
    """Quiver plot of the velocity field on a 2D mesh.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (2D).
    ax : matplotlib Axes or None
        If None, a new figure is created.
    scale : float
        Arrow scaling factor.
    bV : set or None
        If provided, boundary vertices are marked differently.
    title : str or None
        Plot title.
    xlim : tuple of (float, float) or None
        Restrict x-axis range for zooming.
    ylim : tuple of (float, float) or None
        Restrict y-axis range for zooming.
    **quiver_kwargs
        Forwarded to ``ax.quiver()``.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    x = np.array([float(v.x_a[0]) for v in HC.V])
    y = np.array([float(v.x_a[1]) for v in HC.V])
    u = np.array([float(v.u[0]) for v in HC.V])
    w = np.array([float(v.u[1]) for v in HC.V])

    ax.quiver(x, y, u, w, scale=scale, **quiver_kwargs)

    if bV is not None:
        bx = np.array([float(v.x_a[0]) for v in bV])
        by = np.array([float(v.x_a[1]) for v in bV])
        ax.scatter(bx, by, c='red', s=10, zorder=5, label='boundary')
        ax.legend()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    return fig, ax


def plot_mesh_2d(
    HC,
    ax=None,
    bV: set = None,
    vertex_color: str = 'blue',
    edge_color: str = 'gray',
    boundary_color: str = 'red',
    vertex_size: float = 15,
    title: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
):
    """Plot mesh vertices and edges for a 2D simplicial complex.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (2D).
    ax : matplotlib Axes or None
        If None, a new figure is created.
    bV : set or None
        Boundary vertices highlighted in boundary_color.
    vertex_color, edge_color, boundary_color : str
        Colors for vertices, edges, boundary vertices.
    vertex_size : float
        Marker size for vertices.
    title : str or None
        Plot title.
    xlim : tuple of (float, float) or None
        Restrict x-axis range for zooming.
    ylim : tuple of (float, float) or None
        Restrict y-axis range for zooming.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Draw edges
    seen = set()
    for v in HC.V:
        for nb in v.nn:
            key = frozenset((v.x, nb.x))
            if key not in seen:
                seen.add(key)
                ax.plot(
                    [v.x_a[0], nb.x_a[0]],
                    [v.x_a[1], nb.x_a[1]],
                    color=edge_color, lw=0.5, alpha=0.5,
                )

    # Draw vertices
    interior = [v for v in HC.V if bV is None or v not in bV]
    if interior:
        ix = [v.x_a[0] for v in interior]
        iy = [v.x_a[1] for v in interior]
        ax.scatter(ix, iy, c=vertex_color, s=vertex_size, zorder=3)

    if bV:
        bx = [v.x_a[0] for v in bV]
        by = [v.x_a[1] for v in bV]
        ax.scatter(bx, by, c=boundary_color, s=vertex_size, zorder=4,
                   label='boundary')
        ax.legend()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    return fig, ax


# ---------------------------------------------------------------------------
# Interpolated (pointwise) field visualization
# ---------------------------------------------------------------------------

def _get_dual_measure(v, HC, dim):
    """Get the dual cell measure for a vertex (length/area/volume for 1D/2D/3D).

    Uses ``dual_volume`` from ``ddgclib.operators.stress`` (which dispatches
    to ``d_area`` for 2D or ``v_star`` for 3D).  Falls back to ``d_area``
    from ``hyperct.ddg`` if stress module is unavailable.

    Returns 0.0 for boundary vertices without proper dual cells.
    """
    try:
        from ddgclib.operators.stress import dual_volume
        return dual_volume(v, HC, dim)
    except Exception:
        pass
    try:
        from hyperct.ddg import d_area
        return d_area(v)
    except Exception:
        return 0.0


def plot_interpolated_scalar_2d(
    HC,
    field: str = 'p',
    ax=None,
    cmap: str = 'viridis',
    title: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
    colorbar: bool = True,
    levels: int = 32,
    method: str = 'tricontourf',
):
    """Interpolated (smooth) plot of a scalar field normalized by dual area.

    Because the DDG stress tensor pipeline stores *integrated* quantities
    on vertices, the raw vertex value is an extensive (cell-averaged) quantity.
    To obtain a pointwise (intensive) field for smooth visualization, we
    divide each vertex value by the local dual cell area (2D):

        field_pointwise(v) = field_integrated(v) / dual_area(v)

    The resulting pointwise values are then plotted using Delaunay-based
    ``tricontourf`` (filled contours) or ``tripcolor`` (flat triangle shading).

    Parameters
    ----------
    HC : Complex
        Simplicial complex (2D) with duals computed.
    field : str
        Vertex attribute name (default ``'p'``).
    ax : matplotlib Axes or None
        If None, a new figure is created.
    cmap : str
        Colormap name.
    title : str or None
        Plot title.
    xlim : tuple of (float, float) or None
        Restrict x-axis range for zooming.
    ylim : tuple of (float, float) or None
        Restrict y-axis range for zooming.
    colorbar : bool
        Whether to add a colorbar.
    levels : int
        Number of contour levels (for ``tricontourf``).
    method : str
        ``'tricontourf'`` (smooth contours) or ``'tripcolor'`` (flat shading).

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    dim = 2
    x_arr, y_arr, vals = [], [], []
    for v in HC.V:
        x_arr.append(float(v.x_a[0]))
        y_arr.append(float(v.x_a[1]))
        raw = getattr(v, field)
        raw_val = float(raw) if np.ndim(raw) == 0 else float(raw[0])
        dual_m = _get_dual_measure(v, HC, dim)
        if dual_m > 1e-30:
            vals.append(raw_val / dual_m)
        else:
            vals.append(raw_val)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    vals = np.array(vals)

    triang = mtri.Triangulation(x_arr, y_arr)

    if method == 'tripcolor':
        tc = ax.tripcolor(triang, vals, cmap=cmap, shading='gouraud')
    else:
        tc = ax.tricontourf(triang, vals, levels=levels, cmap=cmap)

    if colorbar:
        fig.colorbar(tc, ax=ax, label=f'{field} (pointwise)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    return fig, ax


def plot_interpolated_vector_2d(
    HC,
    ax=None,
    cmap: str = 'coolwarm',
    title: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
    colorbar: bool = True,
    levels: int = 32,
    component: int = 0,
    method: str = 'tricontourf',
):
    """Interpolated (smooth) plot of a velocity component normalized by dual area.

    Same normalization as ``plot_interpolated_scalar_2d`` but for a vector
    field component: divides the integrated velocity by the dual cell area
    to obtain the pointwise velocity.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (2D) with duals computed.
    ax : matplotlib Axes or None
        If None, a new figure is created.
    cmap : str
        Colormap name.
    title : str or None
        Plot title.
    xlim : tuple of (float, float) or None
        Restrict x-axis range for zooming.
    ylim : tuple of (float, float) or None
        Restrict y-axis range for zooming.
    colorbar : bool
        Whether to add a colorbar.
    levels : int
        Number of contour levels (for ``tricontourf``).
    component : int
        Velocity component index (0=u_x, 1=u_y).
    method : str
        ``'tricontourf'`` (smooth contours) or ``'tripcolor'`` (flat shading).

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    dim = 2
    x_arr, y_arr, vals = [], [], []
    for v in HC.V:
        x_arr.append(float(v.x_a[0]))
        y_arr.append(float(v.x_a[1]))
        u_comp = float(v.u[component])
        dual_m = _get_dual_measure(v, HC, dim)
        if dual_m > 1e-30:
            vals.append(u_comp / dual_m)
        else:
            vals.append(u_comp)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    vals = np.array(vals)

    triang = mtri.Triangulation(x_arr, y_arr)

    if method == 'tripcolor':
        tc = ax.tripcolor(triang, vals, cmap=cmap, shading='gouraud')
    else:
        tc = ax.tricontourf(triang, vals, levels=levels, cmap=cmap)

    comp_label = ['u_x', 'u_y'][component] if component < 2 else f'u_{component}'
    if colorbar:
        fig.colorbar(tc, ax=ax, label=f'{comp_label} (pointwise)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    return fig, ax
