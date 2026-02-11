"""2D visualization: scalar fields, vector fields (quiver), and mesh plots."""

import numpy as np


def plot_scalar_field_2d(
    HC,
    field: str = 'P',
    ax=None,
    cmap: str = 'viridis',
    s: float = 30,
    colorbar: bool = True,
    title: str = None,
    **scatter_kwargs,
):
    """Scatter plot of a scalar field on a 2D mesh.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (2D).
    field : str
        Vertex attribute name (default 'P').
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
    if title:
        ax.set_title(title)

    return fig, ax


def plot_vector_field_2d(
    HC,
    ax=None,
    scale: float = 1.0,
    bV: set = None,
    title: str = None,
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
    if title:
        ax.set_title(title)

    return fig, ax
