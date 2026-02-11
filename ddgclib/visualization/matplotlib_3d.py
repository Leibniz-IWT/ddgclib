"""3D visualization: scatter plots and slice profile extraction."""

import numpy as np


def plot_scalar_field_3d(
    HC,
    field: str = 'P',
    ax=None,
    cmap: str = 'viridis',
    s: float = 15,
    alpha: float = 0.8,
    colorbar: bool = True,
    title: str = None,
    **scatter_kwargs,
):
    """3D scatter plot of a scalar field.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (3D).
    field : str
        Vertex attribute to color by (default 'P').
    ax : mpl_toolkits.mplot3d.Axes3D or None
        If None, a new 3D figure is created.
    cmap : str
        Colormap name.
    s : float
        Marker size.
    alpha : float
        Marker transparency.
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    x = np.asarray([float(v.x_a[0]) for v in HC.V], dtype=np.float64)
    y = np.asarray([float(v.x_a[1]) for v in HC.V], dtype=np.float64)
    z = np.asarray([float(v.x_a[2]) for v in HC.V], dtype=np.float64)

    vals = []
    for v in HC.V:
        val = getattr(v, field)
        vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))
    c = np.asarray(vals, dtype=np.float64)

    sc = ax.scatter(x, y, z, c=c, cmap=cmap, s=s, alpha=alpha, **scatter_kwargs)
    if colorbar:
        fig.colorbar(sc, ax=ax, label=field, shrink=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if title:
        ax.set_title(title)

    return fig, ax


def extract_slice_profile(
    HC,
    axis: int,
    position: float,
    tol: float = 0.05,
    sort_by: int = None,
):
    """Extract vertices near a plane perpendicular to the given axis.

    Generalizes ``extract_radial_profiles`` from the Hagen-Poiseuille case.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    axis : int
        Axis perpendicular to the slice plane (0=x, 1=y, 2=z).
    position : float
        Coordinate value along *axis* for the slice.
    tol : float
        Tolerance for vertex inclusion.
    sort_by : int or None
        If provided, sort returned vertices by this coordinate axis.

    Returns
    -------
    list
        List of vertex objects within the slice.
    """
    verts = [v for v in HC.V if abs(v.x_a[axis] - position) < tol]

    if sort_by is not None and verts:
        verts.sort(key=lambda v: v.x_a[sort_by])

    return verts
