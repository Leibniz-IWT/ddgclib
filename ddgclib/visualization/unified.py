"""Unified dimension-dispatching wrappers for primal and dual mesh visualization.

Provides ``plot_primal`` and ``plot_dual`` functions that detect ``HC.dim``
and delegate to the appropriate 1D / 2D / 3D plotting backend.  Both accept
optional *scalar_field* and *vector_field* parameters for field overlays.
"""

import os

import numpy as np

_DEFAULT_FIG_DIR = os.path.join('results', 'fig')


# Helpers

def _extract_scalar(HC, field: str):
    """Return (positions, values) arrays for a scalar vertex attribute."""
    coords = []
    vals = []
    for v in HC.V:
        coords.append(v.x_a.copy())
        val = getattr(v, field)
        vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))
    return np.array(coords), np.array(vals)


def _extract_vector(HC, field: str, dim: int):
    """Return (positions, vectors) arrays for a vector vertex attribute."""
    coords = []
    vecs = []
    for v in HC.V:
        coords.append(v.x_a[:dim].copy())
        vec = getattr(v, field)
        vecs.append(np.asarray(vec[:dim], dtype=np.float64))
    return np.array(coords), np.array(vecs)


# 1D primal helpers

def _plot_primal_1d(HC, bV, scalar_field, vector_field, ax, **kwargs):
    """Plot 1D primal mesh with optional field overlays."""
    import matplotlib.pyplot as plt

    cmap_name = kwargs.pop('cmap', 'viridis')
    title = kwargs.pop('title', None)
    vertex_size = kwargs.pop('vertex_size', 30)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Draw edges (line segments between neighbors)
    seen = set()
    for v in HC.V:
        for nb in v.nn:
            key = frozenset((v.x, nb.x))
            if key not in seen:
                seen.add(key)
                ax.plot([v.x_a[0], nb.x_a[0]], [0, 0],
                        color='gray', lw=0.8, alpha=0.5)

    # Draw vertices with optional scalar coloring
    x_arr = np.array([float(v.x_a[0]) for v in HC.V])
    y_arr = np.zeros_like(x_arr)

    if scalar_field is not None:
        _, c = _extract_scalar(HC, scalar_field)
        sc = ax.scatter(x_arr, y_arr, c=c, cmap=cmap_name, s=vertex_size,
                        zorder=3, edgecolors='k', linewidths=0.3)
        fig.colorbar(sc, ax=ax, label=scalar_field)
    else:
        # Color boundary vs interior
        if bV is not None:
            interior_x = [v.x_a[0] for v in HC.V if v not in bV]
            boundary_x = [v.x_a[0] for v in bV]
            ax.scatter(interior_x, [0]*len(interior_x), c='blue',
                       s=vertex_size, zorder=3)
            ax.scatter(boundary_x, [0]*len(boundary_x), c='red',
                       s=vertex_size, zorder=4, label='boundary')
            if boundary_x:
                ax.legend()
        else:
            ax.scatter(x_arr, y_arr, c='blue', s=vertex_size, zorder=3)

    # Vector field overlay
    if vector_field is not None:
        pos, vecs = _extract_vector(HC, vector_field, 1)
        magnitudes = np.abs(vecs[:, 0])
        if magnitudes.max() > 0:
            norm_mag = magnitudes / magnitudes.max()
        else:
            norm_mag = magnitudes
        cmap = plt.colormaps.get_cmap('coolwarm')
        colors = cmap(norm_mag)
        ax.quiver(pos[:, 0], np.zeros(len(pos)), vecs[:, 0],
                  np.zeros(len(pos)), color=colors, scale=kwargs.get('scale'),
                  zorder=5)

    ax.set_xlabel('x')
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    return fig, ax


# 2D primal helpers

def _plot_primal_2d(HC, bV, scalar_field, vector_field, ax, **kwargs):
    """Plot 2D primal mesh with optional field overlays."""
    import matplotlib.pyplot as plt

    cmap_name = kwargs.pop('cmap', 'viridis')
    title = kwargs.pop('title', None)
    vertex_size = kwargs.pop('vertex_size', 15)
    edge_color = kwargs.pop('edge_color', 'gray')
    show_edges = kwargs.pop('show_edges', True)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Draw edges
    if show_edges:
        seen = set()
        for v in HC.V:
            for nb in v.nn:
                key = frozenset((v.x, nb.x))
                if key not in seen:
                    seen.add(key)
                    ax.plot([v.x_a[0], nb.x_a[0]], [v.x_a[1], nb.x_a[1]],
                            color=edge_color, lw=0.5, alpha=0.5)

    # Vertices with optional scalar coloring
    x = np.array([float(v.x_a[0]) for v in HC.V])
    y = np.array([float(v.x_a[1]) for v in HC.V])

    if scalar_field is not None:
        _, c = _extract_scalar(HC, scalar_field)
        sc = ax.scatter(x, y, c=c, cmap=cmap_name, s=vertex_size, zorder=3)
        fig.colorbar(sc, ax=ax, label=scalar_field)
    else:
        if bV is not None:
            interior = [v for v in HC.V if v not in bV]
            if interior:
                ix = [v.x_a[0] for v in interior]
                iy = [v.x_a[1] for v in interior]
                ax.scatter(ix, iy, c='blue', s=vertex_size, zorder=3)
            bx = [v.x_a[0] for v in bV]
            by = [v.x_a[1] for v in bV]
            ax.scatter(bx, by, c='red', s=vertex_size, zorder=4,
                       label='boundary')
            ax.legend()
        else:
            ax.scatter(x, y, c='blue', s=vertex_size, zorder=3)

    # Vector field overlay
    if vector_field is not None:
        pos, vecs = _extract_vector(HC, vector_field, 2)
        magnitudes = np.linalg.norm(vecs, axis=1)
        if magnitudes.max() > 0:
            norm_mag = magnitudes / magnitudes.max()
        else:
            norm_mag = magnitudes
        cmap = plt.colormaps.get_cmap('coolwarm')
        colors = cmap(norm_mag)
        ax.quiver(pos[:, 0], pos[:, 1], vecs[:, 0], vecs[:, 1],
                  color=colors, scale=kwargs.get('scale'), zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)

    return fig, ax


# 3D primal helpers

def _plot_primal_3d(HC, bV, scalar_field, vector_field, ax, **kwargs):
    """Plot 3D primal mesh with optional field overlays."""
    import matplotlib.pyplot as plt

    cmap_name = kwargs.pop('cmap', 'viridis')
    title = kwargs.pop('title', None)
    vertex_size = kwargs.pop('vertex_size', 15)
    alpha = kwargs.pop('alpha', 0.8)
    show_edges = kwargs.pop('show_edges', True)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    x = np.array([float(v.x_a[0]) for v in HC.V])
    y = np.array([float(v.x_a[1]) for v in HC.V])
    z = np.array([float(v.x_a[2]) for v in HC.V])

    # Draw edges
    if show_edges:
        seen = set()
        for v in HC.V:
            for nb in v.nn:
                key = frozenset((v.x, nb.x))
                if key not in seen:
                    seen.add(key)
                    ax.plot([v.x_a[0], nb.x_a[0]],
                            [v.x_a[1], nb.x_a[1]],
                            [v.x_a[2], nb.x_a[2]],
                            color='gray', lw=0.3, alpha=0.3)

    # Vertices with optional scalar coloring
    if scalar_field is not None:
        _, c = _extract_scalar(HC, scalar_field)
        sc = ax.scatter(x, y, z, c=c, cmap=cmap_name, s=vertex_size,
                        alpha=alpha, zorder=3)
        fig.colorbar(sc, ax=ax, label=scalar_field, shrink=0.6)
    else:
        ax.scatter(x, y, z, c='blue', s=vertex_size, alpha=alpha, zorder=3)

    # Vector field overlay
    if vector_field is not None:
        pos, vecs = _extract_vector(HC, vector_field, 3)
        magnitudes = np.linalg.norm(vecs, axis=1)
        if magnitudes.max() > 0:
            norm_mag = magnitudes / magnitudes.max()
        else:
            norm_mag = magnitudes
        cmap = plt.colormaps.get_cmap('coolwarm')
        colors = cmap(norm_mag)
        ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                  vecs[:, 0], vecs[:, 1], vecs[:, 2],
                  colors=colors, length=kwargs.get('arrow_length', 0.05),
                  normalize=True, alpha=0.7)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if title:
        ax.set_title(title)

    return fig, ax


# 2D dual helpers

def _plot_dual_edges_2d(HC, bV, scalar_field, vector_field, ax, **kwargs):
    """Plot 2D barycentric dual edges connecting dual vertices.

    Dual vertices (barycenters of triangles sharing an edge) are connected
    to form the dual mesh.  Requires ``compute_vd(HC)`` to have been called
    so that each vertex ``v`` has ``v.vd`` populated and ``HC.Vd`` exists.
    """
    import matplotlib.pyplot as plt

    cmap_name = kwargs.pop('cmap', 'viridis')
    title = kwargs.pop('title', None)
    dual_color = kwargs.pop('dual_color', 'darkorange')
    show_primal = kwargs.pop('show_primal', True)
    vertex_size = kwargs.pop('vertex_size', 10)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Draw primal edges in light gray for reference
    if show_primal:
        seen = set()
        for v in HC.V:
            for nb in v.nn:
                key = frozenset((v.x, nb.x))
                if key not in seen:
                    seen.add(key)
                    ax.plot([v.x_a[0], nb.x_a[0]], [v.x_a[1], nb.x_a[1]],
                            color='lightgray', lw=0.5, alpha=0.5)

    # Draw dual edges: for each primal edge, find shared dual vertices
    # and connect them through the edge midpoint
    has_vd = hasattr(next(iter(HC.V)), 'vd')
    if not has_vd:
        import warnings
        warnings.warn(
            "Dual vertices not computed. Call compute_vd(HC) first. "
            "Showing primal mesh only."
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        if title:
            ax.set_title(title)
        return fig, ax

    # Collect all dual vertices and draw dual edges
    dual_pts_set = set()
    dual_edge_set = set()
    seen = set()
    for v1 in HC.V:
        for v2 in v1.nn:
            key = frozenset((v1.x, v2.x))
            if key in seen:
                continue
            seen.add(key)
            # Shared dual vertices between v1 and v2
            shared_vd = v1.vd.intersection(v2.vd)
            # Edge midpoint (center of primal edge)
            midpt = 0.5 * (v1.x_a + v2.x_a)
            mid_key = tuple(midpt)
            # Draw lines from each shared dual vertex to the midpoint
            for vd in shared_vd:
                dual_pts_set.add(vd.x)
                edge_key = frozenset((vd.x, mid_key))
                if edge_key not in dual_edge_set:
                    dual_edge_set.add(edge_key)
                    ax.plot([vd.x[0], midpt[0]], [vd.x[1], midpt[1]],
                            color=dual_color, lw=0.8, alpha=0.7)

    # Draw dual vertices
    if dual_pts_set:
        dx = [p[0] for p in dual_pts_set]
        dy = [p[1] for p in dual_pts_set]

        if scalar_field is not None:
            # Interpolate scalar field to dual vertices (average of primal
            # vertices whose vd set contains this dual vertex)
            dual_vals = _interpolate_scalar_to_dual_2d(HC, scalar_field,
                                                       dual_pts_set)
            sc = ax.scatter(dx, dy, c=dual_vals, cmap=cmap_name,
                            s=vertex_size, zorder=4, edgecolors='k',
                            linewidths=0.3)
            fig.colorbar(sc, ax=ax, label=f'{scalar_field} (dual)')
        else:
            ax.scatter(dx, dy, c=dual_color, s=vertex_size, zorder=4,
                       edgecolors='k', linewidths=0.3)

    # Vector field overlay on dual vertices
    if vector_field is not None and dual_pts_set:
        _overlay_vector_dual_2d(HC, vector_field, dual_pts_set, ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)

    return fig, ax


def _interpolate_scalar_to_dual_2d(HC, field, dual_pts_set):
    """Interpolate a scalar field from primal to dual vertices by averaging."""
    # Build dual_coord -> list of contributing primal vertex values
    from collections import defaultdict
    accum = defaultdict(list)
    for v in HC.V:
        val = getattr(v, field)
        val = float(val) if np.ndim(val) == 0 else float(val[0])
        if hasattr(v, 'vd'):
            for vd in v.vd:
                if vd.x in dual_pts_set:
                    accum[vd.x].append(val)
    # Return values in the same order as dual_pts_set
    result = []
    for pt in dual_pts_set:
        contributors = accum.get(pt, [0.0])
        result.append(float(np.mean(contributors)))
    return np.array(result)


def _overlay_vector_dual_2d(HC, field, dual_pts_set, ax):
    """Overlay a vector field interpolated to dual vertices as arrows."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    accum = defaultdict(list)
    dim = 2
    for v in HC.V:
        vec = getattr(v, field)
        vec = np.asarray(vec[:dim], dtype=np.float64)
        if hasattr(v, 'vd'):
            for vd in v.vd:
                if vd.x in dual_pts_set:
                    accum[vd.x].append(vec)
    px, py, ux, uy = [], [], [], []
    for pt in dual_pts_set:
        contributors = accum.get(pt, [np.zeros(dim)])
        avg = np.mean(contributors, axis=0)
        px.append(pt[0])
        py.append(pt[1])
        ux.append(avg[0])
        uy.append(avg[1])
    px, py = np.array(px), np.array(py)
    ux, uy = np.array(ux), np.array(uy)
    magnitudes = np.sqrt(ux**2 + uy**2)
    if magnitudes.max() > 0:
        norm_mag = magnitudes / magnitudes.max()
    else:
        norm_mag = magnitudes
    cmap = plt.colormaps.get_cmap('coolwarm')
    colors = cmap(norm_mag)
    ax.quiver(px, py, ux, uy, color=colors, zorder=5)


# 3D dual helpers

def _plot_dual_3d(HC, vertex, bV, scalar_field, vector_field, ax, **kwargs):
    """Plot 3D dual around a single vertex using matplotlib.

    In 3D the dual cell of a single vertex is visualized: the dual vertices
    (barycenters) associated with *vertex* are shown together with the dual
    surface triangles connecting them through the edge midpoints.
    """
    import matplotlib.pyplot as plt

    cmap_name = kwargs.pop('cmap', 'viridis')
    title = kwargs.pop('title', None)
    dual_color = kwargs.pop('dual_color', 'darkorange')
    show_primal = kwargs.pop('show_primal', True)
    alpha = kwargs.pop('alpha', 0.5)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    has_vd = hasattr(vertex, 'vd') and vertex.vd
    if not has_vd:
        import warnings
        warnings.warn(
            "Dual vertices not computed for vertex. Call compute_vd(HC) first."
        )
        if title:
            ax.set_title(title)
        return fig, ax

    # Show primal star (edges from vertex to its 1-ring)
    if show_primal:
        for nb in vertex.nn:
            ax.plot([vertex.x_a[0], nb.x_a[0]],
                    [vertex.x_a[1], nb.x_a[1]],
                    [vertex.x_a[2], nb.x_a[2]],
                    color='gray', lw=0.8, alpha=0.5)
        ax.scatter([vertex.x_a[0]], [vertex.x_a[1]], [vertex.x_a[2]],
                   c='blue', s=30, zorder=5)

    # Plot dual vertices
    dual_pts = list(vertex.vd)
    dx = [vd.x[0] for vd in dual_pts]
    dy = [vd.x[1] for vd in dual_pts]
    dz = [vd.x[2] for vd in dual_pts]

    if scalar_field is not None:
        from collections import defaultdict
        accum = defaultdict(list)
        val = getattr(vertex, scalar_field)
        val = float(val) if np.ndim(val) == 0 else float(val[0])
        for nb in vertex.nn:
            nb_val = getattr(nb, scalar_field)
            nb_val = float(nb_val) if np.ndim(nb_val) == 0 else float(nb_val[0])
            shared = vertex.vd.intersection(nb.vd)
            for vd in shared:
                accum[vd.x].append(np.mean([val, nb_val]))
        c_vals = [float(np.mean(accum.get(vd.x, [val]))) for vd in dual_pts]
        sc = ax.scatter(dx, dy, dz, c=c_vals, cmap=cmap_name, s=20,
                        zorder=4, edgecolors='k', linewidths=0.3)
        fig.colorbar(sc, ax=ax, label=f'{scalar_field} (dual)', shrink=0.6)
    else:
        ax.scatter(dx, dy, dz, c=dual_color, s=20, zorder=4)

    # Draw dual edges connecting dual vertices through edge midpoints
    for nb in vertex.nn:
        midpt = 0.5 * (vertex.x_a + nb.x_a)
        shared = vertex.vd.intersection(nb.vd)
        for vd in shared:
            ax.plot([vd.x[0], midpt[0]], [vd.x[1], midpt[1]],
                    [vd.x[2], midpt[2]], color=dual_color, lw=0.8, alpha=0.6)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if title:
        ax.set_title(title)

    return fig, ax


# 1D dual helpers

def _plot_dual_1d(HC, bV, scalar_field, vector_field, ax, **kwargs):
    """Plot 1D dual mesh.

    In 1D the dual vertices are edge midpoints.  Dual 'cells' are
    intervals between consecutive midpoints, drawn as colored segments.
    """
    import matplotlib.pyplot as plt

    title = kwargs.pop('title', None)
    cmap_name = kwargs.pop('cmap', 'viridis')
    dual_color = kwargs.pop('dual_color', 'darkorange')
    show_primal = kwargs.pop('show_primal', True)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Primal vertices as reference
    if show_primal:
        x_arr = sorted([float(v.x_a[0]) for v in HC.V])
        ax.scatter(x_arr, [0]*len(x_arr), c='blue', s=20, zorder=3,
                   label='primal')

    # Dual vertices = edge midpoints
    midpoints = set()
    seen = set()
    for v in HC.V:
        for nb in v.nn:
            key = frozenset((v.x, nb.x))
            if key not in seen:
                seen.add(key)
                mid = 0.5 * (float(v.x_a[0]) + float(nb.x_a[0]))
                midpoints.add(mid)

    dual_x = sorted(midpoints)
    ax.scatter(dual_x, [0.05]*len(dual_x), c=dual_color, s=20, zorder=4,
               marker='D', label='dual')

    if scalar_field is not None:
        # Interpolate scalar to dual midpoints
        coords_vals = sorted([(float(v.x_a[0]), getattr(v, scalar_field))
                              for v in HC.V])
        for mid in dual_x:
            # Average of the two endpoint values
            below = [cv for cv in coords_vals if cv[0] <= mid + 1e-14]
            above = [cv for cv in coords_vals if cv[0] >= mid - 1e-14]
            if below and above:
                val = 0.5 * (float(below[-1][1]) + float(above[0][1]))
                ax.annotate(f'{val:.2f}', (mid, 0.08), fontsize=7,
                            ha='center')

    ax.legend(fontsize=8)
    ax.set_xlabel('x')
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    return fig, ax


# PUBLIC API

def _save_fig(fig, save_path, dpi=150):
    """Save *fig* to *save_path*, creating parent directories as needed."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')


def plot_primal(
    HC,
    bV=None,
    scalar_field: str = None,
    vector_field: str = None,
    ax=None,
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'primal.png'),
    dpi: int = 150,
    **kwargs,
):
    """Plot the primal mesh with optional scalar/vector field overlays.

    Detects ``HC.dim`` and dispatches to the appropriate 1D / 2D / 3D
    plotting function.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with ``dim`` attribute.
    bV : set or None
        Boundary vertices (highlighted differently).
    scalar_field : str or None
        Vertex attribute name for scalar coloring (e.g. ``'P'``).
    vector_field : str or None
        Vertex attribute name for arrow overlay (e.g. ``'u'``).
    ax : matplotlib Axes or None
        If None a new figure is created.
    save_path : str or None
        File path to save the figure.  Set to ``None`` to skip saving.
        Default: ``results/fig/primal.png``.
    dpi : int
        Resolution for the saved image (default 150).
    **kwargs
        Forwarded to the dimension-specific backend.  Common options:

        - ``title`` (str): plot title
        - ``cmap`` (str): colormap name (default ``'viridis'``)
        - ``vertex_size`` (float): marker size
        - ``show_edges`` (bool): draw primal edges (2D/3D, default True)
        - ``scale`` (float): quiver scale for vector field

    Returns
    -------
    fig, ax
    """
    dim = HC.dim
    if dim == 1:
        fig, ax = _plot_primal_1d(HC, bV, scalar_field, vector_field, ax,
                                   **kwargs)
    elif dim == 2:
        fig, ax = _plot_primal_2d(HC, bV, scalar_field, vector_field, ax,
                                   **kwargs)
    elif dim == 3:
        fig, ax = _plot_primal_3d(HC, bV, scalar_field, vector_field, ax,
                                   **kwargs)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    if save_path is not None:
        _save_fig(fig, save_path, dpi=dpi)

    return fig, ax


def plot_dual(
    HC,
    bV=None,
    scalar_field: str = None,
    vector_field: str = None,
    vertex=None,
    ax=None,
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'dual.png'),
    dpi: int = 150,
    **kwargs,
):
    """Plot the dual mesh with optional scalar/vector field overlays.

    Detects ``HC.dim`` and dispatches to the appropriate backend:

    - **1D**: dual vertices are edge midpoints shown above the primal line.
    - **2D**: full dual edge network connecting barycenters through edge
      midpoints.
    - **3D**: dual cell of a single *vertex* (must be provided).

    Parameters
    ----------
    HC : Complex
        Simplicial complex with ``dim`` attribute.
    bV : set or None
        Boundary vertices.
    scalar_field : str or None
        Vertex attribute name for scalar coloring (e.g. ``'P'``).
        Values are interpolated from primal vertices to dual vertices.
    vector_field : str or None
        Vertex attribute name for arrow overlay (e.g. ``'u'``).
        Values are interpolated from primal vertices to dual vertices.
    vertex : vertex object or None
        **Required for 3D**: the specific primal vertex whose dual cell
        to visualize.
    ax : matplotlib Axes or None
        If None a new figure is created.
    save_path : str or None
        File path to save the figure.  Set to ``None`` to skip saving.
        Default: ``results/fig/dual.png``.
    dpi : int
        Resolution for the saved image (default 150).
    **kwargs
        Forwarded to the dimension-specific backend.  Common options:

        - ``title`` (str): plot title
        - ``cmap`` (str): colormap name
        - ``dual_color`` (str): color for dual edges/vertices
        - ``show_primal`` (bool): draw primal mesh as background reference

    Returns
    -------
    fig, ax
    """
    dim = HC.dim
    if dim == 1:
        fig, ax = _plot_dual_1d(HC, bV, scalar_field, vector_field, ax,
                                 **kwargs)
    elif dim == 2:
        fig, ax = _plot_dual_edges_2d(HC, bV, scalar_field, vector_field, ax,
                                       **kwargs)
    elif dim == 3:
        if vertex is None:
            raise ValueError(
                "For 3D dual plots, a specific *vertex* must be provided "
                "(pass vertex=some_vertex)."
            )
        fig, ax = _plot_dual_3d(HC, vertex, bV, scalar_field, vector_field,
                                 ax, **kwargs)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    if save_path is not None:
        _save_fig(fig, save_path, dpi=dpi)

    return fig, ax


# Polyscope unified wrappers

def plot_primal_polyscope(
    HC,
    scalar_fields: list[str] = None,
    vector_fields: list[str] = None,
    name: str = 'primal',
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'primal_ps.png'),
):
    """Register and display the primal mesh in polyscope with field overlays.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    scalar_fields : list of str or None
        Scalar field names to add as quantities (e.g. ``['P']``).
    vector_fields : list of str or None
        Vector field names to add as quantities (e.g. ``['u']``).
    name : str
        Point cloud name in polyscope.
    save_path : str or None
        File path to save a screenshot.  Set to ``None`` to skip saving.
        Default: ``results/fig/primal_ps.png``.

    Returns
    -------
    ps_cloud
        Polyscope PointCloud object.
    """
    from ddgclib.visualization.polyscope_3d import (
        _check_polyscope, register_point_cloud, update_frame,
    )
    ps = _check_polyscope()
    ps_cloud = register_point_cloud(HC, name=name, dim=HC.dim)
    update_frame(HC, ps_cloud, scalar_fields=scalar_fields,
                 vector_fields=vector_fields, dim=HC.dim)
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ps.screenshot(save_path)
    return ps_cloud


def plot_dual_polyscope(
    HC,
    vertex=None,
    scalar_fields: list[str] = None,
    name: str = 'dual',
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'dual_ps.png'),
):
    """Register dual vertices in polyscope as a point cloud.

    For 3D, *vertex* should be provided to show the dual cell of a
    specific primal vertex.  For 2D, all dual vertices are shown.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with ``compute_vd`` already called.
    vertex : vertex object or None
        For 3D: specific primal vertex.
    scalar_fields : list of str or None
        Scalar field names interpolated to dual vertices.
    name : str
        Point cloud name in polyscope.
    save_path : str or None
        File path to save a screenshot.  Set to ``None`` to skip saving.
        Default: ``results/fig/dual_ps.png``.

    Returns
    -------
    ps_cloud
        Polyscope PointCloud object, or None if polyscope unavailable.
    """
    from ddgclib.visualization.polyscope_3d import _check_polyscope
    ps = _check_polyscope()

    dim = HC.dim
    if dim == 3 and vertex is not None:
        dual_pts = list(vertex.vd)
        points = np.array([vd.x[:dim] for vd in dual_pts], dtype=np.float64)
    else:
        # Collect all unique dual vertices
        all_vd = set()
        for v in HC.V:
            if hasattr(v, 'vd'):
                all_vd.update(v.vd)
        dual_pts = list(all_vd)
        points = np.array([vd.x[:dim] for vd in dual_pts], dtype=np.float64)

    if len(points) == 0:
        import warnings
        warnings.warn("No dual vertices found. Call compute_vd(HC) first.")
        return None

    # Pad to 3D if needed (polyscope requires 3D)
    if points.shape[1] < 3:
        pad = np.zeros((points.shape[0], 3 - points.shape[1]))
        points = np.hstack([points, pad])

    ps_cloud = ps.register_point_cloud(name, points)

    if scalar_fields:
        from collections import defaultdict
        for field in scalar_fields:
            accum = defaultdict(list)
            for v in HC.V:
                val = getattr(v, field, 0.0)
                val = float(val) if np.ndim(val) == 0 else float(val[0])
                if hasattr(v, 'vd'):
                    for vd in v.vd:
                        accum[id(vd)].append(val)
            vals = [float(np.mean(accum.get(id(vd), [0.0])))
                    for vd in dual_pts]
            ps_cloud.add_scalar_quantity(field, np.array(vals))

    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ps.screenshot(save_path)

    return ps_cloud


# HIGH-LEVEL FLUID WRAPPERS

def _restore_snapshot(HC, snapshot: dict, fields: list[str]):
    """Write snapshot field data back onto HC vertices (in-place)."""
    for v in HC.V:
        key = tuple(float(x) for x in v.x_a)
        vdata = snapshot.get(key)
        if vdata is None:
            continue
        for f in fields:
            if f in vdata:
                val = vdata[f]
                if isinstance(val, np.ndarray):
                    setattr(v, f, val.copy())
                else:
                    setattr(v, f, val)


def plot_fluid(
    HC,
    bV=None,
    t: float = 0.00,
    show_mesh: bool = True,
    scalar_field: str = 'P',
    vector_field: str = 'u',
    scalar_label: str = 'Pressure [Pa]',
    vector_label: str = 'Velocity [m/s]',
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'fluid.png'),
    dpi: int = 150,
    **kwargs,
):
    """Plot a fluid-state snapshot: pressure + velocity with optional mesh overlay.

    Creates a multi-panel figure showing:

    - **Left**: scalar field (pressure by default) with colorbar.
    - **Right**: vector field (velocity by default) with colored arrows.
    - Optionally the primal (and dual, if computed) mesh edges are drawn
      underneath both panels.

    A timestamp is displayed as a figure suptitle.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with fields already set on vertices.
    bV : set or None
        Boundary vertices.
    t : float
        Simulation time to display in the title (default 0.00 s).
    show_mesh : bool
        If True, draw primal mesh edges on both panels (default True).
    scalar_field : str or None
        Vertex attribute for colormap panel (default ``'P'``).
        Set to ``None`` to skip the scalar panel entirely.
    vector_field : str or None
        Vertex attribute for quiver panel (default ``'u'``).
        Set to ``None`` to skip the vector panel entirely.
    scalar_label : str
        Colorbar / axis label for the scalar panel.
    vector_label : str
        Title annotation for the vector panel.
    save_path : str or None
        File path to save the figure.  ``None`` to skip.
        Default: ``results/fig/fluid.png``.
    dpi : int
        Resolution for the saved image (default 150).
    **kwargs
        Forwarded to the dimension-specific backends (``cmap``,
        ``vertex_size``, ``show_edges``, etc.).

    Returns
    -------
    fig, axes : Figure and array of Axes
    """
    import matplotlib.pyplot as plt

    dim = HC.dim

    # Determine how many panels we need
    panels = []
    if scalar_field is not None:
        panels.append(('scalar', scalar_field, scalar_label))
    if vector_field is not None:
        panels.append(('vector', vector_field, vector_label))
    n_panels = max(len(panels), 1)

    # Create figure
    if dim == 3:
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(7 * n_panels, 6),
                                 subplot_kw={'projection': '3d'})
    else:
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)

    fig.suptitle(f't = {t:.4f} s', fontsize=13, y=1.02)

    panel_kwargs = dict(kwargs)
    panel_kwargs['show_edges'] = show_mesh

    for ax_panel, (kind, field, label) in zip(axes, panels):
        if kind == 'scalar':
            plot_primal(HC, bV=bV, scalar_field=field, ax=ax_panel,
                        title=label, save_path=None, **dict(panel_kwargs))
        else:
            plot_primal(HC, bV=bV, vector_field=field, ax=ax_panel,
                        title=label, save_path=None, **dict(panel_kwargs))

    plt.tight_layout()

    if save_path is not None:
        _save_fig(fig, save_path, dpi=dpi)

    return fig, axes


def dynamic_plot_fluid(
    history,
    HC,
    bV=None,
    scalar_field: str = 'P',
    vector_field: str = 'u',
    scalar_label: str = 'Pressure [Pa]',
    vector_label: str = 'Velocity [m/s]',
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'fluid.mp4'),
    frame_dir: str = None,
    name: str = 'fluid',
    fps: int = 10,
    dpi: int = 150,
    interval: int = 100,
    writer: str = None,
    **kwargs,
):
    """Create a video of the simulation from a ``StateHistory``.

    For **matplotlib** (default): generates a ``FuncAnimation`` and saves
    to *save_path* (supports ``.mp4``, ``.gif``, etc.).

    Handles both Eulerian (fixed mesh) and Lagrangian (changing topology)
    simulations by plotting directly from snapshot data rather than
    restoring fields onto ``HC``.

    If *frame_dir* is provided, individual frame PNGs are also saved there
    (useful for polyscope workflows where static screenshots are compiled
    separately).

    Parameters
    ----------
    history : StateHistory
        Recorded simulation history containing snapshots.
    HC : Complex
        The simplicial complex (used only for ``dim``).
    bV : set or None
        Boundary vertices (unused in animation — kept for API compat).
    scalar_field : str or None
        Scalar field to animate (default ``'P'``).
    vector_field : str or None
        Vector field to animate (default ``'u'``).
    scalar_label : str
        Label for the scalar panel.
    vector_label : str
        Label for the vector panel.
    save_path : str or None
        Output video path.  Extension determines format (``.mp4``,
        ``.gif``).  ``None`` to return the animation without saving.
        Default: ``results/fig/fluid.mp4``.
    frame_dir : str or None
        If provided, each frame is also saved as a PNG in this directory
        (e.g. ``results/fig/ps_snap/``).  The files are named
        ``{name}_{index:06d}_t{time:.6f}.png``.
    name : str
        Base name prefix for frame files (default ``'fluid'``).
    fps : int
        Frames per second for the output video (default 10).
    dpi : int
        Resolution per frame (default 150).
    interval : int
        Milliseconds between frames for ``FuncAnimation`` (default 100).
    writer : str or None
        Matplotlib animation writer.  ``None`` auto-selects (``'ffmpeg'``
        for ``.mp4``, ``'pillow'`` for ``.gif``).
    **kwargs
        Extra options: ``cmap`` (str), ``vertex_size`` (float),
        ``scale`` (float for quiver), ``arrow_length`` (float for 3D).

    Returns
    -------
    anim : FuncAnimation
        Matplotlib animation object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    if history.n_snapshots == 0:
        return None

    dim = HC.dim
    cmap_name = kwargs.get('cmap', 'viridis')
    vertex_size = kwargs.get('vertex_size', 15 if dim >= 2 else 30)

    # Pre-compute global bounds from ALL snapshots for stable animation
    all_coords = []
    scalar_vals_global = []
    for _, snapshot, _ in history._snapshots:
        for key, vdata in snapshot.items():
            all_coords.append(key)
            if scalar_field is not None and scalar_field in vdata:
                val = vdata[scalar_field]
                scalar_vals_global.append(
                    float(val) if np.ndim(val) == 0 else float(val[0])
                )

    all_coords = np.array(all_coords)
    margin = 0.05
    xlim = (all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ylim = zlim = None
    if dim >= 2:
        ylim = (all_coords[:, 1].min() - margin,
                all_coords[:, 1].max() + margin)
    if dim >= 3:
        zlim = (all_coords[:, 2].min() - margin,
                all_coords[:, 2].max() + margin)

    # Global scalar range for consistent colorbar
    if scalar_vals_global:
        svmin = min(scalar_vals_global)
        svmax = max(scalar_vals_global)
        if svmin == svmax:
            svmin -= 0.5
            svmax += 0.5
    else:
        svmin, svmax = 0.0, 1.0

    # Create figure and panels
    panels = []
    if scalar_field is not None:
        panels.append(('scalar', scalar_field, scalar_label))
    if vector_field is not None:
        panels.append(('vector', vector_field, vector_label))
    n_panels = max(len(panels), 1)

    if dim == 3:
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(7 * n_panels, 6),
                                 subplot_kw={'projection': '3d'})
    else:
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)

    suptitle = fig.suptitle('', fontsize=13)

    # Create a fixed colorbar for the scalar panel (if any)
    sm = None
    cbar = None
    if scalar_field is not None:
        sm = mcm.ScalarMappable(
            cmap=cmap_name,
            norm=mcolors.Normalize(vmin=svmin, vmax=svmax),
        )
        sm.set_array([])
        scalar_ax_idx = next(
            i for i, (k, _, _) in enumerate(panels) if k == 'scalar'
        )
        cbar = fig.colorbar(sm, ax=axes[scalar_ax_idx], label=scalar_label)

    fig.tight_layout()

    # Optional frame directory
    if frame_dir is not None:
        os.makedirs(frame_dir, exist_ok=True)

    # Animation update: plot directly from snapshot data
    def update(frame_idx):
        t, snapshot, _ = history._snapshots[frame_idx]

        coords = np.array(list(snapshot.keys()))
        n_verts = len(coords)

        for ax in axes:
            ax.clear()

        suptitle.set_text(f't = {t:.4f} s')

        for ax_panel, (kind, field, label) in zip(axes, panels):
            if kind == 'scalar':
                # Extract scalar values
                vals = np.array([
                    float(snapshot[k].get(field, 0.0))
                    if np.ndim(snapshot[k].get(field, 0.0)) == 0
                    else float(snapshot[k].get(field, [0.0])[0])
                    for k in snapshot
                ])

                if dim == 1:
                    ax_panel.scatter(
                        coords[:, 0], vals, c=vals, cmap=cmap_name,
                        vmin=svmin, vmax=svmax, s=vertex_size,
                        zorder=3, edgecolors='k', linewidths=0.3,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_ylabel(field)
                    ax_panel.set_xlim(xlim)
                elif dim == 2:
                    ax_panel.scatter(
                        coords[:, 0], coords[:, 1], c=vals,
                        cmap=cmap_name, vmin=svmin, vmax=svmax,
                        s=vertex_size, zorder=3,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_ylabel('y')
                    ax_panel.set_aspect('equal')
                    ax_panel.set_xlim(xlim)
                    ax_panel.set_ylim(ylim)
                elif dim == 3:
                    ax_panel.scatter(
                        coords[:, 0], coords[:, 1], coords[:, 2],
                        c=vals, cmap=cmap_name, vmin=svmin, vmax=svmax,
                        s=vertex_size, alpha=0.8, zorder=3,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_ylabel('y')
                    ax_panel.set_zlabel('z')
                    ax_panel.set_xlim(xlim)
                    ax_panel.set_ylim(ylim)
                    ax_panel.set_zlim(zlim)

                ax_panel.set_title(label)

            else:  # vector
                # Extract velocity vectors
                zero_vec = np.zeros(dim)
                vecs = np.array([
                    np.asarray(snapshot[k].get(field, zero_vec))[:dim]
                    for k in snapshot
                ])

                magnitudes = np.linalg.norm(vecs, axis=1) if dim > 1 \
                    else np.abs(vecs[:, 0])
                max_mag = magnitudes.max()
                if max_mag > 0:
                    norm_mag = magnitudes / max_mag
                else:
                    norm_mag = magnitudes
                cmap_v = plt.colormaps.get_cmap('coolwarm')
                colors = cmap_v(norm_mag)

                if dim == 1:
                    ax_panel.quiver(
                        coords[:, 0], np.zeros(n_verts),
                        vecs[:, 0], np.zeros(n_verts),
                        color=colors, scale=kwargs.get('scale'),
                        zorder=5,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_yticks([])
                    ax_panel.set_xlim(xlim)
                elif dim == 2:
                    ax_panel.quiver(
                        coords[:, 0], coords[:, 1],
                        vecs[:, 0], vecs[:, 1],
                        color=colors, scale=kwargs.get('scale'),
                        zorder=5,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_ylabel('y')
                    ax_panel.set_aspect('equal')
                    ax_panel.set_xlim(xlim)
                    ax_panel.set_ylim(ylim)
                elif dim == 3:
                    ax_panel.quiver(
                        coords[:, 0], coords[:, 1], coords[:, 2],
                        vecs[:, 0], vecs[:, 1], vecs[:, 2],
                        colors=colors,
                        length=kwargs.get('arrow_length', 0.05),
                        normalize=True, alpha=0.7,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_ylabel('y')
                    ax_panel.set_zlabel('z')
                    ax_panel.set_xlim(xlim)
                    ax_panel.set_ylim(ylim)
                    ax_panel.set_zlim(zlim)

                ax_panel.set_title(label)

        # Save individual frame if frame_dir specified
        if frame_dir is not None:
            frame_path = os.path.join(
                frame_dir, f'{name}_{frame_idx:06d}_t{t:.6f}.png'
            )
            fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')

        return axes

    anim = FuncAnimation(
        fig, update,
        frames=history.n_snapshots,
        interval=interval,
        blit=False,
    )

    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Auto-select writer from extension
        if writer is None:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == '.gif':
                writer = 'pillow'
            else:
                writer = 'ffmpeg'
        anim.save(save_path, writer=writer, fps=fps, dpi=dpi)

    return anim


def dynamic_plot_fluid_polyscope(
    history,
    HC,
    scalar_fields: list[str] = None,
    vector_fields: list[str] = None,
    frame_dir: str = os.path.join(_DEFAULT_FIG_DIR, 'ps_snap'),
    name: str = 'fluid',
    video_path: str = None,
    fps: int = 10,
):
    """Save polyscope screenshots per snapshot, optionally compile to video.

    Since polyscope has no native video export, this function:
    1. Iterates over all snapshots in *history*.
    2. Restores vertex fields from each snapshot.
    3. Saves a polyscope screenshot per frame to *frame_dir*.
    4. Optionally compiles the frames into an MP4 using ``ffmpeg`` or
       ``pillow`` (for GIF).

    Parameters
    ----------
    history : StateHistory
        Recorded simulation history.
    HC : Complex
        Simplicial complex (fields will be overwritten per frame).
    scalar_fields : list of str or None
        Scalar field names (default ``['P']`` if None).
    vector_fields : list of str or None
        Vector field names (default ``['u']`` if None).
    frame_dir : str
        Output directory for per-frame PNGs.
        Default: ``results/fig/ps_snap/``.
    name : str
        Filename prefix for frames (default ``'fluid'``).
    video_path : str or None
        If provided, compile frames into a video at this path.
        Supports ``.mp4`` (requires ffmpeg) and ``.gif`` (uses pillow).
    fps : int
        Frames per second for the compiled video (default 10).

    Returns
    -------
    frame_paths : list of str
        Paths to all saved frame images.
    """
    from ddgclib.visualization.polyscope_3d import (
        _check_polyscope, register_point_cloud, update_frame,
    )
    ps = _check_polyscope()

    if scalar_fields is None:
        scalar_fields = ['P']
    if vector_fields is None:
        vector_fields = ['u']

    restore_fields = list(set(scalar_fields + vector_fields))
    dim = HC.dim
    os.makedirs(frame_dir, exist_ok=True)

    ps_cloud = register_point_cloud(HC, name='mesh', dim=dim)
    frame_paths = []

    for i, (t, snapshot, _) in enumerate(history._snapshots):
        _restore_snapshot(HC, snapshot, restore_fields)
        update_frame(HC, ps_cloud, scalar_fields=scalar_fields,
                     vector_fields=vector_fields, dim=dim)
        fpath = os.path.join(frame_dir, f'{name}_{i:06d}_t{t:.6f}.png')
        ps.screenshot(fpath)
        frame_paths.append(fpath)

    # Compile video from frames if requested
    if video_path is not None and frame_paths:
        _compile_video_from_frames(frame_paths, video_path, fps=fps)

    return frame_paths


def _compile_video_from_frames(frame_paths: list[str], output_path: str,
                               fps: int = 10):
    """Compile a list of PNG frame paths into a video.

    Tries ``ffmpeg`` first (for ``.mp4``), falls back to ``pillow``
    (for ``.gif``).
    """
    import subprocess
    ext = os.path.splitext(output_path)[1].lower()
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if ext in ('.mp4', '.avi', '.mov'):
        # Try ffmpeg
        list_path = output_path + '.frames.txt'
        with open(list_path, 'w') as f:
            for p in frame_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
                f.write(f"duration {1.0 / fps:.6f}\n")
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                 '-i', list_path, '-vf', f'fps={fps}',
                 '-pix_fmt', 'yuv420p', output_path],
                check=True, capture_output=True,
            )
        finally:
            if os.path.exists(list_path):
                os.remove(list_path)
    else:
        # Fallback: pillow for GIF
        from PIL import Image
        images = [Image.open(p) for p in frame_paths]
        if images:
            images[0].save(
                output_path, save_all=True,
                append_images=images[1:],
                duration=int(1000 / fps), loop=0,
            )
