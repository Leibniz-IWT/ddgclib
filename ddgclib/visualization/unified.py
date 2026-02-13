"""Unified dimension-dispatching wrappers for primal and dual mesh visualization.

Delegates base mesh rendering to ``hyperct._plotting.plot_complex`` and
``hyperct._plotting.animate_complex``, then overlays fluid-specific fields
(scalar colormap, vector quiver) on the returned matplotlib axes.

Color convention (from ``ddgclib._misc.coldict``):
- ``'db'`` (dark blue) for points and edges
- ``'lb'`` (light blue) for triangle faces
"""

import os

import numpy as np

from ddgclib._misc import coldict

_DEFAULT_FIG_DIR = os.path.join('results', 'fig')


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scalar / vector overlay on existing axes
# ---------------------------------------------------------------------------

def _overlay_scalar(HC, ax, field, dim, cmap_name='viridis', vertex_size=15):
    """Overlay a scalar field as colored scatter on *ax*."""
    import matplotlib.pyplot as plt

    coords, vals = _extract_scalar(HC, field)
    if dim == 1:
        sc = ax.scatter(coords[:, 0], np.zeros(len(coords)),
                        c=vals, cmap=cmap_name, s=vertex_size,
                        zorder=3, edgecolors='k', linewidths=0.3)
    elif dim == 2:
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=vals, cmap=cmap_name, s=vertex_size, zorder=3)
    elif dim == 3:
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=vals, cmap=cmap_name, s=vertex_size,
                        alpha=0.8, zorder=3)
    else:
        return
    ax.get_figure().colorbar(sc, ax=ax, label=field, shrink=0.6 if dim == 3 else 1.0)


def _overlay_vector(HC, ax, field, dim, **kwargs):
    """Overlay a vector field as quiver arrows on *ax*."""
    import matplotlib.pyplot as plt

    pos, vecs = _extract_vector(HC, field, dim)
    magnitudes = np.linalg.norm(vecs, axis=1) if dim > 1 else np.abs(vecs[:, 0])
    max_mag = magnitudes.max()
    norm_mag = magnitudes / max_mag if max_mag > 0 else magnitudes
    cmap = plt.colormaps.get_cmap('coolwarm')
    colors = cmap(norm_mag)

    if dim == 1:
        ax.quiver(pos[:, 0], np.zeros(len(pos)),
                  vecs[:, 0], np.zeros(len(pos)),
                  color=colors, scale=kwargs.get('scale'), zorder=5)
    elif dim == 2:
        ax.quiver(pos[:, 0], pos[:, 1], vecs[:, 0], vecs[:, 1],
                  color=colors, scale=kwargs.get('scale'), zorder=5)
    elif dim == 3:
        ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                  vecs[:, 0], vecs[:, 1], vecs[:, 2],
                  colors=colors, length=kwargs.get('arrow_length', 0.05),
                  normalize=True, alpha=0.7)


# ---------------------------------------------------------------------------
# Dual field interpolation helpers (2D / 3D)
# ---------------------------------------------------------------------------

def _interpolate_scalar_to_dual_2d(HC, field, dual_pts_set):
    """Interpolate a scalar field from primal to dual vertices by averaging."""
    from collections import defaultdict
    accum = defaultdict(list)
    for v in HC.V:
        val = getattr(v, field)
        val = float(val) if np.ndim(val) == 0 else float(val[0])
        if hasattr(v, 'vd'):
            for vd in v.vd:
                if vd.x in dual_pts_set:
                    accum[vd.x].append(val)
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
    norm_mag = magnitudes / magnitudes.max() if magnitudes.max() > 0 else magnitudes
    cmap = plt.colormaps.get_cmap('coolwarm')
    colors = cmap(norm_mag)
    ax.quiver(px, py, ux, uy, color=colors, zorder=5)


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------

def _save_fig(fig, save_path, dpi=150):
    """Save *fig* to *save_path*, creating parent directories as needed."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Triangle face rendering helper
# ---------------------------------------------------------------------------

def _render_faces(HC, ax, dim, face_alpha, face_color=None, edge_color=None):
    """Render filled triangle faces on *ax* using ``HC.vertex_face_mesh()``."""
    if face_alpha <= 0 or dim < 2:
        return
    fc = tuple(face_color) if face_color is not None else tuple(coldict['lb'])
    ec = tuple(edge_color) if edge_color is not None else tuple(coldict['db'])
    try:
        HC.vertex_face_mesh()
        verts_fm = np.array(HC.vertices_fm)
        simps = np.array(HC.simplices_fm_i)
        if len(simps) == 0:
            return
        triangles = verts_fm[simps]
        if dim == 2:
            from matplotlib.collections import PolyCollection
            pc = PolyCollection(
                triangles, facecolors=[fc], edgecolors=[ec],
                alpha=face_alpha, linewidths=0.4, zorder=0,
            )
            ax.add_collection(pc)
        elif dim == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            pc = Poly3DCollection(
                triangles, facecolors=fc, edgecolors=ec,
                alpha=face_alpha, linewidths=0.4,
            )
            ax.add_collection3d(pc)
    except Exception:
        pass  # vertex_face_mesh may not be available for all complexes


def _render_faces_from_coords(coords, ax, dim, face_alpha):
    """Render filled faces from raw coordinates via Delaunay triangulation."""
    if face_alpha <= 0 or dim < 2 or len(coords) < 3:
        return
    fc = tuple(coldict['lb'])
    ec = tuple(coldict['db'])
    try:
        from scipy.spatial import Delaunay
        pts = coords[:, :dim]
        tri = Delaunay(pts)
        triangles = pts[tri.simplices]
        if dim == 2:
            from matplotlib.collections import PolyCollection
            pc = PolyCollection(
                triangles, facecolors=[fc], edgecolors=[ec],
                alpha=face_alpha, linewidths=0.4, zorder=0,
            )
            ax.add_collection(pc)
        elif dim == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            pc = Poly3DCollection(
                triangles, facecolors=fc, edgecolors=ec,
                alpha=face_alpha, linewidths=0.4,
            )
            ax.add_collection3d(pc)
    except Exception:
        pass


def _render_edges_from_coords(coords, ax, dim, edge_color=None, linewidth=0.8):
    """Draw mesh edges from raw coordinates, mimicking ``plot_complex`` style.

    For 1D: lines between sorted adjacent vertices.
    For 2D/3D: Delaunay edges rendered as a ``LineCollection``.
    """
    ec = tuple(edge_color) if edge_color is not None else tuple(coldict['db'])
    if len(coords) < 2:
        return
    try:
        if dim == 1:
            xs = np.sort(coords[:, 0])
            ax.plot(xs, np.zeros_like(xs), color=ec, linewidth=linewidth,
                    zorder=1)
            ax.scatter(xs, np.zeros_like(xs), color=ec, s=15, zorder=2)
        elif dim == 2:
            from scipy.spatial import Delaunay
            from matplotlib.collections import LineCollection
            pts = coords[:, :2]
            tri = Delaunay(pts)
            edges = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    edge = tuple(sorted((simplex[i], simplex[(i + 1) % len(simplex)])))
                    edges.add(edge)
            segments = [[pts[e[0]], pts[e[1]]] for e in edges]
            lc = LineCollection(segments, colors=[ec], linewidths=linewidth,
                                zorder=1)
            ax.add_collection(lc)
            ax.scatter(pts[:, 0], pts[:, 1], color=ec, s=7, zorder=2)
        elif dim == 3:
            from scipy.spatial import Delaunay
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            pts = coords[:, :3]
            tri = Delaunay(pts[:, :3])
            edges = set()
            for simplex in tri.simplices:
                n = len(simplex)
                for i in range(n):
                    for j in range(i + 1, n):
                        edges.add(tuple(sorted((simplex[i], simplex[j]))))
            segments = [[pts[e[0]], pts[e[1]]] for e in edges]
            lc = Line3DCollection(segments, colors=[ec], linewidths=linewidth,
                                  zorder=1)
            ax.add_collection3d(lc)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=ec, s=7,
                       zorder=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# plot_primal — delegates to hyperct.plot_complex
# ---------------------------------------------------------------------------

def plot_primal(
    HC,
    bV=None,
    scalar_field: str = None,
    vector_field: str = None,
    ax=None,
    face_alpha: float = 0.6,
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'primal.png'),
    dpi: int = 150,
    **kwargs,
):
    """Plot the primal mesh with optional scalar/vector field overlays.

    Uses ``hyperct._plotting.plot_complex`` for the base wireframe
    (vertices + edges), renders filled triangle faces with configurable
    transparency, then overlays fluid fields on the returned axes.

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
    face_alpha : float
        Transparency for filled triangle faces (default 0.6).
        Set to 0 to hide faces.
    save_path : str or None
        File path to save the figure.  Set to ``None`` to skip saving.
    dpi : int
        Resolution for the saved image (default 150).
    **kwargs
        Extra options: ``cmap``, ``vertex_size``, ``scale``,
        ``arrow_length``, ``title``, ``show_edges``, ``pointsize``.

    Returns
    -------
    fig, ax
    """
    from hyperct._plotting import plot_complex
    import matplotlib.pyplot as plt

    cmap_name = kwargs.pop('cmap', 'viridis')
    title = kwargs.pop('title', None)
    vertex_size = kwargs.pop('vertex_size', 15)
    pointsize = kwargs.pop('pointsize', 7)
    show_edges = kwargs.pop('show_edges', True)

    # Build figure/axes if needed
    dim = HC.dim
    if ax is None:
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Delegate base wireframe (vertices + edges) to hyperct
    plot_complex(
        HC,
        show=False,
        save_fig=False,
        directed=show_edges,
        complex_plot=True,
        contour_plot=False,
        surface_plot=False,
        surface_field_plot=False,
        minimiser_points=False,
        point_color=tuple(coldict['db']),
        line_color=tuple(coldict['db']),
        complex_color_f=tuple(coldict['lb']),
        complex_color_e=tuple(coldict['db']),
        pointsize=pointsize,
        fig_complex=fig,
        ax_complex=ax,
    )
    # plot_complex stores result in HC.ax_complex
    ax = getattr(HC, 'ax_complex', ax)
    fig = ax.get_figure()

    # Render filled triangle faces (behind wireframe)
    _render_faces(HC, ax, dim, face_alpha)

    # Overlay scalar field
    if scalar_field is not None:
        _overlay_scalar(HC, ax, scalar_field, dim,
                        cmap_name=cmap_name, vertex_size=vertex_size)

    # Overlay vector field
    if vector_field is not None:
        _overlay_vector(HC, ax, vector_field, dim, **kwargs)

    if title:
        ax.set_title(title)

    if save_path is not None:
        _save_fig(fig, save_path, dpi=dpi)

    return fig, ax


# ---------------------------------------------------------------------------
# plot_dual — delegates to hyperct.ddg.plot_dual
# ---------------------------------------------------------------------------

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

    Uses ``hyperct.ddg.plot_dual`` functions for the base dual mesh,
    then overlays fluid fields.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with ``dim`` attribute.
    bV : set or None
        Boundary vertices.
    scalar_field : str or None
        Vertex attribute name for scalar coloring.
    vector_field : str or None
        Vertex attribute name for arrow overlay.
    vertex : vertex object or None
        **Required for 3D**: the specific primal vertex whose dual cell
        to visualize.
    ax : matplotlib Axes or None
        If None a new figure is created.
    save_path : str or None
        File path to save the figure.
    dpi : int
        Resolution for the saved image (default 150).
    **kwargs
        Extra options: ``title``, ``cmap``, ``dual_color``,
        ``show_primal``.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    dim = HC.dim
    title = kwargs.pop('title', None)
    cmap_name = kwargs.pop('cmap', 'viridis')
    dual_color = kwargs.pop('dual_color', 'darkorange')
    show_primal = kwargs.pop('show_primal', True)
    vertex_size = kwargs.pop('vertex_size', 10)

    if dim == 1:
        fig, ax = _plot_dual_1d(HC, bV, scalar_field, vector_field, ax,
                                title=title, cmap=cmap_name,
                                dual_color=dual_color,
                                show_primal=show_primal, **kwargs)
    elif dim == 2:
        # Use hyperct's dual plotting for 2D
        from hyperct.ddg.plot_dual import plot_dual_mesh_2D as _hc_dual_2d

        has_vd = hasattr(next(iter(HC.V)), 'vd')
        if not has_vd:
            import warnings
            warnings.warn(
                "Dual vertices not computed. Call compute_vd(HC) first. "
                "Showing primal mesh only."
            )
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
            if title:
                ax.set_title(title)
            if save_path is not None:
                _save_fig(fig, save_path, dpi=dpi)
            return fig, ax

        fig, ax = _hc_dual_2d(HC, ax=ax, show=False)

        # Overlay scalar field on dual vertices if requested
        if scalar_field is not None or vector_field is not None:
            dual_pts_set = set()
            for v in HC.V:
                if hasattr(v, 'vd'):
                    for vd in v.vd:
                        dual_pts_set.add(vd.x)
            if scalar_field is not None and dual_pts_set:
                dx = [p[0] for p in dual_pts_set]
                dy = [p[1] for p in dual_pts_set]
                dual_vals = _interpolate_scalar_to_dual_2d(HC, scalar_field,
                                                           dual_pts_set)
                sc = ax.scatter(dx, dy, c=dual_vals, cmap=cmap_name,
                                s=vertex_size, zorder=4, edgecolors='k',
                                linewidths=0.3)
                fig.colorbar(sc, ax=ax, label=f'{scalar_field} (dual)')
            if vector_field is not None and dual_pts_set:
                _overlay_vector_dual_2d(HC, vector_field, dual_pts_set, ax)

        if title:
            ax.set_title(title)

    elif dim == 3:
        if vertex is None:
            raise ValueError(
                "For 3D dual plots, a specific *vertex* must be provided "
                "(pass vertex=some_vertex)."
            )
        from hyperct.ddg.plot_dual import plot_dual_mesh_3D as _hc_dual_3d
        fig, ax = _hc_dual_3d(HC, ax=ax, show=False)

        if title:
            ax.set_title(title)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    if save_path is not None:
        _save_fig(fig, save_path, dpi=dpi)

    return fig, ax


# ---------------------------------------------------------------------------
# 1D dual helper (lightweight, no hyperct equivalent needed)
# ---------------------------------------------------------------------------

def _plot_dual_1d(HC, bV, scalar_field, vector_field, ax, **kwargs):
    """Plot 1D dual mesh (edge midpoints above the primal line)."""
    import matplotlib.pyplot as plt

    title = kwargs.pop('title', None)
    cmap_name = kwargs.pop('cmap', 'viridis')
    dual_color = kwargs.pop('dual_color', 'darkorange')
    show_primal = kwargs.pop('show_primal', True)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if show_primal:
        x_arr = sorted([float(v.x_a[0]) for v in HC.V])
        ax.scatter(x_arr, [0]*len(x_arr), c=[coldict['db']], s=20, zorder=3,
                   label='primal')

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
        coords_vals = sorted([(float(v.x_a[0]), getattr(v, scalar_field))
                              for v in HC.V])
        for mid in dual_x:
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


# ---------------------------------------------------------------------------
# Polyscope unified wrappers
# ---------------------------------------------------------------------------

def plot_primal_polyscope(
    HC,
    scalar_fields: list[str] = None,
    vector_fields: list[str] = None,
    name: str = 'primal',
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'primal_ps.png'),
):
    """Register and display the primal mesh in polyscope with field overlays."""
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
    """Register dual vertices in polyscope as a point cloud."""
    from ddgclib.visualization.polyscope_3d import _check_polyscope
    ps = _check_polyscope()

    dim = HC.dim
    if dim == 3 and vertex is not None:
        dual_pts = list(vertex.vd)
        points = np.array([vd.x[:dim] for vd in dual_pts], dtype=np.float64)
    else:
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


# ---------------------------------------------------------------------------
# HIGH-LEVEL FLUID WRAPPERS
# ---------------------------------------------------------------------------

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
    face_alpha: float = 0.6,
    scalar_field: str = 'P',
    vector_field: str = 'u',
    scalar_label: str = 'Pressure [Pa]',
    vector_label: str = 'Velocity [m/s]',
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'fluid.png'),
    dpi: int = 150,
    **kwargs,
):
    """Plot a fluid-state snapshot: pressure + velocity with optional mesh overlay.

    Creates a multi-panel figure showing scalar and vector fields,
    using ``plot_primal`` (which delegates to hyperct) for each panel.

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
    face_alpha : float
        Transparency for filled triangle faces (default 0.6).
        Set to 0 to hide faces.
    scalar_field : str or None
        Vertex attribute for colormap panel (default ``'P'``).
    vector_field : str or None
        Vertex attribute for quiver panel (default ``'u'``).
    scalar_label, vector_label : str
        Labels for each panel.
    save_path : str or None
        File path to save the figure.
    dpi : int
        Resolution for the saved image (default 150).

    Returns
    -------
    fig, axes : Figure and array of Axes
    """
    import matplotlib.pyplot as plt

    dim = HC.dim
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

    fig.suptitle(f't = {t:.4f} s', fontsize=13, y=1.02)

    panel_kwargs = dict(kwargs)
    panel_kwargs['show_edges'] = show_mesh

    for ax_panel, (kind, field, label) in zip(axes, panels):
        if kind == 'scalar':
            plot_primal(HC, bV=bV, scalar_field=field, ax=ax_panel,
                        face_alpha=face_alpha, title=label, save_path=None,
                        **dict(panel_kwargs))
        else:
            plot_primal(HC, bV=bV, vector_field=field, ax=ax_panel,
                        face_alpha=face_alpha, title=label, save_path=None,
                        **dict(panel_kwargs))

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
    face_alpha: float = 0.6,
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

    For Eulerian (fixed mesh) simulations, uses ``animate_complex`` from
    hyperct with an ``update_state`` callback that restores vertex fields
    from each snapshot. For Lagrangian (changing topology) simulations,
    plots directly from snapshot coordinate data.

    Parameters
    ----------
    history : StateHistory
        Recorded simulation history containing snapshots.
    HC : Complex
        The simplicial complex.
    bV : set or None
        Boundary vertices.
    scalar_field : str or None
        Scalar field to animate (default ``'P'``).
    vector_field : str or None
        Vector field to animate (default ``'u'``).
    scalar_label, vector_label : str
        Labels for panels.
    face_alpha : float
        Transparency for filled triangle faces (default 0.6).
        Set to 0 to hide faces.
    save_path : str or None
        Output video path (``.mp4``, ``.gif``).
    frame_dir : str or None
        Save individual frame PNGs here.
    name : str
        Base name for frame files.
    fps, dpi, interval : int
        Animation parameters.
    writer : str or None
        Matplotlib animation writer.

    Returns
    -------
    anim : FuncAnimation
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

    sm = None
    if scalar_field is not None:
        sm = mcm.ScalarMappable(
            cmap=cmap_name,
            norm=mcolors.Normalize(vmin=svmin, vmax=svmax),
        )
        sm.set_array([])
        scalar_ax_idx = next(
            i for i, (k, _, _) in enumerate(panels) if k == 'scalar'
        )
        fig.colorbar(sm, ax=axes[scalar_ax_idx], label=scalar_label)

    fig.tight_layout()

    if frame_dir is not None:
        os.makedirs(frame_dir, exist_ok=True)

    _db = coldict['db']

    def update(frame_idx):
        t, snapshot, _ = history._snapshots[frame_idx]
        coords = np.array(list(snapshot.keys()))
        n_verts = len(coords)

        for ax in axes:
            ax.clear()

        suptitle.set_text(f't = {t:.4f} s')

        _ec = tuple(coldict['db'])

        for ax_panel, (kind, field, label) in zip(axes, panels):
            # Render filled triangle faces (2D/3D only)
            _render_faces_from_coords(coords, ax_panel, dim, face_alpha)

            if kind == 'scalar':
                vals = np.array([
                    float(snapshot[k].get(field, 0.0))
                    if np.ndim(snapshot[k].get(field, 0.0)) == 0
                    else float(snapshot[k].get(field, [0.0])[0])
                    for k in snapshot
                ])
                if dim == 1:
                    # Edges: line through sorted (x, value) pairs
                    order = np.argsort(coords[:, 0])
                    ax_panel.plot(
                        coords[order, 0], vals[order],
                        color=_ec, linewidth=0.8, zorder=1,
                    )
                    ax_panel.scatter(
                        coords[:, 0], vals, c=vals, cmap=cmap_name,
                        vmin=svmin, vmax=svmax, s=vertex_size,
                        zorder=3, edgecolors='k', linewidths=0.3,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_ylabel(field)
                    ax_panel.set_xlim(xlim)
                elif dim == 2:
                    _render_edges_from_coords(coords, ax_panel, dim)
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
                    _render_edges_from_coords(coords, ax_panel, dim)
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
                zero_vec = np.zeros(dim)
                vecs = np.array([
                    np.asarray(snapshot[k].get(field, zero_vec))[:dim]
                    for k in snapshot
                ])
                magnitudes = np.linalg.norm(vecs, axis=1) if dim > 1 \
                    else np.abs(vecs[:, 0])
                max_mag = magnitudes.max()
                norm_mag = magnitudes / max_mag if max_mag > 0 else magnitudes
                cmap_v = plt.colormaps.get_cmap('coolwarm')
                colors = cmap_v(norm_mag)

                if dim == 1:
                    # Edges: line at y=0 through sorted vertices
                    order = np.argsort(coords[:, 0])
                    ax_panel.plot(
                        coords[order, 0], np.zeros(n_verts),
                        color=_ec, linewidth=0.8, zorder=1,
                    )
                    ax_panel.scatter(
                        coords[:, 0], np.zeros(n_verts),
                        color=_ec, s=7, zorder=2,
                    )
                    ax_panel.quiver(
                        coords[:, 0], np.zeros(n_verts),
                        vecs[:, 0], np.zeros(n_verts),
                        color=colors, scale=kwargs.get('scale'), zorder=5,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_yticks([])
                    ax_panel.set_xlim(xlim)
                elif dim == 2:
                    _render_edges_from_coords(coords, ax_panel, dim)
                    ax_panel.quiver(
                        coords[:, 0], coords[:, 1],
                        vecs[:, 0], vecs[:, 1],
                        color=colors, scale=kwargs.get('scale'), zorder=5,
                    )
                    ax_panel.set_xlabel('x')
                    ax_panel.set_ylabel('y')
                    ax_panel.set_aspect('equal')
                    ax_panel.set_xlim(xlim)
                    ax_panel.set_ylim(ylim)
                elif dim == 3:
                    _render_edges_from_coords(coords, ax_panel, dim)
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
        if writer is None:
            ext = os.path.splitext(save_path)[1].lower()
            writer = 'pillow' if ext == '.gif' else 'ffmpeg'
        anim.save(save_path, writer=writer, fps=fps, dpi=dpi)

    return anim


def plot_fluid_ps(
    HC,
    bV=None,
    t: float = 0.0,
    scalar_fields: list[str] = None,
    vector_fields: list[str] = None,
    name: str = 'fluid',
    save_path: str = os.path.join(_DEFAULT_FIG_DIR, 'fluid_ps.png'),
):
    """Plot a fluid-state snapshot using polyscope (surface mesh with fields).

    Registers the mesh as a polyscope surface mesh (triangles) rather than
    a point cloud, then adds scalar and vector field quantities.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with fields already set on vertices.
    bV : set or None
        Boundary vertices (not used by polyscope, kept for API symmetry).
    t : float
        Simulation time (used in window title).
    scalar_fields : list of str or None
        Scalar fields to display (default ``['P']``).
    vector_fields : list of str or None
        Vector fields to display (default ``['u']``).
    name : str
        Polyscope structure name.
    save_path : str or None
        Screenshot path. Set to None to skip saving.

    Returns
    -------
    ps_mesh
        Polyscope SurfaceMesh object.
    """
    from ddgclib.visualization.polyscope_3d import (
        _check_polyscope, register_surface_mesh, update_surface_frame,
    )
    ps = _check_polyscope()

    if scalar_fields is None:
        scalar_fields = ['P']
    if vector_fields is None:
        vector_fields = ['u']

    dim = HC.dim
    ps_mesh = register_surface_mesh(HC, name=name, dim=dim)

    # Add field quantities
    for field in scalar_fields:
        vals = []
        for v in HC.V:
            val = getattr(v, field, 0.0)
            vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))
        ps_mesh.add_scalar_quantity(field, np.array(vals), enabled=True)

    for field in vector_fields:
        vecs = []
        for v in HC.V:
            val = getattr(v, field, np.zeros(dim))
            vec = np.asarray(val[:dim], dtype=np.float64)
            if len(vec) < 3:
                vec = np.concatenate([vec, np.zeros(3 - len(vec))])
            vecs.append(vec)
        ps_mesh.add_vector_quantity(field, np.array(vecs), enabled=True)

    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ps.screenshot(save_path)

    return ps_mesh


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

    Uses ``register_surface_mesh`` to display triangle faces (not just
    point clouds), re-registering each frame for Lagrangian meshes
    with changing topology.
    """
    from ddgclib.visualization.polyscope_3d import (
        _check_polyscope, register_surface_mesh, update_surface_frame,
    )
    ps = _check_polyscope()

    if scalar_fields is None:
        scalar_fields = ['P']
    if vector_fields is None:
        vector_fields = ['u']

    restore_fields = list(set(scalar_fields + vector_fields))
    dim = HC.dim
    os.makedirs(frame_dir, exist_ok=True)

    ps_mesh = register_surface_mesh(HC, name='mesh', dim=dim)
    frame_paths = []

    for i, (t, snapshot, _) in enumerate(history._snapshots):
        _restore_snapshot(HC, snapshot, restore_fields)
        ps_mesh = update_surface_frame(
            HC, ps_mesh, scalar_fields=scalar_fields,
            vector_fields=vector_fields, dim=dim, name='mesh',
        )
        fpath = os.path.join(frame_dir, f'{name}_{i:06d}_t{t:.6f}.png')
        ps.screenshot(fpath)
        frame_paths.append(fpath)

    if video_path is not None and frame_paths:
        _compile_video_from_frames(frame_paths, video_path, fps=fps)

    return frame_paths


def _compile_video_from_frames(frame_paths: list[str], output_path: str,
                               fps: int = 10):
    """Compile a list of PNG frame paths into a video."""
    import subprocess
    ext = os.path.splitext(output_path)[1].lower()
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if ext in ('.mp4', '.avi', '.mov'):
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
        from PIL import Image
        images = [Image.open(p) for p in frame_paths]
        if images:
            images[0].save(
                output_path, save_all=True,
                append_images=images[1:],
                duration=int(1000 / fps), loop=0,
            )
