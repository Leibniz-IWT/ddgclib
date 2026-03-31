"""Unified dimension-dispatching wrappers for primal and dual mesh visualization.

# it is in this path: from ddgclib.visualization.unified

Delegates base mesh rendering to ``hyperct._plotting.plot_complex`` and
``hyperct._plotting.animate_complex``, then overlays fluid-specific fields
(scalar colormap, vector quiver) on the returned matplotlib axes.

Color convention (from ``ddgclib._misc.coldict``):
- ``'db'`` (dark blue) for points and edges
- ``'lb'`` (light blue) for triangle faces

New helpers added vs original
------------------------------
``_rebuild_nn_from_delaunay(HC)``
    Computes a Delaunay triangulation of the current vertex positions,
    wires the resulting adjacency into every ``v.nn`` set, and returns
    ``(tri, simplex_verts)`` where ``simplex_verts`` is a list of
    ``(va, vb, vc)`` vertex-object triples.  This is the ONLY place a
    new Delaunay is computed.  Call it when the vertex count changes
    (injection / deletion events); between events the returned
    ``simplex_verts`` list keeps connectivity stable while positions drift.

``_render_faces_from_simplex_verts(simplex_verts, ax, face_alpha)``
    Renders filled triangle faces from a cached ``simplex_verts`` list.
    Because the list holds vertex *references*, it reads ``v.x_a`` at draw
    time — positions are always current without re-triangulating.

``_render_edges_from_simplex_verts(simplex_verts, ax, edge_color, linewidth)``
    Renders primal edges from the same cached ``simplex_verts`` list.
    Same live-position guarantee as above.

``_render_dual_from_vd(HC, ax, clip_box, color, lw, alpha)``
    Renders the BARYCENTRIC dual mesh stored in ``v.vd`` / ``HC.Vd``
    (populated by ``hyperct.ddg.compute_vd(HC, method='barycentric')``).
    Uses the same edge-drawing logic as ``hyperct.ddg.plot_dual.
    plot_dual_mesh_2D`` — shared dual vertices between primal neighbours
    connected by lines — but batched into a ``LineCollection`` and
    optionally clipped to a bounding box.  No ``scipy.spatial.Voronoi``
    is used; all geometry comes from the barycentric dual already
    embedded in the complex.
"""

import os

import numpy as np
from scipy.spatial import Delaunay, QhullError
from matplotlib.collections import LineCollection, PolyCollection

from ddgclib._misc import coldict

_DEFAULT_FIG_DIR = os.path.join('results', 'fig')


# ---------------------------------------------------------------------------
# Internal Delaunay helper — ONE place where QHull is called
# ---------------------------------------------------------------------------

def _compute_delaunay(coords):
    """Return a ``scipy.spatial.Delaunay`` for *coords* (2-D or 3-D).

    Near-coincident points are deduplicated before the QHull call.
    Returns ``None`` when the point set is too small or degenerate.
    """
    if coords is None or len(coords) < 3:
        return None
    _, unique_idx = np.unique(np.round(coords, 12), axis=0, return_index=True)
    pts = coords[unique_idx]
    if len(pts) < 3:
        return None
    try:
        return Delaunay(pts)
    except QhullError:
        return None


# ---------------------------------------------------------------------------
# Neighbour-rebuild + simplex-vert cache
# ---------------------------------------------------------------------------

def _rebuild_nn_from_delaunay(HC):
    """Rebuild ``v.nn`` for every vertex in *HC* from a fresh Delaunay.

    This is the single entry point for computing mesh connectivity when
    the complex was assembled by vertex insertion (not ``.triangulate()``)
    or when the vertex set changes due to injection / deletion.

    Algorithm
    ---------
    1. Collect all vertex positions and deduplicate (same tolerance as
       ``_compute_delaunay`` so the index mapping is exact).
    2. Run ``scipy.spatial.Delaunay`` once on the deduplicated set.
    3. Clear all ``v.nn`` sets and repopulate from simplex edges.
    4. Build and return a ``simplex_verts`` list — one ``(va, vb, vc)``
       tuple per Delaunay triangle.  The tuples hold vertex *references*
       so downstream renderers read current ``v.x_a`` without needing to
       re-triangulate.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.  ``HC.V`` must be iterable.

    Returns
    -------
    tri : ``scipy.spatial.Delaunay`` or ``None``
    simplex_verts : list of ``(va, vb, vc)`` vertex-object tuples
        Empty list when triangulation fails.
    """
    verts = list(HC.V)
    if len(verts) < 3:
        return None, []

    # -- clear existing adjacency ------------------------------------------
    for v in verts:
        v.nn = set()

    coords = np.array([v.x_a[:2] for v in verts])

    # deduplicate (same rounding as _compute_delaunay uses internally)
    _, unique_idx = np.unique(np.round(coords, 12), axis=0, return_index=True)
    unique_verts = [verts[i] for i in unique_idx]
    unique_coords = coords[unique_idx]

    if len(unique_coords) < 3:
        return None, []

    try:
        tri = Delaunay(unique_coords)
    except QhullError:
        return None, []

    # -- wire adjacency and collect simplex_verts --------------------------
    simplex_verts = []
    for simplex in tri.simplices:
        va = unique_verts[simplex[0]]
        vb = unique_verts[simplex[1]]
        vc = unique_verts[simplex[2]]
        simplex_verts.append((va, vb, vc))
        # populate v.nn
        for i in range(3):
            for j in range(i + 1, 3):
                u = unique_verts[simplex[i]]
                w = unique_verts[simplex[j]]
                u.nn.add(w)
                w.nn.add(u)

    return tri, simplex_verts


# ---------------------------------------------------------------------------
# Simplex-vert-based renderers (stable connectivity, live positions)
# ---------------------------------------------------------------------------

def _render_faces_from_simplex_verts(simplex_verts, ax, face_alpha,
                                      face_color=None):
    """Render filled primal triangle faces from a cached ``simplex_verts`` list.

    Unlike ``_render_faces_from_coords``, this function does NOT call
    ``scipy.spatial.Delaunay``.  It uses the connectivity stored in
    ``simplex_verts`` (vertex-object tuples returned by
    ``_rebuild_nn_from_delaunay``) and reads each vertex's current
    ``v.x_a`` at draw time, so positions are always up to date even when
    vertices have moved since the triangulation was last computed.

    Parameters
    ----------
    simplex_verts : list of ``(va, vb, vc)``
        Cached simplex list from ``_rebuild_nn_from_delaunay``.
    ax : matplotlib Axes
    face_alpha : float
        Transparency of filled faces.  Pass 0 to skip.
    face_color : color spec or None
        Fill colour; defaults to ``coldict['lb']`` (light blue).
    """
    if face_alpha <= 0 or not simplex_verts:
        return
    fc = tuple(face_color) if face_color is not None else tuple(coldict['lb'])
    triangles = [
        np.array([va.x_a[:2], vb.x_a[:2], vc.x_a[:2]])
        for va, vb, vc in simplex_verts
    ]
    pc = PolyCollection(
        triangles,
        facecolors=[fc],
        edgecolors='none',   # edges drawn separately by _render_edges_*
        alpha=face_alpha,
        zorder=0,
    )
    ax.add_collection(pc)


def _render_edges_from_simplex_verts(simplex_verts, ax,
                                      edge_color=None, linewidth=0.8):
    """Render primal mesh edges from a cached ``simplex_verts`` list.

    Deduplicates edges by vertex ``id`` pair so each edge is drawn once.
    Reads current ``v.x_a`` for positions.

    Parameters
    ----------
    simplex_verts : list of ``(va, vb, vc)``
    ax : matplotlib Axes
    edge_color : color spec or None
        Line colour; defaults to ``coldict['db']`` (dark blue).
    linewidth : float
    """
    if not simplex_verts:
        return
    ec = tuple(edge_color) if edge_color is not None else tuple(coldict['db'])
    seen = set()
    segments = []
    for va, vb, vc in simplex_verts:
        for u, w in ((va, vb), (vb, vc), (va, vc)):
            key = (min(id(u), id(w)), max(id(u), id(w)))
            if key not in seen:
                seen.add(key)
                segments.append([u.x_a[:2], w.x_a[:2]])
    if segments:
        lc = LineCollection(segments, colors=[ec], linewidths=linewidth, zorder=1)
        ax.add_collection(lc)


# ---------------------------------------------------------------------------
# Barycentric dual renderer — uses v.vd populated by compute_vd
# ---------------------------------------------------------------------------

def _render_dual_from_vd(HC, ax, clip_box=None,
                          color='tab:orange', lw=1.0, alpha=0.7):
    """Render the barycentric dual mesh stored in ``v.vd`` / ``HC.Vd``.

    Requires ``hyperct.ddg.compute_vd(HC, method='barycentric')`` to have
    been called so that every primal vertex ``v`` has a populated ``v.vd``
    set and ``HC.Vd`` holds the dual vertex cache.

    The rendering logic mirrors ``hyperct.ddg.plot_dual.plot_dual_mesh_2D``:
    for each primal edge ``(v, v2)`` the two dual vertices shared by both
    primal endpoints are connected, forming the Voronoi-like boundary
    between their dual cells.  A ``LineCollection`` is used for efficiency
    instead of one ``ax.plot`` call per edge.

    Only dual edges whose *both* endpoints lie inside *clip_box* are
    drawn; edges straddling the domain boundary (common near injected /
    freshly-deleted vertices) are suppressed to keep the plot clean.

    Parameters
    ----------
    HC : Complex
        Must have ``HC.Vd`` and ``v.vd`` populated (call ``compute_vd``
        with ``method='barycentric'`` first).
    ax : matplotlib Axes
    clip_box : (x_min, x_max, y_min, y_max) or None
        Bounding box for clipping dual edges and vertices.
        ``None`` disables clipping.
    color : color spec
        Line and marker colour for dual elements.
    lw : float
        Line width for dual edges.
    alpha : float
        Transparency.
    """
    if not hasattr(HC, 'Vd') or not hasattr(HC, 'V'):
        return

    if clip_box is not None:
        x_min, x_max, y_min, y_max = clip_box
        def _inside(p):
            return x_min <= p[0] <= x_max and y_min <= p[1] <= y_max
    else:
        def _inside(p):  # noqa: F811
            return True

    # -- dual edges (shared dual vertices between primal neighbours) -------
    # Same logic as plot_dual_mesh_2D; use vd.x (position tuple) for
    # deduplication so each dual edge is drawn exactly once.
    segments = []
    plotted_dual_edges = set()

    for v in HC.V:
        if not hasattr(v, 'vd'):
            continue
        for v2 in v.nn:
            if not hasattr(v2, 'vd'):
                continue
            shared = v.vd.intersection(v2.vd)
            if len(shared) < 2:
                continue
            shared_list = list(shared)
            for i in range(len(shared_list)):
                for j in range(i + 1, len(shared_list)):
                    vd1, vd2 = shared_list[i], shared_list[j]
                    de = tuple(sorted([vd1.x, vd2.x]))
                    if de in plotted_dual_edges:
                        continue
                    p0 = np.asarray(vd1.x_a[:2])
                    p1 = np.asarray(vd2.x_a[:2])
                    if _inside(p0) and _inside(p1):
                        plotted_dual_edges.add(de)
                        segments.append([p0, p1])

    if segments:
        lc = LineCollection(segments, colors=color, linewidths=lw,
                            alpha=alpha, zorder=2)
        ax.add_collection(lc)

    # -- dual vertices (barycentric positions = triangle centroids) --------
    vd_pts = []
    for vd in HC.Vd:
        p = np.asarray(vd.x_a[:2])
        if _inside(p):
            vd_pts.append(p)
    if vd_pts:
        vd_arr = np.array(vd_pts)
        ax.scatter(vd_arr[:, 0], vd_arr[:, 1],
                   c=color, s=9, marker='D', zorder=3, alpha=alpha)


# ---------------------------------------------------------------------------
# Coordinate-based renderers (kept for backward-compat / dynamic_plot_fluid)
# ---------------------------------------------------------------------------

def _render_faces_from_coords(coords, ax, dim, face_alpha, tri=None):
    """Render filled faces from raw coordinates via Delaunay.

    Accepts an optional pre-computed *tri* (``scipy.spatial.Delaunay``)
    to skip recomputation.  When *tri* is supplied its ``tri.points`` array
    is used for triangle vertices — pass a tri whose points match *coords*
    or supply ``tri=None`` to always recompute.
    """
    if face_alpha <= 0 or dim < 2 or len(coords) < 3:
        return
    fc = tuple(coldict['lb'])
    ec = tuple(coldict['db'])
    if tri is None:
        tri = _compute_delaunay(coords)
    if tri is None:
        return
    tri_pts = tri.points if tri.points.shape[0] <= len(coords) else coords[:, :dim]
    triangles = tri_pts[tri.simplices]
    try:
        if dim == 2:
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


def _render_edges_from_coords(coords, ax, dim, edge_color=None, linewidth=0.8,
                               tri=None):
    """Draw mesh edges from raw coordinates.

    Accepts an optional pre-computed *tri* to skip recomputation.
    """
    ec = tuple(edge_color) if edge_color is not None else tuple(coldict['db'])
    if len(coords) < 2:
        return
    try:
        if dim == 1:
            xs = np.sort(coords[:, 0])
            ax.plot(xs, np.zeros_like(xs), color=ec, linewidth=linewidth, zorder=1)
            ax.scatter(xs, np.zeros_like(xs), color=ec, s=15, zorder=2)
        elif dim == 2:
            if tri is None:
                tri = _compute_delaunay(coords)
            if tri is None:
                return
            pts = tri.points
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                    edges.add(edge)
            segments = [[pts[e[0]], pts[e[1]]] for e in edges]
            lc = LineCollection(segments, colors=[ec], linewidths=linewidth, zorder=1)
            ax.add_collection(lc)
            ax.scatter(pts[:, 0], pts[:, 1], color=ec, s=7, zorder=2)
        elif dim == 3:
            if tri is None:
                tri = _compute_delaunay(coords)
            if tri is None:
                return
            pts = tri.points
            edges = set()
            for simplex in tri.simplices:
                n = len(simplex)
                for i in range(n):
                    for j in range(i + 1, n):
                        edges.add(tuple(sorted((simplex[i], simplex[j]))))
            segments = [[pts[e[0]], pts[e[1]]] for e in edges]
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            lc = Line3DCollection(segments, colors=[ec], linewidths=linewidth, zorder=1)
            ax.add_collection3d(lc)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=ec, s=7, zorder=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _extract_scalar(HC, field: str):
    coords, vals = [], []
    for v in HC.V:
        coords.append(v.x_a.copy())
        val = getattr(v, field)
        vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))
    return np.array(coords), np.array(vals)


def _extract_vector(HC, field: str, dim: int):
    coords, vecs = [], []
    for v in HC.V:
        coords.append(v.x_a[:dim].copy())
        vec = getattr(v, field)
        vecs.append(np.asarray(vec[:dim], dtype=np.float64))
    return np.array(coords), np.array(vecs)


# ---------------------------------------------------------------------------
# Scalar / vector overlay on existing axes
# ---------------------------------------------------------------------------

def _overlay_scalar(HC, ax, field, dim, cmap_name='viridis', vertex_size=15):
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
                        c=vals, cmap=cmap_name, s=vertex_size, alpha=0.8, zorder=3)
    else:
        return
    ax.get_figure().colorbar(sc, ax=ax, label=field,
                             shrink=0.6 if dim == 3 else 1.0)


def _overlay_vector(HC, ax, field, dim, **kwargs):
    import matplotlib.pyplot as plt
    pos, vecs = _extract_vector(HC, field, dim)
    magnitudes = np.linalg.norm(vecs, axis=1) if dim > 1 else np.abs(vecs[:, 0])
    max_mag = magnitudes.max()
    norm_mag = magnitudes / max_mag if max_mag > 0 else magnitudes
    colors = plt.colormaps.get_cmap('coolwarm')(norm_mag)
    if dim == 1:
        ax.quiver(pos[:, 0], np.zeros(len(pos)), vecs[:, 0], np.zeros(len(pos)),
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
# Dual field interpolation helpers
# ---------------------------------------------------------------------------

def _interpolate_scalar_to_dual_2d(HC, field, dual_pts_set):
    from collections import defaultdict
    accum = defaultdict(list)
    for v in HC.V:
        val = getattr(v, field)
        val = float(val) if np.ndim(val) == 0 else float(val[0])
        if hasattr(v, 'vd'):
            for vd in v.vd:
                if vd.x in dual_pts_set:
                    accum[vd.x].append(val)
    return np.array([float(np.mean(accum.get(pt, [0.0]))) for pt in dual_pts_set])


def _overlay_vector_dual_2d(HC, field, dual_pts_set, ax):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    accum = defaultdict(list)
    dim = 2
    for v in HC.V:
        vec = np.asarray(getattr(v, field)[:dim], dtype=np.float64)
        if hasattr(v, 'vd'):
            for vd in v.vd:
                if vd.x in dual_pts_set:
                    accum[vd.x].append(vec)
    px, py, ux, uy = [], [], [], []
    for pt in dual_pts_set:
        avg = np.mean(accum.get(pt, [np.zeros(dim)]), axis=0)
        px.append(pt[0]); py.append(pt[1])
        ux.append(avg[0]); uy.append(avg[1])
    px, py = np.array(px), np.array(py)
    ux, uy = np.array(ux), np.array(uy)
    magnitudes = np.sqrt(ux**2 + uy**2)
    norm_mag = magnitudes / magnitudes.max() if magnitudes.max() > 0 else magnitudes
    colors = plt.colormaps.get_cmap('coolwarm')(norm_mag)
    ax.quiver(px, py, ux, uy, color=colors, zorder=5)


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------

def _save_fig(fig, save_path, dpi=150):
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Face rendering using HC.vertex_face_mesh (for triangulated complexes)
# ---------------------------------------------------------------------------

def _render_faces(HC, ax, dim, face_alpha, face_color=None, edge_color=None):
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
            pc = PolyCollection(triangles, facecolors=[fc], edgecolors=[ec],
                                alpha=face_alpha, linewidths=0.4, zorder=0)
            ax.add_collection(pc)
        elif dim == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            pc = Poly3DCollection(triangles, facecolors=fc, edgecolors=ec,
                                  alpha=face_alpha, linewidths=0.4)
            ax.add_collection3d(pc)
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
    """Plot the primal mesh with optional scalar/vector field overlays."""
    from hyperct._plotting import plot_complex
    import matplotlib.pyplot as plt

    cmap_name = kwargs.pop('cmap', 'viridis')
    title = kwargs.pop('title', None)
    vertex_size = kwargs.pop('vertex_size', 15)
    pointsize = kwargs.pop('pointsize', 7)
    show_edges = kwargs.pop('show_edges', True)

    dim = HC.dim
    if ax is None:
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    plot_complex(
        HC, show=False, save_fig=False, directed=show_edges,
        complex_plot=True, contour_plot=False, surface_plot=False,
        surface_field_plot=False, minimiser_points=False,
        point_color=tuple(coldict['db']), line_color=tuple(coldict['db']),
        complex_color_f=tuple(coldict['lb']), complex_color_e=tuple(coldict['db']),
        pointsize=pointsize, fig_complex=fig, ax_complex=ax,
    )
    ax = getattr(HC, 'ax_complex', ax)
    fig = ax.get_figure()
    _render_faces(HC, ax, dim, face_alpha)
    if scalar_field is not None:
        _overlay_scalar(HC, ax, scalar_field, dim,
                        cmap_name=cmap_name, vertex_size=vertex_size)
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
    xlim: tuple = None,
    ylim: tuple = None,
    **kwargs,
):
    """Plot the dual mesh with optional scalar/vector field overlays."""
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
        from hyperct.ddg.plot_dual import plot_dual_mesh_2D as _hc_dual_2d
        has_vd = hasattr(next(iter(HC.V)), 'vd')
        if not has_vd:
            import warnings
            warnings.warn("Dual vertices not computed. Call compute_vd(HC) first.")
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
        if scalar_field is not None or vector_field is not None:
            dual_pts_set = set()
            for v in HC.V:
                if hasattr(v, 'vd'):
                    for vd in v.vd:
                        dual_pts_set.add(vd.x)
            if scalar_field is not None and dual_pts_set:
                dx = [p[0] for p in dual_pts_set]
                dy = [p[1] for p in dual_pts_set]
                dual_vals = _interpolate_scalar_to_dual_2d(HC, scalar_field, dual_pts_set)
                sc = ax.scatter(dx, dy, c=dual_vals, cmap=cmap_name,
                                s=vertex_size, zorder=4, edgecolors='k', linewidths=0.3)
                fig.colorbar(sc, ax=ax, label=f'{scalar_field} (dual)')
            if vector_field is not None and dual_pts_set:
                _overlay_vector_dual_2d(HC, vector_field, dual_pts_set, ax)
        if title:
            ax.set_title(title)
    elif dim == 3:
        if vertex is None:
            raise ValueError("For 3D dual plots, pass vertex=some_vertex.")
        from hyperct.ddg.plot_dual import plot_dual_mesh_3D as _hc_dual_3d
        fig, ax = _hc_dual_3d(HC, ax=ax, show=False)
        if title:
            ax.set_title(title)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if save_path is not None:
        _save_fig(fig, save_path, dpi=dpi)
    return fig, ax


def _plot_dual_1d(HC, bV, scalar_field, vector_field, ax, **kwargs):
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
        ax.scatter(x_arr, [0]*len(x_arr), c=[coldict['db']], s=20, zorder=3, label='primal')
    midpoints, seen = set(), set()
    for v in HC.V:
        for nb in v.nn:
            key = frozenset((v.x, nb.x))
            if key not in seen:
                seen.add(key)
                midpoints.add(0.5 * (float(v.x_a[0]) + float(nb.x_a[0])))
    dual_x = sorted(midpoints)
    ax.scatter(dual_x, [0.05]*len(dual_x), c=dual_color, s=20, zorder=4,
               marker='D', label='dual')
    ax.legend(fontsize=8)
    ax.set_xlabel('x')
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return fig, ax


# ---------------------------------------------------------------------------
# Polyscope wrappers (unchanged)
# ---------------------------------------------------------------------------

def plot_primal_polyscope(HC, scalar_fields=None, vector_fields=None,
                          name='primal',
                          save_path=os.path.join(_DEFAULT_FIG_DIR, 'primal_ps.png')):
    from ddgclib.visualization.polyscope_3d import (
        _check_polyscope, register_point_cloud, update_frame)
    ps = _check_polyscope()
    ps_cloud = register_point_cloud(HC, name=name, dim=HC.dim)
    update_frame(HC, ps_cloud, scalar_fields=scalar_fields,
                 vector_fields=vector_fields, dim=HC.dim)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        ps.screenshot(save_path)
    return ps_cloud


def plot_dual_polyscope(HC, vertex=None, scalar_fields=None, name='dual',
                        save_path=os.path.join(_DEFAULT_FIG_DIR, 'dual_ps.png')):
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
        import warnings; warnings.warn("No dual vertices found.")
        return None
    if points.shape[1] < 3:
        points = np.hstack([points, np.zeros((len(points), 3 - points.shape[1]))])
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
            vals = [float(np.mean(accum.get(id(vd), [0.0]))) for vd in dual_pts]
            ps_cloud.add_scalar_quantity(field, np.array(vals))
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        ps.screenshot(save_path)
    return ps_cloud


# ---------------------------------------------------------------------------
# HIGH-LEVEL FLUID WRAPPERS
# ---------------------------------------------------------------------------

def _restore_snapshot(HC, snapshot: dict, fields: list):
    for v in HC.V:
        key = tuple(float(x) for x in v.x_a)
        vdata = snapshot.get(key)
        if vdata is None:
            continue
        for f in fields:
            if f in vdata:
                val = vdata[f]
                setattr(v, f, val.copy() if isinstance(val, np.ndarray) else val)


def plot_fluid(HC, bV=None, t=0.0, show_mesh=True, face_alpha=0.6,
               scalar_field='p', vector_field='u',
               scalar_label='Pressure [Pa]', vector_label='Velocity [m/s]',
               save_path=os.path.join(_DEFAULT_FIG_DIR, 'fluid.png'),
               dpi=150, xlim=None, ylim=None, **kwargs):
    import matplotlib.pyplot as plt
    dim = HC.dim
    panels = []
    if scalar_field is not None:
        panels.append(('scalar', scalar_field, scalar_label))
    if vector_field is not None:
        panels.append(('vector', vector_field, vector_label))
    n_panels = max(len(panels), 1)
    kw = {'projection': '3d'} if dim == 3 else {}
    fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 5), subplot_kw=kw)
    axes = [axes] if n_panels == 1 else list(axes)
    fig.suptitle(f't = {t:.4f} s', fontsize=13, y=1.02)
    pk = dict(kwargs); pk['show_edges'] = show_mesh
    for ax_panel, (kind, field, label) in zip(axes, panels):
        f = scalar_field if kind == 'scalar' else vector_field
        kf = dict(pk)
        if kind == 'scalar':
            plot_primal(HC, bV=bV, scalar_field=f, ax=ax_panel,
                        face_alpha=face_alpha, title=label, save_path=None, **kf)
        else:
            plot_primal(HC, bV=bV, vector_field=f, ax=ax_panel,
                        face_alpha=face_alpha, title=label, save_path=None, **kf)
    for ax_panel in axes:
        if xlim: ax_panel.set_xlim(xlim)
        if ylim: ax_panel.set_ylim(ylim)
    plt.tight_layout()
    if save_path is not None:
        _save_fig(fig, save_path, dpi=dpi)
    return fig, axes


def dynamic_plot_fluid(history, HC, bV=None, scalar_field='p', vector_field='u',
                       scalar_label='Pressure [Pa]', vector_label='Velocity [m/s]',
                       face_alpha=0.6,
                       save_path=os.path.join(_DEFAULT_FIG_DIR, 'fluid.mp4'),
                       frame_dir=None, name='fluid', fps=10, dpi=150,
                       interval=100, writer=None, xlim=None, ylim=None, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.cm as mcm, matplotlib.colors as mcolors

    if history.n_snapshots == 0:
        return None
    dim = HC.dim
    cmap_name = kwargs.get('cmap', 'viridis')
    vertex_size = kwargs.get('vertex_size', 15 if dim >= 2 else 30)

    all_coords, scalar_vals_global = [], []
    for _, snapshot, _ in history._snapshots:
        for key, vdata in snapshot.items():
            all_coords.append(key)
            if scalar_field and scalar_field in vdata:
                val = vdata[scalar_field]
                scalar_vals_global.append(float(val) if np.ndim(val) == 0 else float(val[0]))
    all_coords = np.array(all_coords)
    margin = 0.05
    _xlim = xlim or (all_coords[:,0].min()-margin, all_coords[:,0].max()+margin)
    _ylim = zlim = None
    if dim >= 2:
        _ylim = ylim or (all_coords[:,1].min()-margin, all_coords[:,1].max()+margin)
    if dim >= 3:
        zlim = (all_coords[:,2].min()-margin, all_coords[:,2].max()+margin)
    svmin = min(scalar_vals_global) if scalar_vals_global else 0.0
    svmax = max(scalar_vals_global) if scalar_vals_global else 1.0
    if svmin == svmax: svmin -= 0.5; svmax += 0.5

    panels = []
    if scalar_field: panels.append(('scalar', scalar_field, scalar_label))
    if vector_field: panels.append(('vector', vector_field, vector_label))
    n_panels = max(len(panels), 1)
    kw = {'projection': '3d'} if dim == 3 else {}
    fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 5), subplot_kw=kw)
    axes = [axes] if n_panels == 1 else list(axes)
    suptitle = fig.suptitle('', fontsize=13)
    if scalar_field:
        sm = mcm.ScalarMappable(cmap=cmap_name,
                                norm=mcolors.Normalize(vmin=svmin, vmax=svmax))
        sm.set_array([])
        sidx = next(i for i,(k,_,_) in enumerate(panels) if k=='scalar')
        fig.colorbar(sm, ax=axes[sidx], label=scalar_label)
    fig.tight_layout()
    if frame_dir: os.makedirs(frame_dir, exist_ok=True)

    _anim_cache = {'n': -1, 'tri': None}

    def _get_tri(coords):
        n = len(coords)
        if n != _anim_cache['n']:
            _anim_cache['tri'] = _compute_delaunay(coords)
            _anim_cache['n'] = n
        return _anim_cache['tri']

    def update(frame_idx):
        t, snapshot, _ = history._snapshots[frame_idx]
        coords = np.array(list(snapshot.keys()))
        for ax in axes: ax.clear()
        suptitle.set_text(f't = {t:.4f} s')
        tri = _get_tri(coords)
        for ax_panel, (kind, field, label) in zip(axes, panels):
            _render_faces_from_coords(coords, ax_panel, dim, face_alpha, tri=tri)
            if kind == 'scalar':
                vals = np.array([
                    float(snapshot[k].get(field, 0.0)) if np.ndim(snapshot[k].get(field, 0.0))==0
                    else float(snapshot[k].get(field, [0.0])[0])
                    for k in snapshot])
                if dim == 2:
                    _render_edges_from_coords(coords, ax_panel, dim, tri=tri)
                    ax_panel.scatter(coords[:,0], coords[:,1], c=vals, cmap=cmap_name,
                                     vmin=svmin, vmax=svmax, s=vertex_size, zorder=3)
                    ax_panel.set_aspect('equal'); ax_panel.set_xlim(_xlim); ax_panel.set_ylim(_ylim)
            else:
                zero_vec = np.zeros(dim)
                vecs = np.array([np.asarray(snapshot[k].get(field, zero_vec))[:dim] for k in snapshot])
                magnitudes = np.linalg.norm(vecs, axis=1) if dim > 1 else np.abs(vecs[:,0])
                mm = magnitudes.max()
                norm_mag = magnitudes/mm if mm > 0 else magnitudes
                colors = plt.colormaps.get_cmap('coolwarm')(norm_mag)
                if dim == 2:
                    _render_edges_from_coords(coords, ax_panel, dim, tri=tri)
                    ax_panel.quiver(coords[:,0], coords[:,1], vecs[:,0], vecs[:,1],
                                    color=colors, scale=kwargs.get('scale'), zorder=5)
                    ax_panel.set_aspect('equal'); ax_panel.set_xlim(_xlim); ax_panel.set_ylim(_ylim)
            ax_panel.set_title(label)
        if frame_dir:
            fig.savefig(os.path.join(frame_dir, f'{name}_{frame_idx:06d}_t{t:.6f}.png'),
                        dpi=dpi, bbox_inches='tight')
        return axes

    anim = FuncAnimation(fig, update, frames=history.n_snapshots,
                         interval=interval, blit=False)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        w = writer or ('pillow' if ext == '.gif' else 'ffmpeg')
        anim.save(save_path, writer=w, fps=fps, dpi=dpi)
    return anim


def plot_fluid_ps(HC, bV=None, t=0.0, scalar_fields=None, vector_fields=None,
                  name='fluid',
                  save_path=os.path.join(_DEFAULT_FIG_DIR, 'fluid_ps.png')):
    from ddgclib.visualization.polyscope_3d import (
        _check_polyscope, register_surface_mesh, update_surface_frame)
    ps = _check_polyscope()
    scalar_fields = scalar_fields or ['p']
    vector_fields = vector_fields or ['u']
    dim = HC.dim
    ps_mesh = register_surface_mesh(HC, name=name, dim=dim)
    for field in scalar_fields:
        vals = [float(getattr(v, field, 0.0)) if np.ndim(getattr(v, field, 0.0))==0
                else float(getattr(v, field, [0.0])[0]) for v in HC.V]
        ps_mesh.add_scalar_quantity(field, np.array(vals), enabled=True)
    for field in vector_fields:
        vecs = []
        for v in HC.V:
            vec = np.asarray(getattr(v, field, np.zeros(dim))[:dim], dtype=np.float64)
            if len(vec) < 3: vec = np.concatenate([vec, np.zeros(3-len(vec))])
            vecs.append(vec)
        ps_mesh.add_vector_quantity(field, np.array(vecs), enabled=True)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        ps.screenshot(save_path)
    return ps_mesh


def dynamic_plot_fluid_polyscope(history, HC, scalar_fields=None, vector_fields=None,
                                  frame_dir=os.path.join(_DEFAULT_FIG_DIR, 'ps_snap'),
                                  name='fluid', video_path=None, fps=10):
    from ddgclib.visualization.polyscope_3d import (
        _check_polyscope, register_surface_mesh, update_surface_frame)
    ps = _check_polyscope()
    scalar_fields = scalar_fields or ['p']
    vector_fields = vector_fields or ['u']
    restore_fields = list(set(scalar_fields + vector_fields))
    dim = HC.dim
    os.makedirs(frame_dir, exist_ok=True)
    ps_mesh = register_surface_mesh(HC, name='mesh', dim=dim)
    frame_paths = []
    for i, (t, snapshot, _) in enumerate(history._snapshots):
        _restore_snapshot(HC, snapshot, restore_fields)
        ps_mesh = update_surface_frame(HC, ps_mesh, scalar_fields=scalar_fields,
                                       vector_fields=vector_fields, dim=dim, name='mesh')
        fpath = os.path.join(frame_dir, f'{name}_{i:06d}_t{t:.6f}.png')
        ps.screenshot(fpath)
        frame_paths.append(fpath)
    if video_path and frame_paths:
        _compile_video_from_frames(frame_paths, video_path, fps=fps)
    return frame_paths


def _compile_video_from_frames(frame_paths, output_path, fps=10):
    import subprocess
    ext = os.path.splitext(output_path)[1].lower()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if ext in ('.mp4', '.avi', '.mov'):
        list_path = output_path + '.frames.txt'
        with open(list_path, 'w') as f:
            for p in frame_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
                f.write(f"duration {1.0/fps:.6f}\n")
        try:
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                            '-i', list_path, '-vf', f'fps={fps}',
                            '-pix_fmt', 'yuv420p', output_path],
                           check=True, capture_output=True)
        finally:
            if os.path.exists(list_path):
                os.remove(list_path)
    else:
        from PIL import Image
        images = [Image.open(p) for p in frame_paths]
        if images:
            images[0].save(output_path, save_all=True, append_images=images[1:],
                           duration=int(1000/fps), loop=0)