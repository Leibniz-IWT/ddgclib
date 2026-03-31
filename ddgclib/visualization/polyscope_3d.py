"""Polyscope-based 3D visualization (optional dependency).

All functions gracefully fail with an ImportError message if polyscope
is not installed.

Usage
-----
::

    from ddgclib.visualization.polyscope_3d import register_point_cloud, update_frame
    ps_cloud = register_point_cloud(HC, name='mesh')
    update_frame(HC, ps_cloud, scalar_fields=['p'], vector_fields=['u'])

Interactive viewer with frame slider::

    from ddgclib.visualization.polyscope_3d import interactive_viewer

    # Register polyscope structures first, then:
    interactive_viewer(
        n_frames=len(frames),
        update_fn=lambda idx: cloud.update_point_positions(positions[idx]),
        info_fn=lambda idx: imgui.Text(f"Frame {idx}"),
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def _check_polyscope():
    try:
        import polyscope
        return polyscope
    except ImportError:
        raise ImportError(
            "polyscope is required for 3D visualization. "
            "Install it with: pip install polyscope"
        )


def register_surface_mesh(HC, name: str = 'mesh', dim: int = 3):
    """Register the mesh as a polyscope surface mesh with triangles.

    Uses ``HC.vertex_face_mesh()`` to get triangle connectivity.
    Falls back to Delaunay triangulation if not available, and
    finally to point cloud if triangulation fails.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    name : str
        Surface mesh name in polyscope.
    dim : int
        Spatial dimension.

    Returns
    -------
    ps_mesh
        Polyscope SurfaceMesh object (or PointCloud as fallback).
    """
    ps = _check_polyscope()
    ps.init()

    points = np.array([v.x_a[:dim] for v in HC.V], dtype=np.float64)
    # Pad to 3D for polyscope
    if points.shape[1] < 3:
        pad = np.zeros((points.shape[0], 3 - points.shape[1]))
        points = np.hstack([points, pad])

    # Try HC.vertex_face_mesh() first
    try:
        HC.vertex_face_mesh()
        simps = np.array(HC.simplices_fm_i)
        if len(simps) > 0:
            verts_fm = np.array(HC.vertices_fm, dtype=np.float64)
            if verts_fm.shape[1] < 3:
                pad = np.zeros((verts_fm.shape[0], 3 - verts_fm.shape[1]))
                verts_fm = np.hstack([verts_fm, pad])
            return ps.register_surface_mesh(name, verts_fm, simps)
    except Exception:
        pass

    # Fallback: Delaunay triangulation
    try:
        from scipy.spatial import Delaunay
        pts_2d = np.array([v.x_a[:min(dim, 2)] for v in HC.V], dtype=np.float64)
        tri = Delaunay(pts_2d)
        return ps.register_surface_mesh(name, points, tri.simplices)
    except Exception:
        pass

    # Final fallback: point cloud
    return ps.register_point_cloud(name, points)


def register_point_cloud(HC, name: str = 'mesh', dim: int = 3):
    """Register the mesh vertices as a polyscope point cloud.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    name : str
        Point cloud name in polyscope.
    dim : int
        Spatial dimension.

    Returns
    -------
    ps_cloud
        Polyscope PointCloud object.
    """
    ps = _check_polyscope()
    ps.init()

    points = np.array([v.x_a[:dim] for v in HC.V], dtype=np.float64)
    ps_cloud = ps.register_point_cloud(name, points)
    return ps_cloud


def update_frame(
    HC,
    ps_cloud,
    scalar_fields: list[str] = None,
    vector_fields: list[str] = None,
    dim: int = 3,
):
    """Update polyscope point cloud with current field data.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    ps_cloud : polyscope PointCloud
        Previously registered point cloud.
    scalar_fields : list of str or None
        Scalar fields to add/update (e.g. ['p', 'm']).
    vector_fields : list of str or None
        Vector fields to add/update (e.g. ['u']).
    dim : int
        Spatial dimension.
    """
    _check_polyscope()

    # Update positions
    points = np.array([v.x_a[:dim] for v in HC.V], dtype=np.float64)
    ps_cloud.update_point_positions(points)

    if scalar_fields:
        for field in scalar_fields:
            vals = []
            for v in HC.V:
                val = getattr(v, field, 0.0)
                vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))
            ps_cloud.add_scalar_quantity(field, np.array(vals))

    if vector_fields:
        for field in vector_fields:
            vecs = []
            for v in HC.V:
                val = getattr(v, field, np.zeros(dim))
                vecs.append(np.asarray(val[:dim], dtype=np.float64))
            ps_cloud.add_vector_quantity(field, np.array(vecs))


def update_surface_frame(
    HC,
    ps_mesh,
    scalar_fields: list[str] = None,
    vector_fields: list[str] = None,
    dim: int = 3,
    name: str = 'mesh',
):
    """Re-register surface mesh with updated positions and fields.

    Polyscope surface meshes don't support in-place vertex updates,
    so we re-register the mesh each frame.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    ps_mesh : polyscope SurfaceMesh
        Previously registered surface mesh.
    scalar_fields : list of str or None
        Scalar fields to add/update (e.g. ['p', 'm']).
    vector_fields : list of str or None
        Vector fields to add/update (e.g. ['u']).
    dim : int
        Spatial dimension.
    name : str
        Mesh name in polyscope (must match registered name).

    Returns
    -------
    ps_mesh
        Updated polyscope SurfaceMesh object.
    """
    ps = _check_polyscope()

    # Re-register the mesh with current geometry
    ps_mesh = register_surface_mesh(HC, name=name, dim=dim)

    if scalar_fields:
        for field in scalar_fields:
            vals = []
            for v in HC.V:
                val = getattr(v, field, 0.0)
                vals.append(float(val) if np.ndim(val) == 0 else float(val[0]))
            ps_mesh.add_scalar_quantity(field, np.array(vals))

    if vector_fields:
        for field in vector_fields:
            vecs = []
            for v in HC.V:
                val = getattr(v, field, np.zeros(dim))
                vec = np.asarray(val[:dim], dtype=np.float64)
                if len(vec) < 3:
                    vec = np.concatenate([vec, np.zeros(3 - len(vec))])
                vecs.append(vec)
            ps_mesh.add_vector_quantity(field, np.array(vecs))

    return ps_mesh


def interactive_viewer(
    n_frames: int,
    update_fn: Callable[[int], None],
    info_fn: Callable[[int], None] | None = None,
    screenshot_dir: Path | str | None = None,
    init: bool = True,
):
    """Launch an interactive polyscope viewer with frame slider and playback.

    Register all polyscope structures **before** calling this function,
    then provide callbacks to update them per frame.

    Parameters
    ----------
    n_frames : int
        Total number of frames.
    update_fn : callable(idx: int)
        Called when the current frame changes.  Should update all
        registered polyscope structures for frame *idx*.
    info_fn : callable(idx: int) | None
        Called every UI tick to display ``imgui.Text(...)`` info.
        Receives ``polyscope.imgui`` as the first implicit import
        within the callback scope.
    screenshot_dir : Path or str or None
        If set, saves a PNG screenshot for every visited frame.
    init : bool
        Whether to call ``polyscope.init()`` (set False if you already
        initialised polyscope yourself).

    Example
    -------
    ::

        import polyscope as ps
        import numpy as np

        ps.init()
        cloud = ps.register_point_cloud("pts", positions[0])

        def update(idx):
            cloud.update_point_positions(positions[idx])

        def info(idx):
            import polyscope.imgui as imgui
            imgui.Text(f"t = {times[idx]:.3f} s")

        interactive_viewer(len(positions), update, info, init=False)
    """
    ps = _check_polyscope()
    if init:
        ps.init()

    if screenshot_dir is not None:
        screenshot_dir = Path(screenshot_dir)
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    state = {"idx": 0, "playing": False, "speed": 1}

    def _callback():
        import polyscope.imgui as imgui

        changed, state["idx"] = imgui.SliderInt(
            "Frame", state["idx"], 0, n_frames - 1,
        )
        _, state["playing"] = imgui.Checkbox("Play", state["playing"])

        imgui.SameLine()
        _, state["speed"] = imgui.SliderInt(
            "Speed", state["speed"], 1, 20,
        )

        if state["playing"]:
            state["idx"] = (state["idx"] + state["speed"]) % n_frames
            changed = True

        if changed:
            update_fn(state["idx"])
            if screenshot_dir is not None:
                ps.screenshot(
                    str(screenshot_dir / f"frame_{state['idx']:05d}.png")
                )

        imgui.Separator()
        if info_fn is not None:
            info_fn(state["idx"])

    update_fn(0)
    ps.set_user_callback(_callback)
    ps.show()


def interactive_history_viewer(
    history,
    HC,
    scalar_fields: list[str] = None,
    vector_fields: list[str] = None,
    name: str = 'fluid',
    screenshot_dir: Path | str | None = None,
    init: bool = True,
):
    """Launch an interactive polyscope viewer with a timeline slider for StateHistory.

    Loads each snapshot from *history* into *HC* on demand.  For Lagrangian
    meshes (vertex count / positions change between snapshots), the point
    cloud is re-registered each frame.

    Parameters
    ----------
    history : StateHistory
        Recorded simulation history containing snapshots.
    HC : Complex
        Simplicial complex (used to obtain dim; vertices are overwritten).
    scalar_fields : list of str or None
        Scalar fields to display (default ``['p']``).
    vector_fields : list of str or None
        Vector fields to display (default ``['u']``).
    name : str
        Polyscope structure name prefix.
    screenshot_dir : Path or str or None
        If set, saves a PNG screenshot for every visited frame.
    init : bool
        Whether to call ``polyscope.init()``.

    Example
    -------
    ::

        from ddgclib.visualization.polyscope_3d import interactive_history_viewer
        interactive_history_viewer(history, HC, scalar_fields=['p'],
                                   vector_fields=['u'])
    """
    ps = _check_polyscope()
    if init:
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("z_up")

    if scalar_fields is None:
        scalar_fields = ['p']
    if vector_fields is None:
        vector_fields = ['u']

    dim = HC.dim
    n_frames = history.n_snapshots
    if n_frames == 0:
        print("No snapshots in history — nothing to display.")
        return

    # Extract times for display
    times = [snap[0] for snap in history._snapshots]

    # State dict for imgui callback
    cloud_ref = {"cloud": None}

    def _load_frame(idx):
        """Restore snapshot idx onto HC and update polyscope."""
        t, snapshot, _ = history._snapshots[idx]

        # Build arrays directly from snapshot data (works for Lagrangian
        # meshes where HC vertex set may differ between frames).
        keys = list(snapshot.keys())
        n_pts = len(keys)
        if n_pts == 0:
            return

        positions = np.array(keys, dtype=np.float64)
        # Pad to 3D for polyscope
        if positions.shape[1] < 3:
            pad = np.zeros((n_pts, 3 - positions.shape[1]))
            positions = np.hstack([positions, pad])

        # Re-register point cloud each frame (vertex count may change)
        cloud = ps.register_point_cloud(name, positions)
        cloud_ref["cloud"] = cloud

        # Add scalar quantities
        for field in scalar_fields:
            vals = np.zeros(n_pts)
            for i, key in enumerate(keys):
                vdata = snapshot.get(key, {})
                val = vdata.get(field, 0.0)
                vals[i] = float(val) if np.ndim(val) == 0 else float(val[0])
            cloud.add_scalar_quantity(field, vals, enabled=(field == scalar_fields[0]))

        # Add vector quantities + velocity magnitude scalar
        for field in vector_fields:
            vecs = np.zeros((n_pts, 3))
            for i, key in enumerate(keys):
                vdata = snapshot.get(key, {})
                val = vdata.get(field, np.zeros(dim))
                arr = np.asarray(val, dtype=np.float64)
                vecs[i, :len(arr)] = arr[:min(len(arr), 3)]
            # Vector arrows (enabled, dark blue for visibility)
            cloud.add_vector_quantity(field, vecs, enabled=True,
                                      color=(0.1, 0.2, 0.6),
                                      radius=0.003)
            # Velocity magnitude as scalar colormap (enabled)
            magnitudes = np.linalg.norm(vecs, axis=1)
            cloud.add_scalar_quantity(
                f"|{field}|", magnitudes, enabled=True, cmap='coolwarm',
            )

    def _info(idx):
        import polyscope.imgui as imgui
        t = times[idx]
        imgui.Text(f"t = {t:.4f} s")
        imgui.Text(f"Frame {idx + 1} / {n_frames}")
        # Vertex count from snapshot
        n_pts = len(history._snapshots[idx][1])
        imgui.Text(f"Vertices: {n_pts}")

    interactive_viewer(
        n_frames=n_frames,
        update_fn=_load_frame,
        info_fn=_info,
        screenshot_dir=screenshot_dir,
        init=False,  # we already init'd above
    )
