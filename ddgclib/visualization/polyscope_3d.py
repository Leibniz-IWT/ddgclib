"""Polyscope-based 3D visualization (optional dependency).

All functions gracefully fail with an ImportError message if polyscope
is not installed.

Usage
-----
    from ddgclib.visualization.polyscope_3d import register_point_cloud, update_frame
    ps_cloud = register_point_cloud(HC, name='mesh')
    update_frame(HC, ps_cloud, scalar_fields=['P'], vector_fields=['u'])
"""

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
        Scalar fields to add/update (e.g. ['P', 'm']).
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
        Scalar fields to add/update (e.g. ['P', 'm']).
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
