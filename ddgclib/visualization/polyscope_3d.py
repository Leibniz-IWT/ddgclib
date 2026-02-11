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
