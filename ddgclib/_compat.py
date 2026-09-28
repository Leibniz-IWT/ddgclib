"""
Compatibility utilities for functions not yet migrated to hyperct.ddg.

These functions were originally in ddgclib.barycentric._duals and are
only used by the Dynamic_caprise_tube case study.
"""
import numpy as np
from scipy.spatial import Delaunay
from hyperct import Complex


def _signed_volume_parallelepiped(u, v, w):
    """Signed volume of a parallelepiped defined by three edge vectors."""
    u, v, w = map(np.array, (u, v, w))
    v_para = np.dot(u, v) * u
    v_ortho = v - v_para
    w_prime = w - 2 * v_para
    return (np.cross(u, v_ortho)).dot(w_prime) / 6


def _volume_parallelepiped(u, v, w):
    """Absolute volume of a parallelepiped defined by three edge vectors."""
    return np.abs(_signed_volume_parallelepiped(u, v, w))


def triang_dual(points, plot_delaunay=False):
    """Compute Delaunay triangulation and wrap in a hyperct Complex.

    Parameters
    ----------
    points : ndarray of shape (n, dim)
        Point coordinates.
    plot_delaunay : bool
        If True, plot the Delaunay triangulation.

    Returns
    -------
    HC : Complex
        hyperct Complex with connectivity from the Delaunay triangulation.
    tri : Delaunay
        scipy Delaunay object.
    """
    dim = points.shape[1]
    tri = Delaunay(points)
    if plot_delaunay:
        import matplotlib.pyplot as plt
        plt.triplot(points[:, 0], points[:, 1], tri.simplices)
        plt.plot(points[:, 0], points[:, 1], 'o')
        plt.show()

    HC = Complex(dim)
    for s in tri.simplices:
        for v1i in s:
            for v2i in s:
                if v1i is v2i:
                    continue
                v1 = tuple(points[v1i])
                v2 = tuple(points[v2i])
                HC.V[v1].connect(HC.V[v2])

    return HC, tri
