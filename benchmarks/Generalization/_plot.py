import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

def plot_quadric_surface(coeffs, title="Quadric Surface"):
    """
    Plot the zero-isosurface of the quadric defined by 10 coefficients.
    coeffs: (a1,a2,a3,a4,a5,a6,b1,b2,b3,c)
    """
    a1, a2, a3, a4, a5, a6, b1, b2, b3, c = coeffs

    def f(x, y, z):
        return (
            a1 * x**2 + a2 * y**2 + a3 * z**2
            + a4 * x * y + a5 * x * z + a6 * y * z
            + b1 * x + b2 * y + b3 * z + c
        )

    # 3D grid
    lim = 6
    n = 60
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    zs = np.linspace(-lim, lim, n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    F = f(X, Y, Z)

    # Extract the isosurface
    verts, faces, normals, values = measure.marching_cubes(
        F, level=0.0, spacing=(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    )
    verts += np.array([xs.min(), ys.min(), zs.min()])

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.6)
    ax.add_collection3d(mesh)

    # Set equal aspect
    all_pts = verts.flatten()
    ax.auto_scale_xyz(all_pts, all_pts, all_pts)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.show()

# Example usage:
# coeffs = (
#     1.0000000000000002,
#     1.0000000000000002,
#     0.0,
#     0.0,
#     -5.196152422706631,
#     -8.999999999999998,
#     0.0,
#     0.0,
#     0.0,
#     -9.0,
# )
# plot_quadric_surface(coeffs, "Original Quadric Surface")
