import numpy as np
from scipy.spatial import Delaunay
from ._benchmark_classes import GeometryBenchmarkBase

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ._benchmark_classes import GeometryBenchmarkBase

class TorusBenchmark(GeometryBenchmarkBase):
    def __init__(self, R_major=2.0, r_minor=1.0, n_u=30, n_v=30, **kwargs):
        super().__init__(name="Torus", **kwargs)
        self.R_major = R_major
        self.r_minor = r_minor
        self.n_u = n_u
        self.n_v = n_v

    def generate_mesh(self):
        u = np.linspace(0, 2 * np.pi, self.n_u)
        v = np.linspace(0, 2 * np.pi, self.n_v)
        u, v = np.meshgrid(u, v)
        self.u = u.flatten()
        self.v = v.flatten()

        x = (self.R_major + self.r_minor * np.cos(self.v)) * np.cos(self.u)
        y = (self.R_major + self.r_minor * np.cos(self.v)) * np.sin(self.u)
        z = self.r_minor * np.sin(self.v)
        self.points = np.vstack((x, y, z)).T

        uv_points = np.vstack((self.u, self.v)).T
        tri = Delaunay(uv_points)
        self.simplices = tri.simplices

    def analytical_values(self):
        R, r = self.R_major, self.r_minor
        self.area_analytical = 4 * np.pi**2 * R * r
        self.volume_analytical = 2 * np.pi**2 * R * r**2

        cos_v = np.cos(self.v)
        denom = 2 * self.r_minor * (self.R_major + self.r_minor * cos_v)
        self.H_analytical = cos_v / denom

    def H(self, u, v):
        """
        Analytical mean curvature H(u, v) on the torus surface.

        Parameters
        ----------
        u, v : float
            Parametric angles (in radians).

        Returns
        -------
        float
            Mean curvature at (u, v).
        """
        R, r = self.R_major, self.r_minor
        return np.cos(v) / (2 * r * (R + r * np.cos(v)))

    def f(self, u, v):
        """Parametric surface mapping from (u,v) to R^3."""
        R, r = self.R_major, self.r_minor
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        return np.array([x, y, z])

    def dA(self, uv_i, uv_j, uv_k, n_samples=3):
        """
        Integrate surface area over a triangle in (u,v)-space.

        Parameters
        ----------
        uv_i, uv_j, uv_k : array-like
            Triangle corners in (u,v) coordinates.
        n_samples : int
            Number of barycentric samples.

        Returns
        -------
        float
            Integrated surface area.
        """
        pts, weights = self._barycentric_samples(uv_i, uv_j, uv_k, n_samples)
        area = 0.0
        for (u, v), w in zip(pts, weights):
            x = self.f(u, v)
            du = (self.f(u + 1e-5, v) - x) / 1e-5
            dv = (self.f(u, v + 1e-5) - x) / 1e-5
            normal = np.cross(du, dv)
            dA = 0.5 * np.linalg.norm(normal)
            area += dA * w
        return area

    def HNdA(self, uv_i, uv_j, uv_k, n_samples=3):
        """
        Integrate mean curvature times area over a triangle in (u,v)-space.

        Parameters
        ----------
        uv_i, uv_j, uv_k : array-like
            Triangle corners in (u,v) coordinates.
        n_samples : int
            Number of barycentric samples.

        Returns
        -------
        float
            Integrated mean curvature over area.
        """
        pts, weights = self._barycentric_samples(uv_i, uv_j, uv_k, n_samples)
        integral = 0.0
        for (u, v), w in zip(pts, weights):
            x = self.f(u, v)
            du = (self.f(u + 1e-5, v) - x) / 1e-5
            dv = (self.f(u, v + 1e-5) - x) / 1e-5
            normal = np.cross(du, dv)
            dA = 0.5 * np.linalg.norm(normal)
            H_val = self.H(u, v)
            integral += H_val * dA * w
        return integral

    def _barycentric_samples(self, uv_i, uv_j, uv_k, n_samples):
        """Generate barycentric integration points and uniform weights."""
        pts = []
        weights = []
        for i in range(n_samples + 1):
            for j in range(n_samples + 1 - i):
                a = i / n_samples
                b = j / n_samples
                c = 1 - a - b
                pts.append(a * uv_i + b * uv_j + c * uv_k)
                weights.append(1)
        pts = np.array(pts)
        weights = np.array(weights)
        weights = weights / weights.sum()
        return pts, weights

    def plot_uv_triangulation(self):
        """Plot the triangulation in the (u,v) parametric plane."""
        plt.figure(figsize=(6, 6))
        for tri in self.simplices:
            uv_coords = np.array([self.u[tri], self.v[tri]]).T
            uv_coords = np.vstack((uv_coords, uv_coords[0]))  # close the triangle
            plt.plot(uv_coords[:, 0], uv_coords[:, 1], 'k-')
        plt.xlabel("u")
        plt.ylabel("v")
        plt.title("Parametric (u,v) Triangulation")
        plt.axis("equal")
        plt.show()

    def plot_surface_mesh(self):
        """Plot the 3D simplicial complex mesh."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(self.points[self.simplices], alpha=0.5)
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], s=1, color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D Torus Surface Mesh")
        ax.auto_scale_xyz(self.points[:, 0], self.points[:, 1], self.points[:, 2])
        plt.tight_layout()
        plt.show()