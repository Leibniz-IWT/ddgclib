import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ._benchmark_classes import GeometryBenchmarkBase
import pandas as pd
# benchmarks/_benchmark_cases.py
from pathlib import Path

# ------------------------- Generic “just a mesh” case -------------------------
class MshCase(GeometryBenchmarkBase):
    """
    Loads a mesh from a .msh file and sets analytical values to NaN.
    Use this to run the benchmark framework with an arbitrary mesh; the
    actual algorithm is chosen via the registry (e.g., method={"volume_method": "curved_volume"}).
    """
    def __init__(self, msh_path, **kwargs):
        super().__init__(name=f"Msh[{Path(msh_path).stem}]", **kwargs)
        self.msh_path = str(msh_path)

    def generate_mesh(self):
        # Prefer the project helper if available
        try:
            from ._benchmark_plotting_utils import read_gmsh_tri
        except Exception as e:
            raise ImportError(
                "read_gmsh_tri not available. Ensure benchmarks/_benchmark_plotting_utils.py exists "
                "or replace this call with your own .msh reader that returns (points, simplices)."
            ) from e
        self.points, self.simplices = read_gmsh_tri(self.msh_path)

    def analytical_values(self):
        # Unknown / not applicable -> NaN (not 0.0)
        self.area_analytical = np.nan
        self.volume_analytical = np.nan
        self.H_analytical = np.nan

class EllipsoidMshBenchmark(MshCase):
    def __init__(self, msh_path="test_cases/Ellip_0_sub0_full.msh",
                 ax=1.5, ay=1.0, az=0.8,
                 complex_dtype="vf",
                 workdir="benchmarks_out/ellipsoid",
                 method=None, **kwargs):
        if method is None:
            method = {"volume_method": "curved_volume", "curvature_i_method": "laplace-beltrami"}
        super().__init__(msh_path=msh_path, method=method, complex_dtype=complex_dtype, **kwargs)

        self.ax, self.ay, self.az = float(ax), float(ay), float(az)
        self.workdir = str(workdir)
        self._workdir_path = Path(self.workdir)

        self.V_theory = self.V_flat = self.V_sum = self.V_total = self.rel_err_percent = None

    def _compute_V_theory(self):
        return (4.0/3.0) * np.pi * self.ax * self.ay * self.az

    def _read_Vsum_from_csv(self):
        stem = Path(self.msh_path).stem
        csv_path = self._workdir_path / f"{stem}_COEFFS_Transformed_DualVolume.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

        vdf = pd.read_csv(csv_path)
        if "Vcorrection" in vdf.columns:
            return float(np.nansum(vdf["Vcorrection"].to_numpy(dtype=float)))
        if "DualVolume" in vdf.columns:
            return float(np.nansum(vdf["DualVolume"].to_numpy(dtype=float)))
        raise ValueError(f"CSV missing Vcorrection/DualVolume columns: {csv_path}")

    def run(self):
        self.V_theory = self._compute_V_theory()

        # ensure method passes workdir into curved pipeline (depends on your base-class implementation)
        self.method = dict(self.method)
        self.method["workdir"] = self.workdir
        self.method["msh_path"] = self.msh_path   # IMPORTANT: ensures CSV uses <stem> not "mesh"
        self._workdir_path.mkdir(parents=True, exist_ok=True)

        # curved run (produces CSV)
        self.run_benchmark()

        # piecewise-linear flat volume
        flat_method = {"volume_method": "default", "curvature_i_method": "laplace-beltrami"}
        flat = MshCase(msh_path=self.msh_path, method=flat_method, complex_dtype=self.complex_dtype)
        flat.run_benchmark()
        self.V_flat = float(flat.volume_computed)

        self.V_sum = self._read_Vsum_from_csv()
        self.V_total = self.V_flat + self.V_sum

        # standard sign: computed - theory
        self.rel_err_percent = (self.V_total - self.V_theory) / self.V_theory * 100.0
        return self

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

class SphereBenchmark(GeometryBenchmarkBase):
    def __init__(self, radius=1.0, n_points=1000, **kwargs):
        super().__init__(name="Sphere", **kwargs)
        self.radius = radius
        self.n_points = n_points

    def generate_mesh(self):
        phi = np.arccos(1 - 2 * np.random.rand(self.n_points))
        theta = 2 * np.pi * np.random.rand(self.n_points)
        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = self.radius * np.cos(phi)
        self.points = np.vstack((x, y, z)).T
        tri = Delaunay(self.points[:, :2])
        self.simplices = tri.simplices

    def analytical_values(self):
        r = self.radius
        self.area_analytical = 4 * np.pi * r**2
        self.volume_analytical = (4/3) * np.pi * r**3
        self.H_analytical = np.full(len(self.points), 1 / r)