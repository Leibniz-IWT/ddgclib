import logging
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ._benchmark_classes import GeometryBenchmarkBase, FlowBenchmarkBase
import pandas as pd
# benchmarks/_benchmark_cases.py
from pathlib import Path

logger = logging.getLogger(__name__)

# Generic “just a mesh” case
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


def _configure_background_mesh_field(model, entity_kind, entity_tags, size_min, size_max,
                                     dist_min, dist_max):
    """
    Create a simple distance-threshold background field around an entity set.

    This keeps the obstacle boundary well-resolved while allowing coarser cells
    away from the sphere.
    """
    if not entity_tags:
        return

    distance = model.mesh.field.add("Distance")
    key = "CurvesList" if entity_kind == "curve" else "FacesList"
    model.mesh.field.setNumbers(distance, key, list(entity_tags))

    threshold = model.mesh.field.add("Threshold")
    model.mesh.field.setNumber(threshold, "InField", distance)
    model.mesh.field.setNumber(threshold, "SizeMin", float(size_min))
    model.mesh.field.setNumber(threshold, "SizeMax", float(size_max))
    model.mesh.field.setNumber(threshold, "DistMin", float(dist_min))
    model.mesh.field.setNumber(threshold, "DistMax", float(dist_max))
    model.mesh.field.setAsBackgroundMesh(threshold)


def _safe_set_gmsh_option(gmsh_module, name, value):
    """Set a Gmsh numeric option, ignoring unsupported-option errors."""
    try:
        gmsh_module.option.setNumber(str(name), float(value))
    except Exception:
        pass


def _generate_stokes_sphere_mesh_3d(path, sphere_radius, outer_radius,
                                    mesh_size_inner, mesh_size_outer,
                                    mesh_order=2,
                                    symm_nonobtuse=False,
                                    equal_edge=False):
    """
    Generate a quadratic tetrahedral shell mesh: outer sphere minus inner sphere.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError("gmsh is required to generate the 3D Stokes sphere mesh") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        _safe_set_gmsh_option(gmsh, "General.Terminal", 1)
        gmsh.model.add("stokes_sphere_3d")
        occ = gmsh.model.occ

        outer = occ.addSphere(0.0, 0.0, 0.0, float(outer_radius))
        inner = occ.addSphere(0.0, 0.0, 0.0, float(sphere_radius))
        shell, _ = occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
        occ.synchronize()

        vol_tags = [tag for dim, tag in shell if dim == 3]
        gmsh.model.addPhysicalGroup(3, vol_tags, FlowBenchmarkBase.volume_physical_id)

        boundary = gmsh.model.getBoundary(shell, oriented=False, recursive=False)
        inner_surfaces = []
        outer_surfaces = []
        for _, tag in boundary:
            bbox = gmsh.model.getBoundingBox(2, tag)
            radius_guess = max(abs(v) for v in bbox)
            if abs(radius_guess - sphere_radius) <= abs(radius_guess - outer_radius):
                inner_surfaces.append(tag)
            else:
                outer_surfaces.append(tag)

        gmsh.model.addPhysicalGroup(2, inner_surfaces, FlowBenchmarkBase.obstacle_physical_id)
        gmsh.model.addPhysicalGroup(2, outer_surfaces, FlowBenchmarkBase.outer_boundary_physical_id)

        # Keep independent inner/outer size control for all variants so boundary
        # element counts can be aligned across benchmark cases.
        _safe_set_gmsh_option(gmsh, "Mesh.CharacteristicLengthMin", float(mesh_size_inner))
        _safe_set_gmsh_option(gmsh, "Mesh.CharacteristicLengthMax", float(mesh_size_outer))
        _safe_set_gmsh_option(gmsh, "Mesh.ElementOrder", int(mesh_order))
        if symm_nonobtuse:
            # Deterministic + quality-focused meshing controls.
            _safe_set_gmsh_option(gmsh, "Mesh.RandomFactor", 0.0)
            _safe_set_gmsh_option(gmsh, "Mesh.RandomFactor3D", 0.0)
            _safe_set_gmsh_option(gmsh, "Mesh.Algorithm", 6)
            _safe_set_gmsh_option(gmsh, "Mesh.Algorithm3D", 10)
            _safe_set_gmsh_option(gmsh, "Mesh.Optimize", 1)
            _safe_set_gmsh_option(gmsh, "Mesh.OptimizeNetgen", 1)
            _safe_set_gmsh_option(gmsh, "Mesh.OptimizeThreshold", 0.35)
            _safe_set_gmsh_option(gmsh, "Mesh.Smoothing", 100)
        if equal_edge:
            # Favor near-equal local edge lengths while preserving boundary
            # count matching via the same inner/outer sizing field.
            _safe_set_gmsh_option(gmsh, "Mesh.Algorithm", 6)
            _safe_set_gmsh_option(gmsh, "Mesh.Algorithm3D", 10)
            _safe_set_gmsh_option(gmsh, "Mesh.Optimize", 1)
            _safe_set_gmsh_option(gmsh, "Mesh.OptimizeNetgen", 1)
            _safe_set_gmsh_option(gmsh, "Mesh.Smoothing", 80)

        _configure_background_mesh_field(
            gmsh.model,
            entity_kind="surface",
            entity_tags=inner_surfaces,
            size_min=mesh_size_inner,
            size_max=mesh_size_outer,
            dist_min=0.25 * sphere_radius,
            dist_max=max(outer_radius - sphere_radius, mesh_size_outer),
        )

        gmsh.model.mesh.generate(3)
        if mesh_order > 1:
            gmsh.model.mesh.setOrder(int(mesh_order))
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    return str(path)


def _generate_stokes_sphere_mesh_2d(path, sphere_radius, outer_radius,
                                    mesh_size_inner, mesh_size_outer,
                                    mesh_order=2,
                                    symm_nonobtuse=False):
    """
    Generate a quadratic axisymmetric meridional mesh on the half-annulus
    ``{(rho, z): a <= sqrt(rho^2 + z^2) <= R, rho >= 0}``.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError("gmsh is required to generate the 2D Stokes sphere mesh") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        _safe_set_gmsh_option(gmsh, "General.Terminal", 1)
        gmsh.model.add("stokes_sphere_2d_axisymmetric")
        occ = gmsh.model.occ

        outer = occ.addDisk(0.0, 0.0, 0.0, float(outer_radius), float(outer_radius))
        inner = occ.addDisk(0.0, 0.0, 0.0, float(sphere_radius), float(sphere_radius))
        annulus, _ = occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)

        half_plane = occ.addRectangle(0.0, -float(outer_radius), 0.0,
                                      float(outer_radius), 2.0 * float(outer_radius))
        half_shell, _ = occ.intersect(annulus, [(2, half_plane)],
                                      removeObject=True, removeTool=True)
        occ.synchronize()

        surf_tags = [tag for dim, tag in half_shell if dim == 2]
        gmsh.model.addPhysicalGroup(2, surf_tags, FlowBenchmarkBase.volume_physical_id)

        boundary = gmsh.model.getBoundary(half_shell, oriented=False, recursive=False)
        inner_curves = []
        outer_curves = []
        axis_curves = []
        axis_tol = max(1.0e-8, 1.0e-6 * float(outer_radius))
        for _, tag in boundary:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
            if max(abs(xmin), abs(xmax)) <= axis_tol:
                axis_curves.append(tag)
                continue

            radius_guess = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            if abs(radius_guess - sphere_radius) <= abs(radius_guess - outer_radius):
                inner_curves.append(tag)
            else:
                outer_curves.append(tag)

        gmsh.model.addPhysicalGroup(1, inner_curves, FlowBenchmarkBase.obstacle_physical_id)
        gmsh.model.addPhysicalGroup(1, outer_curves, FlowBenchmarkBase.outer_boundary_physical_id)
        if axis_curves:
            gmsh.model.addPhysicalGroup(1, axis_curves, FlowBenchmarkBase.axis_physical_id)

        _safe_set_gmsh_option(gmsh, "Mesh.CharacteristicLengthMin", float(mesh_size_inner))
        _safe_set_gmsh_option(gmsh, "Mesh.CharacteristicLengthMax", float(mesh_size_outer))
        _safe_set_gmsh_option(gmsh, "Mesh.ElementOrder", int(mesh_order))
        if symm_nonobtuse:
            # Deterministic + quality-focused 2D controls (best-effort).
            _safe_set_gmsh_option(gmsh, "Mesh.RandomFactor", 0.0)
            _safe_set_gmsh_option(gmsh, "Mesh.Algorithm", 6)
            _safe_set_gmsh_option(gmsh, "Mesh.Optimize", 1)
            _safe_set_gmsh_option(gmsh, "Mesh.OptimizeNetgen", 1)
            _safe_set_gmsh_option(gmsh, "Mesh.OptimizeThreshold", 0.35)
            _safe_set_gmsh_option(gmsh, "Mesh.Smoothing", 100)
        _configure_background_mesh_field(
            gmsh.model,
            entity_kind="curve",
            entity_tags=inner_curves,
            size_min=mesh_size_inner,
            size_max=mesh_size_outer,
            dist_min=0.15 * sphere_radius,
            dist_max=max(outer_radius - sphere_radius, mesh_size_outer),
        )

        gmsh.model.mesh.generate(2)
        if mesh_order > 1:
            gmsh.model.mesh.setOrder(int(mesh_order))
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    return str(path)


def _build_hc_seed_points(dim, bounds, refine_level):
    """
    Build an independently generated simplicial seed cloud with hyperct.
    """
    try:
        from hyperct import Complex
    except ImportError as exc:
        raise ImportError("hyperct is required for HC-generated Stokes meshes") from exc

    HC_seed = Complex(dim, domain=bounds)
    HC_seed.triangulate()
    for _ in range(max(int(refine_level), 0)):
        HC_seed.refine_all()

    points = np.array([np.asarray(v.x_a, dtype=float) for v in HC_seed.V], dtype=float)
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError("Failed to build HC seed points with expected shape")
    return points


def _fibonacci_sphere_points(n_points, radius):
    if n_points <= 0:
        return np.empty((0, 3), dtype=float)
    if n_points == 1:
        return np.array([[0.0, 0.0, float(radius)]], dtype=float)

    k = np.arange(n_points, dtype=float)
    z = 1.0 - 2.0 * k / (n_points - 1.0)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden * k
    r_xy = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    pts = np.column_stack([x, y, z])
    return float(radius) * pts


def _symmetric_sphere_points(n_points, radius):
    """
    Deterministic sphere samples with exact central pairing ``x <-> -x``.
    """
    n_points = int(max(n_points, 0))
    if n_points == 0:
        return np.empty((0, 3), dtype=float)
    if n_points == 1:
        return np.array([[0.0, 0.0, float(radius)]], dtype=float)

    n_half = max(1, n_points // 2)
    base = _fibonacci_sphere_points(n_half, radius)
    paired = np.vstack([base, -base])
    if n_points % 2 == 1:
        paired = np.vstack([paired, np.array([[0.0, 0.0, float(radius)]], dtype=float)])
    return paired[:n_points]


def _symmetrize_points_origin(points, dim=3):
    """
    Enforce central symmetry by adding mirrored points ``-x``.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points
    if dim == 2:
        mirrored = np.column_stack([points[:, 0], -points[:, 1]])
    else:
        mirrored = -points
    return _unique_rows(np.vstack([points, mirrored]), decimals=12)


def _triangle_max_angle_deg(a, b, c):
    def _ang(u, v):
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1.0e-30 or nv < 1.0e-30:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))

    A = _ang(b - a, c - a)
    B = _ang(a - b, c - b)
    C = _ang(a - c, b - c)
    return max(A, B, C)


def _tetra_max_face_angle_deg(points, tet):
    i0, i1, i2, i3 = [int(v) for v in tet[:4]]
    p0, p1, p2, p3 = points[i0], points[i1], points[i2], points[i3]
    return max(
        _triangle_max_angle_deg(p0, p1, p2),
        _triangle_max_angle_deg(p0, p1, p3),
        _triangle_max_angle_deg(p0, p2, p3),
        _triangle_max_angle_deg(p1, p2, p3),
    )


def _augment_points_nonobtuse_shell_3d(points, sphere_radius, outer_radius,
                                       angle_threshold_deg=120.0, max_new_points=1024,
                                       iterations=1):
    """
    Heuristic obtuse-angle reduction by inserting mirrored Steiner points in
    high-angle tetrahedra. This is a best-effort quality improvement, not a
    strict non-obtuse guarantee.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points

    radial_gap = float(outer_radius) - float(sphere_radius)
    inner_safe = float(sphere_radius) + 0.08 * radial_gap
    outer_safe = float(outer_radius) - 0.08 * radial_gap

    for _ in range(max(int(iterations), 0)):
        try:
            tri = Delaunay(points)
        except Exception:
            break
        simplices = np.asarray(tri.simplices, dtype=np.int64)
        if simplices.size == 0:
            break

        centroids = np.mean(points[simplices], axis=1)
        rr = np.linalg.norm(centroids[:, :3], axis=1)
        keep = (rr >= float(sphere_radius) * (1.0 - 1.0e-8)) & (rr <= float(outer_radius) * (1.0 + 1.0e-8))
        simplices = simplices[keep]
        centroids = centroids[keep]
        if simplices.size == 0:
            break

        bad = []
        for idx, tet in enumerate(simplices):
            max_face_ang = _tetra_max_face_angle_deg(points, tet)
            if max_face_ang > float(angle_threshold_deg):
                bad.append((max_face_ang, idx))
        if not bad:
            break

        bad.sort(reverse=True)
        new_pts = []
        for _, idx in bad:
            c = centroids[idx]
            r = float(np.linalg.norm(c))
            if r < inner_safe or r > outer_safe:
                continue
            new_pts.append(c)
            if len(new_pts) >= int(max_new_points // 2):
                break

        if not new_pts:
            break

        new_pts = np.asarray(new_pts, dtype=float)
        points = _unique_rows(np.vstack([points, new_pts, -new_pts]), decimals=12)

    return points


def _augment_points_nonobtuse_shell_2d(points, sphere_radius, outer_radius,
                                       angle_threshold_deg=118.0, max_new_points=512,
                                       iterations=1):
    """
    Best-effort obtuse-angle reduction in the 2D half-annulus by inserting
    mirrored Steiner points in high-angle triangles.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points

    radial_gap = float(outer_radius) - float(sphere_radius)
    inner_safe = float(sphere_radius) + 0.06 * radial_gap
    outer_safe = float(outer_radius) - 0.06 * radial_gap

    for _ in range(max(int(iterations), 0)):
        try:
            tri = Delaunay(points)
        except Exception:
            break

        simplices = np.asarray(tri.simplices, dtype=np.int64)
        if simplices.size == 0:
            break

        centroids = np.mean(points[simplices], axis=1)
        rr = _radius_values(centroids, dim=2)
        keep = (
            (rr >= float(sphere_radius) * (1.0 - 1.0e-8))
            & (rr <= float(outer_radius) * (1.0 + 1.0e-8))
            & (centroids[:, 0] >= -1.0e-10)
        )
        simplices = simplices[keep]
        centroids = centroids[keep]
        if simplices.size == 0:
            break

        bad = []
        for idx, tri_ids in enumerate(simplices):
            a, b, c = points[int(tri_ids[0])], points[int(tri_ids[1])], points[int(tri_ids[2])]
            max_ang = _triangle_max_angle_deg(a, b, c)
            if max_ang > float(angle_threshold_deg):
                bad.append((max_ang, idx))
        if not bad:
            break

        bad.sort(reverse=True)
        new_pts = []
        for _, idx in bad:
            c = centroids[idx]
            r = float(np.linalg.norm(c))
            if c[0] < 0.0 or r < inner_safe or r > outer_safe:
                continue

            cm = np.array([max(float(c[0]), 0.0), -float(c[1])], dtype=float)
            rm = float(np.linalg.norm(cm))
            if cm[0] < 0.0 or rm < inner_safe or rm > outer_safe:
                continue

            new_pts.append(np.array([max(float(c[0]), 0.0), float(c[1])], dtype=float))
            new_pts.append(cm)
            if len(new_pts) >= int(max_new_points):
                break

        if not new_pts:
            break

        points = _unique_rows(np.vstack([points, np.asarray(new_pts, dtype=float)]), decimals=12)

    return points


def _unique_rows(points, decimals=12):
    rounded = np.round(np.asarray(points, dtype=float), decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return np.asarray(points, dtype=float)[np.sort(idx)]


def _radius_values(points, dim):
    points = np.asarray(points, dtype=float)
    if dim == 2:
        return np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    return np.linalg.norm(points[:, :3], axis=1)


def _boundary_facets_from_simplices(simplices, dim):
    simplices = np.asarray(simplices, dtype=np.int64)
    if simplices.size == 0:
        n_nodes = 2 if dim == 2 else 3
        return np.empty((0, n_nodes), dtype=np.int64)

    if dim == 2:
        local_facets = ((0, 1), (1, 2), (2, 0))
    else:
        local_facets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

    counts = {}
    for simplex in simplices:
        for lf in local_facets:
            key = tuple(sorted(int(simplex[i]) for i in lf))
            counts[key] = counts.get(key, 0) + 1

    boundary = [facet for facet, count in counts.items() if count == 1]
    if not boundary:
        n_nodes = 2 if dim == 2 else 3
        return np.empty((0, n_nodes), dtype=np.int64)
    return np.asarray(boundary, dtype=np.int64)


def _classify_shell_boundary_elements(points, facets, dim, sphere_radius, outer_radius):
    points = np.asarray(points, dtype=float)
    facets = np.asarray(facets, dtype=np.int64)

    if facets.size == 0:
        return (
            np.empty((0, 2 if dim == 2 else 3), dtype=np.int64),
            np.empty((0, 2 if dim == 2 else 3), dtype=np.int64),
            np.empty((0, 2), dtype=np.int64) if dim == 2 else np.empty((0, 0), dtype=np.int64),
        )

    axis_tol = max(1.0e-8, 1.0e-6 * float(outer_radius))
    radial_gap = abs(float(outer_radius) - float(sphere_radius))
    inner_band = max(1.0e-6, 0.05 * radial_gap, 0.02 * float(sphere_radius))
    outer_band = max(1.0e-6, 0.05 * radial_gap, 0.02 * float(outer_radius))
    inner = []
    outer = []
    axis = []

    for facet in facets:
        node_ids = np.asarray(facet, dtype=np.int64)
        nodes = points[node_ids]

        if dim == 2 and np.max(np.abs(nodes[:, 0])) <= axis_tol:
            axis.append(node_ids)
            continue

        r_mean = float(np.mean(_radius_values(nodes, dim)))
        if abs(r_mean - sphere_radius) <= inner_band:
            inner.append(node_ids)
        elif abs(r_mean - outer_radius) <= outer_band:
            outer.append(node_ids)

    n_nodes = 2 if dim == 2 else 3
    inner = np.asarray(inner, dtype=np.int64) if inner else np.empty((0, n_nodes), dtype=np.int64)
    outer = np.asarray(outer, dtype=np.int64) if outer else np.empty((0, n_nodes), dtype=np.int64)
    if dim == 2:
        axis = np.asarray(axis, dtype=np.int64) if axis else np.empty((0, 2), dtype=np.int64)
    else:
        axis = np.empty((0, 0), dtype=np.int64)
    return inner, outer, axis


def _generate_stokes_sphere_hc_mesh(dim, sphere_radius, outer_radius, hc_refine=2,
                                    hc_inner_samples=None, hc_outer_samples=None,
                                    symm_nonobtuse=False,
                                    equal_edge=False):
    """
    Generate a shell/annulus mesh independently from hyperct seed points.

    This path does not rely on Gmsh.  It uses:
    1. hyperct to generate/refine a simplicial seed cloud,
    2. geometric shell filtering in the target domain,
    3. scipy Delaunay to reconstruct simplex cells in that filtered cloud.

    When ``symm_nonobtuse=True``, the generator enforces central symmetry on
    sample points and applies a best-effort obtuse-angle reduction heuristic.
    When ``equal_edge=True``, the generator uses quasi-uniform spherical layers
    to favor near-equal edge lengths (near-equilateral behavior).
    """
    if dim == 2:
        bounds = [(0.0, float(outer_radius)), (-float(outer_radius), float(outer_radius))]
        n_inner = int(hc_inner_samples) if hc_inner_samples is not None else max(32, 16 * (2 ** max(hc_refine - 1, 0)))
        n_outer = int(hc_outer_samples) if hc_outer_samples is not None else max(48, 24 * (2 ** max(hc_refine - 1, 0)))
    else:
        bounds = [(-float(outer_radius), float(outer_radius))] * 3
        n_inner = int(hc_inner_samples) if hc_inner_samples is not None else max(96, 64 * (2 ** max(hc_refine - 1, 0)))
        n_outer = int(hc_outer_samples) if hc_outer_samples is not None else max(128, 96 * (2 ** max(hc_refine - 1, 0)))

    if dim == 3 and equal_edge:
        # Build quasi-uniform radial shells with near-equal spacing.
        h_outer = np.sqrt(4.0 * np.pi * float(outer_radius) ** 2 / max(float(n_outer), 1.0))
        h_inner = np.sqrt(4.0 * np.pi * float(sphere_radius) ** 2 / max(float(n_inner), 1.0))
        target_h = max(1.0e-6, 0.5 * (h_outer + h_inner))
        n_layers = max(3, int(np.ceil((float(outer_radius) - float(sphere_radius)) / target_h)) + 1)
        radii = np.linspace(float(sphere_radius), float(outer_radius), n_layers)

        layer_points = []
        for li, r in enumerate(radii):
            n_layer = max(64, int(round(4.0 * np.pi * float(r) ** 2 / max(target_h**2, 1.0e-12))))
            if n_layer % 2 == 1:
                n_layer += 1
            if symm_nonobtuse:
                pts = _symmetric_sphere_points(n_layer, r)
            else:
                pts = _fibonacci_sphere_points(n_layer, r)
            # Rotate each layer to avoid radial alignment artifacts.
            ang = (li + 1) * np.pi * (3.0 - np.sqrt(5.0))
            c, s = np.cos(ang), np.sin(ang)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            pts = pts @ rot.T
            layer_points.append(pts)
        seed_points = np.vstack(layer_points)
    else:
        seed_points = _build_hc_seed_points(dim, bounds, hc_refine)
        if symm_nonobtuse:
            seed_points = _symmetrize_points_origin(seed_points, dim=dim)

    if dim == 2:
        phi_in = np.linspace(0.0, np.pi, n_inner)
        phi_out = np.linspace(0.0, np.pi, n_outer)
        inner_pts = np.column_stack([
            float(sphere_radius) * np.sin(phi_in),
            float(sphere_radius) * np.cos(phi_in),
        ])
        outer_pts = np.column_stack([
            float(outer_radius) * np.sin(phi_out),
            float(outer_radius) * np.cos(phi_out),
        ])
        # Add one interior guard ring near each curved boundary to avoid
        # boundary fan/sliver triangles from unconstrained Delaunay.
        radial_gap = float(outer_radius) - float(sphere_radius)
        inner_guard_r = min(
            float(sphere_radius) + max(0.03 * radial_gap, 0.04 * float(sphere_radius)),
            0.5 * (float(sphere_radius) + float(outer_radius)),
        )
        outer_guard_r = max(
            float(outer_radius) - max(0.03 * radial_gap, 0.04 * float(outer_radius)),
            0.5 * (float(sphere_radius) + float(outer_radius)),
        )
        inner_guard_pts = np.column_stack([
            inner_guard_r * np.sin(phi_in),
            inner_guard_r * np.cos(phi_in),
        ])
        outer_guard_pts = np.column_stack([
            outer_guard_r * np.sin(phi_out),
            outer_guard_r * np.cos(phi_out),
        ])
        # Remove HC seed points in the near-boundary strips: those zones are
        # represented by the explicit boundary and guard-ring samples.
        seed_r = _radius_values(seed_points, dim=2)
        seed_keep = (
            (seed_r >= inner_guard_r + 1.0e-12)
            & (seed_r <= outer_guard_r - 1.0e-12)
            & (seed_points[:, 0] >= -1.0e-10)
        )
        seed_points = seed_points[seed_keep]
    else:
        if symm_nonobtuse:
            inner_pts = _symmetric_sphere_points(n_inner, sphere_radius)
            outer_pts = _symmetric_sphere_points(n_outer, outer_radius)
        else:
            inner_pts = _fibonacci_sphere_points(n_inner, sphere_radius)
            outer_pts = _fibonacci_sphere_points(n_outer, outer_radius)
        inner_guard_pts = np.empty((0, 3), dtype=float)
        outer_guard_pts = np.empty((0, 3), dtype=float)

    points = np.vstack([seed_points, inner_pts, outer_pts, inner_guard_pts, outer_guard_pts])
    radii = _radius_values(points, dim)
    tol = 1.0e-10
    mask = (
        (radii >= float(sphere_radius) * (1.0 - tol))
        & (radii <= float(outer_radius) * (1.0 + tol))
    )
    if dim == 2:
        mask &= points[:, 0] >= -1.0e-10

    points = _unique_rows(points[mask], decimals=12)
    if symm_nonobtuse:
        if dim == 2:
            points = _augment_points_nonobtuse_shell_2d(
                points,
                sphere_radius=sphere_radius,
                outer_radius=outer_radius,
                angle_threshold_deg=116.0,
                max_new_points=768,
                iterations=1,
            )
        else:
            points = _augment_points_nonobtuse_shell_3d(
                points,
                sphere_radius=sphere_radius,
                outer_radius=outer_radius,
                angle_threshold_deg=118.0,
                max_new_points=1024,
                iterations=1,
            )
        points = _unique_rows(points, decimals=12)
    if len(points) <= (dim + 1):
        raise ValueError("Not enough points to triangulate HC-generated shell mesh")

    tri = Delaunay(points)
    simplices = np.asarray(tri.simplices, dtype=np.int64)
    simplex_points = points[simplices]
    centroids = np.mean(simplex_points, axis=1)
    centroid_r = _radius_values(centroids, dim)
    keep = (
        centroid_r >= float(sphere_radius) * (1.0 - 1.0e-8)
    ) & (
        centroid_r <= float(outer_radius) * (1.0 + 1.0e-8)
    )

    if dim == 2:
        keep &= centroids[:, 0] >= -1.0e-8

    simplices = simplices[keep]
    if simplices.size == 0:
        raise ValueError("HC-generated shell triangulation produced no valid simplices")

    facets = _boundary_facets_from_simplices(simplices, dim)
    obstacle, outer_boundary, axis = _classify_shell_boundary_elements(
        points,
        facets,
        dim=dim,
        sphere_radius=sphere_radius,
        outer_radius=outer_radius,
    )

    return points, simplices, obstacle, outer_boundary, axis


class _StokesSphereBenchmarkBase(FlowBenchmarkBase):
    """
    Shared analytical field for creeping flow past a sphere.

    The 3D case uses the full spherical shell domain.  The 2D case is the
    axisymmetric meridional reduction in ``(rho, z)`` and integrates the
    traction with the usual ``2*pi*rho`` weight, so it still validates the
    3D Stokes drag on a sphere.
    """

    def __init__(self, workdir, mesh_name, mesh_size_inner=None, mesh_size_outer=None,
                 remesh=False, **kwargs):
        super().__init__(**kwargs)
        self.workdir = str(workdir)
        self._workdir_path = Path(self.workdir)
        self.mesh_name = mesh_name
        self.remesh = bool(remesh)
        self.mesh_size_inner = float(
            mesh_size_inner if mesh_size_inner is not None else 0.20 * self.sphere_radius
        )
        self.mesh_size_outer = float(
            mesh_size_outer if mesh_size_outer is not None else 0.75 * self.sphere_radius
        )

    def _resolved_flow_speed(self):
        if self.flow_speed is None:
            if abs(self.mu) < 1.0e-30:
                raise ZeroDivisionError("mu must be non-zero when inferring flow_speed from gravity")
            self.flow_speed = (
                2.0 * self.density_difference * self.gravity * self.sphere_radius**2
            ) / (9.0 * self.mu)
        return self.flow_speed

    def analytical_values(self):
        U = self._resolved_flow_speed()
        a = self.sphere_radius
        drag = 6.0 * np.pi * self.mu * a * U
        gravity = (4.0 / 3.0) * np.pi * a**3 * self.density_difference * self.gravity

        self.drag_force_analytical = float(drag)
        self.gravity_force_analytical = float(gravity)
        self.force_vector_analytical = np.zeros(self.vector_dim, dtype=float)
        self.force_vector_analytical[self.flow_axis] = float(drag)

    def _spherical_components(self, x):
        x = np.asarray(x, dtype=float)[: self.vector_dim]
        rel = x - self.center

        if self.vector_dim == 2:
            rho = rel[0]
            z = rel[1]
        else:
            rho = float(np.linalg.norm(rel[:2]))
            z = rel[2]

        r = float(np.sqrt(rho**2 + z**2))
        if r < 1.0e-14:
            raise ValueError("The Stokes sphere analytical field is undefined at the origin")

        sin_theta = rho / r
        cos_theta = z / r
        U = self._resolved_flow_speed()
        a = self.sphere_radius

        u_r = U * cos_theta * (1.0 - 1.5 * a / r + 0.5 * (a**3) / (r**3))
        u_theta = -U * sin_theta * (1.0 - 0.75 * a / r - 0.25 * (a**3) / (r**3))
        p = -1.5 * self.mu * U * a * cos_theta / (r**2)
        return rel, rho, z, r, sin_theta, cos_theta, u_r, u_theta, p

    def analytical_pressure(self, x):
        return self._spherical_components(x)[-1]

    def analytical_velocity(self, x):
        rel, rho, _, _, sin_theta, cos_theta, u_r, u_theta, _ = self._spherical_components(x)
        u_rho = u_r * sin_theta + u_theta * cos_theta
        u_z = u_r * cos_theta - u_theta * sin_theta

        if self.vector_dim == 2:
            return np.array([u_rho, u_z], dtype=float)

        out = np.zeros(3, dtype=float)
        if rho > 1.0e-14:
            out[0] = u_rho * rel[0] / rho
            out[1] = u_rho * rel[1] / rho
        out[2] = u_z
        return out

    def analytical_pressure_cartesian_3d(self, x):
        x = np.asarray(x, dtype=float)[:3]
        rho = float(np.linalg.norm(x[:2]))
        z = float(x[2])
        r = float(np.sqrt(rho**2 + z**2))
        if r < 1.0e-14:
            raise ValueError("The Stokes sphere analytical field is undefined at the origin")

        U = self._resolved_flow_speed()
        return -1.5 * self.mu * U * self.sphere_radius * z / (r**3)

    def analytical_velocity_cartesian_3d(self, x):
        x = np.asarray(x, dtype=float)[:3]
        rho = float(np.linalg.norm(x[:2]))
        z = float(x[2])
        r = float(np.sqrt(rho**2 + z**2))
        if r < 1.0e-14:
            raise ValueError("The Stokes sphere analytical field is undefined at the origin")

        sin_theta = rho / r
        cos_theta = z / r
        U = self._resolved_flow_speed()
        a = self.sphere_radius

        u_r = U * cos_theta * (1.0 - 1.5 * a / r + 0.5 * (a**3) / (r**3))
        u_theta = -U * sin_theta * (1.0 - 0.75 * a / r - 0.25 * (a**3) / (r**3))
        u_rho = u_r * sin_theta + u_theta * cos_theta
        u_z = u_r * cos_theta - u_theta * sin_theta

        out = np.zeros(3, dtype=float)
        if rho > 1.0e-14:
            out[0] = u_rho * x[0] / rho
            out[1] = u_rho * x[1] / rho
        out[2] = u_z
        return out

    def _axisymmetric_cauchy_stress(self, x_meridian):
        from ddgclib.operators.stress import cauchy_stress

        x_meridian = np.asarray(x_meridian, dtype=float)[:2]
        x3 = np.array([x_meridian[0], 0.0, x_meridian[1]], dtype=float)
        h = self.fd_step * max(self.sphere_radius, 1.0)

        du = np.zeros((3, 3), dtype=float)
        for j in range(3):
            dx = np.zeros(3, dtype=float)
            dx[j] = h
            u_plus = self.analytical_velocity_cartesian_3d(x3 + dx)
            u_minus = self.analytical_velocity_cartesian_3d(x3 - dx)
            du[:, j] = (u_plus - u_minus) / (2.0 * h)

        p = self.analytical_pressure_cartesian_3d(x3)
        return cauchy_stress(p=p, du=du, mu=self.mu, dim=3)

    def _axisymmetric_cauchy_stress_reference(self, x_meridian):
        from ddgclib.operators.stress import cauchy_stress

        x_meridian = np.asarray(x_meridian, dtype=float)[:2]
        x3 = np.array([x_meridian[0], 0.0, x_meridian[1]], dtype=float)
        h = self.fd_step * max(self.sphere_radius, 1.0)

        du = np.zeros((3, 3), dtype=float)
        for j in range(3):
            dx = np.zeros(3, dtype=float)
            dx[j] = h
            u_p2 = self.analytical_velocity_cartesian_3d(x3 + 2.0 * dx)
            u_p1 = self.analytical_velocity_cartesian_3d(x3 + 1.0 * dx)
            u_m1 = self.analytical_velocity_cartesian_3d(x3 - 1.0 * dx)
            u_m2 = self.analytical_velocity_cartesian_3d(x3 - 2.0 * dx)
            du[:, j] = (-u_p2 + 8.0 * u_p1 - 8.0 * u_m1 + u_m2) / (12.0 * h)

        p = self.analytical_pressure_cartesian_3d(x3)
        return cauchy_stress(p=p, du=du, mu=self.mu, dim=3)

    def _integrate_axisymmetric_obstacle_force(self):
        """
        Integrate the full 3D Cartesian traction on the meridian contour.

        The 2D benchmark stores meridional coordinates ``(rho, z)``, but the
        Stokes stress itself is a 3D tensor.  Evaluating the full Cartesian
        stress on the meridian avoids treating cylindrical basis components as
        if they were planar Cartesian components.
        """
        force = np.zeros(2, dtype=float)
        s_q, w_q = self._line_quadrature()

        for elem in self.obstacle_elements:
            node_ids = np.asarray(elem, dtype=np.int64)
            nodes = self.points[node_ids, :2]
            use_arc_param = False
            theta_a = 0.0
            theta_b = 0.0
            arc_radius = float(self.sphere_radius)

            # For linear HC boundary facets, nodes lie on the sphere but the
            # segment is a chord. Parameterize by polar angle to integrate on
            # the actual spherical arc (geometry-consistent traction integral).
            if len(node_ids) == 2:
                rel0 = nodes[0] - self.center[:2]
                rel1 = nodes[1] - self.center[:2]
                r0 = float(np.linalg.norm(rel0))
                r1 = float(np.linalg.norm(rel1))
                tol_r = max(1.0e-8, 1.0e-6 * arc_radius)
                if abs(r0 - arc_radius) <= tol_r and abs(r1 - arc_radius) <= tol_r:
                    t0 = float(np.arctan2(rel0[0], rel0[1]))
                    t1 = float(np.arctan2(rel1[0], rel1[1]))
                    theta_a, theta_b = sorted((t0, t1))
                    use_arc_param = True

            for s, w in zip(s_q, w_q):
                N_lump = None
                if use_arc_param:
                    N_lump, _ = self._line_shape_functions(s, 2)
                    theta = 0.5 * ((1.0 - s) * theta_a + (1.0 + s) * theta_b)
                    dtheta_ds = 0.5 * (theta_b - theta_a)
                    x = self.center[:2] + arc_radius * np.array(
                        [np.sin(theta), np.cos(theta)],
                        dtype=float,
                    )
                    jac = arc_radius * abs(dtheta_ds)
                else:
                    N, dN = self._line_shape_functions(s, len(node_ids))
                    N_lump = N
                    x = np.tensordot(N, nodes, axes=(0, 0))
                    dx_ds = np.tensordot(dN, nodes, axes=(0, 0))
                    jac = np.linalg.norm(dx_ds)

                if jac < 1.0e-14:
                    continue

                rho = max(float(x[0]), 0.0)
                if rho < 1.0e-14:
                    continue

                x3 = np.array([x[0], 0.0, x[1]], dtype=float)
                normal3 = x3 / np.linalg.norm(x3)
                sigma = self._axisymmetric_cauchy_stress(x)
                sigma_ref = self._axisymmetric_cauchy_stress_reference(x)
                traction3 = sigma @ normal3
                traction3_ref = sigma_ref @ normal3
                weight = 2.0 * np.pi * rho * jac * w

                force[0] += traction3[0] * weight
                force[1] += traction3[2] * weight
                self.accumulate_vertex_drag_lumped_contribution(
                    node_ids=node_ids[: len(N_lump)],
                    shape_values=N_lump,
                    drag_computed=float(traction3[2] * weight),
                    drag_reference=float(traction3_ref[2] * weight),
                )

        return force

    def _integrate_3d_obstacle_force(self):
        """
        Integrate traction on the obstacle in 3D.

        When obstacle element nodes lie on the sphere (linear or high-order),
        quadrature points are mapped onto the true spherical surface and the
        surface Jacobian is evaluated through this spherical projection.
        This keeps HC and Gmsh traction integration geometrically consistent.
        """
        force = np.zeros(self.vector_dim, dtype=float)
        center3 = np.asarray(self.center[:3], dtype=float)
        sphere_r = float(self.sphere_radius)

        for elem in self.obstacle_elements:
            node_ids = np.asarray(elem, dtype=np.int64)
            nodes = self.points[node_ids, : self.vector_dim]
            n_nodes = len(node_ids)

            use_sphere_param = False
            u_nodes = None
            rel_nodes = nodes - center3
            rr = np.linalg.norm(rel_nodes, axis=1)
            tol_r = max(1.0e-8, 1.0e-6 * sphere_r)
            if np.all(np.abs(rr - sphere_r) <= tol_r):
                safe_rr = np.maximum(rr, 1.0e-14)
                u_nodes = rel_nodes / safe_rr[:, None]
                use_sphere_param = True

            for xi, eta, w in self._triangle_quadrature():
                N, dN_dxi, dN_deta = self._triangle_shape_functions(xi, eta, n_nodes)
                if use_sphere_param:
                    q = np.tensordot(N, u_nodes, axes=(0, 0))
                    q_norm = np.linalg.norm(q)
                    if q_norm < 1.0e-14:
                        continue

                    dq_dxi = np.tensordot(dN_dxi, u_nodes, axes=(0, 0))
                    dq_deta = np.tensordot(dN_deta, u_nodes, axes=(0, 0))

                    proj = np.eye(3, dtype=float) - np.outer(q, q) / (q_norm**2)
                    dx_dxi = sphere_r * (proj @ dq_dxi) / q_norm
                    dx_deta = sphere_r * (proj @ dq_deta) / q_norm
                    x = center3 + sphere_r * q / q_norm
                else:
                    x = np.tensordot(N, nodes, axes=(0, 0))
                    dx_dxi = np.tensordot(dN_dxi, nodes, axes=(0, 0))
                    dx_deta = np.tensordot(dN_deta, nodes, axes=(0, 0))

                jac_vec = np.cross(dx_dxi, dx_deta)
                jac = np.linalg.norm(jac_vec)
                if jac < 1.0e-14:
                    continue

                sigma = self.cauchy_stress_tensor(x)
                sigma_ref = self.cauchy_stress_tensor_reference(x)
                normal = self.obstacle_normal(x)
                traction = sigma @ normal
                traction_ref = sigma_ref @ normal
                force += traction * (jac * w)
                self.accumulate_vertex_drag_lumped_contribution(
                    node_ids=node_ids[: len(N)],
                    shape_values=N,
                    drag_computed=float(np.dot(traction, self.flow_direction) * (jac * w)),
                    drag_reference=float(np.dot(traction_ref, self.flow_direction) * (jac * w)),
                )

        return force

    def run(self):
        self.run_benchmark()
        return self


class StokesSphereBenchmark2D(_StokesSphereBenchmarkBase):
    """
    Axisymmetric meridional Stokes benchmark for a sphere.

    This is the 2D ``(rho, z)`` representation of the 3D creeping-flow
    solution, with a quadratic Lagrange line mesh on the semicircular
    obstacle boundary.
    """

    def __init__(self, sphere_radius=1.0, outer_radius=6.0,
                 mu=1.0, density_difference=1.0, gravity=1.0,
                 flow_speed=None, workdir="benchmarks_out/stokes_sphere_2d",
                 mesh_name="stokes_sphere_2d_axisymmetric.msh",
                 mesh_size_inner=None, mesh_size_outer=None,
                 remesh=False, method=None, complex_dtype="vv",
                 symm_nonobtuse=False, **kwargs):
        super().__init__(
            name="StokesSphere2D",
            method=method,
            complex_dtype=complex_dtype,
            dim=2,
            mu=mu,
            sphere_radius=sphere_radius,
            outer_radius=outer_radius,
            density_difference=density_difference,
            gravity=gravity,
            flow_speed=flow_speed,
            workdir=workdir,
            mesh_name=mesh_name,
            mesh_size_inner=mesh_size_inner,
            mesh_size_outer=mesh_size_outer,
            remesh=remesh,
            **kwargs,
        )
        self.symm_nonobtuse = bool(symm_nonobtuse)

    def generate_mesh(self):
        self._workdir_path.mkdir(parents=True, exist_ok=True)
        mesh_path = self._workdir_path / self.mesh_name
        if self.remesh or not mesh_path.exists():
            _generate_stokes_sphere_mesh_2d(
                mesh_path,
                sphere_radius=self.sphere_radius,
                outer_radius=self.outer_radius,
                mesh_size_inner=self.mesh_size_inner,
                mesh_size_outer=self.mesh_size_outer,
                mesh_order=self.mesh_order,
                symm_nonobtuse=self.symm_nonobtuse,
            )
        self.mesh_path = str(mesh_path)


class StokesSphereBenchmark3D(_StokesSphereBenchmarkBase):
    """
    Full 3D Stokes benchmark on a spherical shell.

    The obstacle boundary is meshed with quadratic triangles so the drag
    integral uses a true Lagrange surface instead of a piecewise-linear
    approximation.
    """

    def __init__(self, sphere_radius=1.0, outer_radius=6.0,
                 mu=1.0, density_difference=1.0, gravity=1.0,
                 flow_speed=None, workdir="benchmarks_out/stokes_sphere_3d",
                 mesh_name="stokes_sphere_3d.msh",
                 mesh_size_inner=None, mesh_size_outer=None,
                 remesh=False, method=None, complex_dtype="vv",
                 symm_nonobtuse=False, equal_edge=False, **kwargs):
        super().__init__(
            name="StokesSphere3D",
            method=method,
            complex_dtype=complex_dtype,
            dim=3,
            mu=mu,
            sphere_radius=sphere_radius,
            outer_radius=outer_radius,
            density_difference=density_difference,
            gravity=gravity,
            flow_speed=flow_speed,
            workdir=workdir,
            mesh_name=mesh_name,
            mesh_size_inner=mesh_size_inner,
            mesh_size_outer=mesh_size_outer,
            remesh=remesh,
            **kwargs,
        )
        self.symm_nonobtuse = bool(symm_nonobtuse)
        self.equal_edge = bool(equal_edge)

    def generate_mesh(self):
        self._workdir_path.mkdir(parents=True, exist_ok=True)
        mesh_path = self._workdir_path / self.mesh_name
        if self.remesh or not mesh_path.exists():
            _generate_stokes_sphere_mesh_3d(
                mesh_path,
                sphere_radius=self.sphere_radius,
                outer_radius=self.outer_radius,
                mesh_size_inner=self.mesh_size_inner,
                mesh_size_outer=self.mesh_size_outer,
                mesh_order=self.mesh_order,
                symm_nonobtuse=self.symm_nonobtuse,
                equal_edge=self.equal_edge,
            )
        self.mesh_path = str(mesh_path)


class _StokesSphereBenchmarkBaseHC(_StokesSphereBenchmarkBase):
    """
    HC-native Stokes benchmark: mesh generation independent from Gmsh.
    """

    def __init__(self, hc_refine=None, hc_inner_samples=None, hc_outer_samples=None,
                 symm_nonobtuse=False, equal_edge=False, **kwargs):
        super().__init__(**kwargs)
        default_refine = 3 if self.dim == 2 else 2
        self.hc_refine = int(hc_refine if hc_refine is not None else default_refine)
        self.hc_inner_samples = hc_inner_samples
        self.hc_outer_samples = hc_outer_samples
        self.symm_nonobtuse = bool(symm_nonobtuse)
        self.equal_edge = bool(equal_edge)

    def generate_mesh(self):
        # Kept for parity with the Gmsh path; no on-disk mesh is required here.
        self.mesh_path = f"hc://stokes_sphere_{self.dim}d_ref{self.hc_refine}"

    def load_mesh(self):
        points, simplices, obstacle, outer_boundary, axis = _generate_stokes_sphere_hc_mesh(
            dim=self.dim,
            sphere_radius=self.sphere_radius,
            outer_radius=self.outer_radius,
            hc_refine=self.hc_refine,
            hc_inner_samples=self.hc_inner_samples,
            hc_outer_samples=self.hc_outer_samples,
            symm_nonobtuse=self.symm_nonobtuse,
            equal_edge=self.equal_edge,
        )
        self.points = np.asarray(points, dtype=float)
        self.high_order_volume_cells = np.asarray(simplices, dtype=np.int64)
        self.volume_cells = np.asarray(simplices, dtype=np.int64)
        self.obstacle_elements = np.asarray(obstacle, dtype=np.int64)
        self.outer_boundary_elements = np.asarray(outer_boundary, dtype=np.int64)
        self.axis_elements = np.asarray(axis, dtype=np.int64) if self.dim == 2 else np.empty((0, 0), dtype=np.int64)
        self._update_boundary_bookkeeping()
        self.mesh_metadata["mesh_source"] = "hc"
        self.mesh_metadata["hc_refine"] = int(self.hc_refine)
        if self.symm_nonobtuse and self.equal_edge:
            variant = "symm_nonobtuse_equal_edge"
        elif self.symm_nonobtuse:
            variant = "symm_nonobtuse"
        elif self.equal_edge:
            variant = "equal_edge"
        else:
            variant = "default"
        self.mesh_metadata["mesh_variant"] = variant


class StokesSphereBenchmark2DHC(_StokesSphereBenchmarkBaseHC):
    def __init__(self, sphere_radius=1.0, outer_radius=6.0,
                 mu=1.0, density_difference=1.0, gravity=1.0,
                 flow_speed=None, workdir="benchmarks_out/stokes_sphere_2d_hc",
                 mesh_name="stokes_sphere_2d_hc",
                 hc_refine=3, hc_inner_samples=None, hc_outer_samples=None,
                 method=None, complex_dtype="vv", **kwargs):
        super().__init__(
            name="StokesSphere2DHC",
            method=method,
            complex_dtype=complex_dtype,
            dim=2,
            mu=mu,
            sphere_radius=sphere_radius,
            outer_radius=outer_radius,
            density_difference=density_difference,
            gravity=gravity,
            flow_speed=flow_speed,
            workdir=workdir,
            mesh_name=mesh_name,
            remesh=False,
            hc_refine=hc_refine,
            hc_inner_samples=hc_inner_samples,
            hc_outer_samples=hc_outer_samples,
            **kwargs,
        )


class StokesSphereBenchmark2DSymmNonobtuse(StokesSphereBenchmark2D):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_2d_symm_nonobtuse")
        kwargs.setdefault("mesh_name", "stokes_sphere_2d_symm_nonobtuse.msh")
        kwargs.setdefault("symm_nonobtuse", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere2D_SymmNonobtuse_Gmsh"


class StokesSphereBenchmark2DHCSymmNonobtuse(StokesSphereBenchmark2DHC):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_2d_hc_symm_nonobtuse")
        kwargs.setdefault("mesh_name", "stokes_sphere_2d_hc_symm_nonobtuse")
        kwargs.setdefault("symm_nonobtuse", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere2D_SymmNonobtuse_HC"


class StokesSphereBenchmark3DHC(_StokesSphereBenchmarkBaseHC):
    def __init__(self, sphere_radius=1.0, outer_radius=6.0,
                 mu=1.0, density_difference=1.0, gravity=1.0,
                 flow_speed=None, workdir="benchmarks_out/stokes_sphere_3d_hc",
                 mesh_name="stokes_sphere_3d_hc",
                 hc_refine=2, hc_inner_samples=None, hc_outer_samples=None,
                 method=None, complex_dtype="vv", **kwargs):
        super().__init__(
            name="StokesSphere3DHC",
            method=method,
            complex_dtype=complex_dtype,
            dim=3,
            mu=mu,
            sphere_radius=sphere_radius,
            outer_radius=outer_radius,
            density_difference=density_difference,
            gravity=gravity,
            flow_speed=flow_speed,
            workdir=workdir,
            mesh_name=mesh_name,
            remesh=False,
            hc_refine=hc_refine,
            hc_inner_samples=hc_inner_samples,
            hc_outer_samples=hc_outer_samples,
            **kwargs,
        )


class StokesSphereBenchmark3DSymmNonobtuse(StokesSphereBenchmark3D):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_3d_symm_nonobtuse")
        kwargs.setdefault("mesh_name", "stokes_sphere_3d_symm_nonobtuse.msh")
        kwargs.setdefault("symm_nonobtuse", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_SymmNonobtuse_Gmsh"


class StokesSphereBenchmark3DHCSymmNonobtuse(StokesSphereBenchmark3DHC):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_3d_hc_symm_nonobtuse")
        kwargs.setdefault("mesh_name", "stokes_sphere_3d_hc_symm_nonobtuse")
        kwargs.setdefault("symm_nonobtuse", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_SymmNonobtuse_HC"


class StokesSphereBenchmark3DEqualEdge(StokesSphereBenchmark3D):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_3d_equal_edge")
        kwargs.setdefault("mesh_name", "stokes_sphere_3d_equal_edge.msh")
        kwargs.setdefault("equal_edge", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_EqualEdge_Gmsh"


class StokesSphereBenchmark3DHCEqualEdge(StokesSphereBenchmark3DHC):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_3d_hc_equal_edge")
        kwargs.setdefault("mesh_name", "stokes_sphere_3d_hc_equal_edge")
        kwargs.setdefault("equal_edge", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_EqualEdge_HC"


def _integrate_obstacle_force_ddg_2d(bench):
    """
    DDG-based traction integration on the axisymmetric obstacle curve.

    DDG nodal gradients are reconstructed on obstacle endpoint nodes and
    linearly interpolated along each obstacle segment. The resulting traction
    is integrated with axisymmetric weight ``2*pi*rho``.
    """
    if bench.HC is None:
        raise RuntimeError("DDG complex is unavailable; cannot run DDG-based traction integration")

    try:
        from ddgclib.operators.stress import cauchy_stress, velocity_difference_tensor_pointwise
    except Exception as exc:
        raise RuntimeError(f"Failed to import DDG stress operators: {exc}") from exc

    ddg_index_map = getattr(bench, "ddg_index_map", {})
    ddg_vertices = getattr(bench, "ddg_vertices", [])
    if not ddg_index_map or not ddg_vertices:
        raise RuntimeError("DDG vertex mapping is missing")

    obstacle = np.asarray(bench.obstacle_elements, dtype=np.int64)
    if obstacle.size == 0:
        raise RuntimeError("Obstacle boundary elements are missing")

    # Use obstacle endpoints only for consistency with simplicial DDG nodes.
    edge_ids = obstacle[:, :2]
    node_ids = np.unique(edge_ids.ravel())

    du_nodes = {}
    p_nodes = {}
    missing_nodes = 0
    failed_nodes = 0
    for node_id in node_ids:
        local_idx = ddg_index_map.get(int(node_id))
        if local_idx is None:
            missing_nodes += 1
            continue
        v = ddg_vertices[local_idx]
        try:
            du = velocity_difference_tensor_pointwise(v, bench.HC, dim=2)
        except Exception:
            failed_nodes += 1
            continue
        du_nodes[int(node_id)] = np.asarray(du, dtype=float)
        p_nodes[int(node_id)] = float(v.p)

    if missing_nodes > 0:
        logger.warning("%s: %d obstacle DDG nodes were not mapped to HC vertices", bench.name, missing_nodes)
    if failed_nodes > 0:
        logger.warning("%s: velocity_difference_tensor_pointwise failed on %d obstacle nodes", bench.name, failed_nodes)
    if not du_nodes:
        raise RuntimeError("No DDG gradients were available on obstacle nodes")

    force = np.zeros(2, dtype=float)
    center2 = np.asarray(bench.center[:2], dtype=float)
    sphere_r = float(bench.sphere_radius)
    skipped_elements = 0
    s_q, w_q = bench._line_quadrature()

    for elem in obstacle:
        line_ids = np.asarray(elem[:2], dtype=np.int64)
        if not all(int(nid) in du_nodes for nid in line_ids):
            skipped_elements += 1
            continue

        nodes = bench.points[line_ids, :2]
        du_elem = np.stack([du_nodes[int(nid)] for nid in line_ids], axis=0)
        p_elem = np.asarray([p_nodes[int(nid)] for nid in line_ids], dtype=float)

        use_arc_param = False
        theta_a = 0.0
        theta_b = 0.0
        rel0 = nodes[0] - center2
        rel1 = nodes[1] - center2
        r0 = float(np.linalg.norm(rel0))
        r1 = float(np.linalg.norm(rel1))
        tol_r = max(1.0e-8, 1.0e-6 * sphere_r)
        if abs(r0 - sphere_r) <= tol_r and abs(r1 - sphere_r) <= tol_r:
            t0 = float(np.arctan2(rel0[0], rel0[1]))
            t1 = float(np.arctan2(rel1[0], rel1[1]))
            theta_a, theta_b = sorted((t0, t1))
            use_arc_param = True

        for s, w in zip(s_q, w_q):
            # Endpoint interpolation for DDG nodal data.
            N_edge, dN_edge = bench._line_shape_functions(s, 2)

            if use_arc_param:
                theta = 0.5 * ((1.0 - s) * theta_a + (1.0 + s) * theta_b)
                dtheta_ds = 0.5 * (theta_b - theta_a)
                x = center2 + sphere_r * np.array(
                    [np.sin(theta), np.cos(theta)],
                    dtype=float,
                )
                jac = sphere_r * abs(dtheta_ds)
            else:
                x = np.tensordot(N_edge, nodes, axes=(0, 0))
                dx_ds = np.tensordot(dN_edge, nodes, axes=(0, 0))
                jac = np.linalg.norm(dx_ds)

            if jac < 1.0e-14:
                continue

            rho = max(float(x[0]), 0.0)
            if rho < 1.0e-14:
                continue

            du_q = np.tensordot(N_edge, du_elem, axes=(0, 0))
            p_q = float(np.dot(N_edge, p_elem))
            sigma = cauchy_stress(p=p_q, du=du_q, mu=bench.mu, dim=2)
            # DDG 2D branch evaluates a 2D traction (rho-z plane), so the
            # reference stress must use the same 2x2 operator shape.
            sigma_ref = bench.cauchy_stress_tensor_reference(x)
            normal = bench.obstacle_normal(x)
            traction = sigma @ normal
            traction_ref = sigma_ref @ normal
            force += traction * (2.0 * np.pi * rho * jac * w)
            bench.accumulate_vertex_drag_lumped_contribution(
                node_ids=line_ids,
                shape_values=N_edge,
                drag_computed=float(traction[1] * (2.0 * np.pi * rho * jac * w)),
                drag_reference=float(traction_ref[1] * (2.0 * np.pi * rho * jac * w)),
            )

    if skipped_elements > 0:
        logger.warning("%s: skipped %d obstacle elements without DDG nodal data", bench.name, skipped_elements)

    return force


def _integrate_obstacle_force_ddg_3d(bench):
    """
    DDG-based traction integration on the spherical obstacle.

    The velocity gradient is reconstructed at obstacle corner vertices with
    ``velocity_difference_tensor_pointwise`` from :mod:`ddgclib.operators.stress`,
    then interpolated on each boundary triangle and integrated as
    ``sigma(du_ddg, p) · n``.
    """
    if bench.HC is None:
        raise RuntimeError("DDG complex is unavailable; cannot run DDG-based traction integration")

    try:
        from ddgclib.operators.stress import cauchy_stress, velocity_difference_tensor_pointwise
    except Exception as exc:
        raise RuntimeError(f"Failed to import DDG stress operators: {exc}") from exc

    ddg_index_map = getattr(bench, "ddg_index_map", {})
    ddg_vertices = getattr(bench, "ddg_vertices", [])
    if not ddg_index_map or not ddg_vertices:
        raise RuntimeError("DDG vertex mapping is missing")

    obstacle = np.asarray(bench.obstacle_elements, dtype=np.int64)
    if obstacle.size == 0:
        raise RuntimeError("Obstacle boundary elements are missing")

    # Use only triangle corner nodes for DDG reconstruction consistency with
    # the simplicial backbone used by compute_vd.
    corner_ids = obstacle[:, :3]
    node_ids = np.unique(corner_ids.ravel())

    du_nodes = {}
    p_nodes = {}
    missing_nodes = 0
    failed_nodes = 0
    for node_id in node_ids:
        local_idx = ddg_index_map.get(int(node_id))
        if local_idx is None:
            missing_nodes += 1
            continue
        v = ddg_vertices[local_idx]
        try:
            du = velocity_difference_tensor_pointwise(v, bench.HC, dim=3)
        except Exception:
            failed_nodes += 1
            continue
        du_nodes[int(node_id)] = np.asarray(du, dtype=float)
        p_nodes[int(node_id)] = float(v.p)

    if missing_nodes > 0:
        logger.warning("%s: %d obstacle DDG nodes were not mapped to HC vertices", bench.name, missing_nodes)
    if failed_nodes > 0:
        logger.warning("%s: velocity_difference_tensor_pointwise failed on %d obstacle nodes", bench.name, failed_nodes)
    if not du_nodes:
        raise RuntimeError("No DDG gradients were available on obstacle nodes")

    force = np.zeros(3, dtype=float)
    center3 = np.asarray(bench.center[:3], dtype=float)
    sphere_r = float(bench.sphere_radius)
    skipped_elements = 0

    for elem in obstacle:
        tri_ids = np.asarray(elem[:3], dtype=np.int64)
        if not all(int(nid) in du_nodes for nid in tri_ids):
            skipped_elements += 1
            continue

        nodes = bench.points[tri_ids, :3]

        use_sphere_param = False
        rel_nodes = nodes - center3
        rr = np.linalg.norm(rel_nodes, axis=1)
        tol_r = max(1.0e-8, 1.0e-6 * sphere_r)
        if np.all(np.abs(rr - sphere_r) <= tol_r):
            u_nodes = rel_nodes / rr[:, None]
            use_sphere_param = True
        else:
            u_nodes = None

        du_elem = np.stack([du_nodes[int(nid)] for nid in tri_ids], axis=0)
        p_elem = np.asarray([p_nodes[int(nid)] for nid in tri_ids], dtype=float)

        for xi, eta, w in bench._triangle_quadrature():
            N, dN_dxi, dN_deta = bench._triangle_shape_functions(xi, eta, 3)
            if use_sphere_param:
                q = np.tensordot(N, u_nodes, axes=(0, 0))
                q_norm = np.linalg.norm(q)
                if q_norm < 1.0e-14:
                    continue

                dq_dxi = np.tensordot(dN_dxi, u_nodes, axes=(0, 0))
                dq_deta = np.tensordot(dN_deta, u_nodes, axes=(0, 0))
                proj = np.eye(3, dtype=float) - np.outer(q, q) / (q_norm**2)
                dx_dxi = sphere_r * (proj @ dq_dxi) / q_norm
                dx_deta = sphere_r * (proj @ dq_deta) / q_norm
                x = center3 + sphere_r * q / q_norm
            else:
                x = np.tensordot(N, nodes, axes=(0, 0))
                dx_dxi = np.tensordot(dN_dxi, nodes, axes=(0, 0))
                dx_deta = np.tensordot(dN_deta, nodes, axes=(0, 0))

            jac_vec = np.cross(dx_dxi, dx_deta)
            jac = np.linalg.norm(jac_vec)
            if jac < 1.0e-14:
                continue

            du_q = np.tensordot(N, du_elem, axes=(0, 0))
            p_q = float(np.dot(N, p_elem))
            sigma = cauchy_stress(p=p_q, du=du_q, mu=bench.mu, dim=3)
            sigma_ref = bench.cauchy_stress_tensor_reference(x)
            normal = bench.obstacle_normal(x)
            traction = sigma @ normal
            traction_ref = sigma_ref @ normal
            force += traction * (jac * w)
            bench.accumulate_vertex_drag_lumped_contribution(
                node_ids=tri_ids,
                shape_values=N,
                drag_computed=float(np.dot(traction, bench.flow_direction) * (jac * w)),
                drag_reference=float(np.dot(traction_ref, bench.flow_direction) * (jac * w)),
            )

    if skipped_elements > 0:
        logger.warning("%s: skipped %d obstacle elements without DDG nodal data", bench.name, skipped_elements)

    return force


class StokesSphereBenchmark2DDDG(StokesSphereBenchmark2D):
    """
    2D axisymmetric Stokes benchmark using DDG gradient reconstruction in
    stress evaluation on the Gmsh mesh.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("ddg_enabled", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere2D_DDG_Gmsh"
        self.ddg_residuals_enabled = False

    def run_benchmark(self):
        self.generate_mesh()
        self.load_mesh()
        self.analytical_values()
        self.reset_vertex_drag_lumped_error_stats()
        self.build_ddg_complex()

        if self.HC is None:
            self.force_vector_computed = np.full(self.vector_dim, np.nan, dtype=float)
            self.drag_force_computed = np.nan
            self.force_error_abs = np.nan
            self.force_error_rel = np.nan
            self.finalize_vertex_drag_lumped_error_stats()
            return

        try:
            self.force_vector_computed = _integrate_obstacle_force_ddg_2d(self)
        except Exception as exc:
            logger.warning("%s: DDG obstacle integration failed (%s)", self.name, exc)
            self.force_vector_computed = np.full(self.vector_dim, np.nan, dtype=float)
            self.drag_force_computed = np.nan
            self.force_error_abs = np.nan
            self.force_error_rel = np.nan
            self.finalize_vertex_drag_lumped_error_stats()
            return

        self.drag_force_computed = float(np.dot(self.force_vector_computed, self.flow_direction))
        self.force_error_abs = float(abs(self.drag_force_computed - self.drag_force_analytical))
        if abs(self.drag_force_analytical) > 1.0e-30:
            self.force_error_rel = self.force_error_abs / abs(self.drag_force_analytical)
        self.finalize_vertex_drag_lumped_error_stats()


class StokesSphereBenchmark2DHCDDG(StokesSphereBenchmark2DHC):
    """
    2D axisymmetric Stokes benchmark using DDG gradient reconstruction in
    stress evaluation on the HC-native mesh.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("ddg_enabled", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere2D_DDG_HC"
        self.ddg_residuals_enabled = False

    def run_benchmark(self):
        self.generate_mesh()
        self.load_mesh()
        self.analytical_values()
        self.reset_vertex_drag_lumped_error_stats()
        self.build_ddg_complex()

        if self.HC is None:
            self.force_vector_computed = np.full(self.vector_dim, np.nan, dtype=float)
            self.drag_force_computed = np.nan
            self.force_error_abs = np.nan
            self.force_error_rel = np.nan
            self.finalize_vertex_drag_lumped_error_stats()
            return

        try:
            self.force_vector_computed = _integrate_obstacle_force_ddg_2d(self)
        except Exception as exc:
            logger.warning("%s: DDG obstacle integration failed (%s)", self.name, exc)
            self.force_vector_computed = np.full(self.vector_dim, np.nan, dtype=float)
            self.drag_force_computed = np.nan
            self.force_error_abs = np.nan
            self.force_error_rel = np.nan
            self.finalize_vertex_drag_lumped_error_stats()
            return

        self.drag_force_computed = float(np.dot(self.force_vector_computed, self.flow_direction))
        self.force_error_abs = float(abs(self.drag_force_computed - self.drag_force_analytical))
        if abs(self.drag_force_analytical) > 1.0e-30:
            self.force_error_rel = self.force_error_abs / abs(self.drag_force_analytical)
        self.finalize_vertex_drag_lumped_error_stats()


class StokesSphereBenchmark2DDDGSymmNonobtuse(StokesSphereBenchmark2DDDG):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_2d_symm_nonobtuse")
        kwargs.setdefault("mesh_name", "stokes_sphere_2d_symm_nonobtuse.msh")
        kwargs.setdefault("symm_nonobtuse", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere2D_DDG_SymmNonobtuse_Gmsh"


class StokesSphereBenchmark2DHCDDGSymmNonobtuse(StokesSphereBenchmark2DHCDDG):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_2d_hc_symm_nonobtuse")
        kwargs.setdefault("mesh_name", "stokes_sphere_2d_hc_symm_nonobtuse")
        kwargs.setdefault("symm_nonobtuse", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere2D_DDG_SymmNonobtuse_HC"


class StokesSphereBenchmark3DDDG(StokesSphereBenchmark3D):
    """
    3D Stokes benchmark using DDG gradient reconstruction in stress evaluation
    on the Gmsh mesh.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("ddg_enabled", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_DDG_Gmsh"
        self.ddg_residuals_enabled = False

    def run_benchmark(self):
        self.generate_mesh()
        self.load_mesh()
        self.analytical_values()
        self.reset_vertex_drag_lumped_error_stats()
        self.build_ddg_complex()

        if self.HC is None:
            self.force_vector_computed = np.full(self.vector_dim, np.nan, dtype=float)
            self.drag_force_computed = np.nan
            self.force_error_abs = np.nan
            self.force_error_rel = np.nan
            self.finalize_vertex_drag_lumped_error_stats()
            return

        try:
            self.force_vector_computed = _integrate_obstacle_force_ddg_3d(self)
        except Exception as exc:
            logger.warning("%s: DDG obstacle integration failed (%s)", self.name, exc)
            self.force_vector_computed = np.full(self.vector_dim, np.nan, dtype=float)
            self.drag_force_computed = np.nan
            self.force_error_abs = np.nan
            self.force_error_rel = np.nan
            self.finalize_vertex_drag_lumped_error_stats()
            return

        self.drag_force_computed = float(np.dot(self.force_vector_computed, self.flow_direction))
        self.force_error_abs = float(abs(self.drag_force_computed - self.drag_force_analytical))
        if abs(self.drag_force_analytical) > 1.0e-30:
            self.force_error_rel = self.force_error_abs / abs(self.drag_force_analytical)
        self.finalize_vertex_drag_lumped_error_stats()


class StokesSphereBenchmark3DHCDDG(StokesSphereBenchmark3DHC):
    """
    3D Stokes benchmark using DDG gradient reconstruction in stress evaluation
    on the HC-native mesh.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("ddg_enabled", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_DDG_HC"
        self.ddg_residuals_enabled = False

    def run_benchmark(self):
        self.generate_mesh()
        self.load_mesh()
        self.analytical_values()
        self.reset_vertex_drag_lumped_error_stats()
        self.build_ddg_complex()

        if self.HC is None:
            self.force_vector_computed = np.full(self.vector_dim, np.nan, dtype=float)
            self.drag_force_computed = np.nan
            self.force_error_abs = np.nan
            self.force_error_rel = np.nan
            self.finalize_vertex_drag_lumped_error_stats()
            return

        try:
            self.force_vector_computed = _integrate_obstacle_force_ddg_3d(self)
        except Exception as exc:
            logger.warning("%s: DDG obstacle integration failed (%s)", self.name, exc)
            self.force_vector_computed = np.full(self.vector_dim, np.nan, dtype=float)
            self.drag_force_computed = np.nan
            self.force_error_abs = np.nan
            self.force_error_rel = np.nan
            self.finalize_vertex_drag_lumped_error_stats()
            return

        self.drag_force_computed = float(np.dot(self.force_vector_computed, self.flow_direction))
        self.force_error_abs = float(abs(self.drag_force_computed - self.drag_force_analytical))
        if abs(self.drag_force_analytical) > 1.0e-30:
            self.force_error_rel = self.force_error_abs / abs(self.drag_force_analytical)
        self.finalize_vertex_drag_lumped_error_stats()


class StokesSphereBenchmark3DDDGSymmNonobtuse(StokesSphereBenchmark3DDDG):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_3d_symm_nonobtuse")
        kwargs.setdefault("mesh_name", "stokes_sphere_3d_symm_nonobtuse.msh")
        kwargs.setdefault("symm_nonobtuse", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_DDG_SymmNonobtuse_Gmsh"


class StokesSphereBenchmark3DHCDDGSymmNonobtuse(StokesSphereBenchmark3DHCDDG):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_3d_hc_symm_nonobtuse")
        kwargs.setdefault("mesh_name", "stokes_sphere_3d_hc_symm_nonobtuse")
        kwargs.setdefault("symm_nonobtuse", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_DDG_SymmNonobtuse_HC"


class StokesSphereBenchmark3DDDGEqualEdge(StokesSphereBenchmark3DDDG):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_3d_equal_edge")
        kwargs.setdefault("mesh_name", "stokes_sphere_3d_equal_edge.msh")
        kwargs.setdefault("equal_edge", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_DDG_EqualEdge_Gmsh"


class StokesSphereBenchmark3DHCDDGEqualEdge(StokesSphereBenchmark3DHCDDG):
    def __init__(self, **kwargs):
        kwargs.setdefault("workdir", "benchmarks_out/stokes_sphere_3d_hc_equal_edge")
        kwargs.setdefault("mesh_name", "stokes_sphere_3d_hc_equal_edge")
        kwargs.setdefault("equal_edge", True)
        super().__init__(**kwargs)
        self.name = "StokesSphere3D_DDG_EqualEdge_HC"
