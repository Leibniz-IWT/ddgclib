import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from abc import ABC, abstractmethod

import numpy as np
from ddgclib._method_wrappers import (
    Curvature_i, Curvature_ijk,
    Area_i, Area_ijk, Area,
    Volume
)

class GeometryBenchmarkBase:
    """
    Base class for benchmarking geometric quantities on simplicial complexes.
    """

    def __init__(self, name="unnamed", method=None, complex_dtype="vf"):
        self.name = name
        self.complex_dtype = complex_dtype
        self.method = method or {}

        self.points = None
        self.simplices = None

        self.local_area = None
        self.H_computed = None
        self.H_analytical = None

        self.area_computed = None
        self.area_analytical = None

        self.volume_computed = None
        self.volume_analytical = None

        # Set methods
        try:
            self.curvature_i = Curvature_i(self.method.get("curvature_i_method", "default"))
        except NotImplementedError as e:
            logger.warning(f"curvature_i method not implemented: {e}")
            self.curvature_i = None

        try:
            self.curvature_ijk = Curvature_ijk(self.method.get("curvature_ijk_method", "default"))
        except NotImplementedError as e:
            logger.warning(f"curvature_ijk method not implemented: {e}")
            self.curvature_ijk = None

        try:
            self.area_i = Area_i(self.method.get("area_i_method", "default"))
        except NotImplementedError as e:
            logger.warning(f"area_i method not implemented: {e}")
            self.area_i = None

        try:
            self.area_ijk = Area_ijk(self.method.get("area_ijk_method", "default"))
        except NotImplementedError as e:
            logger.warning(f"area_ijk method not implemented: {e}")
            self.area_ijk = None

        try:
            self.area = Area(self.method.get("area_method", "default"))
        except NotImplementedError as e:
            logger.warning(f"area method not implemented: {e}")
            self.area = None

        try:
            self.volume = Volume(self.method.get("volume_method", "default"))
        except NotImplementedError as e:
            logger.warning(f"volume method not implemented: {e}")
            self.volume = None
    def generate_mesh(self):
        raise NotImplementedError

    def compute_surface_area(self):
        HC = (self.points, self.simplices) if self.complex_dtype == "vf" else self.points
        self.area_computed = self.area(HC, complex_dtype=self.complex_dtype)

    def compute_volume(self):
        if self.complex_dtype == "vf":
            HC = (self.points, self.simplices)
            vol_kwargs = {k: v for k, v in self.method.items()
                        if not k.endswith("_method")}
            self.volume_computed = self.volume(HC, complex_dtype=self.complex_dtype, **vol_kwargs)


    def compute_curvature(self):
        HC = (self.points, self.simplices) if self.complex_dtype == "vf" else self.points
        if self.complex_dtype == "vv":
            self.H_computed = self.curvature_i(HC, complex_dtype=self.complex_dtype)
        elif self.complex_dtype == "vf":
            # Placeholder for future use:
            # self.H_computed = self.curvature_ijk(HC, complex_dtype=self.complex_dtype)
            pass

    def analytical_values(self):
        raise NotImplementedError

    def run_benchmark(self):
        self.generate_mesh()
        self.compute_surface_area()
        self.compute_volume()
        self.compute_curvature()
        self.analytical_values()

        if self.complex_dtype == "vv":
            # Placeholder: loop over all vertices
            for i in range(len(self.points)):
                pass  # per-vertex error logic
        elif self.complex_dtype == "vf":
            # Placeholder: loop over all simplices
            for s in self.simplices:
                pass  # per-simplex error logic

    def summary(self):
        mean_error_H = np.mean(np.abs(self.H_computed - self.H_analytical))
        std_error_H = np.std(np.abs(self.H_computed - self.H_analytical))
        return {
            "Surface Area": (self.area_computed, self.area_analytical),
            "Volume": (self.volume_computed, self.volume_analytical),
            "Curvature Mean Error": (mean_error_H, 0.0),
            "Curvature Std Error": (std_error_H, 0.0)
        }


def run_all_benchmarks(benchmark_classes, method=None, complex_dtype="vf"):
    """
    Runs all provided benchmark cases with specified method implementations.

    Parameters
    ----------
    benchmark_classes : list of GeometryBenchmarkBase subclasses
        List of benchmark classes to instantiate and run.
    method : dict, optional
        Dictionary specifying method names for each evaluator. Example:
        {
            "curvature_i_method": "laplace-beltrami",
            "area_i_method": "default",
            "area_ijk_method": "default",
            "area_method": "default",
            "volume_method": "default",
        }
        If not provided, all evaluators default to their "default" method.
    complex_dtype : str
        Type of mesh structure. Either 'vf' (vertex-face) or 'vv' (vertex-vertex).
    """
    method = method or {}
    for BenchmarkClass in benchmark_classes:
        bench = BenchmarkClass(method=method, complex_dtype=complex_dtype)
        bench.run_benchmark()
        print(f"Benchmark: {bench.name}")
        for k, (v_comp, v_ana) in bench.summary().items():
            print(f"  {k}: computed={v_comp:.6f}, analytical={v_ana:.6f}")


class FlowBenchmarkBase(ABC):
    """
    Base class for flow benchmarks on curved Gmsh meshes.

    The main use-case is an exact-field validation:

    1. Build or load a volumetric mesh of the fluid domain.
    2. Keep the obstacle boundary as a high-order Lagrange mesh.
    3. Evaluate the analytical velocity and pressure fields.
    4. Integrate the Cauchy traction on the obstacle boundary.
    5. Compare the integrated force against the analytical drag / gravity.

    When ``hyperct`` is available, the class also constructs a simplicial
    complex from the low-order simplex backbone and evaluates a DDG stress
    residual using :mod:`ddgclib.operators.stress`.
    """

    volume_physical_id = 1
    obstacle_physical_id = 2
    outer_boundary_physical_id = 3
    axis_physical_id = 4

    def __init__(
        self,
        name="unnamed_flow",
        method=None,
        complex_dtype="vv",
        dim=3,
        mu=1.0,
        sphere_radius=1.0,
        outer_radius=6.0,
        density_difference=1.0,
        gravity=1.0,
        flow_speed=None,
        mesh_order=2,
        fd_step=1.0e-6,
        ddg_enabled=None,
    ):
        if dim not in (2, 3):
            raise ValueError("FlowBenchmarkBase currently supports dim=2 or dim=3")

        self.name = name
        self.method = method or {}
        self.complex_dtype = complex_dtype
        self.dim = dim
        self.vector_dim = 2 if dim == 2 else 3

        self.mu = float(mu)
        self.sphere_radius = float(sphere_radius)
        self.outer_radius = float(outer_radius)
        self.density_difference = float(density_difference)
        self.gravity = float(gravity)
        self.flow_speed = None if flow_speed is None else float(flow_speed)
        self.mesh_order = int(mesh_order)
        self.fd_step = float(fd_step)
        if ddg_enabled is None:
            # 3D `hyperct.compute_vd` on large generic tetra meshes can fail
            # because it reconstructs local dual topology from neighbor
            # intersections. Keep DDG-on by default for 2D and DDG-off for 3D.
            self.ddg_enabled = (dim == 2)
        else:
            self.ddg_enabled = bool(ddg_enabled)

        self.center = np.zeros(self.vector_dim, dtype=float)
        self.flow_axis = self.vector_dim - 1
        self.flow_direction = np.zeros(self.vector_dim, dtype=float)
        self.flow_direction[self.flow_axis] = 1.0

        self.mesh_path = None
        self.mesh_metadata = {}

        self.points = None
        self.high_order_volume_cells = None
        self.volume_cells = None
        self.obstacle_elements = None
        self.outer_boundary_elements = None
        self.axis_elements = None

        self.boundary_node_ids = set()
        self.obstacle_node_ids = set()
        self.outer_boundary_node_ids = set()
        self.axis_node_ids = set()

        self.HC = None
        self.bV = set()
        self.ddg_vertices = []
        self.ddg_index_map = {}
        self.ddg_point_indices = np.empty((0,), dtype=np.int64)
        self.ddg_boundary_local_ids = set()

        self.force_vector_computed = None
        self.force_vector_analytical = None
        self.drag_force_computed = np.nan
        self.drag_force_analytical = np.nan
        self.gravity_force_analytical = np.nan
        self.force_error_abs = np.nan
        self.force_error_rel = np.nan

        self.ddg_residual_median = np.nan
        self.ddg_residual_max = np.nan
        self.ddg_residual_count = 0
        self.ddg_residuals_enabled = True

        self.vertex_drag_rel_error_mean = np.nan
        self.vertex_drag_rel_error_max = np.nan
        self.vertex_drag_valid_count = 0
        self._vertex_drag_computed_by_node = {}
        self._vertex_drag_reference_by_node = {}

    @abstractmethod
    def generate_mesh(self):
        """Create or select ``self.mesh_path``."""

    @abstractmethod
    def analytical_velocity(self, x):
        """Return the analytical velocity vector at ``x``."""

    @abstractmethod
    def analytical_pressure(self, x):
        """Return the analytical pressure at ``x``."""

    @abstractmethod
    def analytical_values(self):
        """Populate analytical drag / gravity reference values."""

    @property
    def _high_order_volume_cell_types(self):
        return ("triangle6",) if self.dim == 2 else ("tetra10",)

    @property
    def _low_order_volume_cell_types(self):
        return ("triangle",) if self.dim == 2 else ("tetra",)

    @property
    def _high_order_boundary_cell_types(self):
        return ("line3",) if self.dim == 2 else ("triangle6",)

    @property
    def _low_order_boundary_cell_types(self):
        return ("line",) if self.dim == 2 else ("triangle",)

    @staticmethod
    def _extract_cell_blocks(mesh, cell_types, physical_id=None):
        if isinstance(cell_types, str):
            cell_types = (cell_types,)

        blocks = []
        physical_blocks = mesh.cell_data.get("gmsh:physical")
        for idx, block in enumerate(mesh.cells):
            if block.type not in cell_types:
                continue

            data = np.asarray(block.data, dtype=np.int64)
            if data.ndim != 2 or data.size == 0:
                continue

            if physical_id is not None:
                if physical_blocks is None or idx >= len(physical_blocks):
                    continue
                tags = np.asarray(physical_blocks[idx], dtype=np.int64)
                data = data[tags == physical_id]

            if data.size:
                blocks.append(data)

        if not blocks:
            return np.empty((0, 0), dtype=np.int64)
        return np.concatenate(blocks, axis=0)

    def _update_boundary_bookkeeping(self):
        """Refresh boundary-node sets and mesh metadata."""
        self.obstacle_node_ids = set(np.asarray(self.obstacle_elements, dtype=np.int64).ravel().tolist())
        self.outer_boundary_node_ids = set(np.asarray(self.outer_boundary_elements, dtype=np.int64).ravel().tolist())
        self.axis_node_ids = set(np.asarray(self.axis_elements, dtype=np.int64).ravel().tolist())
        self.boundary_node_ids = (
            self.obstacle_node_ids | self.outer_boundary_node_ids | self.axis_node_ids
        )

        self.mesh_metadata = {
            "n_points": int(len(self.points)) if self.points is not None else 0,
            "n_volume_cells": int(len(self.volume_cells)) if self.volume_cells is not None else 0,
            "n_obstacle_elements": int(len(self.obstacle_elements)) if self.obstacle_elements is not None else 0,
            "n_outer_boundary_elements": int(len(self.outer_boundary_elements))
            if self.outer_boundary_elements is not None else 0,
            "n_axis_elements": int(len(self.axis_elements)) if self.axis_elements is not None else 0,
        }

    def load_mesh(self):
        """Load the Gmsh mesh and extract the simplex backbone and boundary groups."""
        if self.mesh_path is None:
            raise ValueError("generate_mesh() must set self.mesh_path before load_mesh()")

        try:
            import meshio
        except ImportError as exc:
            raise ImportError("meshio is required to load Stokes benchmark meshes") from exc

        mesh = meshio.read(self.mesh_path)
        points = np.asarray(mesh.points, dtype=float)
        if points.shape[1] < self.vector_dim:
            pad = np.zeros((len(points), self.vector_dim - points.shape[1]), dtype=float)
            points = np.c_[points, pad]
        self.points = np.asarray(points[:, :self.vector_dim], dtype=float)

        high_order_volume = self._extract_cell_blocks(
            mesh, self._high_order_volume_cell_types, physical_id=self.volume_physical_id
        )
        low_order_volume = self._extract_cell_blocks(
            mesh, self._low_order_volume_cell_types, physical_id=self.volume_physical_id
        )
        if high_order_volume.size:
            self.high_order_volume_cells = high_order_volume
            self.volume_cells = np.asarray(
                high_order_volume[:, : self.vector_dim + 1], dtype=np.int64
            )
        else:
            self.high_order_volume_cells = low_order_volume
            self.volume_cells = low_order_volume

        obstacle = self._extract_cell_blocks(
            mesh, self._high_order_boundary_cell_types, physical_id=self.obstacle_physical_id
        )
        if not obstacle.size:
            obstacle = self._extract_cell_blocks(
                mesh, self._low_order_boundary_cell_types, physical_id=self.obstacle_physical_id
            )
        self.obstacle_elements = obstacle

        outer = self._extract_cell_blocks(
            mesh, self._high_order_boundary_cell_types, physical_id=self.outer_boundary_physical_id
        )
        if not outer.size:
            outer = self._extract_cell_blocks(
                mesh, self._low_order_boundary_cell_types, physical_id=self.outer_boundary_physical_id
            )
        self.outer_boundary_elements = outer

        if self.dim == 2:
            axis = self._extract_cell_blocks(
                mesh, self._high_order_boundary_cell_types, physical_id=self.axis_physical_id
            )
            if not axis.size:
                axis = self._extract_cell_blocks(
                    mesh, self._low_order_boundary_cell_types, physical_id=self.axis_physical_id
                )
            self.axis_elements = axis
        else:
            self.axis_elements = np.empty((0, 0), dtype=np.int64)

        self._update_boundary_bookkeeping()

    def obstacle_normal(self, x):
        """Outward unit normal of the sphere, pointing from the solid into the fluid."""
        vec = np.asarray(x, dtype=float)[: self.vector_dim] - self.center
        norm = np.linalg.norm(vec)
        if norm < 1.0e-14:
            return np.zeros(self.vector_dim, dtype=float)
        return vec / norm

    def velocity_gradient(self, x):
        """
        Finite-difference Jacobian of the analytical velocity field.

        The Stokes benchmark uses this with :func:`ddgclib.operators.stress.cauchy_stress`
        so the constitutive relation is still evaluated by ddgclib.
        """
        x = np.asarray(x, dtype=float)[: self.vector_dim]
        h = self.fd_step * max(self.sphere_radius, 1.0)
        grad = np.zeros((self.vector_dim, self.vector_dim), dtype=float)

        for j in range(self.vector_dim):
            dx = np.zeros(self.vector_dim, dtype=float)
            dx[j] = h
            u_plus = np.asarray(self.analytical_velocity(x + dx), dtype=float)
            u_minus = np.asarray(self.analytical_velocity(x - dx), dtype=float)
            grad[:, j] = (u_plus - u_minus) / (2.0 * h)
        return grad

    def cauchy_stress_tensor(self, x):
        """Pointwise Newtonian Cauchy stress via ddgclib."""
        from ddgclib.operators.stress import cauchy_stress

        du = self.velocity_gradient(x)
        p = float(self.analytical_pressure(x))
        return cauchy_stress(p=p, du=du, mu=self.mu, dim=self.vector_dim)

    def velocity_gradient_reference(self, x):
        """
        Higher-order finite-difference Jacobian of the analytical velocity.

        This is used as a reference operator for nodal drag-lumping diagnostics.
        """
        x = np.asarray(x, dtype=float)[: self.vector_dim]
        h = self.fd_step * max(self.sphere_radius, 1.0)
        grad = np.zeros((self.vector_dim, self.vector_dim), dtype=float)

        for j in range(self.vector_dim):
            dx = np.zeros(self.vector_dim, dtype=float)
            dx[j] = h
            u_p2 = np.asarray(self.analytical_velocity(x + 2.0 * dx), dtype=float)
            u_p1 = np.asarray(self.analytical_velocity(x + 1.0 * dx), dtype=float)
            u_m1 = np.asarray(self.analytical_velocity(x - 1.0 * dx), dtype=float)
            u_m2 = np.asarray(self.analytical_velocity(x - 2.0 * dx), dtype=float)
            grad[:, j] = (-u_p2 + 8.0 * u_p1 - 8.0 * u_m1 + u_m2) / (12.0 * h)
        return grad

    def cauchy_stress_tensor_reference(self, x):
        """Reference Cauchy stress using higher-order analytical-field derivatives."""
        from ddgclib.operators.stress import cauchy_stress

        du = self.velocity_gradient_reference(x)
        p = float(self.analytical_pressure(x))
        return cauchy_stress(p=p, du=du, mu=self.mu, dim=self.vector_dim)

    def reset_vertex_drag_lumped_error_stats(self):
        self.vertex_drag_rel_error_mean = np.nan
        self.vertex_drag_rel_error_max = np.nan
        self.vertex_drag_valid_count = 0
        self._vertex_drag_computed_by_node = {}
        self._vertex_drag_reference_by_node = {}

    def accumulate_vertex_drag_lumped_contribution(self, node_ids, shape_values, drag_computed, drag_reference):
        node_ids = np.asarray(node_ids, dtype=np.int64)
        shape_values = np.asarray(shape_values, dtype=float)
        if node_ids.size == 0 or shape_values.size == 0:
            return
        if node_ids.size != shape_values.size:
            return

        d_comp = float(drag_computed)
        d_ref = float(drag_reference)
        for nid, Ni in zip(node_ids, shape_values):
            key = int(nid)
            w = float(Ni)
            self._vertex_drag_computed_by_node[key] = self._vertex_drag_computed_by_node.get(key, 0.0) + w * d_comp
            self._vertex_drag_reference_by_node[key] = self._vertex_drag_reference_by_node.get(key, 0.0) + w * d_ref

    def finalize_vertex_drag_lumped_error_stats(self):
        keys = sorted(
            set(self._vertex_drag_computed_by_node.keys())
            | set(self._vertex_drag_reference_by_node.keys())
        )
        if not keys:
            self.vertex_drag_rel_error_mean = np.nan
            self.vertex_drag_rel_error_max = np.nan
            self.vertex_drag_valid_count = 0
            return

        comp = np.asarray(
            [self._vertex_drag_computed_by_node.get(k, 0.0) for k in keys],
            dtype=float,
        )
        ref = np.asarray(
            [self._vertex_drag_reference_by_node.get(k, 0.0) for k in keys],
            dtype=float,
        )

        avg_scale = abs(float(self.drag_force_analytical)) / max(len(keys), 1)
        ref_tol = max(1.0e-30, 1.0e-12 * max(avg_scale, 1.0))
        valid = np.abs(ref) > ref_tol

        if not np.any(valid):
            self.vertex_drag_rel_error_mean = np.nan
            self.vertex_drag_rel_error_max = np.nan
            self.vertex_drag_valid_count = 0
            return

        rel = np.abs(comp[valid] - ref[valid]) / np.abs(ref[valid])
        self.vertex_drag_rel_error_mean = float(np.mean(rel))
        self.vertex_drag_rel_error_max = float(np.max(rel))
        self.vertex_drag_valid_count = int(np.count_nonzero(valid))

    @staticmethod
    def _line_shape_functions(s, n_nodes):
        if n_nodes == 2:
            N = np.array([(1.0 - s) * 0.5, (1.0 + s) * 0.5], dtype=float)
            dN = np.array([-0.5, 0.5], dtype=float)
            return N, dN
        if n_nodes == 3:
            # Gmsh/meshio order line3 nodes as [end0, end1, midpoint].
            N = np.array(
                [
                    0.5 * s * (s - 1.0),
                    0.5 * s * (s + 1.0),
                    1.0 - s**2,
                ],
                dtype=float,
            )
            dN = np.array([s - 0.5, s + 0.5, -2.0 * s], dtype=float)
            return N, dN
        raise NotImplementedError(f"Unsupported line element with {n_nodes} nodes")

    @staticmethod
    def _triangle_shape_functions(xi, eta, n_nodes):
        zeta = 1.0 - xi - eta
        if n_nodes == 3:
            N = np.array([zeta, xi, eta], dtype=float)
            dN_dxi = np.array([-1.0, 1.0, 0.0], dtype=float)
            dN_deta = np.array([-1.0, 0.0, 1.0], dtype=float)
            return N, dN_dxi, dN_deta

        if n_nodes == 6:
            N = np.array(
                [
                    zeta * (2.0 * zeta - 1.0),
                    xi * (2.0 * xi - 1.0),
                    eta * (2.0 * eta - 1.0),
                    4.0 * xi * zeta,
                    4.0 * xi * eta,
                    4.0 * eta * zeta,
                ],
                dtype=float,
            )
            dN_dxi = np.array(
                [
                    1.0 - 4.0 * zeta,
                    4.0 * xi - 1.0,
                    0.0,
                    4.0 * (zeta - xi),
                    4.0 * eta,
                    -4.0 * eta,
                ],
                dtype=float,
            )
            dN_deta = np.array(
                [
                    1.0 - 4.0 * zeta,
                    0.0,
                    4.0 * eta - 1.0,
                    -4.0 * xi,
                    4.0 * xi,
                    4.0 * (zeta - eta),
                ],
                dtype=float,
            )
            return N, dN_dxi, dN_deta

        raise NotImplementedError(f"Unsupported triangle element with {n_nodes} nodes")

    @staticmethod
    def _line_quadrature():
        q = np.sqrt(3.0 / 5.0)
        return (
            np.array([-q, 0.0, q], dtype=float),
            np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float),
        )

    @staticmethod
    def _triangle_quadrature():
        return [
            (1.0 / 3.0, 1.0 / 3.0, 0.112500000000000),
            (0.470142064105115, 0.470142064105115, 0.066197076394253),
            (0.470142064105115, 0.059715871789770, 0.066197076394253),
            (0.059715871789770, 0.470142064105115, 0.066197076394253),
            (0.101286507323456, 0.101286507323456, 0.0629695902724135),
            (0.101286507323456, 0.797426985353087, 0.0629695902724135),
            (0.797426985353087, 0.101286507323456, 0.0629695902724135),
        ]

    def integrate_obstacle_force(self):
        """Integrate ``sigma · n`` on the obstacle boundary."""
        if self.obstacle_elements is None or self.obstacle_elements.size == 0:
            raise ValueError("Obstacle boundary elements are missing; load_mesh() first")

        if self.dim == 2:
            return self._integrate_axisymmetric_obstacle_force()
        if self.dim == 3:
            return self._integrate_3d_obstacle_force()
        raise NotImplementedError(f"Unsupported dimension {self.dim}")

    def _integrate_axisymmetric_obstacle_force(self):
        force = np.zeros(self.vector_dim, dtype=float)
        s_q, w_q = self._line_quadrature()

        for elem in self.obstacle_elements:
            node_ids = np.asarray(elem, dtype=np.int64)
            nodes = self.points[node_ids, : self.vector_dim]
            for s, w in zip(s_q, w_q):
                N, dN = self._line_shape_functions(s, len(node_ids))
                x = np.tensordot(N, nodes, axes=(0, 0))
                dx_ds = np.tensordot(dN, nodes, axes=(0, 0))
                jac = np.linalg.norm(dx_ds)
                if jac < 1.0e-14:
                    continue

                rho = max(float(x[0] - self.center[0]), 0.0)
                if rho < 1.0e-14:
                    continue

                sigma = self.cauchy_stress_tensor(x)
                normal = self.obstacle_normal(x)
                traction = sigma @ normal
                force += traction * (2.0 * np.pi * rho * jac * w)

        return force

    def _integrate_3d_obstacle_force(self):
        force = np.zeros(self.vector_dim, dtype=float)
        for elem in self.obstacle_elements:
            node_ids = np.asarray(elem, dtype=np.int64)
            nodes = self.points[node_ids, : self.vector_dim]
            for xi, eta, w in self._triangle_quadrature():
                N, dN_dxi, dN_deta = self._triangle_shape_functions(xi, eta, len(node_ids))
                x = np.tensordot(N, nodes, axes=(0, 0))
                dx_dxi = np.tensordot(dN_dxi, nodes, axes=(0, 0))
                dx_deta = np.tensordot(dN_deta, nodes, axes=(0, 0))
                jac_vec = np.cross(dx_dxi, dx_deta)
                jac = np.linalg.norm(jac_vec)
                if jac < 1.0e-14:
                    continue

                sigma = self.cauchy_stress_tensor(x)
                normal = self.obstacle_normal(x)
                traction = sigma @ normal
                force += traction * (jac * w)

        return force

    def build_ddg_complex(self):
        """
        Build a low-order ``hyperct.Complex`` from the simplex backbone.

        This is optional.  When ``hyperct`` is unavailable the benchmark still
        runs and the curved traction integral remains valid.
        """
        if self.volume_cells is None or self.volume_cells.size == 0:
            logger.warning("%s: no volume cells found; skipping DDG complex build", self.name)
            return

        try:
            from hyperct import Complex
            from hyperct.ddg import compute_vd
        except ImportError as exc:
            logger.warning("%s: hyperct is unavailable (%s); skipping DDG residuals", self.name, exc)
            return

        all_coords = np.asarray(self.points[:, : self.vector_dim], dtype=float)
        simplex_size = self.vector_dim + 1
        simplices = np.asarray(self.volume_cells, dtype=np.int64)
        if simplices.ndim != 2 or simplices.shape[0] == 0:
            logger.warning("%s: no simplices available for DDG build", self.name)
            return

        simplices = np.asarray(simplices[:, :simplex_size], dtype=np.int64)
        active_ids = np.unique(simplices.ravel())
        if active_ids.size == 0:
            logger.warning("%s: no active vertices in simplices for DDG build", self.name)
            return

        coords = np.asarray(all_coords[active_ids], dtype=float)
        index_map = {int(orig): int(local) for local, orig in enumerate(active_ids)}

        def _make_complex():
            HC_local = Complex(self.vector_dim)
            verts_local = [HC_local.V[tuple(coord)] for coord in coords]
            for simplex in simplices:
                simplex = np.asarray([index_map[int(node)] for node in simplex], dtype=np.int64)
                for i in range(simplex_size):
                    for j in range(i + 1, simplex_size):
                        verts_local[simplex[i]].connect(verts_local[simplex[j]])

            bV_local = set()
            boundary_local_ids_local = set()
            for idx, v in enumerate(verts_local):
                x = coords[idx]
                v.u = np.asarray(self.analytical_velocity(x), dtype=float)
                v.p = float(self.analytical_pressure(x))
                v.m = 1.0
                orig_idx = int(active_ids[idx])
                v.boundary = orig_idx in self.boundary_node_ids
                if v.boundary:
                    bV_local.add(v)
                    boundary_local_ids_local.add(idx)
            return HC_local, verts_local, bV_local, boundary_local_ids_local

        backend_candidates = [None]
        try:
            from hyperct._backend import get_backend  # type: ignore
            backend_candidates.append(get_backend("numpy"))
        except Exception:
            pass

        attempts = []
        for method in ("barycentric", "circumcentric"):
            for backend in backend_candidates:
                attempts.append((method, backend))

        HC = None
        verts = []
        bV = set()
        boundary_local_ids = set()
        fail_msgs = []
        for method, backend in attempts:
            HC_try, verts_try, bV_try, boundary_ids_try = _make_complex()
            kwargs = {"method": method}
            if backend is not None:
                kwargs["backend"] = backend
            backend_name = "default" if backend is None else "numpy"
            try:
                compute_vd(HC_try, **kwargs)
            except TypeError:
                try:
                    compute_vd(HC_try, cdist=1.0e-10, **kwargs)
                except Exception as exc:
                    fail_msgs.append(f"{method}/{backend_name}: {exc}")
                    continue
            except Exception as exc:
                fail_msgs.append(f"{method}/{backend_name}: {exc}")
                continue

            HC = HC_try
            verts = verts_try
            bV = bV_try
            boundary_local_ids = boundary_ids_try
            break

        if HC is None:
            detail = "; ".join(fail_msgs) if fail_msgs else "unknown error"
            logger.warning("%s: compute_vd failed (%s); skipping DDG residuals", self.name, detail)
            return

        self.HC = HC
        self.bV = bV
        self.ddg_vertices = verts
        self.ddg_index_map = index_map
        self.ddg_point_indices = np.asarray(active_ids, dtype=np.int64)
        self.ddg_boundary_local_ids = boundary_local_ids
        if self.ddg_residuals_enabled:
            self.compute_ddg_residuals()

    def compute_ddg_residuals(self):
        """Evaluate DDG stress-force residuals on interior vertices."""
        if self.HC is None:
            return

        try:
            from ddgclib.operators.stress import stress_force
        except Exception as exc:
            logger.warning("%s: failed to import stress_force (%s)", self.name, exc)
            return

        boundary_local_ids = getattr(self, "ddg_boundary_local_ids", set())
        residuals = []
        fail_count = 0
        for idx, v in enumerate(self.HC.V):
            if idx in boundary_local_ids:
                continue
            try:
                F = stress_force(v, dim=self.vector_dim, mu=self.mu, HC=self.HC)
            except Exception as exc:
                fail_count += 1
                if fail_count <= 5:
                    logger.warning("%s: stress_force failed at vertex %d (%s)", self.name, idx, exc)
                continue
            residuals.append(float(np.linalg.norm(F)))

        if fail_count > 5:
            logger.warning(
                "%s: stress_force failed at %d additional vertices",
                self.name,
                fail_count - 5,
            )

        if residuals:
            residuals = np.asarray(residuals, dtype=float)
            self.ddg_residual_count = int(residuals.size)
            self.ddg_residual_median = float(np.median(residuals))
            self.ddg_residual_max = float(np.max(residuals))

    def run_benchmark(self):
        """Run the flow benchmark end-to-end."""
        self.generate_mesh()
        self.load_mesh()
        self.analytical_values()
        self.reset_vertex_drag_lumped_error_stats()

        self.force_vector_computed = self.integrate_obstacle_force()
        self.drag_force_computed = float(np.dot(self.force_vector_computed, self.flow_direction))
        self.force_error_abs = float(abs(self.drag_force_computed - self.drag_force_analytical))

        if abs(self.drag_force_analytical) > 1.0e-30:
            self.force_error_rel = self.force_error_abs / abs(self.drag_force_analytical)
        self.finalize_vertex_drag_lumped_error_stats()

        if self.ddg_enabled:
            self.build_ddg_complex()

    def summary(self):
        axes = ("X", "Y", "Z")
        out = {}

        if self.force_vector_computed is not None and self.force_vector_analytical is not None:
            for i in range(self.vector_dim):
                out[f"Force {axes[i]}"] = (
                    float(self.force_vector_computed[i]),
                    float(self.force_vector_analytical[i]),
                )

        out["Drag"] = (float(self.drag_force_computed), float(self.drag_force_analytical))
        out["Gravity"] = (float(self.drag_force_computed), float(self.gravity_force_analytical))
        out["Drag Abs Error"] = (float(self.force_error_abs), 0.0)
        out["Drag Rel Error"] = (float(self.force_error_rel), 0.0)
        out["vertexDragRelErrorMean"] = (float(self.vertex_drag_rel_error_mean), 0.0)
        out["vertexDragRelErrorMax"] = (float(self.vertex_drag_rel_error_max), 0.0)
        out["DDG Residual Median"] = (float(self.ddg_residual_median), 0.0)
        out["DDG Residual Max"] = (float(self.ddg_residual_max), 0.0)
        return out
