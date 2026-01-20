import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
