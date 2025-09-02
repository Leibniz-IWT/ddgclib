# benchmarks/_benchmark_classes.py
import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import only the public method wrappers from the library
from ddgclib._method_wrappers import (
    Curvature_i, Curvature_ijk,
    Area_i, Area_ijk, Area,
    Volume,
)


class GeometryBenchmarkBase:
    """
    Base class for benchmarking geometric quantities on simplicial complexes.

    Responsibilities:
      - provide a uniform flow (generate_mesh -> compute_* -> analytical_values -> summary)
      - stay method-agnostic; it only calls whatever is registered in ddgclib/_method_wrappers.py
      - be robust when some methods are unavailable (set to None) or analytics are unknown (np.nan)
    """

    def __init__(self, name: str = "unnamed", method: Optional[Dict[str, Any]] = None, complex_dtype: str = "vf"):
        self.name = name
        self.complex_dtype = complex_dtype  # 'vf' (vertex-face) or 'vv' (vertex-vertex)
        self.method = method or {}

        # Mesh containers
        self.points: Optional[np.ndarray] = None
        self.simplices: Optional[np.ndarray] = None

        # Results (computed vs. analytical)
        self.local_area: Optional[np.ndarray] = None

        self.H_computed: Optional[np.ndarray] = None
        self.H_analytical: Optional[np.ndarray] = None  # should be array for curvature benchmarks or np.nan otherwise

        self.area_computed: Optional[float] = None
        self.area_analytical: Optional[float] = None  # float or np.nan

        self.volume_computed: Optional[float] = None
        self.volume_analytical: Optional[float] = None  # float or np.nan

        # Resolve evaluators from the registry; if a key/method is missing, keep None (graceful skip)
        try:
            self.curvature_i = Curvature_i(self.method.get("curvature_i_method", "default"))
        except Exception as e:
            logger.info(f"curvature_i unavailable: {e}")
            self.curvature_i = None

        try:
            self.curvature_ijk = Curvature_ijk(self.method.get("curvature_ijk_method", "default"))
        except Exception as e:
            logger.info(f"curvature_ijk unavailable: {e}")
            self.curvature_ijk = None

        try:
            self.area_i = Area_i(self.method.get("area_i_method", "default"))
        except Exception as e:
            logger.info(f"area_i unavailable: {e}")
            self.area_i = None

        try:
            self.area_ijk = Area_ijk(self.method.get("area_ijk_method", "default"))
        except Exception as e:
            logger.info(f"area_ijk unavailable: {e}")
            self.area_ijk = None

        try:
            self.area = Area(self.method.get("area_method", "default"))
        except Exception as e:
            logger.info(f"area unavailable: {e}")
            self.area = None

        try:
            self.volume = Volume(self.method.get("volume_method", "default"))
        except Exception as e:
            logger.info(f"volume unavailable: {e}")
            self.volume = None

    # ------------------------------- Hooks -------------------------------- #

    def generate_mesh(self) -> None:
        """Subclass must set self.points (Nx3) and self.simplices (Mx3) for 'vf' meshes, or appropriate structure for 'vv'."""
        raise NotImplementedError

    def analytical_values(self) -> None:
        """
        Subclass should set *_analytical fields.
        If analytics are unknown, use np.nan (not 0.0).
        """
        raise NotImplementedError

    # --------------------------- Compute stages --------------------------- #

    def compute_surface_area(self) -> None:
        """Compute total surface area if an area method exists."""
        HC = (self.points, self.simplices) if self.complex_dtype == "vf" else self.points
        if self.area is not None:
            try:
                self.area_computed = self.area(HC, complex_dtype=self.complex_dtype)
            except Exception as e:
                logger.warning(f"Area computation failed: {e}")
                self.area_computed = None
        else:
            self.area_computed = None

    def compute_volume(self) -> None:
        """Compute enclosed volume if a volume method exists (vf only)."""
        if self.complex_dtype == "vf" and self.volume is not None:
            HC = (self.points, self.simplices)
            try:
                self.volume_computed = self.volume(HC, complex_dtype=self.complex_dtype)
            except Exception as e:
                logger.warning(f"Volume computation failed: {e}")
                self.volume_computed = None
        else:
            self.volume_computed = None

    def compute_curvature(self) -> None:
        """Compute curvature depending on the type of complex and available evaluator."""
        HC = (self.points, self.simplices) if self.complex_dtype == "vf" else self.points
        try:
            if self.complex_dtype == "vv" and self.curvature_i is not None:
                self.H_computed = self.curvature_i(HC, complex_dtype=self.complex_dtype)
            elif self.complex_dtype == "vf" and self.curvature_ijk is not None:
                # Triangle-level curvature could go here if/when implemented.
                self.H_computed = None
            else:
                self.H_computed = None
        except Exception as e:
            logger.warning(f"Curvature computation failed: {e}")
            self.H_computed = None

    # ----------------------------- Orchestration ----------------------------- #

    def run_benchmark(self) -> None:
        self.generate_mesh()
        self.compute_surface_area()
        self.compute_volume()
        self.compute_curvature()
        self.analytical_values()

    # -------------------------------- Summary -------------------------------- #

    @staticmethod
    def _safe_curv_errors(Hc: Optional[np.ndarray], Ha: Optional[np.ndarray]) -> Tuple[float, float]:
        """
        Return (mean_abs_error, std_abs_error) or (np.nan, 0.0) if not comparable.
        """
        if Hc is None or Ha is None:
            return (np.nan, 0.0)
        try:
            Hc = np.asarray(Hc)
            Ha = np.asarray(Ha)
            if Hc.shape != Ha.shape:
                return (np.nan, 0.0)
            # Allow NaNs in analytical: mask them out
            mask = np.isfinite(Ha) & np.isfinite(Hc)
            if not np.any(mask):
                return (np.nan, 0.0)
            err = np.abs(Hc[mask] - Ha[mask])
            return (float(np.mean(err)), float(np.std(err)))
        except Exception:
            return (np.nan, 0.0)

    @staticmethod
    def _fmt_value(v: Any) -> str:
        """Pretty-print floats/None/np.nan/arrays in summaries."""
        if v is None:
            return "None"
        if isinstance(v, (int, float, np.floating)):
            if np.isnan(v):
                return "nan"
            return f"{float(v):.6f}"
        return str(v)

    def summary(self) -> Dict[str, Tuple[Any, Any]]:
        mean_error_H, std_error_H = self._safe_curv_errors(self.H_computed, self.H_analytical)

        return {
            "Surface Area": (self.area_computed, self.area_analytical),
            "Volume": (self.volume_computed, self.volume_analytical),
            "Curvature Mean Error": (mean_error_H, 0.0),  # target=0.0 for error
            "Curvature Std Error": (std_error_H, 0.0),
        }


def run_all_benchmarks(benchmark_classes: Iterable[type], method: Optional[Dict[str, str]] = None, complex_dtype: str = "vf") -> None:
    """
    Instantiate and run all provided benchmark cases with the specified method registry names.

    Example 'method' dict:
        {
            "curvature_i_method": "laplace-beltrami",
            "area_i_method": "default",
            "area_ijk_method": "default",
            "area_method": "default",
            "volume_method": "default",
        }
    """
    method = method or {}
    for BenchmarkClass in benchmark_classes:
        bench = BenchmarkClass(method=method, complex_dtype=complex_dtype)
        bench.run_benchmark()

        print(f"Benchmark: {bench.name}")
        for k, (v_comp, v_ana) in bench.summary().items():
            vc = GeometryBenchmarkBase._fmt_value(v_comp)
            va = GeometryBenchmarkBase._fmt_value(v_ana)
            print(f"  {k}: computed={vc}, analytical={va}")
