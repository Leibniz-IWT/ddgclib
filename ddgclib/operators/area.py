"""
Area estimators for simplicial complexes.

Moved from ddgclib/_method_wrappers.py with the same API, plus new DualArea_i.
"""

import numpy as np

from ddgclib.operators._registry import MethodRegistry

# Registries
area_i_methods = MethodRegistry("area_i")
area_ijk_methods = MethodRegistry("area_ijk")
area_methods = MethodRegistry("area")


def _ensure_benchmarks_registered():
    if "default" not in area_i_methods:
        import sys
        import os
        _repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        _bench_path = os.path.join(_repo_root, "benchmarks")
        if os.path.isdir(_bench_path) and _bench_path not in sys.path:
            sys.path.append(_bench_path)
        from benchmarks._benchmark_toy_methods import (
            compute_area_vertex_default,
            compute_area_triangle_default,
        )
        area_i_methods.register("default", compute_area_vertex_default)
        area_ijk_methods.register("default", compute_area_triangle_default)
        area_methods.register("default", compute_area_triangle_default)


class Area_i:
    """Vertex dual area estimator.

    Parameters
    ----------
    method : str
        Registered method name (default "default").
    """

    def __init__(self, method: str = "default"):
        _ensure_benchmarks_registered()
        self._func = area_i_methods[method]

    def __call__(self, HC, complex_dtype: str = "vv"):
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices)
        elif complex_dtype == "vv":
            return self._func(HC)
        else:
            raise ValueError(f"Unknown complex_dtype: {complex_dtype!r}")


class Area_ijk:
    """Triangle face area estimator.

    Parameters
    ----------
    method : str
        Registered method name (default "default").
    """

    def __init__(self, method: str = "default"):
        _ensure_benchmarks_registered()
        self._func = area_ijk_methods[method]

    def __call__(self, HC, complex_dtype: str = "vv"):
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices)
        elif complex_dtype == "vv":
            return self._func(HC)
        else:
            raise ValueError(f"Unknown complex_dtype: {complex_dtype!r}")


class Area:
    """Total surface area estimator.

    Parameters
    ----------
    method : str
        Registered method name (default "default").
    """

    def __init__(self, method: str = "default"):
        _ensure_benchmarks_registered()
        self._func = area_methods[method]

    def __call__(self, HC, complex_dtype: str = "vf"):
        if complex_dtype != "vf":
            raise NotImplementedError("Total area only supported for 'vf' complexes")
        areas = self._func(*HC)
        return np.sum(areas)


class DualArea_i:
    """Compute dual area directly from barycentric duals.

    Wraps d_area from ddgclib.barycentric._duals. Requires that
    compute_vd(HC) has been called first to populate v.vd sets.

    Usage
    -----
        from ddgclib.barycentric._duals import compute_vd
        compute_vd(HC, cdist=1e-10)

        dual_area = DualArea_i()
        a = dual_area(v)  # area of dual cell around vertex v
    """

    def __call__(self, v) -> float:
        """Compute dual area for a single vertex.

        Parameters
        ----------
        v : vertex object
            Must have v.vd and v.nn populated (via compute_vd).

        Returns
        -------
        float
            Dual cell area.
        """
        from ddgclib.barycentric._duals import d_area
        return d_area(v)
