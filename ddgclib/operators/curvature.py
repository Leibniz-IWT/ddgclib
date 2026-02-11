"""
Curvature estimators for simplicial complexes.

Moved from ddgclib/_method_wrappers.py with the same API.
"""

import numpy as np

from ddgclib.operators._registry import MethodRegistry

# Registry
curvature_i_methods = MethodRegistry("curvature_i")

# Lazy-load benchmark methods to avoid hard import-time dependency
def _ensure_benchmarks_registered():
    if "laplace-beltrami" not in curvature_i_methods:
        import sys
        import os
        _repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        _bench_path = os.path.join(_repo_root, "benchmarks")
        if os.path.isdir(_bench_path) and _bench_path not in sys.path:
            sys.path.append(_bench_path)
        from benchmarks._benchmark_toy_methods import compute_laplace_beltrami
        curvature_i_methods.register("laplace-beltrami", compute_laplace_beltrami)


class Curvature_i:
    """Vertex-based mean curvature magnitude estimator.

    Parameters
    ----------
    method : str
        Registered method name (default "laplace-beltrami").
    """

    def __init__(self, method: str = "laplace-beltrami"):
        _ensure_benchmarks_registered()
        self._func = curvature_i_methods[method]

    def __call__(self, HC, complex_dtype: str = "vv"):
        """Compute mean curvature at vertices.

        Parameters
        ----------
        HC : tuple or Complex
            (points, simplices) if 'vf', hyperct.Complex if 'vv'.
        complex_dtype : str
            'vf' or 'vv'.

        Returns
        -------
        np.ndarray
            Mean curvature magnitudes at vertices.
        """
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices)
        elif complex_dtype == "vv":
            return self._func(HC)
        else:
            raise ValueError(f"Unknown complex_dtype: {complex_dtype!r}")


class Curvature_ijk:
    """Triangle-based curvature estimator (not yet implemented)."""

    def __init__(self, method: str = "default"):
        raise NotImplementedError("Triangle-level curvature not implemented yet")

    def __call__(self, HC, complex_dtype: str = "vv"):
        raise NotImplementedError("Triangle-level curvature not implemented yet")
