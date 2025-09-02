import numpy as np
import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Make benchmarks/ importable for the toy/demo methods (harmless if already set)
# -----------------------------------------------------------------------------
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_bench_path = os.path.join(_repo_root, "benchmarks")
if os.path.isdir(_bench_path) and _bench_path not in sys.path:
    sys.path.append(_bench_path)

# -----------------------------------------------------------------------------
# Optional toy/demo methods from benchmarks/ (used as defaults/fallbacks)
# -----------------------------------------------------------------------------
try:
    from benchmarks._benchmark_toy_methods import (
        compute_laplace_beltrami,
        compute_area_vertex_default,
        compute_area_triangle_default,
        compute_volume_default,
    )
except Exception as e:  # pragma: no cover
    logger.warning("Toy methods unavailable (benchmarks/_benchmark_toy_methods.py): %s", e)

    def compute_laplace_beltrami(*_args, **_kwargs):
        raise NotImplementedError("compute_laplace_beltrami not available")

    def compute_area_vertex_default(*_args, **_kwargs):
        raise NotImplementedError("compute_area_vertex_default not available")

    def compute_area_triangle_default(*_args, **_kwargs):
        raise NotImplementedError("compute_area_triangle_default not available")

    def compute_volume_default(*_args, **_kwargs):
        raise NotImplementedError("compute_volume_default not available")


# -----------------------------------------------------------------------------
# Registries
# -----------------------------------------------------------------------------
_curvature_i_methods = {
    "laplace-beltrami": compute_laplace_beltrami,
    # alias so callers that pass "default" still work
    "default": compute_laplace_beltrami,
}

_area_i_methods = {
    "default": compute_area_vertex_default,
}

_area_ijk_methods = {
    "default": compute_area_triangle_default,
}

_area_methods = {
    "default": compute_area_triangle_default,
}

# Normalize **volume** registry so every entry has the SAME signature:
#     fn(HC, complex_dtype="vf", **kwargs)   where HC == (points, simplices)
def _volume_default(HC, complex_dtype="vf", **kwargs):
    if complex_dtype != "vf":
        raise NotImplementedError("Total volume only supported for 'vf' complexes")
    points, simplices = HC
    return compute_volume_default(points, simplices)

_volume_methods = {
    "default": _volume_default,
}

# -----------------------------------------------------------------------------
# Optional: register curved-volume method lazily (ddgclib/_curved_volume.py)
# Lets users select with: method={"volume_method": "curved_volume", ...}
# -----------------------------------------------------------------------------
def _volume_curved(HC, complex_dtype="vf", **kwargs):
    # Lazy import so missing deps (meshio, pandas) don't break import-time
    from ._curved_volume import curved_volume
    return curved_volume(HC, complex_dtype=complex_dtype, **kwargs)

# Make the entry visible in the registry unconditionally; the lazy import
# inside _volume_curved will raise if dependencies are missing at call time.
_volume_methods["curved_volume"] = _volume_curved


# -----------------------------------------------------------------------------
# Wrapper classes
# -----------------------------------------------------------------------------
class Curvature_i:
    """Vertex-based mean curvature magnitude estimator."""
    def __init__(self, method="laplace-beltrami"):
        if method not in _curvature_i_methods:
            raise KeyError(f"Unknown curvature_i method: {method}")
        self._func = _curvature_i_methods[method]

    def __call__(self, HC, complex_dtype="vv"):
        """
        HC: (points, simplices) if 'vf', otherwise a hyperct.Complex-like structure.
        """
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices)
        elif complex_dtype == "vv":
            return self._func(HC)
        else:
            raise ValueError(f"Unknown complex_dtype: {complex_dtype}")


class Curvature_ijk:
    """Triangle-based curvature estimator (placeholder)."""
    def __init__(self, method="default"):
        raise NotImplementedError("Triangle-level curvature not implemented yet")

    def __call__(self, HC, complex_dtype="vv"):
        raise NotImplementedError("Triangle-level curvature not implemented yet")


class Area_i:
    """Vertex dual area estimator."""
    def __init__(self, method="default"):
        if method not in _area_i_methods:
            raise KeyError(f"Unknown area_i method: {method}")
        self._func = _area_i_methods[method]

    def __call__(self, HC, complex_dtype="vv"):
        """
        HC: (points, simplices) if 'vf', otherwise a hyperct.Complex-like structure.
        """
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices)
        elif complex_dtype == "vv":
            return self._func(HC)
        else:
            raise ValueError(f"Unknown complex_dtype: {complex_dtype}")


class Area_ijk:
    """Triangle face area estimator."""
    def __init__(self, method="default"):
        if method not in _area_ijk_methods:
            raise KeyError(f"Unknown area_ijk method: {method}")
        self._func = _area_ijk_methods[method]

    def __call__(self, HC, complex_dtype="vv"):
        """
        HC: (points, simplices) if 'vf', otherwise a hyperct.Complex-like structure.
        """
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices)
        elif complex_dtype == "vv":
            return self._func(HC)
        else:
            raise ValueError(f"Unknown complex_dtype: {complex_dtype}")


class Area:
    """Total surface area estimator (sum of per-face areas)."""
    def __init__(self, method="default"):
        if method not in _area_methods:
            raise KeyError(f"Unknown area method: {method}")
        self._func = _area_methods[method]

    def __call__(self, HC, complex_dtype="vf"):
        """
        HC: (points, simplices); complex_dtype must be 'vf'.
        """
        if complex_dtype != "vf":
            raise NotImplementedError("Total area only supported for 'vf' complexes")
        areas = self._func(*HC)
        return float(np.sum(areas))


class Volume:
    """Total volume estimator."""
    def __init__(self, method="default"):
        if method not in _volume_methods:
            raise KeyError(f"Unknown volume method: {method}")
        self._func = _volume_methods[method]

    def __call__(self, HC, complex_dtype="vf"):
        """
        HC: (points, simplices); complex_dtype must be 'vf'.
        NOTE: We pass the HC tuple through unchanged so every registered
              method can use the same signature.
        """
        if complex_dtype != "vf":
            raise NotImplementedError("Volume only supported for 'vf' complexes")
        return float(self._func(HC, complex_dtype=complex_dtype))


# Convenience instances (optional)
curvature_i = Curvature_i()         # laplace-beltrami
area_i = Area_i()                   # default
area_ijk = Area_ijk()               # NotImplementedError on use
area = Area()                       # default
volume = Volume()                   # default (or 'curved_volume' if selected by method)
