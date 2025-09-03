import numpy as np
import logging

import sys
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Ensure the repo's benchmarks/ folder is importable when running from the library
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_bench_path = os.path.join(_repo_root, "benchmarks")
if os.path.isdir(_bench_path) and _bench_path not in sys.path:
    sys.path.append(_bench_path)
    
from benchmarks._benchmark_toy_methods import (
    compute_laplace_beltrami,
    compute_area_vertex_default,
    compute_area_triangle_default,
    compute_volume_default
)

# --- Method Registries ---
_curvature_i_methods = {
    "laplace-beltrami": compute_laplace_beltrami,
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

_volume_methods = {
    "default": compute_volume_default,
}

# --- Curvature ---
class Curvature_i:
    """Vertex-based mean curvature magnitude estimator."""
    def __init__(self, method="laplace-beltrami"):
        self._func = _curvature_i_methods[method]

    def __call__(self, HC, complex_dtype="vv"):
        """
        Parameters
        ----------
        HC : tuple or object
            (points, simplices) if 'vf', otherwise hyperct.Complex structure.
        complex_dtype : str
            'vf' or 'vv'

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
            raise ValueError("Unknown complex_dtype")
        
def _volume_curved(HC, complex_dtype="vf", **kwargs):
    # Lazy import so missing deps (meshio, pandas) don't break import-time
    from ._curved_volume import curved_volume
    return curved_volume(HC, complex_dtype=complex_dtype, **kwargs)

# Make the entry visible in the registry unconditionally; the lazy import
# inside _volume_curved will raise if dependencies are missing at call time.
_volume_methods["curved_volume"] = _volume_curved

class Curvature_ijk:
    """Triangle-based curvature estimator (not yet implemented)."""
    def __init__(self, method="default"):
        raise NotImplementedError("Triangle-level curvature not implemented yet")

    def __call__(self, HC, complex_dtype="vv"):
        raise NotImplementedError("Triangle-level curvature not implemented yet")

# --- Area ---
class Area_i:
    """Vertex dual area estimator."""
    def __init__(self, method="default"):
        self._func = _area_i_methods[method]

    def __call__(self, HC, complex_dtype="vv"):
        """
        Parameters
        ----------
        HC : tuple or object
            (points, simplices) if 'vf', otherwise hyperct.Complex structure.
        complex_dtype : str
            'vf' or 'vv'

        Returns
        -------
        np.ndarray
            Area contribution per vertex.
        """
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices)
        elif complex_dtype == "vv":
            return self._func(HC)
        else:
            raise ValueError("Unknown complex_dtype")

class Area_ijk:
    """Triangle face area estimator."""
    def __init__(self, method="default"):
        self._func = _area_ijk_methods[method]

    def __call__(self, HC, complex_dtype="vv"):
        """
        Parameters
        ----------
        HC : tuple or object
            (points, simplices) if 'vf', otherwise hyperct.Complex structure.
        complex_dtype : str
            'vf' or 'vv'

        Returns
        -------
        np.ndarray
            Area per triangle face.
        """
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices)
        elif complex_dtype == "vv":
            return self._func(HC)
        else:
            raise ValueError("Unknown complex_dtype")

class Area:
    """Total surface area estimator."""
    def __init__(self, method="default"):
        self._func = _area_methods[method]

    def __call__(self, HC, complex_dtype="vf"):
        """
        Parameters
        ----------
        HC : tuple
            (points, simplices)
        complex_dtype : str
            Must be 'vf'.

        Returns
        -------
        float
            Total surface area.
        """
        if complex_dtype != "vf":
            raise NotImplementedError("Total area only supported for 'vf' complexes")
        areas = self._func(*HC)
        return np.sum(areas)

# --- Volume ---
class Volume:
    """Total volume estimator."""
    def __init__(self, method="default"):
        self._func = _volume_methods[method]

    def __call__(self, HC, complex_dtype="vf"):
        """
        HC : tuple -> (points, simplices)
        complex_dtype : must be 'vf'
        """
        if complex_dtype != "vf":
            raise NotImplementedError("Volume only supported for 'vf' complexes")
        # Pass the whole tuple and the dtype as a keyword:
        return float(self._func(HC, complex_dtype=complex_dtype))

# --- Example instantiation ---
curvature_i = Curvature_i()
area_i = Area_i()
area_ijk = Area_ijk()
area = Area()
volume = Volume()