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

# Curvature methods (dual vertex-based):
_curvature_i_methods = {
    "laplace-beltrami": compute_laplace_beltrami,
}

# Dual vertex area methods:
_area_i_methods = {
    "default": compute_area_vertex_default,
}

# Primal triangle area methods:
_area_ijk_methods = {
    "default": compute_area_triangle_default,
}

# Dual triangle area methods:
_area_methods = {
    "default": compute_area_triangle_default,
}

# Primal volume methods:
_volume_ijkm_methods = {
    "default": compute_volume_default,
}

# Dual volume methods:
_volume_i_methods = {
    "default": compute_volume_default,
}

def _read_dualvolume_csv(dual_csv, n):
    import numpy as np
    import pandas as pd
    from pathlib import Path

    dual_csv = Path(dual_csv).resolve()
    if not dual_csv.exists():
        raise FileNotFoundError(f"DualVolume CSV not found: {dual_csv}")

    df = pd.read_csv(dual_csv)
    if "PointID" not in df.columns or "DualVolume" not in df.columns:
        raise ValueError(f"DualVolume CSV missing columns PointID/DualVolume: {dual_csv}")

    ids = df["PointID"].to_numpy(dtype=int)
    vols = df["DualVolume"].to_numpy(dtype=float)

    Vi = np.zeros(n, dtype=float)
    if ids.size == 0:
        return Vi

    # Map 1-based gmsh IDs to 0-based python indices when applicable
    offset = 1 if (ids.min() == 1 and ids.max() == n) else 0
    idx = ids - offset
    m = (idx >= 0) & (idx < n)
    Vi[idx[m]] = vols[m]
    return Vi


def _dual_volume_curved(points, tris, **kwargs):
    from .geometry._curved_volume import curved_volume
    import numpy as np
    import tempfile
    from pathlib import Path

    points = np.asarray(points, float)
    tris = np.asarray(tris, int)
    n = points.shape[0]

    # IMPORTANT: remove keys so we don't pass duplicates via **kwargs
    workdir = kwargs.pop("workdir", None)
    msh_path = kwargs.pop("msh_path", None)

    if workdir is None:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            curved_volume((points, tris), complex_dtype="vf", workdir=str(wd), **kwargs)
            dual_csv = wd / "mesh_COEFFS_Transformed_DualVolume.csv"
            return _read_dualvolume_csv(dual_csv, n)
    else:
        wd = Path(workdir)
        curved_volume((points, tris), complex_dtype="vf", workdir=str(wd), **kwargs)

        stem = Path(msh_path).stem if msh_path else "mesh"
        dual_csv = wd / f"{stem}_COEFFS_Transformed_DualVolume.csv"
        return _read_dualvolume_csv(dual_csv, n)


# Register dual-volume method ONCE
_volume_i_methods["curved_volume"] = _dual_volume_curved

 
# Total volume methods:
def _volume_curved(HC, complex_dtype="vf", **kwargs):
    # Lazy import so missing deps (meshio, pandas) don't break import-time
    #from ._curved_volume import curved_volume
    from .geometry._curved_volume import curved_volume
    return curved_volume(HC, complex_dtype=complex_dtype, **kwargs)

_volume_methods = {
    "default": compute_volume_default,
    "curved_volume": _volume_curved,
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
        

# Make the entry visible in the registry unconditionally; the lazy import
# inside _volume_curved will raise if dependencies are missing at call time.
 

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
        self._method = method
        self._func = _volume_methods[method]

    def __call__(self, HC, complex_dtype="vf", **kwargs):
        """
        HC : tuple -> (points, simplices)
        complex_dtype : must be 'vf'
        """
        if complex_dtype != "vf":
            raise NotImplementedError("Volume only supported for 'vf' complexes")

        # Default volume expects (points, simplices) and no complex_dtype kw
        if self._method == "default":
            points, simplices = HC
            return float(self._func(points, simplices))

        # curved_volume wrapper expects HC and may use complex_dtype/workdir/msh_path
        return float(self._func(HC, complex_dtype=complex_dtype, **kwargs))
    
class Volume_i:
    """Dual vertex volume estimator."""
    def __init__(self, method="default"):
        self._func = _volume_i_methods[method]

    def __call__(self, HC, complex_dtype="vv", **kwargs): #Updated Volume_i to pass kwargs
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
            Volume contribution per vertex.
        """
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices, **kwargs)
        elif complex_dtype == "vv":
            return self._func(HC, **kwargs)
        else:
            raise ValueError("Unknown complex_dtype")

# Example instantiation:
curvature_i = Curvature_i()
area_i = Area_i()
area_ijk = Area_ijk()
area = Area()
volume = Volume()
volume_i = Volume_i()