"""
Volume estimators for simplicial complexes.

Moved from ddgclib/_method_wrappers.py with the same API.
"""

import numpy as np

from ddgclib.operators._registry import MethodRegistry

# Registries
volume_methods = MethodRegistry("volume")
volume_i_methods = MethodRegistry("volume_i")


def _ensure_benchmarks_registered():
    if "default" not in volume_methods:
        import sys
        import os
        _repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        _bench_path = os.path.join(_repo_root, "benchmarks")
        if os.path.isdir(_bench_path) and _bench_path not in sys.path:
            sys.path.append(_bench_path)
        from benchmarks._benchmark_toy_methods import compute_volume_default
        volume_methods.register("default", compute_volume_default)
        volume_i_methods.register("default", compute_volume_default)


def _volume_curved(HC, complex_dtype="vf", **kwargs):
    """Lazy-import curved volume computation."""
    from ddgclib.geometry._curved_volume import curved_volume
    return curved_volume(HC, complex_dtype=complex_dtype, **kwargs)


def _read_dualvolume_csv(dual_csv, n):
    """Read per-vertex dual volumes from CSV produced by curved volume pipeline."""
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
    """Per-vertex dual volumes via curved volume pipeline."""
    import tempfile
    from pathlib import Path

    points = np.asarray(points, float)
    tris = np.asarray(tris, int)
    n = points.shape[0]

    workdir = kwargs.pop("workdir", None)
    msh_path = kwargs.pop("msh_path", None)

    if workdir is None:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            _volume_curved((points, tris), complex_dtype="vf", workdir=str(wd), **kwargs)
            dual_csv = wd / "mesh_COEFFS_Transformed_DualVolume.csv"
            return _read_dualvolume_csv(dual_csv, n)
    else:
        wd = Path(workdir)
        _volume_curved((points, tris), complex_dtype="vf", workdir=str(wd), **kwargs)
        stem = Path(msh_path).stem if msh_path else "mesh"
        dual_csv = wd / f"{stem}_COEFFS_Transformed_DualVolume.csv"
        return _read_dualvolume_csv(dual_csv, n)


def _register_curved_volume():
    """Register curved volume methods (lazy, called on first use)."""
    if "curved_volume" not in volume_methods:
        volume_methods.register("curved_volume", _volume_curved)
        volume_i_methods.register("curved_volume", _dual_volume_curved)


class Volume:
    """Total volume estimator.

    Parameters
    ----------
    method : str
        "default" or "curved_volume".
    """

    def __init__(self, method: str = "default"):
        _ensure_benchmarks_registered()
        _register_curved_volume()
        self._method = method
        self._func = volume_methods[method]

    def __call__(self, HC, complex_dtype: str = "vf", **kwargs):
        if complex_dtype != "vf":
            raise NotImplementedError("Volume only supported for 'vf' complexes")
        if self._method == "default":
            points, simplices = HC
            return float(self._func(points, simplices))
        return float(self._func(HC, complex_dtype=complex_dtype, **kwargs))


class Volume_i:
    """Dual vertex volume estimator.

    Parameters
    ----------
    method : str
        "default" or "curved_volume".
    """

    def __init__(self, method: str = "default"):
        _ensure_benchmarks_registered()
        _register_curved_volume()
        self._func = volume_i_methods[method]

    def __call__(self, HC, complex_dtype: str = "vv", **kwargs):
        if complex_dtype == "vf":
            points, simplices = HC
            return self._func(points, simplices, **kwargs)
        elif complex_dtype == "vv":
            return self._func(HC, **kwargs)
        else:
            raise ValueError(f"Unknown complex_dtype: {complex_dtype!r}")
