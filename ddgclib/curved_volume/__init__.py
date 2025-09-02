"""
ddgclib.curved_volume
=====================

Package with the curved-volume pipeline:
- per-vertex quadric fitting
- canonical transform
- per-face curved patch volumes
- optional thickness-weighted splitting

This __init__ provides light re-exports so other modules (e.g. the facade
ddgclib/_curved_volume.py) can import from a stable namespace.
"""

from __future__ import annotations

# --- Part 1: coefficients (per-triangle/vertex) ---
try:
    from ._1_coeffs_computing import compute_rimsafe_alltri
except Exception:  # keep import-safe if a submodule is missing during partial installs
    compute_rimsafe_alltri = None  # type: ignore

# --- Part 2: transformer to canonical space ---
try:
    # single-file transform: returns (out_csv, err_csv, n_errors)
    from ._2_quadric_transformer import process_file as transform_process_file
except Exception:
    transform_process_file = None  # type: ignore

# --- Part 3: volume engines (example: ellipsoid solid-angle lift) ---
try:
    # reads a *_COEFFS_Transformed.csv and writes *_Volume.csv
    from ._3_5_volume_ellipsoid import process_coeffs_transformed_csv as ellipsoid_volume_from_transformed
except Exception:
    ellipsoid_volume_from_transformed = None  # type: ignore

# --- Shared: thickness-weighted split helper ---
try:
    from ._4_dualvolume_split_patch_volume_thickness_weighted import (
        split_patch_volume_thickness_weighted,
    )
except Exception:
    split_patch_volume_thickness_weighted = None  # type: ignore

# --- Batch drivers (optional CLI-style wrappers; kept import-safe) ---
try:
    # CSV -> per-face volumes (translation-wall flavor)
    from .All_Translation_Volume_Transformed import main as run_translation_driver
except Exception:
    run_translation_driver = None  # type: ignore

try:
    # CSV -> per-face volumes (axisymmetric/rotation flavor)
    from .All_Rotation_Volume_Transformed import main as run_rotation_driver
except Exception:
    run_rotation_driver = None  # type: ignore


__all__ = [
    # Part-1
    "compute_rimsafe_alltri",
    # Part-2
    "transform_process_file",
    # Part-3 (ellipsoid example)
    "ellipsoid_volume_from_transformed",
    # Shared split helper
    "split_patch_volume_thickness_weighted",
    # Batch drivers (optional)
    "run_translation_driver",
    "run_rotation_driver",
]
