#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute transformed patch volumes from a COEFFS_Transformed CSV
and save per-face patch volumes (and scaled variants) to <input>_Volume.csv.

Input CSV (columns must match exactly):
  triangle_id, A_id, B_id, C_id,
  scale_factors1_x, scale_factors1_y, scale_factors1_z,
  A_transformed_x, A_transformed_y, A_transformed_z,
  B_transformed_x, B_transformed_y, B_transformed_z,
  C_transformed_x, C_transformed_y, C_transformed_z,
  ABC_new_A, ABC_new_B, ABC_new_C, ABC_new_D, ABC_new_E,
  ABC_new_F, ABC_new_G, ABC_new_H, ABC_new_I, ABC_new_J

(Each row’s coeffs are optionally normalized to J = -1 when |J| > J_FORCE_THR.)
(Additionally, coefficients are snapped:
   - to 0 when |coef| < SNAP_COEFF_TOL (A..I by default);
   - to +1 when |coef-1| < SNAP_COEFF_TOL;
   - to -1 when |coef+1| < SNAP_COEFF_TOL.
  J is only snapped if SNAP_J_TOO=True.)

Output CSV (one row per NON-PLANAR triangle):
  triangle_id,Vcorrection,V_patch_A,V_patch_B,V_patch_C,
  scaling_factor,Vcorrection_scal,V_patch_A_scal,V_patch_B_scal,V_patch_C_scal,side

Console printouts:
  [input] ...
  Triangles kept (curved): ... | planar skipped: ...
  Patch-based total (sum of patches): ...
  Total elapsed: ... s
  [write] ...
"""

import os
import sys
import glob
import time
import math
import argparse
import numpy as np
import pandas as pd

# --- thresholds / knobs ---
FLAT_TOL = 1e-10         # detect "planar" via max(|A..F|) < FLAT_TOL
J_FORCE_THR = 1e-4       # only force J = -1 when |J| is clearly non-tiny
SNAP_COEFF_TOL = 5e-4    # snap |coef|<tol -> 0, |coef-1|<tol -> 1, |coef+1|<tol -> -1
SNAP_J_TOO = False       # set True to also snap J with the same rule
OBSERVER_OUTSIDE = True  # passed to curvature helper if used

# Default input for this transformed pipeline
DEFAULT_INPUT_CSV = "coarse_paraboloid_COEFFS_Transformed.csv"
DEFAULT_INPUT_CSV = "coarse_hyperboloid_COEFFS_Transformed.csv"
DEFAULT_INPUT_CSV = "CylinderSymm_0_tet_COEFFS_Transformed.csv"

# --- required project helpers ---
try:
    import _3_4_shared_axisymmetric_utils as sau
    from _3_4_shared_axisymmetric_utils import (
        principal_curvatures_at,
        principal_curvatures_rule_label,
        wedge_volume_rotation,
    )
except Exception as e:
    raise ImportError(
        "Could not import shared_axisymmetric_utils. Ensure it's on PYTHONPATH and "
        "exports principal_curvatures_at, principal_curvatures_rule_label, wedge_volume_rotation."
    ) from e

# Wire the numeric swept-segment helper into the shared module namespace
try:
    from _3_3_swept_segment_volume_AB import swept_segment_volume_AB
    sau.swept_segment_volume_AB = swept_segment_volume_AB
except Exception as e:
    raise ImportError(
        "Could not import swept_segment_volume_AB from _swept_segment_volume_AB.py. "
        "This function is required by wedge_volume_rotation."
    ) from e

# Optional per-vertex split (leave equal thirds if unavailable)
split_patch = None
try:
    # If you have this locally, we'll use it; otherwise we fall back to equal thirds
    from _4_dualvolume_split_patch_volume_thickness_weighted import (
        split_patch_volume_thickness_weighted as split_patch,
    )
except Exception:
    split_patch = None

REQUIRED_COLS = [
    "triangle_id","A_id","B_id","C_id",
    "scale_factors1_x","scale_factors1_y","scale_factors1_z",
    "A_transformed_x","A_transformed_y","A_transformed_z",
    "B_transformed_x","B_transformed_y","B_transformed_z",
    "C_transformed_x","C_transformed_y","C_transformed_z",
    "ABC_new_A","ABC_new_B","ABC_new_C","ABC_new_D","ABC_new_E",
    "ABC_new_F","ABC_new_G","ABC_new_H","ABC_new_I","ABC_new_J",
]

def _find_default_input():
    """Prefer DEFAULT_INPUT_CSV; otherwise first '*_COEFFS_Transformed.csv' in CWD."""
    if os.path.isfile(DEFAULT_INPUT_CSV):
        return DEFAULT_INPUT_CSV
    hits = sorted(glob.glob("*_COEFFS_Transformed.csv"))
    return hits[0] if hits else None

def _snap_scalar(v: float, tol: float) -> float:
    """Snap a single value to {0, +1, -1} when within tol."""
    if abs(v) < tol:
        return 0.0
    if abs(v - 1.0) < tol:
        return 1.0
    if abs(v + 1.0) < tol:
        return -1.0
    return v

def _snap_coeffs(coeffs, tol=SNAP_COEFF_TOL, snap_J=False):
    """
    Return coeffs with tiny values snapped to 0 and near-unit values snapped to ±1.
    By default snaps A..I; J only if snap_J=True.
    """
    A,B,C,D,E,F,G,H,I,J = map(float, coeffs)
    parts = [A,B,C,D,E,F,G,H,I,J]
    limit = 10 if snap_J else 9
    for k in range(limit):
        parts[k] = _snap_scalar(parts[k], tol)
    return tuple(parts)

def _normalize_coeffs_J_minus1(coeffs, thr=J_FORCE_THR):
    """
    Ensure J == -1 by dividing all terms by (-J) when |J| > thr.
    If J ≈ -1 (within thr), leave as-is. If J is tiny (<= thr), return unchanged.
    coeffs: iterable (A,B,C,D,E,F,G,H,I,J)
    """
    A,B,C,D,E,F,G,H,I,J = map(float, coeffs)
    if abs(J + 1.0) <= thr:
        return (A,B,C,D,E,F,G,H,I,J)
    if abs(J) > thr:
        s = -1.0 / J
        return (A*s, B*s, C*s, D*s, E*s, F*s, G*s, H*s, I*s, J*s)
    return (A,B,C,D,E,F,G,H,I,J)

def _is_flat_quadric(coeffs, tol=FLAT_TOL):
    """
    Treat a face as planar if all quadratic terms are effectively zero:
      max(|A|,|B|,|C|,|D|,|E|,|F|) < tol
    """
    A,B,C,D,E,F,_,_,_,_ = map(float, coeffs)
    return max(abs(A),abs(B),abs(C),abs(D),abs(E),abs(F)) < tol

def _safe_int(x):
    try:
        if isinstance(x, (int, np.integer)): return int(x)
        return int(round(float(x)))
    except Exception:
        return int(x)

def main():
    parser = argparse.ArgumentParser(
        description="Compute Vcorrection per triangle from *_COEFFS_Transformed.csv and write <basename>_Volume.csv in CWD."
    )
    parser.add_argument("input_csv", nargs="?", default=None,
                        help="Path to *_COEFFS_Transformed.csv (defaults to coarse_paraboloid_COEFFS_Transformed.csv, or first match in CWD).")
    parser.add_argument("output_csv", nargs="?", default=None,
                        help="Optional explicit output path. By default writes <basename>_Volume.csv in CWD.")
    args = parser.parse_args()

    in_csv = args.input_csv or _find_default_input()
    if not in_csv or not os.path.isfile(in_csv):
        raise FileNotFoundError(
            f"No input CSV found. Looked for '{DEFAULT_INPUT_CSV}' or first '*_COEFFS_Transformed.csv' in CWD."
        )

    base = os.path.splitext(os.path.basename(in_csv))[0]
    out_csv = args.output_csv or f"{base}_Volume.csv"

    df = pd.read_csv(in_csv)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # --- Vectorized snap for A..I: 0, +1, -1 ---
    coef_cols_noJ = ["ABC_new_A","ABC_new_B","ABC_new_C","ABC_new_D","ABC_new_E","ABC_new_F","ABC_new_G","ABC_new_H","ABC_new_I"]
    # snap to 0
    df[coef_cols_noJ] = df[coef_cols_noJ].where(df[coef_cols_noJ].abs() >= SNAP_COEFF_TOL, 0.0)
    # snap to +1
    for c in coef_cols_noJ:
        df[c] = np.where(np.abs(df[c] - 1.0) < SNAP_COEFF_TOL, 1.0, df[c])
    # snap to -1
    for c in coef_cols_noJ:
        df[c] = np.where(np.abs(df[c] + 1.0) < SNAP_COEFF_TOL, -1.0, df[c])
    if SNAP_J_TOO:
        # apply same vectorized snapping to J as well
        c = "ABC_new_J"
        df[c] = np.where(np.abs(df[c]) < SNAP_COEFF_TOL, 0.0, df[c])
        df[c] = np.where(np.abs(df[c] - 1.0) < SNAP_COEFF_TOL, 1.0, df[c])
        df[c] = np.where(np.abs(df[c] + 1.0) < SNAP_COEFF_TOL, -1.0, df[c])

    # Share observer flag; create DEBUG_PRINT if your utils reads it
    try:
        sau.OBSERVER_OUTSIDE = OBSERVER_OUTSIDE
        if not hasattr(sau, "DEBUG_PRINT"):
            sau.DEBUG_PRINT = False
    except Exception:
        pass

    kept_rows = []
    planar_skipped = 0
    t0 = time.time()

    for _, row in df.iterrows():
        tri_id = _safe_int(row["triangle_id"])

        coeffs_raw = (
            row["ABC_new_A"], row["ABC_new_B"], row["ABC_new_C"], row["ABC_new_D"], row["ABC_new_E"],
            row["ABC_new_F"], row["ABC_new_G"], row["ABC_new_H"], row["ABC_new_I"], row["ABC_new_J"],
        )

        # Per-row snap (A..I by default) to clean any residual float noise from CSV
        coeffs_pre = _snap_coeffs(coeffs_raw, tol=SNAP_COEFF_TOL, snap_J=SNAP_J_TOO)

        if _is_flat_quadric(coeffs_pre):
            planar_skipped += 1
            continue

        # Normalize J when applicable
        coeffs_norm = _normalize_coeffs_J_minus1(coeffs_pre)

        # Snap again after normalization to kill tiny and near-unit perturbations
        coeffs = _snap_coeffs(coeffs_norm, tol=SNAP_COEFF_TOL, snap_J=SNAP_J_TOO)

        # CRITICAL: set the shared module-global COEFFS so internal calls see it
        try:
            sau.COEFFS = coeffs
        except Exception:
            pass

        A = np.array([row["A_transformed_x"], row["A_transformed_y"], row["A_transformed_z"]], dtype=float)
        B = np.array([row["B_transformed_x"], row["B_transformed_y"], row["B_transformed_z"]], dtype=float)
        C = np.array([row["C_transformed_x"], row["C_transformed_y"], row["C_transformed_z"]], dtype=float)

        # Curvature label (used as side/orientation hint)
        try:
            k1_A, k2_A = principal_curvatures_at(A, coeffs, project=True, observer_outside=OBSERVER_OUTSIDE)
            k1_B, k2_B = principal_curvatures_at(B, coeffs, project=True, observer_outside=OBSERVER_OUTSIDE)
            k1_C, k2_C = principal_curvatures_at(C, coeffs, project=True, observer_outside=OBSERVER_OUTSIDE)
            label = principal_curvatures_rule_label(k2_A, k2_B, k2_C)
        except Exception:
            label = "outside"  # sensible default

        # Core patch volume via rotation wedge
        case_id, V_patch = wedge_volume_rotation(coeffs, A, B, C, tri_id, forced_orientation=label)

        # Per-vertex split
        if split_patch is not None:
            try:
                V_A, V_B, V_C = split_patch(A, B, C, coeffs, V_patch)
            except Exception:
                V_A = V_B = V_C = float(V_patch) / 3.0
        else:
            V_A = V_B = V_C = float(V_patch) / 3.0

        # Scaling factor from anisotropic transform
        s = float(row["scale_factors1_x"]) * float(row["scale_factors1_y"]) * float(row["scale_factors1_z"])
        V_patch_scal = s * float(V_patch)
        V_A_scal = s * float(V_A)
        V_B_scal = s * float(V_B)
        V_C_scal = s * float(V_C)

        kept_rows.append({
            "triangle_id": tri_id,
            "Vcorrection": float(V_patch),
            "V_patch_A": float(V_A),
            "V_patch_B": float(V_B),
            "V_patch_C": float(V_C),
            "scaling_factor": s,
            "Vcorrection_scal": V_patch_scal,
            "V_patch_A_scal": V_A_scal,
            "V_patch_B_scal": V_B_scal,
            "V_patch_C_scal": V_C_scal,
            "side": label,
        })

    out_cols = [
        "triangle_id",
        "Vcorrection",
        "V_patch_A", "V_patch_B", "V_patch_C",
        "scaling_factor",
        "Vcorrection_scal", "V_patch_A_scal", "V_patch_B_scal", "V_patch_C_scal",
        "side",
    ]
    out_df = pd.DataFrame(kept_rows, columns=out_cols)
    out_df.to_csv(out_csv, index=False)

    t1 = time.time()
    V_sum = float(np.nansum(out_df["Vcorrection"].values)) if len(out_df) else 0.0
    print(f"[input] {in_csv}")
    print(f"Triangles kept (curved): {len(out_df)} | planar skipped: {planar_skipped}")
    print(f"Patch-based total (sum of patches): {V_sum:.8f}")
    print(f"Total elapsed: {t1 - t0:.2f} s")
    print(f"[write] {os.path.abspath(out_csv)}")

if __name__ == "__main__":
    main()
