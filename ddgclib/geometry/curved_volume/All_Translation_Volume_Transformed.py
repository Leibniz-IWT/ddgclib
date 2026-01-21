#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute transformed patch volumes from a COEFFS_Transformed CSV
and save per-face patch volumes (and scaled variants) to <input>_Volume.csv.

Inputs (per row, required):
  triangle_id, A_id, B_id, C_id,
  scale_factors1_x, scale_factors1_y, scale_factors1_z,
  A_transformed_x/y/z, B_transformed_x/y/z, C_transformed_x/y/z,
  ABC_new_A .. ABC_new_J  (quadric coefficients; normalized internally to J = -1)

Outputs:
  triangle_id, Vcorrection, V_patch_A, V_patch_B, V_patch_C,
  scaling_factor, Vcorrection_scal, V_patch_A_scal, V_patch_B_scal, V_patch_C_scal, side

Notes:
- Planar faces are auto-detected; those rows are kept with zeros and side="planar".
- Volumes are computed in the transformed space from (A_transformed, B_transformed, C_transformed, ABC_new_*).
- Scaled volumes = raw volumes * (scale_x * scale_y * scale_z).

Usage:
  python Transformed_Patch_Volume.py [INPUT_COEFFS_Transformed.csv]

If INPUT is omitted, the script uses:
  "hyperbola_cylinder_x2_minus_y2_z1slice_COEFFS_Transformed.csv"
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd

# ---------- default input ----------
DEFAULT_INPUT = "hyperbola_cylinder_x2_minus_y2_z1slice_COEFFS_Transformed.csv"
DEFAULT_INPUT = "parabolic_cylinder_y_eq_x2_y1slice_COEFFS_Transformed.csv"
#DEFAULT_INPUT = "CylinderSymm_0_tet_COEFFS_Transformed.csv"
# --- per-vertex splitter ---
try:
    from _4_dualvolume_split_patch_volume_thickness_weighted import (
        split_patch_volume_thickness_weighted as split_patch,
    )
except Exception as e:
    raise ImportError(
        "Could not import split_patch_volume_thickness_weighted from "
        "_4_dualvolume_split_patch_volume_thickness_weighted.py."
    ) from e

# --- project-local conic/volume helpers ---
from _3_1_translation_curved_volume_computing import flat_plane_zero_volume
from _3_1_translation_curved_volume_computing import (
    conic_at_z, length_conic_segment, volume_ABApBp_general_conic,
    g_val, roots_x_of_y, roots_y_of_x, pick_branch,
)

# --- shared translator (classification + patch computation) ---
try:
    import _3_2_shared_translation_conic_utils as scu
except Exception:
    import os as _os, sys as _sys
    _sys.path.append(_os.path.dirname(__file__) or ".")
    import _3_2_shared_translation_conic_utils as scu

# Wire shared module
scu.conic_at_z = conic_at_z
scu.length_conic_segment = length_conic_segment
scu.volume_ABApBp_general_conic = volume_ABApBp_general_conic
scu.g_val = g_val
scu.roots_x_of_y = roots_x_of_y
scu.roots_y_of_x = roots_y_of_x
scu.pick_branch = pick_branch
scu.flat_plane_zero_volume = flat_plane_zero_volume
if not hasattr(scu, "_PROJ_CACHE"):
    scu._PROJ_CACHE = {}
scu.PRINT_PLANAR_SKIPS = False  # keep rows; planar => zero volume

# ---- columns ----
ID_COLS = ["triangle_id", "A_id", "B_id", "C_id"]
SCALE_COLS = ["scale_factors1_x", "scale_factors1_y", "scale_factors1_z"]
A_T = ["A_transformed_x", "A_transformed_y", "A_transformed_z"]
B_T = ["B_transformed_x", "B_transformed_y", "B_transformed_z"]
C_T = ["C_transformed_x", "C_transformed_y", "C_transformed_z"]
XYZ_T = A_T + B_T + C_T
ABC_NEW_COLS = [
    "ABC_new_A","ABC_new_B","ABC_new_C","ABC_new_D","ABC_new_E",
    "ABC_new_F","ABC_new_G","ABC_new_H","ABC_new_I","ABC_new_J"
]

# ---- helpers ----
def _normalize_coeffs_J_minus1(coeffs, thr=1e-3):
    """
    Return coefficients with J = -1 if |J|>thr; otherwise leave as-is.
    coeffs: iterable (A,B,C,D,E,F,G,H,I,J)
    """
    A,B,C,D,E,F,G,H,I,J = map(float, coeffs)
    if abs(J) > thr:
        s = -1.0 / J
        return (A*s, B*s, C*s, D*s, E*s, F*s, G*s, H*s, I*s, J*s)
    return (A,B,C,D,E,F,G,H,I,J)

def _row_points(row):
    A = np.array([row[A_T[0]], row[A_T[1]], row[A_T[2]]], dtype=float)
    B = np.array([row[B_T[0]], row[B_T[1]], row[B_T[2]]], dtype=float)
    C = np.array([row[C_T[0]], row[C_T[1]], row[C_T[2]]], dtype=float)
    return A, B, C

def _row_coeffs(row):
    coeffs = [row[c] for c in ABC_NEW_COLS]
    return _normalize_coeffs_J_minus1(coeffs)

def _row_scaling(row):
    sx, sy, sz = (float(row[SCALE_COLS[0]]), float(row[SCALE_COLS[1]]), float(row[SCALE_COLS[2]]))
    return sx * sy * sz

def _first_valid_coeffs(df):
    sub = df[ABC_NEW_COLS].dropna(how="any")
    if sub.empty:
        raise ValueError("No complete ABC_new_* row found in CSV.")
    return _normalize_coeffs_J_minus1(sub.iloc[0].tolist())

def _is_planar(A, B, C, coeffs):
    # Prefer project-local guard; fall back to axial plane check if needed
    try:
        return bool(flat_plane_zero_volume(A, B, C, coeffs))
    except TypeError:
        try:
            return bool(flat_plane_zero_volume(coeffs, A, B, C))
        except TypeError:
            # simple geometric heuristic for axis-aligned planes
            n = np.cross(B - A, C - A)
            nn = np.linalg.norm(n)
            if nn < 1e-10:
                return True
            n /= nn
            return (abs(n[0]) > 1 - 1e-6) or (abs(n[1]) > 1 - 1e-6) or (abs(n[2]) > 1 - 1e-6)

def main():
    t0_all = time.time()

    # -------- input / output --------
    in_csv = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_INPUT
    if not os.path.isfile(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")
    stem = os.path.splitext(os.path.basename(in_csv))[0]
    out_csv = f"{stem}_Volume.csv"

    # -------- read + validate --------
    df = pd.read_csv(in_csv)
    missing = [c for c in (ID_COLS + SCALE_COLS + XYZ_T + ABC_NEW_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {in_csv}: {missing}")

    # -------- one-time conic classification (for fast-paths) --------
    COEFFS_REF = _first_valid_coeffs(df)
    scu.KIND, scu.PARAMS = scu.classify_extruded_xy_conic(COEFFS_REF)
    scu.IS_PARABOLA = (scu.KIND == "parabola_y_eq_x2")
    scu.IS_CIRCLE   = (scu.KIND == "circle")
    scu.IS_ELLIPSE  = (scu.KIND == "ellipse")
    scu.IS_HYPERB   = (scu.KIND == "hyperbola")

    # -------- per-row computation --------
    rows = []
    sum_V_raw = 0.0
    sum_V_scal = 0.0

    for _, row in df.iterrows():
        tri_id = int(row["triangle_id"])
        A, B, C = _row_points(row)
        coeffs  = _row_coeffs(row)
        sfac    = _row_scaling(row)

        # default outputs
        V = 0.0
        V_A = V_B = V_C = 0.0
        side = "planar" if _is_planar(A, B, C, coeffs) else None

        if side is None:
            # compute signed patch volume in transformed space
            try:
                V = float(scu.compute_V_patch_from_ABC(coeffs, A, B, C, tri_id))
                side, _pointdir = scu.classify_side_and_direction(A, B, C, coeffs)
                VA, VB, VC = split_patch(V, A, B, C, coeffs)
                V_A, V_B, V_C = float(VA), float(VB), float(VC)
            except Exception as e:
                side = f"error:{type(e).__name__}"

        # V_scal = V * sfac
        # V_A_scal = V_A * sfac
        # V_B_scal = V_B * sfac
        # V_C_scal = V_C * sfac

        V_scal = V * 1
        V_A_scal = V_A * 1
        V_B_scal = V_B * 1
        V_C_scal = V_C * 1

        V = V * sfac
        V_A = V_A * sfac
        V_B = V_B * sfac
        V_C = V_C * sfac

        sum_V_raw  += V
        sum_V_scal += V_scal

        rows.append(dict(
            triangle_id=tri_id,
            Vcorrection=V,
            V_patch_A=V_A, V_patch_B=V_B, V_patch_C=V_C,
            scaling_factor=sfac,
            Vcorrection_scal=V_scal,
            V_patch_A_scal=V_A_scal, V_patch_B_scal=V_B_scal, V_patch_C_scal=V_C_scal,
            side=side,
        ))

    # -------- write output --------
    df_out = pd.DataFrame(rows)
    ordered = [
        "triangle_id",
        "Vcorrection",
        "V_patch_A","V_patch_B","V_patch_C",
        "scaling_factor",
        "Vcorrection_scal",
        "V_patch_A_scal","V_patch_B_scal","V_patch_C_scal",
        "side",
    ]
    df_out = df_out[ordered]
    df_out.to_csv(out_csv, index=False)

    # -------- summary --------
    elapsed = time.time() - t0_all
    print(f"Saved: {out_csv}")
    print(f"Sum of Vcorrection_scal = {sum_V_scal:.12f}")
    print(f"Sum of Vcorrection = {sum_V_raw:.12f}")
    print(f"Rows: {len(df_out)}, Elapsed: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
