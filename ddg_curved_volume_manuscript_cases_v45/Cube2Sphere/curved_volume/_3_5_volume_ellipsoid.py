#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, time
import numpy as np
import pandas as pd

# ---------------- configuration ----------------
DEBUG_PER_ROW = True     # set False to silence per-row debug prints
WRITE_DEBUG_COLS = True  # set False to keep only the core output columns

# If you know the true ellipsoid semiaxes (in this transformed frame), set them here.
# Otherwise, the script uses scale_factors1_* (× scale_factors2_* if present) from the CSV.
# Example: axes_override = (1.5, 1.0, 0.8)
axes_override = None

# -------------- import your splitter -----------
for p in (os.getcwd(), "/mnt/data"):
    if p not in sys.path:
        sys.path.append(p)

from _4_dualvolume_split_patch_volume_thickness_weighted import (
    split_patch_volume_thickness_weighted,
)

# we now also need the curvature utils, implemented in the same way
from _3_4_shared_axisymmetric_utils import (
    principal_curvatures_at,
    principal_curvatures_rule_label,
    OBSERVER_OUTSIDE,
)

# ----------------- tolerances ------------------
OMEGA_TOL = 1e-14
PATCH_TOL = 1e-15
DEN_TOL   = 1e-18

# ----------------- helpers ---------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0.0 else (v / n)

def _spherical_excess_robust_unitdirs(a_u: np.ndarray, b_u: np.ndarray, c_u: np.ndarray) -> float:
    """
    Oosterom–Strackee solid angle using atan2.
    Inputs MUST be unit directions.
    """
    u = float(np.dot(a_u, np.cross(b_u, c_u)))  # signed triple product
    v = 1.0 + float(np.dot(a_u, b_u)) + float(np.dot(b_u, c_u)) + float(np.dot(c_u, a_u))
    return 2.0 * math.atan2(abs(u), max(v, DEN_TOL))

def _side_from_orientation_unitdirs(a_u: np.ndarray, b_u: np.ndarray, c_u: np.ndarray) -> str:
    signed_det = float(np.dot(a_u, np.cross(b_u, c_u)))
    return "outward" if signed_det >= 0.0 else "inward"

def _read_scale(row, base_x, base_y, base_z):
    """Read base and optional postscale factors; multiply them componentwise."""
    sx1 = float(row.get(base_x, 1.0))
    sy1 = float(row.get(base_y, 1.0))
    sz1 = float(row.get(base_z, 1.0))
    sx2 = float(row.get("postscale_x", row.get("scale_factors2_x", 1.0)))
    sy2 = float(row.get("postscale_y", row.get("scale_factors2_y", 1.0)))
    sz2 = float(row.get("postscale_z", row.get("scale_factors2_z", 1.0)))
    return np.array([sx1 * sx2, sy1 * sy2, sz1 * sz2], float)

def _get_coeffs(row) -> tuple:
    """Fetch quadric coefficients (A,B,C,D,E,F,G,H,I,J)."""
    keys_new = ["ABC_new_A","ABC_new_B","ABC_new_C","ABC_new_D","ABC_new_E",
                "ABC_new_F","ABC_new_G","ABC_new_H","ABC_new_I","ABC_new_J"]
    keys_old = ["ABC_A","ABC_B","ABC_C","ABC_D","ABC_E","ABC_F","ABC_G","ABC_H","ABC_I","ABC_J"]
    if all(k in row for k in keys_new):
        return tuple(float(row[k]) for k in keys_new)
    elif all(k in row for k in keys_old):
        return tuple(float(row[k]) for k in keys_old)
    else:
        raise KeyError("Quadric coefficient columns not found (ABC_new_* or ABC_*).")

def _split_and_normalize(V_total, A_can, B_can, C_can, coeffs):
    """
    Uses your splitter and then rescales so the parts sum EXACTLY to V_total.
    """
    if not np.isfinite(V_total) or abs(V_total) <= PATCH_TOL:
        return 0.0, 0.0, 0.0

    vA, vB, vC = split_patch_volume_thickness_weighted(
        V_total, A_can, B_can, C_can, coeffs, eps=1e-12
    )
    s = vA + vB + vC
    if (not np.isfinite(s)) or abs(s) < 1e-30:
        vA = vB = vC = V_total / 3.0
    else:
        scale = V_total / s
        vA *= scale
        vB *= scale
        vC = V_total - vA - vB
    return float(vA), float(vB), float(vC)

def _triangle_corner_angles(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[float,float,float]:
    """
    Corner angles (in radians) of the Euclidean triangle with vertices u,v,w.
    Robust to tiny edges via DEN_TOL.
    """
    def ang(p, q, r):
        qp = q - p
        rp = r - p
        n1 = np.linalg.norm(qp)
        n2 = np.linalg.norm(rp)
        if n1 < DEN_TOL or n2 < DEN_TOL:
            return 0.0
        cos_t = float(np.dot(qp, rp) / max(n1*n2, DEN_TOL))
        cos_t = max(-1.0, min(1.0, cos_t))
        return math.acos(cos_t)
    A = ang(u, v, w)
    B = ang(v, w, u)
    C = ang(w, u, v)
    return A, B, C

# ----------------- main processing ---------------------
def process_coeffs_transformed_csv(input_csv: str):
    df = pd.read_csv(input_csv)

    out_rows = []
    planar_skipped = 0

    for _, row in df.iterrows():
        tri_id = int(row["triangle_id"])

        # vertices in the provided "transformed" frame
        A = np.array([row["A_transformed_x"], row["A_transformed_y"], row["A_transformed_z"]], float)
        B = np.array([row["B_transformed_x"], row["B_transformed_y"], row["B_transformed_z"]], float)
        C = np.array([row["C_transformed_x"], row["C_transformed_y"], row["C_transformed_z"]], float)
        A_o = np.array([row["Ax"], row["Ay"], row["Az"]], float)
        B_o = np.array([row["Bx"], row["By"], row["Bz"]], float)
        C_o = np.array([row["Cx"], row["Cy"], row["Cz"]], float)
        coeffs = _get_coeffs(row)

        # JACOBIAN scales (sphere -> ellipsoid) still used for area scaling
        s_csv = _read_scale(row, "scale_factors1_x", "scale_factors1_y", "scale_factors1_z")
        jacobian_scales = np.array(axes_override, float) if axes_override is not None else s_csv
        scaling_factor = float(np.prod(jacobian_scales))

        # ====== use raw A,B,C directly ======
        a = A.copy()
        b = B.copy()
        c = C.copy()

        # Save (raw) a,b,c components
        a_x, a_y, a_z = float(a[0]), float(a[1]), float(a[2])
        b_x, b_y, b_z = float(b[0]), float(b[1]), float(b[2])
        c_x, c_y, c_z = float(c[0]), float(c[1]), float(c[2])

        # Unit directions for spherical formulas (internal only)
        au = _unit(a); bu = _unit(b); cu = _unit(c)

        # spherical wedge and flat tetra (both with unit directions)
        Omega  = _spherical_excess_robust_unitdirs(au, bu, cu)
        Vwedge = Omega / 3.0
        Vflat  = abs(float(np.dot(au, np.cross(bu, cu)))) / 6.0

        # sphere-space correction (unscaled)
        Vcorrection_scal = Vwedge - Vflat
        if Vcorrection_scal < 0.0 and abs(Vcorrection_scal) < PATCH_TOL:
            Vcorrection_scal = 0.0

        # ellipsoid-space correction (scaled by det)
        Vcorrection = Vcorrection_scal * scaling_factor

        # ---------------------------------------------------
        # NEW: compute principal curvatures PER VERTEX
        A_k1, A_k2 = principal_curvatures_at(A, coeffs=coeffs, project=True, observer_outside=OBSERVER_OUTSIDE)
        B_k1, B_k2 = principal_curvatures_at(B, coeffs=coeffs, project=True, observer_outside=OBSERVER_OUTSIDE)
        C_k1, C_k2 = principal_curvatures_at(C, coeffs=coeffs, project=True, observer_outside=OBSERVER_OUTSIDE)

        # triangle-level side from the 3 vertex k2's, same rule as worker(...)
        label = principal_curvatures_rule_label(A_k2, B_k2, C_k2)
        if label == "outward":
            side = "outside"
        elif label == "inward":
            side = "inside"
        else:
            # fallback to geometric orientation (keep your old behavior)
            side = _side_from_orientation_unitdirs(au, bu, cu)
        # ---------------------------------------------------

        # ====== feed raw A,B,C to splitter ======
        A_can = A
        B_can = B
        C_can = C

        # per-vertex splits computed in BOTH spaces
        V_patch_A_scal, V_patch_B_scal, V_patch_C_scal = _split_and_normalize(
            Vcorrection_scal, A_can, B_can, C_can, coeffs
        )

        V_patch_A = V_patch_A_scal * scaling_factor
        V_patch_B = V_patch_B_scal * scaling_factor
        V_patch_C = V_patch_C_scal * scaling_factor

        # -------------------- Curved/flat area + per-vertex area split --------------------
        def _sph_dir(p_u, q_u, r_u, wa, wb, wc):
            s = wa*p_u + wb*q_u + wc*r_u
            n = np.linalg.norm(s)
            return s if n == 0.0 else (s / n)

        # Dunavant degree-5 weights/points
        w_c  = 0.225
        w_ab = 0.13239415278850616
        w_gd = 0.12593918054482715
        alpha, beta  = 0.05971587178976981, 0.47014206410511505
        gamma, delta = 0.7974269853530872, 0.10128650732345633

        pts = [
            (1.0/3.0, 1.0/3.0, 1.0/3.0, w_c),
            (alpha, beta,  beta,  w_ab),
            (beta,  alpha, beta,  w_ab),
            (beta,  beta,  alpha, w_ab),
            (gamma, delta, delta, w_gd),
            (delta, gamma, delta, w_gd),
            (delta, delta, gamma, w_gd),
        ]

        inv_scales = np.array([1.0 / max(jacobian_scales[0], DEN_TOL),
                               1.0 / max(jacobian_scales[1], DEN_TOL),
                               1.0 / max(jacobian_scales[2], DEN_TOL)], float)

        def _dunavant7_area(p_u, q_u, r_u, Omega_pqr):
            acc = 0.0
            for wa, wb, wc, ww in pts:
                sdir = _sph_dir(p_u, q_u, r_u, wa, wb, wc)  # unit direction on sphere
                acc += ww * float(np.linalg.norm(inv_scales * sdir))
            return scaling_factor * acc * Omega_pqr  # abc * E[||T^{-T}s||] * Ω

        def _Omega_of(p_u, q_u, r_u):
            u = float(np.dot(p_u, np.cross(q_u, r_u)))
            v = 1.0 + float(np.dot(p_u, q_u)) + float(np.dot(q_u, r_u)) + float(np.dot(r_u, p_u))
            return 2.0 * math.atan2(abs(u), max(v, DEN_TOL))

        # base estimate with unit directions
        A_curved = _dunavant7_area(au, bu, cu, Omega)

        # Flat area in the provided transformed (ellipsoid) frame
        A_flat = 0.5 * float(np.linalg.norm(np.cross(B_o - A_o, C_o - A_o)))

        # Conditional one-shot spherical subdivision if A_curved < A_flat
        if A_curved + 1e-12 < A_flat:
            s_u = _unit(au + bu + cu)        # spherical centroid direction (unit)
            Om1 = _Omega_of(au, bu, s_u)
            Om2 = _Omega_of(bu, cu, s_u)
            Om3 = _Omega_of(cu, au, s_u)
            A1  = _dunavant7_area(au, bu, s_u, Om1)
            A2  = _dunavant7_area(bu, cu, s_u, Om2)
            A3  = _dunavant7_area(cu, au, s_u, Om3)
            A_curved_refined = A1 + A2 + A3
            if A_curved_refined >= A_flat or A_curved_refined > A_curved:
                A_curved = A_curved_refined

        # Per-vertex area split using corner-angle proportions in Euclidean space of unit dirs
        angA, angB, angC = _triangle_corner_angles(au, bu, cu)
        ang_sum = angA + angB + angC
        if ang_sum < DEN_TOL or not np.isfinite(ang_sum):
            A_patch_A = A_patch_B = A_patch_C = A_curved / 3.0
        else:
            wA = angA / ang_sum
            wB = angB / ang_sum
            A_patch_A = A_curved * wA
            A_patch_B = A_curved * wB
            A_patch_C = A_curved - A_patch_A - A_patch_B
        # ---------------------------------------------------------------------------------------

        if (Omega < OMEGA_TOL) or (abs(Vcorrection) <= PATCH_TOL):
            planar_skipped += 1

        if DEBUG_PER_ROW:
            sum_scal = V_patch_A_scal + V_patch_B_scal + V_patch_C_scal
            sum_ell  = V_patch_A + V_patch_B + V_patch_C
            print(
                f"[tri {tri_id}] ||A||={np.linalg.norm(A):.9f} ||B||={np.linalg.norm(B):.9f} ||C||={np.linalg.norm(C):.9f} | "
                f"det(ellip)={scaling_factor:.9f}\n"
                f"           Omega={Omega:.12f}  Vw={Vwedge:.12f}  Vf={Vflat:.12f}\n"
                f"           Vcor_sphere={Vcorrection_scal:.12f}  sum(parts_sphere)={sum_scal:.12f}\n"
                f"           Vcor_ellip ={Vcorrection:.12f}  sum(parts_ellip) ={sum_ell:.12f}  side={side}\n"
                f"           A_curved={A_curved:.12f}  A_flat={A_flat:.12f}  "
                f"A_split=({A_patch_A:.12f},{A_patch_B:.12f},{A_patch_C:.12f})\n"
                f"           a=({a_x:.12f},{a_y:.12f},{a_z:.12f})  "
                f"b=({b_x:.12f},{b_y:.12f},{b_z:.12f})  c=({c_x:.12f},{c_y:.12f},{c_z:.12f})"
            )

        out_row = {
            "triangle_id": tri_id,
            # ellipsoid-space totals
            "Vcorrection": Vcorrection,
            "V_patch_A": V_patch_A,
            "V_patch_B": V_patch_B,
            "V_patch_C": V_patch_C,
            "scaling_factor": scaling_factor,
            # sphere-space total
            "Vcorrection_scal": Vcorrection_scal,
            # sphere-space per-vertex
            "V_patch_A_scal": V_patch_A_scal,
            "V_patch_B_scal": V_patch_B_scal,
            "V_patch_C_scal": V_patch_C_scal,
            # curvature-based side + per-vertex curvatures
            "A_k1": A_k1, "A_k2": A_k2,
            "B_k1": B_k1, "B_k2": B_k2,
            "C_k1": C_k1, "C_k2": C_k2,
            "side": side,
            # -------------------- area outputs --------------------
            "A_curved": A_curved,
            "A_flat": A_flat,
            "A_patch_A": A_patch_A,
            "A_patch_B": A_patch_B,
            "A_patch_C": A_patch_C,
            # ----------- saved raw a,b,c (now identical to A,B,C) -----------
            "a_x": a_x, "a_y": a_y, "a_z": a_z,
            "b_x": b_x, "b_y": b_y, "b_z": b_z,
            "c_x": c_x, "c_y": c_y, "c_z": c_z,
        }

        if WRITE_DEBUG_COLS:
            out_row.update({
                "Omega_sphere": Omega,
                "Vwedge_sphere": Vwedge,
                "Vflat_sphere": Vflat,
            })

        out_rows.append(out_row)

    # assemble output
    base_cols = [
        "triangle_id","Vcorrection",
        "V_patch_A","V_patch_B","V_patch_C",
        "scaling_factor","Vcorrection_scal",
        "V_patch_A_scal","V_patch_B_scal","V_patch_C_scal",
        # replaced old k1,k2 with per-vertex ones
        "A_k1","A_k2","B_k1","B_k2","C_k1","C_k2",
        "side",
        "A_curved","A_flat","A_patch_A","A_patch_B","A_patch_C",
        "a_x","a_y","a_z","b_x","b_y","b_z","c_x","c_y","c_z",
    ]
    dbg_cols = ["Omega_sphere","Vwedge_sphere","Vflat_sphere"] if WRITE_DEBUG_COLS else []

    out_df = pd.DataFrame(out_rows, columns=base_cols + dbg_cols)
    out_csv = os.path.splitext(input_csv)[0] + "_Volume.csv"
    out_df.to_csv(out_csv, index=False)

    curved_count = len(out_df)
    V_sum = float(np.nansum(out_df["Vcorrection"].values)) if curved_count else 0.0
    return out_csv, curved_count, planar_skipped, V_sum

# ----------------- CLI ---------------------
if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "Ellip_0_sub0_full_COEFFS_Transformed.csv"
    t0 = time.time()
    out_csv, curved_count, planar_skipped, V_sum = process_coeffs_transformed_csv(in_csv)
    t1 = time.time()
    print(f"[input] {in_csv}")
    print(f"Triangles written: {curved_count} | (degenerate≈zero) flagged: {planar_skipped}")
    print(f"Patch-based total (sum of Vcorrection): {V_sum:.8f}")
    print(f"Total elapsed: {t1 - t0:.2f} s")
    print(f"[write] {os.path.abspath(out_csv)}")
