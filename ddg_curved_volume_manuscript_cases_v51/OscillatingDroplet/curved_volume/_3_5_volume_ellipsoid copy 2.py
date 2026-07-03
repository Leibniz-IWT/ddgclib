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
# Ensure we can import from your working dir AND /mnt/data (where you uploaded the file).
for p in (os.getcwd(), "/mnt/data"):
    if p not in sys.path:
        sys.path.append(p)

from _4_dualvolume_split_patch_volume_thickness_weighted import (
    split_patch_volume_thickness_weighted,
)

# ----------------- tolerances ------------------
OMEGA_TOL = 1e-14
PATCH_TOL = 1e-15
DEN_TOL   = 1e-18

# ----------------- helpers ---------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0.0 else (v / n)

def _spherical_excess_robust(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Oosterom–Strackee solid angle using atan2 (robust near degeneracy).
    a,b,c are 3D vectors (not necessarily unit).
    """
    u = float(np.dot(a, np.cross(b, c)))  # signed triple product
    v = 1.0 + float(np.dot(a, b)) + float(np.dot(b, c)) + float(np.dot(c, a))
    return 2.0 * math.atan2(abs(u), max(v, DEN_TOL))

def _side_from_orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> str:
    signed_det = float(np.dot(a, np.cross(b, c)))
    return "outward" if signed_det >= 0.0 else "inward"

def _read_scale(row, base_x, base_y, base_z):
    """Read base and optional postscale factors; multiply them componentwise."""
    sx1 = float(row.get(base_x, 1.0))
    sy1 = float(row.get(base_y, 1.0))
    sz1 = float(row.get(base_z, 1.0))

    # optional second/post scales
    sx2 = float(row.get("postscale_x", row.get("scale_factors2_x", 1.0)))
    sy2 = float(row.get("postscale_y", row.get("scale_factors2_y", 1.0)))
    sz2 = float(row.get("postscale_z", row.get("scale_factors2_z", 1.0)))

    return np.array([sx1 * sx2, sy1 * sy2, sz1 * sz2], float)

def _get_coeffs(row) -> tuple:
    """
    Fetch quadric coefficients in the claimed canonical frame.
    Returns (A,B,C,D,E,F,G,H,I,J).
    """
    keys_new = ["ABC_new_A","ABC_new_B","ABC_new_C","ABC_new_D","ABC_new_E",
                "ABC_new_F","ABC_new_G","ABC_new_H","ABC_new_I","ABC_new_J"]
    keys_old = ["ABC_A","ABC_B","ABC_C","ABC_D","ABC_E","ABC_F","ABC_G","ABC_H","ABC_I","ABC_J"]

    if all(k in row for k in keys_new):
        return tuple(float(row[k]) for k in keys_new)
    elif all(k in row for k in keys_old):
        return tuple(float(row[k]) for k in keys_old)
    else:
        raise KeyError("Quadric coefficient columns not found (ABC_new_* or ABC_*).")

def _looks_like_unit_sphere(coeffs, A, B, C, tol=1e-9) -> bool:
    """
    Heuristic: coeffs == (1,1,1,0,0,0,0,0,0,-1) and ||A||≈||B||≈||C||≈1.
    """
    A_,B_,C_,D,E,F,G,H,I,J = coeffs
    coeff_ok = (abs(A_-1.0) < 1e-8 and abs(B_-1.0) < 1e-8 and abs(C_-1.0) < 1e-8 and
                abs(D) < 1e-10 and abs(E) < 1e-10 and abs(F) < 1e-10 and
                abs(G) < 1e-10 and abs(H) < 1e-10 and abs(I) < 1e-10 and abs(J + 1.0) < 1e-8)
    norms_ok = (abs(np.linalg.norm(A) - 1.0) < tol and
                abs(np.linalg.norm(B) - 1.0) < tol and
                abs(np.linalg.norm(C) - 1.0) < tol)
    return coeff_ok and norms_ok

def _split_and_normalize(V_total, A_can, B_can, C_can, coeffs):
    """
    Uses your splitter and then rescales so the parts sum EXACTLY to V_total.
    This removes tiny numerical drift. If splitter returns zeros or NaNs, falls back to equal thirds.
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
        # force exact sum on the last term to kill rounding drift
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
        # clamp
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

        # coefficients (prefer ABC_new_*, fallback to ABC_*)
        coeffs = _get_coeffs(row)

        # scales for JACOBIAN (sphere -> ellipsoid)
        s_csv = _read_scale(row, "scale_factors1_x", "scale_factors1_y", "scale_factors1_z")
        jacobian_scales = np.array(axes_override, float) if axes_override is not None else s_csv
        scaling_factor = float(np.prod(jacobian_scales))

        # scales to map vertices to the SAME canonical frame as coeffs for solid angle / tet
        if _looks_like_unit_sphere(coeffs, A, B, C):
            s_to_unit = np.array([1.0, 1.0, 1.0], float)
        else:
            s_to_unit = jacobian_scales.astype(float)

        # --- map to unit-sphere frame for Omega and Vflat ---
        a = _unit(A / s_to_unit)
        b = _unit(B / s_to_unit)
        c = _unit(C / s_to_unit)

        # spherical wedge and flat tetra (both on unit sphere)
        Omega  = _spherical_excess_robust(a, b, c)
        Vwedge = Omega / 3.0
        Vflat  = abs(float(np.dot(a, np.cross(b, c)))) / 6.0

        # ----- sphere-space correction (unscaled) -----
        Vcorrection_scal = Vwedge - Vflat
        if Vcorrection_scal < 0.0 and abs(Vcorrection_scal) < PATCH_TOL:
            Vcorrection_scal = 0.0

        # ----- ellipsoid-space correction (scaled by det) -----
        Vcorrection = Vcorrection_scal * scaling_factor

        # side/orientation from unit-sphere frame
        side = _side_from_orientation(a, b, c)

        # vertices in the SAME canonical frame as coeffs (for splitting)
        A_can = A / s_to_unit
        B_can = B / s_to_unit
        C_can = C / s_to_unit

        # ---- per-vertex splits computed by your function in BOTH spaces ----
        # sphere-space split -> should sum to Vcorrection_scal
        V_patch_A_scal, V_patch_B_scal, V_patch_C_scal = _split_and_normalize(
            Vcorrection_scal, A_can, B_can, C_can, coeffs
        )
        # ellipsoid-space split -> should sum to Vcorrection
        V_patch_A, V_patch_B, V_patch_C = _split_and_normalize(
            Vcorrection, A_can, B_can, C_can, coeffs
        )

        # -------------------- Curved/flat area + per-vertex area split --------------------
        # High-accuracy O(1) curved area via Dunavant-7 on spherical triangle
        def _sph_dir(p, q, r, wa, wb, wc):
            s = wa*p + wb*q + wc*r
            n = np.linalg.norm(s)
            return s if n == 0.0 else (s / n)

        # Dunavant degree-5 weights/points (sum(weights)=1 on reference triangle)
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

        def _dunavant7_area(p, q, r, Omega_pqr):
            acc = 0.0
            for wa, wb, wc, ww in pts:
                sdir = _sph_dir(p, q, r, wa, wb, wc)  # on unit sphere
                acc += ww * float(np.linalg.norm(inv_scales * sdir))
            return scaling_factor * acc * Omega_pqr  # abc * E[||T^{-T}s||] * Ω

        def _Omega_of(p, q, r):
            u = float(np.dot(p, np.cross(q, r)))
            v = 1.0 + float(np.dot(p, q)) + float(np.dot(q, r)) + float(np.dot(r, p))
            return 2.0 * math.atan2(abs(u), max(v, DEN_TOL))

        # base estimate
        A_curved = _dunavant7_area(a, b, c, Omega)

        # Flat area in the provided transformed (ellipsoid) frame
        A_flat = 0.5 * float(np.linalg.norm(np.cross(B - A, C - A)))

        # Conditional one-shot spherical subdivision if A_curved < A_flat
        if A_curved + 1e-12 < A_flat:
            s = _unit(a + b + c)        # spherical centroid direction
            Om1 = _Omega_of(a, b, s)
            Om2 = _Omega_of(b, c, s)
            Om3 = _Omega_of(c, a, s)
            A1  = _dunavant7_area(a, b, s, Om1)
            A2  = _dunavant7_area(b, c, s, Om2)
            A3  = _dunavant7_area(c, a, s, Om3)
            A_curved_refined = A1 + A2 + A3
            if A_curved_refined >= A_flat or A_curved_refined > A_curved:
                A_curved = A_curved_refined

        # Per-vertex area split using corner-angle proportions (robust, sums exactly)
        angA, angB, angC = _triangle_corner_angles(a, b, c)
        ang_sum = angA + angB + angC
        if ang_sum < DEN_TOL or not np.isfinite(ang_sum):
            A_patch_A = A_patch_B = A_patch_C = A_curved / 3.0
        else:
            wA = angA / ang_sum
            wB = angB / ang_sum
            # last one exact by construction
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
                f"s_to_unit={tuple(float(x) for x in s_to_unit)} | det(ellip)={scaling_factor:.9f}\n"
                f"           Omega={Omega:.12f}  Vw={Vwedge:.12f}  Vf={Vflat:.12f}\n"
                f"           Vcor_sphere={Vcorrection_scal:.12f}  sum(parts_sphere)={sum_scal:.12f}\n"
                f"           Vcor_ellip ={Vcorrection:.12f}  sum(parts_ellip) ={sum_ell:.12f}  side={side}\n"
                f"           A_curved={A_curved:.12f}  A_flat={A_flat:.12f}  "
                f"A_split=({A_patch_A:.12f},{A_patch_B:.12f},{A_patch_C:.12f})"
            )

        out_row = {
            "triangle_id": tri_id,
            # ellipsoid-space totals
            "Vcorrection": Vcorrection,
            "V_patch_A": V_patch_A,     # ELLIPSOID-SPACE per-vertex
            "V_patch_B": V_patch_B,
            "V_patch_C": V_patch_C,
            "scaling_factor": scaling_factor,
            # sphere-space total (so Vcorrection == Vcorrection_scal * scaling_factor)
            "Vcorrection_scal": Vcorrection_scal,
            # sphere-space per-vertex (sums exactly to Vcorrection_scal)
            "V_patch_A_scal": V_patch_A_scal,
            "V_patch_B_scal": V_patch_B_scal,
            "V_patch_C_scal": V_patch_C_scal,
            "side": side,
            # -------------------- area outputs --------------------
            "A_curved": A_curved,
            "A_flat": A_flat,
            "A_patch_A": A_patch_A,
            "A_patch_B": A_patch_B,
            "A_patch_C": A_patch_C,
        }

        if WRITE_DEBUG_COLS:
            out_row.update({
                "Omega_sphere": Omega,
                "Vwedge_sphere": Vwedge,
                "Vflat_sphere": Vflat,
                "s_to_unit_x": float(s_to_unit[0]),
                "s_to_unit_y": float(s_to_unit[1]),
                "s_to_unit_z": float(s_to_unit[2]),
            })

        out_rows.append(out_row)

    # assemble output
    base_cols = [
        "triangle_id","Vcorrection",
        "V_patch_A","V_patch_B","V_patch_C",                # ellipsoid-space per-vertex
        "scaling_factor","Vcorrection_scal",
        "V_patch_A_scal","V_patch_B_scal","V_patch_C_scal", # sphere-space per-vertex
        "side",
        # ------ area columns in output ------
        "A_curved","A_flat","A_patch_A","A_patch_B","A_patch_C",
    ]
    dbg_cols = ["Omega_sphere","Vwedge_sphere","Vflat_sphere",
                "s_to_unit_x","s_to_unit_y","s_to_unit_z"] if WRITE_DEBUG_COLS else []

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
