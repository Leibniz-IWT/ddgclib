#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained quadric transformer utilities (no external _quadric_transformer.py).

Exports (import as needed):
  - EPS, J_FORCE_THR, ABC_COLS, PT_COLS, META_COLS
  - QuadricResult (dataclass)
  - detect_surface_of_revolution, ellipsoid_volume_from_coeffs
  - quadric_major_z_minor_x, evaluate_quadric, postscale_coeffs
  - analyse_quadric
  - is_flat_quadric, transform_point
  - process_file, run_clean_and_batch

Notes:
- Point transform used here:
    p' = R^T (p - centre)
    p' = p' / scale_factors1
- The output coefficients (ABC_new_*) come from analyse_quadric(..., postscale=True).
"""

from __future__ import annotations
import os
import math
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Tuple, List

# ------------------------ global tolerances ------------------------ #
EPS = 1e-12
J_FORCE_THR = 1e-4   # only force J = -1 when |J| is clearly non-tiny
REL_TOL = 1e-4       # default relative tolerance for near-zero checks
ABS_TOL = 1e-8       # default absolute tolerance fallback

# ------------------------ coefficient snapper ---------------------- #
SNAP_TOL = 5e-5

def _snap_scalar_to_unit_or_zero(v: float, tol: float = SNAP_TOL) -> float:
    """Snap a single float to {-1,0,+1} within tol; else return v unchanged."""
    try:
        if not np.isfinite(v):  # keep NaN/inf as-is
            return float(v)
        if abs(v) <= tol: return 0.0
        if abs(v - 1.0) <= tol: return 1.0
        if abs(v + 1.0) <= tol: return -1.0
        return float(v)
    except Exception:
        return float(v)

def _snap_array_to_unit_or_zero(x, tol: float = SNAP_TOL):
    """Vectorized snap: near 0→0, near +1→+1, near -1→-1; else unchanged."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.abs(x) <= tol, 0.0, x)
    x = np.where(np.abs(x - 1.0) <= tol, 1.0, x)
    x = np.where(np.abs(x + 1.0) <= tol, -1.0, x)
    return x

def snap_ABCDJ_columns(df: pd.DataFrame, tol: float = SNAP_TOL) -> pd.DataFrame:
    """
    Snap only the 10 quadric coefficient columns in-place:
      preferred: ABC_new_A..ABC_new_J
      fallback : ABC_A..ABC_J
    """
    abc_new = [f"ABC_new_{k}" for k in "ABCDEFGHIJ"]
    abc_raw = [f"ABC_{k}" for k in "ABCDEFGHIJ"]

    if all(c in df.columns for c in abc_new):
        cols = abc_new
    elif all(c in df.columns for c in abc_raw):
        cols = abc_raw
    else:
        return df  # nothing to snap

    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[cols] = _snap_array_to_unit_or_zero(df[cols].to_numpy(copy=True), tol)
    return df

def write_csv_snapped(df: pd.DataFrame, path: str, tol: float = SNAP_TOL) -> None:
    """Snap only A..J and write with stable float formatting."""
    snap_ABCDJ_columns(df, tol=tol)
    df.to_csv(path, index=False, float_format="%.12g")

# ---- robust near-zero check ----
def _near0(v: float, base: float, rel: float = 1e-4, abs_tol: float = 1e-8) -> bool:
    """Relative+absolute near-zero test against a problem scale."""
    return abs(v) <= max(abs_tol, rel * base)

def _z_invariant_xy_diag_rel(coeffs, rel: float | None = None, abs_tol: float | None = None) -> bool:
    """
    Nearly z-invariant & xy-diagonal:
      C≈0, E≈0, F≈0, I≈0, and D≈0 under relative/absolute tolerances.
    """
    if rel is None: rel = REL_TOL
    if abs_tol is None: abs_tol = ABS_TOL
    A,B,C,D,E,F,G,H,I,J = map(float, coeffs)
    base = max(abs(A), abs(B), abs(C), 1.0)
    return (_near0(C, base, rel, abs_tol) and
            _near0(E, base, rel, abs_tol) and
            _near0(F, base, rel, abs_tol) and
            _near0(I, base, rel, abs_tol) and
            _near0(D, base, rel, abs_tol))

def _detect_parabolic_axes(coeffs, rel: float | None = None, abs_tol: float | None = None):
    """Detect parabolic axis along x or y when z-invariant & xy-diagonal."""
    if rel is None: rel = REL_TOL
    if abs_tol is None: abs_tol = ABS_TOL
    A,B,C,D,E,F,G,H,I,J = map(float, coeffs)
    base = max(abs(A), abs(B), abs(C), 1.0)
    def near0(v): return abs(v) <= max(abs_tol, rel*base)
    zdiag = near0(C) and near0(E) and near0(F) and near0(I) and near0(D)
    if not zdiag: return (False, False)
    baseAB = max(abs(A), abs(B), 1.0)
    thr = max(abs_tol, rel*baseAB)
    return (abs(A) < thr, abs(B) < thr)

# ---------- expected CSV columns ----------
ABC_COLS = [f"ABC_{k}" for k in "ABCDEFGHIJ"]
PT_COLS = {
    "A": ["Ax", "Ay", "Az", "A_id"],
    "B": ["Bx", "By", "Bz", "B_id"],
    "C": ["Cx", "Cy", "Cz", "C_id"],
}
META_COLS = ["triangle_id"]

@dataclass
class QuadricResult:
    type: str
    axis: np.ndarray | None
    centre: np.ndarray
    new_coeffs: Tuple[float, ...]
    scale_factors1: Tuple[float, float, float]
    scale_factors2: Tuple[float, float, float]
    jacobian: float
    R: np.ndarray

    def asdict(self):
        d = asdict(self)
        if d["axis"] is not None:
            d["axis"] = d["axis"].tolist()
        d["centre"] = d["centre"].tolist()
        d["R"] = d["R"].tolist()
        d["new_coeffs"] = list(d["new_coeffs"])
        d["scale_factors1"] = list(d["scale_factors1"])
        d["scale_factors2"] = list(d["scale_factors2"])
        return d

# ------------------------ utilities & classifiers ------------------------ #

def detect_surface_of_revolution(coeffs: Tuple[float, ...], tol: float = 1e-8) -> str:
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    if abs(D) > tol or abs(E) > tol or abs(F) > tol:
        return "not diagonal"
    vals = [A, B, C]
    n_pos = sum(v > tol for v in vals)
    n_neg = sum(v < -tol for v in vals)
    n_zero = 3 - n_pos - n_neg
    if n_pos == 3:             return "ellipsoid of revolution (spheroid)"
    elif n_neg == 3:           return "imaginary ellipsoid of revolution"
    elif n_pos == 2 and n_neg == 1: return "hyperboloid of revolution (one sheet)"
    elif n_pos == 1 and n_neg == 2: return "hyperboloid of revolution (two sheets)"
    elif (n_pos == 2 and n_neg == 1) or (n_pos == 1 and n_neg == 2): return "cone of revolution"
    elif n_pos == 2 and n_zero == 1: return "cylinder of revolution"
    elif n_pos == 1 and n_neg == 1 and n_zero == 1 and abs(G)+abs(H)+abs(I) <= tol:
        return "hyperbolic cylinder"
    elif n_pos == 1 and n_neg == 1 and n_zero == 1 and abs(G)+abs(H)+abs(I) > tol:
        return "hyperbolic paraboloid"
    else:
        return "other/unknown diag form"

def ellipsoid_volume_from_coeffs(A: float, B: float, C: float, J: float) -> float:
    if J >= 0 or A <= 0 or B <= 0 or C <= 0: return float("nan")
    a2, b2, c2 = -J / A, -J / B, -J / C
    a, b, c = math.sqrt(a2), math.sqrt(b2), math.sqrt(c2)
    return 4.0 / 3.0 * math.pi * a * b * c

def _quadric_matrix(A, B, C, D, E, F):
    return np.array([[A, D/2.0, E/2.0],
                     [D/2.0, B, F/2.0],
                     [E/2.0, F/2.0, C]], dtype=float)

def evaluate_quadric(coeffs: Tuple[float, ...], p: np.ndarray) -> float:
    _A, _B, _C, _D, _E, _F, _G, _H, _I, _J = coeffs
    x, y, z = p
    return (_A*x**2 + _B*y**2 + _C*z**2
            + _D*x*y + _E*x*z + _F*y*z
            + _G*x + _H*y + _I*z + _J)

def postscale_coeffs(coeffs: Tuple[float, ...]) -> Tuple[Tuple[float, ...], Tuple[float, float, float]]:
    a1, a2, a3, a4, a5, a6, b1, b2, b3, c = coeffs
    def scale_factor(a: float) -> float:
        return 1.0 / np.sqrt(abs(a)) if abs(a) > 1e-10 else 1.0
    sx = scale_factor(a1)
    sy = scale_factor(a2)
    sz = scale_factor(a3)
    try:
        px, py = _detect_parabolic_axes((a1,a2,a3,a4,a5,a6,b1,b2,b3,c))
    except Exception:
        px, py = (False, False)
    if px: sx = 1.0
    if py: sy = 1.0
    a1_ = a1 * sx * sx
    a2_ = a2 * sy * sy
    a3_ = a3 * sz * sz
    a4_ = a4 * sx * sy
    a5_ = a5 * sx * sz
    a6_ = a6 * sy * sz
    b1_ = b1 * sx
    b2_ = b2 * sy
    b3_ = b3 * sz
    c_  = c
    return (a1_, a2_, a3_, a4_, a5_, a6_, b1_, b2_, b3_, c_), (sx, sy, sz)

def _clean_coeffs(coeffs, rel: float | None = None, abs_tol: float | None = None):
    if rel is None: rel = REL_TOL
    if abs_tol is None: abs_tol = ABS_TOL
    A,B,C,D,E,F,G,H,I,J = map(float, coeffs)
    base = max(abs(A), abs(B), abs(C), 1.0)
    C  = 0.0 if _near0(C, base, rel, abs_tol) else C
    E  = 0.0 if _near0(E, base, rel, abs_tol) else E
    F  = 0.0 if _near0(F, base, rel, abs_tol) else F
    I  = 0.0 if _near0(I, base, rel, abs_tol) else I
    D  = 0.0 if _near0(D, base, rel, abs_tol) else D
    return (A,B,C,D,E,F,G,H,I,J)

def _z_invariant_xy_diag(coeffs, tol=1e-9) -> bool:
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    return (abs(C) < tol and abs(E) < tol and abs(F) < tol and abs(I) < tol and abs(D) < tol)

def quadric_major_z_minor_x(coeffs, tol=1e-8):
    if _z_invariant_xy_diag(coeffs, tol=1e-5):
       #print("Skipping rotation: z-invariant & xy-diagonal form detected.")
        return tuple(map(float, coeffs)), np.eye(3)
    A, B, C, D, E, F, G, H, I_lin, J = coeffs
    Q = np.array([[A, D/2, E/2],
                  [D/2, B, F/2],
                  [E/2, F/2, C]], dtype=float)
    b = np.array([G, H, I_lin], dtype=float)

    base = max(abs(A), abs(B), abs(C), 1.0)
    zmask = np.isclose([A, B, C], 0.0, atol=tol*base + tol)
    if np.count_nonzero(zmask) == 1:
        zero_idx = int(np.where(zmask)[0][0])
        if zero_idx == 0:
            R = np.array([[0., 0., 1.],[0., 1., 0.],[1., 0., 0.]], float)  # swap x<->z
        elif zero_idx == 1:
            R = np.array([[1., 0., 0.],[0., 0., 1.],[0., 1., 0.]], float)  # swap y<->z
        else:
            R = np.eye(3)
        if np.linalg.det(R) < 0: R[:, 2] *= -1
        Qp = R.T @ Q @ R
        bp = R.T @ b
        A_p, B_p, C_p = Qp[0, 0], Qp[1, 1], Qp[2, 2]
        D_p = 2*Qp[0, 1]; E_p = 2*Qp[0, 2]; F_p = 2*Qp[1, 2]
        G_p, H_p, I_p = bp
        return (A_p, B_p, C_p, D_p, E_p, F_p, G_p, H_p, I_p, J), R

    eigvals, eigvecs = np.linalg.eigh(Q)
    abs_eig = np.abs(eigvals)
    null_mask = np.isclose(eigvals, 0.0, atol=tol*np.max(abs_eig) + tol)

    if np.any(null_mask):
        null_basis = eigvecs[:, null_mask]
        c3 = None
        if np.linalg.norm(b) < tol:
            c3 = null_basis[:, 0]
        elif null_basis.shape[1] == 1:
            v = null_basis[:, 0]
            if abs(np.dot(v, b)) < tol: c3 = v
        else:
            n1, n2 = null_basis.T[:2]
            β1, β2 = np.dot(n1, b), np.dot(n2, b)
            if abs(β1) < tol and abs(β2) < tol:
                c3 = n1
            else:
                θ = math.atan2(-β1, β2)
                c3 = math.cos(θ)*n1 + math.sin(θ)*n2
        if c3 is not None:
            c3 /= np.linalg.norm(c3)
            c1 = eigvecs[:, np.argmax(abs_eig)]
            c1 -= np.dot(c1, c3)*c3; c1 /= np.linalg.norm(c1)
            c2 = np.cross(c3, c1);   c2 /= np.linalg.norm(c2)
            R = np.column_stack((c1, c2, c3))
            if np.linalg.det(R) < 0: R[:, 2] *= -1
            Qp = R.T @ Q @ R; bp = R.T @ b
            A_p, B_p, C_p = np.diag(Qp)
            D_p = 2*Qp[0, 1]; E_p = 2*Qp[0, 2]; F_p = 2*Qp[1, 2]
            G_p, H_p, I_p = bp
            if (abs(C_p) < tol and abs(E_p) < tol and abs(F_p) < tol and abs(I_p) < tol):
                return (A_p, B_p, C_p, D_p, E_p, F_p, G_p, H_p, I_p, J), R

    pos_idx = [i for i in range(3) if eigvals[i] > 0]
    neg_idx = [i for i in range(3) if eigvals[i] < 0]
    if len(pos_idx) >= 2: xy_pair = sorted(pos_idx, key=lambda i: -abs_eig[i])[:2]
    elif len(neg_idx) >= 2: xy_pair = sorted(neg_idx, key=lambda i: -abs_eig[i])[:2]
    else: xy_pair = sorted(range(3), key=lambda i: -abs_eig[i])[:2]
    z_idx = ({0, 1, 2} - set(xy_pair)).pop()
    x_idx, y_idx = xy_pair[0], xy_pair[1]
    v_x, v_y, v_z = eigvecs[:, x_idx], eigvecs[:, y_idx], eigvecs[:, z_idx]
    R = np.column_stack((v_x, v_y, v_z))
    if np.linalg.det(R) < 0: R[:, 2] *= -1
    Qp = R.T @ Q @ R; bp = R.T @ b
    A_p, B_p, C_p = np.diag(Qp)
    D_p = 2*Qp[0, 1]; E_p = 2*Qp[0, 2]; F_p = 2*Qp[1, 2]
    G_p, H_p, I_p = bp
    return (A_p, B_p, C_p, D_p, E_p, F_p, G_p, H_p, I_p, J), R

def analyse_quadric(*coeffs: float,
                    eps: float = EPS,
                    prescale: bool = True,
                    postscale: bool = True) -> QuadricResult:
    coeffs_org = tuple(map(float, coeffs))
    zinv_keepI = _z_invariant_xy_diag_rel(coeffs_org)
    coeffs, Rotation = quadric_major_z_minor_x(coeffs_org)

    scale_factors1 = (1.0, 1.0, 1.0)
    if abs(coeffs[-1]) > J_FORCE_THR and abs(coeffs[-1] + 1.0) > 1e-12:
        scale = -1.0 / coeffs[-1]
        coeffs = tuple(c * scale for c in coeffs)

    if prescale:
        coeffs, scale_factors1 = postscale_coeffs(coeffs)

    _A, _B, _C, _D, _E, _F, _G, _H, _I, _J = map(float, coeffs)
    A = _quadric_matrix(_A, _B, _C, _D, _E, _F)

    lam, V = np.linalg.eigh(A)
    zeros = int(np.sum(np.abs(lam) < eps))

    sym_type = "none"
    axis_vec = None
    centre = np.zeros(3)
    R = np.array(Rotation, dtype=float)
    r = None

    if zeros == 0 and np.allclose(lam, lam[0], atol=eps):
        sym_type = "sphere"
        a = _A; g, f, h, c = _G/2.0, _H/2.0, _I/2.0, _J
        centre = np.array((-g/a, -f/a, -h/a))
        r = np.sqrt(abs(g**2 + f**2 + h**2 - a*c)) / abs(a)
    elif zeros == 2:
        sym_type = "translation1"
        axis_vec = V[:, np.argmin(np.abs(lam))]
    elif zeros == 1:
        sym_type = "cylinder"
        axis_vec = V[:, np.argmin(np.abs(lam))]
    else:
        sym_type = detect_surface_of_revolution((_A,_B,_C,0,0,0,0,0,0,_J))

    postscale_factors = (1.0, 1.0, 1.0)
    new_coeffs = tuple(map(float, (_A, _B, _C, _D, _E, _F, _G, _H, _I, _J)))
    if postscale:
        new_coeffs, postscale_factors = postscale_coeffs(new_coeffs)

    jacobian = postscale_factors[0] * postscale_factors[1] * postscale_factors[2]

    if sym_type == "sphere" and r is not None:
        new_coeffs = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -r**2)

    if abs(new_coeffs[-1]) > J_FORCE_THR and abs(new_coeffs[-1] + 1.0) > 1e-12:
        s = -1.0 / new_coeffs[-1]
        new_coeffs = tuple(c * s for c in new_coeffs)

    if zinv_keepI:
        R = np.eye(3)

    return QuadricResult(sym_type, axis_vec, centre, new_coeffs,
                         scale_factors1, postscale_factors, jacobian, R)

def is_flat_quadric(coeffs: Tuple[float, ...], tol: float = 1e-8) -> bool:
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    Q = _quadric_matrix(A,B,C,D,E,F)
    return np.linalg.norm(Q, ord=2) < tol

def transform_point(p: np.ndarray, centre: np.ndarray, R: np.ndarray,
                    scale_factors1: Tuple[float, float, float]) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(3)
    centre = np.asarray(centre, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3,3)
    sxf = np.asarray(scale_factors1, dtype=float).reshape(3)
    return (R.T @ (p - centre)) / sxf

# ======================= CSV processing helpers =======================

def process_file(in_csv: str) -> tuple[str, str | None, int]:
    df = pd.read_csv(in_csv)

    needed = ABC_COLS + PT_COLS["A"] + PT_COLS["B"] + PT_COLS["C"] + META_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {in_csv}: {missing}")

    out_rows: List[dict] = []
    err_rows: List[dict] = []

    for idx, row in df.iterrows():
        tri_id = row["triangle_id"] if "triangle_id" in row else f"row{idx}"
        try:
            coeffs = tuple(float(row[c]) for c in ABC_COLS)

            # flat-plane passthrough
            if is_flat_quadric(coeffs):
                A_xyz = (float(row["Ax"]), float(row["Ay"]), float(row["Az"]))
                B_xyz = (float(row["Bx"]), float(row["By"]))
                B_xyz = (float(row["Bx"]), float(row["By"]), float(row["Bz"]))
                C_xyz = (float(row["Cx"]), float(row["Cy"]), float(row["Cz"]))
                out_rows.append({
                    "triangle_id": float(tri_id),
                    "A_id": float(row["A_id"]), "B_id": float(row["B_id"]), "C_id": float(row["C_id"]),
                    "scale_factors1_x": 1.0, "scale_factors1_y": 1.0, "scale_factors1_z": 1.0,
                    "A_transformed_x": A_xyz[0], "A_transformed_y": A_xyz[1], "A_transformed_z": A_xyz[2],
                    "B_transformed_x": B_xyz[0], "B_transformed_y": B_xyz[1], "B_transformed_z": B_xyz[2],
                    "C_transformed_x": C_xyz[0], "C_transformed_y": C_xyz[1], "C_transformed_z": C_xyz[2],
                    "ABC_new_A": coeffs[0], "ABC_new_B": coeffs[1], "ABC_new_C": coeffs[2],
                    "ABC_new_D": coeffs[3], "ABC_new_E": coeffs[4], "ABC_new_F": coeffs[5],
                    "ABC_new_G": coeffs[6], "ABC_new_H": coeffs[7], "ABC_new_I": coeffs[8],
                    "ABC_new_J": coeffs[9],
                })
                print(f"[FLAT] {os.path.basename(in_csv)} triangle_id={tri_id} → passthrough")
                continue

            info = analyse_quadric(*coeffs, prescale=True, postscale=True)

            sfx, sfy, sfz = info.scale_factors1
            A_xyz = np.array([float(row["Ax"]), float(row["Ay"]), float(row["Az"])])
            B_xyz = np.array([float(row["Bx"]), float(row["By"]), float(row["Bz"])])
            C_xyz = np.array([float(row["Cx"]), float(row["Cy"]), float(row["Cz"])])

            A_t = transform_point(A_xyz, info.centre, info.R, info.scale_factors1)
            B_t = transform_point(B_xyz, info.centre, info.R, info.scale_factors1)
            C_t = transform_point(C_xyz, info.centre, info.R, info.scale_factors1)

            # snap at SOURCE (per-element), so even intermediate dfs are clean
            newA,newB,newC,newD,newE,newF,newG,newH,newI,newJ = [
                _snap_scalar_to_unit_or_zero(v) for v in info.new_coeffs
            ]

            out_rows.append({
                "triangle_id": float(tri_id),
                "A_id": float(row["A_id"]), "B_id": float(row["B_id"]), "C_id": float(row["C_id"]),
                "scale_factors1_x": sfx, "scale_factors1_y": sfy, "scale_factors1_z": sfz,
                "A_transformed_x": float(A_t[0]), "A_transformed_y": float(A_t[1]), "A_transformed_z": float(A_t[2]),
                "B_transformed_x": float(B_t[0]), "B_transformed_y": float(B_t[1]), "B_transformed_z": float(B_t[2]),
                "C_transformed_x": float(C_t[0]), "C_transformed_y": float(C_t[1]), "C_transformed_z": float(C_t[2]),
                "ABC_new_A": newA, "ABC_new_B": newB, "ABC_new_C": newC,
                "ABC_new_D": newD, "ABC_new_E": newE, "ABC_new_F": newF,
                "ABC_new_G": newG, "ABC_new_H": newH, "ABC_new_I": newI,
                "ABC_new_J": newJ,
            })

        except Exception as e:
            err_entry = {
                "triangle_id": float(tri_id) if isinstance(tri_id, (int,float)) else tri_id,
                "A_id": float(row["A_id"]) if "A_id" in row else np.nan,
                "B_id": float(row["B_id"]) if "B_id" in row else np.nan,
                "C_id": float(row["C_id"]) if "C_id" in row else np.nan,
                "Ax": float(row["Ax"]) if "Ax" in row else np.nan,
                "Ay": float(row["Ay"]) if "Ay" in row else np.nan,
                "Az": float(row["Az"]) if "Az" in row else np.nan,
                "Bx": float(row["Bx"]) if "Bx" in row else np.nan,
                "By": float(row["By"]) if "By" in row else np.nan,
                "Bz": float(row["Bz"]) if "Bz" in row else np.nan,
                "Cx": float(row["Cx"]) if "Cx" in row else np.nan,
                "Cy": float(row["Cy"]) if "Cy" in row else np.nan,
                "Cz": float(row["Cz"]) if "Cz" in row else np.nan,
                "error": type(e).__name__,
                "message": str(e),
            }
            for c in ABC_COLS:
                err_entry[c] = row.get(c, np.nan)
            err_rows.append(err_entry)
            continue

    # write successes
    out_df = pd.DataFrame(out_rows)

    # snap at SINK (DataFrame-wide) + clean float formatting
    stem, _ = os.path.splitext(in_csv)
    out_csv = f"{stem}_Transformed.csv"   # e.g. coarse_hyperboloid_COEFFS_Transformed.csv
    write_csv_snapped(out_df, out_csv, tol=SNAP_TOL)

    # write failures if any
    err_csv = None
    if err_rows:
        err_df = pd.DataFrame(err_rows)
        err_csv = f"{stem}_Transformed_Errors.csv"
        err_df.to_csv(err_csv, index=False, float_format="%.12g")

    return out_csv, err_csv, len(err_rows)

# ======================= batch / CLI helpers =======================
def clean_old_transformed() -> None:
    patterns = [
        "*_COEFFS_Transformed.csv",
        "*_COEFFS_Transformed*.csv",
        "*_Transformed_Errors.csv",
    ]
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                os.remove(path)
            except Exception:
                pass

def run_clean_and_batch(pattern: str = "*_COEFFS.csv"):
    clean_old_transformed()
    results = []
    for path in sorted(glob.glob(pattern)):
        try:
            out_csv, err_csv, n_err = process_file(path)
            print(f"Wrote: {out_csv}" + (f" | Errors: {n_err} -> {err_csv}" if n_err else " | No errors."))
            results.append((out_csv, err_csv, n_err))
        except Exception as e:
            print(f"[ERROR] {path}: {type(e).__name__}: {e}")
    return results

if __name__ == "__main__":
    run_clean_and_batch()
