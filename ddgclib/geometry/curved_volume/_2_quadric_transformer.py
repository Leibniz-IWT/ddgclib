#!/usr/bin/env python3
# coding: utf-8
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
    p' = R^T (p) / scale_factors1 - centre_canon
  where centre_canon is the **canonical-frame center** to subtract.
  If final linear terms (G3,H3,I3) are all ~0, the canonical center is **(0,0,0)**.
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
J_FORCE_THR = 1e-5   # only force J = -1 when |J| is clearly non-tiny
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

# robust near-zero check
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

def _is_diagonal(coeffs, tol=1e-9) -> bool:
    """Diagonal iff off-diagonals are (near) zero: D≈E≈F≈0."""
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    return (abs(D) < tol and abs(E) < tol and abs(F) < tol
            and abs(A) >= tol and abs(B) >= tol and abs(C) >= tol
            )

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

# expected CSV columns
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

def detect_surface_of_revolution(coeffs: Tuple[float, ...], tol: float = 1e-8) -> Tuple[np.ndarray | None, str]:
    """
    Returns (axis_vector_or_None, classification_string).
    Only handles diagonal forms (D=E=F=0). For non-diagonal, returns (None, "not diagonal").
    Axis is one of the canonical basis vectors when determinable.
    """
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    if abs(D) > tol or abs(E) > tol or abs(F) > tol:
        return None, "not diagonal"

    ex = np.array([1.0, 0.0, 0.0]); ey = np.array([0.0, 1.0, 0.0]); ez = np.array([0.0, 0.0, 1.0])
    vals = [A, B, C]
    n_pos = sum(v > tol for v in vals)
    n_neg = sum(v < -tol for v in vals)
    n_zero = 3 - n_pos - n_neg

    axis = None
    eqAB = abs(A - B) <= tol
    eqAC = abs(A - C) <= tol
    eqBC = abs(B - C) <= tol
    if eqAB and not eqAC:   axis = ez
    elif eqAC and not eqAB: axis = ey
    elif eqBC and not eqAB: axis = ex
    if axis is None and n_zero == 1:
        if abs(A) <= tol: axis = ex
        elif abs(B) <= tol: axis = ey
        elif abs(C) <= tol: axis = ez

    if n_pos == 3:
        name = "ellipsoid of revolution (spheroid)" if (eqAB or eqAC or eqBC) else "ellipsoid (no revolution)"
    elif n_neg == 3:
        name = "imaginary ellipsoid of revolution" if (eqAB or eqAC or eqBC) else "imaginary ellipsoid"
    elif n_pos == 2 and n_neg == 1:
        name = "hyperboloid of revolution (one sheet)" if (eqAB or eqAC or eqBC) else "hyperboloid (one sheet)"
    elif n_pos == 1 and n_neg == 2:
        name = "hyperboloid of revolution (two sheets)" if (eqAB or eqAC or eqBC) else "hyperboloid (two sheets)"
    elif (n_pos == 2 and n_zero == 1):
        name = "cylinder of revolution" if (eqAB or eqAC or eqBC) else "cylinder"
    elif (n_pos == 1 and n_neg == 1 and n_zero == 1 and abs(G)+abs(H)+abs(I) <= tol):
        name = "hyperbolic cylinder"
    elif (n_pos == 1 and n_neg == 1 and n_zero == 1 and abs(G)+abs(H)+abs(I) > tol):
        name = "hyperbolic paraboloid"
    elif (n_pos == 1 and n_zero == 2):
        name = "parabolic cylinder"
    else:
        name = "other/unknown diag form"

    return axis, name

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

# NEW: permutation to force unique principal axis -> z (fast path only)
def _permute_to_unique_axis_z(A: float, B: float, C: float, tol: float = 1e-10) -> np.ndarray:
    """
    Return permutation R so the unique principal axis (if any) becomes z.
    If A≈B≠C -> unique is C (already z) -> I
       A≈C≠B -> unique is B (y)       -> swap y<->z
       B≈C≠A -> unique is A (x)       -> swap x<->z
    Otherwise (sphere/fully distinct/noisy): I
    """
    I = np.eye(3)
    RXZ = np.array([[0., 0., 1.],
                    [0., 1., 0.],
                    [1., 0., 0.]], float)  # x<->z
    RYZ = np.array([[1., 0., 0.],
                    [0., 0., 1.],
                    [0., 1., 0.]], float)  # y<->z
    eqAB = abs(A - B) <= tol
    eqAC = abs(A - C) <= tol
    eqBC = abs(B - C) <= tol
    if eqAB and not eqAC:
        return I            # unique already z
    if eqAC and not eqAB:
        return RYZ          # unique y -> map to z
    if eqBC and not eqAB:
        return RXZ          # unique x -> map to z
    return I

def quadric_major_z_minor_x(coeffs, tol=1e-8, offdiag_rel=1e-4): #offdiag_rel=1e-3
    if _is_diagonal(coeffs, tol=1e-5):
        A, B, C, D, E, F, G, H, I_lin, J = map(float, coeffs)
        R0 = _permute_to_unique_axis_z(A, B, C, tol=1e-10)
        if not np.allclose(R0, np.eye(3)):
            Q0 = np.array([[A, D/2.0, E/2.0],
                           [D/2.0, B, F/2.0],
                           [E/2.0, F/2.0, C]], dtype=float)
            b0 = np.array([G, H, I_lin], dtype=float)
            Qp = R0.T @ Q0 @ R0
            bp = R0.T @ b0
            Ap, Bp, Cp = float(Qp[0,0]), float(Qp[1,1]), float(Qp[2,2])
            Gp, Hp, Ip = map(float, bp)
            # off-diagonals remain zero after a pure axis permutation
            return (Ap, Bp, Cp, 0.0, 0.0, 0.0, Gp, Hp, Ip, J), R0
    # fast exit if already z-invariant & xy-diagonal
    if _z_invariant_xy_diag(coeffs, tol=1e-5):
        return tuple(map(float, coeffs)), np.eye(3)

    # unpack
    A, B, C, D, E, F, G, H, I_lin, J = map(float, coeffs)

    # zmask swap (if exactly one of A,B,C ~ 0)
    base = max(abs(A), abs(B), abs(C), 1.0)
    zmask = np.isclose([A, B, C], 0.0, atol=tol * base + tol)
    R_acc = np.eye(3)
    if np.count_nonzero(zmask) == 1:
        zero_idx = int(np.where(zmask)[0][0])
        if zero_idx == 0:
            R0 = np.array([[0., 0., 1.],
                           [0., 1., 0.],
                           [1., 0., 0.]], float)  # swap x<->z
        elif zero_idx == 1:
            R0 = np.array([[1., 0., 0.],
                           [0., 0., 1.],
                           [0., 1., 0.]], float)  # swap y<->z
        else:
            R0 = np.eye(3)

        Q0 = np.array([[A, D/2.0, E/2.0],
                       [D/2.0, B, F/2.0],
                       [E/2.0, F/2.0, C]], dtype=float)
        b0 = np.array([G, H, I_lin], dtype=float)
        Qp = R0.T @ Q0 @ R0
        bp = R0.T @ b0
        A, B, C = float(Qp[0, 0]), float(Qp[1, 1]), float(Qp[2, 2])
        D, E, F = 2*float(Qp[0, 1]), 2*float(Qp[0, 2]), 2*float(Qp[1, 2])
        G, H, I_lin = map(float, bp)
        R_acc = R0

    # near-diagonal snap to kill tiny off-diagonals (prevents eigen-rotate)
    base = max(abs(A), abs(B), abs(C), 1.0)
    if max(abs(D), abs(E), abs(F)) <= offdiag_rel * base:
        return (A, B, C, 0.0, 0.0, 0.0, G, H, I_lin, J), R_acc

    # small-absolute off-diagonal check
    tol_off = tol * base + tol
    if abs(D) <= tol_off and abs(E) <= tol_off and abs(F) <= tol_off:
        return (A, B, C, 0.0, 0.0, 0.0, G, H, I_lin, J), R_acc

    # eigen path
    Q = np.array([[A, D/2.0, E/2.0],
                  [D/2.0, B, F/2.0,],
                  [E/2.0, F/2.0, C ]], dtype=float)
    b = np.array([G, H, I_lin], dtype=float)

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
            if abs(np.dot(v, b)) < tol:
                c3 = v
        else:
            n1, n2 = null_basis.T[:2]
            β1, β2 = np.dot(n1, b), np.dot(n2, b)
            if abs(β1) < tol and abs(β2) < tol:
                c3 = n1
            else:
                θ = math.atan2(-β1, β2)
                c3 = math.cos(θ) * n1 + math.sin(θ) * n2
        if c3 is not None:
            c3 /= np.linalg.norm(c3)
            c1 = eigvecs[:, np.argmax(abs_eig)]
            c1 -= np.dot(c1, c3) * c3; c1 /= np.linalg.norm(c1)
            c2 = np.cross(c3, c1);     c2 /= np.linalg.norm(c2)
            R_eig = np.column_stack((c1, c2, c3))
            if np.linalg.det(R_eig) < 0:
                R_eig[:, 2] *= -1
            R = R_acc @ R_eig

            Qp = R_eig.T @ Q @ R_eig
            bp = R_eig.T @ b
            A_p, B_p, C_p = np.diag(Qp)
            D_p = 2 * Qp[0, 1]; E_p = 2 * Qp[0, 2]; F_p = 2 * Qp[1, 2]
            G_p, H_p, I_p = bp
            thr = tol * max(np.max(abs_eig), 1.0) + tol
            if abs(D_p) < thr: D_p = 0.0
            if abs(E_p) < thr: E_p = 0.0
            if abs(F_p) < thr: F_p = 0.0
            return (A_p, B_p, C_p, D_p, E_p, F_p, G_p, H_p, I_p, J), R

    # choose x,y from the two largest-|eig| directions
    pos_idx = [i for i in range(3) if eigvals[i] > 0]
    neg_idx = [i for i in range(3) if eigvals[i] < 0]
    if len(pos_idx) >= 2:
        xy_pair = sorted(pos_idx, key=lambda i: -abs_eig[i])[:2]
    elif len(neg_idx) >= 2:
        xy_pair = sorted(neg_idx, key=lambda i: -abs_eig[i])[:2]
    else:
        xy_pair = sorted(range(3), key=lambda i: -abs_eig[i])[:2]
    z_idx = ({0, 1, 2} - set(xy_pair)).pop()

    v_x, v_y, v_z = eigvecs[:, xy_pair[0]], eigvecs[:, xy_pair[1]], eigvecs[:, z_idx]
    R_eig = np.column_stack((v_x, v_y, v_z))
    if np.linalg.det(R_eig) < 0:
        R_eig[:, 2] *= -1
    R = R_acc @ R_eig

    Qp = R_eig.T @ Q @ R_eig
    bp = R_eig.T @ b
    A_p, B_p, C_p = np.diag(Qp)
    D_p = 2 * Qp[0, 1]; E_p = 2 * Qp[0, 2]; F_p = 2 * Qp[1, 2]
    G_p, H_p, I_p = bp

    thr = tol * max(np.max(abs_eig), 1.0) + tol
    if abs(D_p) < thr: D_p = 0.0
    if abs(E_p) < thr: E_p = 0.0
    if abs(F_p) < thr: F_p = 0.0

    return (A_p, B_p, C_p, D_p, E_p, F_p, G_p, H_p, I_p, J), R

def analyse_quadric(*coeffs: float,
                    eps: float = EPS,
                    prescale: bool = True,   # kept for signature compat; we scale once at the very end
                    postscale: bool = True) -> QuadricResult:
    # --------- NEW: normalize J to {-1, 0} before anything else --------- #
    coeffs_in = tuple(map(float, coeffs))
    coeffs_org  = tuple(map(float, coeffs))
    print("!!!!debug:input coeffs (raw):", coeffs_in)

    A0,B0,C0,D0,E0,F0,G0,H0,I0,J0 = coeffs_in
    print("A0,B0,C0,D0,E0,F0,G0,H0,I0,J0 =", A0,B0,C0,D0,E0,F0,G0,H0,I0,J0)


    # 1) Rotate based on Q (no scaling here)
    zinv_keepI = _z_invariant_xy_diag_rel(coeffs_org)
    coeffs_rot, Rotation = quadric_major_z_minor_x(coeffs_org)
    print("!!!!debug:after quadric_major_z_minor_x coeffs_rot:", coeffs_rot)
    print("!!!!debug:after quadric_major_z_minor_x Rotation:\n", Rotation)

    # snap tiny coeffs to zero to avoid crazy scales later
    coeffs_rot = tuple(0.0 if abs(float(c)) < SNAP_TOL else float(c) for c in coeffs_rot)

    _A,_B,_C,_D,_E,_F,_G,_H,_I,_J = map(float, coeffs_rot)
    A = _quadric_matrix(_A,_B,_C,_D,_E,_F)      # 3x3 symmetric
    b = np.array([_G,_H,_I], float)             # linear terms

    # eigen-structure of quadratic form
    lam, V = np.linalg.eigh(A)
    zeros = int(np.sum(np.abs(lam) < eps))
    pinv_rcond = max(ABS_TOL, REL_TOL * float(np.max(np.abs(lam)) or 1.0))

    # classify symmetry (rough)
    sym_type = "none"
    axis_vec = None
    R = np.array(Rotation, float)

    if zeros == 0 and np.allclose(lam, lam[0], atol=eps):
        sym_type = "sphere"
    elif zeros == 0:
        sym_type = "translation1"; axis_vec = V[:, np.argmin(np.abs(lam))]
    elif zeros == 1:
        sym_type = "cylinder";     axis_vec = V[:, np.argmin(np.abs(lam))]
    else:
        # fall back to a RoS detector that ignores linear terms
        axis_vec, _ = detect_surface_of_revolution((_A,_B,_C,0,0,0,0,0,0,_J))

    # 2) Complete the square in this rotated frame (center translation)
    if zeros == 0:
        # full-rank: exact solve
        c_rot = -0.5 * np.linalg.solve(A, b)
    elif zeros == 1:
        # rank-2: pseudoinverse in the quadratic subspace
        c_rot = -0.5 * (np.linalg.pinv(A, rcond=pinv_rcond) @ b)
    elif zeros == 2:
        # rank-1: per-axis formula where coefficient is non-tiny
        dx = -_G/(2.0*_A) if abs(_A) > pinv_rcond else 0.0
        dy = -_H/(2.0*_B) if abs(_B) > pinv_rcond else 0.0
        dz = -_I/(2.0*_C) if abs(_C) > pinv_rcond else 0.0
        c_rot = np.array([dx,dy,dz], float)
    else:
        # rank-0 (plane): start at origin; we can still shift later to kill J
        c_rot = np.zeros(3, float)

    dx,dy,dz = map(float, c_rot)
    G1 = _G + 2*_A*dx + _D*dy + _E*dz
    H1 = _H + 2*_B*dy + _D*dx + _F*dz
    I1 = _I + 2*_C*dz + _E*dx + _F*dy
    J1 = (_J + b @ c_rot + c_rot @ A @ c_rot)

    # === Zero linear terms only if they cancel numerically after the translation
    if abs(G1) <= SNAP_TOL: G1 = 0.0
    if abs(H1) <= SNAP_TOL: H1 = 0.0
    if abs(I1) <= SNAP_TOL: I1 = 0.0

    # 3) Optional: rotate inside a (near) nullspace AFTER translation to collapse (H1,I1)->(±||·||,0)
    a1, b1, c1 = _A, _B, _C
    zero_mask = [abs(a1) <= SNAP_TOL, abs(b1) <= SNAP_TOL, abs(c1) <= SNAP_TOL]
    if sum(zero_mask) >= 2:
        plane = [i for i, z in enumerate(zero_mask) if z][:2]
        i0, i1p = plane[0], plane[1]
        lin = np.array([G1, H1, I1], float)
        u = np.array([lin[i0], lin[i1p]], float)
        nrm = float(np.hypot(u[0], u[1]))
        if nrm > SNAP_TOL:
            c = u[0] / nrm
            s = u[1] / nrm
            R2 = np.eye(3)
            R2[i0, i0] = c;  R2[i0, i1p] = -s
            R2[i1p, i0] = s; R2[i1p, i1p] =  c
            lin2 = R2.T @ lin
            G1, H1, I1 = lin2.tolist()
            R = R @ R2
            c_rot = R2.T @ c_rot
            dx,dy,dz = map(float, c_rot)

    # === If J1 != 0 and there is a pure-linear (parabolic) axis, shift along that axis to make J -> 0
    if abs(J1) > SNAP_TOL:
        axes = [(a1, G1, 0), (b1, H1, 1), (c1, I1, 2)]
        for a_lin, lin_val, idx in axes:
            if abs(a_lin) <= SNAP_TOL and abs(lin_val) > SNAP_TOL:
                t = -J1 / lin_val
                if idx == 0: dx += t
                elif idx == 1: dy += t
                else: dz += t
                c_rot = np.array([dx,dy,dz], float)
                J1 = 0.0
                print("#### shifted along pure-linear axis", idx, "by", t, "to set J=0")
                break

    # Build unscaled canonical coeffs after the nullspace rotation (+ optional pure-linear shift)
    new_coeffs_unscaled = (_A, _B, _C, _D, _E, _F, G1, H1, I1, J1)
    print("!!!!debug:before _clean_coeffs new_coeffs_unscaled:", new_coeffs_unscaled)
    new_coeffs_unscaled = _clean_coeffs(new_coeffs_unscaled)
    print("!!!!debug:before scaling (unscaled coeffs):", new_coeffs_unscaled)

    # 4) SINGLE scaling at the very end so A,B,C ∈ {−1,0,+1} and J ∈ {−1,0}
    a1,b1,c1,d1,e1,f1,g1,h1,i1,j1 = new_coeffs_unscaled
    print("a1", a1, "b1", b1, "c1", c1, "d1", d1, "e1", e1, "f1", f1," g1", g1, "h1", h1, "i1", i1, "j1", j1)

    # force j1 = -1 by scaling all coeffs
    if abs(j1) > J_FORCE_THR:
        alphaJ = -1.0 / j1       # multiply both sides by this
        a1 *= alphaJ
        b1 *= alphaJ
        c1 *= alphaJ
        d1 *= alphaJ
        e1 *= alphaJ
        f1 *= alphaJ
        g1 *= alphaJ
        h1 *= alphaJ
        i1 *= alphaJ
        j1 = -1.0                # by construction
    else:
        alphaJ = 1.0 

    # lock target to the *initial* choice (−1 or 0). If drifted, we will re-assert it.
    targetJ = 0.0 if abs(j1) <= J_FORCE_THR else -1.0
    #print("targetJ:", targetJ, "J0",J0,"J_FORCE_THR",J_FORCE_THR)

    def choose_s(a, lin, j, targetJ):
        tiny = 1e-12
        if targetJ == 0.0:
            # try to normalize a quadratic axis or a remaining linear axis
            if abs(a) > SNAP_TOL: return 1.0/ math.sqrt(abs(a))
            if abs(lin) > SNAP_TOL: return 1.0/ abs(lin)
            return 1.0
        else:
            # target J -> -1: pick a scale that relates to current J magnitude
            if abs(a) > SNAP_TOL:
                val = (-j)/abs(a)
                if val > tiny: return math.sqrt(val)
                return 1.0/ math.sqrt(max(abs(a), tiny))
            if abs(lin) > SNAP_TOL:
                return abs(j)/abs(lin) if abs(j) > tiny else 1.0/abs(lin)
            return 1.0

    sx = choose_s(a1, g1, j1, targetJ)
    sy = choose_s(b1, h1, j1, targetJ)
    sz = choose_s(c1, i1, j1, targetJ)

    # apply scaling once
    A2 = a1 * sx * sx;  B2 = b1 * sy * sy;  C2 = c1 * sz * sz
    D2 = d1 * sx * sy;  E2 = e1 * sx * sz;  F2 = f1 * sy * sz
    G2 = g1 * sx;       H2 = h1 * sy;       I2 = i1 * sz;       J2 = targetJ #  j1 # 
    print("sx", sx, "a1",a1,"g1",g1,"j1",j1 ,"sy", sy, "sz", sz)
    print("A2", A2, "B2", B2, "C2", C2, "D2", D2, "E2", E2, "F2", F2," G2", G2, "H2", H2, "I2", I2, "J2", J2)
    # We already normalized J at the beginning; here we only *reassert* {-1,0} if numerically drifted.
    alpha = 1.0

    A3,B3,C3,D3,E3,F3,G3,H3,I3,J3 = (alpha*A2, alpha*B2, alpha*C2,
                                     alpha*D2, alpha*E2, alpha*F2,
                                     alpha*G2, alpha*H2, alpha*I2, alpha*J2)
    print("A3", A3, "B3", B3, "C3", C3, "D3", D3, "E3", E3, "F3", F3," G3", G3, "H3", H3, "I3", I3, "J3", J3)
    # snap A,B,C to {-1,0,+1}; for parabolic axes, keep linear as ±1 if significant

    # final J snap to {-1,0}
    if abs(J3) <= J_FORCE_THR:
        J3 = 0.0
    else:
        J3 = -1.0

    new_coeffs_final = tuple(0.0 if abs(float(v)) < SNAP_TOL else float(v)
                             for v in (A3,B3,C3,D3,E3,F3,G3,H3,I3,J3))

    print("!!!!debug:scale_factors1:", (sx,sy,sz))
    print("!!!!debug:after scaling (final coeffs):", new_coeffs_final)

    # KEY FIX: center used for transform must match the final polynomial
    centre_canon = np.array([dx/max(sx,1e-30), dy/max(sy,1e-30), dz/max(sz,1e-30)], float)
    print("alpha:", alpha)
    scale_factors1 = (float(sx), float(sy), float(sz)) 
    scale_factors2 = (1.0, 1.0, 1.0)
    jacobian = scale_factors1[0] * scale_factors1[1] * scale_factors1[2]
    
    return QuadricResult(sym_type, axis_vec, centre_canon, new_coeffs_final,
                         scale_factors1, scale_factors2, jacobian, R)

def is_flat_quadric(coeffs: Tuple[float, ...], tol: float = 1e-8) -> bool:
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    Q = _quadric_matrix(A,B,C,D,E,F)
    return np.linalg.norm(Q, ord=2) < tol

def transform_point(p: np.ndarray, centre: np.ndarray, R: np.ndarray,
                    scale_factors1: Tuple[float, float, float]) -> np.ndarray:
    """
    Transform points to canonical coordinates:
      p' = (R^T @ p) / scale_factors1 - centre
    where 'centre' is in CANONICAL coordinates (consistent with final new_coeffs).
    """
    p = np.asarray(p, dtype=float).reshape(3)
    centre = np.asarray(centre, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3,3)
    sxf = np.asarray(scale_factors1, dtype=float).reshape(3)
    return (R.T @ p) / sxf - centre

# CSV processing helpers

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
                B_xyz = (float(row["Bx"]), float(row["By"]), float(row["Bz"]))
                C_xyz = (float(row["Cx"]), float(row["Cy"]), float(row["Cz"]))
                out_rows.append({
                    # order matches your requested header sequence
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
                    # appended original coordinates, exactly as requested
                    "Ax": A_xyz[0], "Ay": A_xyz[1], "Az": A_xyz[2],
                    "Bx": B_xyz[0], "By": B_xyz[1], "Bz": B_xyz[2],
                    "Cx": C_xyz[0], "Cy": C_xyz[1], "Cz": C_xyz[2],
                    # appended original ABC_* coeffs from _COEFFS.csv
                    "ABC_A": float(row["ABC_A"]), "ABC_B": float(row["ABC_B"]), "ABC_C": float(row["ABC_C"]),
                    "ABC_D": float(row["ABC_D"]), "ABC_E": float(row["ABC_E"]), "ABC_F": float(row["ABC_F"]),
                    "ABC_G": float(row["ABC_G"]), "ABC_H": float(row["ABC_H"]), "ABC_I": float(row["ABC_I"]),
                    "ABC_J": float(row["ABC_J"]),
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

            newA,newB,newC,newD,newE,newF,newG,newH,newI,newJ = [
                _snap_scalar_to_unit_or_zero(v) for v in info.new_coeffs
            ]

            # special case: [1,1,1,0,0,0,0,0,0,0] or [-1,-1,-1,0,0,0,0,0,0,0]
            if (
                (
                    (newA == 1.0 and newB == 1.0 and newC == 1.0) or
                    (newA == -1.0 and newB == -1.0 and newC == -1.0)
                )
                and newD == 0.0 and newE == 0.0 and newF == 0.0
                and newG == 0.0 and newH == 0.0 and newI == 0.0 and newJ == 0.0
            ):
                sfx_out = sfy_out = sfz_out = 0.0
            else:
                sfx_out, sfy_out, sfz_out = sfx, sfy, sfz

            out_rows.append({
                # order matches your requested header sequence
                "triangle_id": float(tri_id),
                "A_id": float(row["A_id"]), "B_id": float(row["B_id"]), "C_id": float(row["C_id"]),
                "scale_factors1_x": sfx_out, "scale_factors1_y": sfy_out, "scale_factors1_z": sfz_out,
                "A_transformed_x": float(A_t[0]), "A_transformed_y": float(A_t[1]), "A_transformed_z": float(A_t[2]),
                "B_transformed_x": float(B_t[0]), "B_transformed_y": float(B_t[1]), "B_transformed_z": float(B_t[2]),
                "C_transformed_x": float(C_t[0]), "C_transformed_y": float(C_t[1]), "C_transformed_z": float(C_t[2]),
                "ABC_new_A": newA, "ABC_new_B": newB, "ABC_new_C": newC,
                "ABC_new_D": newD, "ABC_new_E": newE, "ABC_new_F": newF,
                "ABC_new_G": newG, "ABC_new_H": newH, "ABC_new_I": newI,
                "ABC_new_J": newJ,
                # appended original coordinates, exactly as requested
                "Ax": float(A_xyz[0]), "Ay": float(A_xyz[1]), "Az": float(A_xyz[2]),
                "Bx": float(B_xyz[0]), "By": float(B_xyz[1]), "Bz": float(B_xyz[2]),
                "Cx": float(C_xyz[0]), "Cy": float(C_xyz[1]), "Cz": float(C_xyz[2]),
                # appended original ABC_* coeffs from _COEFFS.csv
                "ABC_A": float(row["ABC_A"]), "ABC_B": float(row["ABC_B"]), "ABC_C": float(row["ABC_C"]),
                "ABC_D": float(row["ABC_D"]), "ABC_E": float(row["ABC_E"]), "ABC_F": float(row["ABC_F"]),
                "ABC_G": float(row["ABC_G"]), "ABC_H": float(row["ABC_H"]), "ABC_I": float(row["ABC_I"]),
                "ABC_J": float(row["ABC_J"]),
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

# batch / CLI helpers
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
