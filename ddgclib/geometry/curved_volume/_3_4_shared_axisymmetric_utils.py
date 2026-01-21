#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared_axisymmetric_utils.py
Auto-generated on 2025-08-18 12:22:16 by consolidation script.

Contains helper functions shared by:
  - Rotation_paraboloid.py
  - Rotation_hyperboloid.py
  - Rotation_cylinder.py

Selection policy:
  - If a function is defined in multiple files with differences, the most recently modified file wins.

NOTE:
  This file only aggregates *function* definitions that are common to all three scripts, as requested.
  If any of these functions depend on variables or helpers defined elsewhere, ensure those are imported
  where you use this module.
"""
import math
import numpy as np
from _Vcut import Vcut

# Optional: numba for JIT-accelerating numeric kernels (safe no-op fallback)
try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

# ---- Fallbacks for pipeline-level globals (caller should override/import their own) ----
try:
    COEFFS
except NameError:
    COEFFS = None  # caller should pass coeffs explicitly or set this before use

try:
    OBSERVER_OUTSIDE
except NameError:
    OBSERVER_OUTSIDE = True  # default outward-observer convention

try:
    # Optional: numerical swept-segment helper, if available
    from _swept_segment_volume_AB import swept_segment_volume_AB, swept_segment_volume_A1B
except Exception:  # pragma: no cover
    def swept_segment_volume_AB(*args, **kwargs):
        raise NotImplementedError(
            "swept_segment_volume_AB is not available; "
            "import it in your caller or pass an alternative."
        )

# --------------------------------------------------------------------
# Debug + caches
# --------------------------------------------------------------------
# Triangles for which verbose debug printing is enabled (empty in production)
DEBUG_TRI = set()

# Caches for expensive axis-frame computations
_AXIS_TRANSFORM_CACHE = {}  # key: coeffs tuple(10) -> (x0, U, A3p, bp, cp)
_AXIS_CENTER_CACHE = {}     # key: coeffs tuple(10) -> (ok, axis_dir, axis_point, frame)

# --------------------------------------------------------------------
# Tiny, fast tetra-volume helper (used everywhere)
# --------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def _tet_volume_core(apex, P, Q, R):
    """Numba-friendly 3D tetra volume |det([P-apex, Q-apex, R-apex])| / 6."""
    v1_0 = P[0] - apex[0]
    v1_1 = P[1] - apex[1]
    v1_2 = P[2] - apex[2]

    v2_0 = Q[0] - apex[0]
    v2_1 = Q[1] - apex[1]
    v2_2 = Q[2] - apex[2]

    v3_0 = R[0] - apex[0]
    v3_1 = R[1] - apex[1]
    v3_2 = R[2] - apex[2]

    # cross = v2 x v3
    cx = v2_1 * v3_2 - v2_2 * v3_1
    cy = v2_2 * v3_0 - v2_0 * v3_2
    cz = v2_0 * v3_1 - v2_1 * v3_0

    triple = v1_0 * cx + v1_1 * cy + v1_2 * cz
    if triple < 0.0:
        triple = -triple
    return triple / 6.0


def _tet_volume_apex(apex, P, Q, R):
    """Wrapper that ensures numpy arrays + calls the numba core if available."""
    apex = np.asarray(apex, dtype=float)
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    return float(_tet_volume_core(apex, P, Q, R))


# ----------------------------------------------------------------------------

__all__ = [
  "_axis_frame",
  "_axis_transform_from_coeffs",
  "_cells_dict",
  "_detect_axis_of_revolution",
  "_parse_quadric_coeffs",
  "_quadric_center",
  "_quadric_grad_hess",
  "_transform_to_axis_frame",
  "adjust_curvature_signs",
  "cone_volume_patch",
  "cone_volume_patch_inside",
  "curved_tet_volume",
  "cylinder_angle",
  "extend_arc_point_to_z",
  "find_rotated_point",
  "frustum_volume",
  "iso_z_project_axisymmetric",
  "normalize",
  "normalize_three_angles",
  "principal_curvatures_at",
  "principal_curvatures_rule_label",
  "project_to_quadric_normal_step",
  "quadric_kind",
  "tet_volume",
  "to_axis",
  "validate_swept_segment",
  "wedge_volume_rotation",
  "worker",
  # added:
  "rotation_patch_area_transformed",
  "rotation_patch_area_original",
  "curved_patch_area_plane_lift",
]

def coeffs_1x_to_2x(coeffs_1x):
    """
    Convert the 1x convention (a,b,c,d,e,f,g,h,i,j) used by volume_patch
    back to the 2x convention:
      Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0

    Mapping:
      A=a, B=b, C=c,
      D=d/2, E=e/2, F=f/2,
      G=g/2, H=h/2, I=i/2,
      J=j
    """
    a, b, c, d, e, f, g, h, i, j = map(float, coeffs_1x)
    return (a, b, c, d / 2.0, e / 2.0, f / 2.0, g / 2.0, h / 2.0, i / 2.0, j)

def _axis_frame(k):
    k = normalize(np.asarray(k, float))
    t = np.array([1.0, 0.0, 0.0]) if abs(k[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(t - np.dot(t, k) * k)
    v = np.cross(k, u)
    return (u, v, k)

def _axis_transform_from_coeffs(coeffs):
    A3, b, _ = _parse_quadric_coeffs(coeffs)
    ok, axis_dir = _detect_axis_of_revolution(A3)
    if not ok:
        return (None, None)
    x0 = _quadric_center(A3, b)
    u, v, k = _axis_frame(axis_dir)
    U = np.column_stack([u, v, k])
    return (x0, U)

def _cells_dict(mesh):
    if hasattr(mesh, 'cells_dict') and isinstance(mesh.cells_dict, dict):
        return mesh.cells_dict
    d = {}
    for cb in mesh.cells:
        d[cb.type] = cb.data
    return d

def _detect_axis_of_revolution(A3, tol=1e-08):
    w, V = np.linalg.eigh(A3)
    idx = np.argsort(w)
    w = w[idx]
    V = V[:, idx]
    scale = max(1.0, np.linalg.norm(w))
    if abs(w[2] - w[1]) < tol * scale and abs(w[1] - w[0]) < tol * scale:
        return (True, np.array([0.0, 0.0, 1.0]))
    if abs(w[2] - w[1]) >= 10 * tol * scale and abs(w[1] - w[0]) < tol * scale:
        axis = V[:, 2]
        return (True, axis / np.linalg.norm(axis))
    if abs(w[2] - w[1]) < tol * scale and abs(w[1] - w[0]) >= 10 * tol * scale:
        axis = V[:, 0]
        return (True, axis / np.linalg.norm(axis))
    return (False, None)

def _parse_quadric_coeffs(coeffs):
    # coeffs are in 1× form: (A,B,C,D,E,F,G,H,I,J) with D=xy, E=xz, F=yz, G=x, H=y, I=z
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    A3 = np.array([
        [A,   D / 2, E / 2],
        [D / 2, B,   F / 2],
        [E / 2, F / 2, C],
    ], float)
    b = 0.5 * np.array([G, H, I], float)
    return A3, b, J

def _quadric_center(A3, b):
    try:
        return -np.linalg.solve(A3, b)
    except np.linalg.LinAlgError:
        return -np.linalg.pinv(A3) @ b

def _quadric_grad_hess(coeffs, P):
    A3, b, _ = _parse_quadric_coeffs(coeffs)
    P = np.asarray(P, float)
    grad = 2 * (A3 @ P + b)
    H = 2 * A3
    return (grad, H)

def _transform_to_axis_frame(coeffs):
    """Return (x0, U, A3p, bp, cp) where y = U^T (x - x0) are axis-frame coords."""
    A3, b, c_ = _parse_quadric_coeffs(coeffs)
    ok, axis_dir = _detect_axis_of_revolution(A3)
    if not ok:
        return (None, None, None, None, None)
    x0 = _quadric_center(A3, b)
    u, v, k = _axis_frame(axis_dir)
    U = np.column_stack([u, v, k])
    A3p = U.T @ A3 @ U
    bp = U.T @ (A3 @ x0 + b)
    cp = float(x0 @ (A3 @ x0) + 2.0 * b @ x0 + c_)
    return (x0, U, A3p, bp, cp)

def _transform_to_axis_frame_cached(coeffs):
    """Cached wrapper around _transform_to_axis_frame to avoid repeated eigendecompositions."""
    key = tuple(float(c) for c in coeffs)
    cached = _AXIS_TRANSFORM_CACHE.get(key)
    if cached is not None:
        return cached
    res = _transform_to_axis_frame(coeffs)
    _AXIS_TRANSFORM_CACHE[key] = res
    return res

def _axis_frame_and_center_from_coeffs(coeffs):
    """
    Cached combination of:
      - parsing A3, b
      - detecting axis
      - computing axis center and local frame
    Used by wedge_volume_rotation.
    """
    key = tuple(float(c) for c in coeffs)
    cached = _AXIS_CENTER_CACHE.get(key)
    if cached is not None:
        return cached
    A3, b, _ = _parse_quadric_coeffs(coeffs)
    ok, axis_dir = _detect_axis_of_revolution(A3)
    if not ok:
        res = (False, None, None, None)
    else:
        axis_point = _quadric_center(A3, b)
        frame = _axis_frame(axis_dir)
        res = (True, axis_dir, axis_point, frame)
    _AXIS_CENTER_CACHE[key] = res
    return res

def adjust_curvature_signs(k1, k2, coeffs, observer_outside=True):
    kind = quadric_kind(coeffs)
    if kind in ('circular_cylinder', 'sphere', 'elliptic_paraboloid'):
        return (-k1, -k2) if observer_outside else (k1, k2)
    if 'hyperboloid' in kind:
        return (k1, k2)
    return (k1, k2)

    # NOTE: the code block that used to live here (with EPS, rA, rB, rC, etc.)
    # was accidentally inlined and is now replaced by the proper cone_volume_patch()
    # function below. It is intentionally unreachable after the return.

def cone_volume_patch(A, B, C, R_unused, r_plane, k, triangle_id=None):
    EPS = 1e-12

    # ---------- constants/angles ----------
    V_cut = Vcut(A[2], B[2], coeffs_1x_to_2x(COEFFS))
    rA, rB, rC = np.linalg.norm(A[:2]), np.linalg.norm(B[:2]), np.linalg.norm(C[:2])
    O  = np.array([0.0, 0.0, (B[2] + A[2]) / 2.0])
    O1 = np.array([0.0, 0.0, A[2]])
    O2 = np.array([0.0, 0.0, B[2]])

    h_full = abs(B[2] - A[2])
    V_t = frustum_volume(rA, rB, h_full)

    thA = cylinder_angle(A[0], A[1], rA)  # [0,2π)
    thB = cylinder_angle(B[0], B[1], rB)
    thC = cylinder_angle(C[0], C[1], rC)

    ratio_AB_A = (rA ** k) / (rA ** k + rB ** k + 1e-300)
    ratio_AB_B = (rB ** k) / (rA ** k + rB ** k + 1e-300)

    def mid_angle_minor(thA_, thC_):
        """Mid-angle along the MINOR arc between thA and thC using only % (2π)."""
        d = (thC_ - thA_) % (2 * math.pi)  # thA -> thC
        if d <= math.pi:
            return (thA_ + 0.5 * d) % (2 * math.pi)
        else:
            L = (2 * math.pi - d)          # minor arc is thC->thA
            return (thC_ + 0.5 * L) % (2 * math.pi)

    def on_minor_arc(theta, thA_, thC_, tol=1e-12):
        """True iff theta lies on the MINOR arc between thA and thC, using only % (2π)."""
        d_minor = (thC_ - thA_) % (2 * math.pi)
        if d_minor > math.pi:
            d_minor = 2 * math.pi - d_minor
            c2t = (theta - thC_) % (2 * math.pi)
            t2a = (thA_   - theta) % (2 * math.pi)
            return abs((c2t + t2a) - d_minor) <= tol
        else:
            a2t = (theta - thA_) % (2 * math.pi)
            t2c = (thC_   - theta) % (2 * math.pi)
            return abs((a2t + t2c) - d_minor) <= tol

    def on_minor_arc_strict(theta, thP, thQ, tol=1e-12):
        """STRICT: theta strictly inside the MINOR arc between thP and thQ (exclude endpoints)."""
        two_pi = 2 * math.pi
        d = (thQ - thP) % two_pi
        if d > math.pi:
            L = two_pi - d
            q2t = (theta - thQ) % two_pi
            return (q2t > tol) and (q2t < L - tol)
        else:
            L = d
            p2t = (theta - thP) % two_pi
            return (p2t > tol) and (p2t < L - tol)

    def otheta(P, eps_base=1e-12):
        """Snapped polar angle for debug: y=0 & x>=0 -> 0; y=0 & x<0 -> π; else [0,2π) with snap."""
        x = float(P[0])
        y = float(P[1])
        r = math.hypot(x, y)
        eps = max(eps_base, 1e-9 * max(1.0, r))
        if abs(y) <= eps:
            return 0.0 if x >= 0.0 else math.pi
        a = math.atan2(y, x)
        if a < 0.0:
            a += 2.0 * math.pi
        if abs(a - 2.0 * math.pi) <= eps:
            a = 0.0
        return a

    # ======================= CASE 1: C ≈ A =======================
    if abs(C[2] - A[2]) < EPS:
        rAC   = rA
        EPS_R = 1e-14
        thB_eff = thB if rB > EPS_R else mid_angle_minor(thA, thC)

        Bt = np.array([rAC * math.cos(thB_eff), rAC * math.sin(thB_eff), A[2]], float)
        Ab = np.array([rB * math.cos(thA),     rB * math.sin(thA),     B[2]], float)
        Cb = np.array([rB * math.cos(thC),     rB * math.sin(thC),     B[2]], float)

        d_BA = (thB_eff - thA) % (2 * math.pi)
        if d_BA > math.pi:
            d_BA = 2 * math.pi - d_BA
        d_BC = (thC - thB_eff) % (2 * math.pi)
        if d_BC > math.pi:
            d_BC = 2 * math.pi - d_BC
        d_AC = (thC - thA) % (2 * math.pi)
        if d_AC > math.pi:
            d_AC = 2 * math.pi - d_AC

        split_BA = 2.0 * math.pi / (d_BA + 1e-12)
        split_BC = 2.0 * math.pi / (d_BC + 1e-12)

        V_OABBt  = _tet_volume_apex(O,  A,  B,  Bt)
        V_OABAb  = _tet_volume_apex(O,  A,  B,  Ab)
        V_O1OBtA = _tet_volume_apex(O1, O,  Bt, A)
        V_O2OBAb = _tet_volume_apex(O2, O,  B,  Ab)
        V_BBtA   = (V_cut / split_BA - V_OABBt - V_OABAb - V_O1OBtA - V_O2OBAb) * ratio_AB_A

        V_OBCBt  = _tet_volume_apex(O,  B,  C,  Bt)
        V_OBCCb  = _tet_volume_apex(O,  B,  C,  Cb)
        V_O1OBtC = _tet_volume_apex(O1, O,  Bt, C)
        V_O2OBCb = _tet_volume_apex(O2, O,  B,  Cb)
        V_BBtC   = (V_cut / split_BC - V_OBCBt - V_OBCCb - V_O1OBtC - V_O2OBCb) * ratio_AB_A

        V_BBtAC = _tet_volume_apex(B, Bt, A, C)

        Bt_between_CA = on_minor_arc(thB_eff, thA, thC)

        if not Bt_between_CA:
            Vpatch = abs(V_BBtA - V_BBtC) - V_BBtAC
        else:
            Vpatch = (V_BBtA + V_BBtC) + V_BBtAC

        if triangle_id in DEBUG_TRI:
            oBt = otheta(Bt)
            oAb = otheta(Ab)
            oCb = otheta(Cb)
            oC  = otheta(C)
            oA  = otheta(A)
            print("inside-ID", triangle_id, "C~A branch")
            print("A", A, "B", B, "C", C, "Bt", Bt, "Bt_between_CA", Bt_between_CA, "\nVpatch?", Vpatch, "coeffs", COEFFS)
            print("oBt", oBt, "oAb", oAb, "oCb", oCb, "oC", oC, "oA", oA)
            print("d_BC", d_BC, "d_BA", d_BA, "d_AC", d_AC)
            print("V_BBtA", V_BBtA, "V_BBtC", V_BBtC, "V_BBtAC", V_BBtAC)
            print("split_BA, split_BC=", split_BA, split_BC, " ratio(A|B)=", ratio_AB_A)
            print("V_OABBt", V_OABBt, "V_OABAb", V_OABAb, "V_O1OBtA", V_O1OBtA, "V_O2OBAb", V_O2OBAb)
            print("V_OBCBt", V_OBCBt, "V_OBCCb", V_OBCCb, "V_O1OBtC", V_O1OBtC, "V_O2OBCb", V_O2OBCb)
            print("V_BBtA", V_BBtA, "V_BBtC", V_BBtC, "V_BBtAC", V_BBtAC, "Bt_between_CA?", Bt_between_CA)
            print("V_band", V_t, "V_cut", V_cut)

        return Vpatch

    # ======================= CASE 2: C ≈ B =======================
    if abs(C[2] - B[2]) < EPS:
        rAC = rA
        Bt  = np.array([rAC * math.cos(thB), rAC * math.sin(thB), A[2]], float)  # B -> AC circle
        Ab  = np.array([rB * math.cos(thA), rB * math.sin(thA), B[2]], float)    # A -> B-plane circle
        Cb  = np.array([rB * math.cos(thC), rB * math.sin(thC), B[2]], float)    # C -> B-plane circle
        Ct  = np.array([rA * math.cos(thC), rA * math.sin(thC), A[2]], float)    # C -> A-plane circle

        # Avoid Ab/Cb coinciding with B by nudging along minor-arc direction
        eps_ang = 1e-9
        dir_sign = +1 if ((thC - thA) % (2 * math.pi)) <= math.pi else -1
        if np.linalg.norm(Ab - B) < 1e-14:
            thA_eps = (thA + dir_sign * eps_ang) % (2 * math.pi)
            Ab = np.array([rB * math.cos(thA_eps), rB * math.sin(thA_eps), B[2]], float)
        if np.linalg.norm(Cb - B) < 1e-14:
            thC_eps = (thC - dir_sign * eps_ang) % (2 * math.pi)
            Cb = np.array([rB * math.cos(thC_eps), rB * math.sin(thC_eps), B[2]], float)

        d_BA_minor = (thB - thA) % (2 * math.pi)
        if d_BA_minor > math.pi:
            d_BA_minor = 2 * math.pi - d_BA_minor
        split_BA = 2.0 * math.pi / (d_BA_minor + 1e-12)

        d_AC_minor = (thC - thA) % (2 * math.pi)
        if d_AC_minor > math.pi:
            d_AC_minor = 2 * math.pi - d_AC_minor
        split_CA = 2.0 * math.pi / (d_AC_minor + 1e-12)

        # V_AAbB
        V_OABAb  = _tet_volume_apex(O,  A,  B,  Ab)
        V_OABBt  = _tet_volume_apex(O,  A,  B,  Bt)
        V_O1OBtA = _tet_volume_apex(O1, O,  Bt, A)
        V_O2OBAb = _tet_volume_apex(O2, O,  B,  Ab)
        V_AAbB   = (V_cut / split_BA - V_OABAb - V_OABBt - V_O1OBtA - V_O2OBAb) * ratio_AB_B

        # V_AAbC   (use Ct per your algorithm)
        V_OACAb  = _tet_volume_apex(O,  A,  C,  Ab)
        V_OACCt  = _tet_volume_apex(O,  A,  C,  Ct)
        V_O1OACt = _tet_volume_apex(O1, O,  A,  Ct)
        V_O2OCAb = _tet_volume_apex(O2, O,  C,  Ab)
        V_AAbC   = (V_cut / split_CA - V_OACAb - V_OACCt - V_O1OACt - V_O2OCAb) * ratio_AB_B

        # Also compute the *Cb* versions for your debug comparison
        V_OACCb  = _tet_volume_apex(O,  A,  C,  Cb)
        V_O1OACb = _tet_volume_apex(O1, O,  A,  Cb)

        V_AAbBC = _tet_volume_apex(A, Ab, B, C)

        # STRICT between test so Ab==B (endpoint) is NOT "between"
        Ab_between_CB = on_minor_arc_strict(thA, thC, thB)

        if not Ab_between_CB:
            Vpatch = abs(V_AAbB - V_AAbC) - V_AAbBC
        else:
            Vpatch = (V_AAbB + V_AAbC) + V_AAbBC

        if triangle_id in DEBUG_TRI:
            oBt = otheta(Bt)
            oAb = otheta(Ab)
            oCb = otheta(Cb)
            oC  = otheta(C)
            oA  = otheta(A)

            print("inside2-ID", triangle_id, "Vpatch?", Vpatch, "Ab_between_CB", Ab_between_CB)
            print("oBt", oBt, "oAb", oAb, "oCb", oCb, "oC", oC, "oA", oA, "oB", otheta(B))
            print("A", A, "B", B, "C", C, "Ab", Ab, "Ct", Ct)
            print("V_AAbB", V_AAbB, "V_AAbC", V_AAbC, "V_AAbBC", V_AAbBC)
            print((V_cut / split_CA - V_OACAb - V_OACCb - V_O1OACb - V_O2OCAb) * ratio_AB_B)
            print("V_cut/split_CA", V_cut / split_CA, "V_OACAb", V_OACAb, "V_OACCb", V_OACCb, "V_O1OACb", V_O1OACb, "V_O2OCAb", V_O2OCAb)
            print("split_CA", split_CA, "ratio(B|A)=", ratio_AB_B)

        return Vpatch

    # ======================= default (unchanged overall logic) =======================
    dth  = (thC - thA) % (2 * math.pi)
    D    = find_rotated_point(B, rB, -((math.atan2(C[1], C[0]) - math.atan2(A[1], A[0]))))
    if dth < 1e-8:
        return 0.0
    if dth >= math.pi:
        dth = 2.0 * math.pi - dth

    split = 2.0 * math.pi / (dth + 1e-12)
    V_OABC  = _tet_volume_apex(O,  A, B, C)
    V_OABD  = _tet_volume_apex(O,  B, D, A)
    V_O1CAO = _tet_volume_apex(O1, C, A, O)
    V_O2BOD = _tet_volume_apex(O2, B, D, O)

    if abs(rB - rC) < 1e-7:
        V_swept = 0.0
    else:
        V_swept, _mode = swept_segment_volume_AB(COEFFS, C[2], B[2])

    Vpatch = (V_cut / split - V_OABC - V_OABD - V_O1CAO - V_O2BOD) * (rA ** k / (rA ** k + rB ** k))

    if triangle_id in DEBUG_TRI:
        print("inside2_ID", triangle_id, C[2], B[2], "COEFFS 2x form", COEFFS)
        print(f"  patch id={triangle_id}: dth={dth:.6e}, split={split:.6e}, V_t={V_t:.6e}, V_swept={V_swept:.6e}, Vpatch={Vpatch:.6e}")
        print("V_OABC", V_OABC, "V_OABD", V_OABD, "V_O1CAO", V_O1CAO, "V_O2BOD", V_O2BOD)
        print("ratio", (rA ** k) / (rA ** k + rB ** k))
        print("V_band", V_t + V_swept, "V_cut", V_cut)

    return Vpatch

def cone_volume_patch_inside(A, B, C, R_unused, r_plane, k, triangle_id=None):
    EPS = 1e-10
    rA, rB, rC = (np.linalg.norm(A[:2]), np.linalg.norm(B[:2]), np.linalg.norm(C[:2]))
    O = np.array([0.0, 0.0, (B[2] + A[2]) / 2.0])
    h_full = abs(B[2] - A[2])
    V_t = frustum_volume(rA, rB, h_full)
    thA = cylinder_angle(A[0], A[1], rA)
    thB = cylinder_angle(B[0], B[1], rB)
    thC = cylinder_angle(C[0], C[1], rC)
    oA, oB, oC = (np.arctan2(A[1], A[0]), np.arctan2(B[1], B[0]), np.arctan2(C[1], C[0]))
    d_CB, d_CA = (oC - oB, oC - oA)
    if abs(C[2] - A[2]) < EPS:
        theta_CA = (thC - thA) % (2 * np.pi)
        D = find_rotated_point(B, rB, -d_CA)
        if theta_CA < 1e-08:
            return 0.0
        if theta_CA >= np.pi:
            theta_CA = 2 * np.pi - theta_CA
        tri_COA = 0.5 * rA ** 2 * np.sin(theta_CA)
        tri_BOD = 0.5 * rB ** 2 * np.sin(theta_CA)
        split = 2 * np.pi / (theta_CA + 1e-12)
        V_OABC = _tet_volume_apex(O,  A, B, C)
        V_OABD = _tet_volume_apex(O,  B, D, A)
        if abs(rB - rC) < 1e-07:
            V_swept = 0.0
        else:
            V_swept, _mode = swept_segment_volume_AB(COEFFS, C[2], B[2])
        Vpatch = (
            V_swept
            - (
                V_t
                - V_OABC * split
                - V_OABD * split
                - tri_BOD * split * h_full / 3.0 * 0.5
                - tri_COA * split * h_full / 3.0 * 0.5
            )
        ) / split * (rA ** k / (rA ** k + rB ** k))
        return Vpatch
    if abs(C[2] - B[2]) < EPS:
        dth = (thC - thB) % (2 * np.pi)
        if dth < 1e-08:
            return 0.0
        if dth >= np.pi:
            dth = 2 * np.pi - dth
        D = find_rotated_point(A, rA, -d_CB)
        V_OABC = _tet_volume_apex(O,  A, B, C)
        V_OABD = _tet_volume_apex(O,  B, D, A)
        tri_COB = 0.5 * rB ** 2 * np.sin(dth)
        tri_DOA = 0.5 * rA ** 2 * np.sin(dth)
        split = 2 * np.pi / (dth + 1e-12)
        if abs(rB - rA) < 1e-07:
            V_swept = 0.0
        else:
            V_swept, _mode = swept_segment_volume_AB(COEFFS, A[2], C[2])
        Vpatch = (
            V_swept
            - (
                V_t
                - V_OABC * split
                - V_OABD * split
                - tri_COB * split * h_full / 6.0
                - tri_DOA * split * h_full / 6.0
            )
        ) / split * (rB ** k / (rB ** k + rA ** k))
        return Vpatch
    return 0.0

def curved_tet_volume(A, B, R_curve, C, D, N=400):
    """
    Volume of the curved tetra between A,B along a circle of radius R_curve
    and base edge CD. Fully vectorized over arc segments.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    D = np.asarray(D, float)

    thA = math.atan2(A[1], A[0])
    thB = math.atan2(B[1], B[0])
    dth = (thB - thA + 2.0 * math.pi) % (2.0 * math.pi)
    if dth > math.pi:
        dth -= 2.0 * math.pi

    n_seg = max(8, int(N))
    thetas = thA + np.linspace(0.0, dth, n_seg)
    zs = np.linspace(A[2], B[2], n_seg)

    arc = np.stack(
        [R_curve * np.cos(thetas),
         R_curve * np.sin(thetas),
         zs],
        axis=1,
    )

    segs = arc[1:] - arc[:-1]        # (n_seg-1, 3)
    C0 = C - arc[:-1]                # (n_seg-1, 3)
    D0 = D - arc[:-1]                # (n_seg-1, 3)
    crosses = np.cross(C0, D0)       # (n_seg-1, 3)
    triple = np.einsum("ij,ij->i", crosses, segs)
    vol = np.sum(np.abs(triple)) / 6.0
    return float(vol)

def cylinder_angle(x, y, R_unused):
    angle = np.arctan2(y, x)
    if angle < -1e-12:
        angle += 2 * np.pi
    if np.isclose(angle, 2 * np.pi):
        angle = 0.0
    return angle

def extend_C_to_AB(Ap, rA, Bp, rB, Cp, rC, idx=None, eps=1e-12):
    """
    Return:
        Fp : intersection of line Ap->Bp with plane z=zC (midpoint if coplanar;
             if parallel & not coplanar, returns np.nan)
        rF : sqrt(Fx^2 + Fy^2) or np.nan if Fp is nan
        Gp : the crossing point of segment Cp->Fp with the circle r=rC in plane z=zC,
             EXCLUDING Cp if a second crossing exists; otherwise Gp == Cp.

    Behavior changes vs. prior version:
        - If there is no second intersection (or rF <= rC, or no valid s* on the segment),
          Gp is returned as Cp (not np.nan).
        - If the line is parallel to the plane and not coplanar, Fp/rF are np.nan, but Gp is Cp.

    Notes:
        - You can still compute Ep from Fp’s polar angle when Fp exists.
    """
    Ap = np.asarray(Ap, float)
    Bp = np.asarray(Bp, float)
    Cp = np.asarray(Cp, float)

    zA, zB, zC = Ap[2], Bp[2], Cp[2]
    vz = zB - zA

    # --- 1) Fp: line–plane intersection with z=zC (or midpoint if coplanar)
    if abs(vz) > eps:
        t = (zC - zA) / vz
        Fp = Ap + t * (Bp - Ap)
        rF = float(np.hypot(Fp[0], Fp[1]))
    else:
        # Parallel to plane
        if abs(zC - zA) <= eps:
            # Coplanar: deterministic midpoint
            Fp = 0.5 * (Ap + Bp)
            rF = float(np.hypot(Fp[0], Fp[1]))
        else:
            # No intersection; still return Gp = Cp as requested
            if idx is not None and idx in DEBUG_TRI:
                print(f"[extend_C_to_AB] tri {idx}: ApBp ‖ plane z={zC:.6g}; no intersection. Gp=Cp.")
            return np.nan, np.nan, Cp.copy()

    # --- 2) Gp: second crossing of segment Cp->Fp with circle r=rC in plane z=zC
    # Default: no extra crossing → Gp = Cp
    Gp = Cp.copy()

    # Only attempt if Fp is farther than the circle (segment goes outside)
    if rF > rC + eps:
        Cp_xy = Cp[:2]
        Fp_xy = Fp[:2]
        d = Fp_xy - Cp_xy
        d2 = float(np.dot(d, d))
        if d2 > eps:
            # Solve ||Cp_xy + s*d|| = rC for s in (0,1]; s=0 is Cp itself
            # Derivation with Cp on circle gives two roots: s=0 and s* = -2 (Cp·d)/||d||^2
            s_star = float(-2.0 * np.dot(Cp_xy, d) / d2)
            # Accept strictly beyond Cp and within the segment
            if s_star > 1e-12 and s_star <= 1.0 + 1e-12:
                s_star = min(s_star, 1.0)
                G_xy = Cp_xy + s_star * d
                # Snap to exact circle to kill floating error
                rG = float(np.hypot(G_xy[0], G_xy[1]))
                if rG > eps:
                    G_xy *= (rC / rG)
                Gp = np.array([G_xy[0], G_xy[1], zC], float)
            # else: keep Gp = Cp
        # else: degenerate segment; keep Gp = Cp

    return Fp, rF, Gp

def extend_arc_point_to_z(P1, r_1_unused, P2, r_2_unused, Cp, R_target, idx=None):
    """
    Intersect straight edge AB with plane z=zC to get L, keep its polar angle,
    and set radius to R_target at that z. Removes angular bias from interpolation.
    """
    zC = float(Cp[2])
    z1, z2 = (float(P1[2]), float(P2[2]))
    if abs(z2 - z1) < 1e-12:
        theta = math.atan2(P1[1], P1[0])
    else:
        t = (zC - z1) / (z2 - z1)
        L = np.asarray(P1, float) + t * (np.asarray(P2, float) - np.asarray(P1, float))
        theta = math.atan2(L[1], L[0])
    return np.array([R_target * math.cos(theta), R_target * math.sin(theta), zC], float)

def find_rotated_point(B, r_B, dtheta):
    th = np.arctan2(B[1], B[0]) + dtheta
    return np.array([r_B * np.cos(th), r_B * np.sin(th), B[2]], B.dtype)

def frustum_volume(rA, rB, h):
    return np.pi * h / 3.0 * (rA ** 2 + rA * rB + rB ** 2)

def iso_z_project_axisymmetric(P0, coeffs):
    """
    Keep z and place (x,y) on the axisymmetric quadric at that z.
    Fast exact path for the paraboloid x^2 + y^2 - z = 0.
    """
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    if (
        abs(A - 1.0) < 1e-12
        and abs(B - 1.0) < 1e-12
        and (abs(C) < 1e-12)
        and (abs(D) < 1e-12)
        and (abs(E) < 1e-12)
        and (abs(F) < 1e-12)
        and (abs(G) < 1e-12)
        and (abs(H) < 1e-12)
        and (abs(J) < 1e-12)
        and (abs(I + 0.5) < 1e-12)
    ):
        x, y, z = map(float, P0)
        r = math.sqrt(max(z, 0.0))
        th = math.atan2(y, x)
        return np.array([r * math.cos(th), r * math.sin(th), z], float)

    x0, U, A3p, bp, cp = _transform_to_axis_frame_cached(coeffs)
    if x0 is None:
        return project_to_quadric_normal_step(P0, coeffs)

    y = U.T @ (np.asarray(P0, float) - x0)
    z = y[2]
    Axy, Axz, Ayz = (A3p[0, 1], A3p[0, 2], A3p[1, 2])
    if any(
        (abs(v) > 1e-10 * max(1.0, np.linalg.norm(A3p))
         for v in (Axy, Axz, Ayz, bp[0], bp[1]))
    ):
        return project_to_quadric_normal_step(P0, coeffs)

    Ar = 0.5 * (A3p[0, 0] + A3p[1, 1])
    Cz = float(A3p[2, 2])
    jz = float(2.0 * bp[2])
    k0 = float(cp)
    numer = -(Cz * z * z + jz * z + k0)
    if Ar == 0.0:
        return project_to_quadric_normal_step(P0, coeffs)
    r2 = numer / Ar
    if r2 < 0.0:
        if r2 > -1e-12:
            r2 = 0.0
        else:
            return project_to_quadric_normal_step(P0, coeffs)
    r_target = float(np.sqrt(r2))
    xy = y[:2]
    nxy = np.linalg.norm(xy)
    dir_xy = np.array([1.0, 0.0]) if nxy < 1e-15 else xy / nxy
    y_proj = np.array([r_target * dir_xy[0], r_target * dir_xy[1], z], float)
    return x0 + U @ y_proj

def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def normalize_three_angles(theta1, theta2, theta3):
    thetas = np.array([theta1, theta2, theta3], dtype=float)
    thetas = np.mod(thetas, 2 * np.pi)
    min_angle = np.min(thetas)
    thetas_shifted = np.mod(thetas - min_angle, 2 * np.pi)
    if np.ptp(thetas_shifted) > np.pi:
        thetas_shifted = np.mod(thetas_shifted + np.pi, 2 * np.pi)
        min_angle = np.min(thetas_shifted)
        thetas_shifted = np.mod(thetas_shifted - min_angle, 2 * np.pi)
    return tuple(thetas_shifted)

def principal_curvatures_at(P, coeffs=COEFFS, project=True, observer_outside=OBSERVER_OUTSIDE):
    thr = 1e-5
    coeffs = tuple(
        0.0 if (i < 9 and abs(float(c)) < thr) else float(c)
        for i, c in enumerate(coeffs)
    )
    Q = iso_z_project_axisymmetric(P, coeffs) if project else np.asarray(P, float)
    A3, b, _ = _parse_quadric_coeffs(coeffs)
    g = 2.0 * (A3 @ Q + b)
    Hm = 2.0 * A3
    ng = np.linalg.norm(g)
    if not np.isfinite(ng) or ng < 1e-14:
        return (np.nan, np.nan)
    n = g / ng
    t1 = np.array([1.0, 0.0, 0.0], float)
    if abs(np.dot(t1, n)) > 0.9:
        t1 = np.array([0.0, 1.0, 0.0], float)
    t1 = t1 - np.dot(t1, n) * n
    n1 = np.linalg.norm(t1)
    if n1 < 1e-14:
        t1 = np.array([0.0, 0.0, 1.0], float) - np.dot(np.array([0.0, 0.0, 1.0], float), n) * n
        n1 = np.linalg.norm(t1)
        if n1 < 1e-14:
            return (np.nan, np.nan)
    t1 /= n1
    t2 = np.cross(n, t1)
    n2 = np.linalg.norm(t2)
    if n2 < 1e-14:
        return (np.nan, np.nan)
    t2 /= n2
    S11 = -(t1 @ (Hm @ t1)) / ng
    S22 = -(t2 @ (Hm @ t2)) / ng
    S12 = -(t1 @ (Hm @ t2)) / ng
    w = np.linalg.eigvalsh(np.array([[S11, S12], [S12, S22]], float))
    k1, k2 = (float(w[1]), float(w[0]))
    k1, k2 = adjust_curvature_signs(k1, k2, coeffs, observer_outside=observer_outside)
    if k1 < k2:
        k1, k2 = (k2, k1)
    return (k1, k2)

def principal_curvatures_rule_label(k2A, k2B, k2C, tol=1e-12):
    k2s = np.array([k2A, k2B, k2C], float)
    if np.all(np.isfinite(k2s)) and np.all(k2s < -tol):
        return 'inward'
    if np.all(np.isfinite(k2s)) and np.all(k2s >= -tol):
        return 'outward'
    return 'between'

def project_to_quadric_normal_step(P0, coeffs, max_iter=20, tol=1e-12):
    """Generic projection by normal steps onto f(x)=0."""
    A3, b, c_ = _parse_quadric_coeffs(coeffs)
    x = np.asarray(P0, float)
    for _ in range(max_iter):
        f = float(x @ (A3 @ x) + 2.0 * b @ x + c_)
        g = 2.0 * (A3 @ x + b)
        g2 = float(np.dot(g, g))
        if not np.isfinite(g2) or g2 < 1e-30:
            break
        step = f / g2
        x_new = x - step * g
        if np.linalg.norm(x_new - x) < tol * (1.0 + np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
    return x

def quadric_kind(coeffs, tol=1e-10):
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    lin_xy = max(abs(G), abs(H))
    off = max(abs(D), abs(E), abs(F))
    if abs(A - B) <= tol * max(1.0, abs(A), abs(B)) and A > 0 and (abs(C) <= tol) and (off <= tol) and (lin_xy <= tol) and (abs(I) <= tol):
        return 'circular_cylinder'
    if abs(A - B) <= tol * max(1.0, abs(A), abs(B)) and abs(B - C) <= tol * max(1.0, abs(B), abs(C)) and (A > 0) and (off <= tol) and (lin_xy <= tol) and (abs(I) <= tol):
        return 'sphere'
    if abs(A - B) <= tol * max(1.0, abs(A), abs(B)) and A > 0 and (abs(C) <= tol) and (off <= tol) and (lin_xy <= tol) and (abs(I) > tol):
        return 'elliptic_paraboloid'
    if abs(A - B) <= tol * max(1.0, abs(A), abs(B)) and A > 0 and (C < -tol) and (off <= tol) and (lin_xy <= tol):
        return 'hyperboloid_one'
    if abs(A - B) <= tol * max(1.0, abs(A), abs(B)) and A < -tol and (C > tol) and (off <= tol) and (lin_xy <= tol):
        return 'hyperboloid_two'
    return 'other'

def tet_volume(pts):
    p0, p1, p2, p3 = pts
    return _tet_volume_apex(p0, p1, p2, p3)

def to_axis(*args, **kwargs):
    raise NotImplementedError("Function 'to_axis' was not found in the source files during consolidation.")

def validate_swept_segment():
    c_loc = 1.0
    coeffs = (1.0, 1.0, 1.0 / c_loc ** 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    z_A = math.sqrt(2) / 2
    z_B = -math.sqrt(2) / 2
    V_num, mode = swept_segment_volume_AB(coeffs, z_A, z_B)
    V_target = 4.0 * math.pi / 3.0 - math.pi / 2.0 * math.sqrt(2.0) - 2.0 * 0.2432
    V_exact = math.pi * math.sqrt(2.0) / 3.0
    print('\n=== SWEPT SEGMENT VALIDATION (sphere) ===')
    print('Mode used:                    ', mode)
    print('Numeric (this function):       %.12f' % V_num)
    print('Your target (4/3*pi - ...):    %.12f' % V_target)
    print('Closed-form (pi*sqrt(2)/3):    %.12f' % V_exact)
    print('|V - target|                 = %.3e' % abs(V_num - V_target))
    print('|exact - target|             = %.3e' % abs(V_exact - V_target))

def wedge_volume_rotation(Coeffs, A, B, C, idx, forced_orientation):
    if idx in DEBUG_TRI:
        print('+++++ Wedge volume rotation called with idx=', idx,
              'Coeffs:', Coeffs, 'A:', A, 'B:', B, 'C:', C,
              'forced_orientation:', forced_orientation)

    ok, axis_dir, axis_point, frame = _axis_frame_and_center_from_coeffs(Coeffs)
    if not ok:
        return (None, 0.0)

    u, v, k = frame

    def to_axis(P):
        return np.array([
            np.dot(P - axis_point, u),
            np.dot(P - axis_point, v),
            np.dot(P - axis_point, k),
        ], float)

    Ap = to_axis(np.asarray(A, float))
    Bp = to_axis(np.asarray(B, float))
    Cp = to_axis(np.asarray(C, float))

    zvals = [Ap[2], Bp[2], Cp[2]]
    idxs = np.argsort(zvals)
    Ap, Bp, Cp = ([Ap, Bp, Cp][idxs[2]], [Ap, Bp, Cp][idxs[0]], [Ap, Bp, Cp][idxs[1]])

    rA, thA, zA = (np.hypot(Ap[0], Ap[1]), math.atan2(Ap[1], Ap[0]), Ap[2])
    rB, thB, zB = (np.hypot(Bp[0], Bp[1]), math.atan2(Bp[1], Bp[0]), Bp[2])
    rC, thC, zC = (np.hypot(Cp[0], Cp[1]), math.atan2(Cp[1], Cp[0]), Cp[2])

    EPS = 1e-10
    k_exp = 1
    orientation = forced_orientation if forced_orientation in ('inward', 'outward', 'between') else 'outward'
    is_between = orientation == 'between'

    if abs(Cp[2] - Ap[2]) < EPS and abs(Bp[2] - Ap[2]) < EPS:
        return (0, 0.0)

    Ep = extend_arc_point_to_z(Ap, rA, Bp, rB, Cp, rC, idx)
    Fp, rF, Gp = extend_C_to_AB(Ap, rA, Bp, rB, Cp, rC, idx)

    if orientation == 'inward':
        V_ABCE = _tet_volume_apex(Ep, Cp, Ap, Bp)
        V_1 = cone_volume_patch_inside(Ap, Ep, Cp, 0.0, rC, k_exp, idx)
        V_2 = cone_volume_patch_inside(Ep, Bp, Cp, 0.0, rC, k_exp, idx)
        V = (V_1 + V_2)

        if rF < rC:
            V_ABCE = -V_ABCE
        else:
            V_beforeCross = (
                cone_volume_patch_inside(Ap, Gp, Cp, 0.0, rC, k_exp, idx)
                + cone_volume_patch_inside(Gp, Bp, Cp, 0.0, rC, k_exp, idx)
            )
            V_afterCross = (
                cone_volume_patch_inside(Ap, Ep, Gp, 0.0, rC, k_exp, idx)
                + cone_volume_patch_inside(Ep, Bp, Gp, 0.0, rC, k_exp, idx)
            )
            V = V_afterCross + V_beforeCross
            V_ABCE = min(
                _tet_volume_apex(Ep, Gp, Ap, Bp),
                curved_tet_volume(Ep, Gp, rC, Ap, Bp),
            )

        if abs(Cp[2] - Ap[2]) < EPS or abs(Cp[2] - Bp[2]) < EPS:
            V_ABCE_curved = 0.0  # kept for logical clarity; not used

        V = V + V_ABCE
        return (5, -V)

    if abs(Cp[2] - Ap[2]) < EPS and (not is_between):
        dth = (thC - thA) % (2 * np.pi)
        if dth < 1e-08:
            return (0, 0.0)
        if dth >= np.pi:
            dth = 2 * np.pi - dth
        r_plane = 0.5 * (rA + rC)
        Vpatch = cone_volume_patch(Ap, Bp, Cp, 0.0, r_plane, k_exp, idx)
        return (1, Vpatch)

    if abs(Cp[2] - Bp[2]) < EPS and (not is_between):
        dth = (thC - thB) % (2 * np.pi)
        if dth < 1e-08:
            return (0, 0.0)
        if dth >= np.pi:
            dth = 2 * np.pi - dth
        r_plane = 0.5 * (rB + rC)
        Vpatch = cone_volume_patch(Ap, Bp, Cp, 0.0, r_plane, k_exp, idx)
        return (2, Vpatch)

    thC2 = math.atan2(Cp[1], Cp[0])
    thE = math.atan2(Ep[1], Ep[0])
    dth = abs(math.remainder(thE - thC2, 2.0 * math.pi))
    if dth > math.pi:
        dth = 2.0 * math.pi - dth
    split = 2.0 * math.pi / (dth + 1e-12)

    if is_between:
        V1, _m1 = swept_segment_volume_AB(COEFFS, Ap[2], Ep[2])
        V2, _m2 = swept_segment_volume_AB(COEFFS, Bp[2], Ep[2])
        V = (
            V1 * rC ** k_exp / (rA ** k_exp + rC ** k_exp)
            + V2 * rC ** k_exp / (rB ** k_exp + rC ** k_exp)
        ) / split
        V += _tet_volume_apex(Ep, Cp, Ap, Bp)
        return (4, V)

    V1 = cone_volume_patch(Ap, Ep, Cp, 0.0, rC, k_exp, idx)
    V2 = cone_volume_patch(Ep, Bp, Cp, 0.0, rC, k_exp, idx)
    V3 = _tet_volume_apex(Ep, Ap, Bp, Cp)
    return (3, V1 + V2 + V3)

# ===================== ADDED: plane-lift curved area =====================

def _quadric_lift_point_along_normal(P_plane, n_hat, A3, b, c_, root_mode="nearest"):
    """
    Given a point P_plane in the base plane and a unit normal n_hat,
    solve for λ such that F(P_plane + λ n_hat) = 0, where

        F(x) = x^T A3 x + 2 b·x + c_.

    Returns the lifted point on the quadric surface.

    root_mode:
        "nearest"  -> choose the root λ with smallest |λ|
    """
    P0 = np.asarray(P_plane, float)
    n_hat = np.asarray(n_hat, float)

    # Quadratic in λ: a λ^2 + b λ + c = 0
    a_q = float(n_hat @ (A3 @ n_hat))
    b_q = float(2.0 * n_hat @ (A3 @ P0) + 2.0 * (b @ n_hat))
    c_q = float(P0 @ (A3 @ P0) + 2.0 * (b @ P0) + c_)

    # If we are already very close to the surface, do nothing.
    if abs(c_q) < 1e-16 and abs(b_q) < 1e-16:
        return P0

    # Degenerate quadratic -> linear
    if abs(a_q) < 1e-18:
        if abs(b_q) < 1e-18:
            # Completely degenerate; fall back to plane
            return P0
        lam = -c_q / b_q
        return P0 + lam * n_hat

    disc = b_q * b_q - 4.0 * a_q * c_q
    # If disc is slightly negative from roundoff, clamp to 0.
    if disc < 0.0:
        if disc > -1e-12 * max(1.0, b_q * b_q, 4.0 * abs(a_q * c_q)):
            disc = 0.0
        else:
            # No real intersection; fall back to plane
            return P0

    sqrt_disc = math.sqrt(max(0.0, disc))
    lam1 = (-b_q - sqrt_disc) / (2.0 * a_q)
    lam2 = (-b_q + sqrt_disc) / (2.0 * a_q)

    if root_mode == "nearest":
        lam = lam1 if abs(lam1) < abs(lam2) else lam2
    else:
        # Fallback: also use nearest if unknown mode
        lam = lam1 if abs(lam1) < abs(lam2) else lam2

    return P0 + lam * n_hat

def _lift_points_diag_quadric(P_plane, n_hat, coeffs):
    """
    Lift an array of points P_plane (shape (M,3)) along direction n_hat onto
    the quadric defined by

        F(x,y,z) = A x^2 + B y^2 + C z^2 + J = 0

    assuming D=E=F=G=H=I≈0 (diagonal form), with coeffs =
        (A,B,C,D,E,F,G,H,I,J).

    For each point, solve:
        F(P + λ n_hat) = 0
    which is a scalar quadratic in λ:

        α λ^2 + β λ + γ = 0

    and select the root with smallest |λ|.
    """
    P_plane = np.asarray(P_plane, dtype=float)
    A_diag, B_diag, C_diag, _, _, _, _, _, _, J = coeffs

    nx, ny, nz = n_hat
    Px = P_plane[:, 0]
    Py = P_plane[:, 1]
    Pz = P_plane[:, 2]

    # Quadratic coefficients for each point
    alpha = A_diag * nx * nx + B_diag * ny * ny + C_diag * nz * nz  # scalar
    beta = 2.0 * (A_diag * Px * nx + B_diag * Py * ny + C_diag * Pz * nz)
    gamma = A_diag * Px * Px + B_diag * Py * Py + C_diag * Pz * Pz + J

    eps = 1e-14
    lam = np.zeros_like(Px)

    if abs(alpha) < eps:
        # Degenerate quadratic => linear equation beta λ + gamma = 0
        mask_lin = np.abs(beta) > eps
        lam[mask_lin] = -gamma[mask_lin] / beta[mask_lin]
        # Where it's doubly degenerate, keep λ=0 (no movement)
    else:
        # Proper quadratic case
        disc = beta * beta - 4.0 * alpha * gamma

        # Clamp very small negative discriminants to zero (numeric noise)
        disc_clamped = np.where(disc < 0.0, np.maximum(disc, -1e-12), disc)
        sqrt_disc = np.sqrt(np.maximum(disc_clamped, 0.0))

        lam1 = (-beta + sqrt_disc) / (2.0 * alpha)
        lam2 = (-beta - sqrt_disc) / (2.0 * alpha)

        # choose the root with smallest |λ|
        use_lam1 = np.abs(lam1) < np.abs(lam2)
        lam = np.where(use_lam1, lam1, lam2)

        # For truly negative discriminants (beyond tolerance), keep λ = 0
        bad_disc = disc < -1e-12
        lam[bad_disc] = 0.0

    # Lifted points
    P_lifted = P_plane + lam[:, None] * np.array(n_hat, dtype=float)
    return P_lifted

def triangle_flat_area(A, B, C):
    """
    Standard flat triangle area in 3D.
    """
    AB = B - A
    AC = C - A
    return 0.5 * np.linalg.norm(np.cross(AB, AC))

def curved_patch_area_plane_lift_with_D(A, B, C, N, coeffs, root_mode="nearest"):
    """
    [curved_D] A_curved_D (GENERAL VERSION, SELF-CONTAINED)

    Curved area of the quadric patch bounded by triangle ABC on the
    general quadric

        F(x,y,z) = A x^2 + B y^2 + C z^2
                   + D x y + E x z + F y z
                   + G x + H y + I z + J = 0,

    using recursive barycenter (centroid) refinement in the base plane
    and plane-lift projection along the triangle's plane normal.

    Parameters
    ----------
    A, B, C : (3,) array-like
        Triangle vertices in ORIGINAL coordinates.
    N : int
        Number of centroid-refinement levels (N >= 0).
    coeffs : 10-tuple
        (A,B,C,D,E,F,G,H,I,J) quadric coefficients in ORIGINAL frame.
    root_mode : str, optional
        Root selection mode for the line–quadric intersection.
        Currently only "nearest" is supported (pick root with smallest |t|).

    Returns
    -------
    A_curved_D : float
        Approximate curved surface area of the quadric patch.
    """
    # ---- local helpers (self-contained, no external dependencies) ----

    def _local_triangle_flat_area(P, Q, R):
        """Flat area of triangle PQR in 3D."""
        P = np.asarray(P, float)
        Q = np.asarray(Q, float)
        R = np.asarray(R, float)
        return 0.5 * np.linalg.norm(np.cross(Q - P, R - P))

    def _local_project_point_to_quadric_along_normal(P, n, coeffs, root_mode="nearest", eps=1e-14):
        """
        Project point P onto the quadric F(x,y,z)=0 along the line

            L(t) = P + t * n

        by solving F(P + t n) = 0 for t (quadratic in t), and choosing a root.

        This is fully general for coeffs = (A,B,C,D,E,F,G,H,I,J).
        """
        P = np.asarray(P, dtype=float)
        n = np.asarray(n, dtype=float)

        norm_n = np.linalg.norm(n)
        if norm_n < eps:
            # Degenerate normal, just return original point
            return P.copy()
        n = n / norm_n

        x0, y0, z0 = P
        nx, ny, nz = n
        A_q, B_q, C_q, D_q, E_q, F_q, G_q, H_q, I_q, J_q = coeffs

        # Coefficients of F(P + t n) = a t^2 + b t + c
        a = (
            A_q * nx * nx +
            B_q * ny * ny +
            C_q * nz * nz +
            D_q * nx * ny +
            E_q * nx * nz +
            F_q * ny * nz
        )
        b = (
            2.0 * A_q * nx * x0 +
            2.0 * B_q * ny * y0 +
            2.0 * C_q * nz * z0 +
            D_q * (nx * y0 + ny * x0) +
            E_q * (nx * z0 + nz * x0) +
            F_q * (ny * z0 + nz * y0) +
            G_q * nx +
            H_q * ny +
            I_q * nz
        )
        c = (
            A_q * x0 * x0 +
            B_q * y0 * y0 +
            C_q * z0 * z0 +
            D_q * x0 * y0 +
            E_q * x0 * z0 +
            F_q * y0 * z0 +
            G_q * x0 +
            H_q * y0 +
            I_q * z0 +
            J_q
        )

        # Handle degenerate cases
        if abs(a) < eps:
            # Linear or constant
            if abs(b) < eps:
                # F ≈ c; if already near the surface, keep P
                if abs(c) < 1e-10:
                    return P.copy()
                else:
                    # No variation along this normal; fall back to original
                    return P.copy()
            else:
                # Linear equation b t + c = 0
                t = -c / b
                return P + t * n

        # Quadratic case
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            # No real intersection; fall back
            return P.copy()

        sqrt_disc = math.sqrt(max(disc, 0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        if root_mode == "nearest":
            # Choose root closest to t=0
            t = t1 if abs(t1) <= abs(t2) else t2
        else:
            # Fallback: nearest anyway
            t = t1 if abs(t1) <= abs(t2) else t2

        return P + t * n

    # ---- main body of curved_patch_area_plane_lift_with_D ----

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)

    # Base-plane normal (single normal reused for all sub-triangles)
    n_plane = np.cross(B - A, C - A)
    n_norm = np.linalg.norm(n_plane)
    if n_norm == 0.0:
        return 0.0
    n_plane = n_plane / n_norm

    tri_list = [(A, B, C)]
    levels = max(int(N), 0)

    # Recursive centroid refinement
    for _ in range(levels):
        new_list = []
        for (P, Q, R) in tri_list:
            D_plane = (P + Q + R) / 3.0
            new_list.append((P, Q, D_plane))
            new_list.append((Q, R, D_plane))
            new_list.append((R, P, D_plane))
        tri_list = new_list

    area_curved_D = 0.0
    for (P, Q, R) in tri_list:
        P_s = _local_project_point_to_quadric_along_normal(P, n_plane, coeffs, root_mode=root_mode)
        Q_s = _local_project_point_to_quadric_along_normal(Q, n_plane, coeffs, root_mode=root_mode)
        R_s = _local_project_point_to_quadric_along_normal(R, n_plane, coeffs, root_mode=root_mode)
        area_curved_D += _local_triangle_flat_area(P_s, Q_s, R_s)

    return float(area_curved_D)


def curved_patch_area_plane_lift(A, B, C, coeffs=COEFFS, N_sub=8, root_mode="nearest"):
    """
    Physically correct curved area A_curved of the quadric patch bounded by
    the *surface curves* AB, BC, CA, working in a single consistent frame.

    IMPORTANT for your pipeline:
    - A, B, C must be the ORIGINAL triangle vertices from _COEFFS.csv
      (A_x, A_y, A_z, ..., C_z).
    - coeffs must be the ORIGINAL 1× quadric coefficients from the SAME row
      of _COEFFS.csv (A,B,C,D,E,F,G,H,I,J).
    - Do NOT mix them with any *_COEFFS_Transformed.csv or scale_factors.

    Method:
      - Take the plane through A,B,C (the base plane).
      - Subdivide the flat triangle ABC into many small plane triangles.
      - For each small triangle:
          * For each vertex P_plane, move along the base-plane normal n_hat and
            solve F(P_plane + λ n_hat) = 0 for λ (closest root).
          * This gives a lifted vertex on the quadric surface.
          * Accumulate area from these small curved triangles in 3D.

    Inputs
    ------
    A, B, C : (3,) array-like
        Vertices of the triangle in ORIGINAL coordinates.
    coeffs : length-10 iterable
        Quadric coefficients in the 1× convention:
            (A,B,C,D,E,F,G,H,I,J) with cross/linear terms as in _parse_quadric_coeffs.
    N_sub : int
        Subdivision resolution (≈ N_sub^2 small triangles). Typical: 6–10.

    Returns
    -------
    A_curved : float
        Approximate curved area on the quadric in the same frame/units as A,B,C.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)

    # Base-plane normal and flat area
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-16:
        # Degenerate triangle; no area
        return 0.0
    n_hat = n / n_norm

    # Parse quadric
    A3, b, c_ = _parse_quadric_coeffs(coeffs)

    # Helper: lift any point on the base plane
    def lift(P_plane):
        return _quadric_lift_point_along_normal(P_plane, n_hat, A3, b, c_, root_mode=root_mode)

    # Barycentric subdivision of the big triangle ABC
    # P(i,j) = A + (i/N) * AB + (j/N) * AC, with j <= N - i
    N = max(1, int(N_sub))
    area_curved = 0.0

    for i in range(N):
        for j in range(N - i):
            s0 = i / N
            t0 = j / N
            P0_plane = A + s0 * AB + t0 * AC
            P0_surf = lift(P0_plane)

            # First small triangle: (i,j) -> (i+1,j) -> (i,j+1)
            if i + 1 <= N:
                s1 = (i + 1) / N
                t1 = t0
                P1_plane = A + s1 * AB + t1 * AC
                P1_surf = lift(P1_plane)
            else:
                P1_surf = None

            if j + 1 <= N - i:
                s2 = s0
                t2 = (j + 1) / N
                P2_plane = A + s2 * AB + t2 * AC
                P2_surf = lift(P2_plane)
            else:
                P2_surf = None

            if (P1_surf is not None) and (P2_surf is not None):
                v1 = P1_surf - P0_surf
                v2 = P2_surf - P0_surf
                area_curved += 0.5 * np.linalg.norm(np.cross(v1, v2))

            # Second small triangle: (i+1,j) -> (i+1,j+1) -> (i,j+1)
            if i + 1 < N and j + 1 <= N - (i + 1):
                s3 = (i + 1) / N
                t3 = (j + 1) / N
                P3_plane = A + s3 * AB + t3 * AC
                P3_surf = lift(P3_plane)

                if (P1_surf is not None) and (P3_surf is not None) and (P2_surf is not None):
                    v1 = P3_surf - P1_surf
                    v2 = P2_surf - P1_surf
                    area_curved += 0.5 * np.linalg.norm(np.cross(v1, v2))

    return float(area_curved)


def rotation_patch_area_transformed(coeffs, A, B, C, nz=8, root_mode="nearest"):
    """
    Backward-compatible wrapper that computes the curved area of the patch
    bounded by AB, BC, CA on the quadric F(x)=0 described by 'coeffs',
    using the plane-lift method in WHATEVER frame A,B,C live in.

    Historically this name suggested 'axis/transformed frame', but now the
    implementation is general and frame-agnostic. The only requirement is:

        - A, B, C are in the same frame as 'coeffs'.

    For your pipeline you should still follow the rule:
        A, B, C and 'coeffs' must come from the SAME ROW of the SAME CSV file.
        For original geometry, that means _COEFFS.csv (NOT *_COEFFS_Transformed.csv).

    Parameters
    ----------
    coeffs : 10-tuple
        Quadric coefficients in 1× form (A,B,C,D,E,F,G,H,I,J).
    A, B, C : (3,) array-like
        Vertices of the triangle (same coordinates as used for 'coeffs').
    nz : int
        Subdivision resolution for the plane-lift area integration (kept name for compatibility).
    root_mode : str
        "nearest" -> choose the λ-root closest to 0 when lifting.

    Returns
    -------
    A_curved : float
        Curved area on the quadric in the same length units as A,B,C.
    """
    N_sub = max(1, int(nz))
    # If you want transformed-frame curved area, enable the line below
    # return curved_patch_area_plane_lift(A, B, C, coeffs=coeffs, N_sub=N_sub, root_mode=root_mode)
    return 0.0


def rotation_patch_area_original(coeffs, A, B, C, sx=None, sy=None, sz=None, nz=3, nth=8, root_mode="nearest"):
    """
    Physically correct A_curved in ORIGINAL coordinates.

    This keeps the old signature (sx,sy,sz,nz,nth) for compatibility, but the
    scale_factors are no longer needed or used. Everything is computed directly
    in the ORIGINAL frame via plane-lift integration.

    VERY IMPORTANT (as you requested):
      - A, B, C MUST be taken from _COEFFS.csv (columns A_x, A_y, A_z, ..., C_z)
        for this triangle.
      - coeffs MUST be (A,B,C,D,E,F,G,H,I,J) from the SAME ROW of _COEFFS.csv.
      - Do NOT pass coordinates or coeffs from _COEFFS_Transformed.csv here.
        No scale_factors1_x/y/z are used anywhere in this function.

    Parameters
    ----------
    coeffs : 10-tuple
        Quadric coefficients in 1× form in ORIGINAL frame (from _COEFFS.csv).
    A, B, C : (3,) array-like
        Triangle vertices in ORIGINAL coordinates (from _COEFFS.csv).
    sx, sy, sz : float or None
        Ignored (kept for call-site compatibility).
    nz : int
        Subdivision resolution (≈ nz^2 small triangles). Typical effective 6–10.
    nth : int
        Ignored (legacy).
    root_mode : str
        "nearest" (default) chooses the intersection closest to the base plane.

    Returns
    -------
    A_curved : float
        Curved surface area in ORIGINAL coordinates.
    """
    SPHERE_COEFFS = (1, 1, 1, 0, 0, 0, 0, 0, 0, -1)
    #nz=2
    if coeffs != SPHERE_COEFFS:
        #Area = curved_patch_area_plane_lift_with_D(A, B, C, N=nz, coeffs=coeffs)
        Area =  0#curved_patch_area_plane_lift(A, B, C, coeffs=coeffs, N_sub=nz, root_mode=root_mode) #0#
    else:
        Area = 0.0
    return Area 

# ===================== END ADDED PART =====================

def worker(args):
    idx, tri, points = args
    A, B, C = points[tri]
    k1_A, k2_A = principal_curvatures_at(A, COEFFS, project=True, observer_outside=OBSERVER_OUTSIDE)
    k1_B, k2_B = principal_curvatures_at(B, COEFFS, project=True, observer_outside=OBSERVER_OUTSIDE)
    k1_C, k2_C = principal_curvatures_at(C, COEFFS, project=True, observer_outside=OBSERVER_OUTSIDE)
    label = principal_curvatures_rule_label(k2_A, k2_B, k2_C)
    case_id, V = wedge_volume_rotation(COEFFS, A, B, C, idx, forced_orientation=label)
    row = dict(
        triangle_id=idx,
        Vcorrection=(case_id, np.float64(V)),
        surface_triangle=True,
        pointdirection=label,
        side=label,
        k1_A=np.float64(k1_A),
        k2_A=np.float64(k2_A),
        k1_B=np.float64(k1_B),
        k2_B=np.float64(k2_B),
        k1_C=np.float64(k1_C),
        k2_C=np.float64(k2_C),
    )
    return (row, V)
