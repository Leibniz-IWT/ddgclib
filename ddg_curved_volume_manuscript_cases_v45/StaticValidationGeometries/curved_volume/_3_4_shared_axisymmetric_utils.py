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
except Exception:
    def swept_segment_volume_AB(*args, **kwargs):
        raise NotImplementedError("swept_segment_volume_AB is not available; import it in your caller or pass an alternative.")
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
  "worker"
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
    a,b,c,d,e,f,g,h,i,j = map(float, coeffs_1x)
    return (a, b, c, d/2.0, e/2.0, f/2.0, g/2.0, h/2.0, i/2.0, j)

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
    A,B,C,D,E,F,G,H,I,J = map(float, coeffs)
    A3 = np.array([
        [A,   D/2, E/2],
        [D/2, B,   F/2],
        [E/2, F/2, C   ],
    ], float)
    b  = 0.5 * np.array([G, H, I], float)
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

def adjust_curvature_signs(k1, k2, coeffs, observer_outside=True):
    kind = quadric_kind(coeffs)
    if kind in ('circular_cylinder', 'sphere', 'elliptic_paraboloid'):
        return (-k1, -k2) if observer_outside else (k1, k2)
    if 'hyperboloid' in kind:
        return (k1, k2)
    return (k1, k2)


    
    EPS = 1e-12
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
        dth = (thC - thA) % (2 * np.pi)
        D = find_rotated_point(B, rB, -d_CA)
        if dth < 1e-08:
            return 0.0
        if dth >= np.pi:
            dth = 2 * np.pi - dth
        tri_COA = 0.5 * rA ** 2 * np.sin(dth)
        tri_BOD = 0.5 * rB ** 2 * np.sin(dth)
        split = 2 * np.pi / (dth + 1e-12)
        V_OABC = abs(np.linalg.det(np.vstack([A - O, B - O, C - O]))) / 6.0
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6.0
        if abs(rB - rC) < 1e-07:
            V_swept = 0.0
        else:
            V_swept, _mode = swept_segment_volume_AB(COEFFS, C[2], B[2])
            #V_swept, _mode = swept_segment_volume_A1B((1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0), C[2], B[2])

        if triangle_id == 51: print(triangle_id,COEFFS,"V_swept",V_swept)

        Vpatch = (V_t + V_swept - V_OABC * split - V_OABD * split - tri_BOD * split * h_full / 6.0 - tri_COA * split * h_full / 6.0) / split * (rA ** k / (rA ** k + rB ** k))
        return Vpatch
    if abs(C[2] - B[2]) < EPS:
        dth = (thC - thB) % (2 * np.pi)
        if dth < 1e-08:
            return 0.0
        if dth >= np.pi:
            dth = 2 * np.pi - dth
        D = find_rotated_point(A, rA, -d_CB)
        V_OABC = abs(np.linalg.det(np.vstack([A - O, B - O, C - O]))) / 6.0
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6.0
        tri_COB = 0.5 * rB ** 2 * np.sin(dth)
        tri_DOA = 0.5 * rA ** 2 * np.sin(dth)
        split = 2 * np.pi / (dth + 1e-12)
        if abs(rB - rA) < 1e-07:
            V_swept = 0.0
        else:
            V_swept, _mode = swept_segment_volume_AB(COEFFS, A[2], B[2])
        return (V_t + V_swept - V_OABC * split - V_OABD * split - tri_COB * split * h_full / 6.0 - tri_DOA * split * h_full / 6.0) / split * (rB ** k / (rB ** k + rA ** k))
    return 0.0
def cone_volume_patch(A, B, C, R_unused, r_plane, k, triangle_id=None):
    EPS = 1e-12
    import math, numpy as np

    # ---- tiny helpers ----
    def tet(apex, P, Q, R):
        return abs(np.linalg.det(np.vstack([P - apex, Q - apex, R - apex]))) / 6.0

    def mid_angle_minor(thA, thC):
        """Mid-angle along the MINOR arc between thA and thC using only % (2π)."""
        d = (thC - thA) % (2*math.pi)  # thA -> thC
        if d <= math.pi:
            return (thA + 0.5*d) % (2*math.pi)
        else:
            L = (2*math.pi - d)          # minor arc is thC->thA
            return (thC + 0.5*L) % (2*math.pi)

    def on_minor_arc(theta, thA, thC, tol=1e-12):
        """True iff theta lies on the MINOR arc between thA and thC, using only % (2π)."""
        d_minor = (thC - thA) % (2*math.pi)
        if d_minor > math.pi:
            d_minor = 2*math.pi - d_minor
            c2t = (theta - thC) % (2*math.pi)
            t2a = (thA   - theta) % (2*math.pi)
            return abs((c2t + t2a) - d_minor) <= tol
        else:
            a2t = (theta - thA) % (2*math.pi)
            t2c = (thC   - theta) % (2*math.pi)
            return abs((a2t + t2c) - d_minor) <= tol

    def on_minor_arc_strict(theta, thP, thQ, tol=1e-12):
        """STRICT: theta strictly inside the MINOR arc between thP and thQ (exclude endpoints)."""
        two_pi = 2*math.pi
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
        x = float(P[0]); y = float(P[1])
        r = math.hypot(x, y)
        eps = max(eps_base, 1e-9 * max(1.0, r))
        if abs(y) <= eps:
            return 0.0 if x >= 0.0 else math.pi
        a = math.atan2(y, x)
        if a < 0.0:
            a += 2.0 * math.pi
        if abs(a - 2.0*math.pi) <= eps:
            a = 0.0
        return a

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

    #print("A",A, "B", B," coeffs_2x_to_1x(COEFFS)", coeffs_2x_to_1x(COEFFS))


    ratio_AB_A = (rA**k) / (rA**k + rB**k + 1e-300)
    ratio_AB_B = (rB**k) / (rA**k + rB**k + 1e-300)

    #print("ratio_AB_A", ratio_AB_A)
    if B[2]>99999:
        ratio, single_us, avg_ns, _ = time_ratioV(A, B, coeffs_2x_to_1x(COEFFS), repeats=20000)
        ratio_AB_A = ratio/(ratio+1)
        ratio_AB_B = 1-ratio_AB_A
        #print(f"ratioV(A,B)={ratio:.6g} single_us={single_us:.3g} avg_ns={avg_ns:.3g} triangle_id={triangle_id}")
        #print(ratio/(ratio+1))

    eps = 1e-300
    # exact split for paraboloid z = α r^2 + β
    #ratio_AB_A = (3.0*rA + 2.0*rB) / (5.0*(rA + rB) + eps)  # V_t fraction
    #ratio_AB_B = (2.0*rA + 3.0*rB) / (5.0*(rA + rB) + eps)  # V_b fraction

    # ======================= CASE 1: C ≈ A =======================
    if abs(C[2] - A[2]) < EPS:
        rAC   = rA
        EPS_R = 1e-14
        thB_eff = thB if rB > EPS_R else mid_angle_minor(thA, thC)

        Bt = np.array([rAC*math.cos(thB_eff), rAC*math.sin(thB_eff), A[2]], float)
        Ab = np.array([rB *math.cos(thA),     rB *math.sin(thA),     B[2]], float)
        Cb = np.array([rB *math.cos(thC),     rB *math.sin(thC),     B[2]], float)

        d_BA = (thB_eff - thA) % (2*math.pi)
        if d_BA > math.pi: d_BA = 2*math.pi - d_BA
        d_BC = (thC - thB_eff) % (2*math.pi)
        if d_BC > math.pi: d_BC = 2*math.pi - d_BC
        d_AC = (thC - thA) % (2*math.pi)
        if d_AC > math.pi: d_AC = 2*math.pi - d_AC

        split_BA = 2.0*math.pi / (d_BA + 1e-12)
        split_BC = 2.0*math.pi / (d_BC + 1e-12)

        V_OABBt  = tet(O,  A,  B,  Bt)
        V_OABAb  = tet(O,  A,  B,  Ab)
        V_O1OBtA = tet(O1, O,  Bt, A)
        V_O2OBAb = tet(O2, O,  B,  Ab)
        V_BBtA   = (V_cut/split_BA - V_OABBt - V_OABAb - V_O1OBtA - V_O2OBAb) * ratio_AB_A

        V_OBCBt  = tet(O,  B,  C,  Bt)
        V_OBCCb  = tet(O,  B,  C,  Cb)
        V_O1OBtC = tet(O1, O,  Bt, C)
        V_O2OBCb = tet(O2, O,  B,  Cb)
        V_BBtC   = (V_cut/split_BC - V_OBCBt - V_OBCCb - V_O1OBtC - V_O2OBCb) * ratio_AB_A

        V_BBtAC = tet(B, Bt, A, C)

        Bt_between_CA = on_minor_arc(thB_eff, thA, thC)

        if not Bt_between_CA:
            Vpatch = abs(V_BBtA - V_BBtC) - V_BBtAC
        else:
            Vpatch = (V_BBtA + V_BBtC) + V_BBtAC

        if triangle_id in (1,5):
            oBt = otheta(Bt); oAb = otheta(Ab); oCb = otheta(Cb); oC = otheta(C); oA = otheta(A)
            print("inside-ID", triangle_id, "C~A branch")
            print("A", A, "B", B, "C", C, "Bt", Bt, "Bt_between_CA", Bt_between_CA, "\nVpatch?", Vpatch,"coeffs", COEFFS)
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
        Bt  = np.array([rAC*math.cos(thB), rAC*math.sin(thB), A[2]], float)  # B -> AC circle
        Ab  = np.array([rB *math.cos(thA), rB *math.sin(thA), B[2]], float)  # A -> B-plane circle
        Cb  = np.array([rB *math.cos(thC), rB *math.sin(thC), B[2]], float)  # C -> B-plane circle
        Ct  = np.array([rA *math.cos(thC), rA *math.sin(thC), A[2]], float)  # C -> A-plane circle

        # Avoid Ab/Cb coinciding with B by nudging along minor-arc direction
        eps_ang = 1e-9
        dir_sign = +1 if ((thC - thA) % (2*math.pi)) <= math.pi else -1
        if np.linalg.norm(Ab - B) < 1e-14:
            thA_eps = (thA + dir_sign*eps_ang) % (2*math.pi)
            Ab = np.array([rB*math.cos(thA_eps), rB*math.sin(thA_eps), B[2]], float)
        if np.linalg.norm(Cb - B) < 1e-14:
            thC_eps = (thC - dir_sign*eps_ang) % (2*math.pi)
            Cb = np.array([rB*math.cos(thC_eps), rB*math.sin(thC_eps), B[2]], float)

        d_BA_minor = (thB - thA) % (2*math.pi)
        if d_BA_minor > math.pi: d_BA_minor = 2*math.pi - d_BA_minor
        split_BA = 2.0*math.pi / (d_BA_minor + 1e-12)

        d_AC_minor = (thC - thA) % (2*math.pi)
        if d_AC_minor > math.pi: d_AC_minor = 2*math.pi - d_AC_minor
        split_CA = 2.0*math.pi / (d_AC_minor + 1e-12)

        # V_AAbB
        V_OABAb  = tet(O,  A,  B,  Ab)
        V_OABBt  = tet(O,  A,  B,  Bt)
        V_O1OBtA = tet(O1, O,  Bt, A)
        V_O2OBAb = tet(O2, O,  B,  Ab)
        V_AAbB   = (V_cut/split_BA - V_OABAb - V_OABBt - V_O1OBtA - V_O2OBAb) * ratio_AB_B

        # V_AAbC   (use Ct per your algorithm)
        V_OACAb  = tet(O,  A,  C,  Ab)
        V_OACCt  = tet(O,  A,  C,  Ct)
        V_O1OACt = tet(O1, O,  A,  Ct)
        V_O2OCAb = tet(O2, O,  C,  Ab)
        V_AAbC   = (V_cut/split_CA - V_OACAb - V_OACCt - V_O1OACt - V_O2OCAb) * ratio_AB_B

        # Also compute the *Cb* versions for your debug comparison
        V_OACCb  = tet(O,  A,  C,  Cb)
        V_O1OACb = tet(O1, O,  A,  Cb)

        V_AAbBC = tet(A, Ab, B, C)

        # STRICT between test so Ab==B (endpoint) is NOT "between"
        Ab_between_CB = on_minor_arc_strict(thA, thC, thB)

        if not Ab_between_CB:
            Vpatch = abs(V_AAbB - V_AAbC) - V_AAbBC
        else:
            Vpatch = (V_AAbB + V_AAbC) + V_AAbBC

        if triangle_id in (1,5): 
            # angles requested
            oBt = otheta(Bt)
            oAb = otheta(Ab)
            oCb = otheta(Cb)
            oC  = otheta(C)
            oA  = otheta(A)

            print("inside2-ID", triangle_id, "Vpatch?", Vpatch, "Ab_between_CB", Ab_between_CB)
            print("oBt", oBt, "oAb", oAb, "oCb", oCb, "oC", oC, "oA", oA, "oB", otheta(B))
            print("A", A, "B", B, "C", C, "Ab", Ab, "Ct", Ct)
            print("V_AAbB", V_AAbB, "V_AAbC", V_AAbC, "V_AAbBC", V_AAbBC)

            # Your requested debug comparison using Cb-terms:
            print((V_cut/split_CA - V_OACAb - V_OACCb - V_O1OACb - V_O2OCAb) * ratio_AB_B)
            print("V_cut/split_CA", V_cut/split_CA, "V_OACAb", V_OACAb, "V_OACCb", V_OACCb, "V_O1OACb", V_O1OACb, "V_O2OCAb", V_O2OCAb)
            print("split_CA", split_CA, "ratio(B|A)=", ratio_AB_B)

        return Vpatch

    # ======================= default (unchanged) =======================
    dth  = (thC - thA) % (2*math.pi)
    D    = find_rotated_point(B, rB, -((math.atan2(C[1],C[0]) - math.atan2(A[1],A[0]))))
    if dth < 1e-8:
        return 0.0
    if dth >= math.pi:
        dth = 2.0*math.pi - dth

    split = 2.0*math.pi / (dth + 1e-12)
    V_OABC  = abs(np.linalg.det(np.vstack([A-O, B-O, C-O]))) / 6.0
    V_OABD  = abs(np.linalg.det(np.vstack([B-O, D-O, A-O]))) / 6.0
    V_O1CAO = abs(np.linalg.det(np.vstack([C-O1, A-O1, O-O1]))) / 6.0
    V_O2BOD = abs(np.linalg.det(np.vstack([B-O2, D-O2, O-O2]))) / 6.0

    if abs(rB - rC) < 1e-7:
        V_swept = 0.0
    else:
        V_swept, _mode = swept_segment_volume_AB(COEFFS, C[2], B[2])

    Vpatch = (V_cut/split - V_OABC - V_OABD - V_O1CAO - V_O2BOD) * (rA**k/(rA**k + rB**k))

    if triangle_id in (1, 5):
        print("inside2_ID", triangle_id, C[2], B[2], "COEFFS 2x form", COEFFS)
        print(f"  patch id={triangle_id}: dth={dth:.6e}, split={split:.6e}, V_t={V_t:.6e}, V_swept={V_swept:.6e}, Vpatch={Vpatch:.6e}")
        print("V_OABC", V_OABC, "V_OABD", V_OABD, "V_O1CAO", V_O1CAO, "V_O2BOD", V_O2BOD)
        print("ratio", (rA**k)/(rA**k + rB**k))
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
        V_OABC = abs(np.linalg.det(np.vstack([A - O, B - O, C - O]))) / 6.0
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6.0
        if abs(rB - rC) < 1e-07:
            V_swept = 0.0
        else:
            V_swept, _mode = swept_segment_volume_AB(COEFFS, C[2], B[2])
        Vpatch = (V_swept - (V_t - V_OABC * split - V_OABD * split - tri_BOD * split * h_full / 3.0 * 0.5 - tri_COA * split * h_full / 3.0 * 0.5)) / split * (rA ** k / (rA ** k + rB ** k))
        return Vpatch
    if abs(C[2] - B[2]) < EPS:
        dth = (thC - thB) % (2 * np.pi)
        if dth < 1e-08:
            return 0.0
        if dth >= np.pi:
            dth = 2 * np.pi - dth
        D = find_rotated_point(A, rA, -d_CB)
        V_OABC = abs(np.linalg.det(np.vstack([A - O, B - O, C - O]))) / 6.0
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6.0
        tri_COB = 0.5 * rB ** 2 * np.sin(dth)
        tri_DOA = 0.5 * rA ** 2 * np.sin(dth)
        split = 2 * np.pi / (dth + 1e-12)
        if abs(rB - rA) < 1e-07:
            V_swept = 0.0
        else:
            V_swept, _mode = swept_segment_volume_AB(COEFFS, A[2], C[2])
        Vpatch = (V_swept - (V_t - V_OABC * split - V_OABD * split - tri_COB * split * h_full / 6.0 - tri_DOA * split * h_full / 6.0)) / split * (rB ** k / (rB ** k + rA ** k))
        return Vpatch
    return 0.0

def curved_tet_volume(A, B, R_curve, C, D, N=400):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    D = np.asarray(D, float)
    thA = math.atan2(A[1], A[0])
    thB = math.atan2(B[1], B[0])
    dth = (thB - thA + 2 * np.pi) % (2 * np.pi)
    if dth > np.pi:
        dth -= 2 * np.pi
    thetas = thA + np.linspace(0.0, dth, max(8, int(N)))
    zs = np.linspace(A[2], B[2], len(thetas))
    arc = np.stack([R_curve * np.cos(thetas), R_curve * np.sin(thetas), zs], axis=1)
    vol = 0.0
    for i in range(len(arc) - 1):
        P0, P1 = (arc[i], arc[i + 1])
        v1, v2, v3 = (P1 - P0, C - P0, D - P0)
        vol += abs(np.dot(np.cross(v2, v3), v1)) / 6.0
    return vol

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
    import numpy as np, math

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
            if idx is not None:
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
    import math
    import numpy as np
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
    return np.array([r_B * np.cos(th), r_B * np.sin(th), B[2]], float)

def frustum_volume(rA, rB, h):
    return np.pi * h / 3.0 * (rA ** 2 + rA * rB + rB ** 2)

def iso_z_project_axisymmetric(P0, coeffs):
    """
    Keep z and place (x,y) on the axisymmetric quadric at that z.
    Fast exact path for the paraboloid x^2 + y^2 - z = 0.
    """
    A, B, C, D, E, F, G, H, I, J = map(float, coeffs)
    if abs(A - 1.0) < 1e-12 and abs(B - 1.0) < 1e-12 and (abs(C) < 1e-12) and (abs(D) < 1e-12) and (abs(E) < 1e-12) and (abs(F) < 1e-12) and (abs(G) < 1e-12) and (abs(H) < 1e-12) and (abs(J) < 1e-12) and (abs(I + 0.5) < 1e-12):
        x, y, z = map(float, P0)
        r = math.sqrt(max(z, 0.0))
        th = math.atan2(y, x)
        return np.array([r * math.cos(th), r * math.sin(th), z], float)
    x0, U, A3p, bp, cp = _transform_to_axis_frame(coeffs)
    if x0 is None:
        return project_to_quadric_normal_step(P0, coeffs)
    y = U.T @ (np.asarray(P0, float) - x0)
    z = y[2]
    Axy, Axz, Ayz = (A3p[0, 1], A3p[0, 2], A3p[1, 2])
    if any((abs(v) > 1e-10 * max(1.0, np.linalg.norm(A3p)) for v in (Axy, Axz, Ayz, bp[0], bp[1]))):
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
    coeffs = tuple(0.0 if (i < 9 and abs(float(c)) < thr) else float(c) for i, c in enumerate(coeffs))
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
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

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
    if idx == 105: print('+++++ Wedge volume rotation called with idx=',idx,'Ceffs:', Coeffs, 'A:', A, 'B:', B, 'C:', C, 'forced_orientation:', forced_orientation)
    # if idx == 50: print('****** Wedge volume rotation called with idx=51,Ceffs:', Coeffs, 'A:', A, 'B:', B, 'C:', C, 'forced_orientation:', forced_orientation)
    A3, b, _ = _parse_quadric_coeffs(Coeffs)
    # if idx == 50: print('+++++ Wedge volume rotation A3:', A3, 'b:', b)
    # if idx == 51: print('***** Wedge volume rotation A3:', A3, 'b:', b)
    ok, axis_dir = _detect_axis_of_revolution(A3)
    if not ok:
        return (None, 0.0)
    axis_point = _quadric_center(A3, b)
    
    frame = _axis_frame(axis_dir)
    # if idx == 50: print('idx:', idx,'######## Wedge volume rotation axis_point:', axis_point, 'axis_dir:', axis_dir,"frame:", frame)
    # if idx == 51: print('idx:', idx,'******* Wedge volume rotation axis_point:', axis_point, 'axis_dir:', axis_dir,"frame:", frame)
 

    def to_axis(P):
        u, v, k = frame
        return np.array([np.dot(P - axis_point, u), np.dot(P - axis_point, v), np.dot(P - axis_point, k)], float)
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
    k = 1
    orientation = forced_orientation if forced_orientation in ('inward', 'outward', 'between') else 'outward'
    is_between = orientation == 'between'
    if abs(Cp[2] - Ap[2]) < EPS and abs(Bp[2] - Ap[2]) < EPS:
        return (0, 0.0)
    Ep = extend_arc_point_to_z(Ap, rA, Bp, rB, Cp, rC, idx)
    Fp, rF, Gp = extend_C_to_AB(Ap, rA, Bp, rB, Cp, rC, idx)

    if idx == 84 and DEBUG_PRINT:
        print('idx:', idx, 'Ap:', Ap, 'Bp:', Bp, 'Cp:', Cp, 'Ep:', Ep)
    if orientation == 'inward':
        V_ABCE = abs(np.linalg.det(np.vstack([Cp - Ep, Ap - Ep, Bp - Ep]))) / 6.0
        V_1=cone_volume_patch_inside(Ap, Ep, Cp, 0.0, rC, k, idx)
        V_2=cone_volume_patch_inside(Ep, Bp, Cp, 0.0, rC, k, idx)
        V = (V_1 + V_2)

        if rF<rC: V_ABCE = -V_ABCE 
        else: 
            V_beforeCross = cone_volume_patch_inside(Ap, Gp, Cp, 0.0, rC, k, idx) + cone_volume_patch_inside(Gp, Bp, Cp, 0.0, rC, k, idx)
            V_afterCross = cone_volume_patch_inside(Ap, Ep, Gp, 0.0, rC, k, idx) + cone_volume_patch_inside(Ep, Bp, Gp, 0.0, rC, k, idx)
            V = V_afterCross + V_beforeCross 
            V_ABCE = min(abs(np.linalg.det(np.vstack([Gp - Ep, Ap - Ep, Bp - Ep]))) / 6.0,
                     curved_tet_volume(Ep, Gp, rC, Ap, Bp))
            
        #V = cone_volume_patch_inside(Ap, Ep, Cp, 0.0, rC, k, idx) + cone_volume_patch_inside(Ep, Bp, Cp, 0.0, rC, k, idx)
        #V_ABCE_curved = curved_tet_volume(Ep, Cp, rC, Ap, Bp)
        if abs(Cp[2] - Ap[2]) < EPS or abs(Cp[2] - Bp[2]) < EPS:
            V_ABCE_curved = 0.0
        V = V + V_ABCE#min(V_ABCE_curved, V_ABCE)
        #if idx == 51: print('Wedge volume rotation idx:###', idx, 'V:', V)
        return (5, -V)
    if abs(Cp[2] - Ap[2]) < EPS and (not is_between):
        dth = (thC - thA) % (2 * np.pi)
        if dth < 1e-08:
            return (0, 0.0)
        if dth >= np.pi:
            dth = 2 * np.pi - dth
        r_plane = 0.5 * (rA + rC)
        Vpatch = cone_volume_patch(Ap, Bp, Cp, 0.0, r_plane, k, idx)
        
        return (1, Vpatch)
    if abs(Cp[2] - Bp[2]) < EPS and (not is_between):
        dth = (thC - thB) % (2 * np.pi)
        if dth < 1e-08:
            return (0, 0.0)
        if dth >= np.pi:
            dth = 2 * np.pi - dth
        r_plane = 0.5 * (rB + rC)
        Vpatch = cone_volume_patch(Ap, Bp, Cp, 0.0, r_plane, k, idx)
        

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
        V = (V1 * rC ** k / (rA ** k + rC ** k) + V2 * rC ** k / (rB ** k + rC ** k)) / split
        V += abs(np.linalg.det(np.vstack([Cp - Ep, Ap - Ep, Bp - Ep]))) / 6.0
        return (4, V)
    V1 = cone_volume_patch(Ap, Ep, Cp, 0.0, rC, k, idx)
    V2 = cone_volume_patch(Ep, Bp, Cp, 0.0, rC, k, idx)
    V3 = abs(np.linalg.det(np.vstack([Ap - Ep, Bp - Ep, Cp - Ep]))) / 6.0


    if idx == 51: 
        print("Ap:", Ap, "Ep:", Ep, "Cp:", Cp, "rC:", rC, "k:", k, "idx:", idx)
        print("Ep:", Ep, "Bp:", Bp, "Cp:", Cp, "rC:", rC, "k:", k, "idx:", idx)        
        print('Wedge volume rotation idx:', idx, 'V1:', V1, 'V2:', V2, 'V3:', V3)
    
    return (3, V1 + V2 + V3)

def worker(args):
    idx, tri, points = args
    A, B, C = points[tri]
    k1_A, k2_A = principal_curvatures_at(A, COEFFS, project=True, observer_outside=OBSERVER_OUTSIDE)
    k1_B, k2_B = principal_curvatures_at(B, COEFFS, project=True, observer_outside=OBSERVER_OUTSIDE)
    k1_C, k2_C = principal_curvatures_at(C, COEFFS, project=True, observer_outside=OBSERVER_OUTSIDE)
    label = principal_curvatures_rule_label(k2_A, k2_B, k2_C)
    case_id, V = wedge_volume_rotation(COEFFS, A, B, C, idx, forced_orientation=label)
    row = dict(triangle_id=idx, Vcorrection=(case_id, np.float64(V)), surface_triangle=True, pointdirection=label, side=label, k1_A=np.float64(k1_A), k2_A=np.float64(k2_A), k1_B=np.float64(k1_B), k2_B=np.float64(k2_B), k1_C=np.float64(k1_C), k2_C=np.float64(k2_C))
    return (row, V)

