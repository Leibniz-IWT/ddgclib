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

def cone_volume_patch(A, B, C, R_unused, r_plane, k, triangle_id=None):
    
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

        #if triangle_id == 51: print(triangle_id,COEFFS,"V_swept",V_swept)

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
    #if idx == 105: print('+++++ Wedge volume rotation called with idx=',idx,'Ceffs:', Coeffs, 'A:', A, 'B:', B, 'C:', C, 'forced_orientation:', forced_orientation)
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
    if idx == 84 and DEBUG_PRINT:
        print('idx:', idx, 'Ap:', Ap, 'Bp:', Bp, 'Cp:', Cp, 'Ep:', Ep)
    if orientation == 'inward':
        V_ABCE = abs(np.linalg.det(np.vstack([Cp - Ep, Ap - Ep, Bp - Ep]))) / 6.0
        V = cone_volume_patch_inside(Ap, Ep, Cp, 0.0, rC, k, idx) + cone_volume_patch_inside(Ep, Bp, Cp, 0.0, rC, k, idx)
        V_ABCE_curved = curved_tet_volume(Ep, Cp, rC, Ap, Bp)
        if abs(Cp[2] - Ap[2]) < EPS or abs(Cp[2] - Bp[2]) < EPS:
            V_ABCE_curved = 0.0
        V = V + min(V_ABCE_curved, V_ABCE)
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


    # if idx == 51: 
    #     print("Ap:", Ap, "Ep:", Ep, "Cp:", Cp, "rC:", rC, "k:", k, "idx:", idx)
    #     print("Ep:", Ep, "Bp:", Bp, "Cp:", Cp, "rC:", rC, "k:", k, "idx:", idx)        
    #     print('Wedge volume rotation idx:', idx, 'V1:', V1, 'V2:', V2, 'V3:', V3)
    
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

