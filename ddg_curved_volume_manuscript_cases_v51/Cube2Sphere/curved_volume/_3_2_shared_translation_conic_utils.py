#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared_conic_utils.py
Shared helper functions for conic and extrusion geometry.

Design notes
------------
- This module is **dependency-injected**: project-specific functions from your
  codebase (e.g. conic_at_z, length_conic_segment, volume_ABApBp_general_conic,
  g_val, flat_plane_zero_volume) are expected to be assigned into this module
  by the caller *after* import, e.g.:
      import shared_conic_utils as scu
      from _volume import conic_at_z, length_conic_segment, volume_ABApBp_general_conic, g_val
      from flat_plane_guard import flat_plane_zero_volume
      scu.conic_at_z = conic_at_z
      scu.length_conic_segment = length_conic_segment
      scu.volume_ABApBp_general_conic = volume_ABApBp_general_conic
      scu.g_val = g_val
      scu.flat_plane_zero_volume = flat_plane_zero_volume
- Fast-path classification flags (IS_PARABOLA/IS_CIRCLE/IS_ELLIPSE/IS_HYPERB)
  and PARAMS are also set by the caller for the specific COEFFS used:
      scu.KIND, scu.PARAMS = scu.classify_extruded_xy_conic(COEFFS)
      scu.IS_PARABOLA = (scu.KIND == "parabola_y_eq_x2")
      scu.IS_CIRCLE   = (scu.KIND == "circle")
      scu.IS_ELLIPSE  = (scu.KIND == "ellipse")
      scu.IS_HYPERB   = (scu.KIND == "hyperbola")
"""

from __future__ import annotations
import math
import numpy as np
from numpy.polynomial.polynomial import Polynomial as Poly, polyroots
from typing import Dict, Tuple, Any

# --------------- Injection points (set by caller) ---------------
# These will be set from your project at runtime. We keep them as
# module attributes to avoid importing heavy local modules here.
conic_at_z = None                     # fn: (coeffs, z) -> (a,b,c2,d,e,f)
length_conic_segment = None           # fn: (conic, (x1,y1), (x2,y2)) -> float
volume_ABApBp_general_conic = None    # fn: (coeffs, A, B, A_, B_) -> (V, ...)
g_val = None                          # fn: (conic tuple, x, y) -> g(x,y)
flat_plane_zero_volume = None         # fn: (A,B,C,coeffs) -> 0.0 if planar, else nonzero

# Optional logging knob (can be set by caller)
PRINT_PLANAR_SKIPS = False

# --------------- Classification state (set by caller) ---------------
KIND: str = "general"
PARAMS: Dict[str, Any] = {}
IS_PARABOLA = False
IS_CIRCLE   = False
IS_ELLIPSE  = False
IS_HYPERB   = False

# --------------- Numerics & caches ---------------
# 6-point Gauss–Legendre nodes/weights
_GL6_X = np.array([0.2386191860831969,
                   0.6612093864662645,
                   0.9324695142031521], float)
_GL6_W = np.array([0.4679139345726910,
                   0.3607615730481386,
                   0.1713244923791704], float)

# projection cache for parabola/circle projections
_PROJ_CACHE = {}

# --------------- Small utilities ---------------
def _gauss6_integrate(f, L: float) -> float:
    """Integrate f(t) on [0, L] using 6-point Gauss–Legendre (symmetric form)."""
    if L <= 0.0:
        return 0.0
    s = 0.0
    for xi, wi in zip(_GL6_X, _GL6_W):
        t1 = 0.5*L*(1 - xi)
        t2 = 0.5*L*(1 + xi)
        s += wi*(f(t1) + f(t2))
    return 0.5*L*s

def _proj_key(x0, y0): return (round(float(x0), 12), round(float(y0), 12))

def _wrap_pi(dth: float) -> float:
    """Wrap angle to (-π, π]."""
    while dth >  math.pi: dth -= 2*math.pi
    while dth <= -math.pi: dth += 2*math.pi
    return dth

def _to_norm_coords(P, params) -> Tuple[float, float]:
    """
    Map physical point P=(x,y) on ellipse/hyperbola to normalized (u,v) so that:
      ellipse: u^2 + v^2 = 1,   hyperbola: u^2 - v^2 = 1.
    """
    x, y = float(P[0]), float(P[1])
    xc, yc = params["xc"], params["yc"]
    a, b = params["a"], params["b"]
    V = params["eigvecs"]  # columns: principal axes
    X = np.array([x - xc, y - yc], float)
    uv = V.T @ X
    return float(uv[0] / a), float(uv[1] / b)

def _to_norm_coords_for_between(X):
    """
    Return a monotone angle-like parameter for between-arc tests.
    - Circle: true polar angle around (xc, yc)
    - Ellipse: angle in normalized (u,v) coordinates
    """
    if IS_CIRCLE:
        xc, yc = PARAMS["xc"], PARAMS["yc"]
        return math.atan2(X[1] - yc, X[0] - xc)
    if IS_ELLIPSE:
        u, v = _to_norm_coords(X, PARAMS)
        return math.atan2(v, u)
    return None

# --------- local detector for axis-aligned, z-independent parabola ----------
def _is_vertical_parabolic_cylinder_coeffs(coeffs, tol=1e-10):
    """
    Detects a z-independent, axis-aligned parabolic cylinder of the form:
        y = a x^2 + b x + c
    given implicit coeffs (A,B,C,D,E,F,G,H,J,K) for:
        A x^2 + B y^2 + C z^2 + D xy + E yz + F xz + G x + H y + J z + K = 0
    Conditions:
      - z-independent: C=E=F=J≈0
      - axis-aligned parabola (no y^2, no xy): B=0, D=0
      - linear y present: |H|>0
      - quadratic in x present: |A|>0
    """
    A,B,C,D,E,F,G,H,J,K = map(float, coeffs)
    if any(abs(v) > tol for v in (C,E,F,J)): return False
    if abs(B) > tol or abs(D) > tol: return False
    if abs(H) <= tol or abs(A) <= tol: return False
    return True

# --------------- Classification ---------------
def classify_extruded_xy_conic(coeffs, tol=1e-12):
    trl=5e-5 # for parabola y - x^2 = 0, we use a very tight tolerance due to fitted data
    """
    Detect z-independent conic and extract parameters.
    Returns (kind, params) where kind ∈ {'parabola_y_eq_x2','circle','ellipse','hyperbola','general'}.
    For ellipse/hyperbola, params include center, eigvals/vecs, a,b and area scale a*b.
    """
    #print(f"Classifying conic with coeffs: {coeffs}")
    A,B,C,D,E,F,G,H,J,K = map(float, coeffs)

    # must be z-independent in this pipeline
    if any(abs(v) > tol for v in (C,E,F,J)):
        return "general", {}

    # Special parabola y - x^2 = 0
    if abs(A + 1.0) < trl and abs(B) < trl and abs(D) < trl and abs(G) < trl and abs(H - 1.0) < trl and abs(K) < trl:
        return "parabola_y_eq_x2", {}

    # Quadratic form in (x,y): x^T Q x + p·x + K = 0
    Q = np.array([[A, D/2.0],
                  [D/2.0, B]], float)
    p = np.array([G, H], float)

    # Center: 2Q c + p = 0  (parabola -> singular Q -> goes to "general")
    M = 2.0 * Q
    try:
        ctr = -np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        return "general", {}
    xc, yc = float(ctr[0]), float(ctr[1])
    Kp = K + ctr @ (Q @ ctr) + p @ ctr

    # Eigen-decompose Q
    eigvals, eigvecs = np.linalg.eigh(Q)
    l1, l2 = float(eigvals[0]), float(eigvals[1])
    detQ = l1 * l2

    # ellipse (includes circle)
    if l1 > tol and l2 > tol and Kp < 0:
        a = math.sqrt(-Kp / l1)  # semiaxes in principal frame
        b = math.sqrt(-Kp / l2)
        if abs(l1 - l2) < tol and abs(a - b) < tol:
            return "circle", {"xc": xc, "yc": yc, "R": a}
        return "ellipse", {"xc": xc, "yc": yc, "eigvals": eigvals, "eigvecs": eigvecs,
                           "a": a, "b": b, "area_scale": a*b}

    # hyperbola
    if detQ < -tol and Kp < 0:
        if l1 < l2:
            lpos, lneg = l2, l1
            V = eigvecs[:, [1,0]]
        else:
            lpos, lneg = l1, l2
            V = eigvecs.copy()
        a = math.sqrt(-Kp / lpos)
        b = math.sqrt( Kp / lneg)
        return "hyperbola", {"xc": xc, "yc": yc, "eigvals": np.array([lpos, lneg], float),
                             "eigvecs": V, "a": a, "b": b, "area_scale": a*b}

    return "general", {}

# --------------- Geometry basics ---------------
def tet_volume(pts):
    p0, p1, p2, p3 = pts
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

def tetra_volume_signed(P0, P1, P2, P3):
    P0 = np.asarray(P0,float); P1 = np.asarray(P1,float)
    P2 = np.asarray(P2,float); P3 = np.asarray(P3,float)
    return float(np.dot(np.cross(P1-P0, P2-P0), P3-P0) / 6.0)

def point_on_AB_at_z(A, B, z_target, clamp_to_segment=True, eps=1e-12):
    A = np.asarray(A, float); B = np.asarray(B, float)
    dz = B[2] - A[2]
    if abs(dz) < eps:
        if abs(z_target - A[2]) < eps: return A.copy()
        return A.copy() if clamp_to_segment else (_ for _ in ()).throw(ValueError("AB ∥ z=const"))
    t = (z_target - A[2]) / dz
    if clamp_to_segment: t = max(0.0, min(1.0, t))
    return A + t * (B - A)

def coincident_xy(P, Q, eps=1e-12):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    return np.hypot(P[0]-Q[0], P[1]-Q[1]) < eps

# --------------- Projections ---------------
def project_to_parabola_xy(x0, y0):
    # Orthogonal projection to y=x^2 via Newton on: 2x^3 + (1 - 2y0)x - x0 = 0
    k = _proj_key(x0, y0)
    hit = _PROJ_CACHE.get(k)
    if hit is not None: return hit
    x = float(x0)
    if y0 > 0:
        g = math.copysign(math.sqrt(y0), x0) if x0 != 0.0 else math.sqrt(y0)
        x = 0.5*(x + g)
    for _ in range(8):
        f  = 2.0*x*x*x + (1.0 - 2.0*y0)*x - x0
        df = 6.0*x*x   + (1.0 - 2.0*y0)
        if df == 0.0: break
        x_new = x - f/df
        if abs(x_new - x) < 1e-12: x = x_new; break
        x = x_new
    y = x*x
    _PROJ_CACHE[k] = (x, y)
    return x, y

def project_to_circle_xy(x0, y0):
    xc, yc, R = PARAMS["xc"], PARAMS["yc"], PARAMS["R"]
    dx, dy = (x0 - xc), (y0 - yc)
    r = math.hypot(dx, dy)
    if r == 0.0: return xc + R, yc
    s = R / r
    return xc + s*dx, yc + s*dy

def project_xy_to_conic_3d(coeffs, P, z_level):
    x0, y0 = float(P[0]), float(P[1])
    if IS_PARABOLA:
        if abs(y0 - x0*x0) < 1e-10:
            return np.array([x0, y0, float(z_level)], float)
        x, y = project_to_parabola_xy(x0, y0)
        return np.array([x, y, float(z_level)], float)
    if IS_CIRCLE:
        xc, yc, R = PARAMS["xc"], PARAMS["yc"], PARAMS["R"]
        if abs((x0 - xc)**2 + (y0 - yc)**2 - R*R) < 1e-10:
            return np.array([x0, y0, float(z_level)], float)
        x, y = project_to_circle_xy(x0, y0)
        return np.array([x, y, float(z_level)], float)

    # generic projection for ellipse/hyperbola/others
    if conic_at_z is None or g_val is None:
        raise RuntimeError("shared_conic_utils.project_xy_to_conic_3d requires injected conic_at_z and g_val.")
    conic = conic_at_z(coeffs, float(z_level))
    a,b,c2,d,e,f = conic
    M11 = Poly([2.0, 2.0*a]); M22 = Poly([2.0, 2.0*c2]); M12 = Poly([0.0, b])
    detM = M11*M22 - M12*M12
    rhs1 = Poly([2.0*x0, -d]); rhs2 = Poly([2.0*y0, -e])
    Nx = rhs1*M22 - rhs2*M12; Ny = M11*rhs2 - M12*rhs1
    g_num = (a*(Nx*Nx) + b*(Nx*Ny) + c2*(Ny*Ny)
             + d*(Nx*detM) + e*(Ny*detM) + f*(detM*detM))
    coeffs_inc = g_num.coef
    while len(coeffs_inc) > 1 and abs(coeffs_inc[-1]) < 1e-14: coeffs_inc = coeffs_inc[:-1]
    roots = polyroots(coeffs_inc)
    best = None
    for lam in roots:
        if abs(lam.imag) > 1e-10: continue
        lam = lam.real; det_val = detM(lam)
        if abs(det_val) < 1e-12: continue
        x = Nx(lam)/det_val; y = Ny(lam)/det_val
        if abs(g_val(conic, x, y)) > 1e-6: continue
        d2 = (x - x0)**2 + (y - y0)**2
        if (best is None) or (d2 < best[0]): best = (d2, float(x), float(y))
    if best is None:
        raise RuntimeError("No valid projection root found.")
    return np.array([best[1], best[2], float(z_level)], float)

# --------------- Arc lengths (for weights) ---------------
def arc_len_parabola_between_oncurve(P, Q):
    # y=x^2; exact primitive: S(x) = 1/4 (asinh(2x) + 2x*sqrt(1+4x^2))
    def S(x): return 0.25*(math.asinh(2.0*x) + 2.0*x*math.sqrt(1.0 + 4.0*x*x))
    return abs(S(P[0]) - S(Q[0]))

def arc_len_circle_between_oncurve(P, Q, params):
    xc, yc, R = params["xc"], params["yc"], params["R"]
    thP = math.atan2(P[1]-yc, P[0]-xc)
    thQ = math.atan2(Q[1]-yc, Q[0]-xc)
    dth = abs(_wrap_pi(thQ - thP))
    return R * dth

def arc_len_ellipse_between(P, Q, params):
    a, b = params["a"], params["b"]
    uP, vP = _to_norm_coords(P, params)
    uQ, vQ = _to_norm_coords(Q, params)
    thP = math.atan2(vP, uP)
    thQ = math.atan2(vQ, uQ)
    d = _wrap_pi(thQ - thP)
    sgn = 1.0 if d >= 0 else -1.0
    L = abs(d)

    # ds = sqrt( (a sin θ)^2 + (b cos θ)^2 ) dθ
    def f(t):
        th = thP + sgn*t
        return math.hypot(a*math.sin(th), b*math.cos(th))
    return _gauss6_integrate(f, L)

def arc_len_hyperbola_between(P, Q, params):
    a, b = params["a"], params["b"]
    uP, vP = _to_norm_coords(P, params)
    uQ, vQ = _to_norm_coords(Q, params)

    # Ensure same branch (sign(u) must match)
    sP, sQ = math.copysign(1.0, uP), math.copysign(1.0, uQ)
    if sP != sQ:
        return None

    t1 = math.asinh(vP)
    t2 = math.asinh(vQ)
    d  = abs(t2 - t1)

    # ds = sqrt( (a sinh t)^2 + (b cosh t)^2 ) dt
    def f(t):
        T = min(t1, t2) + t
        return math.hypot(a*math.sinh(T), b*math.cosh(T))
    return _gauss6_integrate(f, d)

# --------------- Segment area magnitudes ---------------
def parabola_segment_area_mag(P, Q):
    # endpoints on y=x^2
    x1, x2 = float(P[0]), float(Q[0])
    return (abs(x2 - x1)**3) / 6.0

def circle_segment_area_mag(P, Q, params):
    xc, yc, R = params["xc"], params["yc"], params["R"]
    thP = math.atan2(P[1]-yc, P[0]-xc)
    thQ = math.atan2(Q[1]-yc, Q[0]-xc)
    dth = abs(_wrap_pi(thQ - thP))         # minor arc
    return 0.5 * R*R * (dth - math.sin(dth))

def ellipse_segment_area_mag(P, Q, params):
    # Map to unit circle and use 0.5*(Δθ - sin Δθ), then scale by a*b
    a, b, S = params["a"], params["b"], params["area_scale"]
    uP, vP = _to_norm_coords(P, params)
    uQ, vQ = _to_norm_coords(Q, params)
    thP = math.atan2(vP, uP)
    thQ = math.atan2(vQ, uQ)
    dth = _wrap_pi(thQ - thP)    # signed in (-π, π]
    A_unit = 0.5 * (abs(dth) - math.sin(dth))   # magnitude
    return S * A_unit

def hyperbola_segment_area_mag(P, Q, params):
    # Unit hyperbola coords (u,v): sector area over arc + reversed chord
    uP, vP = _to_norm_coords(P, params)
    uQ, vQ = _to_norm_coords(Q, params)
    sP, sQ = math.copysign(1.0, uP), math.copysign(1.0, uQ)
    if sP != sQ:
        # different branches -> ambiguous; signal None to trigger fallback
        return None
    t1 = math.asinh(vP)
    t2 = math.asinh(vQ)
    s  = sP
    arc_int   = s * (t2 - t1)         # ∫_arc (u dv - v du)
    chord_int = (uQ*vP - uP*vQ)       # reversed chord (Q->P)
    A_unit = 0.5 * abs(arc_int + chord_int)
    return PARAMS.get("area_scale", 1.0) * A_unit

# --------------- Between test ---------------
def arc_between_on_conic(conic, P, Q, R, tol_param=1e-12):
    if IS_PARABOLA:
        xP, xQ, xR = P[0], Q[0], R[0]
        if abs(xR - xP) <= tol_param: return False
        if xR > xP: return (xP + tol_param) < xQ < (xR - tol_param)
        else:       return (xR + tol_param) < xQ < (xP - tol_param)

    if IS_CIRCLE or IS_ELLIPSE:
        thP = _to_norm_coords_for_between(P)
        thQ = _to_norm_coords_for_between(Q)
        thR = _to_norm_coords_for_between(R)
        dPR = _wrap_pi(thR - thP)
        dPQ = _wrap_pi(thQ - thP)
        eps = 1e-14
        if dPR > 0:  return (0.0 + eps) < dPQ < (dPR - eps)
        else:        return (0.0 - eps) > dPQ > (dPR + eps)

    if IS_HYPERB:
        uP, vP = _to_norm_coords(P, PARAMS)
        uQ, vQ = _to_norm_coords(Q, PARAMS)
        uR, vR = _to_norm_coords(R, PARAMS)
        sP, sR = math.copysign(1.0, uP), math.copysign(1.0, uR)
        if sP != math.copysign(1.0, uQ) or sP != sR:  # different branches
            return False
        tP = math.asinh(vP); tQ = math.asinh(vQ); tR = math.asinh(vR)
        if abs(tR - tP) <= tol_param: return False
        if tR > tP: return (tP + tol_param) < tQ < (tR - tol_param)
        else:       return (tR + tol_param) < tQ < (tP - tol_param)

    return False

# --------------- Length & Volume wrappers ---------------
def length_or_zero(conic, P, Q):
    if coincident_xy(P, Q): return 0.0
    if IS_PARABOLA:
        return arc_len_parabola_between_oncurve(P, Q)
    if IS_CIRCLE:
        return arc_len_circle_between_oncurve(P, Q, PARAMS)
    if IS_ELLIPSE:
        return arc_len_ellipse_between(P, Q, PARAMS)
    if IS_HYPERB:
        L = arc_len_hyperbola_between(P, Q, PARAMS)
        if L is not None: return L
    # fallback (rare)
    if length_conic_segment is None:
        raise RuntimeError("shared_conic_utils.length_or_zero needs injected length_conic_segment for generic fallback.")
    try:
        return length_conic_segment(conic, (P[0],P[1]), (Q[0],Q[1]))
    except Exception:
        return 0.0

def volume_or_zero(coeffs, A, B, A_, B_):
    if coincident_xy(A, B) and coincident_xy(A_, B_): return 0.0
    dz = abs(float(A[2]) - float(A_[2]))
    if dz == 0.0: return 0.0

    # === FIX: correct world-space scaling for axis-aligned, z-independent parabola (y = a x^2 + b x + c)
    if _is_vertical_parabolic_cylinder_coeffs(coeffs):
        Acoef,Bcoef,Ccoef,Dcoef,Ecoef,Fcoef,Gcoef,Hcoef,Jcoef,Kcoef = map(float, coeffs)
        a = -Acoef / Hcoef  # from A x^2 + H y + G x + K = 0  =>  y = (-A/H) x^2 + ...
        dx = abs(float(B[0]) - float(A[0]))  # world-space Δx of the chord
        area = (abs(a) * dx**3) / 6.0
        return dz * area
    # === END FIX

    if IS_PARABOLA:
        return dz * parabola_segment_area_mag(A, B)
    if IS_CIRCLE:
        return dz * circle_segment_area_mag(A, B, PARAMS)
    if IS_ELLIPSE:
        return dz * ellipse_segment_area_mag(A, B, PARAMS)
    if IS_HYPERB:
        Aseg = hyperbola_segment_area_mag(A, B, PARAMS)
        if Aseg is not None:
            return dz * Aseg

    # generic fallback
    if volume_ABApBp_general_conic is None:
        raise RuntimeError("shared_conic_utils.volume_or_zero needs injected volume_ABApBp_general_conic for generic fallback.")
    try:
        V, *_ = volume_ABApBp_general_conic(coeffs, A, B, A_, B_)
        return V
    except Exception:
        return 0.0

def safe_ratio(num, den):
    try:
        num = float(num); den = float(den)
    except Exception:
        return 0.0
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0: return 0.0
    return num/den

# --------------- Patch construction & computation ---------------
def construct_multilevel_points(coeffs, A_in, B_in, C_in, clamp_to_segment=True):
    A_in = np.asarray(A_in, float)
    B_in = np.asarray(B_in, float)
    C_in = np.asarray(C_in, float)

    pts_sorted = sorted([A_in, B_in, C_in], key=lambda P: P[2])
    B = pts_sorted[0]            # min z
    A = pts_sorted[2]            # max z
    C = C_in.copy()

    z_top = float(A[2]); z_bot = float(B[2]); z_mid = float(C[2])
    D = point_on_AB_at_z(A, B, z_mid, clamp_to_segment=clamp_to_segment)

    # only D needs projection; A,B,C already lie on the wall
    D_mid = project_xy_to_conic_3d(coeffs, D, z_mid)

    D_top    = np.array([D_mid[0], D_mid[1], z_top], float)
    C_top    = np.array([C[0],     C[1],     z_top], float)
    B_top    = np.array([B[0],     B[1],     z_top], float)

    A_middle = np.array([A[0],     A[1],     z_mid], float)
    B_middle = np.array([B[0],     B[1],     z_mid], float)
    C_middle = np.array([C[0],     C[1],     z_mid], float)
    D_middle = D_mid

    C_bottom = np.array([C[0],     C[1],     z_bot], float)
    D_bottom = np.array([D_mid[0], D_mid[1], z_bot], float)
    A_bottom = np.array([A[0],     A[1],     z_bot], float)

    return {
        "A_top": A, "B_bottom": B, "C": C, "D": D,
        "C_top": C_top, "D_top": D_top, "B_top": B_top,
        "A_middle": A_middle, "B_middle": B_middle, "C_middle": C_middle, "D_middle": D_middle,
        "C_bottom": C_bottom, "D_bottom": D_bottom, "A_bottom": A_bottom,
        "z_top": z_top, "z_mid": z_mid, "z_bot": z_bot,
    }

def compute_V_patch(coeffs, A_top, C_top, D_top, A_middle, C_middle, D_middle, indx=None):
    A_top   = np.asarray(A_top,float)
    C_top   = np.asarray(C_top,float)
    D_top   = np.asarray(D_top,float)
    A_mid   = np.asarray(A_middle,float)
    C_mid   = np.asarray(C_middle,float)
    D_mid   = np.asarray(D_middle,float)

    if conic_at_z is None:
        raise RuntimeError("shared_conic_utils.compute_V_patch needs injected conic_at_z.")

    z_top = float(A_top[2]); z_mid = float(A_mid[2])
    conic_top = conic_at_z(coeffs, z_top)
    conic_mid = conic_at_z(coeffs, z_mid)

    V_AtCtAmCm = volume_or_zero(coeffs, A_top, C_top, A_mid, C_mid)
    V_AtDtAmDm = volume_or_zero(coeffs, A_top, D_top, A_mid, D_mid)

    L_AtCt = length_or_zero(conic_top, A_top, C_top)
    L_AtDt = length_or_zero(conic_top, A_top, D_top)
    L_AmCm = length_or_zero(conic_mid, A_mid, C_mid)
    L_AmDm = length_or_zero(conic_mid, A_mid, D_mid)

    A_between = arc_between_on_conic(conic_top, C_top[:2], A_top[:2], D_top[:2])
    D_between = arc_between_on_conic(conic_top, A_top[:2], D_top[:2], C_top[:2])
    C_between = arc_between_on_conic(conic_top, A_top[:2], C_top[:2], D_top[:2])

    if A_between:
        L_CtAt = length_or_zero(conic_top, C_top, A_top)
        L_CmAm = length_or_zero(conic_mid, C_mid, A_mid)
        L_DtAt = length_or_zero(conic_top, D_top, A_top)
        L_DmAm = length_or_zero(conic_mid, D_mid, A_mid)

        V_CtCmAtAm = volume_or_zero(coeffs, C_top, A_top, C_mid, A_mid)
        V_AtAmDtDm = volume_or_zero(coeffs, A_top, D_top, A_mid, D_mid)
        V_CtDtCmDm = volume_or_zero(coeffs, C_top, D_top, C_mid, D_mid)

        V_AtCtDtCmDm = tetra_volume_signed(A_top, C_top, D_top, C_mid) \
                     + tetra_volume_signed(A_top, D_top, D_mid, C_mid)

        rC = safe_ratio(L_CmAm, (L_CtAt + L_CmAm))
        rD = safe_ratio(L_DmAm, (L_DtAt + L_DmAm))
        return abs(V_CtDtCmDm) - abs(V_CtCmAtAm)*rC - abs(V_AtAmDtDm)*rD - abs(V_AtCtDtCmDm)

    if D_between:
        V_AtAmDmCm = tetra_volume_signed(A_top, A_mid, D_mid, C_mid)
        rA = safe_ratio(L_AtCt, (L_AtCt + L_AmCm))
        rD = safe_ratio(L_AtDt, (L_AtDt + L_AmDm))
        return abs(V_AtCtAmCm) * rA - abs(V_AtDtAmDm) * rD - abs(V_AtAmDmCm)

    if C_between:
        V_AtAmDmCm = tetra_volume_signed(A_top, A_mid, D_mid, C_mid)
        rA = safe_ratio(L_AtCt, (L_AtCt + L_AmCm))
        rD = safe_ratio(L_AtDt, (L_AtDt + L_AmDm))
        return - abs(V_AtCtAmCm) * rA + abs(V_AtDtAmDm) * rD - abs(V_AtAmDmCm)

    V_AtCmDmAm = tetra_volume_signed(A_top, C_mid, D_mid, A_mid)
    r1 = safe_ratio(L_AtCt, (L_AtCt + L_AmCm))
    r2 = safe_ratio(L_AmDm, (L_AmDm + L_AtDt))
    return abs(V_AtCtAmCm) * r1 - abs(V_AtDtAmDm) * r2 - abs(V_AtCmDmAm)

def compute_V_patch_from_ABC(coeffs, A_in, B_in, C_in, idx=None, eps=1e-12):
    # fast zero on planar faces (caps, internal planes, etc.)
    V0 = None
    if flat_plane_zero_volume is not None:
        V0 = flat_plane_zero_volume(A_in, B_in, C_in, coeffs)
        if V0 == 0.0:
            if PRINT_PLANAR_SKIPS and idx is not None:
                print(f"Triangle {idx} is planar (V0=0.0); skipping.")
            return 0.0

    zvals = [A_in[2], B_in[2], C_in[2]]
    idxs = np.argsort(zvals)
    A_in, B_in, C_in = [A_in, B_in, C_in][idxs[2]], [A_in, B_in, C_in][idxs[0]], [A_in, B_in, C_in][idxs[1]]

    pts = construct_multilevel_points(coeffs, A_in, B_in, C_in, clamp_to_segment=True)

    A_top = pts["A_top"]; B_bottom = pts["B_bottom"]; C = pts["C"]; D = pts["D"]
    C_top = pts["C_top"]; B_top = pts["B_top"]; A_bottom = pts["A_bottom"]
    C_bottom = pts["C_bottom"]; B_middle = pts["B_middle"]; C_middle = pts["C_middle"]
    D_top = pts["D_top"]; D_middle = pts["D_middle"]; D_bottom = pts["D_bottom"]
    A_middle = pts["A_middle"]

    z_top = pts["z_top"]; z_mid = pts["z_mid"]; z_bot = pts["z_bot"]

    if abs(z_mid - z_bot) < eps:
        V = compute_V_patch(coeffs, A_top, C_top, B_top, A_bottom, C_bottom, B_bottom)
        return abs(V)

    if abs(z_mid - z_top) < eps:
        z_bot_explicit = min(float(A_in[2]), float(B_in[2]), float(C_in[2]))
        z_top_explicit = max(float(A_in[2]), float(B_in[2]), float(C_in[2]))
        A_top_explicit = np.asarray(A_in, float).copy()
        C_top_explicit = np.asarray(C_in, float).copy()
        B_top_explicit    = np.array([B_in[0], B_in[1], z_top_explicit], float)
        B_bottom_explicit = np.array([B_in[0], B_in[1], z_bot_explicit], float)
        A_bottom_branch   = project_xy_to_conic_3d(coeffs, A_in, z_bot_explicit)
        C_bottom_branch   = project_xy_to_conic_3d(coeffs, C_in, z_bot_explicit)
        V = compute_V_patch(
            coeffs,
            B_bottom_explicit, C_bottom_branch, A_bottom_branch,
            B_top_explicit,    C_top_explicit,  A_top_explicit
        )
        return abs(V)

    V_patch1 = compute_V_patch(coeffs, A_top, C_top, D_top, A_middle, C_middle, D_middle)
    V_patch2 = compute_V_patch(coeffs, B_bottom, C_bottom, D_bottom, B_middle, C_middle, D_middle, idx)
    V_ABCD   = tetra_volume_signed(A_top, B_bottom, C_middle, D_middle)
    if idx == 239:
        print("Debug info for patch idx:", idx)
        print("A_in:", A_in, "B_in:", B_in, "C_in:", C_in) 
        print("A_top:", A_top, "C_top:", C_top, "D_top:", D_top)
        print("A_middle:", A_middle, "C_middle:", C_middle, "D_middle:", D_middle)
        print("B_bottom:", B_bottom, "C_bottom:", C_bottom, "D_bottom:", D_bottom)
        print(f"V_patch1={V_patch1}, V_patch2={V_patch2}, V_ABCD={V_ABCD}")
        print("coeffs:", coeffs)
    return abs(V_patch1) + abs(V_patch2) + abs(V_ABCD)

# --------------- Labels (optional) ---------------
def classify_side_and_direction(A, B, C, coeffs):
    A = np.asarray(A,float); B = np.asarray(B,float); C = np.asarray(C,float)
    Gtri = (A + B + C) / 3.0
    A3 = np.array([[coeffs[0], coeffs[3], coeffs[4]],
                   [coeffs[3], coeffs[1], coeffs[5]],
                   [coeffs[4], coeffs[5], coeffs[2]]], float)
    b  = np.array([coeffs[6], coeffs[7], coeffs[8]], float)
    grad = 2*(A3 @ Gtri + b)
    axis = np.array([0.0, 0.0, 1.0])
    rvec = Gtri - np.dot(Gtri, axis)*axis
    direction = "outward" if np.dot(grad, rvec) >= 0 else "inward"
    side = "outside"
    return side, direction

# --------------- Public API ---------------
__all__ = [
    # classification & context
    "classify_extruded_xy_conic", "KIND", "PARAMS", "IS_PARABOLA", "IS_CIRCLE", "IS_ELLIPSE", "IS_HYPERB",
    # geometry basics
    "tet_volume", "tetra_volume_signed", "point_on_AB_at_z", "coincident_xy",
    # projections
    "project_to_parabola_xy", "project_to_circle_xy", "project_xy_to_conic_3d",
    # arc lengths & areas
    "arc_len_parabola_between_oncurve", "arc_len_circle_between_oncurve", "arc_len_ellipse_between", "arc_len_hyperbola_between",
    "parabola_segment_area_mag", "circle_segment_area_mag", "ellipse_segment_area_mag", "hyperbola_segment_area_mag",
    # between test
    "arc_between_on_conic",
    # length / volume wrappers
    "length_or_zero", "volume_or_zero", "safe_ratio",
    # patch pipeline
    "construct_multilevel_points", "compute_V_patch", "compute_V_patch_from_ABC",
    # labels
    "classify_side_and_direction",
    # knobs
    "PRINT_PLANAR_SKIPS",
    # injection points (exported so caller can assign them)
    "conic_at_z", "length_conic_segment", "volume_ABApBp_general_conic", "g_val", "flat_plane_zero_volume",
]
