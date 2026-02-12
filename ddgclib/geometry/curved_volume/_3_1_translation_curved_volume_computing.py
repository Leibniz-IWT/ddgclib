#!/usr/bin/env python3
# coding: utf-8

"""
_volume.py — conic/cylinder helpers

Exports:
  - fmt_s
  - conic_at_z, g_val
  - roots_x_of_y, roots_y_of_x, pick_branch
  - interior_point_on_conic
  - integrate_conic_Q_dy_yparam, integrate_conic_Q_dy_xparam, conic_segment_Q_dy
  - straight_Qf_dy_integral, straight_x_dy_integral
  - triangle_flux_z
  - cap_area_arc_plus_chord, cap_area_sector_circle
  - length_conic_segment
  - volume_ABApBp_general_conic
  - _is_vertical_cylinder, _circle_radius_from_conic
"""


import time
import math
import numpy as np
from numpy.polynomial.legendre import leggauss


# small util

def fmt_s(s: float) -> str:
    """Pretty time formatting."""
    return f"{s*1e6:.1f} µs" if s < 1e-3 else (f"{s*1e3:.3f} ms" if s < 1 else f"{s:.6f} s")


# core conic helpers

def conic_at_z(coeffs, zC):
    """
    Reduce 3D quadric to a 2D implicit conic at plane z=zC.
      3D:  Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Jz + K = 0
      2D:  a x^2 + b x y + c2 y^2 + d x + e y + f = 0
    """
    A,B,C,D,E,F,G,H,J,K = map(float, coeffs)
    a  = A
    b  = D
    c2 = B
    d  = G + E*zC
    e  = H + F*zC
    f  = C*zC*zC + J*zC + K
    return (a,b,c2,d,e,f)

def g_val(conic, x, y):
    a,b,c2,d,e,f = conic
    return a*x*x + b*x*y + c2*y*y + d*x + e*y + f

# Vector-field primitives used by Green/Divergence formulas
def Q_f(x, y):     # primitive for curved-side flux pieces (generic conic path)
    return (1.0/3.0)*x**3 - x*y**2

def Q_area(x, y):  # x dy gives area by Green's theorem
    return x


# roots / branch picking

def roots_x_of_y(conic, y):
    a,b,c2,d,e,f = conic
    qa = a
    qb = b*y + d
    qc = c2*y*y + e*y + f
    if abs(qa) < 1e-16:
        if abs(qb) < 1e-16: return []
        return [-qc/qb]
    disc = qb*qb - 4.0*qa*qc
    if disc < -1e-14:
        return []
    s = math.sqrt(max(0.0, disc))
    x1 = (-qb + s)/(2.0*qa)
    x2 = (-qb - s)/(2.0*qa)
    return [x1, x2] if abs(x1 - x2) > 1e-14 else [0.5*(x1 + x2)]

def roots_y_of_x(conic, x):
    a,b,c2,d,e,f = conic
    qa = c2
    qb = b*x + e
    qc = a*x*x + d*x + f
    if abs(qa) < 1e-16:
        if abs(qb) < 1e-16: return []
        return [-qc/qb]
    disc = qb*qb - 4.0*qa*qc
    if disc < -1e-14:
        return []
    s = math.sqrt(max(0.0, disc))
    y1 = (-qb + s)/(2.0*qa)
    y2 = (-qb - s)/(2.0*qa)
    return [y1, y2] if abs(y1 - y2) > 1e-14 else [0.5*(y1 + y2)]

def pick_branch(vals, target):
    """From a list of roots, pick the one closest to 'target'."""
    if not vals:
        return None
    arr = np.array(vals, float)
    return float(arr[np.argmin(np.abs(arr - target))])


# interior point on the same branch

def interior_point_on_conic(conic, P0, P1):
    """
    Attempt to find an interior point M on the same branch between P0 and P1.
    Try x-midline, then y-midline; fallback is arithmetic mid (may be slightly off curve).
    """
    x0,y0 = float(P0[0]), float(P0[1])
    x1,y1 = float(P1[0]), float(P1[1])

    xm = 0.5*(x0 + x1)
    ys = roots_y_of_x(conic, xm)
    if ys:
        y_hint = 0.5*(y0 + y1)
        ym = pick_branch(ys, y_hint)
        if ym is not None and abs(g_val(conic, xm, ym)) < 1e-8:
            return np.array([xm, ym], float)

    ym = 0.5*(y0 + y1)
    xs = roots_x_of_y(conic, ym)
    if xs:
        x_hint = 0.5*(x0 + x1)
        xm = pick_branch(xs, x_hint)
        if xm is not None and abs(g_val(conic, xm, ym)) < 1e-8:
            return np.array([xm, ym], float)

    return np.array([0.5*(x0+x1), 0.5*(y0+y1)], float)


# ∫ Q dy along conic segment

def integrate_conic_Q_dy_yparam(conic, P0, P1, Q, tol=1e-14, max_levels=14, n_gl=32):
    """Preferred integrator when y is monotone along the arc: ∫ Q(x(y),y) dy"""
    y0, y1 = float(P0[1]), float(P1[1])
    if abs(g_val(conic, *P0)) > 1e-6 or abs(g_val(conic, *P1)) > 1e-6:
        raise ValueError("Endpoints are not on the conic.")
    if abs(y1 - y0) < 1e-15:
        return 0.0

    nodes, weights = leggauss(n_gl)

    def x_on_branch(y, x_hint):
        xs = roots_x_of_y(conic, y)
        if not xs: return None
        return pick_branch(xs, x_hint)

    def quad_interval(a, b, x_left, x_right):
        half = 0.5*(b - a)
        center = 0.5*(b + a)
        total = 0.0
        for t, w in zip(nodes, weights):
            y = center + half*t
            # linear hint for x(y) along the branch
            hint = x_left + (x_right - x_left)*((y - a)/(b - a))
            x = x_on_branch(y, hint)
            if x is None:
                raise RuntimeError("Failed to find x(y) on conic.")
            total += w * Q(x, y)
        return total * half

    def adapt(a, b, x_a, x_b, level):
        I1 = quad_interval(a, b, x_a, x_b)
        m  = 0.5*(a + b)
        x_m = x_on_branch(m, 0.5*(x_a + x_b))
        if x_m is None:
            raise RuntimeError("Midpoint branch resolution failed.")
        I2 = quad_interval(a, m, x_a, x_m) + quad_interval(m, b, x_m, x_b)
        if abs(I2 - I1) < tol or level >= max_levels:
            return I2
        return adapt(a, m, x_a, x_m, level+1) + adapt(m, b, x_m, x_b, level+1)

    x_a = pick_branch(roots_x_of_y(conic, y0), P0[0])
    x_b = pick_branch(roots_x_of_y(conic, y1), P1[0])
    if x_a is None or x_b is None:
        raise RuntimeError("Endpoint root resolution (y-param) failed.")
    return adapt(y0, y1, x_a, x_b, 0)

def integrate_conic_Q_dy_xparam(conic, P0, P1, Q, tol=1e-14, max_levels=14, n_gl=32):
    """Fallback integrator when x is monotone: ∫ Q(x,y(x)) y'(x) dx"""
    x0, x1 = float(P0[0]), float(P1[0])
    if abs(g_val(conic, *P0)) > 1e-6 or abs(g_val(conic, *P1)) > 1e-6:
        raise ValueError("Endpoints are not on the conic.")
    if abs(x1 - x0) < 1e-15:
        return 0.0

    nodes, weights = leggauss(n_gl)
    a,b,c2,d,e,f = conic
    def gx(x,y): return 2*a*x + b*y + d
    def gy(x,y): return b*x + 2*c2*y + e

    def y_on_branch(x, y_hint):
        ys = roots_y_of_x(conic, x)
        if not ys: return None
        return pick_branch(ys, y_hint)

    def quad_interval(ax, bx, y_left, y_right):
        half = 0.5*(bx - ax)
        center = 0.5*(bx + ax)
        total = 0.0
        for t, w in zip(nodes, weights):
            x = center + half*t
            hint = y_left + (y_right - y_left)*((x - ax)/(bx - ax))
            y = y_on_branch(x, hint)
            if y is None:
                raise RuntimeError("Failed to find y(x) on conic.")
            total += w * Q(x, y) * (-(gx(x,y)/gy(x,y)))  # dy = y'(x) dx = -(gx/gy) dx
        return total * half

    def adapt(ax, bx, y_a, y_b, level):
        I1 = quad_interval(ax, bx, y_a, y_b)
        m  = 0.5*(ax + bx)
        y_m = y_on_branch(m, 0.5*(y_a + y_b))
        if y_m is None:
            raise RuntimeError("Midpoint branch resolution (x-param) failed.")
        I2 = quad_interval(ax, m, y_a, y_m) + quad_interval(m, bx, y_m, y_b)
        if abs(I2 - I1) < tol or level >= max_levels:
            return I2
        return adapt(ax, m, y_a, y_m, level+1) + adapt(m, bx, y_m, y_b, level+1)

    y_a = pick_branch(roots_y_of_x(conic, x0), P0[1])
    y_b = pick_branch(roots_y_of_x(conic, x1), P1[1])
    if y_a is None or y_b is None:
        raise RuntimeError("Endpoint root resolution (x-param) failed.")
    return adapt(x0, x1, y_a, y_b, 0)

def conic_segment_Q_dy(conic, P0, P1, tol=1e-14, Q=None, n_gl=32):
    """
    Robust ∫ Q dy on the conic from P0->P1.
    Prefer y-parameterization; if Δy≈0, split and use two y-param halves;
    otherwise fall back to x-parameterization.
    """
    if Q is None:
        Q = Q_f
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)
    dy = abs(P1[1] - P0[1])
    dx = abs(P1[0] - P0[0])

    try:
        if dy >= 1e-12:
            return integrate_conic_Q_dy_yparam(conic, P0, P1, Q, tol=tol, n_gl=n_gl)
    except Exception:
        pass

    if dy < 1e-12 and dx >= 1e-12:
        M = interior_point_on_conic(conic, P0, P1)
        if abs(g_val(conic, M[0], M[1])) < 1e-8:
            I1 = integrate_conic_Q_dy_yparam(conic, P0, M, Q, tol=tol, n_gl=n_gl)
            I2 = integrate_conic_Q_dy_yparam(conic, M, P1, Q, tol=tol, n_gl=n_gl)
            return I1 + I2

    return integrate_conic_Q_dy_xparam(conic, P0, P1, Q, tol=tol, n_gl=n_gl)


# straight segment line integrals

def straight_Qf_dy_integral(P, Qp):
    """Analytic ∫[(1/3)x^3 - x y^2] dy along a straight segment P->Qp."""
    xb, yb = P
    xbp, ybp = Qp
    dx, dy = xbp - xb, ybp - yb
    a, b, c, d = xb, dx, yb, dy
    return dy * (
        (1/3.)*a**3 + (a*a*b)/2 + (a*b*b)/3 + (b**3)/12
        - (a*c**2) - (a*c*d + (b*c**2)/2)
        - ((a*d**2 + 2*b*c*d)/3) - (b*d**2)/4
    )

def straight_x_dy_integral(P, Qp):
    """Analytic ∫ x dy along straight chord P->Qp."""
    return (Qp[1] - P[1]) * 0.5*(P[0] + Qp[0])


# exact flux over planar triangle

def triangle_flux_z(P, Q, R):
    """
    Flux of F=(0,0,z) across an oriented triangle PQR:
        ∬_T F·n dS = ∬_T z n_z dS
    Since z is linear on a triangle, use exact mean value:
        = (n_z * Area) * (z_P + z_Q + z_R)/3
    with oriented area vector = 0.5 * ((Q-P) x (R-P)).
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float); R = np.asarray(R, float)
    Az = 0.5 * np.cross(Q - P, R - P)[2]
    zbar = (P[2] + Q[2] + R[2]) / 3.0
    return Az * zbar


# cap areas

def cap_area_arc_plus_chord(conic, P0, P1, tol=1e-14):
    """
    Segment cap area enclosed by: arc P0->P1 on the conic, then chord P1->P0.
    Returns: (abs_area, signed_area, area_arc, area_chord).
    Orientation matters for the signed area; abs(...) is provided for convenience only.
    """
    area_arc   = conic_segment_Q_dy(conic, P0, P1, tol=tol, Q=Q_area, n_gl=32)
    area_chord = straight_x_dy_integral(P1, P0)  # chord back P1→P0 to close
    A_signed   = area_arc + area_chord
    return abs(A_signed), A_signed, area_arc, area_chord


# circle utilities / sector/segment areas

def _is_vertical_cylinder(coeffs, tol=1e-14):
    """True for x^2 + y^2 + ... with no z-coupling (C=D=E=F=G=H=J=0)."""
    A,B,C,D,E,F,G,H,J,K = map(float, coeffs)
    return (
        abs(C) < tol and abs(D) < tol and abs(E) < tol and abs(F) < tol
        and abs(G) < tol and abs(H) < tol and abs(J) < tol
    )

def _circle_radius_from_conic(conic, tol=1e-12):
    """If the 2D conic is a circle centered at origin: a x^2 + a y^2 + f = 0, return radius."""
    a,b,c2,d,e,f = conic
    if abs(b) < tol and abs(d) < tol and abs(e) < tol and a > 0 and abs(a - c2) < tol:
        if f < 0:
            return math.sqrt(max(0.0, -f/a))
    return None

def _wrap_to_pi(dth):
    # wrap into (-pi, pi]
    while dth <= -math.pi: dth += 2*math.pi
    while dth >   math.pi: dth -= 2*math.pi
    return dth

def cap_area_sector_circle(conic, P0, P1, use_minor_arc=True):
    """Circle-only: sector area (signed) swept from P0 to P1. Returns (abs_area, signed_area)."""
    r = _circle_radius_from_conic(conic)
    if r is None:
        raise ValueError("cap_area_sector_circle: conic is not a circle centered at origin.")
    th0 = math.atan2(P0[1], P0[0])
    th1 = math.atan2(P1[1], P1[0])
    dth = th1 - th0
    if use_minor_arc:
        dth = _wrap_to_pi(dth)
    A_signed = 0.5 * r * r * dth
    return abs(A_signed), A_signed

def _circle_segment_area_signed(P0, P1, r=1.0):
    """Signed minor circular segment area from P0→P1, using ∮ x dy (CCW positive)."""
    th0 = math.atan2(P0[1], P0[0])
    th1 = math.atan2(P1[1], P1[0])
    dth = _wrap_to_pi(th1 - th0)  # signed minor arc (CCW positive)
    return 0.5 * r * r * (dth - math.sin(dth))


# arc length of conic segment

def length_conic_segment(conic, P0, P1, tol=1e-14, max_levels=14, n_gl=32):
    """
    Arc length of an implicit conic segment g(x,y)=0 from P0->P1.
    Uses y-parameterization when possible; otherwise x-parameterization.
    If Δy≈0 but Δx large, splits at an interior on-branch point and sums two y-param halves.
    Includes a circle fast-path: length = r * |Δθ_minor|.
    """
    a,b,c2,d,e,f = map(float, conic)
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)
    x0,y0 = float(P0[0]), float(P0[1])
    x1,y1 = float(P1[0]), float(P1[1])

    # Circle fast-path
    if abs(b) < 1e-12 and abs(d) < 1e-12 and abs(e) < 1e-12 and a > 0 and abs(a - c2) < 1e-12:
        r2 = -f/a
        if r2 >= 0:
            r = math.sqrt(r2)
            th0 = math.atan2(y0, x0)
            th1 = math.atan2(y1, x1)
            dth = _wrap_to_pi(th1 - th0)
            return abs(r * dth)

    nodes, weights = leggauss(n_gl)

    def gx(x,y): return 2*a*x + b*y + d
    def gy(x,y): return b*x + 2*c2*y + e

    def length_yparam(P, Q):
        Py, Qy = float(P[1]), float(Q[1])
        if abs(Qy - Py) < 1e-15:
            raise RuntimeError("degenerate y-interval")
        xa = pick_branch(roots_x_of_y(conic, Py), P[0])
        xb = pick_branch(roots_x_of_y(conic, Qy), Q[0])
        if xa is None or xb is None:
            raise RuntimeError("end root fail y-param")
        def quad(a_, b_, x_left, x_right):
            half = 0.5*(b_ - a_)
            center = 0.5*(b_ + a_)
            total = 0.0
            for t, w in zip(nodes, weights):
                y = center + half*t
                hint = x_left + (x_right - x_left)*((y - a_)/(b_ - a_))
                xs = roots_x_of_y(conic, y)
                if not xs:
                    raise RuntimeError("root fail y-param")
                x  = pick_branch(xs, hint)
                _gx = gx(x,y); _gy = gy(x,y)
                if abs(_gx) < 1e-14:
                    raise RuntimeError("gx≈0; switch param")
                total += w * math.sqrt(1.0 + (_gy/_gx)**2)
            return total * abs(half)
        return quad(Py, Qy, xa, xb)

    def length_xparam(P, Q):
        Px, Qx = float(P[0]), float(Q[0])
        if abs(Qx - Px) < 1e-15:
            raise RuntimeError("degenerate x-interval")
        ya = pick_branch(roots_y_of_x(conic, Px), P[1])
        yb = pick_branch(roots_y_of_x(conic, Qx), P[1])
        if ya is None or yb is None:
            raise RuntimeError("end root fail x-param")
        def quad(a_, b_, y_left, y_right):
            half = 0.5*(b_ - a_)
            center = 0.5*(b_ + a_)
            total = 0.0
            for t, w in zip(nodes, weights):
                x = center + half*t
                hint = y_left + (y_right - y_left)*((x - a_)/(b_ - a_))
                ys = roots_y_of_x(conic, x)
                if not ys:
                    raise RuntimeError("root fail x-param")
                y  = pick_branch(ys, hint)
                _gx = gx(x,y); _gy = gy(x,y)
                if abs(_gy) < 1e-14:
                    raise RuntimeError("gy≈0; switch param")
                total += w * math.sqrt(1.0 + (_gx/_gy)**2)
            return total * abs(half)
        return quad(Px, Qx, ya, yb)

    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    try:
        if dy >= 1e-12:
            return length_yparam(P0, P1)
    except Exception:
        pass

    if dy < 1e-12 and dx >= 1e-12:
        M = interior_point_on_conic(conic, P0, P1)
        if abs(g_val(conic, M[0], M[1])) < 1e-8:
            return length_yparam(P0, M) + length_yparam(M, P1)

    return length_xparam(P0, P1)


# main volume routine

def volume_ABApBp_general_conic(coeffs, A, B, A_, B_, cap_mode="segment", assume_cylinder=True, debug=False):
    """
    Compute volume bounded by:
      bottom arc A->B on z=z_low,
      vertical edges B->B' and A'->A,
      top arc B'->A' on z=z_high,
      plus the planar quad A-B-B'-A'.

    Sign conventions (as seen from +z, CCW positive):
      - bottom arc uses A->B  (CW)  → signed area negative
      - top arc    uses B'->A' (CCW) → signed area positive
    Assembly:
        V = I_flat - I_curved + z_high*A_top_signed + z_low*A_bot_signed
      For a vertical cylinder, this reduces to:
        V = (z_high - z_low) * (minor-segment area)

    Returns 6-tuple (V, I_flat, I_curved, caps, timings, lengths).
    """
    if debug:
        print("Volume AB'AP'B' for conic:", coeffs, "A:", A, "B:", B, "A':", A_, "B':", B_)

    A  = np.asarray(A,  float)
    B  = np.asarray(B,  float)
    A_ = np.asarray(A_, float)
    B_ = np.asarray(B_, float)

    z_low  = float(A[2])
    z_high = float(A_[2])
    conic_low  = conic_at_z(coeffs, z_low)
    conic_high = conic_at_z(coeffs, z_high)

    t0 = time.perf_counter(); t = {}

    # caps (areas on bottom and top)
    # Bottom: A -> B (CW from +z) => signed area negative
    t['cap_bot_start'] = time.perf_counter()
    if cap_mode == "segment":
        Abot_abs, Abot_signed, Abot_arc, Abot_chord = cap_area_arc_plus_chord(
            conic_low, (A[0], A[1]), (B[0], B[1])
        )
    elif cap_mode == "sector":
        Abot_abs, Abot_signed = cap_area_sector_circle(
            conic_low, (A[0], A[1]), (B[0], B[1]), use_minor_arc=True
        )
        Abot_arc = Abot_chord = None
    else:
        raise ValueError("cap_mode must be 'segment' or 'sector'")
    t['cap_bot'] = time.perf_counter() - t['cap_bot_start']

    # Top: B' -> A' (CCW from +z) => signed area positive
    t['cap_top_start'] = time.perf_counter()
    if cap_mode == "segment":
        Atop_abs, Atop_signed, Atop_arc, Atop_chord = cap_area_arc_plus_chord(
            conic_high, (B_[0], B_[1]), (A_[0], A_[1])
        )
    else:
        Atop_abs, Atop_signed = cap_area_sector_circle(
            conic_high, (B_[0], B_[1]), (A_[0], A_[1]), use_minor_arc=True
        )
        Atop_arc = Atop_chord = None
    t['cap_top'] = time.perf_counter() - t['cap_top_start']

    # cylinder shortcut: vertical faces => zero z-flux; curved side integral => 0
    if assume_cylinder and _is_vertical_cylinder(coeffs):
        I_flat = 0.0
        I_curved = 0.0
        t['flat'] = t['lower'] = t['bb'] = t['upper'] = t['aa'] = 0.0

        # radii (both planes should be the same circle)
        r0 = _circle_radius_from_conic(conic_low)
        r1 = _circle_radius_from_conic(conic_high)
        if r0 is None or r1 is None or abs(r0 - r1) > 1e-12:
            raise RuntimeError("Expected identical circles for cylinder shortcut.")

        # Use exact circle segment areas with the required directions:
        # bottom: A -> B (CW) => negative signed area
        Abot_signed = _circle_segment_area_signed((A[0], A[1]), (B[0], B[1]), r0)
        # top:    B'->A' (CCW) => positive signed area
        Atop_signed = _circle_segment_area_signed((B_[0], B_[1]), (A_[0], A_[1]), r1)

        # diagnostics (optional)
        L_lower = length_conic_segment(conic_low,  (A[0],  A[1]),  (B[0],  B[1]))
        L_upper = length_conic_segment(conic_high, (B_[0], B_[1]), (A_[0], A_[1]))

        V = z_high * Atop_signed + z_low * Abot_signed
        t['total'] = time.perf_counter() - t0

        if debug:
            print("  [cylinder-fast] A_top_signed =", Atop_signed, "A_bot_signed =", Abot_signed)
            print("  z_high*A_top_signed =", z_high * Atop_signed)
            print("  z_low *A_bot_signed =", z_low  * Abot_signed)
            print("Volume:", V)

        timings = {
            'total': t.get('total', 0.0),
            'lower': 0.0, 'bb': 0.0, 'upper': 0.0, 'aa': 0.0, 'flat': 0.0,
            'cap_top': t.get('cap_top', 0.0), 'cap_bot': t.get('cap_bot', 0.0),
        }
        caps = {
            'A_top': abs(Atop_signed), 'A_bot': abs(Abot_signed),
            'z_top': z_high, 'z_bot': z_low,
            'A_top_signed': Atop_signed, 'A_bot_signed': Abot_signed,
            'A_top_arc': None, 'A_top_chord': None,
            'A_bot_arc': None, 'A_bot_chord': None,
        }
        lengths = {'L_lower': L_lower, 'L_upper': L_upper}

        #print("Volume for conic:", coeffs, "A:", A, "B:", B, "A':", A_, "B':", B_, " V:", V)
        return V, I_flat, I_curved, caps, timings, lengths

    # planar quad side (two triangles)
    t['flat_start'] = time.perf_counter()
    I_flat = triangle_flux_z(A, B, B_) + triangle_flux_z(A, B_, A_)
    t['flat'] = time.perf_counter() - t['flat_start']

    # curved side (general conic case)
    t['lower_start'] = time.perf_counter()
    I_lower = conic_segment_Q_dy(conic_low,  (A[0],  A[1]),  (B[0],  B[1]), Q=Q_f)
    t['lower'] = time.perf_counter() - t['lower_start']

    t['bb_start'] = time.perf_counter()
    I_BB = straight_Qf_dy_integral((B[0],  B[1]), (B_[0], B_[1]))
    t['bb'] = time.perf_counter() - t['bb_start']

    t['upper_start'] = time.perf_counter()
    I_upper = conic_segment_Q_dy(conic_high, (B_[0], B_[1]), (A_[0], A_[1]), Q=Q_f)  # B'→A'
    t['upper'] = time.perf_counter() - t['upper_start']

    t['aa_start'] = time.perf_counter()
    I_AA = straight_Qf_dy_integral((A_[0], A_[1]), (A[0],  A[1]))                  # A'→A
    t['aa'] = time.perf_counter() - t['aa_start']

    I_curved = I_lower + I_BB + I_upper + I_AA

    # lengths (diagnostic)
    L_lower = length_conic_segment(conic_low,  (A[0],  A[1]),  (B[0],  B[1]))
    L_upper = length_conic_segment(conic_high, (B_[0], B_[1]), (A_[0], A_[1]))

    # assemble
    V = I_flat - I_curved + z_high * Atop_signed + z_low * Abot_signed
    t['total'] = time.perf_counter() - t0

    if debug:
        print("  I_flat =", I_flat)
        print("  I_curved =", I_curved)
        print("  z_high*A_top_signed =", z_high * Atop_signed)
        print("  z_low *A_bot_signed =", z_low  * Abot_signed)
        print("Volume:", V)

    timings = {
        'total': t.get('total', 0.0),
        'lower': t.get('lower', 0.0), 'bb': t.get('bb', 0.0), 'upper': t.get('upper', 0.0),
        'aa': t.get('aa', 0.0), 'flat': t.get('flat', 0.0),
        'cap_top': t.get('cap_top', 0.0), 'cap_bot': t.get('cap_bot', 0.0),
    }
    caps = {
        'A_top': Atop_abs, 'A_bot': Abot_abs, 'z_top': z_high, 'z_bot': z_low,
        'A_top_signed': Atop_signed, 'A_bot_signed': Abot_signed,
        'A_top_arc': Atop_arc, 'A_top_chord': Atop_chord,
        'A_bot_arc': Abot_arc, 'A_bot_chord': Abot_chord,
    }
    lengths = {'L_lower': L_lower, 'L_upper': L_upper}

    #print("Volume for conic:", coeffs, "A:", A, "B:", B, "A':", A_, "B':", B_, " V:", V)
    return V, I_flat, I_curved, caps, timings, lengths


# flat_plane_guard.py
# coding: utf-8
"""
Flat-plane guard for triangle patches on general quadrics.

API
---
flat_plane_zero_volume(A, B, C, coeffs,
                       tol_coord=1e-8, tol_F=1e-6, tol_k=1e-10)
    -> 0.0 if triangle lies on a flat plane (skip patch),
       None otherwise.

Notes
-----
- "Coordinate" test: flags axis-aligned planes x=const, y=const, z=const.
- "Curvature" test: if A,B,C lie (numerically) on the quadric F=0 defined by
  coeffs = (A,B,C,D,E,F,G,H,J,K), we compute principal curvatures (k1,k2)
  at the triangle centroid using the implicit-surface shape operator.
  If both |k1|,|k2| ≈ 0, the surface is locally planar.

- Works for any quadric: plane, sphere, ellipsoid, cone, cylinder, paraboloid,
  hyperboloid, etc. (Degenerate quadrics that are exactly planes naturally yield
  k1=k2=0 everywhere.)
"""


import math
import numpy as np
from typing import Tuple, Optional, Dict

ArrayLike = np.ndarray | Tuple[float, float, float] | list

# Quadric helpers

def quadric_F(x: float, y: float, z: float, coeffs: Tuple[float, ...]) -> float:
    """
    F(x,y,z) = A x^2 + B y^2 + C z^2 + D x y + E x z + F y z + G x + H y + J z + K
    """
    A,B,C,D,E,Fc,G,H,J,K = map(float, coeffs)
    return (A*x*x + B*y*y + C*z*z +
            D*x*y + E*x*z + Fc*y*z +
            G*x + H*y + J*z + K)

def quadric_grad(x: float, y: float, z: float, coeffs: Tuple[float, ...]) -> np.ndarray:
    """
    ∇F = [2Ax + Dy + Ez + G,  2By + Dx + Fz + H,  2Cz + Ex + Fy + J]
    """
    A,B,C,D,E,Fc,G,H,J,_ = map(float, coeffs)
    gx = 2.0*A*x + D*y + E*z + G
    gy = 2.0*B*y + D*x + Fc*z + H
    gz = 2.0*C*z + E*x + Fc*y + J
    return np.array([gx, gy, gz], dtype=float)

def quadric_hess(coeffs: Tuple[float, ...]) -> np.ndarray:
    """
    Hessian of F is constant for quadrics:
      H = [[2A, D , E ],
           [ D, 2B, F ],
           [ E,  F, 2C]]
    """
    A,B,C,D,E,Fc,_,_,_,_ = map(float, coeffs)
    return np.array([[2.0*A,   D,   E],
                     [   D, 2.0*B,  Fc],
                     [   E,   Fc, 2.0*C]], dtype=float)

# Differential geometry on implicit surfaces

def _tangent_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal tangent vectors given a unit normal n (3,)."""
    n = np.asarray(n, float)
    # pick a vector not parallel to n
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1_norm = np.linalg.norm(t1)
    if t1_norm == 0.0:  # pathological fallback
        a = np.array([0.0, 0.0, 1.0])
        t1 = np.cross(n, a)
        t1_norm = np.linalg.norm(t1)
        if t1_norm == 0.0:
            # final resort: arbitrary basis
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    t1 /= t1_norm
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2

def principal_curvatures_on_quadric(P: ArrayLike,
                                    coeffs: Tuple[float, ...],
                                    eps: float = 1e-12
                                    ) -> Tuple[float, float]:
    """
    Principal curvatures (k1<=k2) of the quadric F=0 at point P.
    Uses shape operator S = -(E^T H E)/||∇F|| on the tangent plane,
    where H is constant Hessian, E=[t1 t2] is an orthonormal tangent basis.
    """
    P = np.asarray(P, float).reshape(3)
    g = quadric_grad(P[0], P[1], P[2], coeffs)
    gnorm = float(np.linalg.norm(g))
    if not np.isfinite(gnorm) or gnorm < eps:
        return math.nan, math.nan
    H = quadric_hess(coeffs)
    n = g / gnorm
    t1, t2 = _tangent_basis_from_normal(n)
    E = np.column_stack((t1, t2))  # 3x2
    S2 = -(E.T @ H @ E) / gnorm    # 2x2 symmetric
    w = np.linalg.eigvalsh(S2)     # eigenvalues sorted ascending
    return float(w[0]), float(w[1])

# Main guard

def flat_plane_zero_volume(A: ArrayLike, B: ArrayLike, C: ArrayLike,
                           coeffs: Tuple[float, ...],
                           tol_coord: float = 1e-8,
                           tol_F: float = 1e-6,
                           tol_k: float = 1e-10
                           ) -> Optional[float]:
    """
    Return 0.0 if triangle ABC lies on a flat plane; otherwise return None.

    Strategy:
      1) Coordinate-aligned plane test (fast): if range(x)≤tol or range(y)≤tol or range(z)≤tol.
      2) Curvature test on the *quadric*:
         If all three vertices are on F=0 within tol_F, compute principal curvatures
         (k1,k2) at triangle centroid. If max(|k1|,|k2|) ≤ tol_k => locally planar.

    Parameters
    ----------
    A,B,C : array-like length 3
        Triangle vertices.
    coeffs : tuple of 10 floats
        Quadric coefficients (A,B,C,D,E,F,G,H,J,K) for F(x,y,z)=0.
    tol_coord : float
        Tolerance for coordinate-equality plane detection.
    tol_F : float
        Tolerance for "on-surface" check |F| ≤ tol_F at A,B,C.
    tol_k : float
        Tolerance for principal curvature magnitudes.

    Returns
    -------
    0.0 if planar; None otherwise.
    """
    A = np.asarray(A, float).reshape(3)
    B = np.asarray(B, float).reshape(3)
    C = np.asarray(C, float).reshape(3)

    # (1) Coordinate-aligned plane checks
    xs = np.array([A[0], B[0], C[0]])
    ys = np.array([A[1], B[1], C[1]])
    zs = np.array([A[2], B[2], C[2]])
    if xs.max() - xs.min() <= tol_coord:
        return 0.0
    if ys.max() - ys.min() <= tol_coord:
        return 0.0
    if zs.max() - zs.min() <= tol_coord:
        return 0.0        

    # Optional: degenerate triangle (tiny area) is effectively planar
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
    if area <= 1e-16:
        return 0.0

    # (2) Curvature test (only meaningful if the vertices sit on the quadric)
    FA = abs(quadric_F(A[0], A[1], A[2], coeffs))
    FB = abs(quadric_F(B[0], B[1], B[2], coeffs))
    FC = abs(quadric_F(C[0], C[1], C[2], coeffs))
    if max(FA, FB, FC) <= tol_F:
        Pc = (A + B + C) / 3.0
        k1, k2 = principal_curvatures_on_quadric(Pc, coeffs)
        if (np.isfinite(k1) and np.isfinite(k2) and
                max(abs(k1), abs(k2)) <= tol_k):
            return 0.0

    # Not planar by these tests
    return None

# Optional convenience: boolean + diagnostics

def is_triangle_planar(A: ArrayLike, B: ArrayLike, C: ArrayLike,
                       coeffs: Tuple[float, ...],
                       tol_coord: float = 1e-8,
                       tol_F: float = 1e-6,
                       tol_k: float = 1e-10
                       ) -> Tuple[bool, Dict[str, float]]:
    """
    Same checks as `flat_plane_zero_volume`, but returns (bool, info).
    """
    V0 = flat_plane_zero_volume(A, B, C, coeffs, tol_coord, tol_F, tol_k)
    info: Dict[str, float] = {}
    planar = (V0 == 0.0)

    if planar:
        info["planar"] = 1.0
        return True, info

    # If not flagged planar, provide curvature diagnostics (if on quadric)
    Pc = (np.asarray(A)+np.asarray(B)+np.asarray(C))/3.0
    FA = abs(quadric_F(*A, coeffs))
    FB = abs(quadric_F(*B, coeffs))
    FC = abs(quadric_F(*C, coeffs))
    if max(FA, FB, FC) <= tol_F:
        k1, k2 = principal_curvatures_on_quadric(Pc, coeffs)
        info.update({"k1": k1, "k2": k2})
    else:
        info.update({"k1": math.nan, "k2": math.nan})
    info["planar"] = 0.0
    return False, info


# Self-test
if __name__ == "__main__":
    # Example 1: triangle on the plane x=2 (coordinate-aligned)
    A = (2.0, -1.5, -1.0)
    B = (2.0,  1.2,  0.3)
    C = (2.0, -0.2,  0.9)

    # A generic quadric (take a cylinder for the test): x^2 + y^2 - 1 = 0
    coeffs_cyl = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)

    print("V (x=const plane):", flat_plane_zero_volume(A, B, C, coeffs_cyl))

    # Example 2: triangle on the cylinder wall (should NOT be planar from curvature)
    z1, z2, z3 = -0.7, 0.0, 0.8
    Aw = (math.sqrt(1 - 0.0**2), -0.6, z1)  # x,y chosen to satisfy x^2+y^2=1 (rough)
    Bw = (0.8, 0.6, z2)
    Cw = (0.6, 0.8, z3)
    print("V (curved wall):", flat_plane_zero_volume(Aw, Bw, Cw, coeffs_cyl))

    # Example 3: exact plane quadric (k1=k2=0 everywhere)
    # F = x + 2y - 3z + 1 = 0  -> coeffs with only linear terms
    coeffs_plane = (0,0,0,0,0,0,1,2,-3,1)
    P1 = (0.0, 0.0, 1.0/3.0)      # satisfies x+2y-3z+1=0
    P2 = (1.0, 0.0, 2.0/3.0)
    P3 = (-1.0, 1.0, 0.0)
    print("V (plane quadric):", flat_plane_zero_volume(P1, P2, P3, coeffs_plane))

