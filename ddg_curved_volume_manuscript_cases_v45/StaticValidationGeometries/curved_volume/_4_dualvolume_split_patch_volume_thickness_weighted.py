#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

# ------------------------------------------------------------
# Utility: numerically safe comparisons
# ------------------------------------------------------------
_EPS = 1e-12

def _assert_close(a, b, tol=1e-12, msg=""):
    if abs(a - b) > tol * max(1.0, abs(b)):
        raise AssertionError(f"{msg}  got={a:.16e}, want={b:.16e}")

# ------------------------------------------------------------
# 1) Reduce a 3D quadric to the face-plane conic  g~(x,y)=0
#    f(x,y,z) = Axx x^2 + Byy y^2 + Czz z^2 + 2 Dxy xy + 2 Exz xz + 2 Fyz yz
#               + 2 Gx x + 2 Hy y + 2 Iz z + J
#    NOTE: your coeffs were in a 1× convention; this reducer accepts that and
#          returns the in-plane 1× convention:
#          g~(x,y)= A' x^2 + D' x y + B' y^2 + G' x + H' y + J'
# ------------------------------------------------------------
def _plane_reduce_conic(A, B, C, coeffs, eps=_EPS):
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)
    Axx, Byy, Czz, Dxy, Exz, Fyz, Gx, Hy, Iz, J = [float(t) for t in coeffs]

    # Plane through A,B,C: n·X + d = 0
    n = np.cross(B - A, C - A)
    nn = float(np.dot(n, n))
    if nn < eps:
        raise ValueError("Degenerate face: A,B,C are nearly collinear.")
    a, b, c = n
    d = -float(np.dot(n, A))

    # Solve plane for the stablest coordinate (max |component|)
    # Write the eliminated variable as  z = α x + β y + γ  (or x(...), y(...))
    # Then substitute into the 3D quadric and collect like terms.
    # We implement all three branches to avoid conditioning issues.
    if abs(c) >= max(abs(a), abs(b)):
        # eliminate z -> z = α x + β y + γ
        alpha = -a / c
        beta  = -b / c
        gamma = -d / c

        # Build substitution helpers
        # Quadratic terms after z-sub:
        # x^2: Axx + Czz*alpha^2 + 2*Exz*alpha
        # y^2: Byy + Czz*beta^2  + 2*Fyz*beta
        # xy:  2*Dxy + 2*Czz*alpha*beta + 2*Exz*beta + 2*Fyz*alpha
        # linear x: 2*Gx + 2*Czz*alpha*gamma + 2*Exz*gamma + 2*Iz*alpha
        # linear y: 2*Hy + 2*Czz*beta*gamma  + 2*Fyz*gamma + 2*Iz*beta
        # const:    J   + Czz*gamma**2 + 2*Iz*gamma

        A_p = Axx + Czz*alpha*alpha + 2.0*Exz*alpha
        B_p = Byy + Czz*beta*beta  + 2.0*Fyz*beta
        D_p = 2.0*Dxy + 2.0*Czz*alpha*beta + 2.0*Exz*beta + 2.0*Fyz*alpha
        G_p = 2.0*Gx + 2.0*Czz*alpha*gamma + 2.0*Exz*gamma + 2.0*Iz*alpha
        H_p = 2.0*Hy + 2.0*Czz*beta*gamma  + 2.0*Fyz*gamma + 2.0*Iz*beta
        J_p = J + Czz*gamma*gamma + 2.0*Iz*gamma

        # Return 1× convention in the (x,y) chart: A',D',B',G',H',J'
        return (A_p, D_p, B_p, G_p, H_p, J_p), ('x','y')

    elif abs(a) >= abs(b):
        # eliminate x -> x = α y + β z + γ  with y,z as chart axes
        # From plane: a x + b y + c z + d = 0 -> x = -(b y + c z + d)/a
        alpha = -b / a
        beta  = -c / a
        gamma = -d / a

        # Substitute into f(x,y,z) but we want a 2D equation in (y,z).
        # We'll name chart variables (u,v)=(y,z), then relabel back to (x,y) at return.
        # After substitution, collect u^2, v^2, uv, u, v, const in 1× convention:
        # For brevity, do it via symbolic-like manual expansion.
        # Build x as linear form in (u=y, v=z): x = alpha*u + beta*v + gamma

        # Helpers for quadratic terms (matrix form is convenient)
        # f = [x y z] Q [x y z]^T + 2 L·[x y z] + J   in 1× convention, where
        # Q = [[Axx, Dxy, Exz],
        #      [Dxy, Byy, Fyz],
        #      [Exz, Fyz, Czz]],  L = [Gx, Hy, Iz]
        Q = np.array([[Axx, Dxy, Exz],
                      [Dxy, Byy, Fyz],
                      [Exz, Fyz, Czz]], dtype=float)
        L = np.array([Gx, Hy, Iz], dtype=float)

        # Substitute x = alpha*u + beta*v + gamma, y = u, z = v
        # Build mapping M s.t. [x y z]^T = M [u v 1]^T
        M = np.array([[alpha, beta, gamma],
                      [1.0,   0.0,  0.0 ],
                      [0.0,   1.0,  0.0 ]], dtype=float)

        # Expand f = (M ξ)^T Q (M ξ) + 2 L^T (M ξ) + J, with ξ=[u v 1]^T
        # Compute the 3x3 block for ξ^T (·) ξ
        QT = M.T @ Q @ M
        LT = (M.T @ L.reshape(3,1)).reshape(3)
        JT = float(J)

        # Now read off coefficients for u^2, uv, v^2, u, v, const (1× convention):
        A_u = QT[0,0]               # u^2
        D_uv = 2.0*QT[0,1]          # uv (note the 1× convention packs xy term without the 2)
        B_v = QT[1,1]               # v^2
        G_u = 2.0*LT[0] + 2.0*QT[0,2]  # u
        H_v = 2.0*LT[1] + 2.0*QT[1,2]  # v
        J_c = JT + QT[2,2] + 2.0*LT[2]

        # We currently have conic in (u,v)=(y,z).
        # Return as (A',D',B',G',H',J') but **label chart** as ('y','z')
        return (A_u, D_uv, B_v, G_u, H_v, J_c), ('y','z')

    else:
        # eliminate y -> y = α x + β z + γ, chart (x,z)
        alpha = -a / b
        beta  = -c / b
        gamma = -d / b

        Q = np.array([[Axx, Dxy, Exz],
                      [Dxy, Byy, Fyz],
                      [Exz, Fyz, Czz]], dtype=float)
        L = np.array([Gx, Hy, Iz], dtype=float)

        # [x y z]^T = M [u v 1]^T with (u,v)=(x,z)
        M = np.array([[1.0,   0.0,  0.0 ],
                      [alpha, beta, gamma],
                      [0.0,   1.0,  0.0 ]], dtype=float)

        QT = M.T @ Q @ M
        LT = (M.T @ L.reshape(3,1)).reshape(3)
        JT = float(J)

        A_u = QT[0,0]                 # u^2 (x^2)
        D_uv = 2.0*QT[0,1]            # u v  (x z)
        B_v = QT[1,1]                 # v^2 (z^2)
        G_u = 2.0*LT[0] + 2.0*QT[0,2] # u
        H_v = 2.0*LT[1] + 2.0*QT[1,2] # v
        J_c = JT + QT[2,2] + 2.0*LT[2]

        # Return as (A',D',B',G',H',J') with chart ('x','z')
        return (A_u, D_uv, B_v, G_u, H_v, J_c), ('x','z')


# ------------------------------------------------------------
# 2) Thickness from the plane-reduced conic: span between two roots
#    Vertical cut (fixed y):  A' x^2 + 2(D' y + G') x + (...) = 0
#    Horizontal cut (fixed x): B' y^2 + 2(D' x + H') y + (...) = 0
# ------------------------------------------------------------
def _thickness_span(conic, x_fixed=None, y_fixed=None, eps=_EPS):
    A1, D1, B1, G1, H1, J1 = conic

    if y_fixed is not None:
        # Quadratic in x: a x^2 + b x + c = 0
        a = A1
        b = 2.0 * (D1 * y_fixed + G1)
        c = B1 * y_fixed * y_fixed + 2.0 * H1 * y_fixed + J1
        # Solve robustly
        if abs(a) < eps:
            if abs(b) < eps:
                return 0.0
            # linear: one root -> thickness = 0
            return 0.0
        disc = b*b - 4.0*a*c
        if disc <= 0.0:
            return 0.0
        s = math.sqrt(disc)
        r1 = (-b - s) / (2.0*a)
        r2 = (-b + s) / (2.0*a)
        return abs(r2 - r1)

    if x_fixed is not None:
        # Quadratic in y
        a = B1
        b = 2.0 * (D1 * x_fixed + H1)
        c = A1 * x_fixed * x_fixed + 2.0 * G1 * x_fixed + J1
        if abs(a) < eps:
            if abs(b) < eps:
                return 0.0
            return 0.0
        disc = b*b - 4.0*a*c
        if disc <= 0.0:
            return 0.0
        s = math.sqrt(disc)
        r1 = (-b - s) / (2.0*a)
        r2 = (-b + s) / (2.0*a)
        return abs(r2 - r1)

    raise ValueError("Specify either x_fixed or y_fixed.")


# ------------------------------------------------------------
# 3) Main: split patch volume using thickness-weighted barycentric stencil
# ------------------------------------------------------------
def split_patch_volume_thickness_weighted(V_patch, A, B, C, coeffs, eps=_EPS):
    """
    Thickness-weighted split of a boundary face patch volume V_patch to vertices A,B,C.
    - Always reduces the 3D quadric to the face-plane conic g~(x,y)=0.
    - Samples thickness (span between the two intersections) at the triangle
      centroid and the three edge midpoints.
    - Uses a simple barycentric-based stencil to form per-vertex moments, then
      normalizes to conserve V_patch.

    coeffs = (Axx, Byy, Czz, Dxy, Exz, Fyz, Gx, Hy, Iz, J)  in 1× convention.
    Returns (V_A, V_B, V_C) with V_A + V_B + V_C == V_patch (up to round-off).
    """
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)

    # Face-plane conic (and the chart it lives in)
    try:
        conic, chart = _plane_reduce_conic(A, B, C, coeffs, eps=eps)
    except ValueError:
        v = V_patch / 3.0
        return v, v, v

    # Sample points on the face (in 3D)
    F   = (A + B + C) / 3.0
    MAB = (A + B) / 2.0
    MBC = (B + C) / 2.0
    MCA = (C + A) / 2.0
    samples = [F, MAB, MBC, MCA]

    # Choose sampling direction by conic conditioning:
    # If |A'| is comfortably larger than |B'|, prefer vertical (x-span at fixed y);
    # otherwise prefer horizontal (y-span at fixed x).
    A1, D1, B1, G1, H1, J1 = conic
    prefer_vertical = abs(A1) >= abs(B1)

    # Build thicknesses at the four sample points
    ts = []
    for P in samples:
        x, y, z = float(P[0]), float(P[1]), float(P[2])

        if chart == ('x','y'):
            # direct chart: (x,y)
            t = _thickness_span(conic, y_fixed=y) if prefer_vertical else _thickness_span(conic, x_fixed=x)
        elif chart == ('y','z'):
            # chart is (u,v)=(y,z): use y as "x", z as "y" within conic;
            # for our thickness we can treat "vertical" as fixed v => u-span, etc.
            u, v = y, z
            t = _thickness_span(conic, y_fixed=v) if prefer_vertical else _thickness_span(conic, x_fixed=u)
        else:  # chart == ('x','z')
            u, v = x, z
            t = _thickness_span(conic, y_fixed=v) if prefer_vertical else _thickness_span(conic, x_fixed=u)

        ts.append(max(0.0, float(t)))

    # If everything degenerated, equal thirds
    if not any(t > eps for t in ts):
        v = V_patch / 3.0
        return v, v, v

    # Map the four thickness samples to per-vertex "moments" using a barycentric stencil
    tF, tAB, tBC, tCA = ts
    hA = (tF/3.0) + (tAB/3.0) + (tCA/3.0)  # centroid (1/3 to A) + two incident mid-edges (1/2 but shared via 1/3 overall weight)
    hB = (tF/3.0) + (tAB/3.0) + (tBC/3.0)
    hC = (tF/3.0) + (tBC/3.0) + (tCA/3.0)

    # Clamp small negatives (round-off) and normalize
    hA = max(0.0, hA); hB = max(0.0, hB); hC = max(0.0, hC)
    S = hA + hB + hC
    if S <= eps:
        v = V_patch / 3.0
        return v, v, v

    wA, wB, wC = hA / S, hB / S, hC / S
    return (wA * V_patch, wB * V_patch, wC * V_patch)


# ------------------------------------------------------------
# 3b) NEW: split patch area using the SAME thickness-weighted stencil
#     (for A_curved → A_curved_A, A_curved_B, A_curved_C)
# ------------------------------------------------------------
def split_patch_area_thickness_weighted(A_patch, A, B, C, coeffs, eps=_EPS):
    """
    Thickness-weighted split of a boundary face **area** A_patch to vertices A,B,C.
    Same logic as split_patch_volume_thickness_weighted(...), but we conserve area.
    Returns (A_A, A_B, A_C) with A_A + A_B + A_C == A_patch (up to round-off).
    """
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)

    # Reduce quadric to in-plane conic
    try:
        conic, chart = _plane_reduce_conic(A, B, C, coeffs, eps=eps)
    except ValueError:
        a3 = A_patch / 3.0
        return a3, a3, a3

    # Sample points
    F   = (A + B + C) / 3.0
    MAB = (A + B) / 2.0
    MBC = (B + C) / 2.0
    MCA = (C + A) / 2.0
    samples = [F, MAB, MBC, MCA]

    A1, D1, B1, G1, H1, J1 = conic
    prefer_vertical = abs(A1) >= abs(B1)

    ts = []
    for P in samples:
        x, y, z = float(P[0]), float(P[1]), float(P[2])

        if chart == ('x','y'):
            t = _thickness_span(conic, y_fixed=y) if prefer_vertical else _thickness_span(conic, x_fixed=x)
        elif chart == ('y','z'):
            u, v = y, z
            t = _thickness_span(conic, y_fixed=v) if prefer_vertical else _thickness_span(conic, x_fixed=u)
        else:
            u, v = x, z
            t = _thickness_span(conic, y_fixed=v) if prefer_vertical else _thickness_span(conic, x_fixed=u)

        ts.append(max(0.0, float(t)))

    # Degenerate → equal thirds
    if not any(t > eps for t in ts):
        a3 = A_patch / 3.0
        return a3, a3, a3

    tF, tAB, tBC, tCA = ts
    hA = (tF/3.0) + (tAB/3.0) + (tCA/3.0)
    hB = (tF/3.0) + (tAB/3.0) + (tBC/3.0)
    hC = (tF/3.0) + (tBC/3.0) + (tCA/3.0)

    hA = max(0.0, hA); hB = max(0.0, hB); hC = max(0.0, hC)
    S = hA + hB + hC
    if S <= eps:
        a3 = A_patch / 3.0
        return a3, a3, a3

    wA, wB, wC = hA / S, hB / S, hC / S
    return (wA * A_patch, wB * A_patch, wC * A_patch)


# ------------------------------------------------------------
# 4) Simple sanity tests (kept from your original, now using plane reduction)
# ------------------------------------------------------------
def _print_case(name, V_patch, A, B, C, coeffs):
    VA, VB, VC = split_patch_volume_thickness_weighted(V_patch, A, B, C, coeffs)
    S = VA + VB + VC
    wA = VA / V_patch if V_patch != 0 else float("nan")
    wB = VB / V_patch if V_patch != 0 else float("nan")
    wC = VC / V_patch if V_patch != 0 else float("nan")
    print(f"\n=== {name} ===")
    print(f"A={A}, B={B}, C={C}")
    print(f"V_patch={V_patch:.6e}  ->  VA={VA:.6e}, VB={VB:.6e}, VC={VC:.6e}  (sum={S:.6e})")
    print(f"weights  wA={wA:.6f}, wB={wB:.6f}, wC={wC:.6f}")
    _assert_close(S, V_patch, 1e-12, msg=f"{name}: VA+VB+VC must equal V_patch")
    return VA, VB, VC

def run_tests():
    V_patch = 1e-4

    # 1) Cylinder: x^2 + y^2 = 1  (translation class; z-independent)
    coeffs_cyl = (1.0, 1.0, 0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, -1.0)
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([math.cos(0.6), math.sin(0.6), 0.25])
    C = np.array([math.cos(1.0), math.sin(1.0), 0.55])
    _print_case("Cylinder (R=1)", V_patch, A, B, C, coeffs_cyl)

    # 2) Hyperbola cylinder: x^2 - y^2 = 1
    coeffs_hyp = (1.0, -1.0, 0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, -1.0)
    A = np.array([math.sqrt(1.0 + 0.0**2), 0.0, 0.0])
    B = np.array([math.sqrt(1.0 + 0.5**2), 0.5, 0.30])
    C = np.array([math.sqrt(1.0 + 1.0**2), 1.0, 0.60])
    _print_case("Hyperbola cylinder (x^2 - y^2 = 1)", V_patch, A, B, C, coeffs_hyp)

    # 3) Parabola cylinder: y = x^2  → x^2 - y = 0 ⇒ Axx=1, Hy=-1/2
    coeffs_par = (1.0, 0.0, 0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, -0.5, 0.5*0.0, 0.0)
    A = np.array([0.0, 0.0, 0.00])
    B = np.array([0.70, 0.70**2, 0.25])
    C = np.array([1.10, 1.10**2, 0.55])
    _print_case("Parabola cylinder (y = x^2)", V_patch, A, B, C, coeffs_par)

    # 4) Sphere x^2 + y^2 + z^2 = 1 (general class; reduction engages)
    coeffs_sph = (1.0, 1.0, 1.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, 0.5*0.0, -1.0)
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([math.sqrt(1.0 - 0.30**2), 0.30, 0.0])
    C = np.array([math.sqrt(1.0 - 0.60**2), 0.60, 0.0])
    VA, VB, VC = _print_case("Sphere (general; uses plane reduction)", V_patch, A, B, C, coeffs_sph)
    _assert_close(VA + VB + VC, V_patch, 1e-12, "Sphere: conservation check")

    print("\nAll tests passed ✅")

if __name__ == "__main__":
    run_tests()
