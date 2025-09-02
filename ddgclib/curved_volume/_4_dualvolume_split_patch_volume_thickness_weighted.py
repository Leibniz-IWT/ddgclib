#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

def split_patch_volume_thickness_weighted(V_patch, A, B, C, coeffs, eps=1e-12):
    """
    Thickness-weighted split of a boundary face patch V_patch across its three vertices.
    coeffs = (Axx, Byy, Czz, Dxy, Exz, Fyz, Gx, Hy, Iz, J)
    Uses only (Axx, Byy, Dxy, Gx, Hy, J) — assumed z-independent (extruded) conic.
    Returns (V_A, V_B, V_C) s.t. V_A + V_B + V_C == V_patch.
    Falls back to equal thirds if ill-conditioned.
    """
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)
    Axx, Byy, Czz, Dxy, Exz, Fyz, Gx, Hy, Iz, J = [float(t) for t in coeffs]

    # plane through A,B,C: a x + b y + c z + d = 0
    n = np.cross(B - A, C - A)
    if np.dot(n, n) < eps:
        v = V_patch / 3.0
        return v, v, v
    a, b, c = n
    d = -float(np.dot(n, A))

    # conic helpers (roots in x or y)
    def roots_x_of_y(y):
        Aq = Axx
        Bq = 2.0 * (Dxy * y + Gx)
        Cq = Byy * y * y + 2.0 * Hy * y + J
        if abs(Aq) < eps:
            if abs(Bq) < eps: return []
            return [-Cq / Bq]
        disc = Bq * Bq - 4.0 * Aq * Cq
        if disc < 0: return []
        s = math.sqrt(max(0.0, disc))
        return [(-Bq - s) / (2.0 * Aq), (-Bq + s) / (2.0 * Aq)]

    def roots_y_of_x(x):
        Aq = Byy
        Bq = 2.0 * (Dxy * x + Hy)
        Cq = Axx * x * x + 2.0 * Gx * x + J
        if abs(Aq) < eps:
            if abs(Bq) < eps: return []
            return [-Cq / Bq]
        disc = Bq * Bq - 4.0 * Aq * Cq
        if disc < 0: return []
        s = math.sqrt(max(0.0, disc))
        return [(-Bq - s) / (2.0 * Aq), (-Bq + s) / (2.0 * Aq)]

    def x_plane(y, z):
        if abs(a) < eps: return None
        return (-b * y - c * z - d) / a

    def y_plane(x, z):
        if abs(b) < eps: return None
        return (-a * x - c * z - d) / b

    # sample points on the face
    F   = (A + B + C) / 3.0
    MAB = (A + B) / 2.0
    MBC = (B + C) / 2.0
    MCA = (C + A) / 2.0
    samples = [F, MAB, MBC, MCA]

    prefer_x = abs(a) >= abs(b)  # choose the more stable plane solve

    def thicknesses_along_x():
        ts = []
        for P in samples:
            y, z = float(P[1]), float(P[2])
            xp = x_plane(y, z)
            if xp is None: return None
            xr = roots_x_of_y(y)
            if not xr: return None
            t = min(abs(xp - xr[0]), abs(xp - xr[-1])) if len(xr) == 2 else abs(xp - xr[0])
            ts.append(t)
        tF, tAB, tBC, tCA = ts
        hA = max(0.0, (tF + tAB + tCA) / 3.0)
        hB = max(0.0, (tF + tAB + tBC) / 3.0)
        hC = max(0.0, (tF + tBC + tCA) / 3.0)
        return hA, hB, hC

    def thicknesses_along_y():
        ts = []
        for P in samples:
            x, z = float(P[0]), float(P[2])
            yp = y_plane(x, z)
            if yp is None: return None
            yr = roots_y_of_x(x)
            if not yr: return None
            t = min(abs(yp - yr[0]), abs(yp - yr[-1])) if len(yr) == 2 else abs(yp - yr[0])
            ts.append(t)
        tF, tAB, tBC, tCA = ts
        hA = max(0.0, (tF + tAB + tCA) / 3.0)
        hB = max(0.0, (tF + tAB + tBC) / 3.0)
        hC = max(0.0, (tF + tBC + tCA) / 3.0)
        return hA, hB, hC

    hs = thicknesses_along_x() if prefer_x else thicknesses_along_y()
    if hs is None:
        hs = thicknesses_along_y() if prefer_x else thicknesses_along_x()

    if (hs is None) or (hs[0] + hs[1] + hs[2] <= eps):
        v = V_patch / 3.0
        return v, v, v

    hA, hB, hC = hs
    S = hA + hB + hC
    wA, wB, wC = hA / S, hB / S, hC / S
    return wA * V_patch, wB * V_patch, wC * V_patch


# -------------------- tests with V_patch = 1e-4 --------------------

def assert_close(a, b, tol=1e-12, msg=""):
    if abs(a - b) > tol * max(1.0, abs(b)):
        raise AssertionError(f"{msg}  got={a:.16e}, want={b:.16e}")

def print_case(name, V_patch, A, B, C, coeffs):
    VA, VB, VC = split_patch_volume_thickness_weighted(V_patch, A, B, C, coeffs)
    S = VA + VB + VC
    wA = VA / V_patch if V_patch != 0 else float("nan")
    wB = VB / V_patch if V_patch != 0 else float("nan")
    wC = VC / V_patch if V_patch != 0 else float("nan")
    print(f"\n=== {name} ===")
    print(f"A={A}, B={B}, C={C}")
    print(f"V_patch={V_patch:.6e}  ->  VA={VA:.6e}, VB={VB:.6e}, VC={VC:.6e}  (sum={S:.6e})")
    print(f"weights  wA={wA:.6f}, wB={wB:.6f}, wC={wC:.6f}")
    assert_close(S, V_patch, 1e-12, msg=f"{name}: VA+VB+VC must equal V_patch")
    return VA, VB, VC

def run_tests():
    V_patch = 1e-4

    # 1) Cylinder: x^2 + y^2 = 1
    coeffs_cyl = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([math.cos(0.6), math.sin(0.6), 0.25])
    C = np.array([math.cos(1.0), math.sin(1.0), 0.55])
    print_case("Cylinder (R=1)", V_patch, A, B, C, coeffs_cyl)

    # 2) Hyperbola cylinder: x^2 - y^2 = 1
    coeffs_hyp = (1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    A = np.array([math.sqrt(1.0 + 0.0**2), 0.0, 0.0])
    B = np.array([math.sqrt(1.0 + 0.5**2), 0.5, 0.30])
    C = np.array([math.sqrt(1.0 + 1.0**2), 1.0, 0.60])
    print_case("Hyperbola cylinder (x^2 - y^2 = 1)", V_patch, A, B, C, coeffs_hyp)

    # 3) Parabola cylinder: y = x^2  → x^2 - y = 0 ⇒ A=1, H=-1/2
    coeffs_par = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0)
    A = np.array([0.0, 0.0, 0.00])
    B = np.array([0.70, 0.70**2, 0.25])
    C = np.array([1.10, 1.10**2, 0.55])
    print_case("Parabola cylinder (y = x^2)", V_patch, A, B, C, coeffs_par)

    # 4) Sphere at z=0 (forces equal-thirds fallback; plane solve ill-conditioned for x/y)
    coeffs_sph = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([math.sqrt(1.0 - 0.30**2), 0.30, 0.0])
    C = np.array([math.sqrt(1.0 - 0.60**2), 0.60, 0.0])
    VA, VB, VC = print_case("Sphere (equal-thirds fallback at z=0)", V_patch, A, B, C, coeffs_sph)
    assert_close(VA, V_patch/3.0, 1e-10, "Sphere fallback: VA")
    assert_close(VB, V_patch/3.0, 1e-10, "Sphere fallback: VB")
    assert_close(VC, V_patch/3.0, 1e-10, "Sphere fallback: VC")

    print("\nAll tests passed ✅")

if __name__ == "__main__":
    run_tests()
