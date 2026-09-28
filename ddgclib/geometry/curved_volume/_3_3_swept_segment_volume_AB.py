#!/usr/bin/env python3
# coding: utf-8

import math
import numpy as np

# small helpers
def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def _parse_quadric_coeffs(coeffs):
    """
    coeffs = (A,B,C,D,E,F,G,H,I,J) for:
      x^2*A + y^2*B + z^2*C + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0
    Returns (A3, b, J) where A3 is 3x3 symmetric matrix, b is 3-vector.
    """
    A,B,C,D,E,F,G,H,I,J = coeffs
    A3 = np.array([[A, D, E],
                   [D, B, F],
                   [E, F, C]], dtype=float)
    b  = np.array([G, H, I], dtype=float)
    return A3, b, float(J)

def _quadric_center(A3, b):
    """
    Center x0 such that A3 x0 + b = 0. Uses solve or pseudoinverse.
    (For paraboloids A3 is singular; pseudoinverse gives a least-norm solution.)
    """
    try:
        return -np.linalg.solve(A3, b)
    except np.linalg.LinAlgError:
        return -np.linalg.pinv(A3) @ b

def _axis_transform_from_coeffs(coeffs, tol=1e-8):
    """
    Compute center x0, orthonormal frame U=[u,v,k] where k is revolution axis (if exists),
    and identify (lambda_rad, mu_ax) = (radial eigenvalue, axial eigenvalue).
    Returns: ok, x0, U, lambda_rad, mu_ax
    - ok=False if not axisymmetric (no two equal eigenvalues within tol), but still provides x0,U.
    """
    A3, b, J = _parse_quadric_coeffs(coeffs)
    x0 = _quadric_center(A3, b)

    # eigendecompose A3
    w, V = np.linalg.eigh(A3)
    idx = np.argsort(w); w = w[idx]; V = V[:, idx]
    scale = max(1.0, np.linalg.norm(w))

    # determine axis as the eigenvector of the distinct eigenvalue (two equal, one distinct)
    if abs(w[0]-w[1]) < tol*scale and abs(w[1]-w[2]) >= 10*tol*scale:
        # w0≈w1 (radial), w2 (axial)
        u = V[:,0]; v = V[:,1]; k = V[:,2]
        lam_rad, mu_ax = w[0], w[2]
        ok = True
    elif abs(w[1]-w[2]) < tol*scale and abs(w[0]-w[1]) >= 10*tol*scale:
        # w1≈w2 (radial), w0 (axial)
        u = V[:,1]; v = V[:,2]; k = V[:,0]
        lam_rad, mu_ax = w[1], w[0]
        ok = True
    elif abs(w[0]-w[1]) < tol*scale and abs(w[1]-w[2]) < tol*scale:
        # fully isotropic (sphere): any orthonormal frame works; treat axis = z of V
        u = V[:,0]; v = V[:,1]; k = V[:,2]
        lam_rad = mu_ax = w[0]
        ok = True
    else:
        # not axisymmetric; still provide some frame
        u = V[:,0]; v = V[:,1]; k = V[:,2]
        lam_rad, mu_ax = np.nan, np.nan
        ok = False

    # ensure U is right-handed
    U = np.column_stack([normalize(u), normalize(v), normalize(k)])
    if np.linalg.det(U) < 0:
        U[:,1] *= -1.0  # flip v

    return ok, x0, U, float(lam_rad), float(mu_ax)

def _polygon_area_centroid(poly_xy):
    """
    Shoelace formula for a simple polygon in the plane.
    Returns (|Area|, Cx, Cy). Centroid uses the signed area internally.
    """
    poly_xy = np.asarray(poly_xy, float)
    x = poly_xy[:,0]; y = poly_xy[:,1]
    x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    cross = x*y1 - y*x1
    A_signed = 0.5 * np.sum(cross)
    if abs(A_signed) < 1e-18:
        return 0.0, 0.0, 0.0
    Cx = (1.0/(6.0*A_signed)) * np.sum((x + x1) * cross)
    Cy = (1.0/(6.0*A_signed)) * np.sum((y + y1) * cross)
    return abs(A_signed), Cx, Cy

# core function (z-only API)
def swept_segment_volume_AB(coeffs_1h, z_A, z_B, N=4096):
    """
    1H (1×) INPUT:
      f = A x² + B y² + C z² + D xy + E xz + F yz + G x + H y + I z + J = 0

    Principal-frame meridian model:
      lam * r² + mu * z² + q*z + Jc = 0, with q = 1× axial linear coeff
      (Internally: matrix form uses x^T A x + 2 b·x + J, so b = 0.5*[G,H,I])
    """
    ok, _x0, U, lam_rad, mu_ax = _axis_transform_from_coeffs(coeffs_1h)  # already 1H
    if not ok:
        return 0.0, "no-axis"

    # 1H parse to matrix form x^T A x + 2 b·x + J
    A,B,C,D,E,F,G,H,I,J = map(float, coeffs_1h)
    A3 = np.array([[A,   D/2, E/2],
                   [D/2, B,   F/2],
                   [E/2, F/2, C   ]], float)  # 1H: off-diagonals are D/2, E/2, F/2
    b  = 0.5 * np.array([G, H, I], float)      # 1H: b = 0.5*[G,H,I] so 2 b·x = Gx+Hy+Iz

    bu, bv, bk = (U.T @ b).astype(float)

    eps = 1e-15
    if abs(lam_rad) < eps:
        return 0.0, "no-axis"
    Jc = J - (bu*bu + bv*bv) / lam_rad

    if abs(mu_ax) >= eps:
        z_shift = bk / mu_ax
        Jp = Jc - (bk*bk) / mu_ax
        def r2_of_z(z):
            zp = z + z_shift
            return -(mu_ax*zp*zp + Jp) / lam_rad
    else:
        # paraboloid-like: lam r² + q*z + Jc = 0 with q = 2*bk (1H axial linear coeff)
        q = 2.0 * bk
        def r2_of_z(z):
            return -(q*z + Jc) / lam_rad

    M = max(8, int(N))
    zs = np.linspace(z_A, z_B, M)
    r2 = r2_of_z(zs)
    valid = r2 > 0
    if not np.any(valid):
        return 0.0, "meridian"

    rs = np.sqrt(r2[valid])
    zs_valid = zs[valid]

    rA = math.sqrt(max(0.0, r2_of_z(z_A)))
    rB = math.sqrt(max(0.0, r2_of_z(z_B)))

    arc_rz = np.column_stack([rs, zs_valid])
    poly_rz = np.vstack([arc_rz, [rB, z_B], [rA, z_A]])

    A_area, C_r, _ = _polygon_area_centroid(poly_rz)
    if A_area <= 0.0 or not np.isfinite(C_r):
        return 0.0, "meridian"

    V = 2.0 * math.pi * abs(C_r) * A_area
    return V, "meridian"
 
def swept_segment_volume_A1B(coeffs, z_A, z_B, N=4096):
    """
    Compute the 'wedge' volume obtained by revolving the planar area enclosed by
    the meridian arc r(z) and the straight chord connecting its endpoints
    about the revolution axis (Pappus).

    Supports axisymmetric quadrics in principal frame:
      lam * r^2 + mu * z^2 + 2*p*z + Jc = 0
    where:
      lam = radial eigenvalue (double),
      mu  = axial eigenvalue,
      p   = linear term along axis (component of b along axis in principal frame),
      Jc  = constant adjusted by removing radial linear terms.

    Returns:
        (V, "meridian") on success
        (0.0, "no-axis") if not axisymmetric
        (0.0, "meridian") if no valid radii on [z_A, z_B]
    """
    # eigenframe / axis detection
    ok, _x0_unused, U, lam_rad, mu_ax = _axis_transform_from_coeffs(coeffs)
    if not ok:
        return 0.0, "no-axis"

    A3, b, J = _parse_quadric_coeffs(coeffs)
    # rotate linear terms into principal frame (u,v,k)
    bu, bv, bk = (U.T @ b).astype(float)

    # remove radial linear terms by translating in (u,v); adjust constant
    eps = 1e-15
    if abs(lam_rad) < eps:
        return 0.0, "no-axis"  # degenerate radial quadric
    Jc = J - (bu*bu + bv*bv) / lam_rad

    # r^2 formula in principal frame
    if abs(mu_ax) >= 1e-15:
        # complete the square in z: z' = z + bk/mu, J' = Jc - bk^2/mu
        z_shift = bk / mu_ax
        Jp = Jc - (bk*bk) / mu_ax
        def r2_of_z(z):
            zp = z + z_shift
            return -(mu_ax*zp*zp + Jp) / lam_rad
    else:
        # paraboloid-like: lam*r^2 + 2*bk*z + Jc = 0
        def r2_of_z(z):
            return -(2.0*bk*z + Jc) / lam_rad

    # sample r(z) along the requested interval
    M = max(8, int(N))
    zs = np.linspace(z_A, z_B, M)
    r2 = r2_of_z(zs)
    valid = r2 > 0
    if not np.any(valid):
        return 0.0, "meridian"

    rs = np.sqrt(r2[valid])
    zs_valid = zs[valid]

    # endpoints (allow zero clamp)
    rA = math.sqrt(max(0.0, r2_of_z(z_A)))
    rB = math.sqrt(max(0.0, r2_of_z(z_B)))

    # polygon in (r,z): arc + closing chord (B->A)
    arc_rz = np.column_stack([rs, zs_valid])
    poly_rz = np.vstack([arc_rz, [rB, z_B], [rA, z_A]])

    A_area, C_r, _ = _polygon_area_centroid(poly_rz)
    if A_area <= 0.0 or not np.isfinite(C_r):
        return 0.0, "meridian"

    V = 2.0 * math.pi * abs(C_r) * A_area
    return V, "meridian"

if __name__ == "__main__":
    # Hyperboloid check (symmetric)
    # x^2 + y^2 - z^2/c^2 = 1  (J = -1)
    c = 1.0
    COEFFS = (1.0, 1.0, -1.0/(c**2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    h = 1.0
    z_A, z_B = +h, -h
    V_num, mode = swept_segment_volume_AB(COEFFS, z_A, z_B, N=8192)
    V_exact = (4.0*math.pi/3.0) * (h**3) / (c**2)  # analytic wedge
    print("--- Meridian mode (hyperboloid) ---")
    print("Mode:", mode)
    print(f"z_A={z_A:.6f}, z_B={z_B:.6f}, c={c:.6f}")
    print(f"Numerical V: {V_num:.12f}")
    print(f"Exact V:     {V_exact:.12f}  (= 4π/3 * h^3 / c^2)")
    print(f"Abs error:   {abs(V_num - V_exact):.3e}")

    # Second hyperboloid params
    h2, c2 = 1.75, 1.5
    COEFFS2 = (1.0, 1.0, -1.0/(c2**2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    z_A2, z_B2 = +h2, -h2
    V_num2, mode2 = swept_segment_volume_AB(COEFFS2, z_A2, z_B2, N=8192)
    V_exact2 = (4.0*math.pi/3.0) * (h2**3) / (c2**2)
    print("\n-- Second case (hyperboloid) --")
    print("Mode:", mode2)
    print(f"z_A2={z_A2:.6f}, z_B2={z_B2:.6f}, c2={c2:.6f}")
    print(f"Numerical V: {V_num2:.12f}")
    print(f"Exact V:     {V_exact2:.12f}")
    print(f"Abs error:   {abs(V_num2 - V_exact2):.3e}")

    # Paraboloid test (requested COEFFS)
    # x^2 + y^2 - z = 0  ->  (A,B,C,D,E,F,G,H,I,J) = (1,1,0,0,0,0,0,0,-0.5,0)
    COEFFS_P = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0)

    # Take the meridian segment from z=0 to z=H.
    # For this case, r(z)^2 = z, so the wedge (curve vs straight chord) has closed-form:
    # V_exact = ∫_0^H π [r(z)^2 - r_chord(z)^2] dz = (π/6) H^2
    H = 2.0
    zA_p, zB_p = H, 0.0  # any order ok
    Vp_num, mode_p = swept_segment_volume_AB(COEFFS_P, zA_p, zB_p, N=8192)
    Vp_exact = (math.pi / 6.0) * (H**2)

    print("\n--- Meridian mode (paraboloid) ---")
    print("Mode:", mode_p)
    print(f"H={H:.6f}  z_A={zA_p:.6f}, z_B={zB_p:.6f}")
    print(f"Numerical V: {Vp_num:.12f}")
    print(f"Exact V:     {Vp_exact:.12f}  (= π/6 * H^2)")
    print(f"Abs error:   {abs(Vp_num - Vp_exact):.3e}")
