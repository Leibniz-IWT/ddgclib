import os
import numpy as np
import meshio
import pandas as pd
from scipy.optimize import root
import multiprocessing
import time 

# ====== Spheroid Parameters ======
geo_name = "Ellip_0_sub0.geo"
surf_mesh_name = "Ellip_0_sub0_surf.msh"
tet_mesh_name  = "Ellip_0_sub0_tet.msh"
ax, ay, az = 1.5, 1.0, 0.8
coeffs = [1/ax**2, 1/ay**2, 1/az**2, 0, 0, 0, 0, 0, 0, -1]
V_theory = 4/3 * np.pi * ax * ay * az

def grad_F(x, y, z, coeffs):
    A, B, C, D, E, F, G, H, I, J = coeffs
    Fx = 2*A*x + D*y + E*z + G
    Fy = 2*B*y + D*x + F*z + H
    Fz = 2*C*z + E*x + F*y + I
    return np.array([Fx, Fy, Fz])

def F(x, y, z, coeffs):
    A, B, C, D, E, F_, G, H, I, J = coeffs
    return (A*x**2 + B*y**2 + C*z**2 + D*x*y + E*x*z + F_*y*z + G*x + H*y + I*z + J)

def normal_projection_to_surface(P0, coeffs, t0=1.0):
    grad = grad_F(*P0, coeffs)
    n = grad / np.linalg.norm(grad)
    def f_t(t):
        P = P0 + t*n
        return F(P[0], P[1], P[2], coeffs)
    sol = root(f_t, t0)
    t_star = sol.x[0]
    P_proj = P0 + t_star * n
    return P_proj

def on_spheroid_surface(P, tol=1e-4):
    x, y, z = P
    return abs(x**2/ax**2 + y**2/ay**2 + z**2/az**2 - 1) < tol

def tet_volume(pts):
    p0, p1, p2, p3 = pts
    return abs(np.dot(p1-p0, np.cross(p2-p0, p3-p0)))/6

def patch_area_and_volume(V0, V1, V2, coeffs):
    area = np.linalg.norm(np.cross(V1-V0, V2-V0)) / 2
    P0 = normal_projection_to_surface(V0, coeffs)
    P1 = normal_projection_to_surface(V1, coeffs)
    P2 = normal_projection_to_surface(V2, coeffs)
    tets = [
        [V0, V1, V2, P1],
        [P0, P1, P2, V0],
        [V0, V2, P1, P2]
    ]
    volume = sum(tet_volume(np.array(tet)) for tet in tets)
    return area, volume

def adaptive_patch_integrate(V0, V1, V2, coeffs, tol=1e-5, depth=0, max_depth=8):
    area, volume = patch_area_and_volume(V0, V1, V2, coeffs)
    if area < 1e-14:
        return [(area, volume)]
    mid01 = (V0 + V1) / 2
    mid12 = (V1 + V2) / 2
    mid20 = (V2 + V0) / 2
    Pm01 = normal_projection_to_surface(mid01, coeffs)
    Pm12 = normal_projection_to_surface(mid12, coeffs)
    Pm20 = normal_projection_to_surface(mid20, coeffs)
    flat_mids = [mid01, mid12, mid20]
    curved_mids = [Pm01, Pm12, Pm20]
    err = max(np.linalg.norm(cm - fm) for cm, fm in zip(curved_mids, flat_mids))
    if (area > tol or err > tol*5) and depth < max_depth:
        return (
            adaptive_patch_integrate(V0, mid01, mid20, coeffs, tol, depth+1, max_depth) +
            adaptive_patch_integrate(mid01, V1, mid12, coeffs, tol, depth+1, max_depth) +
            adaptive_patch_integrate(mid20, mid12, V2, coeffs, tol, depth+1, max_depth) +
            adaptive_patch_integrate(mid01, mid12, mid20, coeffs, tol, depth+1, max_depth)
        )
    else:
        return [(area, volume)]

def intersect_ray_spheroid(P0, direction, coeffs):
    A, B, C, D, E, F_, G, H, I, J = coeffs
    x0, y0, z0 = P0
    nx, ny, nz = direction
    a = A*nx**2 + B*ny**2 + C*nz**2
    b = 2*A*x0*nx + 2*B*y0*ny + 2*C*z0*nz
    c = A*x0**2 + B*y0**2 + C*z0**2 + J
    roots = np.roots([a, b, c])
    return roots

def sample_points_triangle(A, B, C, n_samples=5):
    points = []
    for i in range(n_samples):
        for j in range(n_samples - i):
            u = i / (n_samples - 1)
            v = j / (n_samples - 1)
            w = 1 - u - v
            if w < 0 or w > 1: continue
            P = u * A + v * B + w * C
            points.append(P)
    return points

def adaptive_triangle_worker(args):
    idx, tri, surf_points, coeffs, tol, max_depth = args
    A, B, C = surf_points[tri]
    if (on_spheroid_surface(A) and on_spheroid_surface(B) and on_spheroid_surface(C)):
        result = adaptive_patch_integrate(A, B, C, coeffs, tol, max_depth=max_depth)
        total_area = sum(x[0] for x in result)
        total_volume = sum(x[1] for x in result)
        V_flat = abs(np.dot(A, np.cross(B, C))) / 6
        correction = total_volume - V_flat
        flat_area = np.linalg.norm(np.cross(B - A, C - A)) / 2

        G = (A + B + C) / 3
        n = np.cross(B - A, C - A)
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            n = np.array([0.0, 0.0, 1.0])  # fallback
        else:
            n = n / n_norm
        sampled_points = sample_points_triangle(A, B, C, n_samples=5)
        max_t = -np.inf
        F = None
        for P in sampled_points:
            t_roots = intersect_ray_spheroid(P, n, coeffs)
            t_pos = [t.real for t in t_roots if np.isreal(t) and t.real > 0]
            if t_pos:
                t_here = min(t_pos)
                if t_here > max_t:
                    max_t = t_here
                    F = P + t_here * n
        if F is None:
            F = G  # fallback: use centroid

        V_abcf = abs(np.dot(B - A, np.cross(C - A, F - A))) / 6
        ratio = total_volume / V_abcf if V_abcf != 0 else np.nan

        t_g_roots = intersect_ray_spheroid(G, n, coeffs)
        t_g_positive = [t.real for t in t_g_roots if np.isreal(t) and t.real > 0]
        if len(t_g_positive) == 0:
            G_proj = G
        else:
            t_g = min(t_g_positive)
            G_proj = G + t_g * n

        V_abcg = abs(np.dot(B - A, np.cross(C - A, G_proj - A))) / 6
        ratioG = total_volume / V_abcg if V_abcg != 0 else np.nan

        AB = np.linalg.norm(A - B)
        BC = np.linalg.norm(B - C)
        CA = np.linalg.norm(C - A)
        min_len = min(AB, BC, CA)
        norm_AB = AB / min_len
        norm_BC = BC / min_len
        norm_CA = CA / min_len
        org_ratio_len = f"{AB:.8f}:{BC:.8f}:{CA:.8f}"
        ratio_len = f"{norm_AB:.8f}:{norm_BC:.8f}:{norm_CA:.8f}"
        sum_norm_len = norm_AB + norm_BC + norm_CA

        # === Spherical triangle wedge (Vcap) calculation with user spherical_angle function ===
        def to_sphere(P):
            return np.array([P[0] / ax, P[1] / ay, P[2] / az])
        a = to_sphere(A) / np.linalg.norm(to_sphere(A))
        b = to_sphere(B) / np.linalg.norm(to_sphere(B))
        c = to_sphere(C) / np.linalg.norm(to_sphere(C))

        def spherical_angle(va, vb, vc):
            # angle at va between vb and vc (all normalized)
            ab = np.arccos(np.clip(np.dot(va, vb), -1, 1))
            ac = np.arccos(np.clip(np.dot(va, vc), -1, 1))
            bc = np.arccos(np.clip(np.dot(vb, vc), -1, 1))
            return np.arccos((np.cos(ab) - np.cos(ac) * np.cos(bc)) / (np.sin(ac) * np.sin(bc) + 1e-14))
        
        alpha = spherical_angle(a, b, c)
        beta  = spherical_angle(b, c, a)
        gamma = spherical_angle(c, a, b)

        Omega = alpha + beta + gamma - np.pi
        Vwedge = (1.0 / 3.0) * Omega
        O = np.array([0.0, 0.0, 0.0])
        Vflat = abs(np.dot(a - O, np.cross(b - O, c - O))) / 6.0
        Vcap = (Vwedge - Vflat) * (ax * ay * az)
        print(f"Triangle {idx}: Vcap = {Vcap:.8f}, Omega = {Omega:.8f}, Vwedge = {Vwedge:.8f}, Vflat = {Vflat:.8f}")
        row = {
            'triangle_id': idx,
            'curved_patch_volume': total_volume,
            'Vcap': Vcap,           
            'flat_volume': V_flat,
            'correction': correction,
            'patch_area': total_area,
            'flat_area': flat_area,
            'tetra_volume_abcf': V_abcf,
            'ratio': ratio,
            'tetra_volume_abcg': V_abcg,
            'ratioG': ratioG,
            'org length_ratio': org_ratio_len,
            'length_ratio': ratio_len,
            'length_ratio_sum': f"{sum_norm_len:.8f}",
            'A_x': A[0], 'A_y': A[1], 'A_z': A[2],
            'B_x': B[0], 'B_y': B[1], 'B_z': B[2],
            'C_x': C[0], 'C_y': C[1], 'C_z': C[2],
            'F_x': F[0], 'F_y': F[1], 'F_z': F[2],
            'G_proj_x': G_proj[0], 'G_proj_y': G_proj[1], 'G_proj_z': G_proj[2],
        }
        return row, total_volume
    else:
        row = {
            'triangle_id': idx,
            'curved_patch_volume': np.nan,
            'flat_volume': np.nan,
            'correction': np.nan,
            'patch_area': np.nan,
            'flat_area': np.nan,
            'tetra_volume_abcf': np.nan,
            'ratio': np.nan,
            'tetra_volume_abcg': np.nan,
            'ratioG': np.nan,
            'org length_ratio': np.nan,
            'length_ratio': np.nan,
            'length_ratio_sum': np.nan,
            'Vcap': np.nan,
            'A_x': A[0], 'A_y': A[1], 'A_z': A[2],
            'B_x': B[0], 'B_y': B[1], 'B_z': B[2],
            'C_x': C[0], 'C_y': C[1], 'C_z': C[2],
            'F_x': np.nan, 'F_y': np.nan, 'F_z': np.nan,
            'G_proj_x': np.nan, 'G_proj_y': np.nan, 'G_proj_z': np.nan,
        }
        return row, 0.0

if __name__ == "__main__":
    surf_mesh = meshio.read(surf_mesh_name)
    surf_points = surf_mesh.points
    surf_faces = []
    for cell_block in surf_mesh.cells:
        if cell_block.type in ("triangle", "triangle6"):
            surf_faces = cell_block.data
            break
    if len(surf_faces) == 0:
        raise ValueError("No triangle faces found in this mesh!")

    tet_mesh = meshio.read(tet_mesh_name)
    tet_points = tet_mesh.points
    tet_cells = None
    for cell_block in tet_mesh.cells:
        if cell_block.type in ("tetra", "tet"):
            tet_cells = cell_block.data
            break
    if tet_cells is None:
        raise ValueError("No tetrahedra found in this mesh!")

    V_piecewise = 0.0
    for tet in tet_cells:
        pts4 = tet_points[tet]
        V_piecewise += tet_volume(pts4)

    tol = 1e-4
    max_depth = 6
    args_list = [(idx, tri, surf_points, coeffs, tol, max_depth) for idx, tri in enumerate(surf_faces)]

    print(f"Using {multiprocessing.cpu_count()} CPU cores...")

    t0 = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(adaptive_triangle_worker, args_list)

    csv_rows = []
    V_patch_sum = 0.0
    for row, volume in results:
        csv_rows.append(row)
        V_patch_sum += volume

    df = pd.DataFrame(csv_rows)
    df.to_csv("adaptive_triangle_patch_correction.csv", index=False)
    print("\nWrote triangle patch corrections to adaptive_triangle_patch_correction.csv")

    t1 = time.time()
    rel_error_piecewise = abs(V_piecewise - V_theory) / V_theory * 100
    rel_error_patchsum = abs(V_patch_sum + V_piecewise - V_theory) / V_theory * 100

    print(f"\n==== VOLUME SUMMARY ====")
    print(f"Theoretical value (spheroid):           {V_theory:.8f}")
    print(f"Piecewise sum of all tets:              {V_piecewise:.8f}")
    print(f"Patch-based total (curved patch sum):   {V_patch_sum:.8f}")
    print(f"Number of spheroid surface triangles:   {len(surf_faces)}")
    print(f"Relative error (piecewise tets):        {rel_error_piecewise:.6f}%")
    print(f"Relative error (curved patch sum):      {rel_error_patchsum:.6f}%")
    print(f"Elapsed time (patch sum):               {t1-t0:.2f} s")
