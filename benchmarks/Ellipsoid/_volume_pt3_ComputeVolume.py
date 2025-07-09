import os
import numpy as np
import meshio
import pandas as pd
from scipy.optimize import root
import multiprocessing
import time
from tqdm import tqdm

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

def get_vertex_coeffs(idx, fit_df):
    row = fit_df.loc[fit_df['vertex_index'] == idx]
    if row.empty and fit_df['vertex_index'].dtype == float:
        row = fit_df.loc[fit_df['vertex_index'] == float(idx)]
    if row.empty:
        raise ValueError(f"Vertex {idx} not found in coefficient table")
    return row[['A','B','C','D','E','F','G','H','I','J']].values.astype(float)[0]

def get_patch_coeffs(indices, fit_df):
    coeffs = [get_vertex_coeffs(idx, fit_df) for idx in indices]
    return np.mean(coeffs, axis=0)

def adaptive_triangle_worker(args):
    idx, tri, surf_points, fit_df, tol, max_depth = args
    A, B, C = surf_points[tri]
    coeffs = get_patch_coeffs(tri, fit_df)
    result = adaptive_patch_integrate(A, B, C, coeffs, tol, max_depth=max_depth)
    total_area = sum(x[0] for x in result)
    total_volume = sum(x[1] for x in result)
    V_flat = abs(np.dot(A, np.cross(B, C))) / 6
    correction = total_volume - V_flat
    row = {
        'triangle_id': idx,
        'curved_patch_volume': total_volume,
        'flat_volume': V_flat,
        'correction': correction,
        'patch_area': total_area,
        'A_x': A[0], 'A_y': A[1], 'A_z': A[2],
        'B_x': B[0], 'B_y': B[1], 'B_z': B[2],
        'C_x': C[0], 'C_y': C[1], 'C_z': C[2]
    }
    return row, total_volume

def compute_volume_error(tet_mesh_name, surf_mesh_name, fit_csv, V_theory, prefix="", tol=1e-5, max_depth=8):
    # === Load Surface Mesh (Triangles only) ===
    surf_mesh = meshio.read(surf_mesh_name)
    surf_points = surf_mesh.points
    surf_faces = []
    for cell_block in surf_mesh.cells:
        if cell_block.type in ("triangle", "triangle6"):
            surf_faces = cell_block.data
            break
    if len(surf_faces) == 0:
        raise ValueError("No triangle faces found in this mesh!")

    # === Load Tetrahedral Mesh ===
    tet_mesh = meshio.read(tet_mesh_name)
    tet_points = tet_mesh.points
    tet_cells = None
    for cell_block in tet_mesh.cells:
        if cell_block.type in ("tetra", "tet"):
            tet_cells = cell_block.data
            break
    if tet_cells is None:
        raise ValueError("No tetrahedra found in this mesh!")

    # ========== Compute Piecewise Tet Volume ==========
    V_piecewise = 0.0
    for tet in tet_cells:
        pts4 = tet_points[tet]
        V_piecewise += tet_volume(pts4)

    # ==== Load unified quadric CSV ====
    fit_df = pd.read_csv(fit_csv)

    args_list = [(idx, tri, surf_points, fit_df, tol, max_depth) for idx, tri in enumerate(surf_faces)]

    print(f"Using {multiprocessing.cpu_count()} CPU cores...")

    t0 = time.time()
    results = []
    with multiprocessing.Pool() as pool:
        for r in tqdm(pool.imap_unordered(adaptive_triangle_worker, args_list), total=len(args_list), desc="Integrating triangles"):
            results.append(r)

    csv_rows = []
    V_patch_sum = 0.0
    for row, volume in results:
        csv_rows.append(row)
        V_patch_sum += volume

    df = pd.DataFrame(csv_rows)
    out_csv = f"{prefix}_adaptive_triangle_patch_correction.csv" if prefix else "adaptive_triangle_patch_correction.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote triangle patch corrections to {out_csv}")

    t1 = time.time()
    rel_error_piecewise = abs(V_piecewise - V_theory) / V_theory * 100
    rel_error_patchsum = abs(V_patch_sum + V_piecewise - V_theory) / V_theory * 100

    print(f"\n==== VOLUME SUMMARY ====")
    print(f"Theoretical value:                    {V_theory:.8f}")
    print(f"Piecewise sum of all tets:            {V_piecewise:.8f}")
    print(f"Patch-based total (curved patch sum): {V_patch_sum:.8f}")
    print(f"Number of surface triangles:          {len(surf_faces)}")
    print(f"Relative error (piecewise tets):      {rel_error_piecewise:.6f}%")
    print(f"Relative error (curved patch sum):    {rel_error_patchsum:.6f}%")
    print(f"Elapsed time (patch sum):             {t1-t0:.2f} s")

    # Return a summary dictionary for main.py
    return {
        'prefix': prefix,
        'num_tet': len(tet_cells),
        'rel_error_piecewise': rel_error_piecewise,
        'rel_error_patchsum': rel_error_patchsum,
        'elapsed_time_sec': t1-t0,
        'triangle_correction_csv': out_csv
    }

# Example for standalone testing:
if __name__ == "__main__":
    geo_name = "SpheroidSymm_sub0_sub0.geo"
    surf_mesh_name = "SpheroidSymm_sub0_sub0_surf.msh"
    tet_mesh_name  = "SpheroidSymm_sub0_sub0_tet.msh"
    ax, ay, az = 1.5, 1.0, 0.8
    V_theory = 4/3 * np.pi * ax * ay * az
    fit_csv = "Ellip_fit_tet19578_unified.csv"
    prefix = "Ellip_0"
    compute_volume_error(tet_mesh_name, surf_mesh_name, fit_csv, V_theory, prefix)
