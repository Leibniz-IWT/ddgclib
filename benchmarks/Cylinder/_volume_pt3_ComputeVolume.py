import numpy as np
import meshio
import pandas as pd
import multiprocessing
from tqdm import tqdm
from scipy.optimize import root

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
    return P0 + t_star * n

def tet_volume(pts):
    p0, p1, p2, p3 = pts
    return abs(np.dot(p1-p0, np.cross(p2-p0, p3-p0))) / 6

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
    total_volume = sum(x[1] for x in result)
    return total_volume

def compute_volume_error(tet_mesh_name, fit_csv, V_theory, tol=1e-5, max_depth=8):
    """
    Given:
        tet_mesh_name: .msh file with both tetrahedra and surface triangles
        fit_csv: CSV file with fitted quadric coefficients per vertex
        V_theory: analytic volume
    Returns:
        dictionary with volume errors and mesh stats
    """
    # Load mesh
    mesh = meshio.read(tet_mesh_name)
    points = mesh.points
    tets = None
    tris = None
    for c in mesh.cells:
        if c.type in ("tetra", "tet"):
            tets = c.data
        if c.type in ("triangle", "triangle6"):
            tris = c.data
    if tets is None or tris is None:
        raise ValueError("Tets or triangles not found in mesh file!")

    # Piecewise tet volume
    V_piecewise = np.sum([tet_volume(points[tet]) for tet in tets])

    # Patch-based (curved) surface triangle integration
    fit_df = pd.read_csv(fit_csv)
    args_list = [(idx, tri, points, fit_df, tol, max_depth) for idx, tri in enumerate(tris)]
    V_patch_sum = 0.0
    with multiprocessing.Pool() as pool:
        for volume in tqdm(pool.imap_unordered(adaptive_triangle_worker, args_list), total=len(args_list), desc="Curved patch sum"):
            V_patch_sum += volume

    rel_error_piecewise = abs(V_piecewise - V_theory) / V_theory * 100
    rel_error_patchsum = abs(V_patch_sum + V_piecewise - V_theory) / V_theory * 100

    return {
        'rel_error_piecewise': rel_error_piecewise,
        'rel_error_patchsum': rel_error_patchsum,
        'V_piecewise': V_piecewise,
        'V_patch_sum': V_patch_sum,
        'num_tet': len(tets),
        'num_tris': len(tris)
    }

# Example usage in main.py:
if __name__ == "__main__":
    tet_mesh_name = "CylinderSymm_0_tet.msh"
    fit_csv = "CylinderSymm_0_fit_tet60_unified.csv"
    radius = 1.0
    height = 2.0
    V_theory = np.pi * radius**2 * height
    result = compute_volume_error(tet_mesh_name, fit_csv, V_theory)
    print(result)
