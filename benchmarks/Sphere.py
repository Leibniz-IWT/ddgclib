import os
import numpy as np
import meshio
import pandas as pd
from scipy.optimize import root
from scipy.integrate import quad
import multiprocessing
import time

# ====== Cylinder Parameters ======
mesh_name = "CylinderSymm_0_tet.msh"
R = 1
H = 2
coeffs = [1/R**2, 1/R**2, 0, 0, 0, 0, 0, 0, 0, -1]  # For lateral surface: x^2/R^2 + y^2/R^2 = 1
V_theory = np.pi * R**2 * H

def grad_F(x, y, z, coeffs):
    A, B, C, D, E, F, G, H_, I, J = coeffs
    Fx = 2*A*x + D*y + E*z + G
    Fy = 2*B*y + D*x + F*z + H_
    Fz = 2*C*z + E*x + F*y + I
    return np.array([Fx, Fy, Fz])

def F(x, y, z, coeffs):
    A, B, C, D, E, F_, G, H_, I, J = coeffs
    return (A*x**2 + B*y**2 + C*z**2 + D*x*y + E*x*z + F_*y*z + G*x + H_*y + I*z + J)

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

def cylinder_angle(x, y, R):
    angle = np.arctan2(y, x)
    return angle if angle >= 0 else angle + 2*np.pi

def extend_arc_point_to_z(P1, P2, z_target, R):
    """Extend arc on cylinder from P1 to P2 to reach z_target (linear in z, arc in angle)."""
    theta1 = cylinder_angle(P1[0], P1[1], R)
    theta2 = cylinder_angle(P2[0], P2[1], R)
    dz = P2[2] - P1[2]
    if abs(dz) < 1e-12:
        return np.array([R * np.cos(theta1), R * np.sin(theta1), z_target])
    t = (z_target - P1[2]) / dz
    theta = theta1 + t * (theta2 - theta1)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return np.array([x, y, z_target])

def wedge_volume_geodesic_ABCD(A, B, C, R):
    """
    Returns the volume correction for ABC, with D at z(C), D on geodesic AB (on the cylinder).
    Vcorrection = (V_cylinder - area_COD * h_full) / splittimes, where splittimes = 2pi / theta_CD * 2
    """
    # Sort points so A=top, B=bottom, C=middle
    zvals = [A[2], B[2], C[2]]
    idxs = np.argsort(zvals)
    top = [A, B, C][idxs[2]]
    mid = [A, B, C][idxs[1]]
    bot = [A, B, C][idxs[0]]
    A, B, C = top, bot, mid
    EPS = 1e-10
    h_full = abs(A[2] - B[2])
    # D is on AB geodesic at z = C[2]
    D = extend_arc_point_to_z(A, B, C[2], R)

    # Get angles (relative to center O)
    theta_C = cylinder_angle(C[0], C[1], R)
    theta_D = cylinder_angle(D[0], D[1], R)

    # Ensure theta_CD is always the minor arc [0, pi)
    theta_CD = (theta_D - theta_C) % (2 * np.pi)
    if theta_CD >= np.pi:
        theta_CD = 2 * np.pi - theta_CD
    # Triangle area COD (sector formula for triangle from center)
    #triangle_area_COD = 0.5 * R*np.cos(theta_CD/2) * np.sin(theta_CD/2)*R*2
    triangle_area_COD = 0.5 * R**2 * np.sin(theta_CD)
    V_t = h_full * np.pi * R**2
    splittimes = 2 * np.pi / theta_CD
    Vcorrection = (V_t/splittimes- triangle_area_COD * h_full)/2 

    # Special case: C at same z as A
    if abs(C[2] - A[2]) < EPS:
        theta_CA = (cylinder_angle(C[0], C[1], R) - cylinder_angle(A[0], A[1], R)) % (2 * np.pi)
        if theta_CA >= np.pi:
            theta_CA = 2 * np.pi - theta_CA
        triangle_area_COA = 0.5 * R**2 * np.sin(theta_CA)
        splittimes = 2 * np.pi / theta_CA
        V_t = h_full * np.pi * R**2
        Vcorrection = (V_t/splittimes - triangle_area_COA * h_full)/2 
        print(f"*theta_CD: {theta_CD:.6f}, triangle_area_COD: {triangle_area_COA:.6f}, Vcorrection: {Vcorrection:.6f},splittimes: {splittimes:.6f}, h_full: {h_full:.6f}")
        #return Vcorrection

    print(f"theta_CD: {theta_CD:.6f}, triangle_area_COD: {triangle_area_COD:.6f}, Vcorrection: {Vcorrection:.6f},splittimes: {splittimes:.6f}, h_full: {h_full:.6f}")
    return Vcorrection

def get_surface_triangle_set_from_msh(mesh_name):
    surface_tris = set()
    with open(mesh_name, "r") as f:
        lines = f.readlines()
    in_elem = False
    for line in lines:
        if line.strip() == "$Elements":
            in_elem = True
            continue
        if line.strip() == "$EndElements":
            break
        if in_elem:
            parts = line.strip().split()
            if len(parts) >= 7 and parts[1] == '2':  # triangle
                phys_tag = int(parts[3])
                if phys_tag in {2, 3, 4}:  # surface tags: lateral, bottom, top
                    tri = tuple(int(x) - 1 for x in parts[-3:])  # 1-based to 0-based
                    surface_tris.add(frozenset(tri))
    return surface_tris

def adaptive_triangle_worker(args):
    idx, tri, surf_points, coeffs, tol, max_depth, surface_tris = args
    is_surface_triangle = frozenset(tri) in surface_tris
    A, B, C = surf_points[tri]
    result = adaptive_patch_integrate(A, B, C, coeffs, tol, max_depth=max_depth)
    total_area = sum(x[0] for x in result)
    total_volume = sum(x[1] for x in result)
    V_flat = abs(np.dot(A, np.cross(B, C))) / 6
    correction = total_volume - V_flat
    flat_area = np.linalg.norm(np.cross(B - A, C - A)) / 2
    Vcorrection = wedge_volume_geodesic_ABCD(A, B, C, R)

    row = {
        'triangle_id': idx,
        'curved_patch_volume': total_volume,
        'Vcorrection': Vcorrection,
        'flat_volume': V_flat,
        'correction': correction,
        'patch_area': total_area,
        'flat_area': flat_area,
        'surface_triangle': is_surface_triangle,
    }
    return row, total_volume

if __name__ == "__main__":
    mesh = meshio.read(mesh_name)
    points = mesh.points
    surf_faces = []
    for cell_block in mesh.cells:
        if cell_block.type in ("triangle", "triangle6"):
            surf_faces = cell_block.data
            break
    if len(surf_faces) == 0:
        raise ValueError("No triangle faces found in this mesh!")
    tet_cells = None
    for cell_block in mesh.cells:
        if cell_block.type in ("tetra", "tet"):
            tet_cells = cell_block.data
            break
    if tet_cells is None:
        raise ValueError("No tetrahedra found in this mesh!")
    V_piecewise = 0.0
    for tet in tet_cells:
        pts4 = points[tet]
        V_piecewise += tet_volume(pts4)

    all_faces = surf_faces  # or set to all triangles you want to process
    surface_tris = get_surface_triangle_set_from_msh(mesh_name)

    tol = 1e-4
    max_depth = 6
    args_list = [(idx, tri, points, coeffs, tol, max_depth, surface_tris) for idx, tri in enumerate(all_faces)]

    print(f"Using {multiprocessing.cpu_count()} CPU cores...")

    t0 = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(adaptive_triangle_worker, args_list)

    csv_rows = []
    V_patch_sum = 0.0
    for row, volume in results:
        csv_rows.append(row)
        # print(f"Triangle {row['triangle_id']}: Patch volume = {row['curved_patch_volume']:.8f}, Correction = {row['correction']:.8f}, "
        #       f"Vcorrection = {row['Vcorrection']:.8f}, Surface: {row['surface_triangle']}")
        V_patch_sum += volume

    df = pd.DataFrame(csv_rows)
    df.to_csv("adaptive_triangle_patch_correction.csv", index=False)
    print("\nWrote triangle patch corrections to adaptive_triangle_patch_correction.csv")

    t1 = time.time()
    rel_error_piecewise = abs(V_piecewise - V_theory) / V_theory * 100
    rel_error_patchsum = abs(V_patch_sum + V_piecewise - V_theory) / V_theory * 100

    print(f"\n==== VOLUME SUMMARY ====")
    print(f"Theoretical value (cylinder):           {V_theory:.8f}")
    print(f"Piecewise sum of all tets:              {V_piecewise:.8f}")
    print(f"Patch-based total (curved patch sum):   {V_patch_sum:.8f}")
    print(f"Number of cylinder surface triangles:   {len(surf_faces)}")
    print(f"Relative error (piecewise tets):        {rel_error_piecewise:.6f}%")
    print(f"Relative error (curved patch sum):      {rel_error_patchsum:.6f}%")
    print(f"Elapsed time (patch sum):               {t1-t0:.2f} s")
