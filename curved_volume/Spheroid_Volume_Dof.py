import numpy as np
import trimesh
import meshio
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import least_squares, brentq

from _curvatures_heron import HNdC_ijk

ax, ay, az = 1.5, 1.0, 0.8
subdivs_list = [0, 1, 2, 3]
output_prefix = "SpheroidSymm"

def ellipsoid_equation(p, point):
    x0, y0, z0, a, b, c = p
    x, y, z = point
    return ((x - x0) ** 2) / (a ** 2) + ((y - y0) ** 2) / (b ** 2) + ((z - z0) ** 2) / (c ** 2) - 1

def ellipsoid_fit_function(params, points):
    return [ellipsoid_equation(params, pt) for pt in points]

def fit_ellipsoid_from_points(points):
    points = np.asarray(points)
    centroid = np.mean(points, axis=0)
    axis_guess = (np.max(points, axis=0) - np.min(points, axis=0)) / 2
    axis_guess[axis_guess < 1e-2] = 1.0
    x0_guess, y0_guess, z0_guess = centroid
    a_guess, b_guess, c_guess = axis_guess
    initial_guess = [x0_guess, y0_guess, z0_guess, a_guess, b_guess, c_guess]
    bounds = ([-np.inf, -np.inf, -np.inf, 1e-6, 1e-6, 1e-6], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    result = least_squares(
        ellipsoid_fit_function,
        initial_guess,
        args=(points,),
        bounds=bounds,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=100000,
    )
    if not result.success:
        raise RuntimeError("Ellipsoid fit did not converge!")
    return result.x  # x0, y0, z0, a, b, c

def generate_symmetric_spheroid(ax, ay, az, subdivs, meshname):
    sphere = trimesh.creation.icosphere(subdivisions=subdivs, radius=1.0)
    vertices = np.copy(sphere.vertices)
    vertices[:, 0] *= ax
    vertices[:, 1] *= ay
    vertices[:, 2] *= az
    meshio.write_points_cells(
        meshname,
        points=vertices,
        cells=[("triangle", sphere.faces)],
        file_format="gmsh22"
    )
    return meshname

def read_surface_mesh(filename):
    mesh = meshio.read(filename)
    points = mesh.points
    faces = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            faces.extend(cell_block.data)
    faces = np.array(faces)
    tets = np.zeros((0,4), dtype=int)
    return points, faces, tets

def build_vertex_connectivity(points, faces):
    n = len(points)
    neighbors = [set() for _ in range(n)]
    vertex_faces = [[] for _ in range(n)]
    for idx, tri in enumerate(faces):
        for i in range(3):
            v = tri[i]
            neighbors[v].update([tri[(i+1)%3], tri[(i+2)%3]])
            vertex_faces[v].append(idx)
    return neighbors, vertex_faces

def compute_vertex_voronoi_area(points, faces):
    n = len(points)
    area_per_vertex = np.zeros(n)
    for tri in faces:
        i, j, k = tri
        vi, vj, vk = points[i], points[j], points[k]
        area = 0.5 * np.linalg.norm(np.cross(vj-vi, vk-vi))
        area_per_vertex[i] += area / 3
        area_per_vertex[j] += area / 3
        area_per_vertex[k] += area / 3
    return area_per_vertex

def heron_mean_curvature_vectors(points, faces, neighbors, vertex_faces):
    n = len(points)
    H_vecs = np.zeros((n, 3))
    A_dual = np.zeros(n)
    for tri in faces:
        i, j, k = tri
        vi, vj, vk = points[i], points[j], points[k]
        area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
        A_dual[i] += area / 3
        A_dual[j] += area / 3
        A_dual[k] += area / 3
    for i in range(n):
        HNdA_i = np.zeros(3)
        for j in neighbors[i]:
            shared_faces = [idx for idx in vertex_faces[i] if j in faces[idx]]
            for face_idx in shared_faces:
                tri = faces[face_idx]
                if i in tri and j in tri:
                    k = [v for v in tri if v != i and v != j][0]
                    vi, vj, vk = points[i], points[j], points[k]
                    e_ij = vj - vi
                    e_ik = vk - vi
                    e_jk = vk - vj
                    l_ij = np.linalg.norm(e_ij)
                    l_ik = np.linalg.norm(e_ik)
                    l_jk = np.linalg.norm(e_jk)
                    hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)
                    HNdA_i += hnda_ijk
        H_vecs[i] = HNdA_i
    return H_vecs, A_dual

def gaussian_curvature_angle_defect(points, faces, vertex_faces):
    n = len(points)
    K = np.zeros(n)
    for i in range(n):
        angles_sum = 0.0
        pi = points[i]
        for idx in vertex_faces[i]:
            tri = faces[idx]
            if i in tri:
                a, b = [v for v in tri if v != i]
                va = points[a] - pi
                vb = points[b] - pi
                norm_a = np.linalg.norm(va)
                norm_b = np.linalg.norm(vb)
                if norm_a == 0 or norm_b == 0:
                    continue
                angle = np.arccos(np.clip(np.dot(va, vb)/(norm_a*norm_b), -1, 1))
                angles_sum += angle
        K[i] = 2 * np.pi - angles_sum
    return K

def compute_piecewise_linear_volume(points, faces):
    V = 0.0
    for tri in faces:
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        V += np.dot(np.cross(a, b), c)
    return abs(V) / 6

def compute_2ndOrderVertexBasedParaboloidPatch(points, faces, k1_vertex):
    V_total = compute_piecewise_linear_volume(points, faces)
    for tri in faces:
        A_idx, B_idx, C_idx = tri
        A, B, C = points[A_idx], points[B_idx], points[C_idx]
        area_triangle = 0.5 * np.linalg.norm(np.cross(B-A, C-A))
        bary = (A + B + C) / 3.0
        s_sum = 0.0
        for idx in [A_idx, B_idx, C_idx]:
            k = k1_vertex[idx]
            if np.abs(k) < 1e-8:
                s_i = 0.0
            else:
                R = 1.0 / np.abs(k)
                r = np.linalg.norm(points[idx] - bary)
                under_radical = max(R*R - r*r, 0.0)
                s_i = R - np.sqrt(under_radical)
                if k < 0:
                    s_i = -s_i
            s_sum += s_i
        s_avg = s_sum / 3.0
        V_total += (area_triangle * s_avg) / 3.0
    return V_total

def compute_curved_neighbor_patch_volume(points, faces, k1_vertex):
    return compute_2ndOrderVertexBasedParaboloidPatch(points, faces, k1_vertex)

def get_patch_1ring(points, tri, neighbors):
    patch_vertices = set(tri)
    for v in tri:
        patch_vertices.update(neighbors[v])
    patch_vertices = list(patch_vertices)
    return points[patch_vertices]

def curved_spheroid_patch_volume(A, B, C, a, b, c, tet_volume, area):
    triangleArea = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
    analytical_volume = 4/3 * np.pi * a * b * c
    return (analytical_volume - tet_volume) * triangleArea / area

def curved_spheroid_patch_volume_Integral(A, B, C, ax, ay, az, center=None, npts=7):
    """
    Approximates the volume between triangle ABC and the curved spheroid patch above it.
    Uses barycentric quadrature over the triangle (triangle on the flat base).
    """
    if center is None:
        center = np.zeros(3)
    A, B, C = A - center, B - center, C - center

    # Area of the base triangle
    area_tri = 0.5 * np.linalg.norm(np.cross(B-A, C-A))

    vol_sum = 0.0
    count = 0
    # Loop over barycentric grid
    for i in range(npts + 1):
        for j in range(npts + 1 - i):
            u = i / npts
            v = j / npts
            w = 1 - u - v
            p_flat = u*A + v*B + w*C
            x, y, z = p_flat
            if np.linalg.norm(p_flat) < 1e-12:
                continue  # avoid center
            # Project to spheroid surface: solve s such that spheroid_eq(s)=0
            def spheroid_eq(s):
                return (s*x/ax)**2 + (s*y/ay)**2 + (s*z/az)**2 - 1
            try:
                # s > 1 is outside, s=1 is inside, search s in [1, 10]
                s = brentq(spheroid_eq, 0.99, 10)
            except Exception:
                continue
            p_curved = s * p_flat

            # The local "height" between spheroid surface and triangle at this (u,v)
            h = np.linalg.norm(p_curved - p_flat)
            # Weight for this quadrature point
            vol_sum += h
            count += 1

    # The average height * triangle area gives the patch volume
    if count == 0:
        return 0.0
    return vol_sum / count * area_tri

def save_triangle_patch_csv(faces, per_triangle_patchfit, area, tet_count):
    """
    Save a CSV for each triangle: triangle_idx, vertex_id, a, b, c, triangleArea, analytical_volume, area
    """
    rows = []
    for tri_idx, tri in enumerate(faces):
        fit = per_triangle_patchfit.get(tri_idx)
        if fit is None:
            continue
        a, b, c, triangleArea, analytical_volume = fit
        for local_vid, v_idx in enumerate(tri):
            rows.append({
                "triangle_idx": tri_idx,
                "vertex_id": v_idx,
                "a": a,
                "b": b,
                "c": c,
                "triangleArea": triangleArea,
                "analytical_volume": analytical_volume,
                "area": area,
            })
    df = pd.DataFrame(rows)
    csvname = f"Vmesh_{tet_count}.csv"
    df.to_csv(csvname, index=False)
    print(f"Saved: {csvname}")

if __name__ == "__main__":
    curved_patch_cap_volumes = []
    curved_patch_cap_volumes_integral = []
    triangle_counts = []
    HN_means = []
    HN_stds = []
    k1_means = []
    k2_means = []
    piecewise_volumes = []
    paraboloid_patch_volumes = []
    curved_neighbor_patch_volumes = []
    tet_counts = []
    voronoi_dual_area_totals = []

    for subdivs in subdivs_list:
        meshname = f"{output_prefix}_sub{subdivs}.msh"
        print(f"\n--- Generating symmetric spheroid mesh with subdivs={subdivs} ---")
        generate_symmetric_spheroid(ax, ay, az, subdivs, meshname)
        points, faces, tets = read_surface_mesh(meshname)

        tet_volume = compute_piecewise_linear_volume(points, faces)
        area = sum(
            0.5 * np.linalg.norm(np.cross(points[tri[1]] - points[tri[0]], points[tri[2]] - points[tri[0]]))
            for tri in faces
        )

        print(f"Mesh: {len(points)} vertices, {len(faces)} triangles")

        area_per_vertex = compute_vertex_voronoi_area(points, faces)
        total_surface_area = np.sum(area_per_vertex)
        print(f"Total mesh surface area (Voronoi sum): {total_surface_area:.6f}")
        voronoi_dual_area_totals.append(total_surface_area)

        fig = plt.figure(figsize=(6,6))
        ax3d = fig.add_subplot(111, projection='3d')
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        ax3d.plot_trisurf(
            x, y, faces, z,
            cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.5
        )
        ax3d.set_title(f"Spheroid Mesh (subdivs={subdivs})")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        neighbors, vertex_faces = build_vertex_connectivity(points, faces)
        H_vecs, A_dual = heron_mean_curvature_vectors(points, faces, neighbors, vertex_faces)
        point_norms = np.linalg.norm(points, axis=1)
        normals = points / (point_norms[:, None] + 1e-14)
        HN = -np.einsum('ij,ij->i', H_vecs, normals) / (A_dual + 1e-12)
        H = HN / 2
        K_angle = gaussian_curvature_angle_defect(points, faces, vertex_faces)
        K = K_angle / (A_dual + 1e-12)
        k1 = H + np.sqrt(np.clip(H ** 2 - K, 0, None))
        k2 = H - np.sqrt(np.clip(H ** 2 - K, 0, None))

        HN_means.append(np.mean(HN))
        HN_stds.append(np.std(HN))
        k1_means.append(np.mean(k1))
        k2_means.append(np.mean(k2))
        triangle_counts.append(len(faces))
        tet_counts.append(len(faces))

        V_piecewise = tet_volume
        piecewise_volumes.append(V_piecewise)
        V_paraboloid_patch = compute_2ndOrderVertexBasedParaboloidPatch(points, faces, k1)
        paraboloid_patch_volumes.append(V_paraboloid_patch)
        V_curved_neighbor_patch = compute_curved_neighbor_patch_volume(points, faces, k1)
        curved_neighbor_patch_volumes.append(V_curved_neighbor_patch)

        # ---- Curved spheroid cap patch calculation (LOCAL a,b,c from patch fit) ----
        V_curved_patch_cap_sum = 0.0
        per_triangle_patchfit = {}  # Store patch-fit results for CSV
        for tidx, tri in enumerate(faces):
            pts_patch = get_patch_1ring(points, tri, neighbors)
            if len(pts_patch) >= 6:
                try:
                    x0, y0, z0, a, b, c = fit_ellipsoid_from_points(pts_patch)
                    pa, pb, pc = points[tri[0]], points[tri[1]], points[tri[2]]
                    triangleArea = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa))
                    analytical_volume = 4/3 * np.pi * a * b * c
                    per_triangle_patchfit[tidx] = (a, b, c, triangleArea, analytical_volume)
                    V_curved_patch_cap_sum += curved_spheroid_patch_volume(pa, pb, pc, a, b, c, tet_volume, area)
                except Exception as e:
                    continue
        curved_patch_cap_volumes.append(V_curved_patch_cap_sum)
        print(f"Curved spheroid patch cap sum!!!!!: {V_piecewise + V_curved_patch_cap_sum:.6f}")
        print("theoretical volume", 4/3 * np.pi * ax * ay * az)
        print(f"Piecewise volume: {V_piecewise:.6f}, Patch: {V_paraboloid_patch:.6f}")

        # ---- Curved spheroid cap patch calculation (INTEGRAL version, global ax/ay/az) ----
        V_curved_patch_cap_sum_integral = 0.0
        for tidx, tri in enumerate(faces):
            pa, pb, pc = points[tri[0]], points[tri[1]], points[tri[2]]
            V_curved_patch_cap_sum_integral += curved_spheroid_patch_volume_Integral(pa, pb, pc, ax, ay, az)
        curved_patch_cap_volumes_integral.append(V_curved_patch_cap_sum_integral)
        print(f"Curved spheroid patch cap sum (INTEGRAL): {V_piecewise + V_curved_patch_cap_sum_integral:.6f}")

        # ----- SAVE PER TRIANGLE PATCH CSV -----
        save_triangle_patch_csv(faces, per_triangle_patchfit, area, len(faces))

    analytical_vol = 4/3 * np.pi * ax * ay * az
    piecewise_relerr = [abs(V - analytical_vol) / analytical_vol * 100 for V in piecewise_volumes]
    patch_relerr = [abs(V - analytical_vol) / analytical_vol * 100 for V in paraboloid_patch_volumes]
    curved_patch_relerr = [abs(V - analytical_vol) / analytical_vol * 100 for V in curved_neighbor_patch_volumes]
    curved_patch_cap_relerr = [
        abs(V + piecewise_volumes[i] - analytical_vol) / analytical_vol * 100
        for i, V in enumerate(curved_patch_cap_volumes)
    ]
    curved_patch_cap_relerr_integral = [
        abs(V + piecewise_volumes[i] - analytical_vol) / analytical_vol * 100
        for i, V in enumerate(curved_patch_cap_volumes_integral)
    ]

    volume_error_data = pd.DataFrame({
        "tet_count": tet_counts,
        "piecewise_relerr_percent": piecewise_relerr,
        "patch_relerr_percent": patch_relerr,
        "curved_neighbor_patch_relerr_percent": curved_patch_relerr,
        "curved_patch_cap_relerr_percent": curved_patch_cap_relerr,
        "curved_patch_cap_relerr_integral_percent": curved_patch_cap_relerr_integral,
    })
    volume_error_data.to_csv("volume_error_vs_tet_count.csv", index=False)
    print("Saved: volume_error_vs_tet_count.csv")

    plt.figure(figsize=(8,5))
    plt.plot(tet_counts, piecewise_relerr, '-o', label="Piecewise Linear Volume")
    plt.plot(tet_counts, patch_relerr, '-o', label="2nd Order Paraboloid Patch")
    plt.plot(tet_counts, curved_patch_relerr, '-o', label="Curved Neighbor Patch")
    plt.plot(tet_counts, curved_patch_cap_relerr, '-o', label="Curved Spheroid Patch Cap")
    plt.plot(tet_counts, curved_patch_cap_relerr_integral, '-o', label="Curved Spheroid Patch Cap (Integral)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Tet Elements (log scale)')
    plt.ylabel('Relative Error (%) (log scale)')
    plt.title('Relative Volume Error (%) vs Number of Tetrahedra (Spheroid)')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()
