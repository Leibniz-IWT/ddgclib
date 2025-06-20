import numpy as np
import meshio
import subprocess
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd

# === Import your curvature calculation function ===
from _curvatures_heron import HNdC_ijk

def generate_fine_sphere_geo(filename="sphere.geo", lc=0.05):
    geo_str = f"""
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = {lc};
Mesh.CharacteristicLengthMax = {lc};
Sphere(1) = {{0, 0, 0, 1}};
Physical Surface("surface") = {{1}};
Physical Volume("volume") = {{1}};
"""
    with open(filename, "w") as f:
        f.write(geo_str)

def generate_mesh(geo_filename="sphere.geo", msh_filename="sphere.msh", lc=0.05):
    generate_fine_sphere_geo(geo_filename, lc)
    try:
        subprocess.run(
            ["gmsh", "-3", geo_filename, "-o", msh_filename, "-format", "msh2"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        if os.path.exists(msh_filename):
            print(f"Warning: GMSH reported errors, but {msh_filename} was created.")
        else:
            raise

def read_mesh(filename="sphere.msh"):
    mesh = meshio.read(filename)
    points = mesh.points
    faces = []
    tets = []
    for cell_block in mesh.cells:
        if cell_block.type in ["triangle", "triangle6"]:
            faces.extend(cell_block.data)
        if cell_block.type in ["tetra", "tetra10"]:
            tets.extend(cell_block.data)
    faces = np.array(faces)
    tets = np.array(tets)
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

def heron_mean_curvature_vectors(points, faces, neighbors, vertex_faces):
    n = len(points)
    H_vecs = np.zeros((n, 3))
    A_dual = np.zeros(n)
    for i in range(n):
        HNdA_i = np.zeros(3)
        dual_area = 0.0
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
                    dual_area += c_ijk
        H_vecs[i] = HNdA_i
        A_dual[i] = dual_area if dual_area != 0 else 1e-12
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

def barycentric_weights(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if np.abs(denom) < 1e-14:
        return (1/3, 1/3, 1/3)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return (u, v, w)

def export_per_triangle_edge_vertex(
    points, faces, neighbors, vertex_faces, H_vecs, A_dual, HN, H, k1, k2, mesh_number
):
    header = [
        "triangle_idx", "local_edge_idx", "vertex", "x", "y", "z", "wij",
        "e_ij_x", "e_ij_y", "e_ij_z", "hnda_x", "hnda_y", "hnda_z",
        "HNdA_x", "HNdA_y", "HNdA_z", "H", "k1", "k2", "dual_area"
    ]
    rows = []
    for t_idx, tri in enumerate(faces):
        for local_edge in range(3):
            i = tri[local_edge]
            j = tri[(local_edge + 1) % 3]
            k = tri[(local_edge + 2) % 3]
            vi, vj, vk = points[i], points[j], points[k]
            e_ij = vj - vi
            l_ij = np.linalg.norm(e_ij)
            e_ik = vk - vi
            l_ik = np.linalg.norm(e_ik)
            e_jk = vk - vj
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, wij = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)
            # Use precomputed per-vertex values
            HNdA = H_vecs[i]
            H_scalar = H[i]
            k1_val = k1[i]
            k2_val = k2[i]
            dual_area = A_dual[i]
            rows.append([
                t_idx, local_edge, i, vi[0], vi[1], vi[2], wij,
                e_ij[0], e_ij[1], e_ij[2],
                hnda_ijk[0], hnda_ijk[1], hnda_ijk[2],
                HNdA[0], HNdA[1], HNdA[2],
                H_scalar, k1_val, k2_val, dual_area
            ])
    fname = f"Vmesh_{mesh_number}.csv"
    with open(fname, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote per-triangle/edge/vertex data to {fname}")

def compute_piecewise_linear_volume(points, tets):
    total_volume = 0.0
    for tet in tets:
        a, b, c, d = points[tet[0]], points[tet[1]], points[tet[2]], points[tet[3]]
        v_tet = np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0
        total_volume += v_tet
    return total_volume

def compute_2ndOrderVertexBasedParaboloidPatch(points, tets, surface_faces, k1_vertex):
    from collections import defaultdict

    face_to_tet = defaultdict(list)
    for t_idx, tet in enumerate(tets):
        for local_face, idxs in enumerate([(0,1,2),(0,1,3),(0,2,3),(1,2,3)]):
            face = tuple(sorted([tet[i] for i in idxs]))
            face_to_tet[face].append((t_idx, local_face))
    surface_face_set = set(tuple(sorted(face)) for face in surface_faces)
    boundary_tet_idxs = set()
    for face in surface_face_set:
        for t_idx, local_face in face_to_tet[face]:
            boundary_tet_idxs.add(t_idx)
    all_tet_idxs = set(range(len(tets)))
    interior_tet_idxs = all_tet_idxs - boundary_tet_idxs

    V_total = 0.0
    for t_idx in interior_tet_idxs:
        a, b, c, d = points[tets[t_idx][0]], points[tets[t_idx][1]], points[tets[t_idx][2]], points[tets[t_idx][3]]
        v_flat = np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0
        V_total += v_flat
    for t_idx in boundary_tet_idxs:
        tet = tets[t_idx]
        surf_face = None
        for idxs in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
            face = tuple(sorted([tet[i] for i in idxs]))
            if face in surface_face_set:
                surf_face = [tet[i] for i in idxs]
                other_idx = [idx for idx in range(4) if idx not in idxs][0]
                d = points[tet[other_idx]]
                break
        if surf_face is None:
            continue
        A_idx, B_idx, C_idx = surf_face
        A, B, C = points[A_idx], points[B_idx], points[C_idx]
        v_flat = np.abs(np.dot(A - d, np.cross(B - d, C - d))) / 6.0
        tri_vec1 = B - A
        tri_vec2 = C - A
        area_triangle = 0.5 * np.linalg.norm(np.cross(tri_vec1, tri_vec2))
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
        v_corr = v_flat + (area_triangle * s_avg) / 3.0
        V_total += v_corr
    return V_total

def compute_curved_neighbor_patch_volume(points, tets, surface_faces, k1_vertex):
    """
    Curved Neighbor Patch method for mesh boundary volume correction.
    - For each surface triangle, build an apex offset along the normal.
    - For each edge shared by two triangles, form a wedge between the edge and the two apexes.
    - Sum all wedge volumes and add the paraboloid patch surface correction for surface tets.
    """

    from collections import defaultdict

    def apex_point(A, B, C, kA, kB, kC, eps=1e-8):
        bary = (A + B + C) / 3.0
        n = np.cross(B - A, C - A)
        n = n / (np.linalg.norm(n) + 1e-12)
        s = []
        for v, k in zip([A, B, C], [kA, kB, kC]):
            if np.abs(k) < eps:
                R = 1e10
            else:
                R = 1.0 / np.abs(k)
            r = np.linalg.norm(v - bary)
            under_rad = max(R*R - r*r, 0.0)
            s_i = R - np.sqrt(under_rad)
            if k < 0:
                s_i = -s_i
            s.append(s_i)
        s_avg = np.mean(s)
        k_avg = np.mean([kA, kB, kC])
        apex = bary - np.sign(k_avg) * s_avg * n
        return apex

    # 1. Find boundary tets (like in paraboloid patch)
    face_to_tet = defaultdict(list)
    for t_idx, tet in enumerate(tets):
        for local_face, idxs in enumerate([(0,1,2),(0,1,3),(0,2,3),(1,2,3)]):
            face = tuple(sorted([tet[i] for i in idxs]))
            face_to_tet[face].append((t_idx, local_face))
    surface_face_set = set(tuple(sorted(face)) for face in surface_faces)
    boundary_tet_idxs = set()
    for face in surface_face_set:
        for t_idx, local_face in face_to_tet[face]:
            boundary_tet_idxs.add(t_idx)
    all_tet_idxs = set(range(len(tets)))
    interior_tet_idxs = all_tet_idxs - boundary_tet_idxs

    # 2. Compute paraboloid patch correction for surface tets (same as paraboloid patch)
    V_total = 0.0
    for t_idx in interior_tet_idxs:
        a, b, c, d = points[tets[t_idx][0]], points[tets[t_idx][1]], points[tets[t_idx][2]], points[tets[t_idx][3]]
        v_flat = np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0
        V_total += v_flat
    for t_idx in boundary_tet_idxs:
        tet = tets[t_idx]
        surf_face = None
        for idxs in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
            face = tuple(sorted([tet[i] for i in idxs]))
            if face in surface_face_set:
                surf_face = [tet[i] for i in idxs]
                other_idx = [idx for idx in range(4) if idx not in idxs][0]
                d = points[tet[other_idx]]
                break
        if surf_face is None:
            continue
        A_idx, B_idx, C_idx = surf_face
        A, B, C = points[A_idx], points[B_idx], points[C_idx]
        v_flat = np.abs(np.dot(A - d, np.cross(B - d, C - d))) / 6.0
        tri_vec1 = B - A
        tri_vec2 = C - A
        area_triangle = 0.5 * np.linalg.norm(np.cross(tri_vec1, tri_vec2))
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
        v_corr = v_flat + (area_triangle * s_avg) / 3.0
        V_total += v_corr

    # 3. For each surface triangle, compute apex
    triangle_to_apex = {}
    for tri in surface_faces:
        A_idx, B_idx, C_idx = tri
        A, B, C = points[A_idx], points[B_idx], points[C_idx]
        kA, kB, kC = k1_vertex[A_idx], k1_vertex[B_idx], k1_vertex[C_idx]
        apex = apex_point(A, B, C, kA, kB, kC)
        triangle_to_apex[tuple(sorted(tri))] = apex

    # 4. For each unique boundary edge, form a wedge between the two apexes and the edge
    # Find edges and neighboring triangles
    edge_to_triangles = defaultdict(list)
    for tri in surface_faces:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i+1)%3])))
            edge_to_triangles[edge].append(tuple(sorted(tri)))
    wedge_volume = 0.0
    for edge, tris in edge_to_triangles.items():
        if len(tris) == 2:
            tri1, tri2 = tris
            a_idx, b_idx = edge
            a, b = points[a_idx], points[b_idx]
            apex1 = triangle_to_apex[tri1]
            apex2 = triangle_to_apex[tri2]
            # Compute signed wedge volume
            v_wedge = np.dot(np.cross(b - a, apex1 - a), apex2 - a) / 6.0
            wedge_volume += 0.5 * np.abs(v_wedge)
    V_total += wedge_volume

    return V_total

# =============== ADDED METHOD: Spherical Triangle Wedge Volume ================
def spherical_triangle_wedge_volume(A, B, C, k):
    """
    Compute the volume of the spherical cap above the triangle ABC
    on a sphere with curvature k (i.e., radius R = 1/k).
    A, B, C: 3x3 arrays (cartesian coordinates)
    k: curvature (positive for unit sphere)
    Returns: volume of the wedge/cap above triangle ABC
    """
    R = 1.0 / k
    # Normalize points onto the sphere
    a = A / np.linalg.norm(A)
    b = B / np.linalg.norm(B)
    c = C / np.linalg.norm(C)

    # Compute side lengths (arc)
    ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
    bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
    ca = np.arccos(np.clip(np.dot(c, a), -1, 1))

    # Compute spherical triangle angles using the spherical law of cosines
    def spherical_angle(va, vb, vc):
        # angle at va between vb and vc (all normalized)
        ab = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        ac = np.arccos(np.clip(np.dot(va, vc), -1, 1))
        bc = np.arccos(np.clip(np.dot(vb, vc), -1, 1))
        return np.arccos((np.cos(ab) - np.cos(ac) * np.cos(bc)) / (np.sin(ac) * np.sin(bc) + 1e-14))
    
    alpha = spherical_angle(a, b, c)
    beta  = spherical_angle(b, c, a)
    gamma = spherical_angle(c, a, b)

    # Spherical excess (solid angle)
    Omega = alpha + beta + gamma - np.pi
    # Wedge volume
    V = (R ** 3 / 3) * Omega
    # --- Tetrahedron volume (between center O and triangle ABC) ---
    O = np.zeros(3)
    V_tet = np.abs(np.dot((A - O), np.cross(B - O, C - O))) / 6.0

    # Spherical cap = wedge - flat tetrahedron
    return V - V_tet
# ==============================================================================

if __name__ == "__main__":
    geo = "sphere.geo"
    msh = "sphere.msh"
    lc_values = np.linspace(0.1, 0.8, 8)
    triangle_counts = []
    HN_means = []
    HN_stds = []
    HN_at_exact_north_pole = []
    k1_means = []
    k2_means = []

    piecewise_volumes = []
    paraboloid_patch_volumes = []
    curved_neighbor_patch_volumes = []
    wedge_volumes = []
    tet_counts = []

    for lc in lc_values:
        print(f"\nGenerating mesh with lc={lc:.3f} ...")
        generate_mesh(geo, msh, lc=lc)
        points, faces, tets = read_mesh(msh)  # tets now included
        print(f"Mesh: {len(points)} vertices, {len(faces)} triangles, {len(tets)} tetrahedra")
        neighbors, vertex_faces = build_vertex_connectivity(points, faces)
        H_vecs, A_dual = heron_mean_curvature_vectors(points, faces, neighbors, vertex_faces)
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        HN = -np.einsum('ij,ij->i', H_vecs, normals) / (A_dual + 1e-12)
        H = HN / 2  # Heron method gives 2H*n

        K_angle = gaussian_curvature_angle_defect(points, faces, vertex_faces)
        K = K_angle / (A_dual + 1e-12)
        k1 = H + np.sqrt(np.clip(H ** 2 - K, 0, None))
        k2 = H - np.sqrt(np.clip(H ** 2 - K, 0, None))

        HN_means.append(np.mean(HN))
        HN_stds.append(np.std(HN))
        triangle_counts.append(len(faces))
        k1_means.append(np.mean(k1))
        k2_means.append(np.mean(k2))

        V_piecewise = compute_piecewise_linear_volume(points, tets)
        piecewise_volumes.append(V_piecewise)
        tet_counts.append(len(tets))
        print(f"Piecewise linear volume: {V_piecewise:.6f}")

        V_paraboloid_patch = compute_2ndOrderVertexBasedParaboloidPatch(points, tets, faces, k1)
        paraboloid_patch_volumes.append(V_paraboloid_patch)
        print(f"Curvature-corrected (2nd order paraboloid patch) volume: {V_paraboloid_patch:.6f}")

        V_curved_neighbor_patch = compute_curved_neighbor_patch_volume(points, tets, faces, k1)
        curved_neighbor_patch_volumes.append(V_curved_neighbor_patch)
        print(f"Curved Neighbor Patch (wedge+patch) volume: {V_curved_neighbor_patch:.6f}")

        # --- Spherical Wedge Volume Sum for Full Sphere ---
        k_sphere = 1.0  # unit sphere curvature
        V_spherical_wedge_sum = 0.0
        for tri in faces:
            i, j, k = tri
            V_spherical_wedge_sum += spherical_triangle_wedge_volume(points[i], points[j], points[k], k_sphere)
        wedge_volumes.append(V_spherical_wedge_sum+V_piecewise)  # add piecewise volume for full sphere
        print(f"Spherical Wedge sum (total): {V_spherical_wedge_sum:.6f}")
        print("Therical volume of unit sphere: 4/3 * pi = {:.6f}".format(4/3 * np.pi))

        # ---- HN at north pole ----
        target = np.array([0,0,1])
        min_dist = np.inf
        closest_face = None
        for tri in faces:
            verts = points[tri]
            centroid = np.mean(verts, axis=0)
            dist = np.linalg.norm(centroid - target)
            if dist < min_dist:
                min_dist = dist
                closest_face = tri

        verts = points[closest_face]
        hvecs = H_vecs[closest_face]
        aduals = A_dual[closest_face]
        u, v, w = barycentric_weights(target, verts[0], verts[1], verts[2])
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        w = np.clip(w, 0, 1)
        total = u + v + w
        if total == 0:
            u = v = w = 1/3
        else:
            u /= total
            v /= total
            w /= total
        HN_vec_at_exact = u * hvecs[0] + v * hvecs[1] + w * hvecs[2]
        dual_area_at_exact = u * aduals[0] + v * aduals[1] + w * aduals[2]
        HN_at_north_pole_exact = -np.dot(HN_vec_at_exact, np.array([0,0,1])) / (dual_area_at_exact + 1e-12)
        HN_at_exact_north_pole.append(HN_at_north_pole_exact)

        export_per_triangle_edge_vertex(
            points, faces, neighbors, vertex_faces,
            H_vecs, A_dual, HN, H, k1, k2, len(faces)
        )

    analytical_vol = 4/3 * np.pi
    
    piecewise_relerr = [abs(V - analytical_vol) / analytical_vol * 100 for V in piecewise_volumes]  # percent
    patch_relerr = [abs(V - analytical_vol) / analytical_vol * 100 for V in paraboloid_patch_volumes]  # percent
    curved_patch_relerr = [abs(V - analytical_vol) / analytical_vol * 100 for V in curved_neighbor_patch_volumes]
    wedge_relerr = [abs(V - analytical_vol) / analytical_vol * 100 for V in wedge_volumes]

    # --- Save volume error stats to CSV ---
    volume_error_data = pd.DataFrame({
        "tet_count": tet_counts,
        "piecewise_relerr_percent": piecewise_relerr,
        "patch_relerr_percent": patch_relerr,
        "curved_patch_relerr_percent": curved_patch_relerr,
        "wedge_sum_relerr_percent": wedge_relerr,
        "piecewise_volume": piecewise_volumes,
        "paraboloid_patch_volume": paraboloid_patch_volumes,
        "curved_neighbor_patch_volume": curved_neighbor_patch_volumes,
        "wedge_sum_volume": wedge_volumes
    })
    volume_error_data.to_csv("volume_error_vs_tet_count.csv", index=False)
    print("Saved: volume_error_vs_tet_count.csv")

    plt.figure(figsize=(8,5))
    plt.plot(tet_counts, piecewise_relerr, '-o', label="Piecewise Linear Volume (rel. err)")
    plt.plot(tet_counts, patch_relerr, '-o', label="2nd Order Paraboloid Patch (rel. err)")
    plt.plot(tet_counts, curved_patch_relerr, '-o', label="Curved Neighbor Patch (rel. err)")
    plt.plot(tet_counts, wedge_relerr, '-o', label="Spherical Wedge Sum (rel. err)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Tet Elements (log scale)')
    plt.ylabel('Relative Error (%) (log scale)')
    plt.title('Relative Volume Error (%) vs Number of Tetrahedra (Unit Sphere)')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

    # ===========================
    # CSV-based mean curvature plotting: Unique per vertex!
    # ===========================
    k1_means_csv = []
    k2_means_csv = []
    HN_means_csv = []
    HN_stds_csv = []
    triangle_counts_csv = []

    for tri_count in triangle_counts:
        csv_filename = f"Vmesh_{tri_count}.csv"
        df = pd.read_csv(csv_filename)
        # Group by vertex for unique mean per vertex
        k1_by_vertex = df.groupby('vertex')['k1'].mean().values
        k2_by_vertex = df.groupby('vertex')['k2'].mean().values
        # HN: use column H, HN, or HNdA_z, whatever present
        if 'HN' in df.columns:
            HN_by_vertex = df.groupby('vertex')['HN'].mean().values
        elif 'H' in df.columns:
            HN_by_vertex = df.groupby('vertex')['H'].mean().values
        elif 'HNdA_z' in df.columns:
            HN_by_vertex = df.groupby('vertex')['HNdA_z'].mean().values
        else:
            HN_by_vertex = df.groupby('vertex')['HNdA_z'].mean().values

        k1_means_csv.append(np.mean(k1_by_vertex))
        k2_means_csv.append(np.mean(k2_by_vertex))
        HN_means_csv.append(np.mean(HN_by_vertex))
        HN_stds_csv.append(np.std(HN_by_vertex))
        triangle_counts_csv.append(tri_count)

    # --- Save curvature means (unique vertex) to CSV ---
    curvature_data = pd.DataFrame({
        "triangle_count": triangle_counts_csv,
        "HN_mean": HN_means_csv,
        "HN_std": HN_stds_csv,
        "k1_mean": k1_means_csv,
        "k2_mean": k2_means_csv
    })
    curvature_data.to_csv("curvature_vs_triangle_count.csv", index=False)
    print("Saved: curvature_vs_triangle_count.csv")

    plt.figure(figsize=(8,5))
    plt.plot(triangle_counts_csv, HN_means_csv, '-', label="Mean HN (unique vertex)")
    plt.plot(triangle_counts_csv, k1_means_csv, 'o', label="k1 mean (unique vertex)")
    plt.plot(triangle_counts_csv, k2_means_csv, 'x', label="k2 mean (unique vertex)")
    plt.axhline(1, color="grey", linestyle="dotted", label="k1/k2 analytical (1.0)")
    plt.xlabel("Number of triangles")
    plt.ylabel("Curvature")
    plt.title("Curvatures vs mesh triangles on unit sphere (Heron method, unique vertex mean)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
