import numpy as np
import trimesh
import meshio
import pandas as pd
import os
import subprocess
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_symmetric_spheroid(ax, ay, az, subdivs, surf_meshname):
    sphere = trimesh.creation.icosphere(subdivisions=subdivs, radius=1.0)
    vertices = np.copy(sphere.vertices)
    vertices[:, 0] *= ax
    vertices[:, 1] *= ay
    vertices[:, 2] *= az
    meshio.write_points_cells(
        surf_meshname,
        points=vertices,
        cells=[("triangle", sphere.faces)],
        file_format="gmsh22"
    )
    return vertices, sphere.faces

def convert_msh_to_stl(msh_file, stl_file):
    mesh = meshio.read(msh_file)
    points = mesh.points
    faces = mesh.cells_dict["triangle"]
    m = trimesh.Trimesh(vertices=points, faces=faces)
    m.export(stl_file)
    print(f"Exported STL: {stl_file}")

def write_gmsh_geo(geo_file, stl_file, char_len):
    with open(geo_file, "w") as f:
        f.write(f'Merge "{stl_file}";\n')
        f.write("Surface Loop(1) = {1};\n")
        f.write("Volume(1) = {1};\n")
        f.write("Physical Volume(\"domain\") = {1};\n")
        f.write(f"Mesh.CharacteristicLengthMax = {char_len};\n")
        f.write("Mesh 3;\n")
    print(f"Wrote .geo: {geo_file}")

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

def get_2ring_neighbors(v, neighbors):
    ring1 = set(neighbors[v])
    ring2 = set()
    for u in ring1:
        ring2.update(neighbors[u])
    ring2.update(ring1)
    ring2.add(v)
    return list(ring2)

def fit_quadric_to_points(points):
    X = []
    for pt in points:
        x, y, z = pt
        X.append([x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, 1])
    X = np.array(X)
    U, S, Vt = np.linalg.svd(X)
    coeff = Vt[-1]
    return coeff / np.linalg.norm(coeff)

def fit_quadrics_over_mesh(points, neighbors, min_patch=10):
    fit_rows = []
    for v in range(len(points)):
        nbrs = get_2ring_neighbors(v, neighbors)
        if len(nbrs) < min_patch:
            continue
        fit_pts = points[nbrs]
        coeffs = fit_quadric_to_points(fit_pts)
        fit_rows.append({"vertex_index": v, **{k: c for k, c in zip("ABCDEFGHIJ", coeffs)}})
    return pd.DataFrame(fit_rows)

def extract_surface_faces_from_tet_mesh(tet_mesh):
    if "tetra" not in tet_mesh.cells_dict:
        print("No tets found!")
        return np.array([])
    tets = tet_mesh.cells_dict["tetra"]
    faces = []
    for tet in tets:
        faces += [tuple(sorted(face)) for face in [
            [tet[0], tet[1], tet[2]],
            [tet[0], tet[1], tet[3]],
            [tet[0], tet[2], tet[3]],
            [tet[1], tet[2], tet[3]],
        ]]
    face_counter = Counter(faces)
    surf_faces = [face for face, count in face_counter.items() if count == 1]
    return np.array(surf_faces)

def process_mesh(ax, ay, az, subdivs, output_prefix, char_len):
    surf_meshname = os.path.join(SCRIPT_DIR, f"{output_prefix}_sub{subdivs}_surf.msh")
    stl_file = os.path.join(SCRIPT_DIR, f"{output_prefix}_sub{subdivs}.stl")
    geo_file = os.path.join(SCRIPT_DIR, f"{output_prefix}_sub{subdivs}.geo")
    tet_meshname = os.path.join(SCRIPT_DIR, f"{output_prefix}_sub{subdivs}_tet.msh")
    combined_meshname = os.path.join(SCRIPT_DIR, f"{output_prefix}_sub{subdivs}_full.msh")

    print(f"\n--- Generating symmetric spheroid surface mesh with subdivs={subdivs} ---")
    points, faces = generate_symmetric_spheroid(ax, ay, az, subdivs, surf_meshname)

    print("--- Converting surface mesh to STL ---")
    convert_msh_to_stl(surf_meshname, stl_file)

    print("--- Writing GMSH .geo file for volume meshing ---")
    write_gmsh_geo(geo_file, stl_file, char_len)

    print("--- Calling GMSH to generate tetrahedral volume mesh ---")
    subprocess.run([
        "gmsh", geo_file, "-3", "-format", "msh2",
        "-o", tet_meshname
    ], check=True)
    print(f"Generated volume mesh: {tet_meshname}")

    print("--- Extracting surface triangles from tet mesh for combined file ---")
    tet_mesh = meshio.read(tet_meshname)
    surf_faces = extract_surface_faces_from_tet_mesh(tet_mesh)
    cells = []
    for cell_block in tet_mesh.cells:
        if cell_block.type == "tetra":
            cells.append(("tetra", cell_block.data))
    if len(surf_faces) > 0:
        cells.append(("triangle", surf_faces))
    # Write combined mesh
    meshio.write_points_cells(
        combined_meshname,
        points=tet_mesh.points,
        cells=cells,
        file_format="gmsh22"
    )
    print(f"Wrote combined tet+surf mesh: {combined_meshname}")

    print("--- Computing quadric coefficients (A...J) and saving to CSV ---")
    neighbors, vertex_faces = build_vertex_connectivity(points, faces)
    df = fit_quadrics_over_mesh(points, neighbors)
    tet_count = 0
    for cell_block in tet_mesh.cells:
        if cell_block.type == "tetra":
            tet_count += len(cell_block.data)
    out_csv = os.path.join(SCRIPT_DIR, f"Ellip_fit_tet{tet_count}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved quadric fits for each vertex: {out_csv}")
    return {
        "tet_meshname": tet_meshname,
        "quadric_csv": out_csv,
        "surf_meshname": surf_meshname,
        "combined_meshname": combined_meshname,
        "tet_count": tet_count
    }

def generate_meshes(ax, ay, az, subdivs, output_prefix="SpheroidSymm", char_len=0.1):
    """
    Generates mesh, extracts surface, fits quadrics, and returns key output file paths.
    """
    return process_mesh(ax, ay, az, subdivs, output_prefix, char_len)

def main():
    ax, ay, az = 1.5, 1.0, 0.8
    subdivs_and_charlen = [
        (0, 1.5),
        (1, 1.0),
        (2, 0.5),
        (3, 0.25)
    ]
    output_prefix = "SpheroidSymm"
    for subdivs, char_len in subdivs_and_charlen:
        process_mesh(ax, ay, az, subdivs, output_prefix, char_len)

if __name__ == "__main__":
    main()
