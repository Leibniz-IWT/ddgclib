import numpy as np
import meshio
import pandas as pd
import os
import subprocess
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_cylinder_geo(geo_file, radius=1.0, height=2.0, lc=0.6):
    geo_str = f"""
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = {lc};
Mesh.CharacteristicLengthMax = {lc};
Cylinder(1) = {{0, 0, -{height/2}, 0, 0, {height}, {radius}, 2*Pi}};
// Gmsh auto-labels lateral=1, bottom=2, top=3 for OpenCASCADE Cylinder
Physical Volume("domain") = {{1}};
Physical Surface("lateral") = {{1}};
Physical Surface("bottom") = {{2}};
Physical Surface("top") = {{3}};
Mesh 3;
"""
    with open(geo_file, "w") as f:
        f.write(geo_str)
    print(f"Geo file '{geo_file}' created.")

def run_gmsh(geo_file, msh_file):
    print(f"Running gmsh to generate mesh '{msh_file}' ...")
    try:
        subprocess.run(["gmsh", geo_file, "-3", "-format", "msh2", "-o", msh_file], check=True)
        print("Mesh generation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Gmsh failed: {e}")
        exit(1)

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

def process_cylinder_mesh(radius, height, lc, output_prefix):
    geo_file = os.path.join(SCRIPT_DIR, f"{output_prefix}.geo")
    tet_meshname = os.path.join(SCRIPT_DIR, f"{output_prefix}_tet.msh")
    
    print(f"\n--- Generating cylinder mesh: radius={radius}, height={height}, lc={lc} ---")
    generate_cylinder_geo(geo_file, radius=radius, height=height, lc=lc)
    run_gmsh(geo_file, tet_meshname)
    
    print("--- Reading mesh and fitting quadrics on the surface ---")
    mesh = meshio.read(tet_meshname)
    # Get surface triangles (all boundary triangles, including lateral and caps)
    faces = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
    if faces is None:
        raise RuntimeError("No surface triangles found in the mesh!")
    points = mesh.points
    neighbors, vertex_faces = build_vertex_connectivity(points, faces)
    df = fit_quadrics_over_mesh(points, neighbors)
    tet_count = 0
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tet_count += len(cell_block.data)
    out_csv = os.path.join(SCRIPT_DIR, f"{output_prefix}_fit_tet{tet_count}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved quadric fits for each vertex: {out_csv}")

    return {
        "tet_meshname": tet_meshname,
        "quadric_csv": out_csv,
        "tet_count": tet_count
    }

def generate_meshes(radius, height, lc, output_prefix="CylinderSymm"):
    return process_cylinder_mesh(radius, height, lc, output_prefix)

def main():
    radius = 1.0
    height = 2.0
    mesh_settings = [
        (1.0, "CylinderSymm_lc1.0"),
        (0.6, "CylinderSymm_lc0.6"),
        (0.3, "CylinderSymm_lc0.3"),
        (0.15, "CylinderSymm_lc0.15"),
    ]
    for lc, prefix in mesh_settings:
        generate_meshes(radius, height, lc, prefix)

if __name__ == "__main__":
    main()
