import numpy as np
import meshio
import time

def solid_angle_cap(A, B, C, ax, ay, az):
    """
    Compute the curved wedge (cap) volume Vcap for triangle ABC on an ellipsoid.
    This uses the solid angle approach after mapping to the unit sphere.
    """
    def to_sphere(P):
        mapped = np.array([P[0] / ax, P[1] / ay, P[2] / az])
        return mapped / np.linalg.norm(mapped)
    a = to_sphere(A)
    b = to_sphere(B)
    c = to_sphere(C)

    def spherical_angle(va, vb, vc):
        ab = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        ac = np.arccos(np.clip(np.dot(va, vc), -1, 1))
        bc = np.arccos(np.clip(np.dot(vb, vc), -1, 1))
        return np.arccos((np.cos(ab) - np.cos(ac) * np.cos(bc)) /
                         (np.sin(ac) * np.sin(bc) + 1e-14))
    alpha = spherical_angle(a, b, c)
    beta  = spherical_angle(b, c, a)
    gamma = spherical_angle(c, a, b)
    Omega = alpha + beta + gamma - np.pi
    Vwedge = (1.0 / 3.0) * Omega
    O = np.array([0.0, 0.0, 0.0])
    Vflat = abs(np.dot(a - O, np.cross(b - O, c - O))) / 6.0
    Vcap = (Vwedge - Vflat) * (ax * ay * az)
    return Vcap, Omega

def tet_volume(pts):
    p0, p1, p2, p3 = pts
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6

def compute_volume_error_solid_angle(
    tet_mesh_file, surf_mesh_file, ax, ay, az, prefix=""
):
    """
    Compute V_piecewise, Vcap_sum, errors, and return summary row for benchmarking solid angle patch method.
    """
    V_theory = 4/3 * np.pi * ax * ay * az

    # Read surface mesh
    surf_mesh = meshio.read(surf_mesh_file)
    surf_points = surf_mesh.points
    surf_faces = []
    for cell_block in surf_mesh.cells:
        if cell_block.type in ("triangle", "triangle6"):
            surf_faces = cell_block.data
            break
    if len(surf_faces) == 0:
        raise ValueError(f"No triangle faces found in {surf_mesh_file}")

    # Read tet mesh
    tet_mesh = meshio.read(tet_mesh_file)
    tet_points = tet_mesh.points
    tet_cells = None
    for cell_block in tet_mesh.cells:
        if cell_block.type in ("tetra", "tet"):
            tet_cells = cell_block.data
            break
    if tet_cells is None:
        raise ValueError(f"No tetrahedra found in {tet_mesh_file}")

    # Piecewise volume
    V_piecewise = 0.0
    for tet in tet_cells:
        pts4 = tet_points[tet]
        V_piecewise += tet_volume(pts4)

    # Sum Vcap for all surface triangles
    t0 = time.time()
    Vcap_sum = 0.0
    for tri in surf_faces:
        A, B, C = surf_points[tri[0]], surf_points[tri[1]], surf_points[tri[2]]
        Vcap, _ = solid_angle_cap(A, B, C, ax, ay, az)
        Vcap_sum += Vcap
    elapsed_time = time.time() - t0

    rel_error_piecewise = abs(V_piecewise - V_theory) / V_theory * 100
    rel_error_vcap_sum = abs(Vcap_sum + V_piecewise - V_theory) / V_theory * 100

    summary = {
        'prefix': prefix,
        'num_tet': len(tet_cells),
        'rel_error_piecewise': rel_error_piecewise,
        'rel_error_patchsum': rel_error_vcap_sum,
        'elapsed_time_sec': elapsed_time,
        'triangle_correction_csv': "NA",
    }
    return summary

# Optional: allow testing from the command line for a single mesh
if __name__ == "__main__":
    tet_mesh_file = "Ellip_0_sub0_tet.msh"
    surf_mesh_file = "Ellip_0_sub0_surf.msh"
    ax, ay, az = 1.5, 1.0, 0.8
    prefix = "Ellip_0"
    summary = compute_volume_error_solid_angle(tet_mesh_file, surf_mesh_file, ax, ay, az, prefix)
    print("Solid angle patch volume summary:")
    print(summary)
