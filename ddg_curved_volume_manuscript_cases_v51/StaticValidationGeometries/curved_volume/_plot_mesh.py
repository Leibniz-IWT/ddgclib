#!/usr/bin/env python3
# main.py — embed and display Ellip_0_sub0_full.msh

from pathlib import Path
import sys

MESH_FILE = Path(__file__).with_name("Ellip_0_sub0_full.msh")

def _boundary_triangles_from_cells(points, cells):
    """
    Build boundary triangles from meshio cells.
    - If triangle cells exist, use them.
    - Else, extract boundary faces from tets/hex/wedge/pyramid.
    """
    import numpy as np
    tri_blocks = [c.data for c in cells if c.type == "triangle"]
    if tri_blocks:
        tris = tri_blocks[0]
        # If multiple triangle blocks, concatenate:
        if len(tri_blocks) > 1:
            tris = np.vstack(tri_blocks)
        return points[tris]

    # Otherwise, accumulate faces and keep those that occur once (boundary)
    face_counts = {}
    def add_face(face):
        key = tuple(sorted(face))
        face_counts[key] = face_counts.get(key, 0) + 1

    for cb in cells:
        ct = cb.type
        conn = cb.data
        if ct == "tetra":
            faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        elif ct == "hexahedron":
            faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4),
                     (1,2,6,5), (2,3,7,6), (3,0,4,7)]
        elif ct == "wedge":
            faces = [(0,1,2), (3,4,5), (0,1,4,3), (1,2,5,4), (2,0,3,5)]
        elif ct == "pyramid":
            faces = [(0,1,2,3), (0,1,4), (1,2,4), (2,3,4), (3,0,4)]
        else:
            continue
        for cell in conn:
            for f in faces:
                add_face(cell[list(f)])

    # Keep only faces seen once (boundary), and triangulate quads
    tris = []
    for face, cnt in face_counts.items():
        if cnt != 1:
            continue
        if len(face) == 3:
            tris.append(face)
        elif len(face) == 4:
            a,b,c,d = face
            tris.append((a,b,c))
            tris.append((a,c,d))
    tris = np.array(tris, dtype=int)
    return points[tris]

def show_mesh(mesh_path: Path):
    try:
        import meshio
    except ImportError:
        sys.exit("Missing dependency 'meshio'. Install with: pip install meshio")

    mesh = meshio.read(str(mesh_path))
    print(f"[loaded] {mesh_path.name}")
    print(f"  points: {mesh.points.shape[0]}")
    for cb in mesh.cells:
        print(f"  cells[{cb.type}]: {len(cb.data)}")

    # Try interactive (PyVista)
    try:
        import pyvista as pv
        pv_mesh = pv.from_meshio(mesh)
        # For volumes, show surface for clarity
        plot_mesh = pv_mesh
        try:
            plot_mesh = pv_mesh.extract_surface().triangulate()
            if plot_mesh.n_points == 0:
                plot_mesh = pv_mesh
        except Exception:
            pass

        p = pv.Plotter()
        p.add_axes()
        p.add_mesh(plot_mesh, show_edges=True)
        p.show(title=mesh_path.name)
        return
    except Exception as e:
        print(f"[info] PyVista unavailable or failed ({e}). Using Matplotlib fallback.")

    # Fallback: Matplotlib surface preview
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
    except ImportError:
        sys.exit("Matplotlib fallback requires 'matplotlib'. Install with: pip install matplotlib")

    tris_xyz = _boundary_triangles_from_cells(mesh.points, mesh.cells)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    coll = Poly3DCollection(tris_xyz, linewidths=0.25, alpha=0.8)
    coll.set_edgecolor("k")
    coll.set_facecolor((0.7, 0.8, 1.0, 0.6))
    ax.add_collection3d(coll)

    pts = mesh.points
    ax.auto_scale_xyz(pts[:,0], pts[:,1], pts[:,2])
    ax.set_title(mesh_path.name)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not MESH_FILE.exists():
        sys.exit(f"Mesh not found next to main.py: {MESH_FILE}")
    show_mesh(MESH_FILE)
