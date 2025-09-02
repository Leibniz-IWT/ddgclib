# benchmarks/_benchmark_plotting_utils.py
from __future__ import annotations
import numpy as np

def read_gmsh_tri(path: str):
    """
    Load a triangle surface mesh from a Gmsh .msh file.

    Returns
    -------
    points : (N,3) float64
    tris   : (M,3) int64 (0-based)
    """
    # ---- Preferred: meshio (supports v2/v4, ascii/binary) ----
    try:
        import meshio  # pip install meshio
        m = meshio.read(path)
        P = np.asarray(m.points, dtype=float)
        if P.shape[1] == 2:  # pad Z if 2D
            P = np.c_[P, np.zeros(len(P), dtype=float)]

        tri_blocks = []
        for c in getattr(m, "cells", []):
            if c.type in ("triangle", "tri3", "triangle3"):
                tri_blocks.append(np.asarray(c.data, dtype=np.int64))
        if not tri_blocks:
            # some versions store in cell_sets; try a generic grab
            for name, data in getattr(m, "cells_dict", {}).items():
                if name in ("triangle", "tri3", "triangle3"):
                    tri_blocks.append(np.asarray(data, dtype=np.int64))

        if not tri_blocks:
            raise RuntimeError("No triangle cells found in file.")

        T = np.concatenate(tri_blocks, axis=0)
        return P, T

    except Exception:
        # ---- Fallback: very small ASCII Gmsh v2 parser (triangles only) ----
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
        except Exception as e:
            raise ImportError(f"Failed to open {path}: {e}")

        # Find $Nodes ... $EndNodes
        try:
            i = lines.index("$Nodes") + 1
            n_nodes = int(lines[i]); i += 1
        except Exception as e:
            raise ImportError("Gmsh ASCII v2: missing $Nodes section") from e

        nodes = []
        id_to_idx = {}
        for k in range(n_nodes):
            parts = lines[i + k].strip().split()
            # id x y z
            nid = int(parts[0])
            xyz = [float(parts[1]), float(parts[2]), float(parts[3]) if len(parts) > 3 else 0.0]
            id_to_idx[nid] = len(nodes)
            nodes.append(xyz)
        i = i + n_nodes
        # Skip to $Elements
        try:
            j = lines.index("$Elements") + 1
            n_elem = int(lines[j]); j += 1
        except Exception as e:
            raise ImportError("Gmsh ASCII v2: missing $Elements section") from e

        tris = []
        for k in range(n_elem):
            parts = lines[j + k].strip().split()
            # format: id type ntags [tags...] n1 n2 n3 ...
            # type=2 is 3-node triangle in Gmsh v2
            etype = int(parts[1])
            ntags = int(parts[2])
            if etype == 2:
                n1, n2, n3 = map(int, parts[3 + ntags : 3 + ntags + 3])
                tris.append([id_to_idx[n1], id_to_idx[n2], id_to_idx[n3]])

        if not tris:
            raise ImportError("No triangle (etype=2) elements found in ASCII v2 fallback.")

        P = np.asarray(nodes, dtype=float)
        T = np.asarray(tris, dtype=np.int64)
        return P, T
