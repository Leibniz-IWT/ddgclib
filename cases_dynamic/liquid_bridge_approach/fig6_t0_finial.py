#!/usr/bin/env python3
"""Create the packaged Pitois et al. (2000) Fig. 6 approach t0 mesh.

This script is intentionally self-contained for the upstream PR.  The
tetrahedral source mesh committed in ``fig6_t0_finial/mesh`` was generated
with the PR #35 / Case 12 Gmsh mesh path, using the Fig. 6 approach geometry:

    R = 4.0 mm, V = 1.10 uL, D0/R = 0.200, v = 10 um/s,
    r_CL(t0) = 1.071 mm.

The final solver mesh adds explicit boundary triangle elements to that source
tetrahedral mesh.  Those surface triangles are used by the wall-traction and
GIF/PNG diagnostics; the tetrahedra are unchanged.

References/equation map:
    Pitois, Moucheront, and Chateau, J. Colloid Interface Sci. 231 (2000),
    DOI: 10.1006/jcis.2000.7096: Fig. 6 approach experiment and Eq. (6)
    contact-radius estimate.  Eq. (6) is kept as a diagnostic here; it gives a
    nearly cylindrical bridge for the Fig. 6 t0 gap and was not used as the
    final approach contact radius.

    PR #35 / Case 12 separation workflow:
    source of the Gmsh mesh style and ddgclib force-reporting recipe.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "fig6_t0_finial"
DEFAULT_SOURCE_MESH = DEFAULT_OUT_DIR / "mesh" / "fig6_approach_t0_source_gmsh_iter0012.msh"
DEFAULT_FINAL_MESH = DEFAULT_OUT_DIR / "mesh" / "fig6_approach_t0.msh"

PARTICLE_RADIUS_M = 4.0e-3
BRIDGE_VOLUME_M3 = 1.10e-9
APPROACH_START_D_OVER_R = 0.200
FINAL_CONTACT_RADIUS_M = 1.071e-3
FIG6_APPROACH_SPEED_MPS = 10.0e-6


def pitois_eq6_contact_radius(*, radius_m: float, d_over_r: float, volume_m3: float) -> float:
    """Return the Pitois Eq. (6) radius for comparison only."""

    gap = float(d_over_r) * float(radius_m)
    height_at_contact = math.sqrt(gap * gap + 2.0 * float(volume_m3) / (math.pi * float(radius_m)))
    return math.sqrt(max(float(radius_m) * (height_at_contact - gap), 0.0))


def _parse_gmsh22(path: Path) -> tuple[list[str], list[str], list[list[int]]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        nodes_start = lines.index("$Nodes")
        nodes_end = lines.index("$EndNodes")
        elements_start = lines.index("$Elements")
        elements_end = lines.index("$EndElements")
    except ValueError as exc:
        raise ValueError(f"{path} is not a Gmsh 2.2 mesh with Nodes/Elements blocks") from exc

    node_lines = lines[nodes_start + 2 : nodes_end]
    element_lines = lines[elements_start + 2 : elements_end]
    tets: list[list[int]] = []
    for line in element_lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        element_type = int(parts[1])
        n_tags = int(parts[2])
        node_ids = [int(item) for item in parts[3 + n_tags :]]
        if element_type == 4 and len(node_ids) == 4:
            tets.append(node_ids)
    if not tets:
        raise ValueError(f"{path} contains no 4-node tetrahedra")
    return lines[:3], node_lines, tets


def _boundary_faces(tets: Iterable[list[int]]) -> list[tuple[int, int, int]]:
    counts: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for a, b, c, d in tets:
        faces = ((a, b, c), (a, d, b), (a, c, d), (b, d, c))
        for face in faces:
            key = tuple(sorted(face))
            counts[key] = None if key in counts else face
    return [face for face in counts.values() if face is not None]


def write_gmsh22_with_boundary_triangles(source_mesh: Path, final_mesh: Path) -> dict[str, int | str]:
    header, node_lines, tets = _parse_gmsh22(source_mesh)
    faces = _boundary_faces(tets)
    final_mesh.parent.mkdir(parents=True, exist_ok=True)

    element_lines: list[str] = []
    element_id = 1
    for n1, n2, n3 in faces:
        # type 2 = triangle, two tags: physical entity 2, elementary entity 2
        element_lines.append(f"{element_id} 2 2 2 2 {n1} {n2} {n3}")
        element_id += 1
    for n1, n2, n3, n4 in tets:
        # type 4 = tetrahedron, two tags: physical entity 1, elementary entity 1
        element_lines.append(f"{element_id} 4 2 1 1 {n1} {n2} {n3} {n4}")
        element_id += 1

    with final_mesh.open("w", encoding="utf-8") as handle:
        handle.write("$MeshFormat\n")
        handle.write("2.2 0 8\n")
        handle.write("$EndMeshFormat\n")
        handle.write("$Nodes\n")
        handle.write(f"{len(node_lines)}\n")
        handle.write("\n".join(node_lines))
        handle.write("\n$EndNodes\n")
        handle.write("$Elements\n")
        handle.write(f"{len(element_lines)}\n")
        handle.write("\n".join(element_lines))
        handle.write("\n$EndElements\n")

    return {
        "source_mesh": str(source_mesh.resolve()),
        "final_mesh": str(final_mesh.resolve()),
        "node_count": len(node_lines),
        "surface_triangle_count": len(faces),
        "tetrahedron_count": len(tets),
        "element_count": len(element_lines),
    }


def build_summary(mesh_info: dict[str, int | str]) -> dict[str, object]:
    eq6_rcl = pitois_eq6_contact_radius(
        radius_m=PARTICLE_RADIUS_M,
        d_over_r=APPROACH_START_D_OVER_R,
        volume_m3=BRIDGE_VOLUME_M3,
    )
    return {
        "case": "Pitois 2000 Fig. 6 approach t0 mesh",
        "geometry": {
            "particle_radius_m": PARTICLE_RADIUS_M,
            "bridge_volume_m3": BRIDGE_VOLUME_M3,
            "initial_d_over_r": APPROACH_START_D_OVER_R,
            "approach_speed_mps": FIG6_APPROACH_SPEED_MPS,
            "final_contact_radius_m": FINAL_CONTACT_RADIUS_M,
            "pitois_eq6_contact_radius_m": eq6_rcl,
            "note": "Eq. (6) is diagnostic only; final r_CL was selected to match the t0 approach geometry.",
        },
        "mesh": mesh_info,
        "references": [
            "Pitois, Moucheront, Chateau, J. Colloid Interface Sci. 231 (2000), DOI: 10.1006/jcis.2000.7096, Fig. 6 and Eq. (6).",
            "Leibniz-IWT/ddgclib PR #35 Case 12 separation workflow, used as the mesh/force-reporting base.",
        ],
    }


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", default=str(DEFAULT_SOURCE_MESH))
    parser.add_argument("--out-mesh", default=str(DEFAULT_FINAL_MESH))
    parser.add_argument("--summary", default=str(DEFAULT_OUT_DIR / "summary.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_cli().parse_args(argv)
    source_mesh = Path(args.source_mesh)
    final_mesh = Path(args.out_mesh)
    if not source_mesh.is_file():
        raise FileNotFoundError(source_mesh)

    mesh_info = write_gmsh22_with_boundary_triangles(source_mesh, final_mesh)
    summary = build_summary(mesh_info)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {final_mesh}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
