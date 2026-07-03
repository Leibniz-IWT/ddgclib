#!/usr/bin/env python3
"""Create all method-labeled mesh-preview PNGs for Figure 3-8.

Run from this folder with:
    python3 mesh_plot.py

Static panels 3-8a--3-8f use four refinement levels.  Dynamic panels 3-8g and
3-8h use only the initial saved surface state, t0.
Each preview is labeled by
method so a reader can see whether a method uses the surface mesh, the
Evrard-type background tetrahedral mesh, or the Strobl sphere/hex grid.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import io
import math
import re
import sys
from pathlib import Path
from time import perf_counter


HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bootstrap import ensure_runtime  # noqa: E402

ensure_runtime()

import meshio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import cylinder_mesh  # noqa: E402
import hyperbolic_cylinder_mesh  # noqa: E402
import hyperboloid_mesh  # noqa: E402
import parabolic_cylinder_mesh  # noqa: E402
import paraboloid_mesh  # noqa: E402
import sphere_mesh  # noqa: E402
import strobl_2016  # noqa: E402


OUT_DIR = HERE / "mesh"
STATIC_CASES = [
    ("a", "sphere", sphere_mesh.generate_mesh, [0, 1, 2, 3]),
    ("b", "cylinder", cylinder_mesh.generate_mesh, [0, 1, 2, 3]),
    ("c", "paraboloid", paraboloid_mesh.generate_mesh, [0, 1, 2, 3]),
    ("d", "hyperboloid", hyperboloid_mesh.generate_mesh, [0, 1, 2, 3]),
    ("e", "parabolic_cylinder", parabolic_cylinder_mesh.generate_mesh, [1, 2, 3, 4]),
    ("f", "hyperbolic_cylinder", hyperbolic_cylinder_mesh.generate_mesh, [0, 1, 2, 3]),
]
DYNAMIC_CASES = [
    (
        "g",
        "cube2sphere",
        HERE / "source_data" / "figure_3_3" / "cube_present_surface_states",
    ),
    (
        "h",
        "droplet_oscillation",
        HERE / "source_data" / "figure_3_7" / "droplet_present_surface_states",
    ),
]
METHODS = [
    ("PLIC_PL", "PLIC / PL"),
    ("Evrard_type_paraboloid", "Evrard-type paraboloid"),
    ("THINC_QQ", "THINC/QQ"),
    ("Strobl_sphere_overlap", "Strobl sphere/hex overlap"),
    ("Present_quadric_patch", "Present quadric patch"),
]
STROBL_HEXGRID_N = 4


def compact_faces(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    used = np.unique(faces.ravel())
    remap = {int(old): i for i, old in enumerate(used)}
    compact = np.vectorize(lambda idx: remap[int(idx)])(faces).astype(int)
    return points[used], compact


def mesh_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(i), int(j)))))
    return sorted(edges)


def edge_point_compact(points: np.ndarray, edges: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    used = sorted({idx for edge in edges for idx in edge})
    remap = {old: new for new, old in enumerate(used)}
    return points[used], [(remap[i], remap[j]) for i, j in edges]


def load_msh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mesh = meshio.read(path)
    triangles = [block.data for block in mesh.cells if block.type == "triangle"]
    if not triangles:
        raise ValueError(f"No triangle cells found in {path}")
    return np.asarray(mesh.points[:, :3], dtype=float), np.vstack(triangles).astype(int)


def iter_number_from_name(path: Path) -> int:
    match = re.search(r"surface_iter_(\d+)\.msh$", path.name)
    if not match:
        raise ValueError(f"Cannot parse iteration number from {path.name}")
    return int(match.group(1))


def projected_points(points: np.ndarray, width: int, height: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    centered = points - center

    azim = math.radians(-38.0)
    elev = math.radians(24.0)
    rz = np.array(
        [
            [math.cos(azim), -math.sin(azim), 0.0],
            [math.sin(azim), math.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elev), -math.sin(elev)],
            [0.0, math.sin(elev), math.cos(elev)],
        ]
    )
    rotated = centered @ (rz @ rx).T
    xy = rotated[:, :2]
    depth = rotated[:, 2]

    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1.0e-12)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    xy = (xy - 0.5 * (xy.min(axis=0) + xy.max(axis=0))) * scale
    xy[:, 0] += 0.5 * width
    xy[:, 1] = 0.5 * height - xy[:, 1]
    return xy, depth


def canvas(width: int, height: int, title: str, subtitle: str):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((14, 12), title, fill=(0, 0, 0), font=font)
    draw.text((14, height - 34), subtitle, fill=(45, 45, 45), font=font)
    return image, draw


def draw_surface_png(
    points: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
    *,
    width: int = 900,
    height: int = 700,
) -> None:
    points, faces = compact_faces(points, faces)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = canvas(width, height, title, subtitle)

    plot_top = 50
    plot_bottom = height - 55
    xy, depth = projected_points(points, width, plot_bottom - plot_top, margin=55)
    xy[:, 1] += plot_top

    face_depth = np.asarray([depth[face].mean() for face in faces], dtype=float)
    depth_span = max(np.ptp(face_depth), 1.0e-12)
    for idx in np.argsort(face_depth):
        polygon = [tuple(xy[int(vertex)]) for vertex in faces[idx]]
        shade = int(228 - 36 * (face_depth[idx] - face_depth.min()) / depth_span)
        # Draw each face and its outline in painter order. This hides back-side
        # edges on closed surfaces instead of overlaying every edge afterward.
        draw.polygon(
            polygon,
            fill=(shade, min(245, shade + 14), 255),
            outline=(70, 96, 120),
        )

    image.save(out_path, "PNG", optimize=True)


def draw_cube_surface_png(
    points: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
    *,
    width: int = 900,
    height: int = 700,
) -> None:
    """Draw the cube-to-sphere t0 surface as a readable 3D closed object."""
    points, faces = compact_faces(points, faces)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.015, 0.975, title, ha="left", va="top", fontsize=8, color="black")
    fig.text(0.015, 0.035, subtitle, ha="left", va="bottom", fontsize=7, color=(0.18, 0.18, 0.18))

    ax = fig.add_axes([0.04, 0.08, 0.92, 0.84], projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=23, azim=-42)
    ax.set_axis_off()

    tri_vertices = [points[face] for face in faces]
    surface = Poly3DCollection(
        tri_vertices,
        facecolors=(0.70, 0.78, 0.97, 0.72),
        edgecolors=(0.07, 0.22, 0.40, 0.82),
        linewidths=0.45,
    )
    ax.add_collection3d(surface)

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    corners = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=float,
    )
    box_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for i, j in box_edges:
        xs, ys, zs = zip(corners[i], corners[j])
        ax.plot(xs, ys, zs, color=(0.02, 0.12, 0.25), linewidth=2.0)
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=13, color=(0.02, 0.12, 0.25), depthshade=False)

    center = 0.5 * (lo + hi)
    radius = 0.56 * max(float(np.max(hi - lo)), 1.0e-12)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))

    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def draw_wire_png(
    points: np.ndarray,
    edges: list[tuple[int, int]],
    out_path: Path,
    title: str,
    subtitle: str,
    *,
    width: int = 900,
    height: int = 700,
) -> None:
    points, edges = edge_point_compact(np.asarray(points, dtype=float), edges)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = canvas(width, height, title, subtitle)

    plot_top = 50
    plot_bottom = height - 55
    xy, depth = projected_points(points, width, plot_bottom - plot_top, margin=55)
    xy[:, 1] += plot_top

    edge_depth = [(0.5 * (depth[i] + depth[j]), i, j) for i, j in edges]
    for _z, i, j in sorted(edge_depth):
        draw.line([tuple(xy[i]), tuple(xy[j])], fill=(42, 82, 122), width=1)
    for x, y in xy:
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(18, 62, 104))

    image.save(out_path, "PNG", optimize=True)


def draw_evrard_background_png(
    level: int,
    out_path: Path,
    title: str,
    subtitle: str,
    *,
    width: int = 900,
    height: int = 700,
) -> None:
    """Draw a readable 3D cutaway of the Evrard background tet grid."""
    nseg, nr, nz = paraboloid_mesh.EVRARD_BACKGROUND_LEVELS[level]
    top_z = paraboloid_mesh.TOP_Z
    top_radius = paraboloid_mesh.TOP_RADIUS
    outer_radius = top_radius / math.cos(math.pi / nseg) * (1.0 + 1.0e-12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.015, 0.975, title, ha="left", va="top", fontsize=8, color="black")
    fig.text(0.015, 0.035, subtitle, ha="left", va="bottom", fontsize=7, color=(0.18, 0.18, 0.18))

    ax = fig.add_axes([0.03, 0.08, 0.94, 0.84], projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=23, azim=-46)
    ax.set_axis_off()
    ax.set_box_aspect((2.0 * outer_radius, 2.0 * outer_radius, top_z))

    blue = (0.24, 0.45, 0.68)
    pale = (0.78, 0.88, 0.97)
    red = (0.82, 0.20, 0.20)

    theta_count = min(nseg, 24)
    theta = np.linspace(0.0, 2.0 * math.pi, theta_count + 1)
    z_values = np.linspace(0.0, top_z, min(nz, 8) + 1)
    r_values = np.linspace(0.0, outer_radius, min(nr, 7) + 1)

    # Outer 3D background-grid cylinder/prism wireframe.
    for z in z_values:
        x = outer_radius * np.cos(theta)
        y = outer_radius * np.sin(theta)
        ax.plot(x, y, np.full_like(theta, z), color=blue, lw=0.65, alpha=0.72)
    for th in theta[:-1]:
        x = outer_radius * math.cos(th)
        y = outer_radius * math.sin(th)
        ax.plot([x, x], [y, y], [0.0, top_z], color=blue, lw=0.65, alpha=0.72)
    for z in (0.0, top_z):
        for r in r_values[1:]:
            ax.plot(r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z), color=pale, lw=0.6)
        for th in theta[:-1]:
            ax.plot([0.0, outer_radius * math.cos(th)], [0.0, outer_radius * math.sin(th)], [z, z], color=pale, lw=0.6)

    # Two cut planes expose the radial/z layering and representative prism/tet
    # diagonals, avoiding the visual impression of a flat 2D r-z plot.
    cut_thetas = (math.radians(-35.0), math.radians(65.0))
    cut_r = np.linspace(0.0, outer_radius, min(nr, 8) + 1)
    cut_z = np.linspace(0.0, top_z, min(nz, 8) + 1)
    for plane_id, th in enumerate(cut_thetas):
        c, s = math.cos(th), math.sin(th)
        for z in cut_z:
            ax.plot(cut_r * c, cut_r * s, np.full_like(cut_r, z), color=(0.33, 0.55, 0.75), lw=0.7, alpha=0.78)
        for r in cut_r:
            ax.plot([r * c, r * c], [r * s, r * s], [0.0, top_z], color=(0.33, 0.55, 0.75), lw=0.7, alpha=0.78)
        for iz in range(len(cut_z) - 1):
            for ir in range(len(cut_r) - 1):
                if (ir + iz + plane_id) % 2 == 0:
                    rr = [cut_r[ir], cut_r[ir + 1]]
                    zz = [cut_z[iz], cut_z[iz + 1]]
                else:
                    rr = [cut_r[ir], cut_r[ir + 1]]
                    zz = [cut_z[iz + 1], cut_z[iz]]
                ax.plot([rr[0] * c, rr[1] * c], [rr[0] * s, rr[1] * s], zz, color=(0.60, 0.72, 0.84), lw=0.45, alpha=0.8)

    # Prescribed paraboloid z=r^2, drawn as a red 3D wire surface inside the
    # background mesh.
    rr = np.linspace(0.0, top_radius, 14)
    tt = np.linspace(0.0, 2.0 * math.pi, 37)
    for r in rr[1:]:
        ax.plot(r * np.cos(tt), r * np.sin(tt), np.full_like(tt, r * r), color=red, lw=1.25, alpha=0.9)
    for th in np.linspace(0.0, 2.0 * math.pi, 13, endpoint=False):
        rline = np.linspace(0.0, top_radius, 160)
        ax.plot(rline * math.cos(th), rline * math.sin(th), rline * rline, color=red, lw=1.0, alpha=0.75)

    ax.plot([], [], [], color=blue, lw=1.6, label="3D background tetra grid")
    ax.plot([], [], [], color=red, lw=2.0, label="prescribed paraboloid z = r^2")
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=8, frameon=False)
    pad = 0.08 * outer_radius
    ax.set_xlim(-outer_radius - pad, outer_radius + pad)
    ax.set_ylim(-outer_radius - pad, outer_radius + pad)
    ax.set_zlim(0.0, top_z)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def draw_strobl_hexgrid_png(
    points: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
    *,
    width: int = 900,
    height: int = 700,
) -> tuple[int, int, str]:
    """Draw a readable central cross-section of the fitted sphere/hex grid."""
    sphere = strobl_2016.fit_sphere(np.asarray(points, dtype=float), np.arange(len(points)))
    if sphere is None:
        raise ValueError("Could not fit a sphere for Strobl mesh preview.")
    center, radius = sphere
    center = np.asarray(center, dtype=float)
    radius = float(radius)
    pad = 1.05 * radius
    axes = [np.linspace(center[d] - pad, center[d] + pad, STROBL_HEXGRID_N + 1) for d in range(3)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = canvas(width, height, title, subtitle)

    grid_pts = []
    grid_ids = {}

    def grid_node(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        if key not in grid_ids:
            grid_ids[key] = len(grid_pts)
            grid_pts.append((axes[0][i], axes[1][j], axes[2][k]))
        return grid_ids[key]

    grid_edges: set[tuple[int, int]] = set()
    for i in range(STROBL_HEXGRID_N + 1):
        for j in range(STROBL_HEXGRID_N + 1):
            for k in range(STROBL_HEXGRID_N + 1):
                if i < STROBL_HEXGRID_N:
                    grid_edges.add(tuple(sorted((grid_node(i, j, k), grid_node(i + 1, j, k)))))
                if j < STROBL_HEXGRID_N:
                    grid_edges.add(tuple(sorted((grid_node(i, j, k), grid_node(i, j + 1, k)))))
                if k < STROBL_HEXGRID_N:
                    grid_edges.add(tuple(sorted((grid_node(i, j, k), grid_node(i, j, k + 1)))))

    unit_points = np.zeros((len(grid_ids), 3), dtype=float)
    for key, idx in grid_ids.items():
        unit_points[idx] = key
    i = unit_points[:, 0]
    j = unit_points[:, 1]
    k = unit_points[:, 2]
    # Isometric-like projection of the logical 4 x 4 x 4 brick lattice. Using
    # logical indices instead of physical coordinates keeps the hexahedral
    # volume grid visually cubic for every fitted sphere size.
    grid_xy = np.column_stack(((i - j) * 0.95, (i + j) * 0.48 - k * 0.92))
    grid_depth = i + j + k

    left, top = 72, 82
    grid_w, grid_h = 540, 500
    span = np.maximum(grid_xy.max(axis=0) - grid_xy.min(axis=0), 1.0e-12)
    scale = min(grid_w / span[0], grid_h / span[1]) * 0.88
    grid_xy = (grid_xy - 0.5 * (grid_xy.min(axis=0) + grid_xy.max(axis=0))) * scale
    grid_xy[:, 0] += left + 0.5 * grid_w
    grid_xy[:, 1] = top + 0.5 * grid_h - grid_xy[:, 1]

    font = ImageFont.load_default()
    draw.text((left, top - 18), "3D hexahedral overlap grid", fill=(20, 45, 70), font=font)

    def idx(a: int, b: int, c: int) -> int:
        return grid_ids[(a, b, c)]

    quads: list[tuple[float, list[int], tuple[int, int, int]]] = []
    face_fills = {
        "i0": (232, 240, 250),
        "iN": (205, 224, 244),
        "j0": (218, 232, 247),
        "jN": (238, 244, 251),
        "k0": (226, 237, 249),
        "kN": (199, 220, 242),
    }

    n = STROBL_HEXGRID_N
    for fixed_i, name in ((0, "i0"), (n, "iN")):
        for b in range(n):
            for c in range(n):
                verts = [idx(fixed_i, b, c), idx(fixed_i, b + 1, c), idx(fixed_i, b + 1, c + 1), idx(fixed_i, b, c + 1)]
                quads.append((float(np.mean(grid_depth[verts])), verts, face_fills[name]))
    for fixed_j, name in ((0, "j0"), (n, "jN")):
        for a in range(n):
            for c in range(n):
                verts = [idx(a, fixed_j, c), idx(a + 1, fixed_j, c), idx(a + 1, fixed_j, c + 1), idx(a, fixed_j, c + 1)]
                quads.append((float(np.mean(grid_depth[verts])), verts, face_fills[name]))
    for fixed_k, name in ((0, "k0"), (n, "kN")):
        for a in range(n):
            for b in range(n):
                verts = [idx(a, b, fixed_k), idx(a + 1, b, fixed_k), idx(a + 1, b + 1, fixed_k), idx(a, b + 1, fixed_k)]
                quads.append((float(np.mean(grid_depth[verts])), verts, face_fills[name]))

    for _depth, verts, fill in sorted(quads, key=lambda item: item[0]):
        polygon = [tuple(grid_xy[v]) for v in verts]
        draw.polygon(polygon, fill=fill, outline=(82, 120, 157))

    outer_edges: set[tuple[int, int]] = set()
    for a in (0, n):
        for b in (0, n):
            for c in range(n):
                outer_edges.add(tuple(sorted((idx(a, b, c), idx(a, b, c + 1)))))
    for a in (0, n):
        for c in (0, n):
            for b in range(n):
                outer_edges.add(tuple(sorted((idx(a, b, c), idx(a, b + 1, c)))))
    for b in (0, n):
        for c in (0, n):
            for a in range(n):
                outer_edges.add(tuple(sorted((idx(a, b, c), idx(a + 1, b, c)))))

    for a, b in sorted(outer_edges):
        draw.line([tuple(grid_xy[a]), tuple(grid_xy[b])], fill=(20, 65, 108), width=3)

    side_left, side_top = 650, 90
    side_size = 190
    xmin, xmax = axes[0][0], axes[0][-1]
    ymin, ymax = axes[1][0], axes[1][-1]

    def px(x: float) -> float:
        return side_left + (float(x) - xmin) / (xmax - xmin) * side_size

    def py(y: float) -> float:
        return side_top + side_size - (float(y) - ymin) / (ymax - ymin) * side_size

    draw.text((side_left, side_top - 20), "central slice", fill=(20, 45, 70), font=font)
    draw.rectangle((side_left, side_top, side_left + side_size, side_top + side_size), fill=(244, 249, 255), outline=(95, 116, 138))
    for x in axes[0]:
        draw.line((px(x), side_top, px(x), side_top + side_size), fill=(128, 151, 174), width=1)
    for y in axes[1]:
        draw.line((side_left, py(y), side_left + side_size, py(y)), fill=(128, 151, 174), width=1)
    bbox = (
        px(center[0] - radius),
        py(center[1] + radius),
        px(center[0] + radius),
        py(center[1] - radius),
    )
    draw.ellipse(bbox, outline=(205, 52, 52), width=3)
    draw.ellipse((px(center[0]) - 3, py(center[1]) - 3, px(center[0]) + 3, py(center[1]) + 3), fill=(205, 52, 52))
    draw.line((side_left, side_top + side_size + 38, side_left + 50, side_top + side_size + 38), fill=(55, 93, 131), width=3)
    draw.text((side_left + 60, side_top + side_size + 31), "hexahedral grid", fill=(20, 45, 70), font=font)
    draw.line((side_left, side_top + side_size + 64, side_left + 50, side_top + side_size + 64), fill=(205, 52, 52), width=3)
    draw.text((side_left + 60, side_top + side_size + 57), "fitted sphere", fill=(20, 45, 70), font=font)
    draw.text(
        (side_left, side_top + side_size + 96),
        f"full grid: {STROBL_HEXGRID_N} x {STROBL_HEXGRID_N} x {STROBL_HEXGRID_N} hexahedra",
        fill=(20, 45, 70),
        font=font,
    )

    image.save(out_path, "PNG", optimize=True)
    note = f"fitted sphere center=({center[0]:.3g},{center[1]:.3g},{center[2]:.3g}), R={radius:.3g}, hexgrid={STROBL_HEXGRID_N}^3"
    return (STROBL_HEXGRID_N + 1) ** 3, 3 * STROBL_HEXGRID_N * (STROBL_HEXGRID_N + 1) ** 2, note


def tet_boundary_faces(tets: np.ndarray) -> np.ndarray:
    counts: Counter[tuple[int, int, int]] = Counter()
    oriented: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tet in np.asarray(tets, dtype=int):
        a, b, c, d = [int(x) for x in tet]
        for face in ((a, b, c), (a, d, b), (b, d, c), (c, d, a)):
            key = tuple(sorted(face))
            counts[key] += 1
            oriented.setdefault(key, face)
    return np.asarray([oriented[key] for key, count in counts.items() if count == 1], dtype=int)


def hexgrid_edges_from_surface(points: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], str]:
    sphere = strobl_2016.fit_sphere(points, np.arange(len(points)))
    if sphere is None:
        raise ValueError("Could not fit a sphere for Strobl mesh preview.")
    center, radius = sphere
    center = np.asarray(center, dtype=float)
    pad = 1.05 * float(radius)
    axes = [np.linspace(center[d] - pad, center[d] + pad, STROBL_HEXGRID_N + 1) for d in range(3)]

    pts: list[tuple[float, float, float]] = []
    ids: dict[tuple[int, int, int], int] = {}

    def node(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        if key not in ids:
            ids[key] = len(pts)
            pts.append((float(axes[0][i]), float(axes[1][j]), float(axes[2][k])))
        return ids[key]

    edges: set[tuple[int, int]] = set()
    for i in range(STROBL_HEXGRID_N + 1):
        for j in range(STROBL_HEXGRID_N + 1):
            for k in range(STROBL_HEXGRID_N + 1):
                if i < STROBL_HEXGRID_N:
                    edges.add(tuple(sorted((node(i, j, k), node(i + 1, j, k)))))
                if j < STROBL_HEXGRID_N:
                    edges.add(tuple(sorted((node(i, j, k), node(i, j + 1, k)))))
                if k < STROBL_HEXGRID_N:
                    edges.add(tuple(sorted((node(i, j, k), node(i, j, k + 1)))))
    note = f"fitted sphere center=({center[0]:.3g},{center[1]:.3g},{center[2]:.3g}), R={radius:.3g}, hexgrid={STROBL_HEXGRID_N}^3"
    return np.asarray(pts, dtype=float), sorted(edges), note


def clean_mesh_dir() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for path in OUT_DIR.glob("*.png"):
        path.unlink()
    for name in ("manifest.csv", "README.txt"):
        path = OUT_DIR / name
        if path.exists():
            path.unlink()


def write_readme(total_pngs: int) -> None:
    (OUT_DIR / "README.txt").write_text(
        "\n".join(
            [
                "Figure 3-8 method-labeled mesh previews",
                "",
                "This folder is generated by mesh_plot.py, which is called by recompute_all_cases.py.",
                "Static panels 3-8a through 3-8f include four refinement levels for each plotted method.",
                "Dynamic panels 3-8g and 3-8h include only the initial saved surface state t0 for each method.",
                "Most methods use the same current surface mesh. The preview is still labeled by method so this is explicit.",
                "The Evrard-type paraboloid panel uses a separate background tetrahedral mesh for its model-matched forward operator; its preview is a 3D wireframe cutaway of that full grid.",
                "The Strobl comparison uses a fitted sphere and a 4^3 hexahedral overlap grid; its preview shows the full 3D grid with a central cross-section inset.",
                "Cube-to-sphere t0 previews show the actual surface triangles with a dark bounding-box outline so the initial cube is visible.",
                "manifest.csv records panel, case, method, source, level/iteration, point/cell counts, and PNG path.",
                f"Total PNG files generated: {total_pngs}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "panel",
        "case",
        "kind",
        "method",
        "method_name",
        "mesh_role",
        "level",
        "iteration",
        "source",
        "points",
        "faces",
        "cells",
        "edges",
        "png",
    ]
    with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def surface_mesh_note(method: str) -> str:
    if method == "PLIC_PL":
        return "same surface mesh; planar PL volume baseline"
    if method == "Evrard_type_paraboloid":
        return "same surface mesh; local paraboloid Vpatch operator"
    if method == "THINC_QQ":
        return "same surface mesh; quadratic Gaussian-quadrature operator"
    if method == "Present_quadric_patch":
        return "same surface mesh; class-aware quadric Vpatch operator"
    return "surface triangle mesh"


def draw_method_preview(
    rows: list[dict[str, object]],
    *,
    letter: str,
    case: str,
    kind: str,
    method: str,
    method_name: str,
    points: np.ndarray,
    faces: np.ndarray,
    level: int | str = "",
    iteration: int | str = "",
    source: str,
) -> None:
    panel = f"3-8{letter}"
    suffix = f"level_{int(level):02d}" if level != "" else f"t{iteration}"
    out = OUT_DIR / f"Figure_{panel}_{case}_{suffix}_{safe(method)}.png"
    title = f"Figure {panel}: {case}, {method_name}"

    mesh_role = "surface triangle mesh"
    cells = ""
    edges = ""
    draw_source = source
    if method == "Strobl_sphere_overlap":
        draw_points, edges, note = draw_strobl_hexgrid_png(
            points,
            out,
            title,
            f"3D hexahedral-grid preview; {STROBL_HEXGRID_N}^3 hexahedra; source surface={source}",
        )
        subtitle = f"{note}; source surface={source}"
        mesh_role = "fitted-sphere hexahedral overlap grid (central cross-section preview)"
        draw_faces = ""
        draw_source = f"{source}; Strobl fitted-sphere hexgrid"
    elif case == "paraboloid" and kind == "static" and method == "Evrard_type_paraboloid":
        bg_points, bg_tets = paraboloid_mesh.generate_evrard_background_tets(int(level))
        bg_faces = tet_boundary_faces(bg_tets)
        subtitle = f"3D wireframe cutaway preview; full mesh points={len(bg_points)}, tets={len(bg_tets)}, boundary faces={len(bg_faces)}"
        draw_evrard_background_png(int(level), out, title, subtitle)
        mesh_role = "Evrard background tetrahedral mesh (3D wireframe cutaway preview)"
        draw_points = len(bg_points)
        draw_faces = len(bg_faces)
        cells = len(bg_tets)
        draw_source = f"paraboloid_mesh.generate_evrard_background_tets({level})"
    elif case == "cube2sphere" and kind == "dynamic_t0":
        subtitle = (
            f"{surface_mesh_note(method)}; initial cube-like closed surface t0; "
            f"source={source}; points={len(points)}, faces={len(faces)}"
        )
        draw_cube_surface_png(points, faces, out, title, subtitle)
        mesh_role = "surface triangle mesh (3D cube-outline preview)"
        draw_points = len(points)
        draw_faces = len(faces)
    else:
        subtitle = f"{surface_mesh_note(method)}; source={source}; points={len(points)}, faces={len(faces)}"
        draw_surface_png(points, faces, out, title, subtitle)
        draw_points = len(points)
        draw_faces = len(faces)

    rows.append(
        {
            "panel": panel,
            "case": case,
            "kind": kind,
            "method": method,
            "method_name": method_name,
            "mesh_role": mesh_role,
            "level": level,
            "iteration": iteration,
            "source": draw_source,
            "points": draw_points,
            "faces": draw_faces,
            "cells": cells,
            "edges": edges,
            "png": out.name,
        }
    )
    print(f"mesh: wrote {out.name}", flush=True)


def generate_static(rows: list[dict[str, object]]) -> int:
    before = len(rows)
    for letter, case, generator, levels in STATIC_CASES:
        for level in levels:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                points, faces = generator(level)
            source = f"{case}_mesh.generate_mesh({level})"
            for method, method_name in METHODS:
                draw_method_preview(
                    rows,
                    letter=letter,
                    case=case,
                    kind="static",
                    method=method,
                    method_name=method_name,
                    points=points,
                    faces=faces,
                    level=level,
                    source=source,
                )
    return len(rows) - before


def generate_dynamic_t0(rows: list[dict[str, object]]) -> int:
    before = len(rows)
    for letter, case, mesh_dir in DYNAMIC_CASES:
        paths = sorted(mesh_dir.glob("surface_iter_*.msh"), key=iter_number_from_name)
        if not paths:
            raise FileNotFoundError(f"No surface_iter_*.msh files found in {mesh_dir}")
        path = paths[0]
        iteration = iter_number_from_name(path)
        if iteration != 0:
            print(f"mesh: first {case} mesh is iteration {iteration}, not 0", flush=True)
        points, faces = load_msh(path)
        source = str(path.relative_to(HERE))
        for method, method_name in METHODS:
            draw_method_preview(
                rows,
                letter=letter,
                case=case,
                kind="dynamic_t0",
                method=method,
                method_name=method_name,
                points=points,
                faces=faces,
                iteration=iteration,
                source=source,
            )
    return len(rows) - before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dynamic-all",
        action="store_true",
        help="Deprecated compatibility flag; dynamic previews are intentionally limited to t0.",
    )
    args = parser.parse_args()
    if args.dynamic_all:
        print("mesh: --dynamic-all ignored; dynamic previews are limited to t0 by design.", flush=True)

    start = perf_counter()
    clean_mesh_dir()
    rows: list[dict[str, object]] = []
    n_static = generate_static(rows)
    n_dynamic = generate_dynamic_t0(rows)
    write_manifest(rows)
    total = n_static + n_dynamic
    write_readme(total)
    print(f"Generated {total} method-labeled mesh PNGs in {OUT_DIR} in {perf_counter() - start:.2f} s", flush=True)


if __name__ == "__main__":
    main()
