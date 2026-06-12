from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

import meshio
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ddgclib-mpl")

import matplotlib


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MSH_FILE = (
    SCRIPT_DIR
    / "out"
    / "Case_2b_axisym_initialshape_Gmsh"
    / "fig"
    / "mesh_iter0012.msh"
)

# USER CONTROLS
# Edit these values when you want a different default interactive view.
USER_INTERACTIVE_ELEV_DEG = 15.0 #0# # side-on reference: 0
USER_INTERACTIVE_AZIM_DEG = 45.0  #90# # side-on reference: -87
USER_LIQUID_BRIDGE_ALPHA = 0.42
USER_SPHERE_ALPHA = 1
USER_SPHERE_VISIBLE_FRACTION = 1.0 / 10 #0.028#
USER_SHOW_SPHERES = False
USER_SHOW_COORDINATE_SYSTEM = False
USER_SHOW_COORDINATE_ARROWS = True
USER_COORDINATE_ARROW_LENGTH_MM = 0.25
USER_COORDINATE_ARROW_GAP_MM = 0.08
USER_COORDINATE_ARROW_LINEWIDTH = 1.3
USER_COORDINATE_ARROW_LABEL_SIZE = 7
USER_SAVE_PNG = True
USER_SAVE_PNG_NAME = "mesh_PPT_vol.png"
USER_TRIM_PNG_WHITESPACE = True
USER_TRIM_PNG_PAD_PX = 18
USER_SHOW_VOLUME_SURFACE = True
USER_SHOW_CUTAWAY = True
USER_CUT_AXIS = "y"
USER_CUT_POSITION_MM = 0.0
USER_CUT_KEEP_SIDE = "negative" #positive
USER_VERTEX_SIZE = 0.0
USER_VOLUME_EDGE_LINEWIDTH = 0.10
USER_VOLUME_EDGE_ALPHA = 0.20
USER_SURFACE_EDGE_LINEWIDTH = 0.28
USER_X_LIMITS_MM = (-3.0, 3.0)
USER_Y_LIMITS_MM = (-3.0, 3.0)
USER_Z_LIMITS_MM = (-2, 2)
USER_CLIP_TO_AXIS_LIMITS = True

DEFAULT_PARTICLE_RADIUS_M = 4.0e-3
LIQUID_FACE_COLOR = "#8fc59b"
LIQUID_EDGE_COLOR = "#000000"
LIQUID_FACE_COLOR = "#acd2e9"
LIQUID_EDGE_COLOR = "#5C748B"
SPHERE_FACE_COLOR = "#c8c6bc"


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display the full volumetric tetrahedral mesh of a Case 2b Gmsh mesh."
    )
    parser.add_argument(
        "msh_file",
        nargs="?",
        type=Path,
        default=DEFAULT_MSH_FILE,
        help=f"Path to the .msh file. Default: {DEFAULT_MSH_FILE}",
    )
    parser.add_argument("--elev", type=float, default=USER_INTERACTIVE_ELEV_DEG, help="Initial 3D camera elevation.")
    parser.add_argument("--azim", type=float, default=USER_INTERACTIVE_AZIM_DEG, help="Initial 3D camera azimuth.")
    parser.add_argument(
        "--liquid-alpha",
        type=float,
        default=USER_LIQUID_BRIDGE_ALPHA,
        help="Exterior volume surface opacity from 0 to 1.",
    )
    surface_group = parser.add_mutually_exclusive_group()
    surface_group.add_argument(
        "--show-volume-surface",
        dest="show_volume_surface",
        action="store_true",
        default=USER_SHOW_VOLUME_SURFACE,
        help="Show the exterior boundary surface of the volumetric mesh.",
    )
    surface_group.add_argument(
        "--hide-volume-surface",
        dest="show_volume_surface",
        action="store_false",
        help="Hide the exterior boundary surface and show only the volume wireframe.",
    )
    cutaway_group = parser.add_mutually_exclusive_group()
    cutaway_group.add_argument(
        "--cutaway",
        dest="show_cutaway",
        action="store_true",
        default=USER_SHOW_CUTAWAY,
        help="Show a cutaway volume mesh so internal tetra faces are visible.",
    )
    cutaway_group.add_argument(
        "--no-cutaway",
        dest="show_cutaway",
        action="store_false",
        help="Show the complete volume mesh without removing a side.",
    )
    parser.add_argument(
        "--cut-axis",
        choices=("x", "y", "z"),
        default=USER_CUT_AXIS,
        help="Axis normal to the cutaway plane.",
    )
    parser.add_argument(
        "--cut-position-mm",
        type=float,
        default=USER_CUT_POSITION_MM,
        help="Cutaway plane position in mm.",
    )
    parser.add_argument(
        "--cut-keep-side",
        choices=("positive", "negative"),
        default=USER_CUT_KEEP_SIDE,
        help="Which side of the cutaway plane to keep.",
    )
    parser.add_argument(
        "--sphere-alpha",
        type=float,
        default=USER_SPHERE_ALPHA,
        help="Particle sphere opacity from 0 to 1.",
    )
    parser.add_argument(
        "--sphere-radius-mm",
        type=float,
        default=DEFAULT_PARTICLE_RADIUS_M * 1.0e3,
        help="Particle sphere radius in mm. Default: 4.0",
    )
    sphere_group = parser.add_mutually_exclusive_group()
    sphere_group.add_argument(
        "--show-spheres",
        dest="show_spheres",
        action="store_true",
        default=USER_SHOW_SPHERES,
        help="Show the two particle spheres.",
    )
    sphere_group.add_argument(
        "--no-spheres",
        dest="show_spheres",
        action="store_false",
        help="Hide the two particle spheres.",
    )
    parser.add_argument(
        "--full-spheres",
        action="store_true",
        help="Draw complete spheres instead of only the local cap near the bridge.",
    )
    parser.add_argument(
        "--sphere-visible-fraction",
        type=float,
        default=USER_SPHERE_VISIBLE_FRACTION,
        help="Fraction of each sphere surface to draw near the bridge attachment.",
    )
    parser.add_argument(
        "--vertex-size",
        type=float,
        default=USER_VERTEX_SIZE,
        help="Volumetric mesh vertex marker size.",
    )
    parser.add_argument(
        "--volume-edge-linewidth",
        type=float,
        default=USER_VOLUME_EDGE_LINEWIDTH,
        help="Line width for tetrahedral volume mesh edges.",
    )
    parser.add_argument(
        "--volume-edge-alpha",
        type=float,
        default=USER_VOLUME_EDGE_ALPHA,
        help="Opacity for tetrahedral volume mesh edges.",
    )
    parser.add_argument(
        "--surface-edge-linewidth",
        type=float,
        default=USER_SURFACE_EDGE_LINEWIDTH,
        help="Line width for exposed tetra face outlines.",
    )
    parser.add_argument("--x-limits", nargs=2, type=float, default=USER_X_LIMITS_MM, metavar=("MIN", "MAX"), help="Initial x-axis display limits in mm.")
    parser.add_argument("--y-limits", nargs=2, type=float, default=USER_Y_LIMITS_MM, metavar=("MIN", "MAX"), help="Initial y-axis display limits in mm.")
    parser.add_argument("--z-limits", nargs=2, type=float, default=USER_Z_LIMITS_MM, metavar=("MIN", "MAX"), help="Initial z-axis display limits in mm.")
    parser.add_argument("--no-axis-clip", action="store_true", help="Allow geometry outside the configured axis limits to remain visible.")
    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument(
        "--show-coordinate-system",
        dest="show_coordinate_system",
        action="store_true",
        default=USER_SHOW_COORDINATE_SYSTEM,
        help="Show axes, labels, grid, and title.",
    )
    coord_group.add_argument(
        "--hide-coordinate-system",
        dest="show_coordinate_system",
        action="store_false",
        help="Hide axes, labels, grid, and title.",
    )
    coord_arrow_group = parser.add_mutually_exclusive_group()
    coord_arrow_group.add_argument(
        "--show-coordinate-arrows",
        dest="show_coordinate_arrows",
        action="store_true",
        default=USER_SHOW_COORDINATE_ARROWS,
        help="Show a small colored XYZ coordinate arrow triad.",
    )
    coord_arrow_group.add_argument(
        "--hide-coordinate-arrows",
        dest="show_coordinate_arrows",
        action="store_false",
        help="Hide the colored XYZ coordinate arrow triad.",
    )
    parser.add_argument(
        "--coordinate-arrow-length-mm",
        type=float,
        default=USER_COORDINATE_ARROW_LENGTH_MM,
        help="Length of the in-scene coordinate arrows in mm.",
    )
    parser.add_argument(
        "--coordinate-arrow-gap-mm",
        type=float,
        default=USER_COORDINATE_ARROW_GAP_MM,
        help="Gap between the mesh bounding box and the coordinate arrow triad in mm.",
    )
    parser.add_argument(
        "--coordinate-arrow-linewidth",
        type=float,
        default=USER_COORDINATE_ARROW_LINEWIDTH,
        help="Line width of the coordinate arrows.",
    )
    parser.add_argument(
        "--coordinate-arrow-label-size",
        type=float,
        default=USER_COORDINATE_ARROW_LABEL_SIZE,
        help="Font size of the X/Y/Z coordinate arrow labels.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=SCRIPT_DIR / USER_SAVE_PNG_NAME if USER_SAVE_PNG else None,
        help="PNG path to save. Default: mesh_PPT_vol.png next to this script.",
    )
    trim_group = parser.add_mutually_exclusive_group()
    trim_group.add_argument(
        "--trim-png",
        dest="trim_png",
        action="store_true",
        default=USER_TRIM_PNG_WHITESPACE,
        help="Trim white margins from the saved PNG.",
    )
    trim_group.add_argument(
        "--no-trim-png",
        dest="trim_png",
        action="store_false",
        help="Keep the full Matplotlib figure canvas in the saved PNG.",
    )
    parser.add_argument(
        "--trim-png-pad-px",
        type=int,
        default=USER_TRIM_PNG_PAD_PX,
        help="White padding to keep around the trimmed PNG.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open a window; useful with --save.")
    return parser


def _configure_matplotlib(*, no_show: bool) -> None:
    if no_show:
        matplotlib.use("Agg", force=True)
        return

    if os.environ.get("MPLBACKEND"):
        return

    # Prefer a real GUI backend for the requested interactive window.
    for backend in ("MacOSX", "QtAgg", "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue


def _tetra_blocks(mesh: meshio.Mesh) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for cell_block in mesh.cells:
        if cell_block.type in {"tetra", "tetra10"}:
            blocks.append(np.asarray(cell_block.data[:, :4], dtype=int))
    if not blocks:
        raise ValueError("No tetra cells found in the .msh file.")
    return np.vstack(blocks)


def _signed_tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return float(np.dot(a - d, np.cross(b - d, c - d)) / 6.0)


def _boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    face_count: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    for tet in np.asarray(tets, dtype=int):
        for face in (
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ):
            face_count[tuple(sorted(int(v) for v in face))] += 1

    faces = [face for face, count in face_count.items() if count == 1]
    if not faces:
        raise ValueError("No exposed tetra boundary faces found.")
    return np.asarray(faces, dtype=int)


def _liquid_air_vertex_mask(points: np.ndarray) -> np.ndarray:
    """Find rings on the monotone bridge side profile, excluding particle caps."""

    pts = np.asarray(points, dtype=float)
    z_abs = np.abs(pts[:, 2])
    radii = np.linalg.norm(pts[:, :2], axis=1)

    # At each |z|, the liquid-air profile is the largest radius.  The spherical
    # wetted caps then get rejected by enforcing the monotone neck-to-contact
    # profile used by the Case 2b axisymmetric importer.
    max_radius_by_abs_z: dict[float, float] = {}
    for z_val, radius in zip(z_abs, radii):
        key = round(float(z_val), 15)
        max_radius_by_abs_z[key] = max(max_radius_by_abs_z.get(key, 0.0), float(radius))

    half_z_all = np.asarray(sorted(max_radius_by_abs_z), dtype=float)
    half_r_all = np.asarray([max_radius_by_abs_z[float(z)] for z in half_z_all], dtype=float)
    if half_z_all.size < 2:
        raise ValueError("Could not infer an axisymmetric outer profile from the mesh points.")

    profile: list[tuple[float, float]] = []
    current_max_radius = -float("inf")
    monotone_tol = max(1.0e-12, 1.0e-9 * float(np.max(half_r_all)))
    for z_val, radius in zip(half_z_all, half_r_all):
        if float(radius) + monotone_tol < current_max_radius:
            continue
        profile.append((float(z_val), float(radius)))
        current_max_radius = max(current_max_radius, float(radius))

    z_span = max(float(np.max(pts[:, 2]) - np.min(pts[:, 2])), 1.0e-12)
    r_max = max(float(np.max(radii)), 1.0e-12)
    z_tol = max(5.0e-12, 1.0e-8 * z_span)
    r_tol = max(5.0e-12, 1.0e-6 * r_max)

    mask = np.zeros(len(pts), dtype=bool)
    for z_profile, r_profile in profile:
        mask |= (np.abs(z_abs - z_profile) <= z_tol) & (np.abs(radii - r_profile) <= r_tol)

    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("Could not identify enough liquid-air surface vertices.")
    return mask


def _outer_liquid_surface_faces(points: np.ndarray, boundary_faces: np.ndarray) -> np.ndarray:
    on_liquid_air = _liquid_air_vertex_mask(points)
    keep = np.all(on_liquid_air[np.asarray(boundary_faces, dtype=int)], axis=1)
    faces = np.asarray(boundary_faces, dtype=int)[keep]
    if faces.size == 0:
        raise ValueError("No boundary faces matched the liquid-air outer surface.")
    return faces


def _triangle_edges(triangles: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tri in np.asarray(triangles, dtype=int):
        edges.add(tuple(sorted((int(tri[0]), int(tri[1])))))
        edges.add(tuple(sorted((int(tri[1]), int(tri[2])))))
        edges.add(tuple(sorted((int(tri[2]), int(tri[0])))))
    return np.asarray(sorted(edges), dtype=int)


def _tetra_edges(tets: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tet in np.asarray(tets, dtype=int):
        edges.add(tuple(sorted((int(tet[0]), int(tet[1])))))
        edges.add(tuple(sorted((int(tet[0]), int(tet[2])))))
        edges.add(tuple(sorted((int(tet[0]), int(tet[3])))))
        edges.add(tuple(sorted((int(tet[1]), int(tet[2])))))
        edges.add(tuple(sorted((int(tet[1]), int(tet[3])))))
        edges.add(tuple(sorted((int(tet[2]), int(tet[3])))))
    if not edges:
        raise ValueError("No tetrahedral edges found.")
    return np.asarray(sorted(edges), dtype=int)


def _cutaway_tets(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    cut_axis: str,
    cut_position_mm: float,
    cut_keep_side: str,
) -> np.ndarray:
    axis_index = {"x": 0, "y": 1, "z": 2}[str(cut_axis).lower()]
    centroid_mm = 1.0e3 * np.mean(np.asarray(points, dtype=float)[np.asarray(tets, dtype=int)], axis=1)
    cut_value = float(cut_position_mm)
    if str(cut_keep_side).lower() == "positive":
        keep = centroid_mm[:, axis_index] >= cut_value
    else:
        keep = centroid_mm[:, axis_index] <= cut_value
    kept_tets = np.asarray(tets, dtype=int)[keep]
    if kept_tets.size == 0:
        raise ValueError("Cutaway removed all tetrahedra. Adjust USER_CUT_POSITION_MM or USER_CUT_KEEP_SIDE.")
    return kept_tets


def _clamp_alpha(alpha: float) -> float:
    return float(np.clip(float(alpha), 0.0, 1.0))


def _clamp_fraction(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _sphere_geometry_from_bridge(
    points: np.ndarray,
    surface_faces: np.ndarray,
    *,
    sphere_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    pts = np.asarray(points, dtype=float)
    surface_points = pts[np.unique(np.asarray(surface_faces, dtype=int).reshape(-1))]
    radii = np.linalg.norm(surface_points[:, :2], axis=1)
    contact_radius = float(np.max(radii))
    if sphere_radius_m <= contact_radius:
        raise ValueError(
            f"Sphere radius {sphere_radius_m * 1.0e3:.6g} mm must be larger than "
            f"contact radius {contact_radius * 1.0e3:.6g} mm."
        )

    contact_tol = max(5.0e-12, 1.0e-6 * contact_radius)
    contact_points = surface_points[np.abs(radii - contact_radius) <= contact_tol]
    if contact_points.size == 0:
        contact_points = surface_points

    bottom_contact_z = float(np.min(contact_points[:, 2]))
    top_contact_z = float(np.max(contact_points[:, 2]))
    axial_offset = float(np.sqrt(max(sphere_radius_m**2 - contact_radius**2, 0.0)))
    bottom_center = np.array([0.0, 0.0, bottom_contact_z - axial_offset], dtype=float)
    top_center = np.array([0.0, 0.0, top_contact_z + axial_offset], dtype=float)
    return bottom_center, top_center, contact_radius, bottom_contact_z, top_contact_z


def _sphere_surface_xyz(
    center: np.ndarray,
    radius_m: float,
    *,
    phi_min: float = 0.0,
    phi_max: float = np.pi,
    n_theta: int = 64,
    n_phi: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta) + 1, dtype=float)
    pole_eps = 1.0e-4
    phi = np.linspace(
        max(float(phi_min), pole_eps),
        min(float(phi_max), np.pi - pole_eps),
        int(n_phi),
        dtype=float,
    )
    sin_phi = np.sin(phi)[:, None]
    x = center[0] + radius_m * sin_phi * np.cos(theta)[None, :]
    y = center[1] + radius_m * sin_phi * np.sin(theta)[None, :]
    z = center[2] + radius_m * np.cos(phi)[:, None] * np.ones_like(theta)[None, :]
    return 1.0e3 * x, 1.0e3 * y, 1.0e3 * z


def _sphere_facecolors(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    center: np.ndarray,
    radius_m: float,
    *,
    alpha: float,
) -> np.ndarray:
    from matplotlib.colors import to_rgb

    center_mm = 1.0e3 * np.asarray(center, dtype=float)
    radius_mm = 1.0e3 * float(radius_m)
    normals = np.stack(
        (
            sx - center_mm[0],
            sy - center_mm[1],
            sz - center_mm[2],
        ),
        axis=-1,
    ) / max(radius_mm, 1.0e-12)
    light = np.array([-0.45, -0.35, 0.82], dtype=float)
    light /= max(float(np.linalg.norm(light)), 1.0e-12)
    intensity = np.clip(0.62 + 0.38 * np.maximum(normals @ light, 0.0), 0.52, 1.0)
    base = np.array(to_rgb(SPHERE_FACE_COLOR), dtype=float)
    colors = np.empty(normals.shape[:-1] + (4,), dtype=float)
    colors[..., :3] = np.clip(base[None, None, :] * intensity[..., None], 0.0, 1.0)
    colors[..., 3] = _clamp_alpha(alpha)
    return colors


def _apply_axis_ranges(ax, center: np.ndarray, ranges: np.ndarray) -> None:
    ranges = np.maximum(np.asarray(ranges, dtype=float), 1.0e-9)
    half_ranges = 0.5 * ranges
    ax.set_xlim(center[0] - half_ranges[0], center[0] + half_ranges[0])
    ax.set_ylim(center[1] - half_ranges[1], center[1] + half_ranges[1])
    ax.set_zlim(center[2] - half_ranges[2], center[2] + half_ranges[2])
    ax.set_box_aspect(tuple(ranges))


def _apply_axis_limits(ax, limits: np.ndarray) -> None:
    limits = np.asarray(limits, dtype=float)
    ranges = np.maximum(limits[:, 1] - limits[:, 0], 1.0e-9)
    ax.set_xlim(float(limits[0, 0]), float(limits[0, 1]))
    ax.set_ylim(float(limits[1, 0]), float(limits[1, 1]))
    ax.set_zlim(float(limits[2, 0]), float(limits[2, 1]))
    ax.set_box_aspect(tuple(ranges))


def _set_axes_equal(
    ax,
    coords: np.ndarray,
    *,
    pad_fraction: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(coords, dtype=float).reshape((-1, 3))
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1.0e-9)
    radius *= 1.0 + float(pad_fraction)
    ranges = np.full(3, 2.0 * radius, dtype=float)
    _apply_axis_ranges(ax, center, ranges)
    return center, ranges


def _initial_axis_limits(
    auto_center: np.ndarray,
    auto_ranges: np.ndarray,
    *,
    x_limits_mm: tuple[float, float] | list[float] | None,
    y_limits_mm: tuple[float, float] | list[float] | None,
    z_limits_mm: tuple[float, float] | list[float] | None,
) -> np.ndarray:
    center = np.asarray(auto_center, dtype=float)
    ranges = np.asarray(auto_ranges, dtype=float)
    half_ranges = 0.5 * ranges
    limits = np.column_stack((center - half_ranges, center + half_ranges))
    for idx, value in enumerate((x_limits_mm, y_limits_mm, z_limits_mm)):
        if value is not None:
            low, high = float(value[0]), float(value[1])
            if high <= low:
                raise ValueError("Axis limits must be ordered as (min, max).")
            limits[idx] = [low, high]
    return limits


def _sphere_cap_phi_limits(
    center: np.ndarray,
    contact_z: float,
    radius_m: float,
    visible_fraction: float,
) -> tuple[float, float]:
    fraction = _clamp_fraction(visible_fraction)
    if fraction >= 1.0:
        return 0.0, np.pi
    if fraction <= 0.0:
        return 0.0, 0.0

    contact_phi = float(
        np.arccos(
            np.clip(
                (float(contact_z) - float(center[2])) / max(float(radius_m), 1.0e-30),
                -1.0,
                1.0,
            )
        )
    )

    # Draw a dry spherical band adjacent to the contact line.  The wetted cap
    # under the bridge is intentionally omitted so the liquid does not look
    # buried inside an opaque particle.
    if float(center[2]) < 0.0:
        cos_outer = np.cos(contact_phi) - 2.0 * fraction
        phi_outer = float(np.arccos(np.clip(cos_outer, -1.0, 1.0)))
        return contact_phi, phi_outer

    cos_outer = np.cos(contact_phi) + 2.0 * fraction
    phi_outer = float(np.arccos(np.clip(cos_outer, -1.0, 1.0)))
    return phi_outer, contact_phi


def _trim_png_whitespace(path: Path, *, pad_px: int) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("Warning: Pillow is not available; saved PNG was not whitespace-trimmed.")
        return

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        pixels = np.asarray(rgba)
        alpha = pixels[:, :, 3] > 0
        non_white = np.any(pixels[:, :, :3] < 250, axis=2) & alpha
        if not np.any(non_white):
            return

        ys, xs = np.nonzero(non_white)
        pad = max(int(pad_px), 0)
        left = max(int(xs.min()) - pad, 0)
        upper = max(int(ys.min()) - pad, 0)
        right = min(int(xs.max()) + pad + 1, rgba.width)
        lower = min(int(ys.max()) + pad + 1, rgba.height)
        rgba.crop((left, upper, right, lower)).save(path)


def _coordinate_triad_points(origin: np.ndarray, *, length_mm: float) -> np.ndarray:
    origin = np.asarray(origin, dtype=float)
    length = max(float(length_mm), 0.0)
    directions = np.eye(3, dtype=float)
    points = [origin]
    for direction in directions:
        points.append(origin + length * direction)
        points.append(origin + 1.35 * length * direction)
    return np.vstack(points)


def _coordinate_triad_origin(
    ax,
    mesh_coords: np.ndarray,
    axis_limits: np.ndarray,
    *,
    length_mm: float,
    gap_mm: float,
) -> np.ndarray:
    from mpl_toolkits.mplot3d import proj3d

    xyz = np.asarray(mesh_coords, dtype=float).reshape((-1, 3))
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    limits = np.asarray(axis_limits, dtype=float)
    ranges = np.maximum(limits[:, 1] - limits[:, 0], 1.0e-9)
    length = max(float(length_mm), 1.0e-9)
    gap = max(float(gap_mm), 0.0)

    projection = ax.get_proj()
    mesh_screen_x, mesh_screen_y, _mesh_screen_z = proj3d.proj_transform(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        projection,
    )
    target = np.array(
        (
            float(np.max(mesh_screen_x)) + 0.010,
            float(np.min(mesh_screen_y)) - 0.008,
        ),
        dtype=float,
    )

    candidate_values = (
        (mins[0] - gap - length, mins[0] - gap, 0.5 * (mins[0] + maxs[0]), maxs[0] + gap, maxs[0] + gap + length),
        (mins[1] - gap - length, mins[1] - gap, 0.5 * (mins[1] + maxs[1]), maxs[1] + gap, maxs[1] + gap + length),
        (mins[2] - gap - length, mins[2] - gap, 0.5 * (mins[2] + maxs[2]), maxs[2] + gap, maxs[2] + gap + length),
    )
    candidates: list[tuple[float, np.ndarray]] = []
    for x_value in candidate_values[0]:
        for y_value in candidate_values[1]:
            for z_value in candidate_values[2]:
                origin = np.array((x_value, y_value, z_value), dtype=float)
                outside_mesh = bool(np.any((origin < mins) | (origin > maxs)))
                if not outside_mesh:
                    continue

                triad_points = _coordinate_triad_points(origin, length_mm=length)
                inside_limits = bool(np.all((triad_points >= limits[:, 0]) & (triad_points <= limits[:, 1])))
                screen_x, screen_y, _screen_z = proj3d.proj_transform(
                    origin[0],
                    origin[1],
                    origin[2],
                    projection,
                )
                # Pick a 3D origin that projects just to the lower-right of
                # the mesh in the initial camera; after that it rotates with
                # the mesh because it is drawn in the same 3D axes.
                screen_point = np.array((float(screen_x), float(screen_y)), dtype=float)
                score = float(np.linalg.norm(screen_point - target))
                score += 4.0 * max(0.0, float(np.max(mesh_screen_x) - screen_x))
                score += 4.0 * max(0.0, float(screen_y - np.min(mesh_screen_y)))
                if not inside_limits:
                    score += 100.0
                candidates.append((score, origin))

    best_origin = min(candidates, key=lambda item: item[0])[1]
    margin = 0.03 * ranges
    low = limits[:, 0] + margin
    high = limits[:, 1] - margin - 1.35 * length
    for idx in range(3):
        if high[idx] <= low[idx]:
            best_origin[idx] = 0.5 * (limits[idx, 0] + limits[idx, 1] - length)
        else:
            best_origin[idx] = float(np.clip(best_origin[idx], low[idx], high[idx]))
    return best_origin


def _add_coordinate_triad(
    ax,
    mesh_coords: np.ndarray,
    axis_limits: np.ndarray,
    *,
    length_mm: float,
    gap_mm: float,
    linewidth: float,
    label_size: float,
) -> None:
    length = max(float(length_mm), 0.0)
    if length <= 0.0:
        return

    origin = _coordinate_triad_origin(
        ax,
        mesh_coords,
        axis_limits,
        length_mm=length,
        gap_mm=gap_mm,
    )
    arrows = (
        ("X", "#d62728", np.array((1.0, 0.0, 0.0), dtype=float)),
        ("Y", "#2ca02c", np.array((0.0, 1.0, 0.0), dtype=float)),
        ("Z", "#1f77b4", np.array((0.0, 0.0, 1.0), dtype=float)),
    )
    ax.scatter(
        [origin[0]],
        [origin[1]],
        [origin[2]],
        s=10,
        c="#777777",
        depthshade=False,
        zorder=30,
    )
    for label, color, direction in arrows:
        vector = length * direction
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vector[0],
            vector[1],
            vector[2],
            color=color,
            linewidth=max(float(linewidth), 0.1),
            arrow_length_ratio=0.25,
            normalize=False,
            zorder=31,
        )
        label_pos = origin + 1.18 * vector
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            label,
            color=color,
            fontsize=max(float(label_size), 1.0),
            fontweight="bold",
            ha="center",
            va="center",
            zorder=32,
        )


def _plot_volumetric_mesh(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    title: str,
    elev: float,
    azim: float,
    sphere_radius_m: float,
    show_spheres: bool,
    liquid_alpha: float,
    sphere_alpha: float,
    sphere_visible_fraction: float,
    show_volume_surface: bool,
    show_cutaway: bool,
    cut_axis: str,
    cut_position_mm: float,
    cut_keep_side: str,
    vertex_size: float,
    volume_edge_linewidth: float,
    volume_edge_alpha: float,
    surface_edge_linewidth: float,
    x_limits_mm: tuple[float, float] | list[float] | None,
    y_limits_mm: tuple[float, float] | list[float] | None,
    z_limits_mm: tuple[float, float] | list[float] | None,
    clip_to_axis_limits: bool,
    show_coordinate_system: bool,
    show_coordinate_arrows: bool,
    coordinate_arrow_length_mm: float,
    coordinate_arrow_gap_mm: float,
    coordinate_arrow_linewidth: float,
    coordinate_arrow_label_size: float,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    plot_points = 1.0e3 * np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=int)
    render_tets = (
        _cutaway_tets(
            points,
            tets,
            cut_axis=cut_axis,
            cut_position_mm=float(cut_position_mm),
            cut_keep_side=cut_keep_side,
        )
        if show_cutaway
        else tets
    )
    mesh_vertices = np.unique(render_tets.reshape(-1))
    mesh_edges = _tetra_edges(render_tets)
    boundary_faces = _boundary_faces_from_tets(render_tets)

    edge_segments = plot_points[mesh_edges]
    boundary_triangles = plot_points[boundary_faces]

    fig = plt.figure(figsize=(8.4, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    axis_points = [plot_points[mesh_vertices]]
    if show_volume_surface:
        surface_collection = Poly3DCollection(
            boundary_triangles,
            facecolor=LIQUID_FACE_COLOR,
            edgecolor=LIQUID_EDGE_COLOR,
            linewidths=max(float(surface_edge_linewidth), 0.0),
            alpha=_clamp_alpha(liquid_alpha),
            zorder=6,
            axlim_clip=bool(clip_to_axis_limits),
        )
        ax.add_collection3d(surface_collection)

    if show_spheres:
        surface_faces = _outer_liquid_surface_faces(points, boundary_faces)
        (
            bottom_center,
            top_center,
            _contact_radius,
            bottom_contact_z,
            top_contact_z,
        ) = _sphere_geometry_from_bridge(
            points,
            surface_faces,
            sphere_radius_m=float(sphere_radius_m),
        )
        sphere_specs = (
            (bottom_center, bottom_contact_z),
            (top_center, top_contact_z),
        )
        for center, contact_z in sphere_specs:
            phi_min, phi_max = _sphere_cap_phi_limits(
                center,
                contact_z,
                float(sphere_radius_m),
                sphere_visible_fraction,
            )
            if phi_max <= phi_min:
                continue

            sx, sy, sz = _sphere_surface_xyz(
                center,
                float(sphere_radius_m),
                phi_min=phi_min,
                phi_max=phi_max,
            )
            ax.plot_surface(
                sx,
                sy,
                sz,
                facecolors=_sphere_facecolors(
                    sx,
                    sy,
                    sz,
                    center,
                    float(sphere_radius_m),
                    alpha=sphere_alpha,
                ),
                linewidth=0.0,
                antialiased=True,
                shade=False,
                zorder=1,
                axlim_clip=bool(clip_to_axis_limits),
            )
            axis_points.append(np.column_stack((sx.reshape(-1), sy.reshape(-1), sz.reshape(-1))))

    if float(volume_edge_linewidth) > 0.0 and float(volume_edge_alpha) > 0.0:
        edge_collection = Line3DCollection(
            edge_segments,
            colors=LIQUID_EDGE_COLOR,
            linewidths=max(float(volume_edge_linewidth), 0.0),
            alpha=_clamp_alpha(volume_edge_alpha),
            zorder=11,
            axlim_clip=bool(clip_to_axis_limits),
        )
        ax.add_collection3d(edge_collection)
    vertex_size = max(float(vertex_size), 0.0)
    if vertex_size > 0.0:
        ax.scatter(
            plot_points[mesh_vertices, 0],
            plot_points[mesh_vertices, 1],
            plot_points[mesh_vertices, 2],
            s=vertex_size,
            c=LIQUID_EDGE_COLOR,
            alpha=0.70,
            depthshade=False,
            zorder=12,
            axlim_clip=bool(clip_to_axis_limits),
        )

    axis_center, axis_ranges = _set_axes_equal(ax, np.vstack(axis_points))
    axis_limits = _initial_axis_limits(
        axis_center,
        axis_ranges,
        x_limits_mm=x_limits_mm,
        y_limits_mm=y_limits_mm,
        z_limits_mm=z_limits_mm,
    )
    _apply_axis_limits(ax, axis_limits)
    ax.set_proj_type("ortho")
    ax.view_init(elev=float(elev), azim=float(azim))
    if show_coordinate_arrows:
        _add_coordinate_triad(
            ax,
            plot_points[mesh_vertices],
            axis_limits,
            length_mm=float(coordinate_arrow_length_mm),
            gap_mm=float(coordinate_arrow_gap_mm),
            linewidth=float(coordinate_arrow_linewidth),
            label_size=float(coordinate_arrow_label_size),
        )
    if show_coordinate_system:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.set_title(title, pad=12)
        ax.grid(True, alpha=0.28)
    else:
        ax.set_axis_off()
        ax.grid(False)
    if show_coordinate_system:
        fig.tight_layout()
    else:
        ax.set_position((0.0, 0.0, 1.0, 1.0))
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    return fig


def main() -> None:
    args = _build_cli().parse_args()
    _configure_matplotlib(no_show=bool(args.no_show))

    msh_file = Path(args.msh_file).expanduser().resolve()
    if not msh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {msh_file}")

    print("Reading mesh:", msh_file)
    mesh = meshio.read(msh_file)
    points = np.asarray(mesh.points, dtype=float)
    tets = _tetra_blocks(mesh)
    render_tets = (
        _cutaway_tets(
            points,
            tets,
            cut_axis=str(args.cut_axis),
            cut_position_mm=float(args.cut_position_mm),
            cut_keep_side=str(args.cut_keep_side),
        )
        if bool(args.show_cutaway)
        else tets
    )
    boundary_faces = _boundary_faces_from_tets(render_tets)

    vol_tetra = np.array(
        [
            _signed_tet_volume(points[t[0]], points[t[1]], points[t[2]], points[t[3]])
            for t in tets
        ],
        dtype=float,
    )

    volume_edges = _tetra_edges(render_tets)
    volume_vertex_count = int(np.unique(render_tets.reshape(-1)).size)

    print("sum(abs(vol_tetra))       =", np.sum(np.abs(vol_tetra)))
    print("sum(vol_tetra)            =", np.sum(vol_tetra))
    print("min(vol_tetra)            =", np.min(vol_tetra))
    print("max(vol_tetra)            =", np.max(vol_tetra))
    print("number of tetrahedra      =", len(tets))
    print("rendered tetrahedra       =", len(render_tets))
    print("volume boundary faces     =", len(boundary_faces))
    print("volume mesh edges         =", len(volume_edges))
    print("volume mesh vertices      =", volume_vertex_count)

    title = (
        f"{msh_file.stem}: volumetric tetrahedral mesh\n"
        f"vertices={volume_vertex_count}, tetrahedra={len(render_tets)}, edges={len(volume_edges)}"
    )
    fig = _plot_volumetric_mesh(
        points,
        tets,
        title=title,
        elev=args.elev,
        azim=args.azim,
        sphere_radius_m=float(args.sphere_radius_mm) * 1.0e-3,
        show_spheres=bool(args.show_spheres),
        liquid_alpha=float(args.liquid_alpha),
        sphere_alpha=float(args.sphere_alpha),
        sphere_visible_fraction=1.0 if bool(args.full_spheres) else float(args.sphere_visible_fraction),
        show_volume_surface=bool(args.show_volume_surface),
        show_cutaway=bool(args.show_cutaway),
        cut_axis=str(args.cut_axis),
        cut_position_mm=float(args.cut_position_mm),
        cut_keep_side=str(args.cut_keep_side),
        vertex_size=float(args.vertex_size),
        volume_edge_linewidth=float(args.volume_edge_linewidth),
        volume_edge_alpha=float(args.volume_edge_alpha),
        surface_edge_linewidth=float(args.surface_edge_linewidth),
        x_limits_mm=args.x_limits,
        y_limits_mm=args.y_limits,
        z_limits_mm=args.z_limits,
        clip_to_axis_limits=bool(USER_CLIP_TO_AXIS_LIMITS) and not bool(args.no_axis_clip),
        show_coordinate_system=bool(args.show_coordinate_system),
        show_coordinate_arrows=bool(args.show_coordinate_arrows),
        coordinate_arrow_length_mm=float(args.coordinate_arrow_length_mm),
        coordinate_arrow_gap_mm=float(args.coordinate_arrow_gap_mm),
        coordinate_arrow_linewidth=float(args.coordinate_arrow_linewidth),
        coordinate_arrow_label_size=float(args.coordinate_arrow_label_size),
    )

    if args.save is not None:
        save_path = Path(args.save).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.0)
        if bool(args.trim_png) and save_path.suffix.lower() == ".png":
            _trim_png_whitespace(save_path, pad_px=int(args.trim_png_pad_px))
        print("Wrote:", save_path)

    if args.no_show:
        import matplotlib.pyplot as plt

        plt.close(fig)
        return

    backend = matplotlib.get_backend()
    if backend.lower().endswith("agg"):
        print(f"Warning: Matplotlib backend is {backend!r}, so no interactive window may appear.")
    print("Opening interactive Matplotlib window. Use the mouse to rotate, zoom, and pan.")

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
