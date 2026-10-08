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
DEFAULT_BATCH_DIR = (
    SCRIPT_DIR
    / "Case_2b_axisym_pitois2000_separation_Gmsh_recovered_2026-05-01_v3"
)

# USER CONTROLS
# Edit these values when you want a different default interactive view.
USER_INTERACTIVE_ELEV_DEG = 7#15.0 #0# # side-on reference: 0
USER_INTERACTIVE_AZIM_DEG = 45.0  #90# # side-on reference: -87
USER_LIQUID_BRIDGE_ALPHA = 1
USER_SPHERE_ALPHA = 1
USER_SPHERE_VISIBLE_FRACTION = 1.0 / 10 #0.028#
USER_VERTEX_SIZE = 0.0
USER_X_LIMITS_MM = (-3.0, 3.0)
USER_Y_LIMITS_MM = (-3.0, 3.0)
USER_Z_LIMITS_MM = (-2, 2)
USER_CLIP_TO_AXIS_LIMITS = True
USER_BATCH_INPUT_DIR = DEFAULT_BATCH_DIR
USER_BATCH_MSH_PATTERN = "*.msh"
USER_BATCH_OUTPUT_SUFFIX = "_video"
USER_BATCH_OVERWRITE = True
USER_USE_DEFINED_SPHERE_MOTION = True
USER_SPHERE_MOTION_DT_S = 0.01
USER_SPHERE_MOTION_SPEED_MPS = 5.0e-6
USER_SPHERE_MOTION_DIRECTION = (0.0, 0.0, -1.0)
USER_KEEP_INTERACTIVE_WINDOWS = True
# Number of figures to keep open after PNG export. Use -1 to keep every figure.
USER_INTERACTIVE_WINDOW_LIMIT = 1
USER_CREATE_GIF = True
USER_GIF_OUTPUT_NAME = "mesh_video.gif"
USER_GIF_FRAME_DURATION_MS = 90
USER_GIF_LOOP = 0

DEFAULT_PARTICLE_RADIUS_M = 4.0e-3
LIQUID_FACE_COLOR = "#8fb0c5"
LIQUID_EDGE_COLOR = "#637988"
SPHERE_FACE_COLOR = "#c8c6bc"
LIQUID_FACE_COLOR = "#acd2e9"
LIQUID_EDGE_COLOR = "#5C748B"

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-render outer liquid-air surface PNGs for Case 2b Gmsh tetra meshes."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=USER_BATCH_INPUT_DIR,
        help=f"Directory containing .msh files. Default: {USER_BATCH_INPUT_DIR}",
    )
    parser.add_argument("--pattern", default=USER_BATCH_MSH_PATTERN, help="Glob pattern for input .msh files.")
    parser.add_argument("--output-suffix", default=USER_BATCH_OUTPUT_SUFFIX, help="Suffix added before .png.")
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N meshes; useful for testing.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing output PNG files.")
    parser.add_argument("--elev", type=float, default=USER_INTERACTIVE_ELEV_DEG, help="Initial 3D camera elevation.")
    parser.add_argument("--azim", type=float, default=USER_INTERACTIVE_AZIM_DEG, help="Initial 3D camera azimuth.")
    parser.add_argument(
        "--liquid-alpha",
        type=float,
        default=USER_LIQUID_BRIDGE_ALPHA,
        help="Liquid bridge surface opacity from 0 to 1.",
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
    parser.add_argument("--no-spheres", action="store_true", help="Hide the two particle spheres.")
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
        "--sphere-motion",
        choices=("defined", "mesh"),
        default="defined" if USER_USE_DEFINED_SPHERE_MOTION else "mesh",
        help="Use the Case 2b prescribed sphere motion, or infer both centers from each mesh.",
    )
    parser.add_argument(
        "--sphere-motion-dt-s",
        type=float,
        default=USER_SPHERE_MOTION_DT_S,
        help="Time step used by the prescribed sphere motion.",
    )
    parser.add_argument(
        "--sphere-motion-speed-mps",
        type=float,
        default=USER_SPHERE_MOTION_SPEED_MPS,
        help="Moving sphere speed used by the prescribed sphere motion.",
    )
    parser.add_argument(
        "--vertex-size",
        type=float,
        default=USER_VERTEX_SIZE,
        help="Initial liquid bridge vertex marker size.",
    )
    parser.add_argument("--x-limits", nargs=2, type=float, default=USER_X_LIMITS_MM, metavar=("MIN", "MAX"), help="Initial x-axis display limits in mm.")
    parser.add_argument("--y-limits", nargs=2, type=float, default=USER_Y_LIMITS_MM, metavar=("MIN", "MAX"), help="Initial y-axis display limits in mm.")
    parser.add_argument("--z-limits", nargs=2, type=float, default=USER_Z_LIMITS_MM, metavar=("MIN", "MAX"), help="Initial z-axis display limits in mm.")
    parser.add_argument("--no-axis-clip", action="store_true", help="Allow geometry outside the configured axis limits to remain visible.")
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=USER_KEEP_INTERACTIVE_WINDOWS,
        help="Keep Matplotlib figure windows open after the batch render completes.",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Save PNGs only and do not keep Matplotlib windows open.",
    )
    parser.add_argument(
        "--interactive-window-limit",
        type=int,
        default=USER_INTERACTIVE_WINDOW_LIMIT,
        help="How many rendered figures to keep open. Use -1 to keep every rendered figure.",
    )
    parser.add_argument(
        "--gif",
        dest="create_gif",
        action="store_true",
        default=USER_CREATE_GIF,
        help="Create a GIF from the generated PNG frames after rendering.",
    )
    parser.add_argument(
        "--no-gif",
        dest="create_gif",
        action="store_false",
        help="Do not create a GIF after rendering.",
    )
    parser.add_argument(
        "--gif-output",
        default=USER_GIF_OUTPUT_NAME,
        help="GIF output path. Relative paths are written inside the input directory.",
    )
    parser.add_argument(
        "--gif-frame-duration-ms",
        type=int,
        default=USER_GIF_FRAME_DURATION_MS,
        help="GIF frame duration in milliseconds.",
    )
    parser.add_argument(
        "--gif-loop",
        type=int,
        default=USER_GIF_LOOP,
        help="GIF loop count. 0 means loop forever.",
    )
    return parser


def _configure_matplotlib(*, interactive: bool) -> None:
    if not interactive:
        matplotlib.use("Agg", force=True)


def _tetra_blocks(mesh: meshio.Mesh) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for cell_block in mesh.cells:
        if cell_block.type in {"tetra", "tetra10"}:
            blocks.append(np.asarray(cell_block.data[:, :4], dtype=int))
    if not blocks:
        raise ValueError("No tetra cells found in the .msh file.")
    return np.vstack(blocks)


def _triangle_blocks(mesh: meshio.Mesh) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for cell_block in mesh.cells:
        if cell_block.type in {"triangle", "triangle6"}:
            blocks.append(np.asarray(cell_block.data[:, :3], dtype=int))
    if not blocks:
        return np.empty((0, 3), dtype=int)
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


def _faces_without_sphere_caps(
    points: np.ndarray,
    faces: np.ndarray,
    sphere_centers: tuple[np.ndarray, np.ndarray],
    *,
    sphere_radius_m: float,
) -> np.ndarray:
    faces = np.asarray(faces, dtype=int)
    if faces.size == 0:
        raise ValueError("No faces available for liquid-air surface extraction.")

    face_points = np.asarray(points, dtype=float)[faces]
    radius = float(sphere_radius_m)
    tol = max(2.0e-8, 2.0e-5 * max(radius, 1.0e-30))
    on_any_cap = np.zeros(len(faces), dtype=bool)
    for center in sphere_centers:
        distances = np.linalg.norm(face_points - np.asarray(center, dtype=float)[None, None, :], axis=2)
        on_any_cap |= np.all(np.abs(distances - radius) <= tol, axis=1)

    liquid_faces = faces[~on_any_cap]
    if liquid_faces.size == 0:
        raise ValueError("All candidate faces were classified as sphere caps.")
    return liquid_faces


def _outer_liquid_surface_faces(
    points: np.ndarray,
    boundary_faces: np.ndarray,
    *,
    sphere_centers: tuple[np.ndarray, np.ndarray] | None = None,
    sphere_radius_m: float | None = None,
) -> np.ndarray:
    if sphere_centers is not None and sphere_radius_m is not None:
        return _faces_without_sphere_caps(
            points,
            boundary_faces,
            sphere_centers,
            sphere_radius_m=float(sphere_radius_m),
        )

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
    (
        bottom_contact_z,
        top_contact_z,
        bottom_contact_radius,
        top_contact_radius,
    ) = _bridge_contact_endpoints(points, surface_faces)
    contact_radius = max(bottom_contact_radius, top_contact_radius)
    if sphere_radius_m <= contact_radius:
        raise ValueError(
            f"Sphere radius {sphere_radius_m * 1.0e3:.6g} mm must be larger than "
            f"contact radius {contact_radius * 1.0e3:.6g} mm."
        )

    bottom_offset = float(np.sqrt(max(sphere_radius_m**2 - bottom_contact_radius**2, 0.0)))
    top_offset = float(np.sqrt(max(sphere_radius_m**2 - top_contact_radius**2, 0.0)))
    bottom_center = np.array([0.0, 0.0, bottom_contact_z - bottom_offset], dtype=float)
    top_center = np.array([0.0, 0.0, top_contact_z + top_offset], dtype=float)
    return bottom_center, top_center, contact_radius, bottom_contact_z, top_contact_z


def _bridge_contact_endpoints(
    points: np.ndarray,
    surface_faces: np.ndarray,
) -> tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=float)
    surface_points = pts[np.unique(np.asarray(surface_faces, dtype=int).reshape(-1))]
    if surface_points.size == 0:
        raise ValueError("No surface points available for sphere placement.")

    z_values = surface_points[:, 2]
    radii = np.linalg.norm(surface_points[:, :2], axis=1)
    bottom_contact_z = float(np.min(z_values))
    top_contact_z = float(np.max(z_values))
    z_tol = max(5.0e-12, 1.0e-8 * max(top_contact_z - bottom_contact_z, 1.0e-12))
    bottom_mask = np.abs(z_values - bottom_contact_z) <= z_tol
    top_mask = np.abs(z_values - top_contact_z) <= z_tol
    if not np.any(bottom_mask) or not np.any(top_mask):
        raise ValueError("Could not identify bridge endpoint contact rings.")

    bottom_contact_radius = float(np.max(radii[bottom_mask]))
    top_contact_radius = float(np.max(radii[top_mask]))
    return bottom_contact_z, top_contact_z, bottom_contact_radius, top_contact_radius


def _mesh_iteration_number(path: Path) -> int:
    stem = path.stem
    marker = "mesh_iter"
    if stem.startswith(marker):
        suffix = stem[len(marker) :]
        if suffix.isdigit():
            return int(suffix)
    return 0


def _defined_sphere_centers(
    reference_centers: tuple[np.ndarray, np.ndarray],
    msh_file: Path,
    *,
    dt_s: float,
    speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    bottom_reference, top_reference = reference_centers
    direction = np.asarray(USER_SPHERE_MOTION_DIRECTION, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-30:
        raise ValueError("USER_SPHERE_MOTION_DIRECTION must be nonzero.")
    direction = direction / norm
    step = _mesh_iteration_number(msh_file)
    displacement = float(step) * float(dt_s) * float(speed_mps) * direction
    return (
        np.asarray(bottom_reference, dtype=float) + displacement,
        np.asarray(top_reference, dtype=float),
    )


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


def _plot_surface(
    points: np.ndarray,
    surface_faces: np.ndarray,
    *,
    title: str,
    elev: float,
    azim: float,
    sphere_radius_m: float,
    show_spheres: bool,
    liquid_alpha: float,
    sphere_alpha: float,
    sphere_visible_fraction: float,
    vertex_size: float,
    x_limits_mm: tuple[float, float] | list[float] | None,
    y_limits_mm: tuple[float, float] | list[float] | None,
    z_limits_mm: tuple[float, float] | list[float] | None,
    clip_to_axis_limits: bool,
    sphere_centers: tuple[np.ndarray, np.ndarray] | None,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    plot_points = 1.0e3 * np.asarray(points, dtype=float)
    surface_faces = np.asarray(surface_faces, dtype=int)
    surface_vertices = np.unique(surface_faces.reshape(-1))
    surface_edges = _triangle_edges(surface_faces)

    triangles_xyz = plot_points[surface_faces]
    edge_segments = plot_points[surface_edges]

    fig = plt.figure(figsize=(8.4, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    axis_points = [plot_points[surface_vertices]]
    if show_spheres:
        (
            bottom_contact_z,
            top_contact_z,
            _bottom_contact_radius,
            _top_contact_radius,
        ) = _bridge_contact_endpoints(points, surface_faces)
        (
            bottom_center,
            top_center,
        ) = (
            sphere_centers
            if sphere_centers is not None
            else _sphere_geometry_from_bridge(
                points,
                surface_faces,
                sphere_radius_m=float(sphere_radius_m),
            )[:2]
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

    liquid_collection = Poly3DCollection(
        triangles_xyz,
        facecolor=LIQUID_FACE_COLOR,
        edgecolor="none",
        alpha=_clamp_alpha(liquid_alpha),
        zorder=10,
        axlim_clip=bool(clip_to_axis_limits),
    )
    ax.add_collection3d(liquid_collection)
    edge_collection = Line3DCollection(
        edge_segments,
        colors=LIQUID_EDGE_COLOR,
        linewidths=0.46,
        alpha=0.78,
        zorder=11,
        axlim_clip=bool(clip_to_axis_limits),
    )
    ax.add_collection3d(edge_collection)
    vertex_size = max(float(vertex_size), 0.0)
    ax.scatter(
        plot_points[surface_vertices, 0],
        plot_points[surface_vertices, 1],
        plot_points[surface_vertices, 2],
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
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(title, pad=12)
    ax.set_proj_type("ortho")
    ax.view_init(elev=float(elev), azim=float(azim))
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    return fig


def _mesh_stem_sort_key(stem: str) -> tuple[int, str]:
    if stem == "mesh_initial":
        return -1, stem
    marker = "mesh_iter"
    if stem.startswith(marker):
        suffix = stem[len(marker) :]
        if suffix.isdigit():
            return int(suffix), stem
    return 10**18, stem


def _mesh_sort_key(path: Path) -> tuple[int, str]:
    return _mesh_stem_sort_key(path.stem)


def _video_png_path(msh_file: Path, output_suffix: str) -> Path:
    return msh_file.with_name(f"{msh_file.stem}{output_suffix}.png")


def _gif_frame_sort_key(png_file: Path, output_suffix: str) -> tuple[int, str]:
    stem = png_file.stem
    if output_suffix and stem.endswith(output_suffix):
        stem = stem[: -len(output_suffix)]
    return _mesh_stem_sort_key(stem)


def _resolve_gif_output_path(input_dir: Path, gif_output: str | Path) -> Path:
    path = Path(gif_output).expanduser()
    if path.is_absolute():
        return path
    return input_dir / path


def _write_gif(
    frame_paths: list[Path],
    output_path: Path,
    *,
    duration_ms: int,
    loop: int,
) -> None:
    from PIL import Image

    frames = [Path(path) for path in frame_paths if Path(path).is_file()]
    if not frames:
        raise FileNotFoundError("No PNG frames are available for GIF creation.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette_mode = Image.Palette.ADAPTIVE
    with Image.open(frames[0]) as first_image:
        base_size = first_image.size
        first_frame = first_image.convert("RGB").convert("P", palette=palette_mode)

    append_frames = []
    for frame_path in frames[1:]:
        with Image.open(frame_path) as image:
            frame = image.convert("RGB")
            if frame.size != base_size:
                frame = frame.resize(base_size, Image.Resampling.LANCZOS)
            append_frames.append(frame.convert("P", palette=palette_mode))

    first_frame.save(
        output_path,
        save_all=True,
        append_images=append_frames,
        duration=max(int(duration_ms), 1),
        loop=max(int(loop), 0),
        optimize=False,
    )


def _read_reference_sphere_centers(
    msh_file: Path,
    *,
    sphere_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(msh_file)
    points = np.asarray(mesh.points, dtype=float)
    tets = _tetra_blocks(mesh)
    boundary_faces = _boundary_faces_from_tets(tets)
    surface_faces = _outer_liquid_surface_faces(points, boundary_faces)
    bottom_center, top_center, _contact_radius, _bottom_contact_z, _top_contact_z = _sphere_geometry_from_bridge(
        points,
        surface_faces,
        sphere_radius_m=float(sphere_radius_m),
    )
    return bottom_center, top_center


def _render_mesh_png(
    args: argparse.Namespace,
    msh_file: Path,
    png_file: Path,
    *,
    sphere_reference_centers: tuple[np.ndarray, np.ndarray] | None,
    keep_figure: bool,
):
    import matplotlib.pyplot as plt

    mesh = meshio.read(msh_file)
    points = np.asarray(mesh.points, dtype=float)
    candidate_faces = _triangle_blocks(mesh)
    if candidate_faces.size == 0:
        candidate_faces = _boundary_faces_from_tets(_tetra_blocks(mesh))

    sphere_centers = None
    if args.sphere_motion == "defined":
        if sphere_reference_centers is None:
            raise ValueError("Defined sphere motion needs reference sphere centers.")
        sphere_centers = _defined_sphere_centers(
            sphere_reference_centers,
            msh_file,
            dt_s=float(args.sphere_motion_dt_s),
            speed_mps=float(args.sphere_motion_speed_mps),
        )
    surface_faces = _outer_liquid_surface_faces(
        points,
        candidate_faces,
        sphere_centers=sphere_centers,
        sphere_radius_m=float(args.sphere_radius_mm) * 1.0e-3 if sphere_centers is not None else None,
    )
    surface_vertex_count = int(np.unique(surface_faces.reshape(-1)).size)

    title = (
        f"{msh_file.stem}: liquid-air outer surface only\n"
        f"surface vertices={surface_vertex_count}, surface faces={len(surface_faces)}"
    )
    fig = _plot_surface(
        points,
        surface_faces,
        title=title,
        elev=args.elev,
        azim=args.azim,
        sphere_radius_m=float(args.sphere_radius_mm) * 1.0e-3,
        show_spheres=not bool(args.no_spheres),
        liquid_alpha=float(args.liquid_alpha),
        sphere_alpha=float(args.sphere_alpha),
        sphere_visible_fraction=1.0 if bool(args.full_spheres) else float(args.sphere_visible_fraction),
        vertex_size=float(args.vertex_size),
        x_limits_mm=args.x_limits,
        y_limits_mm=args.y_limits,
        z_limits_mm=args.z_limits,
        clip_to_axis_limits=bool(USER_CLIP_TO_AXIS_LIMITS) and not bool(args.no_axis_clip),
        sphere_centers=sphere_centers,
    )
    png_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_file, dpi=180, bbox_inches="tight")
    if keep_figure:
        return fig
    plt.close(fig)
    return None


def main() -> None:
    args = _build_cli().parse_args()
    _configure_matplotlib(interactive=bool(args.interactive))

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    msh_files = sorted(input_dir.glob(str(args.pattern)), key=_mesh_sort_key)
    msh_files = [path for path in msh_files if path.is_file()]
    if args.limit is not None:
        msh_files = msh_files[: max(int(args.limit), 0)]
    if not msh_files:
        raise FileNotFoundError(f"No .msh files matched {args.pattern!r} in {input_dir}")

    sphere_reference_centers = None
    if args.sphere_motion == "defined":
        reference_msh = input_dir / "mesh_iter0000.msh"
        if not reference_msh.is_file():
            reference_msh = next((path for path in msh_files if _mesh_iteration_number(path) == 0), msh_files[0])
        sphere_reference_centers = _read_reference_sphere_centers(
            reference_msh,
            sphere_radius_m=float(args.sphere_radius_mm) * 1.0e-3,
        )

    overwrite = bool(USER_BATCH_OVERWRITE) and not bool(args.skip_existing)
    print(f"Input directory: {input_dir}")
    print(f"Meshes matched : {len(msh_files)}")
    if sphere_reference_centers is not None:
        bottom_center, top_center = sphere_reference_centers
        print(
            "Sphere motion : defined "
            f"(reference bottom z={bottom_center[2] * 1.0e3:.6f} mm, "
            f"top z={top_center[2] * 1.0e3:.6f} mm)"
        )
    interactive_limit = int(args.interactive_window_limit)
    kept_figures = []
    output_pngs = [_video_png_path(msh_file, str(args.output_suffix)) for msh_file in msh_files]
    for idx, msh_file in enumerate(msh_files, start=1):
        png_file = output_pngs[idx - 1]
        if png_file.exists() and not overwrite:
            print(f"[{idx}/{len(msh_files)}] skip existing {png_file.name}")
            continue
        print(f"[{idx}/{len(msh_files)}] {msh_file.name} -> {png_file.name}")
        keep_figure = bool(args.interactive) and (
            interactive_limit < 0 or (interactive_limit > 0 and idx > len(msh_files) - interactive_limit)
        )
        fig = _render_mesh_png(
            args,
            msh_file,
            png_file,
            sphere_reference_centers=sphere_reference_centers,
            keep_figure=keep_figure,
        )
        if fig is not None:
            kept_figures.append(fig)

    if bool(args.create_gif):
        gif_frames = sorted(
            [png_file for png_file in output_pngs if png_file.is_file()],
            key=lambda path: _gif_frame_sort_key(path, str(args.output_suffix)),
        )
        gif_output = _resolve_gif_output_path(input_dir, args.gif_output)
        print(f"Creating GIF from {len(gif_frames)} frame(s): {gif_output}")
        _write_gif(
            gif_frames,
            gif_output,
            duration_ms=int(args.gif_frame_duration_ms),
            loop=int(args.gif_loop),
        )

    print("Done.")
    if kept_figures:
        import matplotlib.pyplot as plt

        print(f"Keeping {len(kept_figures)} Matplotlib window(s) open. Close them to exit.")
        plt.show(block=True)


if __name__ == "__main__":
    main()
