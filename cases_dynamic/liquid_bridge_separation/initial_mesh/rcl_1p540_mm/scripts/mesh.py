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
USER_LIQUID_BRIDGE_ALPHA = 1
USER_SPHERE_ALPHA = 1
USER_SPHERE_VISIBLE_FRACTION = 1.0 / 10 #0.028#
USER_VERTEX_SIZE = 0.0
USER_SAVE_PNG = True
USER_SAVE_PNG_NAME = "mesh.png"
USER_X_LIMITS_MM = (-3.0, 3.0)
USER_Y_LIMITS_MM = (-3.0, 3.0)
USER_Z_LIMITS_MM = (-2, 2)
USER_CLIP_TO_AXIS_LIMITS = True

DEFAULT_PARTICLE_RADIUS_M = 4.0e-3
LIQUID_FACE_COLOR = "#8fb0c5"
LIQUID_EDGE_COLOR = "#637988"
SPHERE_FACE_COLOR = "#c8c6bc"


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display only the outer liquid-air surface of a Case 2b Gmsh tetra mesh."
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
        "--save",
        type=Path,
        default=SCRIPT_DIR / USER_SAVE_PNG_NAME if USER_SAVE_PNG else None,
        help="PNG path to save. Default: mesh.png next to this script.",
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
        print(cell_block.type, cell_block.data.shape)
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

    vol_tetra = np.array(
        [
            _signed_tet_volume(points[t[0]], points[t[1]], points[t[2]], points[t[3]])
            for t in tets
        ],
        dtype=float,
    )

    boundary_faces = _boundary_faces_from_tets(tets)
    surface_faces = _outer_liquid_surface_faces(points, boundary_faces)
    surface_vertex_count = int(np.unique(surface_faces.reshape(-1)).size)
    surface_edge_count = int(len(_triangle_edges(surface_faces)))

    print("sum(abs(vol_tetra))       =", np.sum(np.abs(vol_tetra)))
    print("sum(vol_tetra)            =", np.sum(vol_tetra))
    print("min(vol_tetra)            =", np.min(vol_tetra))
    print("max(vol_tetra)            =", np.max(vol_tetra))
    print("number of tetrahedra      =", len(tets))
    print("number of boundary faces  =", len(boundary_faces))
    print("liquid-air surface faces  =", len(surface_faces))
    print("liquid-air surface edges  =", surface_edge_count)
    print("liquid-air surface verts  =", surface_vertex_count)

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
    )

    if args.save is not None:
        save_path = Path(args.save).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
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
