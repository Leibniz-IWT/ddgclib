from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path

import meshio
import numpy as np


ROOT = Path(__file__).resolve().parent
MESH_BATCH_PATH = ROOT / "mesh_batch.py"
CASES = (
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case1",
        "case1_mesh_batch_style_fixed_top.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case2",
        "case2_mesh_batch_style_fixed_bottom_cl.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case3",
        "case3_mesh_batch_style_force_calibrated.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case4",
        "case4_mesh_batch_style_force_calibrated.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case5",
        "case5_mesh_batch_style_fixed_top.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case6",
        "case6_mesh_batch_style_fixed_bottom_cl.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case7",
        "case7_mesh_batch_style_hydrostatic_topcl.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case8",
        "case8_mesh_batch_style_hydrostatic_topcl.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case9",
        "case9_mesh_batch_style_cauchy_pr33bt_kinfgate_prefzero_hydrostatic_topcl.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case10",
        "case10_mesh_batch_style_cauchy_pr33bt_hydrostatic_topcl.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case11",
        "case11_mesh_batch_style_cauchy_pr33bt_prefheron_from_oldgood9.gif",
    ),
    (
        "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case12",
        "case12_mesh_batch_style_cauchy_pr33bt_kinfgate_prefzero_hydrostatic_topcl.gif",
    ),
)
OUTPUT_SUFFIX = "_meshbatch100_fixedtop"
SPHERE_RADIUS_M = 4.0e-3
SPHERE_MOTION_DT_S = 0.01
SPHERE_MOTION_SPEED_MPS = 5.0e-6
SPHERE_VISIBLE_FRACTION = 0.060
SPHERE_CONTACT_RING_COLOR = "#55534b"
SPHERE_OUTER_RING_COLOR = "#8d8a7f"
SPHERE_RING_LINEWIDTH = 1.05


def _load_mesh_batch():
    spec = importlib.util.spec_from_file_location("case_2b_mesh_batch", MESH_BATCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MESH_BATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mb = _load_mesh_batch()


def _surface_faces_for_file(msh_file: Path, sphere_reference_centers):
    mesh = meshio.read(msh_file)
    points = np.asarray(mesh.points, dtype=float)
    candidate_faces = mb._triangle_blocks(mesh)
    if candidate_faces.size == 0:
        candidate_faces = mb._boundary_faces_from_tets(mb._tetra_blocks(mesh))
    if msh_file.name == "mesh_initial.msh":
        sphere_centers = sphere_reference_centers
    else:
        sphere_centers = mb._defined_sphere_centers(
            sphere_reference_centers,
            msh_file,
            dt_s=SPHERE_MOTION_DT_S,
            speed_mps=SPHERE_MOTION_SPEED_MPS,
        )
    surface_faces = mb._outer_liquid_surface_faces(
        points,
        candidate_faces,
        sphere_centers=sphere_centers,
        sphere_radius_m=SPHERE_RADIUS_M,
    )
    return points, surface_faces, sphere_centers


def _initial_mesh_file(case_dir: Path) -> Path:
    mesh_iter0 = case_dir / "mesh_iter0000.msh"
    if mesh_iter0.exists():
        return mesh_iter0
    mesh_initial = case_dir / "mesh_initial.msh"
    if mesh_initial.exists():
        return mesh_initial
    raise FileNotFoundError(f"{case_dir} has neither mesh_iter0000.msh nor mesh_initial.msh")


def _mesh_frame_sort_key(path: Path) -> int:
    if path.name == "mesh_initial.msh":
        return 0
    return int(path.stem.replace("mesh_iter", ""))


def _png_frame_sort_key(path: Path) -> int:
    stem = path.stem.replace(OUTPUT_SUFFIX, "")
    if stem == "mesh_initial":
        return 0
    return int(stem.replace("mesh_iter", ""))


def _case_mesh_files(case_dir: Path) -> list[Path]:
    files = sorted(case_dir.glob("mesh_iter*.msh"), key=_mesh_frame_sort_key)
    regular_files = [
        path
        for path in files
        if path.name == "mesh_iter0000.msh" or _mesh_frame_sort_key(path) % 100 == 0
    ]
    if len(regular_files) >= 50:
        files = regular_files
    if files and files[0].name == "mesh_iter0000.msh":
        return files
    mesh_initial = case_dir / "mesh_initial.msh"
    if mesh_initial.exists():
        return [mesh_initial] + files
    return files


def _case_png_files(case_dir: Path) -> list[Path]:
    files = sorted(case_dir.glob(f"*{OUTPUT_SUFFIX}.png"), key=_png_frame_sort_key)
    return files


def _reference_top_contact_z(case_dir: Path, sphere_reference_centers) -> float:
    points, surface_faces, _sphere_centers = _surface_faces_for_file(
        _initial_mesh_file(case_dir),
        sphere_reference_centers,
    )
    _bottom_z, top_z, _bottom_r, _top_r = mb._bridge_contact_endpoints(points, surface_faces)
    return float(top_z)


def _plot_surface_fixed_top_cap(
    points: np.ndarray,
    surface_faces: np.ndarray,
    *,
    title: str,
    sphere_centers,
    fixed_top_contact_z: float,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    plot_points = 1.0e3 * np.asarray(points, dtype=float)
    surface_faces = np.asarray(surface_faces, dtype=int)
    surface_vertices = np.unique(surface_faces.reshape(-1))
    surface_edges = mb._triangle_edges(surface_faces)

    triangles_xyz = plot_points[surface_faces]
    edge_segments = plot_points[surface_edges]

    fig = plt.figure(figsize=(8.4, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    axis_points = [plot_points[surface_vertices]]
    bottom_contact_z, _top_contact_z, _bottom_radius, _top_radius = mb._bridge_contact_endpoints(
        points,
        surface_faces,
    )
    bottom_center, top_center = sphere_centers
    sphere_specs = (
        (bottom_center, bottom_contact_z),
        (top_center, fixed_top_contact_z),
    )
    for center, contact_z in sphere_specs:
        phi_min, phi_max = mb._sphere_cap_phi_limits(
            center,
            contact_z,
            SPHERE_RADIUS_M,
            SPHERE_VISIBLE_FRACTION,
        )
        if phi_max <= phi_min:
            continue
        sx, sy, sz = mb._sphere_surface_xyz(
            center,
            SPHERE_RADIUS_M,
            phi_min=phi_min,
            phi_max=phi_max,
        )
        ax.plot_surface(
            sx,
            sy,
            sz,
            facecolors=mb._sphere_facecolors(
                sx,
                sy,
                sz,
                center,
                SPHERE_RADIUS_M,
                alpha=mb.USER_SPHERE_ALPHA,
            ),
            linewidth=0.0,
            antialiased=True,
            shade=False,
            zorder=1,
            axlim_clip=bool(mb.USER_CLIP_TO_AXIS_LIMITS),
        )
        for ring_idx, phi in enumerate((phi_min, phi_max)):
            theta = np.linspace(0.0, 2.0 * np.pi, 193, dtype=float)
            radius_mm = 1.0e3 * SPHERE_RADIUS_M
            center_mm = 1.0e3 * np.asarray(center, dtype=float)
            x_ring = center_mm[0] + radius_mm * np.sin(phi) * np.cos(theta)
            y_ring = center_mm[1] + radius_mm * np.sin(phi) * np.sin(theta)
            z_ring = center_mm[2] + radius_mm * np.cos(phi) * np.ones_like(theta)
            ring_color = SPHERE_CONTACT_RING_COLOR
            if (ring_idx == 0 and center[2] > 0.0) or (ring_idx == 1 and center[2] < 0.0):
                ring_color = SPHERE_OUTER_RING_COLOR
            ax.plot(
                x_ring,
                y_ring,
                z_ring,
                color=ring_color,
                linewidth=SPHERE_RING_LINEWIDTH,
                alpha=0.92,
                zorder=7,
            )
        axis_points.append(np.column_stack((sx.reshape(-1), sy.reshape(-1), sz.reshape(-1))))

    liquid_collection = Poly3DCollection(
        triangles_xyz,
        facecolor=mb.LIQUID_FACE_COLOR,
        edgecolor="none",
        alpha=mb._clamp_alpha(mb.USER_LIQUID_BRIDGE_ALPHA),
        zorder=10,
        axlim_clip=bool(mb.USER_CLIP_TO_AXIS_LIMITS),
    )
    ax.add_collection3d(liquid_collection)
    edge_collection = Line3DCollection(
        edge_segments,
        colors=mb.LIQUID_EDGE_COLOR,
        linewidths=0.46,
        alpha=0.78,
        zorder=11,
        axlim_clip=bool(mb.USER_CLIP_TO_AXIS_LIMITS),
    )
    ax.add_collection3d(edge_collection)

    vertex_size = max(float(mb.USER_VERTEX_SIZE), 0.0)
    ax.scatter(
        plot_points[surface_vertices, 0],
        plot_points[surface_vertices, 1],
        plot_points[surface_vertices, 2],
        s=vertex_size,
        c=mb.LIQUID_EDGE_COLOR,
        alpha=0.70,
        depthshade=False,
        zorder=12,
        axlim_clip=bool(mb.USER_CLIP_TO_AXIS_LIMITS),
    )

    axis_center, axis_ranges = mb._set_axes_equal(ax, np.vstack(axis_points))
    axis_limits = mb._initial_axis_limits(
        axis_center,
        axis_ranges,
        x_limits_mm=mb.USER_X_LIMITS_MM,
        y_limits_mm=mb.USER_Y_LIMITS_MM,
        z_limits_mm=mb.USER_Z_LIMITS_MM,
    )
    mb._apply_axis_limits(ax, axis_limits)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(title, pad=12)
    ax.set_proj_type("ortho")
    ax.view_init(elev=float(mb.USER_INTERACTIVE_ELEV_DEG), azim=float(mb.USER_INTERACTIVE_AZIM_DEG))
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    return fig


def _render_case(case_dir: Path, gif_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    msh_files = _case_mesh_files(case_dir)
    if not msh_files:
        raise RuntimeError(f"{case_dir} has no mesh frames")
    if len(msh_files) != 51:
        print(f"Warning: {case_dir.name} has {len(msh_files)} mesh frames, not 51")

    sphere_reference_centers = mb._read_reference_sphere_centers(
        _initial_mesh_file(case_dir),
        sphere_radius_m=SPHERE_RADIUS_M,
    )
    fixed_top_contact_z = _reference_top_contact_z(case_dir, sphere_reference_centers)
    print(
        f"{case_dir.name}: fixed top center z={sphere_reference_centers[1][2] * 1.0e3:.6f} mm, "
        f"fixed top contact z={fixed_top_contact_z * 1.0e3:.6f} mm"
    )

    output_pngs = []
    for idx, msh_file in enumerate(msh_files, start=1):
        png_file = msh_file.with_name(f"{msh_file.stem}{OUTPUT_SUFFIX}.png")
        points, surface_faces, sphere_centers = _surface_faces_for_file(
            msh_file,
            sphere_reference_centers,
        )
        surface_vertex_count = int(np.unique(surface_faces.reshape(-1)).size)
        title = (
            f"{msh_file.stem}: liquid-air outer surface only\n"
            f"surface vertices={surface_vertex_count}, surface faces={len(surface_faces)}"
        )
        fig = _plot_surface_fixed_top_cap(
            points,
            surface_faces,
            title=title,
            sphere_centers=sphere_centers,
            fixed_top_contact_z=fixed_top_contact_z,
        )
        fig.savefig(png_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_pngs.append(png_file)
        print(f"[{idx}/{len(msh_files)}] {msh_file.name} -> {png_file.name}")

    gif_path = case_dir / gif_name
    gif_frames = sorted(output_pngs, key=_png_frame_sort_key)
    mb._write_gif(
        gif_frames,
        gif_path,
        duration_ms=mb.USER_GIF_FRAME_DURATION_MS,
        loop=mb.USER_GIF_LOOP,
    )
    print(f"Wrote {gif_path.resolve()}")


def _write_case_gif_from_pngs(case_dir: Path, gif_name: str) -> None:
    png_files = _case_png_files(case_dir)
    if not png_files:
        raise RuntimeError(
            f"{case_dir} has no *{OUTPUT_SUFFIX}.png frames. "
            "Run without --gif-only first to render them from the mesh frames."
        )
    if len(png_files) != 51:
        print(f"Warning: {case_dir.name} has {len(png_files)} PNG frames, not 51")
    gif_path = case_dir / gif_name
    mb._write_gif(
        png_files,
        gif_path,
        duration_ms=mb.USER_GIF_FRAME_DURATION_MS,
        loop=mb.USER_GIF_LOOP,
    )
    print(f"Wrote {gif_path.resolve()} from {len(png_files)} PNG frames")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Render only case names or prefixes such as case7 case8.",
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Create GIFs from existing fixed-top PNG frames without rerendering meshes.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    selected = CASES
    if args.only:
        prefixes = tuple(str(item) for item in args.only)
        selected = tuple(
            (case_name, gif_name)
            for case_name, gif_name in CASES
            if case_name.startswith(prefixes)
            )
        if not selected:
            raise RuntimeError(f"No cases matched --only {args.only!r}")
    for case_name, gif_name in selected:
        case_dir = root / case_name
        if args.gif_only:
            _write_case_gif_from_pngs(case_dir, gif_name)
        else:
            _render_case(case_dir, gif_name)


if __name__ == "__main__":
    main()
