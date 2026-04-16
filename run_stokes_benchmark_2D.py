#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

from benchmarks._benchmark_cases import (
    StokesSphereBenchmark2D,
    StokesSphereBenchmark2DDDG,
    StokesSphereBenchmark2DHC,
    StokesSphereBenchmark2DHCDDG,
    StokesSphereBenchmark2DHCDDGSymmNonobtuse,
    StokesSphereBenchmark2DHCSymmNonobtuse,
    StokesSphereBenchmark2DDDGSymmNonobtuse,
    StokesSphereBenchmark2DSymmNonobtuse,
    _generate_stokes_sphere_hc_mesh,
)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run the 2D axisymmetric Stokes-sphere benchmark.",
    )
    parser.add_argument("--sphere-radius", type=float, default=1.0)
    parser.add_argument("--outer-radius", type=float, default=6.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--density-difference", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--flow-speed", type=float, default=None)
    parser.add_argument("--mesh-size-inner", type=float, default=None)
    parser.add_argument("--mesh-size-outer", type=float, default=None)
    parser.add_argument("--mesh-order", type=int, default=2)
    parser.add_argument(
        "--mesh-source",
        type=str,
        choices=("gmsh", "hc", "both"),
        default="both",
        help="Mesh source: gmsh, hc, or both (for side-by-side comparison).",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="benchmarks_out/stokes_sphere_2d",
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="stokes_sphere_2d_axisymmetric.msh",
    )
    parser.add_argument(
        "--remesh",
        action="store_true",
        help="Regenerate the Gmsh mesh even if it already exists.",
    )
    parser.add_argument("--hc-refine", type=int, default=3)
    parser.add_argument("--hc-inner-samples", type=int, default=None)
    parser.add_argument("--hc-outer-samples", type=int, default=None)
    parser.add_argument(
        "--auto-match-hc-cells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In 'both' mode, auto-tune HC mesh parameters to match Gmsh volume-cell count.",
    )
    parser.add_argument(
        "--auto-match-mode",
        type=str,
        choices=("cells", "balanced", "gmsh_style"),
        default="gmsh_style",
        help=(
            "HC auto-match objective in 'both' mode: "
            "'cells' (match volume cells), "
            "'balanced' (trade off cells and boundary density), "
            "'gmsh_style' (match Gmsh boundary density first)."
        ),
    )
    parser.add_argument(
        "--save-mesh-pngs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save mesh snapshots as PNG files.",
    )
    parser.add_argument(
        "--mesh-png-dir",
        type=str,
        default=None,
        help="Directory where mesh PNG files are saved (defaults to <workdir>/mesh_pngs).",
    )
    return parser


def _format_value(value):
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "nan"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _print_summary(bench):
    print(f"Benchmark: {bench.name}")
    print(f"Mesh: {bench.mesh_path}")
    print("")
    meta = getattr(bench, "mesh_metadata", {}) or {}
    n_obstacle = int(meta.get("n_obstacle_elements", 0))
    n_outer = int(meta.get("n_outer_boundary_elements", 0))
    n_cells = int(meta.get("n_volume_cells", 0))
    print(f"Boundary Elements: obstacle={n_obstacle}, outer={n_outer}, volume_cells={n_cells}")
    print("")
    for metric, (computed, analytical) in bench.summary().items():
        print(
            f"{metric}: "
            f"computed={_format_value(computed)}, "
            f"analytical={_format_value(analytical)}"
        )


def _run_and_print(bench):
    bench.run_benchmark()
    _print_summary(bench)
    return bench


def _print_case_header(case_idx, title):
    print(f"=== Case {int(case_idx)}. {title} ===")


def _print_rel_error_by_case(case_entries):
    print("\n=== Relative Error by Case ===")
    for idx, (label, bench) in enumerate(case_entries, start=1):
        if bench is None:
            value = "not evaluated"
        else:
            value = _format_value(getattr(bench, "force_error_rel", np.nan))
        print(f"Case {idx} ({label}): Drag Rel Error = {value}")


def _print_vertex_rel_error_by_case(case_entries):
    print("\n=== Vertex Drag Relative Error by Case ===")
    for idx, (label, bench) in enumerate(case_entries, start=1):
        if bench is None:
            mean_val = "not evaluated"
            max_val = "not evaluated"
        else:
            mean_val = _format_value(getattr(bench, "vertex_drag_rel_error_mean", np.nan))
            max_val = _format_value(getattr(bench, "vertex_drag_rel_error_max", np.nan))
        print(f"Case {idx} ({label}): vertexDragRelErrorMean = {mean_val}, vertexDragRelErrorMax = {max_val}")


def _run_case(case_idx, title, benchmark_cls, kwargs):
    print("" if int(case_idx) > 1 else "")
    _print_case_header(case_idx, title)
    try:
        return _run_and_print(benchmark_cls(**kwargs))
    except ImportError as exc:
        print(f"{title} skipped: {exc}")
        return None


def _save_mesh_png(bench, out_path, title):
    points = np.asarray(bench.points, dtype=float)
    simplices = np.asarray(bench.volume_cells, dtype=np.int64)
    if points.ndim != 2 or simplices.ndim != 2 or simplices.shape[1] != 3:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 1050
    margin = 40
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    xy = points[:, :2]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    dx = max(x_max - x_min, 1.0e-12)
    dy = max(y_max - y_min, 1.0e-12)
    scale = min((width - 2 * margin) / dx, (height - 2 * margin) / dy)
    x_pad = (width - scale * dx) * 0.5
    y_pad = (height - scale * dy) * 0.5

    def to_px(p):
        px = x_pad + (float(p[0]) - x_min) * scale
        py = height - (y_pad + (float(p[1]) - y_min) * scale)
        return (px, py)

    for tri in simplices:
        a, b, c = [to_px(points[int(i), :2]) for i in tri]
        draw.line([a, b], fill=(140, 140, 140), width=1)
        draw.line([b, c], fill=(140, 140, 140), width=1)
        draw.line([c, a], fill=(140, 140, 140), width=1)

    def draw_boundary(elements, color, width_px):
        if elements is None:
            return
        arr = np.asarray(elements, dtype=np.int64)
        if arr.size == 0:
            return
        for elem in arr:
            node_ids = np.asarray(elem, dtype=np.int64)
            nodes = points[node_ids, :2]
            if len(node_ids) == 3:
                # line3 in gmsh/meshio ordering: [end0, end1, midpoint]
                nodes = nodes[[0, 2, 1]]
            pts = [to_px(p) for p in nodes]
            draw.line(pts, fill=color, width=width_px)

    draw_boundary(bench.outer_boundary_elements, color=(42, 157, 143), width_px=2)
    draw_boundary(bench.obstacle_elements, color=(230, 57, 70), width_px=3)
    draw_boundary(bench.axis_elements, color=(29, 53, 87), width_px=2)
    img.save(out_path)
    return str(out_path)


def _default_hc_samples(refine_level):
    refine_level = int(max(refine_level, 0))
    n_inner = max(32, 16 * (2 ** max(refine_level - 1, 0)))
    n_outer = max(48, 24 * (2 ** max(refine_level - 1, 0)))
    return int(n_inner), int(n_outer)


def _default_gmsh_sizes_2d(args):
    h_inner = args.mesh_size_inner if args.mesh_size_inner is not None else 0.20 * float(args.sphere_radius)
    h_outer = args.mesh_size_outer if args.mesh_size_outer is not None else 0.75 * float(args.sphere_radius)
    return float(h_inner), float(h_outer)


def _triangle_quality_stats(points, simplices):
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(simplices, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or tri.size == 0:
        return 0.0, 0.0

    a = pts[tri[:, 0], :2]
    b = pts[tri[:, 1], :2]
    c = pts[tri[:, 2], :2]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    quality = (2.0 * np.sqrt(3.0) * area2) / (ab * ab + bc * bc + ca * ca + 1.0e-30)
    quality = np.asarray(quality, dtype=float)
    finite = np.isfinite(quality)
    if not np.any(finite):
        return 0.0, 0.0
    quality = quality[finite]
    return float(np.median(quality)), float(np.min(quality))


def _estimate_hc_volume_cells(args, refine_level, inner_samples, outer_samples, symm_nonobtuse=False):
    points, simplices, obstacle, outer_boundary, _ = _generate_stokes_sphere_hc_mesh(
        dim=2,
        sphere_radius=args.sphere_radius,
        outer_radius=args.outer_radius,
        hc_refine=int(refine_level),
        hc_inner_samples=int(inner_samples),
        hc_outer_samples=int(outer_samples),
        symm_nonobtuse=bool(symm_nonobtuse),
    )
    q_med, q_min = _triangle_quality_stats(points, simplices)
    return int(simplices.shape[0]), q_med, q_min, int(len(obstacle)), int(len(outer_boundary))


def _auto_tune_hc_to_target_cells(
    args,
    target_cells,
    target_obstacle=None,
    target_outer=None,
    match_mode="cells",
    symm_nonobtuse=False,
):
    base_ref = int(max(args.hc_refine, 0))
    if args.hc_inner_samples is None or args.hc_outer_samples is None:
        base_inner, base_outer = _default_hc_samples(base_ref)
    else:
        base_inner = int(args.hc_inner_samples)
        base_outer = int(args.hc_outer_samples)

    refine_candidates = list(range(max(0, base_ref - 2), base_ref + 3))
    scale_candidates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

    best = None
    for refine in refine_candidates:
        if args.hc_inner_samples is None or args.hc_outer_samples is None:
            ref_inner, ref_outer = _default_hc_samples(refine)
        else:
            ref_inner, ref_outer = base_inner, base_outer

        inner_candidates = {max(8, int(round(ref_inner * s_in))) for s_in in scale_candidates}
        outer_candidates = {max(12, int(round(ref_outer * s_out))) for s_out in scale_candidates}
        if target_obstacle is not None and target_outer is not None and match_mode in ("balanced", "gmsh_style"):
            inner_target = max(8, int(target_obstacle) + 1)
            outer_target = max(12, int(target_outer) + 1)
            for f in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
                inner_candidates.add(max(8, int(round(inner_target * f))))
                outer_candidates.add(max(12, int(round(outer_target * f))))

        for inner in sorted(inner_candidates):
            for outer in sorted(outer_candidates):
                if outer < inner:
                    continue
                try:
                    n_cells, q_med, q_min, n_obstacle, n_outer = _estimate_hc_volume_cells(
                        args,
                        refine,
                        inner,
                        outer,
                        symm_nonobtuse=symm_nonobtuse,
                    )
                except Exception:
                    continue

                err = abs(n_cells - target_cells)
                quality_class = 0 if (q_med >= 0.60 and q_min >= 0.04) else 1
                cell_scale_class = 0 if err <= max(50, int(0.25 * target_cells)) else 1
                if target_obstacle is None or target_outer is None:
                    err_boundary = 0
                    objective = (err,)
                else:
                    err_boundary = abs(int(n_obstacle) - int(target_obstacle)) + abs(int(n_outer) - int(target_outer))
                    if match_mode == "gmsh_style":
                        objective = (err_boundary, err)
                    elif match_mode == "balanced":
                        objective = (3 * err_boundary + err, err_boundary, err)
                    else:
                        objective = (err, err_boundary)

                candidate = (
                    quality_class,
                    cell_scale_class,
                    *objective,
                    -q_med,
                    -q_min,
                    n_cells,
                    n_obstacle,
                    n_outer,
                    refine,
                    inner,
                    outer,
                )
                if best is None or candidate < best:
                    best = candidate

    if best is None:
        return None

    quality_class = int(best[0])
    neg_q_med = float(best[-8])
    neg_q_min = float(best[-7])
    n_cells = int(best[-6])
    n_obstacle = int(best[-5])
    n_outer = int(best[-4])
    refine = int(best[-3])
    inner = int(best[-2])
    outer = int(best[-1])
    err = abs(n_cells - int(target_cells))
    err_boundary = None
    if target_obstacle is not None and target_outer is not None:
        err_boundary = abs(n_obstacle - int(target_obstacle)) + abs(n_outer - int(target_outer))
    return {
        "hc_refine": refine,
        "hc_inner_samples": inner,
        "hc_outer_samples": outer,
        "n_cells": n_cells,
        "target_cells": int(target_cells),
        "abs_error": int(err),
        "n_obstacle": n_obstacle,
        "n_outer": n_outer,
        "target_obstacle": None if target_obstacle is None else int(target_obstacle),
        "target_outer": None if target_outer is None else int(target_outer),
        "boundary_abs_error": None if err_boundary is None else int(err_boundary),
        "quality_class": int(quality_class),
        "quality_median": float(-neg_q_med),
        "quality_min": float(-neg_q_min),
    }


def _probe_mesh_metadata(benchmark_cls, kwargs):
    probe = benchmark_cls(**kwargs)
    probe.generate_mesh()
    probe.load_mesh()
    return dict(getattr(probe, "mesh_metadata", {}) or {})


def _auto_tune_gmsh_to_target_boundaries_2d(
    args,
    benchmark_cls,
    base_kwargs,
    target_obstacle,
    target_outer,
    max_iter=3,
):
    target_obstacle = int(target_obstacle) if target_obstacle is not None else 0
    target_outer = int(target_outer) if target_outer is not None else 0
    if target_obstacle <= 0 or target_outer <= 0:
        return None

    kwargs = dict(base_kwargs)
    kwargs["remesh"] = True
    inner0, outer0 = _default_gmsh_sizes_2d(args)
    if kwargs.get("mesh_size_inner") is not None:
        inner0 = float(kwargs["mesh_size_inner"])
    if kwargs.get("mesh_size_outer") is not None:
        outer0 = float(kwargs["mesh_size_outer"])
    inner = float(inner0)
    outer = float(max(outer0, inner))

    min_h = 0.02 * float(args.sphere_radius)
    max_h = 2.0 * float(args.outer_radius)
    best = None
    stagnant_iters = 0
    prev_counts = None

    for _ in range(max(int(max_iter), 1)):
        kwargs["mesh_size_inner"] = float(inner)
        kwargs["mesh_size_outer"] = float(outer)
        try:
            meta = _probe_mesh_metadata(benchmark_cls, kwargs)
        except Exception:
            break

        n_obstacle = int(meta.get("n_obstacle_elements", 0))
        n_outer = int(meta.get("n_outer_boundary_elements", 0))
        n_cells = int(meta.get("n_volume_cells", 0))
        err_obstacle = abs(n_obstacle - target_obstacle)
        err_outer = abs(n_outer - target_outer)
        err_sum = err_obstacle + err_outer
        candidate = (
            int(err_sum),
            int(err_obstacle),
            int(err_outer),
            float(inner),
            float(outer),
            int(n_obstacle),
            int(n_outer),
            int(n_cells),
        )
        if best is None or candidate < best:
            best = candidate
        if err_sum == 0:
            break

        current_counts = (int(n_obstacle), int(n_outer))
        if prev_counts is not None and current_counts == prev_counts:
            stagnant_iters += 1
        else:
            stagnant_iters = 0
        prev_counts = current_counts
        if stagnant_iters >= 1:
            break

        ratio_inner = max(float(n_obstacle), 1.0) / max(float(target_obstacle), 1.0)
        ratio_outer = max(float(n_outer), 1.0) / max(float(target_outer), 1.0)
        inner = float(np.clip(inner * np.sqrt(ratio_inner), min_h, max_h))
        outer = float(np.clip(outer * np.sqrt(ratio_outer), min_h, max_h))
        if outer < inner:
            outer = inner

    if best is None:
        return None

    tuned_kwargs = dict(base_kwargs)
    tuned_kwargs["mesh_size_inner"] = float(best[3])
    tuned_kwargs["mesh_size_outer"] = float(best[4])
    tuned_kwargs["remesh"] = True
    return {
        "kwargs": tuned_kwargs,
        "mesh_size_inner": float(best[3]),
        "mesh_size_outer": float(best[4]),
        "n_obstacle": int(best[5]),
        "n_outer": int(best[6]),
        "n_cells": int(best[7]),
        "target_obstacle": int(target_obstacle),
        "target_outer": int(target_outer),
        "boundary_abs_error": int(best[0]),
    }


def main():
    args = _build_parser().parse_args()
    gmsh_kwargs = dict(
        sphere_radius=args.sphere_radius,
        outer_radius=args.outer_radius,
        mu=args.mu,
        density_difference=args.density_difference,
        gravity=args.gravity,
        flow_speed=args.flow_speed,
        mesh_size_inner=args.mesh_size_inner,
        mesh_size_outer=args.mesh_size_outer,
        mesh_order=args.mesh_order,
        workdir=args.workdir,
        mesh_name=args.mesh_name,
        remesh=args.remesh,
    )
    hc_kwargs = dict(
        sphere_radius=args.sphere_radius,
        outer_radius=args.outer_radius,
        mu=args.mu,
        density_difference=args.density_difference,
        gravity=args.gravity,
        flow_speed=args.flow_speed,
        workdir=f"{args.workdir}_hc",
        hc_refine=args.hc_refine,
        hc_inner_samples=args.hc_inner_samples,
        hc_outer_samples=args.hc_outer_samples,
    )
    gmsh_symm_kwargs = dict(
        sphere_radius=args.sphere_radius,
        outer_radius=args.outer_radius,
        mu=args.mu,
        density_difference=args.density_difference,
        gravity=args.gravity,
        flow_speed=args.flow_speed,
        mesh_size_inner=args.mesh_size_inner,
        mesh_size_outer=args.mesh_size_outer,
        mesh_order=args.mesh_order,
        workdir=f"{args.workdir}_symm_nonobtuse",
        mesh_name="stokes_sphere_2d_symm_nonobtuse.msh",
        remesh=args.remesh,
        symm_nonobtuse=True,
    )
    hc_symm_kwargs = dict(
        sphere_radius=args.sphere_radius,
        outer_radius=args.outer_radius,
        mu=args.mu,
        density_difference=args.density_difference,
        gravity=args.gravity,
        flow_speed=args.flow_speed,
        workdir=f"{args.workdir}_hc_symm_nonobtuse",
        hc_refine=args.hc_refine,
        hc_inner_samples=args.hc_inner_samples,
        hc_outer_samples=args.hc_outer_samples,
        symm_nonobtuse=True,
    )
    png_dir = Path(args.mesh_png_dir) if args.mesh_png_dir else (Path(args.workdir) / "mesh_pngs")

    if args.mesh_source == "gmsh":
        gmsh_bench = _run_case(1, "Gmsh Unsymm+Obtuse Finite-Difference Operator Benchmark", StokesSphereBenchmark2D, gmsh_kwargs)
        gmsh_ddg_bench = _run_case(2, "Gmsh Unsymm+Obtuse DDG Operator Benchmark", StokesSphereBenchmark2DDDG, gmsh_kwargs)
        gmsh_symm_bench = _run_case(
            3,
            "Gmsh Symm+Nonobtuse Finite-Difference Operator Benchmark",
            StokesSphereBenchmark2DSymmNonobtuse,
            gmsh_symm_kwargs,
        )
        gmsh_symm_ddg_bench = _run_case(
            4,
            "Gmsh Symm+Nonobtuse DDG Operator Benchmark",
            StokesSphereBenchmark2DDDGSymmNonobtuse,
            gmsh_symm_kwargs,
        )
        if args.save_mesh_pngs:
            if gmsh_bench is not None:
                png = _save_mesh_png(gmsh_bench, png_dir / "stokes_2d_gmsh_mesh.png", "StokesSphere2D (Gmsh)")
                if png is not None:
                    print(f"Mesh PNG (Case 1): {png}")
            if gmsh_symm_bench is not None:
                png = _save_mesh_png(
                    gmsh_symm_bench,
                    png_dir / "stokes_2d_gmsh_symm_nonobtuse_mesh.png",
                    "StokesSphere2D Symm+Nonobtuse (Gmsh)",
                )
                if png is not None:
                    print(f"Mesh PNG (Case 3): {png}")
        _print_rel_error_by_case([
            ("Gmsh Unsymm+Obtuse Finite-Difference Operator Benchmark", gmsh_bench),
            ("Gmsh Unsymm+Obtuse DDG Operator Benchmark", gmsh_ddg_bench),
            ("Gmsh Symm+Nonobtuse Finite-Difference Operator Benchmark", gmsh_symm_bench),
            ("Gmsh Symm+Nonobtuse DDG Operator Benchmark", gmsh_symm_ddg_bench),
        ])
        _print_vertex_rel_error_by_case([
            ("Gmsh Unsymm+Obtuse Finite-Difference Operator Benchmark", gmsh_bench),
            ("Gmsh Unsymm+Obtuse DDG Operator Benchmark", gmsh_ddg_bench),
            ("Gmsh Symm+Nonobtuse Finite-Difference Operator Benchmark", gmsh_symm_bench),
            ("Gmsh Symm+Nonobtuse DDG Operator Benchmark", gmsh_symm_ddg_bench),
        ])
        return

    if args.mesh_source == "hc":
        hc_bench = _run_case(1, "HC Unsymm+Obtuse Finite-Difference Operator Benchmark", StokesSphereBenchmark2DHC, hc_kwargs)
        hc_ddg_bench = _run_case(2, "HC Unsymm+Obtuse DDG Operator Benchmark", StokesSphereBenchmark2DHCDDG, hc_kwargs)
        hc_symm_bench = _run_case(
            3,
            "HC Symm+Nonobtuse Finite-Difference Operator Benchmark",
            StokesSphereBenchmark2DHCSymmNonobtuse,
            hc_symm_kwargs,
        )
        hc_symm_ddg_bench = _run_case(
            4,
            "HC Symm+Nonobtuse DDG Operator Benchmark",
            StokesSphereBenchmark2DHCDDGSymmNonobtuse,
            hc_symm_kwargs,
        )
        if args.save_mesh_pngs:
            if hc_bench is not None:
                png = _save_mesh_png(hc_bench, png_dir / "stokes_2d_hc_mesh.png", "StokesSphere2DHC (HC)")
                if png is not None:
                    print(f"Mesh PNG (Case 1): {png}")
            if hc_symm_bench is not None:
                png = _save_mesh_png(
                    hc_symm_bench,
                    png_dir / "stokes_2d_hc_symm_nonobtuse_mesh.png",
                    "StokesSphere2D Symm+Nonobtuse (HC)",
                )
                if png is not None:
                    print(f"Mesh PNG (Case 3): {png}")
        _print_rel_error_by_case([
            ("HC Unsymm+Obtuse Finite-Difference Operator Benchmark", hc_bench),
            ("HC Unsymm+Obtuse DDG Operator Benchmark", hc_ddg_bench),
            ("HC Symm+Nonobtuse Finite-Difference Operator Benchmark", hc_symm_bench),
            ("HC Symm+Nonobtuse DDG Operator Benchmark", hc_symm_ddg_bench),
        ])
        _print_vertex_rel_error_by_case([
            ("HC Unsymm+Obtuse Finite-Difference Operator Benchmark", hc_bench),
            ("HC Unsymm+Obtuse DDG Operator Benchmark", hc_ddg_bench),
            ("HC Symm+Nonobtuse Finite-Difference Operator Benchmark", hc_symm_bench),
            ("HC Symm+Nonobtuse DDG Operator Benchmark", hc_symm_ddg_bench),
        ])
        return

    gmsh_bench = _run_case(1, "Gmsh Unsymm+Obtuse Finite-Difference Operator Benchmark", StokesSphereBenchmark2D, gmsh_kwargs)
    global_target_cells = None
    global_target_obstacle = None
    global_target_outer = None
    if gmsh_bench is not None:
        global_target_cells = int(gmsh_bench.mesh_metadata.get("n_volume_cells", 0))
        global_target_obstacle = int(gmsh_bench.mesh_metadata.get("n_obstacle_elements", 0))
        global_target_outer = int(gmsh_bench.mesh_metadata.get("n_outer_boundary_elements", 0))

    if gmsh_bench is not None and args.auto_match_hc_cells:
        tuned = _auto_tune_hc_to_target_cells(
            args,
            global_target_cells,
            target_obstacle=global_target_obstacle,
            target_outer=global_target_outer,
            match_mode=args.auto_match_mode,
            symm_nonobtuse=False,
        )
        if tuned is not None:
            hc_kwargs["hc_refine"] = tuned["hc_refine"]
            hc_kwargs["hc_inner_samples"] = tuned["hc_inner_samples"]
            hc_kwargs["hc_outer_samples"] = tuned["hc_outer_samples"]
            print(
                "\nHC auto-match: "
                f"mode={args.auto_match_mode}, "
                f"target_cells={tuned['target_cells']}, "
                f"chosen_cells={tuned['n_cells']}, "
                f"hc_refine={tuned['hc_refine']}, "
                f"hc_inner_samples={tuned['hc_inner_samples']}, "
                f"hc_outer_samples={tuned['hc_outer_samples']}, "
                f"target_obstacle={_format_value(tuned['target_obstacle'])}, "
                f"chosen_obstacle={tuned['n_obstacle']}, "
                f"target_outer={_format_value(tuned['target_outer'])}, "
                f"chosen_outer={tuned['n_outer']}, "
                f"quality_median={_format_value(tuned['quality_median'])}, "
                f"quality_min={_format_value(tuned['quality_min'])}"
            )
        else:
            print("\nHC auto-match skipped: no valid HC candidate mesh was produced.")

    hc_bench = _run_case(2, "HC Unsymm+Obtuse Finite-Difference Operator Benchmark", StokesSphereBenchmark2DHC, hc_kwargs)
    gmsh_ddg_bench = _run_case(3, "Gmsh Unsymm+Obtuse DDG Operator Benchmark", StokesSphereBenchmark2DDDG, gmsh_kwargs)
    hc_ddg_bench = _run_case(4, "HC Unsymm+Obtuse DDG Operator Benchmark", StokesSphereBenchmark2DHCDDG, hc_kwargs)

    if global_target_obstacle is not None and global_target_outer is not None and args.auto_match_hc_cells:
        print("\nGmsh auto-match (symm_nonobtuse): tuning to global boundary target ...")
        tuned_gmsh_symm = _auto_tune_gmsh_to_target_boundaries_2d(
            args,
            StokesSphereBenchmark2DSymmNonobtuse,
            gmsh_symm_kwargs,
            target_obstacle=global_target_obstacle,
            target_outer=global_target_outer,
        )
        if tuned_gmsh_symm is not None:
            gmsh_symm_kwargs = tuned_gmsh_symm["kwargs"]
            print(
                "\nGmsh auto-match (symm_nonobtuse): "
                f"target_obstacle={tuned_gmsh_symm['target_obstacle']}, "
                f"chosen_obstacle={tuned_gmsh_symm['n_obstacle']}, "
                f"target_outer={tuned_gmsh_symm['target_outer']}, "
                f"chosen_outer={tuned_gmsh_symm['n_outer']}, "
                f"mesh_size_inner={tuned_gmsh_symm['mesh_size_inner']:.6g}, "
                f"mesh_size_outer={tuned_gmsh_symm['mesh_size_outer']:.6g}"
            )
        else:
            print("\nGmsh auto-match (symm_nonobtuse) skipped: no valid candidate mesh was produced.")

    gmsh_symm_bench = _run_case(
        5,
        "Gmsh Symm+Nonobtuse Finite-Difference Operator Benchmark",
        StokesSphereBenchmark2DSymmNonobtuse,
        gmsh_symm_kwargs,
    )

    if gmsh_symm_bench is not None and args.auto_match_hc_cells and global_target_cells is not None:
        tuned_symm = _auto_tune_hc_to_target_cells(
            args,
            global_target_cells,
            target_obstacle=global_target_obstacle,
            target_outer=global_target_outer,
            match_mode=args.auto_match_mode,
            symm_nonobtuse=True,
        )
        if tuned_symm is not None:
            hc_symm_kwargs["hc_refine"] = tuned_symm["hc_refine"]
            hc_symm_kwargs["hc_inner_samples"] = tuned_symm["hc_inner_samples"]
            hc_symm_kwargs["hc_outer_samples"] = tuned_symm["hc_outer_samples"]
            print(
                "\nHC auto-match (symm_nonobtuse): "
                f"mode={args.auto_match_mode}, "
                f"target_cells={tuned_symm['target_cells']}, "
                f"chosen_cells={tuned_symm['n_cells']}, "
                f"hc_refine={tuned_symm['hc_refine']}, "
                f"hc_inner_samples={tuned_symm['hc_inner_samples']}, "
                f"hc_outer_samples={tuned_symm['hc_outer_samples']}, "
                f"target_obstacle={_format_value(tuned_symm['target_obstacle'])}, "
                f"chosen_obstacle={tuned_symm['n_obstacle']}, "
                f"target_outer={_format_value(tuned_symm['target_outer'])}, "
                f"chosen_outer={tuned_symm['n_outer']}, "
                f"quality_median={_format_value(tuned_symm['quality_median'])}, "
                f"quality_min={_format_value(tuned_symm['quality_min'])}"
            )
        else:
            print("\nHC auto-match (symm_nonobtuse) skipped: no valid HC candidate mesh was produced.")

    hc_symm_bench = _run_case(
        6,
        "HC Symm+Nonobtuse Finite-Difference Operator Benchmark",
        StokesSphereBenchmark2DHCSymmNonobtuse,
        hc_symm_kwargs,
    )
    gmsh_symm_ddg_bench = _run_case(
        7,
        "Gmsh Symm+Nonobtuse DDG Operator Benchmark",
        StokesSphereBenchmark2DDDGSymmNonobtuse,
        gmsh_symm_kwargs,
    )
    hc_symm_ddg_bench = _run_case(
        8,
        "HC Symm+Nonobtuse DDG Operator Benchmark",
        StokesSphereBenchmark2DHCDDGSymmNonobtuse,
        hc_symm_kwargs,
    )

    if args.save_mesh_pngs:
        png_entries = []
        if gmsh_bench is not None:
            png_entries.append(("Case 1", _save_mesh_png(gmsh_bench, png_dir / "stokes_2d_gmsh_mesh.png", "StokesSphere2D (Gmsh)")))
        if hc_bench is not None:
            png_entries.append(("Case 2", _save_mesh_png(hc_bench, png_dir / "stokes_2d_hc_mesh.png", "StokesSphere2DHC (HC)")))
        if gmsh_symm_bench is not None:
            png_entries.append(("Case 5", _save_mesh_png(
                gmsh_symm_bench,
                png_dir / "stokes_2d_gmsh_symm_nonobtuse_mesh.png",
                "StokesSphere2D Symm+Nonobtuse (Gmsh)",
            )))
        if hc_symm_bench is not None:
            png_entries.append(("Case 6", _save_mesh_png(
                hc_symm_bench,
                png_dir / "stokes_2d_hc_symm_nonobtuse_mesh.png",
                "StokesSphere2D Symm+Nonobtuse (HC)",
            )))
        for label, png_path in png_entries:
            if png_path is not None:
                print(f"Mesh PNG ({label}): {png_path}")

    _print_rel_error_by_case([
        ("Gmsh Unsymm+Obtuse Finite-Difference Operator Benchmark", gmsh_bench),
        ("HC Unsymm+Obtuse Finite-Difference Operator Benchmark", hc_bench),
        ("Gmsh Unsymm+Obtuse DDG Operator Benchmark", gmsh_ddg_bench),
        ("HC Unsymm+Obtuse DDG Operator Benchmark", hc_ddg_bench),
        ("Gmsh Symm+Nonobtuse Finite-Difference Operator Benchmark", gmsh_symm_bench),
        ("HC Symm+Nonobtuse Finite-Difference Operator Benchmark", hc_symm_bench),
        ("Gmsh Symm+Nonobtuse DDG Operator Benchmark", gmsh_symm_ddg_bench),
        ("HC Symm+Nonobtuse DDG Operator Benchmark", hc_symm_ddg_bench),
    ])
    _print_vertex_rel_error_by_case([
        ("Gmsh Unsymm+Obtuse Finite-Difference Operator Benchmark", gmsh_bench),
        ("HC Unsymm+Obtuse Finite-Difference Operator Benchmark", hc_bench),
        ("Gmsh Unsymm+Obtuse DDG Operator Benchmark", gmsh_ddg_bench),
        ("HC Unsymm+Obtuse DDG Operator Benchmark", hc_ddg_bench),
        ("Gmsh Symm+Nonobtuse Finite-Difference Operator Benchmark", gmsh_symm_bench),
        ("HC Symm+Nonobtuse Finite-Difference Operator Benchmark", hc_symm_bench),
        ("Gmsh Symm+Nonobtuse DDG Operator Benchmark", gmsh_symm_ddg_bench),
        ("HC Symm+Nonobtuse DDG Operator Benchmark", hc_symm_ddg_bench),
    ])


if __name__ == "__main__":
    main()
