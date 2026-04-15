#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

from benchmarks._benchmark_cases import (
    StokesSphereBenchmark3D,
    StokesSphereBenchmark3DDDG,
    StokesSphereBenchmark3DHC,
    StokesSphereBenchmark3DHCSymmNonobtuse,
    StokesSphereBenchmark3DHCDDG,
    StokesSphereBenchmark3DHCDDGSymmNonobtuse,
    StokesSphereBenchmark3DSymmNonobtuse,
    StokesSphereBenchmark3DDDGSymmNonobtuse,
    _generate_stokes_sphere_hc_mesh,
)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run the 3D Stokes-sphere benchmark.",
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
        default="benchmarks_out/stokes_sphere_3d",
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="stokes_sphere_3d.msh",
    )
    parser.add_argument(
        "--remesh",
        action="store_true",
        help="Regenerate the Gmsh mesh even if it already exists.",
    )
    parser.add_argument("--hc-refine", type=int, default=2)
    parser.add_argument("--hc-inner-samples", type=int, default=None)
    parser.add_argument("--hc-outer-samples", type=int, default=None)
    parser.add_argument(
        "--auto-match-hc-cells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In 'both' mode, auto-tune HC mesh parameters to match Gmsh 3D volume-cell count.",
    )
    parser.add_argument(
        "--auto-match-mode",
        type=str,
        choices=("surface", "cells", "balanced"),
        default="surface",
        help=(
            "HC auto-match objective in 'both' mode: "
            "'surface' (match obstacle/outer surface density first), "
            "'cells' (match 3D volume cells first), "
            "'balanced' (trade off surface and volume matching)."
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
    """
    Print drag relative error for each available benchmark case.

    Parameters
    ----------
    case_entries : list[tuple[str, object]]
        Items of ``(label, benchmark_or_None)``.
    """
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


def _default_hc_samples_3d(refine_level):
    refine_level = int(max(refine_level, 0))
    n_inner = max(96, 64 * (2 ** max(refine_level - 1, 0)))
    n_outer = max(128, 96 * (2 ** max(refine_level - 1, 0)))
    return int(n_inner), int(n_outer)


def _default_gmsh_sizes_3d(args):
    h_inner = args.mesh_size_inner if args.mesh_size_inner is not None else 0.20 * float(args.sphere_radius)
    h_outer = args.mesh_size_outer if args.mesh_size_outer is not None else 0.75 * float(args.sphere_radius)
    return float(h_inner), float(h_outer)


def _probe_mesh_metadata(benchmark_cls, kwargs):
    probe = benchmark_cls(**kwargs)
    probe.generate_mesh()
    probe.load_mesh()
    return dict(getattr(probe, "mesh_metadata", {}) or {})


def _auto_tune_gmsh_to_target_surfaces_3d(
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
    inner0, outer0 = _default_gmsh_sizes_3d(args)
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


def _estimate_hc_volume_cells_3d(
    args, refine_level, inner_samples, outer_samples, symm_nonobtuse=False, equal_edge=False
):
    points, simplices, obstacle, outer_boundary, _ = _generate_stokes_sphere_hc_mesh(
        dim=3,
        sphere_radius=args.sphere_radius,
        outer_radius=args.outer_radius,
        hc_refine=int(refine_level),
        hc_inner_samples=int(inner_samples),
        hc_outer_samples=int(outer_samples),
        symm_nonobtuse=bool(symm_nonobtuse),
        equal_edge=bool(equal_edge),
    )
    return (
        int(simplices.shape[0]),
        int(points.shape[0]),
        int(len(obstacle)),
        int(len(outer_boundary)),
    )


def _auto_tune_hc_to_target_cells_3d(
    args,
    target_cells,
    target_obstacle=None,
    target_outer=None,
    match_mode="surface",
    symm_nonobtuse=False,
    equal_edge=False,
):
    base_ref = int(max(args.hc_refine, 1))
    if args.hc_inner_samples is None or args.hc_outer_samples is None:
        base_inner, base_outer = _default_hc_samples_3d(base_ref)
    else:
        base_inner, base_outer = int(args.hc_inner_samples), int(args.hc_outer_samples)

    # Fast path for default surface mode: directly map Gmsh boundary counts
    # to HC sampling, then choose a refinement level heuristically from the
    # target volume-cell scale. This avoids long candidate sweeps.
    if (
        (not equal_edge)
        and match_mode == "surface"
        and target_obstacle is not None
        and target_outer is not None
    ):
        inner = max(96, int(round(0.5 * (int(target_obstacle) + 4))))
        outer = max(128, int(round(0.5 * (int(target_outer) - 4))))
        if outer < inner:
            outer = inner

        if int(target_cells) >= 20000:
            refine = max(base_ref, 4)
        elif int(target_cells) >= 7000:
            refine = max(base_ref, 3)
        else:
            refine = max(base_ref, 2)

        n_cells, n_points, n_obstacle, n_outer = _estimate_hc_volume_cells_3d(
            args, refine, inner, outer, symm_nonobtuse=symm_nonobtuse, equal_edge=equal_edge
        )
        err_cells = abs(n_cells - int(target_cells))
        err_surface = abs(n_obstacle - int(target_obstacle)) + abs(n_outer - int(target_outer))
        return {
            "hc_refine": int(refine),
            "hc_inner_samples": int(inner),
            "hc_outer_samples": int(outer),
            "n_cells": int(n_cells),
            "n_points": int(n_points),
            "target_cells": int(target_cells),
            "abs_error": int(err_cells),
            "n_obstacle": int(n_obstacle),
            "n_outer": int(n_outer),
            "target_obstacle": int(target_obstacle),
            "target_outer": int(target_outer),
            "boundary_abs_error": int(err_surface),
        }

    refine_candidates = sorted(set([max(1, base_ref - 1), base_ref, base_ref + 1, base_ref + 2]))

    # Probe each refine level at its base sampling, then search multipliers
    # only on the most promising levels to keep runtime reasonable.
    refine_probes = []
    for refine in refine_candidates:
        if args.hc_inner_samples is None or args.hc_outer_samples is None:
            inner0, outer0 = _default_hc_samples_3d(refine)
        else:
            inner0, outer0 = base_inner, base_outer
        try:
            n_cells0, _, _, _ = _estimate_hc_volume_cells_3d(
                args,
                refine,
                inner0,
                outer0,
                symm_nonobtuse=symm_nonobtuse,
                equal_edge=equal_edge,
            )
        except Exception:
            continue
        refine_probes.append((abs(n_cells0 - target_cells), refine, inner0, outer0))

    if not refine_probes:
        return None

    refine_probes.sort()
    top_refines = refine_probes[:2]
    scale_candidates = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    best = None
    for _, refine, inner0, outer0 in top_refines:
        inner_candidates = {max(96, int(round(inner0 * scale))) for scale in scale_candidates}
        outer_candidates = {max(128, int(round(outer0 * scale))) for scale in scale_candidates}

        if (not equal_edge) and target_obstacle is not None and target_outer is not None:
            # In this HC generator, these sample-count heuristics map well to
            # boundary element counts:
            # obstacle ≈ 2*inner - 4, outer ≈ 2*outer + 4.
            inner_from_surface = max(96, int(round(0.5 * (int(target_obstacle) + 4))))
            outer_from_surface = max(128, int(round(0.5 * (int(target_outer) - 4))))
            for d in (-2, -1, 0, 1, 2):
                inner_candidates.add(max(96, inner_from_surface + d))
            for d in (-8, -4, 0, 4, 8):
                outer_candidates.add(max(128, outer_from_surface + d))

        for inner in sorted(inner_candidates):
            for outer in sorted(outer_candidates):
                if outer < inner:
                    continue
                try:
                    n_cells, n_points, n_obstacle, n_outer = _estimate_hc_volume_cells_3d(
                        args,
                        refine,
                        inner,
                        outer,
                        symm_nonobtuse=symm_nonobtuse,
                        equal_edge=equal_edge,
                    )
                except Exception:
                    continue

                err_cells = abs(n_cells - int(target_cells))
                if target_obstacle is None or target_outer is None:
                    err_surface = 0
                else:
                    err_surface = abs(n_obstacle - int(target_obstacle)) + abs(n_outer - int(target_outer))

                if match_mode == "cells":
                    objective = (err_cells, err_surface)
                elif match_mode == "balanced":
                    objective = (3 * err_surface + err_cells, err_surface, err_cells)
                else:
                    objective = (err_surface, err_cells)

                candidate = (
                    *objective,
                    n_points,
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

    n_points = int(best[-7])
    n_cells = int(best[-6])
    n_obstacle = int(best[-5])
    n_outer = int(best[-4])
    refine = int(best[-3])
    inner = int(best[-2])
    outer = int(best[-1])
    err_cells = abs(n_cells - int(target_cells))
    err_surface = None
    if target_obstacle is not None and target_outer is not None:
        err_surface = abs(n_obstacle - int(target_obstacle)) + abs(n_outer - int(target_outer))

    return {
        "hc_refine": int(refine),
        "hc_inner_samples": int(inner),
        "hc_outer_samples": int(outer),
        "n_cells": int(n_cells),
        "n_points": int(n_points),
        "target_cells": int(target_cells),
        "abs_error": int(err_cells),
        "n_obstacle": int(n_obstacle),
        "n_outer": int(n_outer),
        "target_obstacle": None if target_obstacle is None else int(target_obstacle),
        "target_outer": None if target_outer is None else int(target_outer),
        "boundary_abs_error": None if err_surface is None else int(err_surface),
    }


def _save_mesh_png_3d(bench, out_path, title):
    points = np.asarray(bench.points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 1050
    margin = 40
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Simple orthographic camera for a stable 3D snapshot.
    az = np.deg2rad(35.0)
    el = np.deg2rad(20.0)
    Rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ],
        dtype=float,
    )
    R = Rx @ Rz

    centered = np.ascontiguousarray(
        points[:, :3] - np.mean(points[:, :3], axis=0, keepdims=True),
        dtype=np.float64,
    )
    proj = np.einsum("ij,jk->ik", centered, R.T, optimize=True)
    x2 = proj[:, 0]
    y2 = proj[:, 1]
    z2 = proj[:, 2]

    x_min, y_min = float(np.min(x2)), float(np.min(y2))
    x_max, y_max = float(np.max(x2)), float(np.max(y2))
    dx = max(x_max - x_min, 1.0e-12)
    dy = max(y_max - y_min, 1.0e-12)
    scale = min((width - 2 * margin) / dx, (height - 2 * margin) / dy)
    x_pad = (width - scale * dx) * 0.5
    y_pad = (height - scale * dy) * 0.5

    def to_px(idx):
        px = x_pad + (x2[idx] - x_min) * scale
        py = height - (y_pad + (y2[idx] - y_min) * scale)
        return (float(px), float(py))

    def element_corner_ids(elem):
        n = len(elem)
        if n >= 3:
            return int(elem[0]), int(elem[1]), int(elem[2])
        return None

    lines = []

    def add_surface(elements, color, width_px):
        if elements is None:
            return
        arr = np.asarray(elements, dtype=np.int64)
        if arr.size == 0:
            return
        for elem in arr:
            tri = element_corner_ids(elem)
            if tri is None:
                continue
            i, j, k = tri
            for a, b in ((i, j), (j, k), (k, i)):
                depth = 0.5 * (z2[a] + z2[b])
                lines.append((depth, a, b, color, width_px))

    add_surface(bench.outer_boundary_elements, color=(42, 157, 143), width_px=1)
    add_surface(bench.obstacle_elements, color=(230, 57, 70), width_px=2)

    lines.sort(key=lambda it: it[0])
    for _, a, b, color, width_px in lines:
        draw.line([to_px(a), to_px(b)], fill=color, width=width_px)

    img.save(out_path)
    return str(out_path)


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
        mesh_name="stokes_sphere_3d_symm_nonobtuse.msh",
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
        gmsh_bench = _run_case(1, "Gmsh Unsymm+Obtuse Finite-Difference Operator Benchmark", StokesSphereBenchmark3D, gmsh_kwargs)
        gmsh_ddg_bench = _run_case(2, "Gmsh Unsymm+Obtuse DDG Operator Benchmark", StokesSphereBenchmark3DDDG, gmsh_kwargs)
        gmsh_symm_bench = _run_case(
            3,
            "Gmsh Symm+Nonobtuse Finite-Difference Operator Benchmark",
            StokesSphereBenchmark3DSymmNonobtuse,
            gmsh_symm_kwargs,
        )
        gmsh_symm_ddg_bench = _run_case(
            4,
            "Gmsh Symm+Nonobtuse DDG Operator Benchmark",
            StokesSphereBenchmark3DDDGSymmNonobtuse,
            gmsh_symm_kwargs,
        )
        if args.save_mesh_pngs:
            if gmsh_bench is not None:
                png = _save_mesh_png_3d(gmsh_bench, png_dir / "stokes_3d_gmsh_mesh.png", "StokesSphere3D (Gmsh)")
                if png is not None:
                    print(f"Mesh PNG (Case 1): {png}")
            if gmsh_symm_bench is not None:
                png = _save_mesh_png_3d(
                    gmsh_symm_bench,
                    png_dir / "stokes_3d_gmsh_symm_nonobtuse_mesh.png",
                    "StokesSphere3D Symm+Nonobtuse (Gmsh)",
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
        hc_bench = _run_case(1, "HC Unsymm+Obtuse Finite-Difference Operator Benchmark", StokesSphereBenchmark3DHC, hc_kwargs)
        hc_ddg_bench = _run_case(2, "HC Unsymm+Obtuse DDG Operator Benchmark", StokesSphereBenchmark3DHCDDG, hc_kwargs)
        hc_symm_bench = _run_case(
            3,
            "HC Symm+Nonobtuse Finite-Difference Operator Benchmark",
            StokesSphereBenchmark3DHCSymmNonobtuse,
            hc_symm_kwargs,
        )
        hc_symm_ddg_bench = _run_case(
            4,
            "HC Symm+Nonobtuse DDG Operator Benchmark",
            StokesSphereBenchmark3DHCDDGSymmNonobtuse,
            hc_symm_kwargs,
        )
        if args.save_mesh_pngs:
            if hc_bench is not None:
                png = _save_mesh_png_3d(hc_bench, png_dir / "stokes_3d_hc_mesh.png", "StokesSphere3DHC (HC)")
                if png is not None:
                    print(f"Mesh PNG (Case 1): {png}")
            if hc_symm_bench is not None:
                png = _save_mesh_png_3d(
                    hc_symm_bench,
                    png_dir / "stokes_3d_hc_symm_nonobtuse_mesh.png",
                    "StokesSphere3D Symm+Nonobtuse (HC)",
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

    gmsh_bench = _run_case(1, "Gmsh Unsymm+Obtuse Finite-Difference Operator Benchmark", StokesSphereBenchmark3D, gmsh_kwargs)
    global_target_cells = None
    global_target_obstacle = None
    global_target_outer = None
    if gmsh_bench is not None:
        global_target_cells = int(gmsh_bench.mesh_metadata.get("n_volume_cells", 0))
        global_target_obstacle = int(gmsh_bench.mesh_metadata.get("n_obstacle_elements", 0))
        global_target_outer = int(gmsh_bench.mesh_metadata.get("n_outer_boundary_elements", 0))

    if gmsh_bench is not None and args.auto_match_hc_cells:
        print(f"\nHC auto-match: tuning (mode={args.auto_match_mode}) ...")
        tuned = _auto_tune_hc_to_target_cells_3d(
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
                f"chosen_points={tuned['n_points']}, "
                f"hc_refine={tuned['hc_refine']}, "
                f"hc_inner_samples={tuned['hc_inner_samples']}, "
                f"hc_outer_samples={tuned['hc_outer_samples']}, "
                f"target_obstacle={tuned['target_obstacle']}, "
                f"chosen_obstacle={tuned['n_obstacle']}, "
                f"target_outer={tuned['target_outer']}, "
                f"chosen_outer={tuned['n_outer']}"
            )
        else:
            print("\nHC auto-match skipped: no valid HC candidate mesh was produced.")

    hc_bench = _run_case(2, "HC Unsymm+Obtuse Finite-Difference Operator Benchmark", StokesSphereBenchmark3DHC, hc_kwargs)
    gmsh_ddg_bench = _run_case(3, "Gmsh Unsymm+Obtuse DDG Operator Benchmark", StokesSphereBenchmark3DDDG, gmsh_kwargs)
    hc_ddg_bench = _run_case(4, "HC Unsymm+Obtuse DDG Operator Benchmark", StokesSphereBenchmark3DHCDDG, hc_kwargs)

    gmsh_symm_bench = None

    if global_target_obstacle is not None and global_target_outer is not None and args.auto_match_hc_cells:
        print("\nGmsh auto-match (symm_nonobtuse): tuning to global boundary target ...")
        tuned_gmsh_symm = _auto_tune_gmsh_to_target_surfaces_3d(
            args,
            StokesSphereBenchmark3DSymmNonobtuse,
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
        StokesSphereBenchmark3DSymmNonobtuse,
        gmsh_symm_kwargs,
    )

    if gmsh_symm_bench is not None and args.auto_match_hc_cells and global_target_cells is not None:
        print(f"\nHC auto-match (symm_nonobtuse): tuning to global target (mode={args.auto_match_mode}) ...")
        tuned_symm = _auto_tune_hc_to_target_cells_3d(
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
                f"chosen_points={tuned_symm['n_points']}, "
                f"hc_refine={tuned_symm['hc_refine']}, "
                f"hc_inner_samples={tuned_symm['hc_inner_samples']}, "
                f"hc_outer_samples={tuned_symm['hc_outer_samples']}, "
                f"target_obstacle={tuned_symm['target_obstacle']}, "
                f"chosen_obstacle={tuned_symm['n_obstacle']}, "
                f"target_outer={tuned_symm['target_outer']}, "
                f"chosen_outer={tuned_symm['n_outer']}"
            )
        else:
            print("\nHC auto-match (symm_nonobtuse) skipped: no valid HC candidate mesh was produced.")

    hc_symm_bench = _run_case(
        6,
        "HC Symm+Nonobtuse Finite-Difference Operator Benchmark",
        StokesSphereBenchmark3DHCSymmNonobtuse,
        hc_symm_kwargs,
    )
    gmsh_symm_ddg_bench = _run_case(
        7,
        "Gmsh Symm+Nonobtuse DDG Operator Benchmark",
        StokesSphereBenchmark3DDDGSymmNonobtuse,
        gmsh_symm_kwargs,
    )
    hc_symm_ddg_bench = _run_case(
        8,
        "HC Symm+Nonobtuse DDG Operator Benchmark",
        StokesSphereBenchmark3DHCDDGSymmNonobtuse,
        hc_symm_kwargs,
    )

    if args.save_mesh_pngs:
        png_entries = []
        if gmsh_bench is not None:
            png_entries.append(("Case 1", _save_mesh_png_3d(gmsh_bench, png_dir / "stokes_3d_gmsh_mesh.png", "StokesSphere3D (Gmsh)")))
        if hc_bench is not None:
            png_entries.append(("Case 2", _save_mesh_png_3d(hc_bench, png_dir / "stokes_3d_hc_mesh.png", "StokesSphere3DHC (HC)")))
        if gmsh_symm_bench is not None:
            png_entries.append(("Case 5", _save_mesh_png_3d(
                gmsh_symm_bench,
                png_dir / "stokes_3d_gmsh_symm_nonobtuse_mesh.png",
                "StokesSphere3D Symm+Nonobtuse (Gmsh)",
            )))
        if hc_symm_bench is not None:
            png_entries.append(("Case 6", _save_mesh_png_3d(
                hc_symm_bench,
                png_dir / "stokes_3d_hc_symm_nonobtuse_mesh.png",
                "StokesSphere3D Symm+Nonobtuse (HC)",
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
