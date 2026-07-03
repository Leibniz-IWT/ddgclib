#!/usr/bin/env python3
"""Recompute dynamic Figure 3-8 cases from local simulation artifacts.

The PL and present-method curves are read from bundled copies of the simulation
summaries that created the manuscript figures. The comparison curves are
recomputed from bundled surface meshes, so the result table is generated from
traceable numerical artifacts rather than from a copied plotting CSV.

The top-level author-year files document and implement the equations used for
the CSV columns written by this helper.

This is a helper used by cube2sphere_all_methods.py and
droplet_oscillation_all_methods.py. If it is run directly without arguments from
some unrelated folder, it delegates to ../recompute_all_cases.py so a user does
not get a confusing "unknown case" error.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

COMMON = Path(__file__).resolve().parent
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
BUNDLE_ROOT = COMMON.parent
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from bootstrap import ensure_runtime  # noqa: E402
from method_timing import new_timings, perf_now, print_timing_summary  # noqa: E402

ensure_runtime()

import meshio
import numpy as np

import evrard_2023
import strobl_2016
import xie_xiao_2017


METHODS = {
    "plic_pl": ("PLIC_PL", "PLIC / PL baseline"),
    "evrard_type_paraboloid": ("Evrard_type_paraboloid", "Evrard-type paraboloid Vpatch"),
    "thinc_qq": ("THINC_QQ", "THINC/QQ quadratic Gaussian-quadrature Vpatch"),
    "strobl_sphere_overlap": ("Strobl_sphere_overlap", "Strobl-type sphere/hex overlap Vpatch"),
    "present_quadric_patch": ("Present_quadric_patch", "Present class-aware quadric-patch Vpatch"),
}

STROBL_HEXGRID_N = 4


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def case_config(case: str) -> dict[str, object]:
    sources = bundle_root() / "source_data"
    if case == "cube2sphere":
        fig_dir = sources / "figure_3_3"
        return {
            "mesh_dir": fig_dir / "cube_present_surface_states",
            "pl_summary": fig_dir / "_summary_PL.csv",
            "present_summary": fig_dir / "_summary_CurvedVolume.csv",
            "reference_volume": 1.0e-9,
            "default_max_iter": 1000,
        }
    if case == "droplet_oscillation":
        fig_dir = sources / "figure_3_7"
        return {
            "mesh_dir": fig_dir / "droplet_present_surface_states",
            "pl_summary": fig_dir / "_summary_PL.csv",
            "present_summary": fig_dir / "_summary_CurvedVolume.csv",
            "reference_volume": 4.0 * math.pi * (1.0e-3) ** 3 / 3.0,
            "default_max_iter": 2000,
        }
    raise ValueError(f"Unknown dynamic case: {case}")


def to_float(value: str | None, *, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_summary(path: Path, max_iter: int, stride: int) -> dict[int, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            iteration = int(to_float(row.get("iter"), default=-1))
            if iteration < 0 or iteration >= max_iter or iteration % stride:
                continue
            out[iteration] = {
                "V_total": to_float(row.get("V_total")),
                "Vi_true_plus_total": to_float(row.get("Vi_true_plus_total")),
                "rel_error_percent": to_float(row.get("rel.error.V%")),
                "REF_VOLUME": to_float(row.get("REF_VOLUME")),
            }
    if not out:
        raise ValueError(f"No usable rows found in {path}")
    return out


def load_surface_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        mesh = meshio.read(path)
    triangles = [block.data for block in mesh.cells if block.type == "triangle"]
    if not triangles:
        raise ValueError(f"No triangle cells found in {path}")
    return np.asarray(mesh.points[:, :3], dtype=float), np.vstack(triangles).astype(int)


def err_percent(value: float, reference_volume: float) -> float:
    return abs((value - reference_volume) / reference_volume * 100.0)


def recompute_comparison_job(job: dict[str, object]) -> dict[str, float | int]:
    iteration = int(job["iter"])
    reference_volume = float(job["reference_volume"])
    msh_path = Path(job["mesh_path"])
    points, faces = load_surface_mesh(msh_path)
    is_curved_face = lambda _points, _face: True

    t0 = perf_now()
    v_evrard = evrard_2023.surface_ppic_volume(points, faces, is_curved_face)
    t_evrard = perf_now() - t0
    t0 = perf_now()
    v_thinc = xie_xiao_2017.local_quadratic_gq_volume(points, faces, is_curved_face)
    t_thinc = perf_now() - t0
    t0 = perf_now()
    v_strobl = strobl_2016.sphere_hexgrid_volume(points)
    t_strobl = perf_now() - t0
    return {
        "iter": iteration,
        "V_evrard_type_paraboloid": v_evrard,
        "V_thinc_qq": v_thinc,
        "V_strobl_sphere_overlap": v_strobl,
        "evrard_type_paraboloid_error_percent": err_percent(v_evrard, reference_volume),
        "thinc_qq_error_percent": err_percent(v_thinc, reference_volume),
        "strobl_sphere_overlap_error_percent": err_percent(v_strobl, reference_volume),
        "time_evrard_type_paraboloid": t_evrard,
        "time_thinc_qq": t_thinc,
        "time_strobl_sphere_overlap": t_strobl,
    }


def recompute_case(case_dir: Path, max_iter: int | None, stride: int, workers: int) -> Path:
    case = case_dir.name
    case_start = perf_now()
    timings = new_timings()
    config = case_config(case)
    mesh_dir = Path(config["mesh_dir"])
    if max_iter is None:
        max_iter = int(config["default_max_iter"])

    t0 = perf_now()
    pl_summary = load_summary(Path(config["pl_summary"]), max_iter=max_iter, stride=stride)
    timings["plic_pl"] += perf_now() - t0
    t0 = perf_now()
    present_summary = load_summary(Path(config["present_summary"]), max_iter=max_iter, stride=stride)
    timings["present_quadric_patch"] += perf_now() - t0
    iterations = sorted(set(pl_summary) & set(present_summary))
    if not iterations:
        raise ValueError(f"No matching PL/present summary iterations for {case}")

    out_csv = case_dir / f"{case}_all_methods_result.csv"
    out_log = case_dir / f"{case}_all_methods_result.txt"
    rows_by_iter: dict[int, dict[str, float | int]] = {}
    jobs = []
    for iteration in iterations:
        msh_path = mesh_dir / f"surface_iter_{iteration:04d}.msh"
        if not msh_path.exists():
            raise FileNotFoundError(msh_path)

        reference_volume = present_summary[iteration]["REF_VOLUME"]
        if not math.isfinite(reference_volume) or reference_volume <= 0.0:
            reference_volume = float(config["reference_volume"])

        v_pl = pl_summary[iteration]["V_total"]
        v_present = present_summary[iteration]["Vi_true_plus_total"]
        if not math.isfinite(v_present):
            v_present = present_summary[iteration]["V_total"]
        rows_by_iter[iteration] = {
            "iter": iteration,
            "V_plic_pl": v_pl,
            "V_present_quadric_patch": v_present,
            "plic_pl_error_percent": abs(pl_summary[iteration]["rel_error_percent"]),
            "present_quadric_patch_error_percent": abs(present_summary[iteration]["rel_error_percent"]),
        }
        jobs.append(
            {
                "iter": iteration,
                "mesh_path": str(msh_path),
                "reference_volume": reference_volume,
            }
        )

    workers = max(1, min(workers, os.cpu_count() or 1, len(jobs)))
    if workers == 1:
        for idx, job in enumerate(jobs, start=1):
            result = recompute_comparison_job(job)
            timings["evrard_type_paraboloid"] += float(result.pop("time_evrard_type_paraboloid"))
            timings["thinc_qq"] += float(result.pop("time_thinc_qq"))
            timings["strobl_sphere_overlap"] += float(result.pop("time_strobl_sphere_overlap"))
            rows_by_iter[int(result["iter"])].update(result)
            print(f"{case}: computed {idx}/{len(jobs)} iter={result['iter']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_iter = {pool.submit(recompute_comparison_job, job): int(job["iter"]) for job in jobs}
            for idx, future in enumerate(as_completed(future_to_iter), start=1):
                result = future.result()
                timings["evrard_type_paraboloid"] += float(result.pop("time_evrard_type_paraboloid"))
                timings["thinc_qq"] += float(result.pop("time_thinc_qq"))
                timings["strobl_sphere_overlap"] += float(result.pop("time_strobl_sphere_overlap"))
                rows_by_iter[int(result["iter"])].update(result)
                if idx == 1 or idx % 50 == 0 or idx == len(jobs):
                    print(f"{case}: computed {idx}/{len(jobs)}", flush=True)

    rows = [rows_by_iter[iteration] for iteration in iterations]

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    timing_note = (
        "Timing note: dynamic method times are aggregate worker times; case total is wall-clock. "
        "PL and present dynamic values are loaded from saved full-solver summaries."
    )
    out_log.write_text(
        "\n".join(
            [
                f"Case: {case}",
                f"Mesh directory: {mesh_dir}",
                f"PL summary: {config['pl_summary']}",
                f"Present-method summary: {config['present_summary']}",
                f"Workers: {workers}",
                f"Rows: {len(rows)}",
                f"CSV: {out_csv.name}",
                timing_note,
                *[f"{method}_seconds: {timings[method]:.6f}" for method in METHODS],
                f"case_total_seconds: {perf_now() - case_start:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(timing_note, flush=True)
    print_timing_summary(case, timings, perf_now() - case_start)
    return out_csv


def write_method_result(method_dir: Path, method: str, all_csv: Path) -> Path:
    method_name, method_description = METHODS[method]
    out_csv = method_dir / f"{method_name}_result.csv"
    out_txt = method_dir / f"{method_name}_result.txt"
    err_col = f"{method}_error_percent"
    vol_col = f"V_{method}"

    rows = []
    with all_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "iter": row["iter"],
                    "volume": row[vol_col],
                    "error_percent": row[err_col],
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "volume", "error_percent"])
        writer.writeheader()
        writer.writerows(rows)

    out_txt.write_text(
        "\n".join(
            [
                f"Method: {method_description}",
                f"Rows: {len(rows)}",
                f"CSV: {out_csv.name}",
                f"Final error percent: {float(rows[-1]['error_percent']):.16e}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--method", choices=sorted(METHODS))
    parser.add_argument("--method-dir", type=Path)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("FIGURE_C1_WORKERS", "8")))
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="For method folders, reuse an existing parent *_all_methods_result.csv when present.",
    )
    args = parser.parse_args()

    if args.case_dir is None and args.method is None:
        default_case_dir = Path.cwd().resolve()
        if default_case_dir.name not in {"cube2sphere", "droplet_oscillation"}:
            full_runner = bundle_root() / "recompute_all_cases.py"
            print(
                "No dynamic case folder was provided. Running the full Figure 3-8 "
                f"recompute script instead: {full_runner}",
                flush=True,
            )
            subprocess.run([sys.executable, str(full_runner)], cwd=str(bundle_root()), check=True)
            return
        case_dir = default_case_dir
    else:
        case_dir = (args.case_dir or Path.cwd()).resolve()

    existing_csv = case_dir / f"{case_dir.name}_all_methods_result.csv"
    if args.reuse_existing and existing_csv.exists():
        all_csv = existing_csv
    else:
        all_csv = recompute_case(
            case_dir,
            max_iter=args.max_iter,
            stride=args.stride,
            workers=args.workers,
        )
    if args.method:
        method_dir = args.method_dir.resolve() if args.method_dir else Path.cwd()
        out = write_method_result(method_dir, args.method, all_csv)
        print(out)
    else:
        for method_id, (folder, _description) in METHODS.items():
            method_dir = args.case_dir.resolve() / folder
            method_dir.mkdir(exist_ok=True)
            print(write_method_result(method_dir, method_id, all_csv))


if __name__ == "__main__":
    main()
