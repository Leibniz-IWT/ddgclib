#!/usr/bin/env python3
"""Recompute all Figure 3-8 cases from source scripts/mesh states, then plot.

The method equations, literature references, and callable functions are kept in
the top-level author-year files: evrard_2023.py, xie_xiao_2017.py,
strobl_2016.py, plic_pl_1998_1999.py, and present_quadric_patch_2026.py.

This script intentionally deletes previous generated result files first. If it
is named recompute, it should recompute from the bundled source scripts, meshes,
and simulation states instead of silently reusing old plotting CSVs.
"""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime, runtime_env  # noqa: E402
from method_timing import format_duration, perf_now  # noqa: E402

ensure_runtime()

STATIC_CASES = [
    "sphere",
    "cylinder",
    "paraboloid",
    "hyperboloid",
    "parabolic_cylinder",
    "hyperbolic_cylinder",
]
DYNAMIC_CASES = ["cube2sphere", "droplet_oscillation"]
EXPECTED_ROWS = {
    "sphere": 4,
    "cylinder": 4,
    "paraboloid": 4,
    "hyperboloid": 4,
    "parabolic_cylinder": 4,
    "hyperbolic_cylinder": 4,
    "cube2sphere": 1000,
    "droplet_oscillation": 2000,
}
EXPECTED_METHODS = 5
EXPECTED_MESH_PNGS = (6 * 4 * EXPECTED_METHODS) + (2 * EXPECTED_METHODS)
EXPECTED_PANELS = [f"Figure_3-8{letter}.png" for letter in "abcdefgh"]
GENERATED_PATTERNS = [
    "Figure_3-8*.png",
    "Figure_3-8.pdf",
    "Figure_3-8_VolumeOperatorComparison_result.txt",
    "Figure_C-1*.png",
    "Figure_C-1.pdf",
    "Figure_C-1_Rebuttal_result.txt",
    "mesh/*.png",
    "mesh/manifest.csv",
    "mesh/README.txt",
    "*/*_all_methods_result.csv",
    "*/*_all_methods_result.txt",
    "*/*/*_result.csv",
    "*/*/*_result.txt",
]
TEMP_DIR_NAMES = {"_work", "__pycache__"}


def case_csv_path(case: str) -> Path:
    return HERE / case / f"{case}_all_methods_result.csv"


def run(cmd: list[str], cwd: Path) -> float:
    start = perf_now()
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=runtime_env())
    elapsed = perf_now() - start
    print(f"Finished {' '.join(cmd)} in {format_duration(elapsed)}", flush=True)
    return elapsed


def clean_generated_outputs() -> None:
    removed = 0
    for pattern in GENERATED_PATTERNS:
        for path in HERE.glob(pattern):
            if not path.is_file():
                continue
            path.unlink()
            removed += 1
    print(f"Removed {removed} old generated CSV/TXT/PNG/PDF files.", flush=True)


def clean_temporary_dirs() -> None:
    removed = 0
    for path in sorted(HERE.rglob("*"), reverse=True):
        if path.is_dir() and path.name in TEMP_DIR_NAMES:
            shutil.rmtree(path)
            removed += 1
    print(f"Removed {removed} temporary _work/__pycache__ folders.", flush=True)


def missing_result_csvs(cases: list[str]) -> list[Path]:
    return [case_csv_path(case) for case in cases if not case_csv_path(case).exists()]


def ensure_required_result_csvs() -> None:
    """Make sure plotting inputs exist before Figure_3-8_VolumeOperatorComparison.py runs."""
    missing_static = missing_result_csvs(STATIC_CASES)
    if missing_static:
        print("Static result CSVs are missing before plotting:", flush=True)
        for path in missing_static:
            print(f"  missing {path.relative_to(HERE)}", flush=True)
        print("Rerunning static cases to rebuild the missing plotting inputs.", flush=True)
        run([sys.executable, "recompute_all_static_cases.py"], HERE)

    missing = missing_result_csvs(STATIC_CASES + DYNAMIC_CASES)
    if missing:
        details = "\n".join(f"- {path.relative_to(HERE)}" for path in missing)
        raise RuntimeError(
            "Cannot plot Figure 3-8 because these required result CSVs are missing:\n"
            f"{details}"
        )


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def pdf_page_count(path: Path) -> int | None:
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    return len(PdfReader(str(path)).pages)


def validate_outputs() -> None:
    print("Validating regenerated Figure 3-8 outputs...", flush=True)
    errors: list[str] = []
    for case, expected in EXPECTED_ROWS.items():
        csv_path = HERE / case / f"{case}_all_methods_result.csv"
        if not csv_path.exists():
            errors.append(f"missing {csv_path.relative_to(HERE)}")
            continue
        rows = csv_row_count(csv_path)
        print(f"  {csv_path.relative_to(HERE)}: {rows} rows", flush=True)
        if rows != expected:
            errors.append(f"{csv_path.relative_to(HERE)} has {rows} rows; expected {expected}")

    for panel in EXPECTED_PANELS:
        panel_path = HERE / panel
        if not panel_path.exists() or panel_path.stat().st_size == 0:
            errors.append(f"missing or empty {panel}")
        else:
            print(f"  {panel}: {panel_path.stat().st_size} bytes", flush=True)

    pdf_path = HERE / "Figure_3-8.pdf"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        errors.append("missing or empty Figure_3-8.pdf")
    else:
        pages = pdf_page_count(pdf_path)
        if pages is None:
            print("  Figure_3-8.pdf: exists; page count unavailable in this Python", flush=True)
        else:
            print(f"  Figure_3-8.pdf: {pages} pages", flush=True)
            if pages != 8:
                errors.append(f"Figure_3-8.pdf has {pages} pages; expected 8")

    if errors:
        raise RuntimeError("Output validation failed:\n- " + "\n- ".join(errors))
    print("Output validation passed.", flush=True)


def validate_mesh_outputs() -> None:
    print("Validating regenerated mesh previews...", flush=True)
    errors: list[str] = []
    mesh_dir = HERE / "mesh"
    pngs = sorted(mesh_dir.glob("*.png"))
    manifest = mesh_dir / "manifest.csv"
    readme = mesh_dir / "README.txt"
    if len(pngs) != EXPECTED_MESH_PNGS:
        errors.append(f"mesh has {len(pngs)} PNGs; expected {EXPECTED_MESH_PNGS}")
    else:
        print(f"  mesh PNGs: {len(pngs)}", flush=True)
    if not manifest.exists():
        errors.append("missing mesh/manifest.csv")
    else:
        rows = csv_row_count(manifest)
        print(f"  mesh/manifest.csv: {rows} rows", flush=True)
        if rows != EXPECTED_MESH_PNGS:
            errors.append(f"mesh/manifest.csv has {rows} rows; expected {EXPECTED_MESH_PNGS}")
    if not readme.exists():
        errors.append("missing mesh/README.txt")
    if errors:
        raise RuntimeError("Mesh-preview validation failed:\n- " + "\n- ".join(errors))
    print("Mesh-preview validation passed.", flush=True)


def main() -> None:
    total_start = perf_now()
    print("Recomputing Figure 3-8 from bundled source scripts, local meshes, and local simulation states.", flush=True)
    clean_generated_outputs()
    run([sys.executable, "recompute_all_static_cases.py"], HERE)
    for case in DYNAMIC_CASES:
        run([sys.executable, f"{case}_all_methods.py"], HERE / case)
    ensure_required_result_csvs()
    run([sys.executable, "Figure_3-8_VolumeOperatorComparison.py"], HERE)
    run([sys.executable, "mesh_plot.py"], HERE)
    validate_outputs()
    validate_mesh_outputs()
    package_dir = HERE.parents[1] / "01_Manuscript" / "Rebuttal_package"
    if package_dir.exists():
        shutil.copy2(HERE / "Figure_3-8.pdf", package_dir / "Figure_3-8.pdf")
        print(f"Copied Figure_3-8.pdf to {package_dir}", flush=True)
    clean_temporary_dirs()
    print(f"Full Figure 3-8 recomputation finished in {format_duration(perf_now() - total_start)}", flush=True)


if __name__ == "__main__":
    main()
