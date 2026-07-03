#!/usr/bin/env python3
"""Shared recomputation helper for Figure 3-8 method folders.

The top-level author-year files document and implement the equations used for
the generated columns.  This helper converts the case-script output into CSV
tables and per-method CSV files.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from bootstrap import runtime_env


METHODS = {
    "plic_pl": ("PLIC_PL", "PLIC / PL baseline"),
    "evrard_type_paraboloid": ("Evrard_type_paraboloid", "Evrard-type paraboloid Vpatch"),
    "thinc_qq": ("THINC_QQ", "THINC/QQ quadratic Gaussian-quadrature Vpatch"),
    "strobl_sphere_overlap": ("Strobl_sphere_overlap", "Strobl-type sphere/hex overlap Vpatch"),
    "present_quadric_patch": ("Present_quadric_patch", "Present class-aware quadric-patch Vpatch"),
}

CASE_LETTERS = {
    "sphere": "a",
    "cylinder": "b",
    "paraboloid": "c",
    "hyperboloid": "d",
    "parabolic_cylinder": "e",
    "hyperbolic_cylinder": "f",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_rows(case: str, text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not re.match(r"\s*\d+\s+", line):
            continue
        values = [
            float(value)
            for value in re.findall(
                r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
                line,
            )
        ]
        if case == "sphere":
            if len(values) < 8:
                continue
            level, faces, hmean, pl, evrard, thinc, strobl, present = values[:8]
            row = {
                "level": int(level),
                "faces": int(faces),
                "h_mean": hmean,
                "plic_pl": pl,
                "evrard_type_paraboloid": evrard,
                "thinc_qq": thinc,
                "strobl_sphere_overlap": strobl,
                "present_quadric_patch": present,
            }
        else:
            if len(values) < 9:
                continue
            level, curved_faces, total_faces, hmean, pl, evrard, thinc, strobl, present = values[:9]
            row = {
                "level": int(level),
                "curved_faces": int(curved_faces),
                "total_faces": int(total_faces),
                "h_mean": hmean,
                "plic_pl": pl,
                "evrard_type_paraboloid": evrard,
                "thinc_qq": thinc,
                "strobl_sphere_overlap": strobl,
                "present_quadric_patch": present,
            }
        row["label"] = "Table 3-1 mesh" if "Table 3-1" in line else ""
        rows.append(row)
    if not rows:
        raise ValueError(f"No numerical rows parsed for {case}")
    return rows


def run_case(case_dir: Path) -> tuple[str, list[dict[str, str]]]:
    case = case_dir.name
    script = case_dir / f"{case}_all_methods.py"
    if not script.exists():
        raise FileNotFoundError(script)

    env = runtime_env(os.environ.copy())
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(case_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    returncode = proc.wait()
    output = "".join(chunks)
    result_path = case_dir / f"{case}_all_methods_result.txt"
    result_path.write_text(output, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"{script} failed; see {result_path}")

    rows = parse_rows(case, output)
    all_csv = case_dir / f"{case}_all_methods_result.csv"
    fieldnames = ["level"]
    if any("faces" in row for row in rows):
        fieldnames.append("faces")
    if any("curved_faces" in row for row in rows):
        fieldnames.extend(["curved_faces", "total_faces"])
    fieldnames.extend(
        [
            "h_mean",
            "plic_pl_error_percent",
            "evrard_type_paraboloid_error_percent",
            "thinc_qq_error_percent",
            "strobl_sphere_overlap_error_percent",
            "present_quadric_patch_error_percent",
            "label",
        ]
    )
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "level": row["level"],
                "h_mean": f"{float(row['h_mean']):.16e}",
                "plic_pl_error_percent": f"{float(row['plic_pl']):.16e}",
                "evrard_type_paraboloid_error_percent": f"{float(row['evrard_type_paraboloid']):.16e}",
                "thinc_qq_error_percent": f"{float(row['thinc_qq']):.16e}",
                "strobl_sphere_overlap_error_percent": f"{float(row['strobl_sphere_overlap']):.16e}",
                "present_quadric_patch_error_percent": f"{float(row['present_quadric_patch']):.16e}",
                "label": row.get("label", ""),
            }
            if "faces" in row:
                out["faces"] = row["faces"]
            if "curved_faces" in row:
                out["curved_faces"] = row["curved_faces"]
                out["total_faces"] = row["total_faces"]
            writer.writerow(out)
    return output, rows


def read_existing_rows(case_dir: Path) -> list[dict[str, str]]:
    case = case_dir.name
    all_csv = case_dir / f"{case}_all_methods_result.csv"
    if not all_csv.exists():
        raise FileNotFoundError(all_csv)

    rows: list[dict[str, str]] = []
    with all_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for source in reader:
            row = {
                "level": int(float(source["level"])),
                "h_mean": float(source["h_mean"]),
                "plic_pl": float(source["plic_pl_error_percent"]),
                "evrard_type_paraboloid": float(source["evrard_type_paraboloid_error_percent"]),
                "thinc_qq": float(source["thinc_qq_error_percent"]),
                "strobl_sphere_overlap": float(source["strobl_sphere_overlap_error_percent"]),
                "present_quadric_patch": float(source["present_quadric_patch_error_percent"]),
                "label": source.get("label", ""),
            }
            if source.get("faces"):
                row["faces"] = int(float(source["faces"]))
            if source.get("curved_faces"):
                row["curved_faces"] = int(float(source["curved_faces"]))
                row["total_faces"] = int(float(source["total_faces"]))
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {all_csv}")
    return rows


def write_method_result(method_dir: Path, method: str, rows: list[dict[str, str]]) -> Path:
    if method not in METHODS:
        raise KeyError(f"Unknown method: {method}")
    method_name, method_description = METHODS[method]
    out_csv = method_dir / f"{method_name}_result.csv"
    fieldnames = ["level", "h_mean", "error_percent", "label"]
    if any("curved_faces" in row for row in rows):
        fieldnames[1:1] = ["curved_faces", "total_faces"]
    elif any("faces" in row for row in rows):
        fieldnames[1:1] = ["faces"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "level": row["level"],
                "h_mean": f"{float(row['h_mean']):.16e}",
                "error_percent": f"{float(row[method]):.16e}",
                "label": row.get("label", ""),
            }
            if "curved_faces" in row:
                out["curved_faces"] = row["curved_faces"]
                out["total_faces"] = row["total_faces"]
            if "faces" in row:
                out["faces"] = row["faces"]
            writer.writerow(out)

    out_txt = method_dir / f"{method_name}_result.txt"
    out_txt.write_text(
        "\n".join(
            [
                f"Method: {method_description}",
                f"Rows: {len(rows)}",
                f"CSV: {out_csv.name}",
                f"Finest-grid error percent: {float(rows[-1][method]):.16e}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out_csv


def recompute(method_dir: Path, method: str, reuse_existing: bool = False) -> Path:
    case_dir = method_dir.parent
    if reuse_existing and (case_dir / f"{case_dir.name}_all_methods_result.csv").exists():
        rows = read_existing_rows(case_dir)
    else:
        _output, rows = run_case(case_dir)
    return write_method_result(method_dir, method, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--method-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an existing parent *_all_methods_result.csv when present.",
    )
    args = parser.parse_args()

    out = recompute(args.method_dir.resolve(), args.method, reuse_existing=args.reuse_existing)
    print(out)


if __name__ == "__main__":
    main()
