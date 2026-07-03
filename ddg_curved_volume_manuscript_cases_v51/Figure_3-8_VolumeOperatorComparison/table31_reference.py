#!/usr/bin/env python3
"""Table 3-1 anchor values for Figure 3-8 static benchmark rows.

Figure 3-8 uses refinement sweeps, but the first static row is the exact mesh
reported in Table 3-1.  These helpers recompute that row from the same PL mesh
files and the same per-triangle correction CSVs used for Table 3-1, so the
Section 3.4 plot cannot drift from the manuscript table through a second code
path.
"""

from __future__ import annotations

import csv
import importlib
import math
from pathlib import Path
import sys

BUNDLE_ROOT = Path(__file__).resolve().parent
COMMON = BUNDLE_ROOT / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime  # noqa: E402

ensure_runtime()

import numpy as np  # noqa: E402


TABLE_DATA = BUNDLE_ROOT / "source_data" / "table_3_1"
MESH_DIR = BUNDLE_ROOT / "msh"

CASE_DATA = {
    "cylinder": {
        "mesh_module": "cylinder_mesh",
        "volume_csv": "cylinder_COEFFS_Transformed_Volume.csv",
        "exact_volume": math.pi,
    },
    "paraboloid": {
        "mesh_module": "paraboloid_mesh",
        "volume_csv": "snapped_paraboloid_ascii_COEFFS_Transformed_Volume.csv",
        "exact_volume": 2.0 * math.pi,
    },
    "hyperboloid": {
        "mesh_module": "hyperboloid_mesh",
        "volume_csv": "coarse_hyperboloid_COEFFS_Transformed_Volume.csv",
        "exact_volume": math.pi * (2.0 + 8.0 / 12.0),
    },
    "parabolic_cylinder": {
        "mesh_module": "parabolic_cylinder_mesh",
        "table_level": 1,
        "volume_csv": "snapped_y_eq_x2_ascii_COEFFS_Transformed_Volume.csv",
        # Table 3-1 used this rounded reference constant in the notebook.
        "exact_volume": 5.65685425 * (4.0 / 3.0),
    },
    "hyperbolic_cylinder": {
        "mesh_module": "hyperbolic_cylinder_mesh",
        "volume_csv": "snapped_x2_minus_y2_ascii_COEFFS_Transformed_Volume.csv",
        "exact_volume": 2.0 * (
            2.0 * 3.0 * 2.0
            - (2.0 * math.sqrt(1.0 + 2.0**2) + math.asinh(2.0))
        ),
    },
}


def _closed_surface_volume(points: np.ndarray, faces: np.ndarray) -> float:
    volume = 0.0
    for face in faces:
        a, b, c = points[face]
        volume += float(np.dot(a, np.cross(b, c))) / 6.0
    return abs(volume)


def _sum_vcorrection(csv_path: Path) -> float:
    total = 0.0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "Vcorrection" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing Vcorrection")
        for row in reader:
            total += float(row["Vcorrection"])
    return total


def table31_reference_row(case: str) -> dict[str, float]:
    if case not in CASE_DATA:
        raise KeyError(f"No Table 3-1 reference row configured for {case}")

    config = CASE_DATA[case]
    csv_path = TABLE_DATA / config["volume_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    geometry = importlib.import_module(str(config["mesh_module"]))
    table_level = int(config.get("table_level", 0))
    points, faces = geometry.generate_mesh(table_level)
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    v_flat = _closed_surface_volume(points, faces)
    v_patch = _sum_vcorrection(csv_path)
    v_present = v_flat + v_patch
    v_exact = float(config["exact_volume"])
    return {
        "v_flat": v_flat,
        "v_patch": v_patch,
        "v_present": v_present,
        "v_exact": v_exact,
        "pl_err": abs(v_flat - v_exact) / v_exact,
        "present_err": abs(v_present - v_exact) / v_exact,
    }


def apply_table31_reference(row: dict, case: str, *, table_label_required: bool = False) -> dict:
    """Replace level-0 PL and present errors with Table 3-1 values."""

    label = str(row.get("label", ""))
    if table_label_required:
        if not label.startswith("Table 3-1"):
            return row
    elif int(row.get("level", -1)) != 0:
        return row

    ref = table31_reference_row(case)
    row["pl_err"] = ref["pl_err"]
    row["present_err"] = ref["present_err"]
    row["table31_v_flat"] = ref["v_flat"]
    row["table31_v_patch"] = ref["v_patch"]
    row["table31_v_present"] = ref["v_present"]
    if "pl" in row:
        row["pl"] = ref["v_flat"]
    if "present" in row:
        row["present"] = ref["v_present"]
    return row
