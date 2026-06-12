#!/usr/bin/env python3
"""Regenerate the rCL = 1.486 mm force calibrated initial mesh.

The called script documents the mesh equations: Pitois Fig. 5 first force
point, root solve for r_CL, and ddgclib/FHeron gorge-force calibration.
"""

from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "rcl_1p486_mm" / "scripts" / "rcl_1p486_mm.py"
OUTPUT_DIR = ROOT / "rcl_1p486_mm" / "generated" / "Case_2b_initialmesh_t0_force_calibrated"
OUTPUT_MESH = OUTPUT_DIR / "mesh_initial.msh"
CANONICAL_MESH = ROOT / "rcl_1p486_mm" / "mesh" / "rcl_1p486_mm.msh"


def main() -> None:
    if not SCRIPT.is_file():
        raise FileNotFoundError(f"Missing generator script: {SCRIPT}")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out-dir", str(OUTPUT_DIR)],
        cwd=SCRIPT.parent,
        check=True,
    )
    CANONICAL_MESH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_MESH, CANONICAL_MESH)
    print(f"rCL = 1.486 mm mesh: {CANONICAL_MESH}")


if __name__ == "__main__":
    main()
