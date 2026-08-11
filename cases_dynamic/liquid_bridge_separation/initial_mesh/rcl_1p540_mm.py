#!/usr/bin/env python3
"""Regenerate the rCL = 1.540 mm initial mesh archive source.

The called script documents the mesh equations: Pitois Fig. 5 geometry,
contact-plane relation, smooth volume-matched free-surface profile, Pappus
volume estimate, and Gmsh tetrahedral volume correction.
"""

from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "rcl_1p540_mm" / "scripts" / "rcl_1p540_mm.py"
OUTPUT_MESH = (
    ROOT
    / "rcl_1p540_mm"
    / "scripts"
    / "out"
    / "Case_2b_axisym_initialshape_Gmsh"
    / "fig"
    / "mesh_iter0012.msh"
)
CANONICAL_MESH = ROOT / "rcl_1p540_mm" / "mesh" / "rcl_1p540_mm.msh"


def main() -> None:
    if not SCRIPT.is_file():
        raise FileNotFoundError(f"Missing generator script: {SCRIPT}")
    subprocess.run([sys.executable, str(SCRIPT)], cwd=SCRIPT.parent, check=True)
    CANONICAL_MESH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_MESH, CANONICAL_MESH)
    print(f"rCL = 1.540 mm mesh: {CANONICAL_MESH}")


if __name__ == "__main__":
    main()
