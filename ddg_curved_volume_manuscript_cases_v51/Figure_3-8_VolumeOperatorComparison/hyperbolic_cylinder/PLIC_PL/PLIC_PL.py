#!/usr/bin/env python3
"""Run the PLIC / PL baseline recomputation for this geometry."""

from pathlib import Path
import subprocess
import sys

COMMON = Path(__file__).resolve().parents[2] / "common"
RUNNER = COMMON / "recompute_case.py"

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--method",
            "plic_pl",
            "--method-dir",
            str(Path(__file__).resolve().parent),
            "--reuse-existing",
        ],
        check=True,
    )
