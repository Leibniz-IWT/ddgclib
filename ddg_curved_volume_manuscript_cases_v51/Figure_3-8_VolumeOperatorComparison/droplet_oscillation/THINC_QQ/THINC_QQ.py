#!/usr/bin/env python3
"""Run the THINC/QQ-type dynamic-volume recomputation for this case."""

from pathlib import Path
import subprocess
import sys

RUNNER = Path(__file__).resolve().parents[2] / "common" / "recompute_dynamic_case.py"

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--case-dir",
            str(Path(__file__).resolve().parents[1]),
            "--method",
            "thinc_qq",
            "--method-dir",
            str(Path(__file__).resolve().parent),
            "--reuse-existing",
        ],
        check=True,
    )
