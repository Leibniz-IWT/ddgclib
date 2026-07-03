#!/usr/bin/env python3
"""Press/run this file to regenerate Figure 3-8 CSVs, PNGs, and PDF."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "recompute_all_cases.py"


if __name__ == "__main__":
    subprocess.run([sys.executable, str(RUNNER)], cwd=str(HERE), check=True)
