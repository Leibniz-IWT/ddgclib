#!/usr/bin/env python3
"""Regenerate all Figure 3-8 method curves for the droplet_oscillation dynamic case.

The equations, literature references, and callable functions are kept directly
in the top-level author-year files in this Figure_3-8_VolumeOperatorComparison folder.
"""

from pathlib import Path
import subprocess
import sys

RUNNER = Path(__file__).resolve().parents[1] / "common" / "recompute_dynamic_case.py"

if __name__ == "__main__":
    subprocess.run([sys.executable, str(RUNNER), "--case-dir", str(Path(__file__).resolve().parent)], check=True)
