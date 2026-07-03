#!/usr/bin/env python3
"""Recompute all static Figure 3-8 benchmark cases and method result files.

The method equations, literature references, and callable functions are kept in
the top-level author-year files next to this runner.
"""

from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime  # noqa: E402
from recompute_case import METHODS, run_case, write_method_result  # noqa: E402

ensure_runtime()


CASES = [
    "sphere",
    "cylinder",
    "paraboloid",
    "hyperboloid",
    "parabolic_cylinder",
    "hyperbolic_cylinder",
]


def main() -> None:
    for case in CASES:
        case_dir = HERE / case
        print(f"Recomputing {case}...")
        _output, rows = run_case(case_dir)
        for method_id, (folder, _description) in METHODS.items():
            method_dir = case_dir / folder
            method_dir.mkdir(exist_ok=True)
            out = write_method_result(method_dir, method_id, rows)
            print(f"  wrote {out.relative_to(HERE)}")

    print("Static recomputation complete.")
    print("Run Figure_3-8_VolumeOperatorComparison.py after dynamic case result CSVs have also been generated.")


if __name__ == "__main__":
    main()
