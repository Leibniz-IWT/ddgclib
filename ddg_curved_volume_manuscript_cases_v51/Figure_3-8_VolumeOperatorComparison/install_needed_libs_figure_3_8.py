#!/usr/bin/env python3
"""Install/check Python packages needed to regenerate Figure 3-8.

Run this file from the Figure_3-8_VolumeOperatorComparison folder if the recomputation scripts
report missing packages. It installs third-party packages into the local
_python_deps folder; manuscript helper code is already bundled beside this
script as _curved_volume.py, curved_volume/, and ddgclib/.
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime, deps_dir  # noqa: E402


def main() -> None:
    ensure_runtime()
    print(f"Figure 3-8 Python packages are available. Local package folder: {deps_dir()}")


if __name__ == "__main__":
    main()
