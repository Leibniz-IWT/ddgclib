#!/usr/bin/env python3
"""PLIC / PL volume baseline for Figure 3-8.

Literature context:
- Rider and Kothe (1998)
- Scardovelli and Zaleski (1999)

Equation used by the CSV generators:

    V_PL = (1/6) * sum_{(a,b,c) in faces} x_a . (x_b x x_c)

The scripts use ``closed_surface_volume`` as the planar baseline before any
curved-volume correction is added.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime  # noqa: E402

ensure_runtime()

import numpy as np


def signed_closed_surface_volume(points, faces) -> float:
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    total = 0.0
    for a, b, c in faces:
        total += float(np.dot(points[a], np.cross(points[b], points[c]))) / 6.0
    return total


def closed_surface_volume(points, faces) -> float:
    return abs(signed_closed_surface_volume(points, faces))


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
