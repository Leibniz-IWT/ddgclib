#!/usr/bin/env python3
"""Strobl-type sphere/mesh-element overlap operator for Figure 3-8.

Literature context:
- Strobl et al. (2016), exact calculation of sphere/mesh-element overlap.

The comparison fits a sphere and sums sphere/hexahedron overlaps using the
public ``overlap`` package.  This is a sphere-special operator and is therefore
model-matched only for spherical geometry.
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
import overlap


def fit_sphere(points: np.ndarray, vertex_ids) -> tuple[np.ndarray, float] | None:
    """Least-squares sphere fit used before the Strobl sphere-overlap call.

    The comparison is intentionally sphere-special: it first estimates

        |x - center|^2 = R^2

    from the available mesh vertices, then evaluates sphere/hexahedron overlap
    volumes.  This keeps the fitted-sphere step visible in the author-year file
    instead of hiding it in a shared helper.
    """
    xyz = points[np.asarray(vertex_ids, dtype=int)]
    mat = np.column_stack((xyz[:, 0], xyz[:, 1], xyz[:, 2], np.ones(len(xyz))))
    rhs = -np.sum(xyz * xyz, axis=1)
    coeffs, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
    center = -0.5 * coeffs[:3]
    radius2 = float(np.dot(center, center) - coeffs[3])
    if radius2 <= 0.0:
        return None
    return center, float(np.sqrt(radius2))


def sphere_hexgrid_volume(points, *, hexgrid_n: int = 4) -> float:
    """Fit a sphere and sum sphere/hexahedron overlap volumes."""
    points = np.asarray(points, dtype=float)
    sphere = fit_sphere(points, np.arange(len(points)))
    if sphere is None:
        raise ValueError("Could not fit a sphere for the Strobl overlap comparison.")

    center, radius = sphere
    center = np.asarray(center, dtype=float)
    radius = float(radius)
    sphere_obj = overlap.Sphere(center, radius)

    pad = 1.05 * radius
    axes = [np.linspace(center[d] - pad, center[d] + pad, hexgrid_n + 1) for d in range(3)]
    total = 0.0
    for i in range(hexgrid_n):
        for j in range(hexgrid_n):
            for k in range(hexgrid_n):
                x0, x1 = axes[0][i], axes[0][i + 1]
                y0, y1 = axes[1][j], axes[1][j + 1]
                z0, z1 = axes[2][k], axes[2][k + 1]
                vertices = np.array(
                    [
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x1, y1, z0],
                        [x0, y1, z0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x1, y1, z1],
                        [x0, y1, z1],
                    ],
                    dtype=float,
                )
                total += overlap.overlap_volume(sphere_obj, overlap.Hexahedron(vertices))
    return float(total)


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
