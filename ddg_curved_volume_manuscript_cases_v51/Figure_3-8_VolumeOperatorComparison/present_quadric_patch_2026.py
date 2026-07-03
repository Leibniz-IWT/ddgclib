#!/usr/bin/env python3
"""Present class-aware quadric-patch volume operator for Figure 3-8.

This wrapper calls the manuscript curved-volume implementation used for the
present-method line.  It is placed at the top of Figure_3-8_VolumeOperatorComparison so the
present operator is invoked beside the literature comparison modules.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CURVED_VOLUME = HERE / "_curved_volume.py"


def _import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manuscript_curved_volume(
    points,
    faces,
    workdir,
    *,
    msh_path=None,
    coeffs_kwargs=None,
    complex_dtype="vf",
) -> float:
    """Call the manuscript curved-volume operator and suppress helper chatter."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "complex_dtype": complex_dtype,
        "workdir": workdir,
    }
    if coeffs_kwargs is not None:
        kwargs["coeffs_kwargs"] = coeffs_kwargs
    if msh_path is not None:
        kwargs["msh_path"] = msh_path

    cv = _import_from_path("curved_volume_wrapper", CURVED_VOLUME)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(cv.curved_volume((points, faces), **kwargs))


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
