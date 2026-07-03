#!/usr/bin/env python3
"""Runtime bootstrap for Figure 3-8 volume-operator comparison scripts.

The figure scripts are intended to be runnable from a plain terminal. This
module keeps third-party dependencies local to the Figure_3-8_VolumeOperatorComparison folder
when they are missing from the active Python interpreter.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "meshio": "meshio",
    "overlap": "overlap",
    "pypdf": "pypdf",
    "reportlab": "reportlab",
    "PIL": "Pillow",
}


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def deps_dir() -> Path:
    return bundle_root() / "_python_deps"


def add_local_deps_to_path() -> None:
    path = deps_dir()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def missing_imports() -> list[str]:
    add_local_deps_to_path()
    missing: list[str] = []
    for import_name in REQUIRED:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(import_name)
    return missing


def candidate_pythons() -> list[Path]:
    candidates: list[Path] = []
    env_python = os.environ.get("FIGURE_C1_PYTHON")
    if env_python:
        candidates.append(Path(env_python))

    candidates.append(
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "python"
        / "bin"
        / "python3"
    )

    for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3.9"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key not in seen and resolved.exists() and os.access(resolved, os.X_OK):
            seen.add(key)
            out.append(resolved)
    return out


def python_has_deps(python: Path) -> bool:
    code = "import " + ",".join(REQUIRED.keys())
    env = os.environ.copy()
    local = deps_dir()
    if local.exists():
        env["PYTHONPATH"] = str(local) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [str(python), "-c", code],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def maybe_reexec_with_working_python() -> None:
    current = Path(sys.executable).resolve()
    for python in candidate_pythons():
        if python == current:
            continue
        if python_has_deps(python):
            print(f"Re-executing with dependency-ready Python: {python}", flush=True)
            os.execv(str(python), [str(python), *sys.argv])


def install_missing_locally(missing: list[str]) -> None:
    target = deps_dir()
    target.mkdir(parents=True, exist_ok=True)
    specs = [REQUIRED[name] for name in missing]
    print("Installing missing Figure 3-8 dependencies locally:", ", ".join(specs), flush=True)
    print(f"Target: {target}", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(target),
        *specs,
    ]
    try:
        subprocess.check_call(cmd)
    except Exception as exc:
        raise SystemExit(
            "\nCould not install the missing Python packages automatically.\n"
            f"Current Python: {sys.executable}\n"
            f"Missing imports: {', '.join(missing)}\n\n"
            "The 'overlap' package currently provides wheels through common "
            "Python 3.9-3.13 environments. If you are using Python 3.14 and "
            "the install fails, rerun with Python 3.12 or set FIGURE_C1_PYTHON "
            "to a dependency-ready interpreter.\n\n"
            "Example:\n"
            "  FIGURE_C1_PYTHON=/path/to/python3.12 python3 recompute_all_static_cases.py\n"
        ) from exc
    add_local_deps_to_path()


def ensure_runtime() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    missing = missing_imports()
    if not missing:
        return
    maybe_reexec_with_working_python()
    install_missing_locally(missing)
    still_missing = missing_imports()
    if still_missing:
        raise SystemExit(
            "Figure 3-8 dependencies are still missing after installation: "
            + ", ".join(still_missing)
        )


def runtime_env(base: dict[str, str] | None = None) -> dict[str, str]:
    ensure_runtime()
    env = dict(os.environ if base is None else base)
    local = deps_dir()
    if local.exists():
        env["PYTHONPATH"] = str(local) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    return env
