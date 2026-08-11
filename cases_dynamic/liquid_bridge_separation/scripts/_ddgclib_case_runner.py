"""Small entry point helper for the 12 Pitois liquid-bridge benchmark cases.

The numbered case scripts intentionally stay thin.  They only select the
validated case parameters; the ddgclib/FHeron/PR33 implementation lives in
``_ddgclib_case_core.py``.

Equation/reference map for the case switches below:
    Pitois, Moucheront, Chateau, J. Colloid Interface Sci. 231 (2000):
        benchmark geometry, Fig. 5 force curve, and Eq. [6] contact radius.
    ddgclib FHeron path:
        surface-tension force from Heron/dual-area geometry.  The DDG geometry
        background follows Meyer et al. (2003), "Discrete Differential-Geometry
        Operators for Triangulated 2-Manifolds".
    Newtonian tet-Cauchy viscous path:
        sigma = -p I + mu (grad u + grad u^T).  Standard continuum-mechanics
        form, e.g. Batchelor, "An Introduction to Fluid Dynamics" (1967).
    Incompressible / incompressible-limit projection:
        pressure/volume projection in the spirit of Chorin (1968).  In this
        code the PR33 operator path assembles the volume-gradient form
        B M^-1 B^T p = residual.
    PR33 pressure force:
        internal ddgclib/PR33 operator test, F_p = B^T p, implemented in
        cases_dynamic/oscillating_droplet_p_ref/scripts/pr33_operators.py.

Notes:
    Cases 1-8 keep PR33 pressure force off.  Cases 9-12 turn it on.
    These twelve selected cases do not pass --solver-bulk-viscosity-pa, so the
    optional zeta div(u) I term is not part of the selected case definitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

from v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib import main as _base_main
from _ddgclib_case_core import main as _core_main


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
R154_MESH = PACKAGE_ROOT / "initial_mesh" / "rcl_1p540_mm" / "mesh" / "rcl_1p540_mm.msh"
FORCE_CALIBRATED_MESH = PACKAGE_ROOT / "initial_mesh" / "rcl_1p486_mm" / "mesh" / "rcl_1p486_mm.msh"
CASE_SCRIPT_PREFIX = "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case"


def _case_output_dir(case_id: str) -> Path:
    return ROOT / f"{CASE_SCRIPT_PREFIX}{case_id}"


COMMON = [
    "--steps",
    "5000",
    "--initial-contact-radius-mode",
    "loaded",
    "--no-cox-contact-line-force",
    "--gorge-pressure-model",
    "axisym_fit",
    "--fheron-neck-pressure-factor",
    "1.0",
    "--contact-line-max-slide-um",
    "5.0",
    "--history-png-every",
    "0",
]


CASE_ARGS: dict[str, list[str]] = {
    "1": [
        "--closure",
        "compressible",
        "--use-pitois-eq6-contact-radius",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--allow-contact-line-growth",
        "--solver-viscous-force-model",
        "edge_flux",
        "--no-solver-pr33-pressure-force",
        "--no-solver-hydrostatic-force",
        "--max-acceleration",
        "3.8095238095238094e-6",
        "--record-every",
        "100",
        "--mesh-snapshot-every",
        "100",
        "--out-dir",
        str(_case_output_dir("1")),
    ],
    "2": [
        "--closure",
        "incompressible",
        "--use-pitois-eq6-contact-radius",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--solver-viscous-force-model",
        "edge_flux",
        "--no-solver-pr33-pressure-force",
        "--no-solver-hydrostatic-force",
        "--max-acceleration",
        "3.8095238095238094e-6",
        "--record-every",
        "100",
        "--mesh-snapshot-every",
        "100",
        "--out-dir",
        str(_case_output_dir("2")),
    ],
    "3": [
        "--closure",
        "compressible",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(FORCE_CALIBRATED_MESH),
        "--gmsh-contact-angle-kinematics",
        "--compressible-incompressible-limit-projection",
        "--allow-contact-line-growth",
        "--solver-viscous-force-model",
        "edge_flux",
        "--no-solver-pr33-pressure-force",
        "--no-solver-hydrostatic-force",
        "--record-every",
        "1",
        "--mesh-snapshot-every",
        "100",
        "--no-raw-mesh-png",
        "--out-dir",
        str(_case_output_dir("3")),
    ],
    "4": [
        "--closure",
        "incompressible",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(FORCE_CALIBRATED_MESH),
        "--solver-viscous-force-model",
        "edge_flux",
        "--no-solver-pr33-pressure-force",
        "--no-solver-hydrostatic-force",
        "--record-every",
        "1",
        "--mesh-snapshot-every",
        "100",
        "--no-raw-mesh-png",
        "--out-dir",
        str(_case_output_dir("4")),
    ],
    "5": [
        "--closure",
        "compressible",
        "--use-pitois-eq6-contact-radius",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--allow-contact-line-growth",
        "--solver-viscous-force-model",
        "tet_cauchy",
        "--no-solver-pr33-pressure-force",
        "--no-solver-hydrostatic-force",
        "--max-acceleration",
        "3.8095238095238094e-6",
        "--record-every",
        "100",
        "--mesh-snapshot-every",
        "0",
        "--out-dir",
        str(_case_output_dir("5")),
    ],
    "6": [
        "--closure",
        "incompressible",
        "--use-pitois-eq6-contact-radius",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--solver-viscous-force-model",
        "tet_cauchy",
        "--no-solver-pr33-pressure-force",
        "--no-solver-hydrostatic-force",
        "--max-acceleration",
        "3.8095238095238094e-6",
        "--record-every",
        "100",
        "--mesh-snapshot-every",
        "0",
        "--out-dir",
        str(_case_output_dir("6")),
    ],
    "7": [
        "--closure",
        "compressible",
        "--use-pitois-eq6-contact-radius",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--solver-viscous-force-model",
        "edge_flux",
        "--no-solver-pr33-pressure-force",
        "--solver-hydrostatic-force",
        "--max-acceleration",
        "3.8095238095238094e-6",
        "--hydrostatic-zref",
        "top_cl",
        "--record-every",
        "1",
        "--mesh-snapshot-every",
        "100",
        "--no-raw-mesh-png",
        "--out-dir",
        str(_case_output_dir("7")),
    ],
    "8": [
        "--closure",
        "incompressible",
        "--use-pitois-eq6-contact-radius",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--solver-viscous-force-model",
        "edge_flux",
        "--no-solver-pr33-pressure-force",
        "--solver-hydrostatic-force",
        "--max-acceleration",
        "3.8095238095238094e-6",
        "--hydrostatic-zref",
        "top_cl",
        "--record-every",
        "1",
        "--mesh-snapshot-every",
        "100",
        "--no-raw-mesh-png",
        "--out-dir",
        str(_case_output_dir("8")),
    ],
    "9": [
        "--closure",
        "compressible",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--solver-viscous-force-model",
        "tet_cauchy",
        "--solver-pr33-pressure-force",
        "--compressible-incompressible-limit-projection",
        "--solver-pr33-pressure-reference-mode",
        "zero",
        "--solver-hydrostatic-force",
        "--hydrostatic-zref",
        "top_cl",
        "--record-every",
        "1",
        "--mesh-snapshot-every",
        "100",
        "--no-raw-mesh-png",
        "--out-dir",
        str(_case_output_dir("9")),
    ],
    "10": [
        "--closure",
        "incompressible",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--solver-viscous-force-model",
        "tet_cauchy",
        "--solver-pr33-pressure-force",
        "--solver-hydrostatic-force",
        "--hydrostatic-zref",
        "top_cl",
        "--record-every",
        "1",
        "--mesh-snapshot-every",
        "100",
        "--no-raw-mesh-png",
        "--out-dir",
        str(_case_output_dir("10")),
    ],
    "11": [
        "--closure",
        "compressible",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(R154_MESH),
        "--solver-viscous-force-model",
        "tet_cauchy",
        "--solver-pr33-pressure-force",
        "--compressible-incompressible-limit-projection",
        "--solver-pr33-pressure-reference-mode",
        "heron",
        "--solver-hydrostatic-force",
        "--hydrostatic-zref",
        "top_cl",
        "--record-every",
        "1",
        "--mesh-snapshot-every",
        "100",
        "--no-raw-mesh-png",
        "--out-dir",
        str(_case_output_dir("11")),
    ],
    "12": [
        "--closure",
        "compressible",
        "--compressible-bulk-pa",
        "1.0e5",
        "--initial-msh-path",
        str(FORCE_CALIBRATED_MESH),
        "--solver-viscous-force-model",
        "tet_cauchy",
        "--solver-pr33-pressure-force",
        "--compressible-incompressible-limit-projection",
        "--solver-pr33-pressure-reference-mode",
        "zero",
        "--solver-hydrostatic-force",
        "--hydrostatic-zref",
        "top_cl",
        "--record-every",
        "1",
        "--mesh-snapshot-every",
        "500",
        "--no-raw-mesh-png",
        "--out-dir",
        str(_case_output_dir("12")),
    ],
}

CORE_CASES = {"9", "10", "11", "12"}


def show_case_process(case_id: str, title: str, steps: tuple[str, ...]) -> None:
    """Print the high-level process selected by a thin case wrapper."""
    key = str(case_id)
    print(f"[case {key}] {title}", flush=True)
    for index, step in enumerate(steps, start=1):
        print(f"[case {key}] process {index}: {step}", flush=True)


def run_case(case_id: str, argv: list[str] | None = None) -> None:
    """Run one numbered case; explicit user CLI args override defaults."""
    key = str(case_id)
    if key not in CASE_ARGS:
        raise SystemExit(f"Unknown case {case_id!r}; expected one of {', '.join(sorted(CASE_ARGS))}")
    user_args = list(sys.argv[1:] if argv is None else argv)
    solver_main = _core_main if key in CORE_CASES else _base_main
    solver_main(COMMON + CASE_ARGS[key] + user_args)
