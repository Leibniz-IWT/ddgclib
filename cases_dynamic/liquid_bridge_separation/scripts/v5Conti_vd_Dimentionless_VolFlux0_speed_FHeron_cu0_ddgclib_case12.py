"""Case 12: rCL = 1.486 mm force-calibrated mesh with PR33 pressure force."""

from _ddgclib_case_runner import run_case, show_case_process


CASE_ID = "12"
CASE_TITLE = "rCL = 1.486 mm, compressible label, PR33 zero gauge"


def initial_mesh_process() -> str:
    return "load force-calibrated rCL = 1.486 mm mesh"


def pressure_process() -> str:
    return "use compressible label with incompressible-limit projection"


def force_process() -> str:
    return "use ddgclib FHeron capillary force and tetrahedral Cauchy viscous force"


def optional_force_process() -> str:
    return "call PR33 operator path, add Fp = B^T p, use p_ref = 0, hydrostatic top reference on"


def contact_line_process() -> str:
    return "outward growth constrained; PR33 contact-line mobility on"


def output_process() -> str:
    return "write case12 output, record every step, mesh PNG every 500 steps"


def process_steps() -> tuple[str, ...]:
    return (
        initial_mesh_process(),
        pressure_process(),
        force_process(),
        optional_force_process(),
        contact_line_process(),
        output_process(),
    )


def main(argv: list[str] | None = None) -> None:
    show_case_process(CASE_ID, CASE_TITLE, process_steps())
    run_case(CASE_ID, argv=argv)


if __name__ == "__main__":
    main()
