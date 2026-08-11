"""Case 3: rCL = 1.486 mm compressible label with limit projection."""

from _ddgclib_case_runner import run_case, show_case_process


CASE_ID = "3"
CASE_TITLE = "rCL = 1.486 mm force-calibrated mesh, compressible label"


def initial_mesh_process() -> str:
    return "load force-calibrated rCL = 1.486 mm mesh"


def pressure_process() -> str:
    return "use compressible label with incompressible-limit projection"


def force_process() -> str:
    return "use ddgclib FHeron capillary force and edge-flux viscous force"


def optional_force_process() -> str:
    return "PR33 pressure force off, hydrostatic force off"


def contact_line_process() -> str:
    return "contact line can grow outward"


def output_process() -> str:
    return "write case3 output, record every step, mesh PNG every 100 steps"


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
