from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from lxml import etree

from patch_office_math_slide2 import (
    MATHML2OMML,
    equation_mathml,
    math as mathml,
    math_shape,
    omml_from_mathml,
    panel,
    root_with_math_namespaces,
    shape_end,
    shape_start,
    text_shape,
    max_shape_id,
)

IWT_EQUATION_FONT_SIZE = 14
DUAL_VOLUME_EQUATION_FONT_SIZE = 15
FULL_FLUX_EQUATION_FONT_SIZE = 13
PAIRWISE_EQUATION_FONT_SIZE = 14
FREE_SURFACE_EQUATION_FONT_SIZE = 15


def dual_volume_equation_mathml() -> list[tuple[str, str]]:
    return [
        (
            "dual 1",
            mathml(
                """
                <mrow>
                  <msubsup><mi>V</mi><mi>i</mi><mi mathvariant="normal">HC</mi></msubsup>
                  <mo>→</mo>
                  <msub><mi>ρ</mi><mi>i</mi></msub>
                  <mo>=</mo>
                  <mfrac><msub><mi>m</mi><mi>i</mi></msub><msubsup><mi>V</mi><mi>i</mi><mi mathvariant="normal">HC</mi></msubsup></mfrac>
                  <mo>→</mo>
                  <msub><mi>p</mi><mi>i</mi></msub>
                  <mo>=</mo>
                  <mi>p</mi><mo stretchy="false">(</mo><msub><mi>ρ</mi><mi>i</mi></msub><mo stretchy="false">)</mo>
                </mrow>
                """
            ),
        ),
        (
            "dual 2",
            mathml(
                """
                <mrow>
                  <mi mathvariant="normal">Delaunay</mi><mo>&#160;</mo><mi mathvariant="normal">flip</mi>
                  <mo>→</mo>
                  <mi>Δ</mi><msubsup><mi>V</mi><mi>i</mi><mi mathvariant="normal">HC</mi></msubsup>
                  <mo>→</mo>
                  <mi>Δ</mi><msub><mi>ρ</mi><mi>i</mi></msub>
                  <mo>,</mo>
                  <mi>Δ</mi><msub><mi>p</mi><mi>i</mi></msub>
                </mrow>
                """
            ),
        ),
        (
            "dual 3",
            mathml(
                """
                <mrow>
                  <msubsup><mi>V</mi><mi>i</mi><mi mathvariant="normal">bary</mi></msubsup>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>t</mi><mo>∋</mo><mi>i</mi></mrow></msub>
                  <mfrac><msub><mi>V</mi><mi>t</mi></msub><mn>4</mn></mfrac>
                </mrow>
                """
            ),
        ),
        (
            "dual 4",
            mathml(
                """
                <mrow>
                  <msub><mi mathvariant="bold">B</mi><mrow><mi>t</mi><mo>,</mo><mi>i</mi></mrow></msub>
                  <mo>=</mo>
                  <mfrac><mrow><mi>∂</mi><msub><mi>V</mi><mi>t</mi></msub></mrow><mrow><mi>∂</mi><msub><mi mathvariant="bold">x</mi><mi>i</mi></msub></mrow></mfrac>
                </mrow>
                """
            ),
        ),
        (
            "dual 5",
            mathml(
                """
                <mrow>
                  <msup><mi mathvariant="bold">T</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msup>
                  <mo>=</mo>
                  <mi mathvariant="normal">Delaunay</mi>
                  <mo stretchy="false">(</mo><msup><mi mathvariant="bold">x</mi><mi>n</mi></msup><mo stretchy="false">)</mo>
                </mrow>
                """
            ),
        ),
        (
            "dual 6",
            mathml(
                """
                <mrow>
                  <mo stretchy="false">{</mo>
                  <msub><mi>V</mi><mi>t</mi></msub>
                  <mo>,</mo>
                  <mi mathvariant="bold">B</mi>
                  <mo>,</mo>
                  <msub><mi>m</mi><mi>t</mi></msub>
                  <mo>,</mo>
                  <msubsup><mi>V</mi><mi>t</mi><mi mathvariant="normal">tar</mi></msubsup>
                  <mo stretchy="false">}</mo>
                  <mo>&#160;</mo><mi mathvariant="normal">rebuilt</mi>
                </mrow>
                """
            ),
        ),
    ]


def patch_iwt_dual_volume_slide(xml: str, xslt: etree.XSLT) -> str:
    xml = root_with_math_namespaces(xml)
    next_id = max_shape_id(xml) + 1
    equations = {
        name: omml_from_mathml(xslt, eq, size=DUAL_VOLUME_EQUATION_FONT_SIZE)
        for name, eq in dual_volume_equation_mathml()
    }

    shapes: list[str] = []

    def new_id() -> int:
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    shapes.append(shape_start(new_id(), "IWT dual volume body cover", 18, 104, 1230, 592, "FFFFFF", None) + shape_end())
    panels = [
        (32, 128, 565, 246, "GitHub / HC pressure path", "A61B1B"),
        (632, 128, 585, 246, "Benchmark pressure path", "087E8B"),
        (32, 420, 565, 146, "Why V/4 helps", "F07D00"),
        (632, 420, 585, 146, "What is not claimed", "B7791F"),
    ]
    for x, y, w, h, title, color in panels:
        shapes.append(panel(new_id(), x, y, w, h))
        shapes.append(shape_start(new_id(), "IWT dual volume color bar", x, y, 7, h, color, None) + shape_end())
        shapes.append(text_shape(new_id(), f"IWT dual volume title {title}", title, x + 22, y + 16, w - 44, 28, font_size=21, bold=True))

    labels = {
        "dual 1": "density/pressure path",
        "dual 2": "connectivity artifact",
        "dual 3": "barycentric volume",
        "dual 4": "volume gradient",
        "dual 5": "active tet list",
        "dual 6": "consistent remap",
    }

    def add_eq(name: str, x: float, y: float, w: float, label_x: float) -> None:
        shapes.append(math_shape(new_id(), f"IWT Office dual equation {name}", equations[name], x, y, w, 38))
        shapes.append(text_shape(new_id(), f"IWT dual label {name}", labels[name], label_x, y + 9, 130, 18, font_size=8, color="5B6470"))

    add_eq("dual 1", 62, 190, 405, 482)
    add_eq("dual 2", 62, 260, 405, 482)
    add_eq("dual 3", 662, 188, 405, 1092)
    add_eq("dual 4", 662, 248, 405, 1092)
    add_eq("dual 5", 662, 308, 405, 1092)
    shapes.append(
        text_shape(
            new_id(),
            "IWT dual github issue text",
            "If the HC dual-cell polyhedron changes after a Delaunay flip, V_i^HC can jump even when the physical liquid did not locally compress by that amount.",
            62,
            318,
            450,
            34,
            font_size=10.5,
            bold=True,
            color="A61B1B",
        )
    )
    add_eq("dual 6", 662, 458, 405, 1092)
    shapes.append(
        text_shape(
            new_id(),
            "IWT dual v4 note",
            "Every tet volume is partitioned exactly to its four vertices, so the nodal volumes sum to the total tet volume.",
            62,
            474,
            450,
            34,
            font_size=11,
            bold=True,
            color="17202A",
        )
    )
    shapes.append(
        text_shape(
            new_id(),
            "IWT dual limitation note",
            "This is a controlled benchmark workaround, not a full conservative old-dual/new-dual overlap remap.",
            662,
            505,
            450,
            34,
            font_size=11,
            bold=True,
            color="17202A",
        )
    )
    shapes.append(panel(new_id(), 32, 608, 1185, 52))
    shapes.append(
        text_shape(
            new_id(),
            "IWT dual bottom takeaway",
            "Deck wording: GitHub problem = Delaunay flip changes HC dual volume and can create artificial density/pressure. Our workaround = tet-volume barycentric volume plus mass/target-volume remap after retriangulation.",
            58,
            620,
            1130,
            24,
            font_size=11.2,
            bold=True,
            color="17202A",
        )
    )
    return xml.replace("</p:spTree>", "".join(shapes) + "</p:spTree>", 1)


def patch_iwt_equation_slide(xml: str, xslt: etree.XSLT) -> str:
    xml = root_with_math_namespaces(xml)
    next_id = max_shape_id(xml) + 1
    equations = {
        name: omml_from_mathml(xslt, eq, size=IWT_EQUATION_FONT_SIZE)
        for name, eq, _size, _group in equation_mathml()
    }

    shapes: list[str] = []

    def new_id() -> int:
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    # Cover only the equation body, preserving IWT header, logo, and footer.
    shapes.append(shape_start(new_id(), "IWT equation body cover", 18, 100, 1230, 592, "FFFFFF", None) + shape_end())

    left_x, right_x = 22, 626
    top_y, bottom_y = 102, 360
    top_panel_h, bottom_panel_h = 250, 265
    panel_w = 588
    for x, y, w, h in [
        (left_x, top_y, panel_w, top_panel_h),
        (right_x, top_y, 598, top_panel_h),
        (left_x, bottom_y, panel_w, bottom_panel_h),
        (right_x, bottom_y, 598, bottom_panel_h),
    ]:
        shapes.append(panel(new_id(), x, y, w, h))
        shapes.append(shape_start(new_id(), "IWT equation orange bar", x, y, 7, h, "F07D00", None) + shape_end())

    labels = {
        "force 1": "component split",
        "force 2": "HC Heron force",
        "force 4": "damping force",
        "force 6": "time integration",
        "closure 7": "EOS pressure force",
        "closure 0": "reference pressure",
        "closure 1": "density target",
        "closure 2": "stiffness entry",
        "closure 6": "EOS pressure",
        "closure 3": "compressible solve",
        "closure 8": "projection force",
        "closure 4": "incomp pressure solve",
        "closure 5": "velocity constraint",
        "flux 1": "material face flux",
        "flux 7": "Lagrangian face",
        "flux 2": "upwind momentum",
        "flux 5": "face mass source",
        "flux 6": "mass update",
    }

    def add_rows(
        names: list[str],
        x: float,
        y: float,
        row0: float,
        step: float,
        eq_w: float,
        label_x: float,
        eq_h: float = 34,
    ) -> None:
        for idx, name in enumerate(names):
            row_y = y + row0 + idx * step
            shapes.append(math_shape(new_id(), f"IWT Office equation {name}", equations[name], x + 20, row_y, eq_w, eq_h))
            shapes.append(
                text_shape(
                    new_id(),
                    f"IWT equation label {name}",
                    labels[name],
                    x + label_x,
                    row_y + 8,
                    108,
                    17,
                    font_size=8,
                    color="5B6470",
                )
            )

    shapes.append(text_shape(new_id(), "IWT force title", "Common force/update", left_x + 22, top_y + 18, 350, 28, font_size=21, bold=True))
    add_rows(["force 1", "force 2", "force 4", "force 6"], left_x, top_y, 58, 46, 432, 462, 38)

    shapes.append(text_shape(new_id(), "IWT compressible title", "Compressible EOS closure", right_x + 22, top_y + 18, 380, 28, font_size=21, bold=True))
    add_rows(["closure 7", "closure 0", "closure 1", "closure 6", "closure 3"], right_x, top_y, 52, 40, 432, 462, 38)

    shapes.append(text_shape(new_id(), "IWT flux title", "ALE flux and mass update", left_x + 22, bottom_y + 18, 380, 28, font_size=21, bold=True))
    add_rows(["flux 1", "flux 7", "flux 2", "flux 5", "flux 6"], left_x, bottom_y, 50, 42, 432, 462, 38)

    shapes.append(text_shape(new_id(), "IWT incompressible title", "Incompressible projection", right_x + 22, bottom_y + 18, 390, 28, font_size=21, bold=True))
    add_rows(["closure 8", "closure 2", "closure 4", "closure 5"], right_x, bottom_y, 52, 52, 432, 462, 42)

    shapes.append(
        text_shape(
            new_id(),
            "IWT notation convention",
            "Notation: scalars italic; vectors bold upright; matrices/tensors/operators bold upright; multi-letter labels upright roman.",
            42,
            626,
            1120,
            16,
            font_size=8,
            color="5B6470",
        )
    )

    shapes.append(panel(new_id(), 22, 638, 1202, 50))
    shapes.append(
        text_shape(
            new_id(),
            "IWT reference label",
            "Literature references",
            46,
            650,
            180,
            20,
            font_size=12,
            bold=True,
            color="F07D00",
        )
    )
    shapes.append(
        text_shape(
            new_id(),
            "IWT projection reference",
            "Incompressible projection: Chorin, A. J. (1968), Numerical solution of the Navier-Stokes equations, Math. Comput. 22(104), 745-762. doi: 10.1090/S0025-5718-1968-0242392-2.",
            236,
            642,
            972,
            18,
            font_size=9,
            color="17202A",
        )
    )
    shapes.append(
        text_shape(
            new_id(),
            "IWT compressible reference",
            "Compressible EOS pressure solve: Chen, Z. and Przekwas, A. (2010), A coupled pressure-based computational method for incompressible/compressible flows, JCP 229(24), 9150-9165. doi: 10.1016/j.jcp.2010.08.029.",
            236,
            664,
            972,
            18,
            font_size=9,
            color="17202A",
        )
    )
    return xml.replace("</p:spTree>", "".join(shapes) + "</p:spTree>", 1)


def patch_iwt_full_flux_slide(xml: str, xslt: etree.XSLT) -> str:
    xml = root_with_math_namespaces(xml)
    next_id = max_shape_id(xml) + 1
    equations = {
        name: omml_from_mathml(xslt, eq, size=FULL_FLUX_EQUATION_FONT_SIZE)
        for name, eq, _size, _group in equation_mathml()
    }

    shapes: list[str] = []

    def new_id() -> int:
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    shapes.append(shape_start(new_id(), "IWT full flux body cover", 18, 104, 1230, 592, "FFFFFF", None) + shape_end())

    panels = [
        (28, 112, 1200, 150, "Cell-face mass flux"),
        (28, 280, 1200, 150, "Cell-face momentum flux"),
        (28, 448, 1200, 158, "Mass, limiter, and vertex-force update"),
    ]
    for x, y, w, h, title in panels:
        shapes.append(panel(new_id(), x, y, w, h))
        shapes.append(shape_start(new_id(), "IWT full flux orange bar", x, y, 7, h, "F07D00", None) + shape_end())
        shapes.append(text_shape(new_id(), f"IWT full flux title {title}", title, x + 22, y + 14, 520, 26, font_size=21, bold=True))

    labels = {
        "flux 1": "relative face flux",
        "flux 3": "Rusanov mass flux",
        "flux 2": "upwind momentum",
        "flux 4": "Rusanov momentum",
        "flux 5": "face mass source",
        "flux 6": "limited mass update",
        "force 5": "vertex flux force",
    }

    def add_eq(name: str, x: float, y: float, w: float, h: float, label: str) -> None:
        shapes.append(math_shape(new_id(), f"IWT Office full flux equation {name}", equations[name], x, y, w, h))
        shapes.append(
            text_shape(
                new_id(),
                f"IWT full flux label {name}",
                label,
                1034,
                y + 9,
                150,
                18,
                font_size=9,
                color="5B6470",
            )
        )

    add_eq("flux 1", 58, 162, 950, 38, labels["flux 1"])
    add_eq("flux 3", 58, 212, 965, 42, labels["flux 3"])
    add_eq("flux 2", 58, 330, 950, 38, labels["flux 2"])
    add_eq("flux 4", 58, 380, 965, 42, labels["flux 4"])
    add_eq("flux 5", 58, 488, 950, 34, labels["flux 5"])
    add_eq("flux 6", 58, 528, 950, 34, labels["flux 6"])
    add_eq("force 5", 58, 568, 950, 30, labels["force 5"])

    shapes.append(panel(new_id(), 28, 620, 1200, 62))
    shapes.append(
        text_shape(
            new_id(),
            "IWT full flux note",
            "Impact cases #7/#8 use lambda_m = 1 and lambda_F = 1 with nonzero cell-average/Rusanov flux. This is a diagnostic stress test, not the physical Lagrangian material-face flux used for #5/#6.",
            52,
            630,
            735,
            30,
            font_size=11,
            bold=True,
            color="A61B1B",
        )
    )
    shapes.append(
        text_shape(
            new_id(),
            "IWT full flux references",
            "References: Hirt, Amsden & Cook (1974), JCP 14, 227-253; Toro (2009), Riemann Solvers and Numerical Methods for Fluid Dynamics.",
            805,
            630,
            390,
            30,
            font_size=10,
            color="17202A",
        )
    )
    shapes.append(
        text_shape(
            new_id(),
            "IWT full flux notation convention",
            "Notation: scalars italic; vectors bold upright; matrices/tensors/operators bold upright; multi-letter labels upright roman.",
            52,
            664,
            735,
            14,
            font_size=8,
            color="5B6470",
        )
    )
    return xml.replace("</p:spTree>", "".join(shapes) + "</p:spTree>", 1)


def pairwise_equation_mathml() -> list[tuple[str, str]]:
    return [
        (
            "pair 1",
            mathml(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mrow><mi mathvariant="normal">p</mi><mo>,</mo><mi>A</mi></mrow></msubsup>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>j</mi><mo>∈</mo><mi mathvariant="bold">N</mi><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msub>
                  <mo>-</mo><mfrac><mn>1</mn><mn>2</mn></mfrac>
                  <mo stretchy="false">(</mo><msub><mi>p</mi><mi>i</mi></msub><mo>+</mo><msub><mi>p</mi><mi>j</mi></msub><mo stretchy="false">)</mo>
                  <msub><mi mathvariant="bold">A</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub>
                </mrow>
                """
            ),
        ),
        (
            "pair 2",
            mathml(
                """
                <mrow>
                  <msub><mi>p</mi><mi>i</mi></msub>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>t</mi><mo>∋</mo><mi>i</mi></mrow></msub>
                  <msub><mi>w</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub>
                  <msub><mi>p</mi><mi>t</mi></msub>
                  <mo>,</mo><mo>&#160;</mo>
                  <msub><mi>w</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub>
                  <mo>=</mo>
                  <mfrac>
                    <msub><mi>V</mi><mi>t</mi></msub>
                    <mrow><msub><mi mathvariant="normal">∑</mi><mrow><mi>s</mi><mo>∋</mo><mi>i</mi></mrow></msub><msub><mi>V</mi><mi>s</mi></msub></mrow>
                  </mfrac>
                </mrow>
                """
            ),
        ),
        (
            "pair 3",
            mathml(
                """
                <mrow>
                  <msub><mi mathvariant="bold">P</mi><mi>A</mi></msub>
                  <mi mathvariant="bold">p</mi>
                  <mo>=</mo>
                  <msup><mi mathvariant="bold">F</mi><mrow><mi mathvariant="normal">p</mi><mo>,</mo><mi>A</mi></mrow></msup>
                  <mo>,</mo><mo>&#160;</mo>
                  <msub><mi mathvariant="bold">S</mi><mi>A</mi></msub>
                  <mo>=</mo>
                  <mi mathvariant="bold">B</mi>
                  <msup><mi mathvariant="bold">M</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
                  <msub><mi mathvariant="bold">P</mi><mi>A</mi></msub>
                </mrow>
                """
            ),
        ),
        (
            "pair 4",
            mathml(
                """
                <mrow>
                  <mo stretchy="false">[</mo>
                  <mi mathvariant="bold">I</mi>
                  <mo>+</mo>
                  <msup><mrow><mi>Δ</mi><mi>t</mi></mrow><mn>2</mn></msup>
                  <msub><mi mathvariant="bold">D</mi><mi>K</mi></msub>
                  <msub><mi mathvariant="bold">S</mi><mi>A</mi></msub>
                  <mo stretchy="false">]</mo>
                  <mi>δ</mi><mi mathvariant="bold">p</mi>
                  <mo>=</mo>
                  <mo>-</mo>
                  <msub><mi mathvariant="bold">D</mi><mi>K</mi></msub>
                  <mo stretchy="false">(</mo>
                  <mi mathvariant="bold">V</mi>
                  <mo>-</mo>
                  <msup><mi mathvariant="bold">V</mi><mi mathvariant="normal">tar</mi></msup>
                  <mo>+</mo>
                  <mi>Δ</mi><mi>t</mi><mi mathvariant="bold">B</mi><msup><mi mathvariant="bold">u</mi><mo>*</mo></msup>
                  <mo stretchy="false">)</mo>
                </mrow>
                """
            ),
        ),
        (
            "pair 5",
            mathml(
                """
                <mrow>
                  <msub><mi mathvariant="bold">S</mi><mi>A</mi></msub>
                  <mi mathvariant="bold">p</mi>
                  <mo>=</mo>
                  <mfrac>
                    <mrow><mi mathvariant="bold">r</mi><mo>-</mo><mi mathvariant="bold">B</mi><mi mathvariant="bold">u</mi></mrow>
                    <mrow><mi>Δ</mi><mi>t</mi></mrow>
                  </mfrac>
                  <mo>-</mo>
                  <mi mathvariant="bold">B</mi>
                  <msup><mi mathvariant="bold">M</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
                  <msub><mi mathvariant="bold">F</mi><mi mathvariant="normal">np</mi></msub>
                </mrow>
                """
            ),
        ),
        (
            "pair 6",
            mathml(
                """
                <mrow>
                  <mi mathvariant="bold">B</mi>
                  <msup><mi mathvariant="bold">u</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msup>
                  <mo>=</mo>
                  <mi mathvariant="bold">r</mi>
                </mrow>
                """
            ),
        ),
        (
            "pair 7",
            mathml(
                """
                <mrow>
                  <msub><mi mathvariant="bold">S</mi><mi>A</mi></msub>
                  <mo>=</mo>
                  <mi mathvariant="bold">B</mi>
                  <msup><mi mathvariant="bold">M</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
                  <msub><mi mathvariant="bold">P</mi><mi>A</mi></msub>
                  <mo>≠</mo>
                  <mi mathvariant="bold">B</mi>
                  <msup><mi mathvariant="bold">M</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
                  <msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup>
                </mrow>
                """
            ),
        ),
        (
            "pair 8",
            mathml(
                """
                <mrow>
                  <msub><mi mathvariant="bold">P</mi><mi>A</mi></msub>
                  <mo>≠</mo>
                  <msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup>
                </mrow>
                """
            ),
        ),
        (
            "pair 9",
            mathml(
                """
                <mrow>
                  <mi mathvariant="bold">P</mi>
                  <mo>=</mo>
                  <msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup>
                </mrow>
                """
            ),
        ),
        (
            "pair 10",
            mathml(
                """
                <mrow>
                  <mi mathvariant="bold">B</mi>
                  <mi mathvariant="bold">u</mi>
                  <mo>=</mo>
                  <mi mathvariant="bold">r</mi>
                </mrow>
                """
            ),
        ),
    ]


def patch_iwt_pairwise_pressure_slide(xml: str, xslt: etree.XSLT) -> str:
    xml = root_with_math_namespaces(xml)
    next_id = max_shape_id(xml) + 1
    equations = {
        name: omml_from_mathml(xslt, eq, size=PAIRWISE_EQUATION_FONT_SIZE)
        for name, eq in pairwise_equation_mathml()
    }

    shapes: list[str] = []

    def new_id() -> int:
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    shapes.append(shape_start(new_id(), "IWT pairwise body cover", 18, 104, 1230, 592, "FFFFFF", None) + shape_end())
    panels = [
        (28, 116, 586, 188, "Pairwise force law"),
        (642, 116, 586, 188, "#9 compressible trial"),
        (28, 336, 586, 178, "#10 incompressible trial"),
        (642, 326, 586, 210, "Observed result"),
    ]
    for x, y, w, h, title in panels:
        shapes.append(panel(new_id(), x, y, w, h))
        shapes.append(shape_start(new_id(), "IWT pairwise orange bar", x, y, 7, h, "F07D00", None) + shape_end())
        shapes.append(text_shape(new_id(), f"IWT pairwise title {title}", title, x + 22, y + 14, 500, 26, font_size=21, bold=True))

    labels = {
        "pair 1": "edge pressure force",
        "pair 2": "vertex pressure map",
        "pair 3": "mixed operator",
        "pair 4": "compressible solve",
        "pair 5": "projection solve",
        "pair 6": "velocity constraint",
        "pair 7": "mixed stiffness",
        "pair 8": "not adjoint",
        "pair 9": "consistent force",
        "pair 10": "constraint",
    }

    def add_eq(name: str, x: float, y: float, w: float, h: float, label_x: float) -> None:
        shapes.append(math_shape(new_id(), f"IWT Office pairwise equation {name}", equations[name], x, y, w, h))
        shapes.append(
            text_shape(
                new_id(),
                f"IWT pairwise label {name}",
                labels[name],
                label_x,
                y + 9,
                118,
                18,
                font_size=8,
                color="5B6470",
            )
        )

    add_eq("pair 1", 58, 166, 420, 42, 492)
    add_eq("pair 2", 58, 224, 420, 42, 492)
    add_eq("pair 3", 672, 166, 420, 42, 1104)
    add_eq("pair 4", 672, 224, 420, 42, 1104)
    add_eq("pair 5", 58, 392, 420, 42, 492)
    add_eq("pair 6", 58, 452, 420, 34, 492)

    shapes.append(text_shape(new_id(), "IWT pairwise observed intro", "#9 includes B in the residual, but the force operator is pairwise Aij:", 672, 376, 470, 18, font_size=10, bold=True, color="17202A"))
    add_eq("pair 7", 672, 402, 360, 32, 1104)
    add_eq("pair 8", 672, 442, 210, 30, 1104)
    shapes.append(text_shape(new_id(), "IWT pairwise observed control", "Local EOS modes are not controlled.", 890, 444, 210, 28, font_size=10, bold=True, color="A61B1B"))
    shapes.append(text_shape(new_id(), "IWT pairwise observed consistent", "Consistent local-volume pressure requires:", 672, 482, 270, 18, font_size=10, color="A61B1B"))
    add_eq("pair 9", 940, 478, 120, 28, 1104)
    shapes.append(text_shape(new_id(), "IWT pairwise observed projection", "#10 works because projection enforces:", 672, 512, 270, 18, font_size=10, color="17202A"))
    add_eq("pair 10", 940, 508, 120, 28, 1104)

    shapes.append(panel(new_id(), 28, 574, 1200, 68))
    shapes.append(
        text_shape(
            new_id(),
            "IWT pairwise interpretation label",
            "Interpretation",
            52,
            590,
            160,
            20,
            font_size=12,
            bold=True,
            color="F07D00",
        )
    )
    shapes.append(
        text_shape(
            new_id(),
            "IWT pairwise interpretation",
            "For uniform pressure the pairwise Aij force can reproduce the capillary-scale direction. For compressible local EOS, nonuniform tet pressure corrections need a volume-gradient-compatible operator.",
            218,
            584,
            950,
            30,
            font_size=11,
            bold=True,
            color="17202A",
        )
    )
    shapes.append(
        text_shape(
            new_id(),
            "IWT pairwise references",
            "Refs: Chorin (1968), Math. Comput. 22(104), 745-762; Chen & Przekwas (2010), JCP 229(24), 9150-9165; Desbrun et al. (1999), Implicit fairing of irregular meshes using diffusion and curvature flow.",
            218,
            622,
            950,
            16,
            font_size=8,
            color="5B6470",
        )
    )
    return xml.replace("</p:spTree>", "".join(shapes) + "</p:spTree>", 1)


def free_surface_equation_mathml() -> list[tuple[str, str]]:
    return [
        (
            "free 1",
            mathml(
                """
                <mrow>
                  <msub><mi mathvariant="bold">T</mi><mi mathvariant="normal">GH</mi></msub>
                  <mo>=</mo>
                  <mi mathvariant="normal">droplet_in_box_3d</mi>
                  <mo stretchy="false">(</mo><msub><mi>R</mi><mn>0</mn></msub><mo>,</mo><mi>L</mi><mo stretchy="false">)</mo>
                </mrow>
                """
            ),
        ),
        (
            "free 2",
            mathml(
                """
                <mrow>
                  <mi mathvariant="normal">phase</mi><mo>&#160;</mo><mn>0</mn><mo>=</mo><mi mathvariant="normal">gas</mi>
                  <mo>,</mo><mo>&#160;</mo>
                  <mi mathvariant="normal">phase</mi><mo>&#160;</mo><mn>1</mn><mo>=</mo><mi mathvariant="normal">liquid</mi>
                </mrow>
                """
            ),
        ),
        (
            "free 3",
            mathml(
                """
                <mrow>
                  <mi>Γ</mi>
                  <mo>=</mo>
                  <mi mathvariant="normal">interface</mi><mo>&#160;</mo><mi mathvariant="normal">subcomplex</mi>
                </mrow>
                """
            ),
        ),
        (
            "free 4",
            mathml(
                """
                <mrow>
                  <msubsup><mi>V</mi><mi>i</mi><mi mathvariant="normal">bary</mi></msubsup>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>t</mi><mo>∋</mo><mi>i</mi></mrow></msub>
                  <mfrac><msub><mi>V</mi><mi>t</mi></msub><mn>4</mn></mfrac>
                </mrow>
                """
            ),
        ),
        (
            "free 5",
            mathml(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">Heron</mi></msubsup>
                  <mo>=</mo>
                  <mo>-</mo><mi>γ</mi>
                  <msub><mrow><mo stretchy="false">(</mo><mi mathvariant="normal">HN</mi><mi>dA</mi><mo stretchy="false">)</mo></mrow><mi>i</mi></msub>
                </mrow>
                """
            ),
        ),
        (
            "free 6",
            mathml(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">p</mi></msubsup>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>t</mi><mo>∋</mo><mi>i</mi></mrow></msub>
                  <msub><mi>p</mi><mi>t</mi></msub>
                  <msub><mi mathvariant="bold">B</mi><mrow><mi>t</mi><mo>,</mo><mi>i</mi></mrow></msub>
                </mrow>
                """
            ),
        ),
        (
            "free 7",
            mathml(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">T</mi><mi mathvariant="normal">display</mi><mi>n</mi></msubsup>
                  <mo>=</mo>
                  <mi mathvariant="normal">Delaunay</mi>
                  <mo stretchy="false">(</mo>
                  <msup><mi mathvariant="bold">x</mi><mi>n</mi></msup>
                  <mo stretchy="false">)</mo>
                </mrow>
                """
            ),
        ),
        (
            "free 8",
            mathml(
                """
                <mrow>
                  <mo stretchy="false">{</mo>
                  <msub><mi>V</mi><mi>t</mi></msub><mo>,</mo>
                  <mi mathvariant="bold">B</mi><mo>,</mo>
                  <msub><mi>m</mi><mi>t</mi></msub><mo>,</mo>
                  <msubsup><mi>V</mi><mi>t</mi><mi mathvariant="normal">tar</mi></msubsup>
                  <mo stretchy="false">}</mo>
                  <mo>&#160;</mo><mi mathvariant="normal">rebuilt</mi>
                </mrow>
                """
            ),
        ),
    ]


def patch_iwt_free_surface_slide(xml: str, xslt: etree.XSLT) -> str:
    xml = root_with_math_namespaces(xml)
    next_id = max_shape_id(xml) + 1
    equations = {
        name: omml_from_mathml(xslt, eq, size=FREE_SURFACE_EQUATION_FONT_SIZE)
        for name, eq in free_surface_equation_mathml()
    }

    shapes: list[str] = []

    def new_id() -> int:
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    shapes.append(shape_start(new_id(), "IWT free-surface body cover", 18, 104, 1230, 592, "FFFFFF", None) + shape_end())
    panels = [
        (34, 120, 590, 210, "Displayed triangulation"),
        (656, 120, 560, 210, "Stable dynamic closure"),
        (34, 374, 590, 190, "Active tet retopology"),
        (656, 374, 560, 190, "Important interpretation"),
    ]
    for x, y, w, h, title in panels:
        shapes.append(panel(new_id(), x, y, w, h))
        shapes.append(shape_start(new_id(), "IWT free-surface orange bar", x, y, 7, h, "F07D00", None) + shape_end())
        shapes.append(text_shape(new_id(), f"IWT free-surface title {title}", title, x + 22, y + 16, 510, 26, font_size=21, bold=True))

    labels = {
        "free 1": "GitHub source mesh",
        "free 2": "phase labels",
        "free 3": "interface mesh",
        "free 4": "bary volume",
        "free 5": "HC Heron force",
        "free 6": "pressure force",
        "free 7": "active tets",
        "free 8": "consistent rebuild",
    }

    def add_eq(name: str, x: float, y: float, w: float, label_x: float) -> None:
        shapes.append(math_shape(new_id(), f"IWT Office free-surface equation {name}", equations[name], x, y, w, 36))
        shapes.append(
            text_shape(
                new_id(),
                f"IWT free-surface label {name}",
                labels[name],
                label_x,
                y + 9,
                122,
                18,
                font_size=8,
                color="5B6470",
            )
        )

    add_eq("free 1", 64, 178, 390, 492)
    add_eq("free 2", 64, 228, 390, 492)
    add_eq("free 3", 64, 278, 390, 492)
    add_eq("free 4", 686, 186, 330, 1092)
    add_eq("free 5", 686, 236, 330, 1092)
    add_eq("free 6", 686, 286, 330, 1092)
    add_eq("free 7", 64, 438, 390, 492)
    add_eq("free 8", 64, 492, 390, 492)
    shapes.append(
        text_shape(
            new_id(),
            "IWT free-surface meaning",
            "The successful Rayleigh response is not the original GitHub multiphase stress solver. It is the GitHub two-phase triangulation view plus the corrected tet-volume FHeron EOS/projection dynamics.",
            686,
            436,
            430,
            70,
            font_size=11,
            bold=True,
            color="17202A",
        )
    )
    shapes.append(panel(new_id(), 34, 600, 1182, 74))
    shapes.append(
        text_shape(
            new_id(),
            "IWT free-surface note",
            "#11/#12: visible phase mesh comes from GitHub droplet_in_box_3d. The stable pressure/volume dynamics use barycentric tet volumes and rebuilt V_t, B, m_t, V_t^tar after active retopology.",
            58,
            612,
            1128,
            32,
            font_size=11,
            bold=True,
            color="17202A",
        )
    )
    shapes.append(
        text_shape(
            new_id(),
            "IWT free-surface references",
            "Refs: Edelsbrunner (2001), Geometry and Topology for Mesh Generation, for Delaunay triangulation; Chorin (1968) and Chen & Przekwas (2010) for pressure/projection closures.",
            58,
            652,
            1128,
            14,
            font_size=8,
            color="5B6470",
        )
    )
    return xml.replace("</p:spTree>", "".join(shapes) + "</p:spTree>", 1)


def patch_pptx(input_path: Path, output_path: Path) -> None:
    if not MATHML2OMML.exists():
        raise FileNotFoundError(f"Missing Microsoft Word MathML converter: {MATHML2OMML}")
    xslt = etree.XSLT(etree.parse(str(MATHML2OMML)))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "patched.pptx"
        with zipfile.ZipFile(input_path, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "ppt/slides/slide3.xml":
                    data = patch_iwt_dual_volume_slide(data.decode("utf-8"), xslt).encode("utf-8")
                elif item.filename == "ppt/slides/slide4.xml":
                    data = patch_iwt_equation_slide(data.decode("utf-8"), xslt).encode("utf-8")
                elif item.filename == "ppt/slides/slide5.xml":
                    data = patch_iwt_full_flux_slide(data.decode("utf-8"), xslt).encode("utf-8")
                elif item.filename == "ppt/slides/slide6.xml":
                    data = patch_iwt_pairwise_pressure_slide(data.decode("utf-8"), xslt).encode("utf-8")
                elif item.filename == "ppt/slides/slide7.xml":
                    data = patch_iwt_free_surface_slide(data.decode("utf-8"), xslt).encode("utf-8")
                zout.writestr(item, data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tmp_path.read_bytes())


def main() -> int:
    base = Path(__file__).resolve().parent
    input_path = base / "outputs/manual-20260602-fheron-ppt/presentations/fheron-flux-cases/output/fheron-hc-twelve-cases-34slides-iwt-template.pptx"
    output_path = base / "outputs/manual-20260602-fheron-ppt/presentations/fheron-flux-cases/output/fheron-hc-twelve-cases-34slides-iwt-template-office-math.pptx"
    patch_pptx(input_path, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
