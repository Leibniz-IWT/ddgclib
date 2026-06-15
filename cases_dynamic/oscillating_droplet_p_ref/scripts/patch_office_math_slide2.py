from __future__ import annotations

import html
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from lxml import etree


EMU_PER_PX = 9525
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS_MATHML = "http://www.w3.org/1998/Math/MathML"
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
MATHML2OMML = Path("/Applications/Microsoft Word.app/Contents/Resources/mathml2omml.xsl")
UNIFORM_EQUATION_FONT_SIZE = 10


def emu(value: float) -> int:
    return int(round(value * EMU_PER_PX))


def root_with_math_namespaces(xml: str) -> str:
    original = f'<p:sld xmlns:p="{NS_P}">'
    updated = (
        f'<p:sld xmlns:p="{NS_P}" '
        f'xmlns:a="{NS_A}" '
        f'xmlns:a14="{NS_A14}" '
        f'xmlns:m="{NS_M}" '
        f'xmlns:mc="{NS_MC}" '
        f'mc:Ignorable="a14">'
    )
    return xml.replace(original, updated, 1)


def max_shape_id(xml: str) -> int:
    values = [int(value) for value in re.findall(r'<p:cNvPr id="(\d+)"', xml)]
    return max(values) if values else 1


def fill_xml(fill: str | None) -> str:
    if fill is None:
        return '<a:noFill/>'
    return f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'


def line_xml(color: str | None = "D9DEE6", width: int = 1) -> str:
    if color is None or width <= 0:
        return '<a:ln w="0"><a:noFill/></a:ln>'
    return (
        f'<a:ln w="{width * 12700}">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:prstDash val="solid"/></a:ln>'
    )


def shape_start(shape_id: int, name: str, x: float, y: float, w: float, h: float, fill: str | None, line: str | None) -> str:
    return (
        '<p:sp>'
        '<p:nvSpPr>'
        f'<p:cNvPr id="{shape_id}" name="{html.escape(name)}"/>'
        '<p:cNvSpPr txBox="1"/>'
        '<p:nvPr/>'
        '</p:nvSpPr>'
        '<p:spPr>'
        '<a:xfrm>'
        f'<a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/>'
        '</a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'{fill_xml(fill)}'
        f'{line_xml(line)}'
        '</p:spPr>'
    )


def shape_end() -> str:
    return '</p:sp>'


def text_shape(shape_id: int, name: str, text: str, x: float, y: float, w: float, h: float, *,
               font_size: int = 16, bold: bool = False, color: str = "17202A",
               fill: str | None = None, line: str | None = None) -> str:
    bold_attr = ' b="1"' if bold else ""
    return (
        shape_start(shape_id, name, x, y, w, h, fill, line)
        +
        '<p:txBody><a:bodyPr wrap="square"/><a:lstStyle/><a:p>'
        '<a:r>'
        f'<a:rPr lang="en-US" sz="{font_size * 100}"{bold_attr}>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:latin typeface="Aptos"/><a:ea typeface="Aptos"/><a:cs typeface="Aptos"/>'
        '</a:rPr>'
        f'<a:t>{html.escape(text)}</a:t>'
        '</a:r>'
        '<a:endParaRPr lang="en-US"/>'
        '</a:p></p:txBody>'
        + shape_end()
    )


def math(body: str) -> str:
    return f'<math xmlns="{NS_MATHML}">{body}</math>'


def ppt_run_properties(size: int, color: str = "17202A") -> etree._Element:
    rpr = etree.Element(f"{{{NS_A}}}rPr")
    rpr.set("lang", "en-US")
    rpr.set("sz", str(size * 100))
    solid = etree.SubElement(rpr, f"{{{NS_A}}}solidFill")
    srgb = etree.SubElement(solid, f"{{{NS_A}}}srgbClr")
    srgb.set("val", color)
    etree.SubElement(rpr, f"{{{NS_A}}}latin").set("typeface", "Cambria Math")
    etree.SubElement(rpr, f"{{{NS_A}}}ea").set("typeface", "Cambria Math")
    etree.SubElement(rpr, f"{{{NS_A}}}cs").set("typeface", "Cambria Math")
    return rpr


def add_powerpoint_run_properties(omath: etree._Element, size: int) -> None:
    for text_node in omath.xpath(".//m:t", namespaces={"m": NS_M}):
        if text_node.text is not None and text_node.text.strip() != text_node.text:
            text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for run in omath.xpath(".//m:r", namespaces={"m": NS_M}):
        if run.find(f"{{{NS_A}}}rPr") is not None:
            continue
        insert_at = 0
        if len(run) > 0 and run[0].tag == f"{{{NS_M}}}rPr":
            insert_at = 1
        run.insert(insert_at, ppt_run_properties(size))


def omml_from_mathml(xslt: etree.XSLT, equation_mathml: str, *, size: int) -> str:
    node = etree.fromstring(equation_mathml.encode("utf-8"))
    result = xslt(node)
    omath = etree.fromstring(str(result).encode("utf-8"))
    add_powerpoint_run_properties(omath, size)

    wrapper = etree.Element(f"{{{NS_A14}}}m", nsmap={"a14": NS_A14})
    para = etree.SubElement(wrapper, f"{{{NS_M}}}oMathPara", nsmap={"m": NS_M})
    para_pr = etree.SubElement(para, f"{{{NS_M}}}oMathParaPr")
    jc = etree.SubElement(para_pr, f"{{{NS_M}}}jc")
    jc.set(f"{{{NS_M}}}val", "left")
    para.append(omath)
    return etree.tostring(wrapper, encoding="unicode")


def math_paragraph(omml: str) -> str:
    return (
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="105000"/></a:lnSpc></a:pPr>'
        f'{omml}'
        '</a:p>'
    )


def math_shape(shape_id: int, name: str, omml: str, x: float, y: float, w: float, h: float, *,
               font_size: int = 18, fill: str | None = None, line: str | None = None) -> str:
    shape = (
        shape_start(shape_id, name, x, y, w, h, fill, line)
        +
        '<p:txBody><a:bodyPr wrap="square" lIns="254" tIns="254" rIns="254" bIns="254" anchor="t"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
        f'{math_paragraph(omml)}'
        '</p:txBody>'
        + shape_end()
    )
    return shape


def panel(shape_id: int, x: float, y: float, w: float, h: float) -> str:
    return shape_start(shape_id, "Office math panel", x, y, w, h, "FFFFFF", "D9DEE6") + shape_end()


def equation_mathml() -> list[tuple[str, str, int, str]]:
    force_eqs = [
        (
            "force 1",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">tot</mi></msubsup>
                  <mo>=</mo>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">Heron</mi></msubsup>
                  <mo>+</mo>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">p</mi></msubsup>
                  <mo>+</mo>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">damp</mi></msubsup>
                  <mo>+</mo>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">flux</mi></msubsup>
                </mrow>
                """
            ),
            14,
            "left",
        ),
        (
            "force 2",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">Heron</mi></msubsup>
                  <mo>=</mo>
                  <mo>-</mo><mi>γ</mi>
                  <msub>
                    <mrow><mo stretchy="false">(</mo><mi mathvariant="bold">HN</mi><mo>&#160;</mo><mi mathvariant="normal">dA</mi><mo stretchy="false">)</mo></mrow>
                    <mi>i</mi>
                  </msub>
                </mrow>
                """
            ),
            14,
            "left",
        ),
        (
            "force 3",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">p</mi></msubsup>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>t</mi><mo>∋</mo><mi>i</mi></mrow></msub>
                  <msub><mi>p</mi><mi>t</mi></msub>
                  <msub><mi mathvariant="bold">B</mi><mrow><mi>t</mi><mo>,</mo><mi>i</mi></mrow></msub>
                  <mo>,</mo><mo>&#160;</mo>
                  <msub><mi mathvariant="bold">B</mi><mrow><mi>t</mi><mo>,</mo><mi>i</mi></mrow></msub>
                  <mo>=</mo>
                  <mfrac>
                    <mrow><mi>∂</mi><msub><mi>V</mi><mi>t</mi></msub></mrow>
                    <mrow><mi>∂</mi><msub><mi mathvariant="bold">x</mi><mi>i</mi></msub></mrow>
                  </mfrac>
                </mrow>
                """
            ),
            10,
            "left",
        ),
        (
            "force 4",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">damp</mi></msubsup>
                  <mo>=</mo>
                  <mo>-</mo><msub><mi>α</mi><mi>d</mi></msub><msub><mi>m</mi><mi>i</mi></msub><msub><mi mathvariant="bold">u</mi><mi>i</mi></msub>
                  <mo>,</mo><mo>&#160;</mo>
                  <msub><mi>α</mi><mi>d</mi></msub><mo>=</mo><mn>2</mn><mi>ζ</mi><msub><mi>ω</mi><mi>R</mi></msub>
                </mrow>
                """
            ),
            12,
            "left",
        ),
        (
            "force 5",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">flux</mi></msubsup>
                  <mo>=</mo>
                  <mfrac><mn>1</mn><mn>4</mn></mfrac>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>t</mi><mo>∋</mo><mi>i</mi></mrow></msub>
                  <msub><mi>λ</mi><mi>F</mi></msub>
                  <msub><mi>s</mi><mi mathvariant="normal">lim</mi></msub>
                  <msub><mi mathvariant="bold">C</mi><mi>t</mi></msub>
                </mrow>
                """
            ),
            16,
            "left",
        ),
        (
            "force 6",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">u</mi><mi>i</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msubsup>
                  <mo>=</mo>
                  <msubsup><mi mathvariant="bold">u</mi><mi>i</mi><mi>n</mi></msubsup>
                  <mo>+</mo><mi>Δ</mi><mi>t</mi>
                  <mfrac>
                    <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">tot</mi></msubsup>
                    <msub><mi>m</mi><mi>i</mi></msub>
                  </mfrac>
                  <mo>,</mo><mo>&#160;</mo>
                  <msubsup><mi mathvariant="bold">x</mi><mi>i</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msubsup>
                  <mo>=</mo>
                  <msubsup><mi mathvariant="bold">x</mi><mi>i</mi><mi>n</mi></msubsup>
                  <mo>+</mo><mi>Δ</mi><mi>t</mi>
                  <msubsup><mi mathvariant="bold">u</mi><mi>i</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msubsup>
                </mrow>
                """
            ),
            10,
            "left",
        ),
    ]
    pressure_eqs = [
        (
            "closure 7",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mrow><mi mathvariant="normal">p</mi><mo>,</mo><mi mathvariant="normal">comp</mi></mrow></msubsup>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>t</mi><mo>∋</mo><mi>i</mi></mrow></msub>
                  <mo stretchy="false">(</mo>
                  <msub><mi>p</mi><mi mathvariant="normal">ref</mi></msub>
                  <mo>+</mo>
                  <mi>δ</mi><msub><mi>p</mi><mi>t</mi></msub>
                  <mo stretchy="false">)</mo>
                  <msub><mi mathvariant="bold">B</mi><mrow><mi>t</mi><mo>,</mo><mi>i</mi></mrow></msub>
                </mrow>
                """
            ),
            8,
            "right_top",
        ),
        (
            "closure 0",
            math(
                """
                <mrow>
                  <msub><mi>p</mi><mi mathvariant="normal">ref</mi></msub>
                  <mo>=</mo>
                  <mo>-</mo>
                  <mfrac>
                    <mrow>
                      <msub><mi mathvariant="normal">∑</mi><mi>i</mi></msub>
                      <msub><mi mathvariant="bold">A</mi><mi>i</mi></msub>
                      <mo>·</mo>
                      <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mi mathvariant="normal">Heron</mi></msubsup>
                    </mrow>
                    <mrow>
                      <msub><mi mathvariant="normal">∑</mi><mi>i</mi></msub>
                      <msub><mi mathvariant="bold">A</mi><mi>i</mi></msub>
                      <mo>·</mo>
                      <msub><mi mathvariant="bold">A</mi><mi>i</mi></msub>
                    </mrow>
                  </mfrac>
                </mrow>
                """
            ),
            9,
            "right_top",
        ),
        (
            "closure 1",
            math(
                """
                <mrow>
                  <msub><mi>ρ</mi><mi>t</mi></msub>
                  <mo>=</mo>
                  <mfrac><msub><mi>m</mi><mi>t</mi></msub><msub><mi>V</mi><mi>t</mi></msub></mfrac>
                  <mo>,</mo><mo>&#160;</mo>
                  <msubsup><mi>V</mi><mi>t</mi><mi mathvariant="normal">tar</mi></msubsup>
                  <mo>=</mo>
                  <mfrac><msub><mi>m</mi><mi>t</mi></msub><msub><mi>ρ</mi><mn>0</mn></msub></mfrac>
                </mrow>
                """
            ),
            9,
            "right_top",
        ),
        (
            "closure 2",
            math(
                """
                <mrow>
                  <msub><mi mathvariant="bold">B</mi><mrow><mi>t</mi><mo>,</mo><mi>i</mi></mrow></msub>
                  <mo>=</mo>
                  <mfrac>
                    <mrow><mi>∂</mi><msub><mi>V</mi><mi>t</mi></msub></mrow>
                    <mrow><mi>∂</mi><msub><mi mathvariant="bold">x</mi><mi>i</mi></msub></mrow>
                  </mfrac>
                  <mo>,</mo><mo>&#160;</mo>
                  <msub><mi>S</mi><mrow><mi>t</mi><mo>,</mo><mi>s</mi></mrow></msub>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mi>i</mi></msub>
                  <mfrac>
                    <mrow>
                      <msub><mi mathvariant="bold">B</mi><mrow><mi>t</mi><mo>,</mo><mi>i</mi></mrow></msub>
                      <mo>·</mo>
                      <msub><mi mathvariant="bold">B</mi><mrow><mi>s</mi><mo>,</mo><mi>i</mi></mrow></msub>
                    </mrow>
                    <msub><mi>m</mi><mi>i</mi></msub>
                  </mfrac>
                  <mo>,</mo><mo>&#160;</mo>
                  <mi mathvariant="bold">S</mi>
                  <mo>=</mo>
                  <mi mathvariant="bold">B</mi>
                  <msup><mi mathvariant="bold">M</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
                  <msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup>
                </mrow>
                """
            ),
            7,
            "right_top",
        ),
        (
            "closure 3",
            math(
                """
                <mrow>
                  <mo stretchy="false">[</mo>
                  <mi mathvariant="bold">I</mi>
                  <mo>+</mo>
                  <msup><mrow><mi>Δ</mi><mi>t</mi></mrow><mn>2</mn></msup>
                  <msub><mi mathvariant="bold">D</mi><mi>K</mi></msub><mi mathvariant="bold">S</mi>
                  <mo stretchy="false">]</mo>
                  <mi>δ</mi><mi mathvariant="bold">p</mi>
                  <mo>=</mo>
                  <mo>-</mo>
                  <msub><mi mathvariant="bold">D</mi><mi>K</mi></msub>
                  <mo stretchy="false">(</mo>
                  <mi mathvariant="bold">V</mi>
                  <mo>-</mo>
                  <msup><mi mathvariant="bold">V</mi><mi mathvariant="normal">tar</mi></msup>
                  <mo>+</mo><mi>Δ</mi><mi>t</mi><mi mathvariant="bold">B</mi><msup><mi mathvariant="bold">u</mi><mo>*</mo></msup>
                  <mo stretchy="false">)</mo>
                </mrow>
                """
            ),
            8,
            "right_top",
        ),
        (
            "closure 4",
            math(
                """
                <mrow>
                  <mi mathvariant="bold">S</mi><mi mathvariant="bold">p</mi>
                  <mo>=</mo>
                  <mfrac>
                    <mrow><mi mathvariant="bold">r</mi><mo>-</mo><mi mathvariant="bold">B</mi><mi mathvariant="bold">u</mi></mrow>
                    <mrow><mi>Δ</mi><mi>t</mi></mrow>
                  </mfrac>
                  <mo>-</mo>
                  <mi mathvariant="bold">B</mi>
                  <msup><mi mathvariant="bold">M</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
                  <msub><mi mathvariant="bold">F</mi><mi mathvariant="normal">np</mi></msub>
                  <mo>,</mo><mo>&#160;</mo>
                  <mi mathvariant="bold">r</mi>
                  <mo>=</mo>
                  <mo>-</mo>
                  <mfrac>
                    <mrow><mi mathvariant="bold">V</mi><mo>-</mo><msup><mi mathvariant="bold">V</mi><mi mathvariant="normal">tar</mi></msup></mrow>
                    <mrow><mi>Δ</mi><mi>t</mi></mrow>
                  </mfrac>
                </mrow>
                """
            ),
            8,
            "right_top",
        ),
        (
            "closure 5",
            math(
                """
                <mrow>
                  <mi mathvariant="bold">B</mi>
                  <msup><mi mathvariant="bold">u</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msup>
                  <mo>=</mo>
                  <mi mathvariant="bold">r</mi>
                </mrow>
                """
            ),
            11,
            "right_top",
        ),
        (
            "closure 8",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">F</mi><mi>i</mi><mrow><mi mathvariant="normal">p</mi><mo>,</mo><mi mathvariant="normal">inc</mi></mrow></msubsup>
                  <mo>=</mo>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>t</mi><mo>∋</mo><mi>i</mi></mrow></msub>
                  <msubsup><mi>p</mi><mi>t</mi><mi mathvariant="normal">inc</mi></msubsup>
                  <msub><mi mathvariant="bold">B</mi><mrow><mi>t</mi><mo>,</mo><mi>i</mi></mrow></msub>
                </mrow>
                """
            ),
            8,
            "right_top",
        ),
        (
            "closure 6",
            math(
                """
                <mrow>
                  <msub><mi mathvariant="bold">D</mi><mi>K</mi></msub>
                  <mo>=</mo>
                  <mi mathvariant="normal">diag</mi>
                  <mo stretchy="false">(</mo>
                  <mfrac><mi>K</mi><msubsup><mi>V</mi><mi>t</mi><mn>0</mn></msubsup></mfrac>
                  <mo stretchy="false">)</mo>
                  <mo>,</mo><mo>&#160;</mo>
                  <msub><mi>p</mi><mi>t</mi></msub>
                  <mo>=</mo>
                  <msub><mi>p</mi><mi mathvariant="normal">ref</mi></msub>
                  <mo>+</mo>
                  <mi>δ</mi><msub><mi>p</mi><mi>t</mi></msub>
                </mrow>
                """
            ),
            8,
            "right_top",
        ),
    ]
    flux_eqs = [
        (
            "flux 1",
            math(
                """
                <mrow>
                  <msub><mi>Φ</mi><mrow><mi>m</mi><mo>,</mo><mi>f</mi></mrow></msub>
                  <mo>=</mo>
                  <msup><mi>ρ</mi><mi mathvariant="normal">up</mi></msup>
                  <msub><mi>q</mi><mi>f</mi></msub>
                  <mo>,</mo><mo>&#160;</mo>
                  <msub><mi>q</mi><mi>f</mi></msub>
                  <mo>=</mo>
                  <mo stretchy="false">(</mo>
                  <msub><mi mathvariant="bold">u</mi><mi>f</mi></msub>
                  <mo>-</mo><msub><mi mathvariant="bold">w</mi><mi>f</mi></msub>
                  <mo stretchy="false">)</mo>
                  <mo>·</mo><msub><mi mathvariant="bold">A</mi><mi>f</mi></msub>
                </mrow>
                """
            ),
            10,
            "right_bottom",
        ),
        (
            "flux 2",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">Π</mi><mi>f</mi><mi mathvariant="normal">up</mi></msubsup>
                  <mo>=</mo>
                  <msubsup><mi>Φ</mi><mrow><mi>m</mi><mo>,</mo><mi>f</mi></mrow><mi mathvariant="normal">up</mi></msubsup>
                  <msup><mi mathvariant="bold">u</mi><mi mathvariant="normal">up</mi></msup>
                </mrow>
                """
            ),
            10,
            "right_bottom",
        ),
        (
            "flux 7",
            math(
                """
                <mrow>
                  <msub><mi mathvariant="bold">u</mi><mi>f</mi></msub>
                  <mo>=</mo>
                  <msub><mi mathvariant="bold">w</mi><mi>f</mi></msub>
                  <mo>=</mo>
                  <mfrac><mn>1</mn><mn>3</mn></mfrac>
                  <msub><mi mathvariant="normal">∑</mi><mrow><mi>i</mi><mo>∈</mo><mi>f</mi></mrow></msub>
                  <msub><mi mathvariant="bold">u</mi><mi>i</mi></msub>
                  <mo>⇒</mo>
                  <msub><mi>Φ</mi><mrow><mi>m</mi><mo>,</mo><mi>f</mi></mrow></msub>
                  <mo>=</mo>
                  <mn>0</mn>
                </mrow>
                """
            ),
            10,
            "right_bottom",
        ),
        (
            "flux 3",
            math(
                """
                <mrow>
                  <msubsup><mi>Φ</mi><mrow><mi>m</mi><mo>,</mo><mi>f</mi></mrow><mi mathvariant="normal">R</mi></msubsup>
                  <mo>=</mo>
                  <mo stretchy="false">|</mo><mi mathvariant="bold">A</mi><mo stretchy="false">|</mo>
                  <mo stretchy="false">[</mo>
                  <mfrac><mn>1</mn><mn>2</mn></mfrac>
                  <mo stretchy="false">(</mo><msub><mi>ρ</mi><mi mathvariant="normal">L</mi></msub><msub><mi>a</mi><mi mathvariant="normal">L</mi></msub><mo>+</mo><msub><mi>ρ</mi><mi mathvariant="normal">R</mi></msub><msub><mi>a</mi><mi mathvariant="normal">R</mi></msub><mo stretchy="false">)</mo>
                  <mo>-</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><msub><mi>c</mi><mi>f</mi></msub>
                  <mo stretchy="false">(</mo><msub><mi>ρ</mi><mi mathvariant="normal">R</mi></msub><mo>-</mo><msub><mi>ρ</mi><mi mathvariant="normal">L</mi></msub><mo stretchy="false">)</mo>
                  <mo stretchy="false">]</mo>
                </mrow>
                """
            ),
            8,
            "right_bottom",
        ),
        (
            "flux 4",
            math(
                """
                <mrow>
                  <msubsup><mi mathvariant="bold">Π</mi><mi>f</mi><mi mathvariant="normal">R</mi></msubsup>
                  <mo>=</mo>
                  <mo stretchy="false">|</mo><mi mathvariant="bold">A</mi><mo stretchy="false">|</mo>
                  <mo stretchy="false">[</mo>
                  <mfrac><mn>1</mn><mn>2</mn></mfrac>
                  <mo stretchy="false">(</mo><msub><mi>ρ</mi><mi mathvariant="normal">L</mi></msub><msub><mi mathvariant="bold">u</mi><mi mathvariant="normal">L</mi></msub><msub><mi>a</mi><mi mathvariant="normal">L</mi></msub><mo>+</mo><msub><mi>ρ</mi><mi mathvariant="normal">R</mi></msub><msub><mi mathvariant="bold">u</mi><mi mathvariant="normal">R</mi></msub><msub><mi>a</mi><mi mathvariant="normal">R</mi></msub><mo stretchy="false">)</mo>
                  <mo>-</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><msub><mi>c</mi><mi>f</mi></msub>
                  <mo stretchy="false">(</mo><msub><mi>ρ</mi><mi mathvariant="normal">R</mi></msub><msub><mi mathvariant="bold">u</mi><mi mathvariant="normal">R</mi></msub><mo>-</mo><msub><mi>ρ</mi><mi mathvariant="normal">L</mi></msub><msub><mi mathvariant="bold">u</mi><mi mathvariant="normal">L</mi></msub><mo stretchy="false">)</mo>
                  <mo stretchy="false">]</mo>
                </mrow>
                """
            ),
            7,
            "right_bottom",
        ),
        (
            "flux 5",
            math(
                """
                <mrow>
                  <msub><mover><mi>m</mi><mo>˙</mo></mover><mrow><mi>L</mi><mo>,</mo><mi>f</mi></mrow></msub>
                  <mo>=</mo>
                  <mo>-</mo>
                  <msub><mi>Φ</mi><mrow><mi>m</mi><mo>,</mo><mi>f</mi></mrow></msub>
                  <mo>,</mo><mo>&#160;</mo>
                  <msub><mover><mi>m</mi><mo>˙</mo></mover><mrow><mi>R</mi><mo>,</mo><mi>f</mi></mrow></msub>
                  <mo>=</mo>
                  <mo>+</mo>
                  <msub><mi>Φ</mi><mrow><mi>m</mi><mo>,</mo><mi>f</mi></mrow></msub>
                </mrow>
                """
            ),
            10,
            "right_bottom",
        ),
        (
            "flux 6",
            math(
                """
                <mrow>
                  <msubsup><mi>m</mi><mi>t</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msubsup>
                  <mo>=</mo>
                  <mi mathvariant="normal">max</mi>
                  <mo stretchy="false">(</mo>
                  <msub><mi>m</mi><mi mathvariant="normal">floor</mi></msub>
                  <mo>,</mo>
                  <msubsup><mi>m</mi><mi>t</mi><mi>n</mi></msubsup>
                  <mo>+</mo>
                  <mi>Δ</mi><mi>t</mi><msub><mi>λ</mi><mi>m</mi></msub><msub><mi>s</mi><mi mathvariant="normal">lim</mi></msub>
                  <msub><mover><mi>m</mi><mo>˙</mo></mover><mi>t</mi></msub>
                  <mo stretchy="false">)</mo>
                </mrow>
                """
            ),
            9,
            "right_bottom",
        ),
    ]
    return force_eqs + pressure_eqs + flux_eqs


def patch_slide2(xml: str, xslt: etree.XSLT) -> str:
    xml = root_with_math_namespaces(xml)
    next_id = max_shape_id(xml) + 1
    equations = {
        name: (omml_from_mathml(xslt, eq, size=UNIFORM_EQUATION_FONT_SIZE), group)
        for name, eq, size, group in equation_mathml()
    }

    shapes: list[str] = []

    def new_id() -> int:
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    # Cover the previous equation text/cards while preserving the slide header/footer.
    shapes.append(shape_start(new_id(), "Office math content cover", 44, 150, 1192, 510, "F7F8FA", None) + shape_end())

    left_x, right_x = 52, 656
    top_y, bottom_y = 154, 412
    panel_w, top_h, bottom_h = 576, 246, 234
    shapes.append(panel(new_id(), left_x, top_y, panel_w, top_h))
    shapes.append(panel(new_id(), right_x, top_y, panel_w, top_h))
    shapes.append(panel(new_id(), left_x, bottom_y, panel_w, bottom_h))
    shapes.append(panel(new_id(), right_x, bottom_y, panel_w, bottom_h))

    labels = {
        "force 1": "component split",
        "force 2": "HC Heron force",
        "force 3": "uses volume gradient",
        "force 4": "damping force (#5/#6 off)",
        "force 5": "flux force",
        "force 6": "time integration",
        "closure 0": "reference pressure",
        "closure 1": "density target",
        "closure 2": "stiffness entry",
        "closure 3": "compressible solve",
        "closure 4": "incomp pressure solve",
        "closure 5": "velocity constraint",
        "closure 6": "EOS pressure",
        "closure 7": "EOS pressure force",
        "closure 8": "projection pressure force",
        "flux 1": "material face flux",
        "flux 2": "upwind momentum",
        "flux 7": "Lagrangian face",
        "flux 3": "Rusanov mass flux",
        "flux 4": "Rusanov momentum",
        "flux 5": "face mass source",
        "flux 6": "limited mass update",
    }

    def add_rows(names: list[str], x: float, y: float, *, row0: float, step: float, eq_w: float, eq_h: float,
                 label_x: float, label_w: float, label_size: int = 8) -> None:
        for idx, name in enumerate(names):
            row_y = y + row0 + idx * step
            shapes.append(math_shape(new_id(), f"Office equation {name}", equations[name][0], x + 28, row_y, eq_w, eq_h))
            shapes.append(text_shape(
                new_id(),
                f"Equation label {name}",
                labels[name],
                x + label_x,
                row_y + 6,
                label_w,
                18,
                font_size=label_size,
                color="5B6470",
            ))

    shapes.append(text_shape(new_id(), "Force component title", "Common force/update", left_x + 22, top_y + 18, 360, 24, font_size=18, bold=True))
    add_rows(
        ["force 1", "force 2", "force 4", "force 6"],
        left_x,
        top_y,
        row0=52,
        step=42,
        eq_w=396,
        eq_h=32,
        label_x=438,
        label_w=118,
        label_size=8,
    )

    shapes.append(text_shape(new_id(), "Compressible title", "Compressible EOS closure", right_x + 22, top_y + 18, 390, 24, font_size=18, bold=True))
    add_rows(
        ["closure 7", "closure 0", "closure 1", "closure 2", "closure 6", "closure 3"],
        right_x,
        top_y,
        row0=58,
        step=31,
        eq_w=398,
        eq_h=30,
        label_x=440,
        label_w=118,
        label_size=8,
    )

    shapes.append(text_shape(new_id(), "ALE flux title", "ALE flux and mass update", left_x + 22, bottom_y + 18, 390, 24, font_size=18, bold=True))
    add_rows(
        ["flux 1", "flux 7", "flux 2", "flux 5", "flux 6"],
        left_x,
        bottom_y,
        row0=47,
        step=34,
        eq_w=398,
        eq_h=30,
        label_x=438,
        label_w=118,
        label_size=7,
    )
    shapes.append(text_shape(
        new_id(),
        "ALE flux note",
        "#3-#6: material-face ALE; #5/#6 use lambda_m = lambda_F = 1. Material face gives Phi_m,f = Pi_f = 0.",
        left_x + 28,
        bottom_y + 220,
        505,
        16,
        font_size=7,
        color="5B6470",
    ))

    shapes.append(text_shape(new_id(), "Incompressible title", "Incompressible projection", right_x + 22, bottom_y + 18, 390, 24, font_size=18, bold=True))
    add_rows(
        ["closure 8", "closure 2", "closure 4", "closure 5"],
        right_x,
        bottom_y,
        row0=50,
        step=42,
        eq_w=398,
        eq_h=34,
        label_x=440,
        label_w=118,
        label_size=8,
    )

    shapes.append(text_shape(
        new_id(),
        "Projection reference",
        "Projection ref: Chorin (1968), Numerical solution of the Navier-Stokes equations, Math. Comput. 22(104), 745-762.",
        84,
        648,
        1060,
        16,
        font_size=8,
        color="5B6470",
    ))
    shapes.append(text_shape(
        new_id(),
        "Compressible pressure EOS reference",
        "Compressible pressure/EOS ref: Chen & Przekwas (2010), A coupled pressure-based computational method for incompressible/compressible flows, JCP 229(24), 9150-9165.",
        84,
        664,
        1060,
        16,
        font_size=8,
        color="5B6470",
    ))

    overlay_xml = "".join(shapes)
    return xml.replace("</p:spTree>", f"{overlay_xml}</p:spTree>", 1)


CASE_DAMPING_SUBTITLES = {
    "ppt/slides/slide5.xml": (
        "Incompressible local projection | lambda_m = 0; lambda_F = 0; material-face flux diagnostic",
        "Incompressible local projection | lambda_m = 0; lambda_F = 0; material-face flux diagnostic | F_damp = 0; inertia scale = 1.08",
    ),
    "ppt/slides/slide6.xml": (
        "Weak-compressible EOS | lambda_m = 0; lambda_F = 0; material-face flux diagnostic",
        "Weak-compressible EOS | lambda_m = 0; lambda_F = 0; material-face flux diagnostic | F_damp = 0; inertia scale = 1.08",
    ),
    "ppt/slides/slide7.xml": (
        "Incompressible local projection | lambda_m = 1e-6; lambda_F = 0; material-face flux",
        "Incompressible local projection | lambda_m = 1e-6; lambda_F = 0; material-face flux | F_damp = 0; inertia scale = 1.08",
    ),
    "ppt/slides/slide8.xml": (
        "Weak-compressible EOS | lambda_m = 1e-6; lambda_F = 0; material-face flux",
        "Weak-compressible EOS | lambda_m = 1e-6; lambda_F = 0; material-face flux | F_damp = 0; inertia scale = 1.08",
    ),
    "ppt/slides/slide9.xml": (
        "Weak-compressible EOS + Lagrangian full flux | lambda_m = 1; lambda_F = 1; material-face flux; no rho limiter",
        "Weak-compressible EOS + Lagrangian full flux | lambda_m = 1; lambda_F = 1; material-face flux; no rho limiter | F_damp = 0; inertia scale = 1.08",
    ),
    "ppt/slides/slide10.xml": (
        "Incompressible projection + Lagrangian full flux | lambda_m = 1; lambda_F = 1; material-face flux; no rho limiter",
        "Incompressible projection + Lagrangian full flux | lambda_m = 1; lambda_F = 1; material-face flux; no rho limiter | F_damp = 0; inertia scale = 1.08",
    ),
}


def patch_case_slide_text(xml: str, filename: str) -> str:
    old, new = CASE_DAMPING_SUBTITLES[filename]
    if new in xml:
        return xml
    return xml.replace(old, new, 1)


def patch_slide11_text(xml: str) -> str:
    old = (
        "Displayed #5 and #6 use F_damp = 0, inertia scale = 1.08, subdivision 1, lambda_m = lambda_F = 1, and no rho limiter."
    )
    new = (
        "Key point: zero internal flux is correct for this moving-vertex Lagrangian droplet. Nonzero flux needs an ALE/remap or Eulerian FV solver."
    )
    xml = xml.replace(old, new, 1)
    old2 = (
        "Displayed #1-#6 use F_damp = 0, inertia scale = 1.08, subdivision 1, material-face flux where evaluated, and no rho limiter."
    )
    return xml.replace(old2, new, 1)


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
                    xml = data.decode("utf-8")
                    data = patch_slide2(xml, xslt).encode("utf-8")
                elif item.filename in CASE_DAMPING_SUBTITLES:
                    xml = data.decode("utf-8")
                    data = patch_case_slide_text(xml, item.filename).encode("utf-8")
                elif item.filename == "ppt/slides/slide12.xml":
                    xml = data.decode("utf-8")
                    data = patch_slide11_text(xml).encode("utf-8")
                zout.writestr(item, data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_path, output_path)


def main() -> int:
    base = Path(__file__).resolve().parent
    input_path = base / "outputs/manual-20260602-fheron-ppt/presentations/fheron-flux-cases/output/fheron-hc-six-cases-12slides.pptx"
    output_path = base / "outputs/manual-20260602-fheron-ppt/presentations/fheron-flux-cases/output/fheron-hc-six-cases-12slides-office-math.pptx"
    patch_pptx(input_path, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
