#!/usr/bin/env python3
"""Plot Figure 3-8 from generated recomputation result files.

This script does not read copied plotting CSVs. It expects each case folder to
contain a *_all_methods_result.csv file produced by its recomputation script.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime  # noqa: E402

ensure_runtime()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

RESULT_TXT = HERE / "Figure_3-8_VolumeOperatorComparison_result.txt"

STYLE_STATIC = HERE / "neatplot-main" / "standard.mplstyle"
STYLE_CUBE = STYLE_STATIC
STYLE_DROPLET = STYLE_STATIC

LABELS = {
    "plic_pl": "PLIC / PL (Rider & Kothe, 1998; Scardovelli & Zaleski, 1999)",
    "evrard_type_paraboloid": "Evrard-type paraboloid (Evrard et al., 2023)",
    "thinc_qq": "THINC/QQ quadratic (Xie & Xiao, 2017)",
    "strobl_sphere_overlap": "sphere/hex overlap (Strobl et al., 2016)",
    "present_quadric_patch": "Curved (Quadric Patch) Volume",
}

METHOD_ORDER = [
    ("plic_pl", "o-"),
    ("evrard_type_paraboloid", "^-"),
    ("thinc_qq", "s--"),
    ("strobl_sphere_overlap", "x:"),
    ("present_quadric_patch", "D-"),
]

STATIC_CASES = [
    ("a", "sphere", False),
    ("b", "cylinder", True),
    ("c", "paraboloid", True),
    ("d", "hyperboloid", True),
    ("e", "parabolic_cylinder", True),
    ("f", "hyperbolic_cylinder", True),
]

STATIC_XLABELS = {
    "sphere": "sphere surface triangles",
    "cylinder": "cylinder surface triangles",
    "paraboloid": "paraboloid-cap surface triangles",
    "hyperboloid": "hyperboloid surface triangles",
    "parabolic_cylinder": "parabolic-cylinder surface triangles",
    "hyperbolic_cylinder": "hyperbolic-cylinder surface triangles",
}

DYNAMIC_CASES = [
    ("g", "cube2sphere", STYLE_CUBE),
    ("h", "droplet_oscillation", STYLE_DROPLET),
]

FRAME_LINEWIDTH = 1.0
TICK_LINEWIDTH = 0.75
SAVE_PAD_INCHES = 0.01


def use_style(path: Path) -> None:
    if path.exists():
        plt.style.use(str(path))


def normalize_axes_frame(ax) -> None:
    """Keep all four spines and tick marks at the same visual weight."""
    for spine in ax.spines.values():
        spine.set_linewidth(FRAME_LINEWIDTH)
        spine.set_edgecolor("black")
    ax.tick_params(axis="both", which="both", width=TICK_LINEWIDTH)


def read_generated_csv(case: str) -> list[dict[str, str]]:
    path = HERE / case / f"{case}_all_methods_result.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run the recomputation script in {HERE / case} first."
        )
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def rows_for_plot(case: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if case == "parabolic_cylinder":
        table_idx = next(
            (i for i, row in enumerate(rows) if "Table 3-1" in row.get("label", "")),
            None,
        )
        if table_idx is not None:
            return rows[table_idx:]
    return rows


def values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def values_first_available(rows: list[dict[str, str]], columns: tuple[str, ...]) -> np.ndarray:
    for column in columns:
        if column in rows[0]:
            return values(rows, column)
    raise KeyError(f"None of these columns were found: {columns}")


def positive(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    pos = y[y > 0]
    if pos.size == 0:
        return y
    return np.where(y > 0, y, pos.min() * 0.5)


def plot_static(letter: str, case: str, mark_table: bool) -> str:
    use_style(STYLE_STATIC)
    rows = rows_for_plot(case, read_generated_csv(case))
    surface_triangles = values_first_available(rows, ("total_faces", "faces"))

    fig, ax = plt.subplots()
    for method, style in METHOD_ORDER:
        y = positive(values(rows, f"{method}_error_percent"))
        kwargs = {}
        if method == "thinc_qq":
            kwargs["markerfacecolor"] = "none"
        ax.loglog(surface_triangles, y, style, label=LABELS[method], **kwargs)

    if mark_table and any(row.get("label") for row in rows):
        idx = next(i for i, row in enumerate(rows) if row.get("label"))
        present = values(rows, "present_quadric_patch_error_percent")
        ax.loglog(
            surface_triangles[idx],
            max(present[idx], 1.0e-16),
            "o",
            markerfacecolor="none",
            markeredgecolor="k",
            label="_nolegend_",
        )

    pl = values(rows, "plic_pl_error_percent")
    ref = pl[-1] * (surface_triangles / surface_triangles[-1]) ** -1.0
    ax.loglog(surface_triangles, ref, "k--", label="_nolegend_")
    ax.set_xlabel(STATIC_XLABELS[case])
    ax.set_ylabel("relative volume error (%)")
    ax.xaxis.set_minor_formatter(NullFormatter())
    normalize_axes_frame(ax)
    if case == "sphere":
        legend = ax.legend(
            loc="center",
            bbox_to_anchor=(0.52, 0.47),
            borderaxespad=0.0,
            title="Shared legend for panels (a)-(f)",
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(legend.get_texts()[0].get_fontsize())

    out = HERE / f"Figure_3-8{letter}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return f"{out.name}: source={case}/{case}_all_methods_result.csv, rows={len(rows)}"


def plot_dynamic(letter: str, case: str, style_path: Path) -> str:
    use_style(style_path)
    rows = read_generated_csv(case)
    x = values(rows, "iter")

    fig, ax = plt.subplots()
    for method, _style in METHOD_ORDER:
        y = positive(values(rows, f"{method}_error_percent"))
        ax.plot(x, y, label=LABELS[method])
    ax.set_xlabel("iteration")
    ax.set_ylabel("|rel.error.V%|")
    ax.set_yscale("log")
    normalize_axes_frame(ax)
    ax.legend(loc="lower right")

    out = HERE / f"Figure_3-8{letter}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return f"{out.name}: source={case}/{case}_all_methods_result.csv, rows={len(rows)}"


def write_panel_pdf() -> str:
    pngs = [HERE / f"Figure_3-8{letter}.png" for letter in "abcdefgh"]
    out = HERE / "Figure_3-8.pdf"
    pdf = None
    for png in pngs:
        if not png.exists():
            raise FileNotFoundError(png)
        image = Image.open(png).convert("RGB")
        dpi = image.info.get("dpi", (600.0, 600.0))[0] or 600.0
        width_pt = image.width / dpi * 72.0
        height_pt = image.height / dpi * 72.0
        if pdf is None:
            pdf = canvas.Canvas(str(out), pagesize=(width_pt, height_pt))
            pdf.setTitle("Figure_3-8")
        else:
            pdf.setPageSize((width_pt, height_pt))
        pdf.drawImage(ImageReader(image), 0, 0, width=width_pt, height=height_pt)
        pdf.showPage()
    if pdf is None:
        raise RuntimeError("No Figure 3-8 panels were generated.")
    pdf.save()
    return f"{out.name}: pages=8"


def main() -> None:
    lines = [
        "Figure 3-8 volume-operator comparison panels plotted from generated recomputation outputs",
        "",
        "Generated files:",
    ]
    for letter, case, mark_table in STATIC_CASES:
        lines.append("- " + plot_static(letter, case, mark_table))
    for letter, case, style_path in DYNAMIC_CASES:
        lines.append("- " + plot_dynamic(letter, case, style_path))
    lines.append("- " + write_panel_pdf())

    RESULT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
