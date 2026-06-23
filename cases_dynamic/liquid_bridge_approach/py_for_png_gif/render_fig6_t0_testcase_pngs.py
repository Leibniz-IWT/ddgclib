#!/usr/bin/env python3
"""Render the Fig. 6 t0 approach mesh with the packaged display scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


APPROACH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH = APPROACH_ROOT / "fig6_t0_finial" / "mesh" / "fig6_approach_t0.msh"
DEFAULT_OUT_DIR = APPROACH_ROOT / "fig6_t0_finial" / "display_pngs_from_testcase"
SCRIPT_DIR = Path(__file__).resolve().parent


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser


def run_renderer(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _scaled_image(path: Path, *, max_w: int, max_h: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def write_overview(out_dir: Path) -> None:
    from PIL import Image, ImageDraw

    items = [
        ("Side / XZ view", _scaled_image(out_dir / "mesh_xz.png", max_w=1120, max_h=560)),
        ("Volume cutaway", _scaled_image(out_dir / "mesh_xyz.png", max_w=540, max_h=420)),
        ("Top / XYZ view", _scaled_image(out_dir / "mesh_top.png", max_w=540, max_h=420)),
    ]
    pad = 18
    label_h = 26
    width = max(items[0][1].width + 2 * pad, items[1][1].width + items[2][1].width + 3 * pad)
    height = label_h + items[0][1].height + label_h + max(items[1][1].height, items[2][1].height) + 3 * pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(15)
    title_color = (170, 0, 40)
    border_color = (218, 218, 218)

    x = pad
    y = pad
    draw.text((x, y - 2), items[0][0], fill=title_color, font=title_font)
    canvas.paste(items[0][1], (x, y + label_h))
    draw.rectangle((x, y + label_h, x + items[0][1].width - 1, y + label_h + items[0][1].height - 1), outline=border_color)

    y2 = y + label_h + items[0][1].height + pad
    x1 = pad
    x2 = pad * 2 + items[1][1].width
    draw.text((x1, y2 - 2), items[1][0], fill=title_color, font=title_font)
    canvas.paste(items[1][1], (x1, y2 + label_h))
    draw.rectangle((x1, y2 + label_h, x1 + items[1][1].width - 1, y2 + label_h + items[1][1].height - 1), outline=border_color)
    draw.text((x2, y2 - 2), items[2][0], fill=title_color, font=title_font)
    canvas.paste(items[2][1], (x2, y2 + label_h))
    draw.rectangle((x2, y2 + label_h, x2 + items[2][1].width - 1, y2 + label_h + items[2][1].height - 1), outline=border_color)

    canvas.save(out_dir / "mesh_overview.png")


def main(argv: list[str] | None = None) -> None:
    args = build_cli().parse_args(argv)
    mesh = Path(args.mesh).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    py = str(Path(args.python).resolve())

    if not mesh.exists():
        raise FileNotFoundError(mesh)

    surface_renderer = SCRIPT_DIR / "mesh_PPT.py"
    volume_renderer = SCRIPT_DIR / "mesh_PPT_vol.py"

    for renderer in (surface_renderer, volume_renderer):
        if not renderer.exists():
            raise FileNotFoundError(renderer)

    run_renderer(
        [
            py,
            str(surface_renderer),
            str(mesh),
            "--elev",
            "0",
            "--azim",
            "90",
            "--save",
            str(out_dir / "mesh_xz.png"),
            "--no-show",
        ]
    )
    run_renderer(
        [
            py,
            str(volume_renderer),
            str(mesh),
            "--save",
            str(out_dir / "mesh_xyz.png"),
            "--no-show",
        ]
    )
    run_renderer(
        [
            py,
            str(surface_renderer),
            str(mesh),
            "--save",
            str(out_dir / "mesh_top.png"),
            "--no-show",
        ]
    )

    write_overview(out_dir)
    print(f"Wrote PPT-style PNGs under {out_dir}")


if __name__ == "__main__":
    main()
