from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


ROOT = Path(__file__).resolve().parent
GIFS = [
    ROOT / "out/sphere_fheron_noflux_0p016/flux_incompressible_sphere.gif",
    ROOT / "out/sphere_fheron_noflux_0p016/flux_compressible_sphere.gif",
    ROOT / "out/sphere_fheron_flux_fv/flux_incompressible_sphere_tfinal_0p016.gif",
    ROOT / "out/sphere_fheron_flux_fv/flux_compressible_sphere_tfinal_0p016.gif",
    ROOT / "out/sphere_fheron_flux_fv_fullflux_stabilized/flux_compressible_sphere.gif",
    ROOT / "out/sphere_fheron_flux_fv_fullflux_projected_upwind_nolimit/flux_incompressible_sphere.gif",
]


def tight_name(path: Path) -> Path:
    return path.with_name(f"{path.stem}_tight{path.suffix}")


def frame_bbox(frame: Image.Image, threshold: int = 245) -> tuple[int, int, int, int] | None:
    array = np.asarray(frame.convert("RGB"))
    mask = np.any(array < threshold, axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def union_bbox(path: Path, padding: int = 12) -> tuple[int, int, int, int]:
    image = Image.open(path)
    union: list[int] | None = None
    for frame in ImageSequence.Iterator(image):
        box = frame_bbox(frame)
        if box is None:
            continue
        if union is None:
            union = list(box)
        else:
            union = [
                min(union[0], box[0]),
                min(union[1], box[1]),
                max(union[2], box[2]),
                max(union[3], box[3]),
            ]
    if union is None:
        return 0, 0, image.width, image.height
    return (
        max(0, union[0] - padding),
        max(0, union[1] - padding),
        min(image.width, union[2] + padding),
        min(image.height, union[3] + padding),
    )


def crop_gif(path: Path) -> Path:
    image = Image.open(path)
    crop = union_bbox(path)
    durations: list[int] = []
    frames: list[Image.Image] = []
    for frame in ImageSequence.Iterator(image):
        durations.append(int(frame.info.get("duration", image.info.get("duration", 80))))
        frames.append(frame.convert("RGB").crop(crop))
    output = tight_name(path)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=int(image.info.get("loop", 0)),
        optimize=True,
        disposal=2,
    )
    return output


def main() -> int:
    for path in GIFS:
        output = crop_gif(path)
        with Image.open(path) as original, Image.open(output) as cropped:
            print(f"{path.relative_to(ROOT)} {original.size} -> {output.relative_to(ROOT)} {cropped.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
