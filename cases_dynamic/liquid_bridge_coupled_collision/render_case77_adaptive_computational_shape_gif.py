#!/usr/bin/env python3
"""Render accepted Case 77 states without importing earlier case renderers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "case77_best_treatment_finite_initial_0to3600"
STATE_DIR_NAME = "young_laplace_mesh_states"
HISTORY_NAME = "case77_real_mesh_evolution_history.csv"


@dataclass(frozen=True)
class Snapshot:
    path: Path
    time_s: float


def _read_snapshots(out_dir: Path) -> list[Snapshot]:
    state_dir = out_dir / STATE_DIR_NAME
    if not state_dir.is_dir():
        raise FileNotFoundError(
            f"Missing accepted Case 77 state directory: {state_dir}"
        )
    snapshots: list[Snapshot] = []
    for path in state_dir.glob("*.npz"):
        with np.load(path, allow_pickle=False) as state:
            snapshots.append(Snapshot(path=path, time_s=float(state["time_s"])))
    snapshots.sort(key=lambda item: item.time_s)
    if not snapshots:
        raise RuntimeError(f"No accepted Case 77 states found in {state_dir}")
    return snapshots


def _select_snapshots(
    snapshots: list[Snapshot],
    maximum_frames: int | None,
) -> list[Snapshot]:
    if maximum_frames is None or maximum_frames >= len(snapshots):
        return snapshots
    if maximum_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    indices = np.linspace(0, len(snapshots) - 1, maximum_frames)
    return [snapshots[int(round(index))] for index in indices]


def _read_history(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Case 77 history: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Case 77 history is empty: {path}")
    columns: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[name]))
            except (TypeError, ValueError):
                values.append(float("nan"))
        column = np.asarray(values, dtype=float)
        if np.any(np.isfinite(column)):
            columns[name] = column
    return columns


def _profile(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as state:
        vertices = np.asarray(state["vertices_m"], dtype=float)
        rings = np.asarray(state["ring_index"], dtype=int)
        regions = np.asarray(state["ring_region"], dtype=int)
    radius_mm = np.asarray(
        [np.mean(np.hypot(vertices[ring, 0], vertices[ring, 1])) for ring in rings]
    ) * 1.0e3
    height_um = np.asarray(
        [np.mean(vertices[ring, 2]) for ring in rings]
    ) * 1.0e6
    return radius_mm, height_um, regions


def _nearest_history_index(history: dict[str, np.ndarray], time_s: float) -> int:
    return int(np.argmin(np.abs(history["t_s"] - float(time_s))))


def _local_film_minimum(path: Path) -> float:
    radius_mm, height_um, regions = _profile(path)
    film = regions == 1
    if not np.any(film):
        return float("nan")
    local_limit_mm = 0.75 * float(np.max(radius_mm[film]))
    local = film & (radius_mm <= local_limit_mm)
    return float(np.min(height_um[local]))


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[np.isfinite(values)]


def _render_frame(
    snapshot: Snapshot,
    history: dict[str, np.ndarray],
    initial_profile: tuple[np.ndarray, np.ndarray],
    film_minimum_times_s: np.ndarray,
    film_minimum_um: np.ndarray,
) -> Image.Image:
    radius_mm, height_um, regions = _profile(snapshot.path)
    bridge = regions == 0
    film = regions == 1
    index = _nearest_history_index(history, snapshot.time_s)
    times = history["t_s"]
    bridge_volume = history["bridge_volume_ul"]
    mode_code = history.get(
        "case77_momentum_mode_code",
        np.ones_like(times),
    )
    mode = "dynamic" if mode_code[index] > 0.5 else "Stokes"

    figure = plt.figure(figsize=(12.0, 7.0), dpi=120, facecolor="#f7f8fa")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.75, 1.0),
        height_ratios=(1.0, 1.0),
        hspace=0.34,
        wspace=0.28,
    )
    bridge_ax = figure.add_subplot(grid[0, 0])
    film_ax = figure.add_subplot(grid[1, 0])
    history_ax = figure.add_subplot(grid[:, 1])

    for axis in (bridge_ax, film_ax, history_ax):
        axis.set_facecolor("white")
        axis.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)

    if np.any(bridge):
        bridge_ax.plot(
            radius_mm[bridge],
            height_um[bridge],
            color="#146c94",
            linewidth=2.6,
            label="computed bridge",
        )
    if np.any(film):
        local_film = film & (radius_mm <= 5.0)
        bridge_ax.plot(
            radius_mm[local_film],
            height_um[local_film],
            color="#d97706",
            linewidth=2.2,
            label="transported film",
        )
    bridge_ax.set_xlim(0.0, 5.0)
    bridge_values = _finite(height_um[radius_mm <= 5.0])
    bridge_top = max(float(np.max(bridge_values)) if bridge_values.size else 100.0, 120.0)
    bridge_ax.set_ylim(0.0, 1.10 * bridge_top)
    bridge_ax.set_xlabel("radius r [mm]")
    bridge_ax.set_ylabel("height z [um]")
    bridge_ax.set_title("Accepted bridge-film meridian")
    bridge_ax.legend(loc="upper right", frameon=False)

    initial_r, initial_h = initial_profile
    film_ax.plot(
        initial_r,
        initial_h,
        color="#6b7280",
        linewidth=1.5,
        linestyle="--",
        label="analytic initial film",
    )
    if np.any(film):
        order = np.argsort(radius_mm[film])
        film_ax.plot(
            radius_mm[film][order],
            height_um[film][order],
            color="#d97706",
            linewidth=2.2,
            label="accepted V-shaped film",
        )
    film_ax.set_xlim(0.0, max(float(np.max(initial_r)), 12.0))
    film_values = _finite(height_um[film])
    film_top = max(
        float(np.max(_finite(initial_h))),
        float(np.max(film_values)) if film_values.size else 100.0,
        100.0,
    )
    film_ax.set_ylim(0.0, 1.10 * film_top)
    film_ax.set_xlabel("radius r [mm]")
    film_ax.set_ylabel("film height h [um]")
    film_ax.set_title("Finite-substrate film")
    film_ax.legend(loc="upper right", frameon=False)

    history_ax.plot(
        times,
        bridge_volume,
        color="#146c94",
        linewidth=2.3,
        label="bridge volume [uL]",
    )
    history_ax.scatter(
        [times[index]],
        [bridge_volume[index]],
        color="#146c94",
        edgecolor="white",
        linewidth=0.8,
        s=54,
        zorder=4,
    )
    history_ax.set_xscale("symlog", linthresh=1.0)
    history_ax.set_xlim(0.0, max(float(times[-1]), 1.0))
    history_ax.set_xlabel("time [s]")
    history_ax.set_ylabel("bridge volume [uL]", color="#146c94")
    history_ax.tick_params(axis="y", colors="#146c94")
    history_ax.set_title("Conservative prediction history")

    minimum_ax = history_ax.twinx()
    minimum_ax.plot(
        film_minimum_times_s,
        film_minimum_um,
        color="#b45309",
        linewidth=1.8,
        label="accepted film minimum [um]",
    )
    minimum_index = int(
        np.argmin(np.abs(film_minimum_times_s - snapshot.time_s))
    )
    minimum_ax.scatter(
        [film_minimum_times_s[minimum_index]],
        [film_minimum_um[minimum_index]],
        color="#b45309",
        edgecolor="white",
        linewidth=0.8,
        s=46,
        zorder=4,
    )
    minimum_ax.set_ylabel("accepted film minimum [um]", color="#b45309")
    minimum_ax.tick_params(axis="y", colors="#b45309")
    minimum_ax.spines["top"].set_visible(False)

    figure.suptitle(
        f"Case 77 predictive bridge-film solver | t = {snapshot.time_s:g} s | {mode}",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.015,
        "5 through-film node levels | resolved K_mu | K_lub = 0 | no uCFL | Young-Laplace pressure feedback",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba, mode="RGBA").convert(
        "P",
        palette=Image.Palette.ADAPTIVE,
        colors=256,
    )


def render(
    out_dir: Path,
    output: Path | None,
    duration_ms: int,
    maximum_frames: int | None,
) -> Path:
    snapshots = _select_snapshots(_read_snapshots(out_dir), maximum_frames)
    history = _read_history(out_dir / HISTORY_NAME)
    initial_radius, initial_height, initial_regions = _profile(snapshots[0].path)
    initial_film = initial_regions == 1
    order = np.argsort(initial_radius[initial_film])
    initial_profile = (
        initial_radius[initial_film][order],
        initial_height[initial_film][order],
    )
    film_minimum_times_s = np.asarray(
        [snapshot.time_s for snapshot in snapshots],
        dtype=float,
    )
    film_minimum_um = np.asarray(
        [_local_film_minimum(snapshot.path) for snapshot in snapshots],
        dtype=float,
    )
    frames = [
        _render_frame(
            snapshot,
            history,
            initial_profile,
            film_minimum_times_s,
            film_minimum_um,
        )
        for snapshot in snapshots
    ]
    final_label = f"{snapshots[-1].time_s:g}".replace(".", "p")
    gif_path = (
        output
        if output is not None
        else out_dir / f"case77_flux_inventory_prediction_0to{final_label}.gif"
    )
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(int(duration_ms), 20),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return gif_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--duration-ms", type=int, default=420)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    path = render(
        arguments.out_dir.expanduser().resolve(),
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else None,
        arguments.duration_ms,
        arguments.max_frames,
    )
    print(path)


if __name__ == "__main__":
    main()
