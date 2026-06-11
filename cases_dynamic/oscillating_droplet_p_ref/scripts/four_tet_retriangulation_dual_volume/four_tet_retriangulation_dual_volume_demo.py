#!/usr/bin/env python3
"""Delaunay-valid HC retriangulation demo for interface dual volume.

This script builds a minimal two-snapshot case:

* step 0: vertices are triangulated through ddgclib's dynamic-integrator
  retopology path, which rebuilds the HC connectivity with SciPy Delaunay.
* step 1: vertices have moved slightly, then the same retopology path is run
  again. Both snapshots therefore follow the same ddgclib triangulation rule.

The interface vertices lie on a unit sphere.  The selected-vertex liquid dual
volume is computed from the barycentric HC tetrahedra: each incident tetrahedron
contributes one quarter of its tetrahedral volume to the selected vertex's
barycentric dual cell, then those contributions are split by phase.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "four_tet_retriangulation_dual_volume"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize


SPHERE_CENTER = np.zeros(3)
SPHERE_RADIUS = 1.0
SELECTED_LABEL = "a"

AIR_COLOR = "#9aa0a6"
LIQUID_COLOR = "#1f77b4"
INTERFACE_COLOR = "#2ca25f"
SELECTED_COLOR = "#c62828"
EDGE_COLOR = "#222222"
REMOVED_EDGE_COLOR = "#d62728"
ADDED_EDGE_COLOR = "#2ca02c"
CHANGE_ARROW_COLOR = "#8e24aa"
MOVED_VERTEX_COLOR = "#ff7f0e"
INTERFACE_LABELS = ("a", "b", "c", "d", "e", "f", "g", "h")


@dataclass(frozen=True)
class TetRecord:
    name: str
    phase: str
    labels: tuple[str, str, str, str]


@dataclass
class Snapshot:
    step: int
    name: str
    tets: list[TetRecord]
    hc_dual_vertices: int
    liquid_dual_volume: float
    air_dual_volume: float
    total_dual_volume: float
    per_tet: list[dict[str, object]]


@dataclass(frozen=True)
class Comparison:
    delta_liquid_volume: float
    delta_liquid_percent: float
    max_vertex_position_delta: float
    removed_edges: tuple[tuple[str, str], ...]
    added_edges: tuple[tuple[str, str], ...]
    moved_vertices: tuple[str, ...]


def sphere_point(y: float, z: float) -> np.ndarray:
    """Point on the positive-x cap of the unit sphere."""
    x2 = SPHERE_RADIUS * SPHERE_RADIUS - y * y - z * z
    if x2 <= 0.0:
        raise ValueError("interface point is outside the requested sphere cap")
    return np.array([np.sqrt(x2), y, z], dtype=float)


def build_before_points() -> dict[str, np.ndarray]:
    """Local vertices before motion: interface cap, one air, one liquid."""
    return {
        "a": np.array([1.0, 0.0, 0.0], dtype=float),
        "b": sphere_point(0.24000000, 0.20000000),
        "c": sphere_point(-0.10000000, 0.29000000),
        "d": sphere_point(0.34000000, -0.08000000),
        "e": sphere_point(-0.26000000, -0.15000000),
        "f": sphere_point(0.06000000, -0.33000000),
        "g": sphere_point(-0.36000000, 0.08000000),
        "h": sphere_point(0.26000000, 0.34000000),
        "air": np.array([1.32, 0.04, 0.02], dtype=float),
        "liq": np.array([0.64, -0.02, -0.02], dtype=float),
    }


def build_after_points() -> dict[str, np.ndarray]:
    """Local vertices after one point moves, still on the sphere cap."""
    points = build_before_points()
    points["e"] = sphere_point(-0.23500000, -0.07500000)
    return points


def connect_tetrahedron(vertices: dict[str, object], labels: Iterable[str]) -> None:
    tet_vertices = [vertices[label] for label in labels]
    for i, vi in enumerate(tet_vertices):
        for vj in tet_vertices[i + 1 :]:
            vi.connect(vj)


def build_hc(
    points: dict[str, np.ndarray],
    tets: list[TetRecord],
) -> tuple[Complex, dict[str, object]]:
    hc = Complex(
        3,
        domain=[
            (0.55, 1.35),
            (-0.50, 0.50),
            (-0.40, 0.40),
        ],
    )
    vertices = {label: hc.V[tuple(coord)] for label, coord in points.items()}
    for label, vertex in vertices.items():
        vertex.label = label
        vertex.u = np.zeros(3)
        vertex.p = 0.0
        vertex.m = 1.0
        vertex.is_interface = label in INTERFACE_LABELS
        if vertex.is_interface:
            vertex.phase = "interface"
        else:
            vertex.phase = "liquid" if label == "liq" else "air"

    for tet in tets:
        connect_tetrahedron(vertices, tet.labels)
    return hc, vertices


def compute_hc_duals(hc: Complex) -> int:
    """Populate HC barycentric dual data for the current connectivity."""
    boundary_vertices = hc.boundary()
    for vertex in hc.V:
        vertex.boundary = vertex in boundary_vertices
    with np.errstate(invalid="ignore", divide="ignore"):
        compute_vd(hc, method="barycentric")
    return len(hc.Vd)


def tet_volume(points: dict[str, np.ndarray], labels: tuple[str, str, str, str]) -> float:
    p0, p1, p2, p3 = (points[label] for label in labels)
    mat = np.vstack((p1 - p0, p2 - p0, p3 - p0))
    return abs(float(np.linalg.det(mat))) / 6.0


def percent_change(before: float, after: float) -> float:
    if abs(before) < 1e-30:
        return float("nan")
    return 100.0 * (after - before) / before


def phase_from_tet(points: dict[str, np.ndarray], labels: tuple[str, str, str, str]) -> str:
    if "liq" in labels and "air" not in labels:
        return "liquid"
    if "air" in labels and "liq" not in labels:
        return "air"
    centroid = np.mean([points[label] for label in labels], axis=0)
    r = np.linalg.norm(centroid - SPHERE_CENTER)
    return "liquid" if r <= SPHERE_RADIUS else "air"


def tetrahedra_from_hc_cliques(
    hc: Complex,
    points: dict[str, np.ndarray],
) -> list[TetRecord]:
    """Recover tetrahedra as 4-cliques in the current HC connectivity graph."""
    clique_labels: list[tuple[str, str, str, str]] = []
    for vertices in itertools.combinations(list(hc.V), 4):
        is_tetrahedron = all(
            vj in vi.nn for vi, vj in itertools.combinations(vertices, 2)
        )
        if is_tetrahedron:
            clique_labels.append(tuple(sorted(vertex.label for vertex in vertices)))

    clique_labels = sorted(set(clique_labels))
    records: list[TetRecord] = []
    for index, tet_labels in enumerate(clique_labels):
        phase = phase_from_tet(points, tet_labels)
        records.append(TetRecord(f"T{index:02d}", phase, tet_labels))
    return records


def selected_dual_simplex_vertices(
    points: dict[str, np.ndarray],
    tet_labels: tuple[str, str, str, str],
    selected_label: str,
) -> list[np.ndarray]:
    """Barycentric subdivision tets belonging to one selected primal vertex."""
    if selected_label not in tet_labels:
        return []

    selected = points[selected_label]
    others = [label for label in tet_labels if label != selected_label]
    tet_centroid = np.mean([points[label] for label in tet_labels], axis=0)

    small_tets = []
    for order in itertools.permutations(others):
        p1 = 0.5 * (selected + points[order[0]])
        p2 = (selected + points[order[0]] + points[order[1]]) / 3.0
        small_tets.append(np.array([selected, p1, p2, tet_centroid], dtype=float))
    return small_tets


def summarize_snapshot(
    step: int,
    name: str,
    hc: Complex,
    points: dict[str, np.ndarray],
    tets: list[TetRecord],
    selected_label: str,
) -> Snapshot:
    hc_dual_vertices = compute_hc_duals(hc)

    per_tet = []
    liquid_dual_volume = 0.0
    air_dual_volume = 0.0
    for tet in tets:
        full_volume = tet_volume(points, tet.labels)
        selected_volume = full_volume / 4.0 if selected_label in tet.labels else 0.0
        if tet.phase == "liquid":
            liquid_dual_volume += selected_volume
        elif tet.phase == "air":
            air_dual_volume += selected_volume
        per_tet.append(
            {
                "name": tet.name,
                "phase": tet.phase,
                "labels": list(tet.labels),
                "tet_volume": full_volume,
                "selected_dual_volume": selected_volume,
            }
        )

    return Snapshot(
        step=step,
        name=name,
        tets=tets,
        hc_dual_vertices=hc_dual_vertices,
        liquid_dual_volume=liquid_dual_volume,
        air_dual_volume=air_dual_volume,
        total_dual_volume=liquid_dual_volume + air_dual_volume,
        per_tet=per_tet,
    )


def tetra_faces(vertices: np.ndarray) -> list[np.ndarray]:
    return [
        vertices[[0, 1, 2]],
        vertices[[0, 1, 3]],
        vertices[[0, 2, 3]],
        vertices[[1, 2, 3]],
    ]


def unique_edges(tets: list[TetRecord]) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for tet in tets:
        for a_label, b_label in itertools.combinations(tet.labels, 2):
            edges.add(tuple(sorted((a_label, b_label))))
    return sorted(edges)


def edge_midpoint(points: dict[str, np.ndarray], edge: tuple[str, str]) -> np.ndarray:
    return 0.5 * (points[edge[0]] + points[edge[1]])


def changed_edges(
    before: Snapshot,
    after: Snapshot,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    before_edges = set(unique_edges(before.tets))
    after_edges = set(unique_edges(after.tets))
    removed_edges = tuple(sorted(before_edges - after_edges))
    added_edges = tuple(sorted(after_edges - before_edges))
    return removed_edges, added_edges


def max_position_delta(
    before_points: dict[str, np.ndarray],
    after_points: dict[str, np.ndarray],
) -> float:
    shared_labels = sorted(set(before_points) & set(after_points))
    return max(
        float(np.linalg.norm(after_points[label] - before_points[label]))
        for label in shared_labels
    )


def moved_vertex_labels(
    before_points: dict[str, np.ndarray],
    after_points: dict[str, np.ndarray],
    tol: float = 1.0e-12,
) -> tuple[str, ...]:
    shared_labels = sorted(set(before_points) & set(after_points))
    return tuple(
        label
        for label in shared_labels
        if np.linalg.norm(after_points[label] - before_points[label]) > tol
    )


def retopologized_snapshot(
    step: int,
    name: str,
    points: dict[str, np.ndarray],
    selected_label: str,
) -> tuple[Snapshot, Complex]:
    """Build HC from positions only, then let ddgclib Delaunay-connect it."""
    hc, _ = build_hc(points, [])
    _retopologize(hc, set(), 3)
    tets = tetrahedra_from_hc_cliques(hc, points)
    if not tets:
        raise RuntimeError(f"retopology produced no tetrahedra for {name}")
    snapshot = summarize_snapshot(step, name, hc, points, tets, selected_label)
    return snapshot, hc


def draw_edge_highlight(
    ax,
    points: dict[str, np.ndarray],
    edge: tuple[str, str],
    *,
    color: str,
    label: str,
) -> None:
    p0 = points[edge[0]]
    p1 = points[edge[1]]
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        color=color,
        linewidth=3.0,
        alpha=0.95,
    )
    midpoint = edge_midpoint(points, edge)
    ax.text(
        *(midpoint + np.array([0.016, 0.016, 0.016])),
        label,
        color=color,
        fontsize=8,
        weight="bold",
    )


def draw_connectivity_change_arrow(
    ax,
    points: dict[str, np.ndarray],
    removed_edges: tuple[tuple[str, str], ...],
    added_edges: tuple[tuple[str, str], ...],
) -> None:
    if not removed_edges or not added_edges:
        return
    start = edge_midpoint(points, removed_edges[0])
    end = edge_midpoint(points, added_edges[0])
    delta = end - start
    ax.quiver(
        start[0],
        start[1],
        start[2],
        delta[0],
        delta[1],
        delta[2],
        color=CHANGE_ARROW_COLOR,
        linewidth=2.2,
        arrow_length_ratio=0.25,
        alpha=0.95,
    )


def draw_motion_arrows(
    ax,
    before_points: dict[str, np.ndarray],
    after_points: dict[str, np.ndarray],
    moved_vertices: Iterable[str],
) -> None:
    for label in moved_vertices:
        start = before_points[label]
        end = after_points[label]
        delta = end - start
        ax.quiver(
            start[0],
            start[1],
            start[2],
            delta[0],
            delta[1],
            delta[2],
            color=MOVED_VERTEX_COLOR,
            linewidth=2.6,
            arrow_length_ratio=0.35,
            alpha=0.98,
        )
        ax.text(
            *(end + np.array([0.014, -0.018, 0.014])),
            f"moved {label}",
            color=MOVED_VERTEX_COLOR,
            fontsize=8,
            weight="bold",
        )


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = np.array(
        [
            abs(x_limits[1] - x_limits[0]),
            abs(y_limits[1] - y_limits[0]),
            abs(z_limits[1] - z_limits[0]),
        ]
    )
    centers = np.array(
        [
            np.mean(x_limits),
            np.mean(y_limits),
            np.mean(z_limits),
        ]
    )
    radius = 0.5 * max(ranges)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def draw_interface_sphere(ax) -> None:
    y = np.linspace(-0.46, 0.46, 25)
    z = np.linspace(-0.36, 0.36, 25)
    yy, zz = np.meshgrid(y, z)
    xx2 = SPHERE_RADIUS * SPHERE_RADIUS - yy * yy - zz * zz
    xx = np.sqrt(np.clip(xx2, 0.0, None))
    ax.plot_wireframe(
        xx,
        yy,
        zz,
        rstride=4,
        cstride=4,
        color=INTERFACE_COLOR,
        linewidth=0.45,
        alpha=0.30,
    )


def plot_snapshot(
    ax,
    points: dict[str, np.ndarray],
    snapshot: Snapshot,
    selected_label: str,
    moved_vertices: Iterable[str] = (),
) -> None:
    draw_interface_sphere(ax)
    moved_vertex_set = set(moved_vertices)

    for tet in snapshot.tets:
        small_tets = selected_dual_simplex_vertices(points, tet.labels, selected_label)
        if not small_tets:
            continue
        faces = []
        for small_tet in small_tets:
            faces.extend(tetra_faces(small_tet))
        color = LIQUID_COLOR if tet.phase == "liquid" else AIR_COLOR
        alpha = 0.54 if tet.phase == "liquid" else 0.36
        collection = Poly3DCollection(
            faces,
            facecolor=color,
            edgecolor=color,
            linewidth=0.25,
            alpha=alpha,
        )
        ax.add_collection3d(collection)

    edge_segments = [
        [points[a_label], points[b_label]]
        for a_label, b_label in unique_edges(snapshot.tets)
    ]
    ax.add_collection3d(
        Line3DCollection(edge_segments, colors=EDGE_COLOR, linewidths=0.9, alpha=0.72)
    )

    for label, coord in points.items():
        if label in moved_vertex_set:
            ax.scatter(
                *coord,
                color=MOVED_VERTEX_COLOR,
                marker="D",
                s=72,
                depthshade=False,
            )
        elif label == selected_label:
            ax.scatter(*coord, color=SELECTED_COLOR, s=54, depthshade=False)
        elif label in {"air", "liq"}:
            color = AIR_COLOR if label == "air" else LIQUID_COLOR
            ax.scatter(*coord, color=color, s=40, depthshade=False)
        else:
            ax.scatter(*coord, color=INTERFACE_COLOR, s=28, depthshade=False)
        text_color = MOVED_VERTEX_COLOR if label in moved_vertex_set else "black"
        text = f"{label}*" if label in moved_vertex_set else label
        ax.text(
            *(coord + np.array([0.012, 0.012, 0.012])),
            text,
            fontsize=8,
            color=text_color,
            weight="bold" if label in moved_vertex_set else "normal",
        )

    ax.set_title(
        f"step {snapshot.step}: {snapshot.name}\n"
        f"selected {selected_label} liquid dual volume = "
        f"{snapshot.liquid_dual_volume:.9e}",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-62)
    ax.set_xlim(0.62, 1.32)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(-0.36, 0.34)
    set_axes_equal(ax)


def save_snapshot_png(
    out_dir: Path,
    points: dict[str, np.ndarray],
    snapshot: Snapshot,
    selected_label: str,
    moved_vertices: Iterable[str] = (),
) -> Path:
    fig = plt.figure(figsize=(8.0, 6.6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    plot_snapshot(ax, points, snapshot, selected_label, moved_vertices)
    fig.tight_layout()
    path = out_dir / f"step_{snapshot.step:03d}_{snapshot.name.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_comparison_png(
    out_dir: Path,
    before_points: dict[str, np.ndarray],
    after_points: dict[str, np.ndarray],
    before: Snapshot,
    after: Snapshot,
    selected_label: str,
    comparison: Comparison,
) -> Path:
    fig = plt.figure(figsize=(13.0, 6.2), dpi=180)
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122, projection="3d")
    plot_snapshot(ax0, before_points, before, selected_label, comparison.moved_vertices)
    plot_snapshot(ax1, after_points, after, selected_label, comparison.moved_vertices)
    for edge in comparison.removed_edges:
        draw_edge_highlight(
            ax0,
            before_points,
            edge,
            color=REMOVED_EDGE_COLOR,
            label=f"removed {'-'.join(edge)}",
        )
    for edge in comparison.added_edges:
        draw_edge_highlight(
            ax1,
            after_points,
            edge,
            color=ADDED_EDGE_COLOR,
            label=f"added {'-'.join(edge)}",
        )
    draw_connectivity_change_arrow(
        ax0,
        before_points,
        comparison.removed_edges,
        comparison.added_edges,
    )
    draw_connectivity_change_arrow(
        ax1,
        after_points,
        comparison.removed_edges,
        comparison.added_edges,
    )
    draw_motion_arrows(
        ax0,
        before_points,
        after_points,
        comparison.moved_vertices,
    )
    draw_motion_arrows(
        ax1,
        before_points,
        after_points,
        comparison.moved_vertices,
    )
    if comparison.removed_edges and comparison.added_edges:
        connectivity_text = (
            f", connectivity: {'-'.join(comparison.removed_edges[0])} "
            f"-> {'-'.join(comparison.added_edges[0])}"
        )
    else:
        connectivity_text = ""
    fig.suptitle(
        "Delta liquid dual volume: "
        f"{comparison.delta_liquid_volume:.9e} "
        f"({comparison.delta_liquid_percent:+.3f}%), "
        f"tetrahedra: {len(before.tets)} -> {len(after.tets)}, "
        f"moved: {','.join(comparison.moved_vertices)}, "
        "max vertex position delta: "
        f"{comparison.max_vertex_position_delta:.3e}"
        f"{connectivity_text}",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / "before_after_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def rgba_hex(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_plotly_sphere(fig, row: int, col: int) -> None:
    y_values = np.linspace(-0.46, 0.46, 9)
    z_values = np.linspace(-0.36, 0.36, 9)
    for y in y_values:
        z = np.linspace(-0.36, 0.36, 60)
        x = np.sqrt(np.clip(SPHERE_RADIUS * SPHERE_RADIUS - y * y - z * z, 0.0, None))
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=np.full_like(x, y),
                z=z,
                mode="lines",
                line=dict(color=rgba_hex(INTERFACE_COLOR, 0.25), width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    for z in z_values:
        y = np.linspace(-0.46, 0.46, 60)
        x = np.sqrt(np.clip(SPHERE_RADIUS * SPHERE_RADIUS - y * y - z * z, 0.0, None))
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=np.full_like(x, z),
                mode="lines",
                line=dict(color=rgba_hex(INTERFACE_COLOR, 0.25), width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def add_plotly_dual_mesh(
    fig,
    points: dict[str, np.ndarray],
    snapshot: Snapshot,
    selected_label: str,
    row: int,
    col: int,
) -> None:
    for tet in snapshot.tets:
        small_tets = selected_dual_simplex_vertices(points, tet.labels, selected_label)
        if not small_tets:
            continue
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        i_faces: list[int] = []
        j_faces: list[int] = []
        k_faces: list[int] = []
        for small_tet in small_tets:
            for face in tetra_faces(small_tet):
                idx0 = len(xs)
                for coord in face:
                    xs.append(float(coord[0]))
                    ys.append(float(coord[1]))
                    zs.append(float(coord[2]))
                i_faces.append(idx0)
                j_faces.append(idx0 + 1)
                k_faces.append(idx0 + 2)
        color = LIQUID_COLOR if tet.phase == "liquid" else AIR_COLOR
        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=i_faces,
                j=j_faces,
                k=k_faces,
                color=color,
                opacity=0.50 if tet.phase == "liquid" else 0.32,
                name=f"{snapshot.name} {tet.name} {tet.phase}",
                hovertemplate=(
                    f"{tet.name} {tet.phase}<br>"
                    f"tet: {'-'.join(tet.labels)}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def add_plotly_edges(
    fig,
    points: dict[str, np.ndarray],
    edges: Iterable[tuple[str, str]],
    row: int,
    col: int,
    *,
    color: str,
    width: float,
    name: str,
    showlegend: bool,
) -> None:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    hover: list[str | None] = []
    for edge in edges:
        p0 = points[edge[0]]
        p1 = points[edge[1]]
        xs.extend([float(p0[0]), float(p1[0]), None])
        ys.extend([float(p0[1]), float(p1[1]), None])
        zs.extend([float(p0[2]), float(p1[2]), None])
        hover.extend([f"{edge[0]}-{edge[1]}", f"{edge[0]}-{edge[1]}", None])
    if not xs:
        return
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def add_plotly_points(
    fig,
    points: dict[str, np.ndarray],
    selected_label: str,
    row: int,
    col: int,
    *,
    showlegend: bool,
    moved_vertices: Iterable[str] = (),
) -> None:
    labels = list(points.keys())
    coords = np.array([points[label] for label in labels])
    moved_vertex_set = set(moved_vertices)
    colors = []
    sizes = []
    symbols = []
    for label in labels:
        if label in moved_vertex_set:
            colors.append(MOVED_VERTEX_COLOR)
            sizes.append(8)
            symbols.append("diamond")
        elif label == selected_label:
            colors.append(SELECTED_COLOR)
            sizes.append(7)
            symbols.append("circle")
        elif label == "liq":
            colors.append(LIQUID_COLOR)
            sizes.append(6)
            symbols.append("circle")
        elif label == "air":
            colors.append(AIR_COLOR)
            sizes.append(6)
            symbols.append("circle")
        else:
            colors.append(INTERFACE_COLOR)
            sizes.append(5)
            symbols.append("circle")
    display_labels = [
        f"{label}*" if label in moved_vertex_set else label for label in labels
    ]
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers+text",
            marker=dict(color=colors, size=sizes, symbol=symbols),
            text=display_labels,
            textposition="top center",
            name="vertices",
            hovertext=labels,
            hovertemplate="%{hovertext}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def add_plotly_motion_arrows(
    fig,
    before_points: dict[str, np.ndarray],
    after_points: dict[str, np.ndarray],
    moved_vertices: Iterable[str],
    row: int,
    col: int,
    *,
    showlegend: bool,
) -> None:
    for label in moved_vertices:
        start = before_points[label]
        end = after_points[label]
        delta = end - start
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=MOVED_VERTEX_COLOR, width=7),
                name="moved vertex",
                hovertemplate=f"moved {label}<extra></extra>",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Cone(
                x=[end[0]],
                y=[end[1]],
                z=[end[2]],
                u=[delta[0]],
                v=[delta[1]],
                w=[delta[2]],
                sizemode="absolute",
                sizeref=0.045,
                anchor="tip",
                colorscale=[[0.0, MOVED_VERTEX_COLOR], [1.0, MOVED_VERTEX_COLOR]],
                showscale=False,
                name="moved vertex arrow head",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def add_plotly_change_arrow(
    fig,
    points: dict[str, np.ndarray],
    comparison: Comparison,
    row: int,
    col: int,
    *,
    showlegend: bool,
) -> None:
    if not comparison.removed_edges or not comparison.added_edges:
        return
    start = edge_midpoint(points, comparison.removed_edges[0])
    end = edge_midpoint(points, comparison.added_edges[0])
    delta = end - start
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=CHANGE_ARROW_COLOR, width=6),
            name="connectivity arrow",
            hovertemplate=(
                f"{'-'.join(comparison.removed_edges[0])} -> "
                f"{'-'.join(comparison.added_edges[0])}<extra></extra>"
            ),
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Cone(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            u=[delta[0]],
            v=[delta[1]],
            w=[delta[2]],
            sizemode="absolute",
            sizeref=0.055,
            anchor="tip",
            colorscale=[[0.0, CHANGE_ARROW_COLOR], [1.0, CHANGE_ARROW_COLOR]],
            showscale=False,
            name="arrow head",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def add_plotly_snapshot(
    fig,
    points: dict[str, np.ndarray],
    snapshot: Snapshot,
    selected_label: str,
    comparison: Comparison,
    row: int,
    col: int,
) -> None:
    add_plotly_sphere(fig, row, col)
    add_plotly_dual_mesh(fig, points, snapshot, selected_label, row, col)
    add_plotly_edges(
        fig,
        points,
        unique_edges(snapshot.tets),
        row,
        col,
        color=EDGE_COLOR,
        width=3,
        name="HC edges",
        showlegend=col == 1,
    )
    if snapshot.step == 0:
        add_plotly_edges(
            fig,
            points,
            comparison.removed_edges,
            row,
            col,
            color=REMOVED_EDGE_COLOR,
            width=8,
            name="removed edge",
            showlegend=True,
        )
    else:
        add_plotly_edges(
            fig,
            points,
            comparison.added_edges,
            row,
            col,
            color=ADDED_EDGE_COLOR,
            width=8,
            name="added edge",
            showlegend=True,
    )
    add_plotly_change_arrow(fig, points, comparison, row, col, showlegend=col == 1)
    add_plotly_points(
        fig,
        points,
        selected_label,
        row,
        col,
        showlegend=col == 1,
        moved_vertices=comparison.moved_vertices,
    )


def save_interactive_html(
    out_dir: Path,
    before_points: dict[str, np.ndarray],
    after_points: dict[str, np.ndarray],
    before: Snapshot,
    after: Snapshot,
    selected_label: str,
    comparison: Comparison,
) -> Path:
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"step 0 before, V_liquid={before.liquid_dual_volume:.9e}",
            f"step 1 after, V_liquid={after.liquid_dual_volume:.9e}",
        ),
        horizontal_spacing=0.02,
    )
    add_plotly_snapshot(fig, before_points, before, selected_label, comparison, 1, 1)
    add_plotly_snapshot(fig, after_points, after, selected_label, comparison, 1, 2)
    add_plotly_motion_arrows(
        fig,
        before_points,
        after_points,
        comparison.moved_vertices,
        1,
        1,
        showlegend=True,
    )
    add_plotly_motion_arrows(
        fig,
        before_points,
        after_points,
        comparison.moved_vertices,
        1,
        2,
        showlegend=False,
    )

    scene_settings = dict(
        xaxis=dict(title="x", range=[0.62, 1.32]),
        yaxis=dict(title="y", range=[-0.45, 0.45]),
        zaxis=dict(title="z", range=[-0.36, 0.34]),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.35, y=-1.65, z=1.0)),
    )
    if comparison.removed_edges and comparison.added_edges:
        connectivity_text = (
            f", connectivity {'-'.join(comparison.removed_edges[0])} -> "
            f"{'-'.join(comparison.added_edges[0])}"
        )
    else:
        connectivity_text = ""
    fig.update_layout(
        title=(
            "Delaunay-valid HC retriangulation: "
            f"{len(before.tets)} -> {len(after.tets)} tetrahedra, "
            f"Delta={comparison.delta_liquid_volume:.9e} "
            f"({comparison.delta_liquid_percent:+.3f}%)"
            f"{connectivity_text}, "
            f"moved {','.join(comparison.moved_vertices)}, "
            f"max position delta={comparison.max_vertex_position_delta:.3e}"
        ),
        scene=scene_settings,
        scene2=scene_settings,
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=72, b=0),
        height=760,
    )
    path = out_dir / "before_after_interactive.html"
    sync_script = """
(function() {
  const gd = document.getElementById('{plot_id}');
  let syncing = false;

  function hasCameraUpdate(sceneName, eventData) {
    const cameraKey = sceneName + '.camera';
    const cameraPrefix = cameraKey + '.';
    return Object.keys(eventData).some(function(key) {
      return key === cameraKey || key.startsWith(cameraPrefix);
    });
  }

  function cloneCamera(camera) {
    return JSON.parse(JSON.stringify(camera || {}));
  }

  function cameraFromEvent(sceneName, eventData) {
    const cameraKey = sceneName + '.camera';
    const cameraPrefix = cameraKey + '.';
    if (eventData[cameraKey]) {
      return cloneCamera(eventData[cameraKey]);
    }

    const camera = cloneCamera((gd.layout[sceneName] || {}).camera);
    Object.keys(eventData).forEach(function(key) {
      if (!key.startsWith(cameraPrefix)) {
        return;
      }
      const path = key.slice(cameraPrefix.length).split('.');
      let target = camera;
      for (let index = 0; index < path.length - 1; index += 1) {
        target[path[index]] = target[path[index]] || {};
        target = target[path[index]];
      }
      target[path[path.length - 1]] = eventData[key];
    });
    return camera;
  }

  gd.on('plotly_relayout', function(eventData) {
    if (syncing || !eventData) {
      return;
    }

    const update = {};
    if (hasCameraUpdate('scene', eventData)) {
      update['scene2.camera'] = cameraFromEvent('scene', eventData);
    }
    if (hasCameraUpdate('scene2', eventData)) {
      update['scene.camera'] = cameraFromEvent('scene2', eventData);
    }
    if (!Object.keys(update).length) {
      return;
    }

    syncing = true;
    Plotly.relayout(gd, update).then(function() {
      syncing = false;
    }).catch(function() {
      syncing = false;
    });
  });

  window.__fourTetCameraSync = true;
})();
"""
    fig.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        post_script=sync_script,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    return path


def write_summary(
    out_dir: Path,
    snapshots: list[Snapshot],
    selected_label: str,
    comparison: Comparison,
) -> tuple[Path, Path]:
    csv_path = out_dir / "liquid_dual_volume_summary.csv"
    json_path = out_dir / "liquid_dual_volume_summary.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "snapshot",
                "selected_vertex",
                "liquid_dual_volume",
                "air_dual_volume",
                "total_selected_dual_volume",
                "delta_liquid_volume_vs_previous",
                "delta_liquid_percent_vs_previous",
                "hc_dual_vertices",
                "incident_tets",
            ],
        )
        writer.writeheader()
        for snapshot in snapshots:
            writer.writerow(
                {
                    "step": snapshot.step,
                    "snapshot": snapshot.name,
                    "selected_vertex": selected_label,
                    "liquid_dual_volume": f"{snapshot.liquid_dual_volume:.17e}",
                    "air_dual_volume": f"{snapshot.air_dual_volume:.17e}",
                    "total_selected_dual_volume": f"{snapshot.total_dual_volume:.17e}",
                    "delta_liquid_volume_vs_previous": (
                        f"{comparison.delta_liquid_volume:.17e}"
                        if snapshot.step == snapshots[-1].step
                        else ""
                    ),
                    "delta_liquid_percent_vs_previous": (
                        f"{comparison.delta_liquid_percent:.9f}"
                        if snapshot.step == snapshots[-1].step
                        else ""
                    ),
                    "hc_dual_vertices": snapshot.hc_dual_vertices,
                    "incident_tets": ";".join(
                        f"{tet.name}:{tet.phase}:{'-'.join(tet.labels)}"
                        for tet in snapshot.tets
                        if selected_label in tet.labels
                    ),
                }
            )

    payload = {
        "selected_vertex": selected_label,
        "sphere_center": SPHERE_CENTER.tolist(),
        "sphere_radius": SPHERE_RADIUS,
        "method": (
            "both snapshots use ddgclib dynamic _retopologize, which rebuilds "
            "HC connectivity with SciPy Delaunay; barycentric HC dual split; "
            "each incident tetrahedron contributes tet_volume/4 to the "
            "selected vertex"
        ),
        "comparison": {
            "delta_liquid_volume": comparison.delta_liquid_volume,
            "delta_liquid_percent": comparison.delta_liquid_percent,
            "max_vertex_position_delta": comparison.max_vertex_position_delta,
            "moved_vertices": list(comparison.moved_vertices),
            "removed_edges": [list(edge) for edge in comparison.removed_edges],
            "added_edges": [list(edge) for edge in comparison.added_edges],
            "position_check": (
                "only vertex e moves between the two timesteps; each timestep "
                "is then triangulated by the same ddgclib Delaunay retopology rule"
            ),
        },
        "snapshots": [
            {
                "step": snapshot.step,
                "name": snapshot.name,
                "hc_dual_vertices": snapshot.hc_dual_vertices,
                "liquid_dual_volume": snapshot.liquid_dual_volume,
                "air_dual_volume": snapshot.air_dual_volume,
                "total_selected_dual_volume": snapshot.total_dual_volume,
                "per_tet": snapshot.per_tet,
            }
            for snapshot in snapshots
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return csv_path, json_path


def run(out_dir: Path, selected_label: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    before_points = build_before_points()
    after_points = build_after_points()

    before_snapshot, _ = retopologized_snapshot(
        0,
        "before_delaunay_retriangulation",
        before_points,
        selected_label,
    )
    after_snapshot, _ = retopologized_snapshot(
        1,
        "after_delaunay_retriangulation",
        after_points,
        selected_label,
    )
    removed_edges, added_edges = changed_edges(before_snapshot, after_snapshot)
    moved_vertices = moved_vertex_labels(before_points, after_points)
    comparison = Comparison(
        delta_liquid_volume=after_snapshot.liquid_dual_volume
        - before_snapshot.liquid_dual_volume,
        delta_liquid_percent=percent_change(
            before_snapshot.liquid_dual_volume,
            after_snapshot.liquid_dual_volume,
        ),
        max_vertex_position_delta=max_position_delta(before_points, after_points),
        removed_edges=removed_edges,
        added_edges=added_edges,
        moved_vertices=moved_vertices,
    )

    before_png = save_snapshot_png(
        out_dir,
        before_points,
        before_snapshot,
        selected_label,
        comparison.moved_vertices,
    )
    after_png = save_snapshot_png(
        out_dir,
        after_points,
        after_snapshot,
        selected_label,
        comparison.moved_vertices,
    )
    comparison_png = save_comparison_png(
        out_dir,
        before_points,
        after_points,
        before_snapshot,
        after_snapshot,
        selected_label,
        comparison,
    )
    interactive_html = save_interactive_html(
        out_dir,
        before_points,
        after_points,
        before_snapshot,
        after_snapshot,
        selected_label,
        comparison,
    )
    csv_path, json_path = write_summary(
        out_dir,
        [before_snapshot, after_snapshot],
        selected_label,
        comparison,
    )

    return {
        "before": before_snapshot,
        "after": after_snapshot,
        "before_png": before_png,
        "after_png": after_png,
        "comparison_png": comparison_png,
        "interactive_html": interactive_html,
        "csv": csv_path,
        "json": json_path,
        "comparison": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Delaunay-valid HC retriangulation dual-volume demo."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for PNG and volume summary files.",
    )
    parser.add_argument(
        "--selected",
        default=SELECTED_LABEL,
        choices=list(INTERFACE_LABELS),
        help="Interface vertex whose barycentric dual volume is shown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.out_dir, args.selected)
    before = result["before"]
    after = result["after"]
    comparison = result["comparison"]

    print("Delaunay-valid HC retriangulation dual-volume demo")
    print(f"selected vertex: {args.selected}")
    print(f"before tetrahedra: {len(before.tets)}")
    print(f"after  tetrahedra: {len(after.tets)}")
    print(f"before liquid dual volume: {before.liquid_dual_volume:.17e}")
    print(f"after  liquid dual volume: {after.liquid_dual_volume:.17e}")
    print(f"delta liquid dual volume: {comparison.delta_liquid_volume:.17e}")
    print(f"delta liquid percent: {comparison.delta_liquid_percent:+.9f}%")
    print(f"moved vertex labels: {comparison.moved_vertices}")
    print(
        "connectivity change: removed "
        f"{comparison.removed_edges}, added {comparison.added_edges}"
    )
    print(
        "max vertex position delta between timesteps: "
        f"{comparison.max_vertex_position_delta:.17e}"
    )
    print(f"before PNG: {result['before_png']}")
    print(f"after PNG: {result['after_png']}")
    print(f"comparison PNG: {result['comparison_png']}")
    print(f"interactive HTML: {result['interactive_html']}")
    print(f"CSV: {result['csv']}")
    print(f"JSON: {result['json']}")


if __name__ == "__main__":
    main()
