"""Polyscope 3D visualization of the liquid bridge case study.

Loads results from ``results/history.json`` and animates the two-particle
approach and bridge formation using polyscope.  Uses the library's
:func:`~ddgclib.visualization.polyscope_3d.interactive_viewer` for the
frame slider, play/pause, and speed controls.

Usage::

    python visualize_bridge.py               # interactive
    python visualize_bridge.py --screenshot   # save frames to fig/frames/

Requires ``polyscope`` (pip install polyscope).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import polyscope as ps
except ImportError:
    print("polyscope is required for 3D visualization.")
    print("Install with:  pip install polyscope")
    sys.exit(1)


def load_history(path: Path | None = None) -> list[dict]:
    """Load history from JSON."""
    if path is None:
        path = Path(__file__).parent / "results" / "history.json"
    return json.loads(path.read_text())


def visualize(history: list[dict], screenshot: bool = False):
    """Animate the two-particle liquid bridge simulation in polyscope."""
    from ddgclib.visualization.polyscope_3d import interactive_viewer
    from cases_dynamic.liquid_bridge_dem.src._params import R

    n_frames = len(history)
    all_x1 = np.array([h["x1"] for h in history])
    all_x2 = np.array([h["x2"] for h in history])
    bridge_active = np.array([h["n_bridges"] > 0 for h in history])

    # ── Polyscope setup ──────────────────────────────────────────────
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    positions = np.vstack([all_x1[0], all_x2[0]])
    cloud = ps.register_point_cloud("particles", positions)
    cloud.add_scalar_quantity("radius", np.array([R, R]))
    cloud.set_point_radius_quantity("radius", autoscale=False)
    cloud.set_color((0.3, 0.5, 0.8))

    bridge_net_ref = {"net": None}

    # ── Callbacks ─────────────────────────────────────────────────────
    def update(idx):
        pos = np.vstack([all_x1[idx], all_x2[idx]])
        cloud.update_point_positions(pos)

        if bridge_active[idx]:
            bridge_net_ref["net"] = ps.register_curve_network(
                "bridge", pos, np.array([[0, 1]]),
            )
            bridge_net_ref["net"].set_color((0.0, 1.0, 1.0))
            bridge_net_ref["net"].set_radius(R * 0.15)
        else:
            if bridge_net_ref["net"] is not None:
                ps.remove_curve_network("bridge")
                bridge_net_ref["net"] = None

    def info(idx):
        import polyscope.imgui as imgui

        h = history[idx]
        imgui.Text(f"t = {h['t']*1e3:.3f} ms")
        imgui.Text(f"sep = {h['sep']*1e6:.1f} um")
        imgui.Text(f"bridges = {h['n_bridges']}")
        imgui.Text(f"F_cap = {h['capillary_force_mag']*1e6:.2f} uN")

    # ── Launch viewer ─────────────────────────────────────────────────
    screenshot_dir = None
    if screenshot:
        screenshot_dir = Path(__file__).parent / "fig" / "frames"

    interactive_viewer(
        n_frames=n_frames,
        update_fn=update,
        info_fn=info,
        screenshot_dir=screenshot_dir,
        init=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Liquid bridge polyscope viz")
    parser.add_argument("--screenshot", action="store_true",
                        help="Save frames to fig/frames/")
    parser.add_argument("--history", type=str, default=None,
                        help="Path to history.json")
    args = parser.parse_args()

    history_path = Path(args.history) if args.history else None
    history = load_history(history_path)
    print(f"Loaded {len(history)} frames")
    visualize(history, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
