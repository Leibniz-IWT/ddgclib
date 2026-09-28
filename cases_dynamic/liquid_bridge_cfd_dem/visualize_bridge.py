"""Polyscope 3D visualization of the CFD-DEM liquid bridge case.

Shows the DEM particles and the dynamic fluid film surface mesh with
curvature scalar field and velocity vectors.  Uses the library's
:func:`~ddgclib.visualization.polyscope_3d.interactive_viewer` with a
frame slider, play/pause, and speed controls.

Usage::

    python visualize_bridge.py
    python visualize_bridge.py --screenshot

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
    if path is None:
        path = Path(__file__).parent / "results" / "history.json"
    return json.loads(path.read_text())


def _triangles_from_edges(edges: list[list[int]], n_verts: int) -> np.ndarray:
    """Find triangles from an edge list (mutual-neighbour triples)."""
    adj = [set() for _ in range(n_verts)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    triangles = set()
    for i in range(n_verts):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k <= j:
                    continue
                triangles.add((i, j, k))

    return np.array(sorted(triangles)) if triangles else np.empty((0, 3), dtype=int)


def visualize(history: list[dict], screenshot: bool = False):
    """Animate particles + fluid film mesh in polyscope."""
    from ddgclib.visualization.polyscope_3d import interactive_viewer
    from cases_dynamic.liquid_bridge_cfd_dem.src._params import R

    n_frames = len(history)

    # Pre-extract particle data
    all_x1 = np.array([h["x1"] for h in history])
    all_x2 = np.array([h["x2"] for h in history])

    # Pre-extract film data
    has_film = "film" in history[0]

    # ── Polyscope setup ──────────────────────────────────────────────
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    # -- Particles --
    particle_pos = np.vstack([all_x1[0], all_x2[0]])
    cloud = ps.register_point_cloud("particles", particle_pos)
    cloud.add_scalar_quantity("radius", np.array([R, R]))
    cloud.set_point_radius_quantity("radius", autoscale=False)
    cloud.set_color((0.3, 0.5, 0.8))

    # -- Film mesh --
    film_mesh_ref = {"mesh": None, "net": None}

    def _register_film(film_data):
        """Register the film surface mesh in polyscope."""
        # Remove old
        if film_mesh_ref["mesh"] is not None:
            try:
                ps.remove_surface_mesh("film")
            except Exception:
                pass
        if film_mesh_ref["net"] is not None:
            try:
                ps.remove_curve_network("film_edges")
            except Exception:
                pass

        verts = np.array(film_data["vertices"])
        edges = np.array(film_data["edges"]) if film_data["edges"] else np.empty((0, 2), int)
        n_v = len(verts)

        triangles = _triangles_from_edges(film_data["edges"], n_v)

        if len(triangles) > 0:
            mesh = ps.register_surface_mesh("film", verts, triangles)

            # Color by particle_id
            pids = np.array(film_data.get("particle_ids", [0] * n_v), dtype=float)
            mesh.add_scalar_quantity(
                "particle_id", pids,
                defined_on="vertices", cmap="coolwarm", enabled=True,
            )

            # Curvature
            if film_data.get("curvature"):
                curv = np.array(film_data["curvature"])
                mesh.add_scalar_quantity(
                    "curvature", curv,
                    defined_on="vertices", cmap="viridis", enabled=False,
                )

            # Velocity vectors
            if film_data.get("velocities"):
                vel = np.array(film_data["velocities"])
                mesh.add_vector_quantity(
                    "velocity", vel, defined_on="vertices",
                    vectortype="standard", enabled=False,
                )

            # Boundary
            if film_data.get("boundary"):
                bnd = np.array(film_data["boundary"], dtype=float)
                mesh.add_scalar_quantity(
                    "boundary", bnd,
                    defined_on="vertices", cmap="blues", enabled=False,
                )

            mesh.set_transparency(0.5)
            mesh.set_smooth_shade(True)
            film_mesh_ref["mesh"] = mesh
        elif len(edges) > 0:
            net = ps.register_curve_network("film_edges", verts, edges)
            net.set_color((0.4, 0.6, 0.9))
            net.set_radius(0.00005, relative=False)
            film_mesh_ref["net"] = net

    if has_film:
        _register_film(history[0]["film"])

    # -- Bridge indicator --
    bridge_net_ref = {"net": None}

    # ── Callbacks ─────────────────────────────────────────────────────
    def update(idx):
        # Update particles
        pos = np.vstack([all_x1[idx], all_x2[idx]])
        cloud.update_point_positions(pos)

        # Update bridge indicator
        if history[idx]["n_bridges"] > 0:
            bridge_net_ref["net"] = ps.register_curve_network(
                "bridge", pos, np.array([[0, 1]]),
            )
            bridge_net_ref["net"].set_color((0.0, 1.0, 1.0))
            bridge_net_ref["net"].set_radius(R * 0.15)
        else:
            if bridge_net_ref["net"] is not None:
                ps.remove_curve_network("bridge")
                bridge_net_ref["net"] = None

        # Update film mesh
        if has_film and "film" in history[idx]:
            _register_film(history[idx]["film"])

    def info(idx):
        import polyscope.imgui as imgui

        h = history[idx]
        imgui.Text(f"t = {h['t']*1e3:.3f} ms")
        imgui.Text(f"sep = {h['sep']*1e6:.1f} um")
        imgui.Text(f"DEM bridges = {h['n_bridges']}")
        imgui.Separator()
        imgui.Text(f"|F_cap| = {h['capillary_force_mag']*1e6:.4f} uN")

        if h.get("film_bridge_formed"):
            imgui.Text("Film bridge: FORMED")
        else:
            imgui.Text("Film bridge: not formed")

        if has_film and "film" in h:
            film = h["film"]
            imgui.Separator()
            imgui.Text(f"Film: {film['n_vertices']} verts, {film['n_edges']} edges")

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
    parser = argparse.ArgumentParser(
        description="CFD-DEM liquid bridge polyscope visualization"
    )
    parser.add_argument("--screenshot", action="store_true")
    parser.add_argument("--history", type=str, default=None)
    args = parser.parse_args()

    history_path = Path(args.history) if args.history else None
    history = load_history(history_path)
    print(f"Loaded {len(history)} frames")
    if "film" in history[0]:
        film = history[0]["film"]
        #print(f"Film mesh: {film['n_vertices']} vertices, {film['n_edges']} edges")
    visualize(history, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
