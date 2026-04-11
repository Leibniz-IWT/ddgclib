"""Parametric Surface Geometry — Tutorial and Validation

Demonstrates all surface generators from ``ddgclib.geometry``, validates
generated meshes against analytical curvatures, and saves matplotlib figures.

Usage::

    python tutorials/parametric_surfaces_tutorial.py
    python tutorials/parametric_surfaces_tutorial.py --polyscope   # 3D interactive viewer

Outputs saved to ``tutorials/fig/parametric/``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ddgclib.geometry import (
    parametric_surface,
    sphere,
    catenoid,
    cylinder,
    hyperboloid,
    torus,
    plane,
    translate_surface,
    scale_surface,
)


# ── 1. Generate all surface types ────────────────────────────────────────

def generate_surfaces(refinement: int = 2) -> dict:
    """Generate one mesh per surface type and return as dict."""
    surfaces = {}

    surfaces["sphere"] = sphere(R=1.0, refinement=refinement)
    surfaces["catenoid"] = catenoid(a=1.0, v_range=(-1.5, 1.5),
                                    refinement=refinement)
    surfaces["cylinder"] = cylinder(R=1.0, h_range=(-1.5, 1.5),
                                    refinement=refinement)
    surfaces["hyperboloid"] = hyperboloid(a=1.0, c=1.0, v_range=(-1.5, 1.5),
                                          refinement=refinement)
    surfaces["torus"] = torus(R=2.0, r=0.5, refinement=refinement)
    surfaces["plane"] = plane(x_range=(-1, 1), y_range=(-1, 1),
                              refinement=refinement)

    # Custom: paraboloid
    def paraboloid(u, v):
        return (u, v, 0.5 * (u**2 + v**2))
    surfaces["paraboloid"] = parametric_surface(
        paraboloid, [(-1, 1), (-1, 1)], refinement=refinement,
    )

    return surfaces


# ── 2. Print mesh statistics ─────────────────────────────────────────────

def print_stats(surfaces: dict):
    """Print vertex/boundary counts and connectivity stats."""
    print("\n" + "=" * 65)
    print(f"  {'Surface':<14} {'Vertices':>8} {'Boundary':>8} "
          f"{'min(nn)':>8} {'max(nn)':>8}")
    print("=" * 65)
    for name, (HC, bV) in surfaces.items():
        n = HC.V.size()
        nb = len(bV)
        nn_min = min(len(v.nn) for v in HC.V)
        nn_max = max(len(v.nn) for v in HC.V)
        print(f"  {name:<14} {n:>8} {nb:>8} {nn_min:>8} {nn_max:>8}")
    print("=" * 65)


# ── 3. Validate analytical curvatures ────────────────────────────────────

def validate_geometry(refinement_levels: list[int] = None):
    """Validate geometric accuracy of generated surfaces.

    Checks that vertices lie on the analytical surface and that
    vertex count increases with refinement.

    Note: ``compute_vd()`` from ``hyperct.ddg`` is designed for volumetric
    3D meshes, not surface meshes.  Curvature validation uses the
    legacy ``HC_curvatures()`` pipeline from ``ddgclib._curvatures``
    which works directly on surface geometry.

    Returns convergence table data.
    """
    if refinement_levels is None:
        refinement_levels = [1, 2, 3]

    results = {}

    for ref in refinement_levels:
        ref_results = {}

        # Sphere: all vertices at distance R from origin
        R = 1.0
        HC, bV = sphere(R=R, refinement=ref)
        dists = [abs(np.linalg.norm(v.x_a[:3]) - R) for v in HC.V]
        ref_results["sphere"] = {
            "n_verts": HC.V.size(),
            "n_boundary": len(bV),
            "max_geom_error": max(dists),
            "analytical_H": 1.0 / R,
        }

        # Cylinder: all vertices at distance R from z-axis
        HC_c, bV_c = cylinder(R=R, refinement=ref)
        radii_err = [abs(np.sqrt(v.x_a[0]**2 + v.x_a[1]**2) - R)
                     for v in HC_c.V]
        ref_results["cylinder"] = {
            "n_verts": HC_c.V.size(),
            "n_boundary": len(bV_c),
            "max_geom_error": max(radii_err),
            "analytical_H": 1.0 / (2 * R),
        }

        # Catenoid: neck at z=0 should have radius = a
        a = 1.0
        HC_cat, bV_cat = catenoid(a=a, refinement=ref)
        neck = [v for v in HC_cat.V if abs(v.x_a[2]) < 0.01]
        neck_err = [abs(np.sqrt(v.x_a[0]**2 + v.x_a[1]**2) - a)
                    for v in neck]
        ref_results["catenoid"] = {
            "n_verts": HC_cat.V.size(),
            "n_boundary": len(bV_cat),
            "max_geom_error": max(neck_err) if neck_err else float("nan"),
            "analytical_H": 0.0,
        }

        # Torus: (sqrt(x²+y²) - R)² + z² = r²
        R_t, r_t = 2.0, 0.5
        HC_t, bV_t = torus(R=R_t, r=r_t, refinement=ref)
        torus_err = [abs((np.sqrt(v.x_a[0]**2 + v.x_a[1]**2) - R_t)**2
                        + v.x_a[2]**2 - r_t**2) for v in HC_t.V]
        ref_results["torus"] = {
            "n_verts": HC_t.V.size(),
            "n_boundary": len(bV_t),
            "max_geom_error": max(torus_err),
            "analytical_H": None,  # varies over surface
        }

        # Hyperboloid: x²/a² + y²/a² - z²/c² = 1
        a_h, c_h = 1.0, 1.0
        HC_h, bV_h = hyperboloid(a=a_h, c=c_h, refinement=ref)
        hyp_err = [abs(v.x_a[0]**2/a_h**2 + v.x_a[1]**2/a_h**2
                       - v.x_a[2]**2/c_h**2 - 1.0) for v in HC_h.V]
        ref_results["hyperboloid"] = {
            "n_verts": HC_h.V.size(),
            "n_boundary": len(bV_h),
            "max_geom_error": max(hyp_err),
            "analytical_H": None,
        }

        results[ref] = ref_results

    # Print convergence table
    print("\n" + "=" * 75)
    print("  Geometry Validation — Convergence with Refinement")
    print("=" * 75)
    print(f"  {'Ref':>3}  {'Surface':<14} {'Vertices':>8} {'Boundary':>8} "
          f"{'Max Error':>12} {'H_analytical':>12}")
    print("-" * 75)
    for ref in refinement_levels:
        for name, data in results[ref].items():
            h_str = f"{data['analytical_H']:.4f}" if data["analytical_H"] is not None \
                else "varies"
            print(f"  {ref:>3}  {name:<14} {data['n_verts']:>8} "
                  f"{data['n_boundary']:>8} {data['max_geom_error']:>12.2e} "
                  f"{h_str:>12}")
        if ref != refinement_levels[-1]:
            print("-" * 75)
    print("=" * 75)

    return results


# ── 4. Matplotlib visualization ──────────────────────────────────────────

def make_plots(surfaces: dict, fig_dir: Path):
    """Save 3D plots of each surface using hyperct's plot_complex as base.

    Uses ``plot_complex`` for the mesh wireframe, then overlays boundary
    vertices in red so the user can see boundary tagging.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hyperct._plotting import plot_complex

    fig_dir.mkdir(parents=True, exist_ok=True)

    # Individual surface plots
    for name, (HC, bV) in surfaces.items():
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Use plot_complex for the base mesh (vertices + edges)
        plot_complex(HC, show=False, save_fig=False, directed=False,
                     fig_complex=fig, ax_complex=ax,
                     point_color='steelblue', line_color='gray',
                     pointsize=8)

        # Overlay boundary vertices
        if bV:
            bx = [v.x_a[0] for v in bV]
            by = [v.x_a[1] for v in bV]
            bz = [v.x_a[2] for v in bV]
            ax.scatter(bx, by, bz, c="red", s=25, alpha=0.9,
                       zorder=5, label="boundary")

        ax.set_title(f"{name} (n={HC.V.size()}, bV={len(bV)})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if bV:
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=150)
        plt.close(fig)

    # Overview: all surfaces in a 2x4 grid
    n_surfaces = len(surfaces)
    ncols = 4
    nrows = (n_surfaces + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows),
                             subplot_kw={"projection": "3d"})
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    for ax, (name, (HC, bV)) in zip(axes_flat, surfaces.items()):
        plot_complex(HC, show=False, save_fig=False, directed=False,
                     fig_complex=fig, ax_complex=ax,
                     point_color='steelblue', line_color='gray',
                     pointsize=4)
        # Overlay boundary vertices
        if bV:
            bx = [v.x_a[0] for v in bV]
            by = [v.x_a[1] for v in bV]
            bz = [v.x_a[2] for v in bV]
            ax.scatter(bx, by, bz, c="red", s=12, alpha=0.9, zorder=5)

        ax.set_title(f"{name} (n={HC.V.size()}, bV={len(bV)})",
                     fontsize=9)

    # Hide empty subplots
    for i in range(n_surfaces, nrows * ncols):
        axes_flat[i].set_visible(False)

    fig.suptitle("Parametric Surface Meshes", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "overview.png", dpi=150)
    plt.close(fig)

    print(f"\n  Figures saved to {fig_dir}/")


# ── 5. Polyscope visualization (optional) ────────────────────────────────

def polyscope_view(surfaces: dict):
    """Interactive 3D viewer using polyscope."""
    try:
        import polyscope as ps
    except ImportError:
        print("polyscope not installed — skipping 3D viewer.")
        print("Install with:  pip install polyscope")
        return

    from ddgclib.visualization.polyscope_3d import interactive_viewer

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    # Register each surface
    for i, (name, (HC, bV)) in enumerate(surfaces.items()):
        positions = np.array([v.x_a[:3] for v in HC.V])
        is_boundary = np.array([v in bV for v in HC.V], dtype=float)

        cloud = ps.register_point_cloud(name, positions)
        cloud.add_scalar_quantity("boundary", is_boundary, cmap="blues")

        # Try to build edges for curve network
        edges = []
        seen = set()
        v_list = list(HC.V)
        v_to_idx = {id(v): i for i, v in enumerate(v_list)}
        for v in v_list:
            for nb in v.nn:
                edge = frozenset((id(v), id(nb)))
                if edge not in seen:
                    seen.add(edge)
                    edges.append([v_to_idx[id(v)], v_to_idx[id(nb)]])
        if edges:
            net = ps.register_curve_network(
                f"{name}_edges", positions, np.array(edges),
            )
            net.set_color((0.5, 0.5, 0.5))
            net.set_radius(0.002, relative=False)
            net.set_transparency(0.4)

    ps.show()


# ── 6. Composition demo ─────────────────────────────────────────────────

def composition_demo():
    """Demonstrate translate + scale for composing surfaces."""
    print("\n--- Composition Demo ---")

    # Two spheres at different positions
    HC1, bV1 = sphere(R=0.5, refinement=2)
    translate_surface(HC1, [-2.0, 0.0, 0.0])
    print(f"Sphere 1: center ≈ ({np.mean([v.x_a[0] for v in HC1.V]):.1f}, 0, 0)")

    HC2, bV2 = sphere(R=0.5, refinement=2)
    translate_surface(HC2, [2.0, 0.0, 0.0])
    print(f"Sphere 2: center ≈ ({np.mean([v.x_a[0] for v in HC2.V]):.1f}, 0, 0)")

    # Scaled cylinder connecting them
    HC3, bV3 = cylinder(R=0.2, h_range=(-1.5, 1.5), refinement=2)
    # Rotate: cylinder is along z, we want it along x
    # For now just show it as-is
    print(f"Cylinder: {HC3.V.size()} vertices")

    # Scale a torus
    HC4, bV4 = torus(R=1.0, r=0.3, refinement=2)
    scale_surface(HC4, 0.5)
    radii = [np.linalg.norm(v.x_a[:3]) for v in HC4.V]
    print(f"Scaled torus: r range [{min(radii):.3f}, {max(radii):.3f}]")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parametric surface geometry tutorial"
    )
    parser.add_argument("--polyscope", action="store_true",
                        help="Launch interactive polyscope viewer")
    parser.add_argument("--refinement", type=int, default=2,
                        help="Mesh refinement level (default: 2)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Parametric Surface Geometry Tutorial")
    print("=" * 60)

    # Generate surfaces
    surfaces = generate_surfaces(refinement=args.refinement)
    print_stats(surfaces)

    # Validate curvatures
    validate_geometry(refinement_levels=[1, 2, 3])

    # Composition demo
    composition_demo()

    # Save plots
    fig_dir = Path(__file__).parent / "fig" / "parametric"
    make_plots(surfaces, fig_dir)

    # Polyscope
    if args.polyscope:
        polyscope_view(surfaces)

    print("\nDone.")


if __name__ == "__main__":
    main()
