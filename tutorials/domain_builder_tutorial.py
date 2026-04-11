#!/usr/bin/env python
"""Domain Builder Tutorial
========================

Demonstrates how to construct common CFD simulation domains using the
``ddgclib.geometry.domains`` module.  Each domain builder returns a
:class:`DomainResult` with the simplicial complex, boundary vertices,
and named boundary groups ready for use with initial conditions,
boundary conditions, and integrators.

Sections
--------
1. Quick start — build and inspect each domain type
2. Customization — refinement levels, distribution laws, custom origins
3. Composition — combining with transforms and extrusion
4. Full simulation setup — domain + ICs + BCs + integrator
5. Agent recipe — the standard pattern for AI-generated geometry

Run
---
    python tutorials/domain_builder_tutorial.py

Requires: numpy, matplotlib (optional, for plots).
"""

import sys
import numpy as np

# ── Section 1: Quick Start ──────────────────────────────────────────────

def quick_start():
    """Build and inspect every available domain type."""
    from ddgclib.geometry.domains import (
        rectangle, l_shape, disk, annulus,
        box, cylinder_volume, pipe, ball,
    )

    print("=" * 60)
    print("Section 1: Quick Start — All Domain Types")
    print("=" * 60)

    # --- 2D Domains ---

    # 2D rectangular channel
    r = rectangle(L=4.0, h=1.0, refinement=2, flow_axis=0)
    print(f"\nrectangle:  {r.summary()}")
    print(f"  groups: {list(r.boundary_groups.keys())}")

    # L-shaped domain
    r = l_shape(L=2.0, h=1.0, notch_L=1.0, notch_h=0.5, refinement=2)
    print(f"\nl_shape:    {r.summary()}")
    print(f"  groups: {list(r.boundary_groups.keys())}")

    # Filled disk
    r = disk(R=1.0, refinement=2)
    print(f"\ndisk:       {r.summary()}")
    print(f"  groups: {list(r.boundary_groups.keys())}")

    # Annular ring
    r = annulus(R_outer=1.0, R_inner=0.3, refinement=2)
    print(f"\nannulus:    {r.summary()}")
    print(f"  groups: {list(r.boundary_groups.keys())}")

    # --- 3D Domains ---

    # 3D box
    r = box(Lx=2.0, Ly=1.0, Lz=1.0, refinement=1, flow_axis=0)
    print(f"\nbox:        {r.summary()}")
    print(f"  groups: {list(r.boundary_groups.keys())}")

    # Filled cylinder
    r = cylinder_volume(R=0.5, L=2.0, refinement=1, flow_axis=2)
    print(f"\ncylinder:   {r.summary()}")
    print(f"  groups: {list(r.boundary_groups.keys())}")

    # Long pipe (uses extrusion)
    r = pipe(R=0.5, L=5.0, refinement=1, flow_axis=2)
    print(f"\npipe:       {r.summary()}")
    print(f"  groups: {list(r.boundary_groups.keys())}")

    # Filled sphere
    r = ball(R=1.0, refinement=1)
    print(f"\nball:       {r.summary()}")
    print(f"  groups: {list(r.boundary_groups.keys())}")


# ── Section 2: Customization ────────────────────────────────────────────

def customization():
    """Show how refinement, distribution laws, and origins affect domains."""
    from ddgclib.geometry.domains import rectangle, disk, cylinder_volume

    print("\n" + "=" * 60)
    print("Section 2: Customization")
    print("=" * 60)

    # Refinement convergence
    print("\nRectangle vertex count vs refinement:")
    for ref in [1, 2, 3, 4]:
        r = rectangle(L=1.0, h=1.0, refinement=ref)
        print(f"  refinement={ref}: {r.HC.V.size():5d} vertices")

    # Distribution laws for disks
    print("\nDisk vertex distributions (R=1.0, refinement=2):")
    for law in ['sinusoidal', 'linear', 'power', 'log']:
        r = disk(R=1.0, refinement=2, distr_law=law)
        # Compute mean distance of interior vertices from center
        dists = []
        for v in r.HC.V:
            if v not in r.bV:
                dists.append(np.linalg.norm(v.x_a[:2]))
        mean_dist = np.mean(dists) if dists else 0
        print(f"  {law:12s}: mean interior distance = {mean_dist:.4f}")

    # Custom origin
    print("\nRectangle with custom origin (5.0, 2.0):")
    r = rectangle(L=2.0, h=1.0, origin=(5.0, 2.0), refinement=1)
    xs = [v.x_a[0] for v in r.HC.V]
    ys = [v.x_a[1] for v in r.HC.V]
    print(f"  x range: [{min(xs):.1f}, {max(xs):.1f}]")
    print(f"  y range: [{min(ys):.1f}, {max(ys):.1f}]")

    # Flow axis direction
    print("\nCylinder with different flow axes:")
    for ax in [0, 1, 2]:
        r = cylinder_volume(R=0.5, L=1.0, refinement=1, flow_axis=ax)
        inlet_ax = [v.x_a[ax] for v in r.boundary_groups['inlet']]
        print(f"  flow_axis={ax}: inlet at axis[{ax}] = {min(inlet_ax):.1f}")


# ── Section 3: Composition ──────────────────────────────────────────────

def composition():
    """Combine domains with transforms and extrusion."""
    from ddgclib.geometry.domains import disk, rectangle
    from ddgclib.geometry import translate_surface, scale_surface

    print("\n" + "=" * 60)
    print("Section 3: Composition with Transforms")
    print("=" * 60)

    # Translate a disk
    r = disk(R=0.5, refinement=2)
    print(f"\nDisk before translate: center approx "
          f"({np.mean([v.x_a[0] for v in r.HC.V]):.2f}, "
          f"{np.mean([v.x_a[1] for v in r.HC.V]):.2f})")

    translate_surface(r.HC, [3.0, 2.0])
    print(f"Disk after translate [3, 2]: center approx "
          f"({np.mean([v.x_a[0] for v in r.HC.V]):.2f}, "
          f"{np.mean([v.x_a[1] for v in r.HC.V]):.2f})")

    # Scale a rectangle
    r = rectangle(L=1.0, h=1.0, refinement=1)
    scale_surface(r.HC, 2.0, center=[0.5, 0.5])
    xs = [v.x_a[0] for v in r.HC.V]
    ys = [v.x_a[1] for v in r.HC.V]
    print(f"\nRectangle scaled 2x about center: "
          f"x=[{min(xs):.1f}, {max(xs):.1f}], "
          f"y=[{min(ys):.1f}, {max(ys):.1f}]")


# ── Section 4: Full Simulation Setup ────────────────────────────────────

def full_simulation_setup():
    """Complete flow setup: domain + ICs + BCs + integrator."""
    from ddgclib.geometry.domains import rectangle
    from ddgclib._boundary_conditions import (
        BoundaryConditionSet,
        NoSlipWallBC,
    )
    from ddgclib.initial_conditions import (
        CompositeIC,
        PoiseuillePlanar,
        LinearPressureGradient,
        UniformMass,
    )
    from hyperct.ddg import compute_vd
    from functools import partial
    from ddgclib.operators.stress import dudt_i

    print("\n" + "=" * 60)
    print("Section 4: Full 2D Poiseuille Flow Setup")
    print("=" * 60)

    # Parameters
    G = 1.0     # pressure gradient
    mu = 0.1    # dynamic viscosity
    rho = 1.0   # density
    dim = 2

    # Step 1: Domain
    result = rectangle(L=2.0, h=1.0, refinement=3, flow_axis=0)
    HC, bV = result.HC, result.bV
    L = result.metadata['L']
    h = result.metadata['h']
    print(f"\n1. Domain: {result.summary()}")

    # Step 2: Boundary conditions
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), result.boundary_groups['walls'])
    print(f"2. BCs: NoSlipWallBC on {len(result.boundary_groups['walls'])} wall vertices")

    # Step 3: Initial conditions
    ic = CompositeIC(
        PoiseuillePlanar(G=G, mu=mu, y_lb=0.0, y_ub=h,
                         flow_axis=0, normal_axis=1, dim=dim),
        LinearPressureGradient(G=G, axis=0),
        UniformMass(total_volume=result.metadata['volume'], rho=rho),
    )
    ic.apply(HC, bV)
    print("3. ICs: Poiseuille profile + linear pressure gradient + uniform mass")

    # Step 4: Compute dual mesh
    compute_vd(HC, cdist=1e-10)
    print("4. Dual mesh computed")

    # Step 5: Set up acceleration function
    dudt_fn = partial(dudt_i, dim=dim, mu=mu, HC=HC)
    print("5. Acceleration function ready (dudt_i with partial binding)")

    # Verify: check velocity profile at inlet
    inlet_v = list(result.boundary_groups['inlet'])
    if inlet_v:
        u_vals = [v.u[0] for v in inlet_v if hasattr(v, 'u')]
        if u_vals:
            print(f"\n   Inlet velocity range: [{min(u_vals):.4f}, {max(u_vals):.4f}]")
            U_max = G * h**2 / (8 * mu)
            print(f"   Analytical U_max: {U_max:.4f}")

    print("\nSimulation setup complete. Ready for time integration.")


# ── Section 5: Agent Recipe ─────────────────────────────────────────────

def agent_recipe():
    """The standard pattern for AI-generated geometry.

    This is the recommended pattern for AI agents building simulation
    domains from natural language descriptions.
    """
    print("\n" + "=" * 60)
    print("Section 5: Agent Recipe")
    print("=" * 60)

    print("""
Standard pattern for an AI agent building a CFD domain:

    from ddgclib.geometry.domains import rectangle  # or disk, cylinder_volume, etc.

    # 1. Build domain (one line)
    result = rectangle(L=10.0, h=1.0, refinement=3, flow_axis=0)

    # 2. Unpack
    HC, bV = result.HC, result.bV

    # 3. Set up BCs using named boundary groups
    from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=2), result.boundary_groups['walls'])

    # 4. Set up ICs
    from ddgclib.initial_conditions import CompositeIC, ZeroVelocity, UniformMass
    ic = CompositeIC(
        ZeroVelocity(dim=2),
        UniformMass(total_volume=result.metadata['volume'], rho=1.0),
    )
    ic.apply(HC, bV)

    # 5. Compute dual mesh
    from hyperct.ddg import compute_vd
    compute_vd(HC, cdist=1e-10)

    # 6. Set up integrator
    from functools import partial
    from ddgclib.operators.stress import dudt_i
    dudt_fn = partial(dudt_i, dim=2, mu=0.1, HC=HC)

Available domains:
  2D: rectangle(), l_shape(), disk(), annulus()
  3D: box(), cylinder_volume(), pipe(), ball()

All return DomainResult with:
  .HC               — simplicial complex
  .bV               — all boundary vertices
  .boundary_groups  — dict of named vertex sets
  .metadata         — dict with volume, dimensions, etc.
  .summary()        — one-line description
""")


# ── Visualization (optional) ────────────────────────────────────────────

def make_plots():
    """Generate visualization of all domain types (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nSkipping plots (matplotlib not available)")
        return

    from ddgclib.geometry.domains import (
        rectangle, l_shape, disk, annulus,
        box, cylinder_volume, ball,
    )
    from pathlib import Path

    fig_dir = Path("tutorials/fig/domains")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 2D domains
    domains_2d = {
        'rectangle': rectangle(L=2.0, h=1.0, refinement=3),
        'l_shape': l_shape(L=2.0, h=1.0, refinement=3),
        'disk': disk(R=1.0, refinement=3),
        'annulus': annulus(R_outer=1.0, R_inner=0.3, refinement=3),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, result) in zip(axes.flat, domains_2d.items()):
        xs = [v.x_a[0] for v in result.HC.V]
        ys = [v.x_a[1] for v in result.HC.V]
        bxs = [v.x_a[0] for v in result.bV]
        bys = [v.x_a[1] for v in result.bV]

        # Plot edges
        for v in result.HC.V:
            for nb in v.nn:
                ax.plot([v.x_a[0], nb.x_a[0]], [v.x_a[1], nb.x_a[1]],
                        'b-', alpha=0.2, linewidth=0.5)
        ax.plot(xs, ys, 'b.', markersize=2, label='interior')
        ax.plot(bxs, bys, 'r.', markersize=4, label='boundary')
        ax.set_title(f"{name} ({result.HC.V.size()} vertices)")
        ax.set_aspect('equal')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / "domains_2d.png", dpi=150)
    plt.close(fig)
    print(f"\n2D domain plots saved to {fig_dir / 'domains_2d.png'}")

    # 3D domains
    domains_3d = {
        'box': box(Lx=2.0, Ly=1.0, Lz=1.0, refinement=1),
        'cylinder': cylinder_volume(R=0.5, L=1.0, refinement=1),
        'ball': ball(R=1.0, refinement=1),
    }

    fig = plt.figure(figsize=(15, 5))
    for i, (name, result) in enumerate(domains_3d.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        xs = [v.x_a[0] for v in result.HC.V]
        ys = [v.x_a[1] for v in result.HC.V]
        zs = [v.x_a[2] for v in result.HC.V]
        bxs = [v.x_a[0] for v in result.bV]
        bys = [v.x_a[1] for v in result.bV]
        bzs = [v.x_a[2] for v in result.bV]

        ax.scatter(xs, ys, zs, c='blue', s=2, alpha=0.3, label='interior')
        ax.scatter(bxs, bys, bzs, c='red', s=8, alpha=0.6, label='boundary')
        ax.set_title(f"{name} ({result.HC.V.size()} verts)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / "domains_3d.png", dpi=150)
    plt.close(fig)
    print(f"3D domain plots saved to {fig_dir / 'domains_3d.png'}")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    quick_start()
    customization()
    composition()
    full_simulation_setup()
    agent_recipe()
    make_plots()

    print("\n" + "=" * 60)
    print("Tutorial complete.")
    print("=" * 60)
