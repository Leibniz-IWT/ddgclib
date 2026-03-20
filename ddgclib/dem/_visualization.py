"""Visualization utilities for DEM particles.

Provides matplotlib-based scatter plots for 2D/3D particle systems,
and optional polyscope point cloud rendering.

Usage
-----
::

    from ddgclib.dem import plot_particles, plot_bridges, plot_bonds

    fig, ax = plot_particles(ps)
    plot_bridges(ps, bridge_mgr, ax=ax)
    plt.show()
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ddgclib.dem._particle import ParticleSystem


def plot_particles(
    ps: ParticleSystem,
    ax=None,
    color: str = "steelblue",
    alpha: float = 0.6,
    show_velocity: bool = False,
    velocity_scale: float = 1.0,
    title: Optional[str] = None,
):
    """Plot particles as circles (2D) or spheres (3D).

    Parameters
    ----------
    ps : ParticleSystem
    ax : matplotlib Axes or None
        If ``None``, creates a new figure.
    color : str
        Particle fill colour.
    alpha : float
        Transparency.
    show_velocity : bool
        If ``True``, draw velocity quivers.
    velocity_scale : float
        Quiver arrow scaling.
    title : str or None
        Plot title.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    dim = ps.dim
    if dim not in (2, 3):
        raise ValueError(f"Plotting supports dim=2 or 3, got {dim}")

    positions = ps.positions()
    radii = ps.radii()

    if ax is None:
        fig = plt.figure()
        if dim == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    if dim == 2:
        # Draw circles
        for i, p in enumerate(ps.particles):
            circle = plt.Circle(
                (positions[i, 0], positions[i, 1]),
                radii[i],
                color=color,
                alpha=alpha,
                fill=True,
            )
            ax.add_patch(circle)

        if show_velocity:
            vels = ps.velocities()
            ax.quiver(
                positions[:, 0],
                positions[:, 1],
                vels[:, 0],
                vels[:, 1],
                scale=1.0 / max(velocity_scale, 1e-30),
                color="red",
                alpha=0.7,
            )

        ax.set_aspect("equal")
        ax.autoscale_view()

    else:  # 3D
        # Scatter with size proportional to radius
        sizes = radii * 1000  # scale for visibility
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=sizes**2,
            c=color,
            alpha=alpha,
        )

        if show_velocity:
            vels = ps.velocities()
            ax.quiver(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                vels[:, 0],
                vels[:, 1],
                vels[:, 2],
                length=velocity_scale,
                color="red",
                alpha=0.7,
            )

    if title:
        ax.set_title(title)

    return fig, ax


def plot_bridges(
    ps: ParticleSystem,
    bridge_manager,
    ax=None,
    color: str = "cyan",
    linewidth: float = 1.5,
    alpha: float = 0.8,
):
    """Draw lines between particles connected by liquid bridges.

    Parameters
    ----------
    ps : ParticleSystem
    bridge_manager : LiquidBridgeManager
    ax : matplotlib Axes or None
    color : str
    linewidth : float
    alpha : float

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    dim = ps.dim

    if ax is None:
        fig = plt.figure()
        if dim == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    for bridge in bridge_manager.bridges:
        if not bridge.active:
            continue
        xi = bridge.p_i.x_a[:dim]
        xj = bridge.p_j.x_a[:dim]

        if dim == 2:
            ax.plot(
                [xi[0], xj[0]], [xi[1], xj[1]],
                color=color, linewidth=linewidth, alpha=alpha,
            )
        else:
            ax.plot(
                [xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]],
                color=color, linewidth=linewidth, alpha=alpha,
            )

    return fig, ax


def plot_bonds(
    ps: ParticleSystem,
    bond_manager,
    ax=None,
    color: str = "orange",
    linewidth: float = 2.0,
    alpha: float = 0.8,
    show_neck: bool = False,
):
    """Draw sintered bonds between particles.

    Parameters
    ----------
    ps : ParticleSystem
    bond_manager : BondManager
    ax : matplotlib Axes or None
    color : str
    linewidth : float
    alpha : float
    show_neck : bool
        If ``True``, scale line width proportional to neck_radius.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    dim = ps.dim

    if ax is None:
        fig = plt.figure()
        if dim == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    for bond in bond_manager.bonds:
        if not bond.active:
            continue
        xi = bond.p_i.x_a[:dim]
        xj = bond.p_j.x_a[:dim]
        lw = linewidth
        if show_neck:
            # Scale linewidth by neck/radius ratio
            r_min = min(bond.p_i.radius, bond.p_j.radius)
            lw = max(0.5, linewidth * bond.neck_radius / max(r_min, 1e-30))

        if dim == 2:
            ax.plot(
                [xi[0], xj[0]], [xi[1], xj[1]],
                color=color, linewidth=lw, alpha=alpha,
            )
        else:
            ax.plot(
                [xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]],
                color=color, linewidth=lw, alpha=alpha,
            )

    return fig, ax


def plot_particles_polyscope(
    ps: ParticleSystem,
    name: str = "DEM particles",
    color: Optional[tuple[float, float, float]] = None,
    radius_scale: float = 1.0,
):
    """Register particles as a polyscope point cloud.

    Requires ``polyscope`` to be installed.

    Parameters
    ----------
    ps : ParticleSystem
    name : str
        Point cloud name in polyscope.
    color : tuple or None
        RGB colour (0-1).
    radius_scale : float
        Scale factor for particle radii.
    """
    import polyscope as pols

    positions = ps.positions()
    radii = ps.radii() * radius_scale

    cloud = pols.register_point_cloud(name, positions)
    cloud.add_scalar_quantity("radius", radii, enabled=True)

    if color is not None:
        cloud.set_color(color)

    return cloud


def plot_bridges_polyscope(
    ps: ParticleSystem,
    bridge_manager,
    name: str = "bridges",
    color: tuple[float, float, float] = (0.0, 1.0, 1.0),
    radius: float = 0.0002,
):
    """Register liquid bridge connections as a polyscope curve network.

    Parameters
    ----------
    ps : ParticleSystem
    bridge_manager : LiquidBridgeManager
    name : str
        Curve network name in polyscope.
    color : tuple
        RGB colour (0-1).
    radius : float
        Tube radius for the curve network.

    Returns
    -------
    net : polyscope CurveNetwork or None
        ``None`` if no active bridges.
    """
    import polyscope as pols

    dim = ps.dim
    active = [b for b in bridge_manager.bridges if b.active]
    if not active:
        return None

    # Collect unique node positions from bridge endpoints
    node_map: dict[int, int] = {}  # particle id → node index
    nodes: list[np.ndarray] = []

    def _get_node(p):
        if p.id not in node_map:
            node_map[p.id] = len(nodes)
            nodes.append(p.x_a[:dim] if dim == 3 else np.append(p.x_a[:dim], 0.0))
        return node_map[p.id]

    edges = []
    for b in active:
        i = _get_node(b.p_i)
        j = _get_node(b.p_j)
        edges.append([i, j])

    nodes_arr = np.array(nodes)
    edges_arr = np.array(edges)

    net = pols.register_curve_network(name, nodes_arr, edges_arr)
    net.set_color(color)
    net.set_radius(radius)
    return net


def plot_bonds_polyscope(
    ps: ParticleSystem,
    bond_manager,
    name: str = "bonds",
    color: tuple[float, float, float] = (1.0, 0.5, 0.0),
    radius: float = 0.0003,
):
    """Register sintered bonds as a polyscope curve network.

    Parameters
    ----------
    ps : ParticleSystem
    bond_manager : BondManager
    name : str
        Curve network name in polyscope.
    color : tuple
        RGB colour (0-1).
    radius : float
        Tube radius for the curve network.

    Returns
    -------
    net : polyscope CurveNetwork or None
        ``None`` if no active bonds.
    """
    import polyscope as pols

    dim = ps.dim
    active = [b for b in bond_manager.bonds if b.active]
    if not active:
        return None

    node_map: dict[int, int] = {}
    nodes: list[np.ndarray] = []

    def _get_node(p):
        if p.id not in node_map:
            node_map[p.id] = len(nodes)
            nodes.append(p.x_a[:dim] if dim == 3 else np.append(p.x_a[:dim], 0.0))
        return node_map[p.id]

    edges = []
    for b in active:
        i = _get_node(b.p_i)
        j = _get_node(b.p_j)
        edges.append([i, j])

    nodes_arr = np.array(nodes)
    edges_arr = np.array(edges)

    net = pols.register_curve_network(name, nodes_arr, edges_arr)
    net.set_color(color)
    net.set_radius(radius)
    return net


def update_particles_polyscope(
    ps: ParticleSystem,
    cloud,
    radius_scale: float = 1.0,
):
    """Update an existing polyscope point cloud with current particle positions.

    Parameters
    ----------
    ps : ParticleSystem
    cloud : polyscope PointCloud
        The cloud returned by :func:`plot_particles_polyscope`.
    radius_scale : float
        Scale factor for particle radii.
    """
    cloud.update_point_positions(ps.positions())
    cloud.add_scalar_quantity(
        "radius", ps.radii() * radius_scale, enabled=True,
    )
