"""Particle cloud import/export for DEM.

Supports saving and loading :class:`ParticleSystem` state in JSON format,
and importing particle clouds from external generation codes via NumPy arrays.

File format: ``ddgclib_dem_state_v1`` JSON with particle arrays.

Usage
-----
::

    from ddgclib.dem import save_particles, load_particles, import_particle_cloud

    # Save/load round-trip
    save_particles(ps, t=0.5, path="state.json")
    ps_loaded, t_loaded = load_particles("state.json")

    # Import from external code
    ps = import_particle_cloud(positions, radii=1e-3, rho_s=2500.0, dim=3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ddgclib.dem._particle import Particle, ParticleSystem


def save_particles(
    ps: ParticleSystem,
    t: float = 0.0,
    path: Union[str, Path] = "dem_state.json",
) -> Path:
    """Save a ParticleSystem to JSON.

    Parameters
    ----------
    ps : ParticleSystem
        The particle system to save.
    t : float
        Current simulation time.
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The path written to.
    """
    path = Path(path)
    particles_data = []
    for p in ps.particles:
        particles_data.append({
            "id": p.id,
            "x_a": p.x_a.tolist(),
            "u": p.u.tolist(),
            "omega": p.omega.tolist(),
            "radius": p.radius,
            "m": p.m,
            "I": p.I,
            "rho_s": p.rho_s,
            "boundary": p.boundary,
            "cluster_id": p.cluster_id,
            "wetted": p.wetted,
            "wetting_angle": p.wetting_angle,
            "liquid_volume": p.liquid_volume,
        })

    state = {
        "format": "ddgclib_dem_state_v1",
        "dim": ps.dim,
        "gravity": ps.gravity.tolist(),
        "time": t,
        "n_particles": len(ps),
        "particles": particles_data,
    }

    path.write_text(json.dumps(state, indent=2))
    return path


def load_particles(
    path: Union[str, Path],
) -> tuple[ParticleSystem, float]:
    """Load a ParticleSystem from JSON.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    ps : ParticleSystem
        Reconstructed particle system.
    t : float
        Simulation time stored in file.
    """
    path = Path(path)
    state = json.loads(path.read_text())

    fmt = state.get("format", "")
    if fmt != "ddgclib_dem_state_v1":
        raise ValueError(
            f"Unknown DEM state format: {fmt!r}. "
            f"Expected 'ddgclib_dem_state_v1'."
        )

    dim = state["dim"]
    gravity = np.array(state["gravity"])
    t = state["time"]

    ps = ParticleSystem(dim=dim, gravity=gravity)

    for pd in state["particles"]:
        p = Particle(
            x_a=np.array(pd["x_a"]),
            u=np.array(pd["u"]),
            omega=np.array(pd["omega"]),
            radius=pd["radius"],
            m=pd["m"],
            I=pd["I"],
            rho_s=pd["rho_s"],
            force=np.zeros(dim),
            torque=np.zeros(3 if dim == 3 else 1),
            id=pd["id"],
            cluster_id=pd.get("cluster_id"),
            boundary=pd.get("boundary", False),
            wetted=pd.get("wetted", False),
            wetting_angle=pd.get("wetting_angle", 0.0),
            liquid_volume=pd.get("liquid_volume", 0.0),
        )
        ps.add(p)

    return ps, t


def import_particle_cloud(
    positions: np.ndarray,
    radii: Union[float, np.ndarray] = 1e-3,
    rho_s: float = 2500.0,
    dim: int = 3,
    velocities: Optional[np.ndarray] = None,
    gravity: Optional[np.ndarray] = None,
    wetted: bool = False,
    wetting_angle: float = 0.0,
    liquid_volume: float = 0.0,
) -> ParticleSystem:
    """Import a particle cloud from external data.

    Primary entry point for external particle film generation codes.

    Parameters
    ----------
    positions : np.ndarray
        Shape ``(N, dim)`` array of particle centre positions.
    radii : float or np.ndarray
        Scalar (monodisperse) or ``(N,)`` array of particle radii.
    rho_s : float
        Solid density [kg/m^3].
    dim : int
        Spatial dimension.
    velocities : np.ndarray or None
        Shape ``(N, dim)`` initial velocities. Zero if ``None``.
    gravity : np.ndarray or None
        Gravity vector. Zero if ``None``.
    wetted : bool
        Whether particles are wetted (for liquid bridge formation).
    wetting_angle : float
        Contact angle [rad] for wetted particles.
    liquid_volume : float
        Liquid volume per particle [m^3] for bridge formation.

    Returns
    -------
    ParticleSystem
        Configured particle system ready for simulation.
    """
    positions = np.asarray(positions, dtype=float)
    N = positions.shape[0]

    if positions.ndim != 2 or positions.shape[1] != dim:
        raise ValueError(
            f"positions must have shape (N, {dim}), got {positions.shape}"
        )

    if np.isscalar(radii):
        radii_arr = np.full(N, radii, dtype=float)
    else:
        radii_arr = np.asarray(radii, dtype=float)
        if radii_arr.shape != (N,):
            raise ValueError(
                f"radii must be scalar or shape ({N},), got {radii_arr.shape}"
            )

    if velocities is not None:
        velocities = np.asarray(velocities, dtype=float)
        if velocities.shape != (N, dim):
            raise ValueError(
                f"velocities must have shape ({N}, {dim}), "
                f"got {velocities.shape}"
            )

    if gravity is None:
        gravity = np.zeros(dim)

    ps = ParticleSystem(dim=dim, gravity=gravity)

    for i in range(N):
        u = velocities[i] if velocities is not None else None
        ps.add(Particle.sphere(
            x=positions[i],
            radius=float(radii_arr[i]),
            rho_s=rho_s,
            dim=dim,
            u=u,
            wetted=wetted,
            wetting_angle=wetting_angle,
            liquid_volume=liquid_volume,
        ))

    return ps
