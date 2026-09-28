"""DEM time integrators synchronized with the fluid integration timestep.

These integrators advance DEM particle positions and velocities using the
same ``dt`` as the fluid integrators. They follow the same structural
conventions as :mod:`ddgclib.dynamic_integrators._integrators_dynamic`.

The primary integrator is **Velocity Verlet** (symplectic, second-order).
For compatibility with the fluid ``symplectic_euler``, a symplectic Euler
integrator is also provided.

Usage
-----
::

    from ddgclib.dem import dem_step, ParticleSystem, HertzContact, ContactDetector

    ps = ParticleSystem(dim=3)
    # ... add particles ...
    contact_model = HertzContact()
    detector = ContactDetector(ps)

    # Single step (called once per fluid timestep):
    dem_step(ps, detector, contact_model, dt=1e-4, dim=3)

    # With sub-stepping (when contact stiffness demands smaller dt):
    dem_step(ps, detector, contact_model, dt=1e-4, dim=3, n_sub=10)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ddgclib.dem._particle import ParticleSystem
from ddgclib.dem._contact import ContactDetector
from ddgclib.dem._force_models import ContactForceModel


def _accumulate_contact_forces(
    ps: ParticleSystem,
    detector: ContactDetector,
    model: ContactForceModel,
) -> int:
    """Detect contacts and accumulate forces/torques on all particles.

    Returns the number of contacts detected.
    """
    contacts = detector.detect()
    for c in contacts:
        result = model.compute(c)
        c.p_i.force += result.total_force_on_i
        c.p_j.force += result.total_force_on_j
        c.p_i.torque += result.torque_i
        c.p_j.torque += result.torque_j
    return len(contacts)


def _accumulate_bond_forces(ps: ParticleSystem, bond_manager=None) -> None:
    """Accumulate forces from sintered bonds if present."""
    if bond_manager is not None:
        bond_manager.apply_forces(ps)


def _accumulate_bridge_forces(ps: ParticleSystem, bridge_manager=None) -> None:
    """Accumulate forces from liquid bridges if present."""
    if bridge_manager is not None:
        bridge_manager.apply_forces(ps)


def _accumulate_all_forces(
    ps: ParticleSystem,
    detector: ContactDetector,
    model: ContactForceModel,
    bond_manager=None,
    bridge_manager=None,
    external_forces_fn: Optional[Callable] = None,
) -> None:
    """Reset and accumulate all force contributions."""
    ps.reset_all_forces()
    ps.apply_gravity()
    _accumulate_contact_forces(ps, detector, model)
    _accumulate_bond_forces(ps, bond_manager)
    _accumulate_bridge_forces(ps, bridge_manager)
    if external_forces_fn is not None:
        external_forces_fn(ps)


def dem_velocity_verlet(
    ps: ParticleSystem,
    detector: ContactDetector,
    model: ContactForceModel,
    dt: float,
    dim: int = 3,
    bond_manager=None,
    bridge_manager=None,
    external_forces_fn: Optional[Callable] = None,
) -> None:
    """Single Velocity Verlet step for all particles.

    Update rule::

        x^{n+1}     = x^n + dt * u^n + 0.5 * dt^2 * a^n
        (recompute forces at x^{n+1})
        u^{n+1}     = u^n + 0.5 * dt * (a^n + a^{n+1})
        omega^{n+1} = omega^n + 0.5 * dt * (alpha^n + alpha^{n+1})

    Parameters
    ----------
    ps : ParticleSystem
    detector : ContactDetector
    model : ContactForceModel
    dt : float
        Time step (same as fluid timestep, or a sub-step).
    dim : int
    bond_manager : BondManager or None
    bridge_manager : LiquidBridgeManager or None
    external_forces_fn : callable or None
        ``fn(ps)`` that adds external forces (e.g. fluid coupling).
    """
    # Store old accelerations (computed from forces at current state)
    old_accels: dict[int, np.ndarray] = {}
    old_alphas: dict[int, np.ndarray] = {}
    for p in ps.particles:
        if not p.boundary:
            old_accels[p.id] = p.force / p.m
            old_alphas[p.id] = p.torque / p.I

    # Half-step position update: x += dt*u + 0.5*dt²*a
    for p in ps.particles:
        if not p.boundary:
            a_old = old_accels[p.id]
            p.x_a[:dim] += dt * p.u[:dim] + 0.5 * dt**2 * a_old[:dim]

    # Recompute all forces at new positions
    _accumulate_all_forces(
        ps, detector, model, bond_manager, bridge_manager, external_forces_fn
    )

    # Half-step velocity update: u += 0.5*dt*(a_old + a_new)
    for p in ps.particles:
        if not p.boundary:
            a_new = p.force / p.m
            alpha_new = p.torque / p.I
            p.u[:dim] += 0.5 * dt * (old_accels[p.id][:dim] + a_new[:dim])
            p.omega += 0.5 * dt * (old_alphas[p.id] + alpha_new)


def dem_symplectic_euler(
    ps: ParticleSystem,
    detector: ContactDetector,
    model: ContactForceModel,
    dt: float,
    dim: int = 3,
    bond_manager=None,
    bridge_manager=None,
    external_forces_fn: Optional[Callable] = None,
) -> None:
    """Single symplectic Euler step (matches fluid ``symplectic_euler``).

    Update rule::

        a^n       = F^n / m
        u^{n+1}   = u^n + dt * a^n       (velocity first)
        x^{n+1}   = x^n + dt * u^{n+1}   (new velocity for position)
    """
    # Compute forces at current state
    _accumulate_all_forces(
        ps, detector, model, bond_manager, bridge_manager, external_forces_fn
    )

    # Update velocity then position (symplectic ordering)
    for p in ps.particles:
        if not p.boundary:
            a = p.force / p.m
            alpha = p.torque / p.I
            p.u[:dim] += dt * a[:dim]
            p.omega += dt * alpha
            p.x_a[:dim] += dt * p.u[:dim]


def dem_step(
    ps: ParticleSystem,
    detector: ContactDetector,
    model: ContactForceModel,
    dt: float,
    dim: int = 3,
    n_sub: int = 1,
    method: str = "velocity_verlet",
    bond_manager=None,
    bridge_manager=None,
    external_forces_fn: Optional[Callable] = None,
    callback: Optional[Callable] = None,
) -> None:
    """Advance DEM system by one fluid timestep ``dt``, with optional sub-stepping.

    This is the main entry point called from the coupled stepping loop.

    Parameters
    ----------
    ps : ParticleSystem
    detector : ContactDetector
    model : ContactForceModel
    dt : float
        Total time to advance (same as fluid ``dt``).
    dim : int
    n_sub : int
        Number of DEM sub-steps within ``dt``.
        Total DEM ``dt_sub = dt / n_sub``.
        Use ``n_sub > 1`` when contact stiffness requires smaller DEM steps.
    method : str
        ``"velocity_verlet"`` or ``"symplectic_euler"``.
    bond_manager : BondManager or None
    bridge_manager : LiquidBridgeManager or None
    external_forces_fn : callable or None
    callback : callable or None
        ``fn(sub_step, t_sub, ps)`` called after each sub-step.
    """
    integrators = {
        "velocity_verlet": dem_velocity_verlet,
        "symplectic_euler": dem_symplectic_euler,
    }
    if method not in integrators:
        raise ValueError(
            f"Unknown DEM method {method!r}. "
            f"Available: {list(integrators.keys())}"
        )
    integrator_fn = integrators[method]

    dt_sub = dt / n_sub
    for sub in range(n_sub):
        integrator_fn(
            ps,
            detector,
            model,
            dt_sub,
            dim=dim,
            bond_manager=bond_manager,
            bridge_manager=bridge_manager,
            external_forces_fn=external_forces_fn,
        )
        if callback is not None:
            callback(sub, (sub + 1) * dt_sub, ps)
