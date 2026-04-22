"""Setup function for the oscillating droplet test case.

Constructs the multiphase mesh, applies initial conditions with
an ellipsoidal perturbation, and returns all objects needed to
run the simulation.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties, mass_conserving_merge
from ddgclib.initial_conditions import ZeroVelocity
from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
from ddgclib.geometry.domains import droplet_in_box_2d, droplet_in_box_3d
from ddgclib.operators.multiphase_stress import multiphase_dudt_i
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _retopologize_multiphase,
)


def setup_oscillating_droplet(
    dim: int = 2,
    R0: float = 0.01,
    epsilon: float = 0.05,
    l: int = 2,
    rho_d: float = 800.0,
    rho_o: float = 1000.0,
    mu_d: float = 0.5,
    mu_o: float = 0.1,
    gamma: float = 0.05,
    K_d: float | None = None,
    K_o: float | None = None,
    L_domain: float = 0.05,
    refinement_outer: int = 2,
    refinement_droplet: int = 3,
    P0: float = 0.0,
    distr_law: str = "sinusoidal",
):
    """Set up oscillating droplet problem (2D or 3D).

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    R0 : float
        Equilibrium droplet radius [m].
    epsilon : float
        Relative perturbation amplitude.
    l : int
        Oscillation mode number.
    rho_d, rho_o : float
        Droplet and outer fluid densities [kg/m³].
    mu_d, mu_o : float
        Droplet and outer fluid viscosities [Pa·s].
    gamma : float
        Surface tension [N/m].
    K_d, K_o : float or None
        Bulk moduli [Pa].  If None, computed from density and sound speed.
    L_domain : float
        Half-side of outer box domain [m].
    refinement_outer, refinement_droplet : int
        Mesh refinement levels.
    P0 : float
        Reference pressure [Pa].
    distr_law : str
        Radial distribution law for droplet mesh vertices.

    Returns
    -------
    HC : Complex
        Simplicial complex with duals computed.
    bV : set
        Boundary vertices (outer walls).
    mps : MultiphaseSystem
        Multiphase system configuration.
    bc_set : BoundaryConditionSet
        Boundary condition container.
    dudt_fn : callable
        Acceleration function for integrators.
    params : dict
        All parameters for reference.
    """
    # -- Compute bulk moduli if not given --
    if K_d is None:
        c_s = max(10.0 * epsilon * R0 * 1000.0, 1.0)
        K_d = rho_d * c_s**2
    if K_o is None:
        c_s = max(10.0 * epsilon * R0 * 1000.0, 1.0)
        K_o = rho_o * c_s**2

    # -- Build multiphase system --
    eos_outer = TaitMurnaghan(rho0=rho_o, P0=P0, K=K_o, n=7.15,
                               rho_clip=(0.8, 1.2))
    eos_drop = TaitMurnaghan(rho0=rho_d, P0=P0, K=K_d, n=7.15,
                              rho_clip=(0.8, 1.2))

    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_outer, mu=mu_o, rho0=rho_o, name="outer"),
            PhaseProperties(eos=eos_drop, mu=mu_d, rho0=rho_d, name="droplet"),
        ],
        gamma={(0, 1): gamma},
    )

    # -- Build mesh --
    if dim == 2:
        result = droplet_in_box_2d(
            R=R0, L=L_domain,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
            distr_law=distr_law,
        )
    elif dim == 3:
        result = droplet_in_box_3d(
            R=R0, L=L_domain,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
            distr_law=distr_law,
        )
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    HC = result.HC
    bV = result.bV

    # -- Apply initial conditions --
    # Phase labelling is done by the domain builder via
    # simplex-centroid test (primal subcomplex model).  Transfer the
    # criterion function so that mps.refresh can re-label after
    # retriangulations.
    builder_mps = result.metadata['mps']
    mps.simplex_phase = builder_mps.simplex_phase
    mps._simplex_criterion_fn = builder_mps._simplex_criterion_fn

    # 1. Zero velocity
    ZeroVelocity(dim=dim).apply(HC, bV)

    # 2. Initialise all per-phase fields, split dual volumes, set
    #    per-phase mass and pressure using the new n-phase data model.
    split_method = 'neighbour_count'
    mps.refresh(HC, dim, reset_mass=True, split_method=split_method)

    # 3. Apply ellipsoidal perturbation to interface vertices.
    #    Interface vertices (phase 1 on the circle) are moved to the
    #    perturbed radius R(theta) = R0*(1 + epsilon*cos(l*theta)).
    #    They keep phase=1 — they represent the deformed droplet boundary.
    _apply_perturbation(HC, R0, epsilon, l, dim)

    # 3b. Merge near-duplicate vertices created by perturbation
    mass_conserving_merge(HC, cdist=1e-10)

    # 4. Apply Young-Laplace equilibrium: set inner-phase mass so that
    #    the EOS pressure at the equilibrium density equals the Laplace
    #    jump.  P_d(rho_d_eq) = P_o(rho_o) + gamma * kappa.
    curvature = (dim - 1) / R0  # kappa = 1/R (2D), 2/R (3D)
    gamma_val = mps.get_gamma_pair(0, 1)
    delta_p = gamma_val * curvature
    p_outer = float(eos_outer.pressure(rho_o))
    rho_d_eq = float(eos_drop.density(p_outer + delta_p))
    for v in HC.V:
        # Adjust droplet-phase (phase 1) mass for vertices that have
        # a non-zero phase-1 dual volume.  This covers bulk phase-1
        # vertices AND interface vertices (v.phase == INTERFACE_PHASE).
        vol_d = v.dual_vol_phase[1]
        if vol_d > 1e-30:
            v.m_phase[1] = rho_d_eq * vol_d
            v.m = float(np.sum(v.m_phase))

    # 5. Final refresh: recompute pressures from the adjusted densities.
    #    reset_mass=False so the adjusted masses are preserved.
    mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

    # -- Boundary conditions --
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), result.boundary_groups['walls'])

    # -- Build acceleration function --
    meos = MultiphaseEOS([eos_outer, eos_drop])
    dudt_fn = partial(
        multiphase_dudt_i,
        dim=dim, mps=mps, HC=HC, pressure_model=meos,
    )

    # -- Retopologize function --
    # Stays on global Delaunay for now.  hyperct.remesh.adaptive_remesh
    # is interface-preserving but has upstream issues that blow up this
    # case:
    #   1. edge_split_2d inserts a midpoint vertex with mass ~= mean of
    #      its endpoints, silently inflating total mass on every split.
    #      (hyperct/remesh/_operations_2d.py:198)
    #   2. The driver uses a single global h_local, so a mixed
    #      fine-droplet / coarse-outer mesh triggers unbounded splits
    #      in the outer region when L_max is sized to the droplet.
    adaptive_kwargs = None
    retopo_fn = partial(_retopologize_multiphase, mps=mps,
                        split_method=split_method)

    # -- Collect params --
    params = {
        'dim': dim, 'R0': R0, 'epsilon': epsilon, 'l': l,
        'rho_d': rho_d, 'rho_o': rho_o, 'mu_d': mu_d, 'mu_o': mu_o,
        'gamma': gamma, 'K_d': K_d, 'K_o': K_o,
        'L_domain': L_domain, 'P0': P0,
        'refinement_outer': refinement_outer,
        'refinement_droplet': refinement_droplet,
        'remesh_mode': 'delaunay',
        'remesh_kwargs': adaptive_kwargs,
    }

    return HC, bV, mps, bc_set, dudt_fn, retopo_fn, params


def _apply_perturbation(HC, R0: float, epsilon: float, l: int, dim: int):
    """Move interface vertices to perturbed ellipsoidal shape.

    For mode l=2, the perturbation is:
        R(theta) = R0 * (1 + epsilon * cos(l * theta))

    where theta is the polar angle from the x-axis (2D) or z-axis (3D).
    """
    for v in HC.V:
        if not getattr(v, 'is_interface', False):
            continue

        x = v.x_a[:dim].copy()
        r = np.linalg.norm(x)
        if r < 1e-30:
            continue

        # Compute angle
        if dim == 2:
            theta = np.arctan2(x[1], x[0])
        else:  # dim == 3
            theta = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])

        # Perturbed radius
        R_perturbed = R0 * (1.0 + epsilon * np.cos(l * theta))

        # Scale position to perturbed radius
        scale = R_perturbed / r
        new_pos = x * scale

        # Pad to full coordinate array length
        full_pos = v.x_a.copy()
        full_pos[:dim] = new_pos
        HC.V.move(v, tuple(full_pos))
