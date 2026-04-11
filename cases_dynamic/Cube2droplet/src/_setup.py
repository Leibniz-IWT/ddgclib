"""Setup function for the cube-to-droplet relaxation case.

Constructs a multiphase mesh with a square (2D) or cube (3D) droplet
region embedded in an outer fluid, and returns all objects needed
to run the simulation.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.initial_conditions import ZeroVelocity, PhaseAssignment
from ddgclib._boundary_conditions import (
    BoundaryCondition, BoundaryConditionSet, NoSlipWallBC,
)
from ddgclib.operators.multiphase_stress import multiphase_dudt_i
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _retopologize_multiphase,
)


class AtmosphericPressureBC(BoundaryCondition):
    """Enforce atmospheric pressure on gas-phase vertices near the walls.

    After each time step, adjusts the mass of target vertices so that
    ``rho = m / dual_vol = rho0``, which makes the EOS return ``P = P0``
    (atmospheric).  This acts as a mass source/sink — the boundary
    "breathes" to maintain constant pressure.

    Only affects vertices whose phase matches ``gas_phase``.  Droplet
    vertices near the wall (if any) are left unchanged.
    """

    def __init__(self, rho0: float, gas_phase: int = 0):
        super().__init__()
        self.rho0 = rho0
        self.gas_phase = gas_phase

    def apply(self, mesh, dt, target_vertices=None):
        verts = target_vertices if target_vertices is not None else mesh.V
        count = 0
        for v in verts:
            if v.phase != self.gas_phase:
                continue
            vol = getattr(v, 'dual_vol', 0.0)
            if vol < 1e-30:
                continue
            v.m = self.rho0 * vol
            # Also reset per-phase mass if present
            mp = getattr(v, 'm_phase', None)
            if mp is not None:
                mp[self.gas_phase] = v.m
            count += 1
        return count


def setup_cube_to_droplet(
    dim: int = 2,
    R: float = 0.01,
    L_domain: float = 0.03,
    rho_d: float = 800.0,
    rho_o: float = 1000.0,
    mu_d: float = 2.0,
    mu_o: float = 1.0,
    gamma: float = 0.01,
    K_d: float = 100.0,
    K_o: float = 125.0,
    P0: float = 0.0,
    n_refine: int = 4,
):
    """Set up the cube-to-droplet relaxation problem.

    Returns
    -------
    HC, bV, mps, meos, bc_set, dudt_fn, retopo_fn, params
    """
    # -- Build mesh --
    bounds = [(-L_domain, L_domain)] * dim
    HC = Complex(dim, domain=bounds)
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = HC.boundary()
    for v in HC.V:
        v.boundary = v in bV
    compute_vd(HC, method="barycentric")

    from ddgclib.operators.stress import dual_volume
    for v in HC.V:
        try:
            v.dual_vol = dual_volume(v, HC, dim)
        except (ValueError, IndexError):
            v.dual_vol = 0.0

    # -- Phases --
    PhaseAssignment(
        lambda x: 1 if all(abs(x[i]) <= R for i in range(dim)) else 0
    ).apply(HC, bV)

    eos_outer = TaitMurnaghan(rho0=rho_o, P0=P0, K=K_o, n=1.0,
                               rho_clip=(0.1, 10.0))
    eos_drop = TaitMurnaghan(rho0=rho_d, P0=P0, K=K_d, n=1.0,
                              rho_clip=(0.1, 10.0))

    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_outer, mu=mu_o, rho0=rho_o, name="outer"),
            PhaseProperties(eos=eos_drop, mu=mu_d, rho0=rho_d, name="droplet"),
        ],
        gamma={(0, 1): gamma},
    )

    # -- Interface + per-phase fields --
    mps.identify_interface(HC)
    mps.init_phase_fields(HC)
    mps.split_dual_volumes(HC, dim)
    mps.compute_phase_masses(HC)

    # -- ICs: zero velocity, mass from phase density * dual volume --
    ZeroVelocity(dim=dim).apply(HC, bV)
    # Mass is set by compute_phase_masses (m_i = sum_k rho_k * V_k)
    # Pressure starts at P0 — EOS will update on first force evaluation
    for v in HC.V:
        v.p = P0

    # -- BCs --
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV)

    # Atmospheric pressure on gas-phase vertices adjacent to walls.
    # Wall vertices themselves have dual_vol=0 (boundary), so we target
    # their interior gas-phase neighbours — the first ring of interior
    # vertices that actually have nonzero dual volumes.
    atm_verts = set()
    for v_wall in bV:
        for nb in v_wall.nn:
            if nb not in bV and nb.phase == 0:
                atm_verts.add(nb)
    if atm_verts:
        bc_set.add(AtmosphericPressureBC(rho0=rho_o, gas_phase=0),
                    atm_verts)

    # -- Acceleration function --
    meos = MultiphaseEOS([eos_outer, eos_drop])
    dudt_fn = partial(
        multiphase_dudt_i,
        dim=dim, mps=mps, HC=HC, pressure_model=meos,
    )

    # -- Retopo (mass-conserving, no mass reset) --
    # Accept **kwargs so that the integrator can forward remesh_mode /
    # remesh_kwargs (see _do_retopologize's inspect.signature dispatch).
    def retopo_fn(HC, bV, dim, **kwargs):
        _retopologize_multiphase(HC, bV, dim, mps=mps, **kwargs)

    # -- Equivalent radius --
    if dim == 2:
        R_eq = R * 2.0 / np.sqrt(np.pi)
    else:
        R_eq = R * (6.0 / np.pi) ** (1 / 3)

    params = {
        'dim': dim, 'R': R, 'L_domain': L_domain,
        'rho_d': rho_d, 'rho_o': rho_o, 'mu_d': mu_d, 'mu_o': mu_o,
        'gamma': gamma, 'K_d': K_d, 'K_o': K_o, 'P0': P0,
        'n_refine': n_refine, 'R_eq': R_eq,
    }

    return HC, bV, mps, meos, bc_set, dudt_fn, retopo_fn, params
