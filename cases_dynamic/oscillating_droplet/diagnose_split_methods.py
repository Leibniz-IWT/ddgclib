#!/usr/bin/env python3
"""Compare neighbour_count vs exact split on force balance."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet,
)
from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties, mass_conserving_merge
from ddgclib.initial_conditions import ZeroVelocity
from ddgclib.geometry.domains import droplet_in_box_2d
from ddgclib.operators.multiphase_stress import multiphase_stress_force
from ddgclib.dynamic_integrators import symplectic_euler
from hyperct.ddg import compute_vd
from ddgclib.operators.stress import cache_dual_volumes
from functools import partial


def build_case(split_method):
    """Build static droplet with the specified split method."""
    dim = 2
    eos_outer = TaitMurnaghan(rho0=rho_o, P0=0.0, K=K_o, n=7.15, rho_clip=(0.8, 1.2))
    eos_drop = TaitMurnaghan(rho0=rho_d, P0=0.0, K=K_d, n=7.15, rho_clip=(0.8, 1.2))
    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_outer, mu=mu_o, rho0=rho_o, name="outer"),
            PhaseProperties(eos=eos_drop, mu=mu_d, rho0=rho_d, name="droplet"),
        ],
        gamma={(0, 1): gamma},
    )
    result = droplet_in_box_2d(R=R0, L=L_domain,
                                refinement_outer=n_refine_outer,
                                refinement_droplet=n_refine_droplet)
    HC = result.HC
    bV = result.bV
    builder_mps = result.metadata['mps']
    mps.simplex_phase = builder_mps.simplex_phase
    mps._simplex_criterion_fn = builder_mps._simplex_criterion_fn

    ZeroVelocity(dim=dim).apply(HC, bV)
    mps.refresh(HC, dim, reset_mass=True, split_method=split_method)

    # Young-Laplace
    kappa = 1.0 / R0
    delta_p = gamma * kappa
    p_outer = float(eos_outer.pressure(rho_o))
    rho_d_eq = float(eos_drop.density(p_outer + delta_p))
    for v in HC.V:
        vol_d = v.dual_vol_phase[1]
        if vol_d > 1e-30:
            v.m_phase[1] = rho_d_eq * vol_d
            v.m = float(np.sum(v.m_phase))

    mps.refresh(HC, dim, reset_mass=False, split_method=split_method)
    return HC, bV, mps


def dual_only_retopo(HC, bV, dim, _mps=None, **_kw):
    dV = HC.boundary()
    for v in HC.V:
        v.boundary = v in dV
    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)
    if _mps is not None:
        _mps.split_dual_volumes(HC, dim)
    bV.clear()
    bV.update(dV)


def run_test(split_method, n_steps=100):
    dim = 2
    HC, bV, mps = build_case(split_method)

    iface = [v for v in HC.V if getattr(v, 'is_interface', False)]
    forces_t0 = [np.linalg.norm(multiphase_stress_force(v, dim=dim, mps=mps, HC=HC))
                 for v in iface]

    # Check volume fractions at interface
    vol_fracs = [v.dual_vol_phase[1] / (v.dual_vol_phase[0] + v.dual_vol_phase[1])
                 for v in iface if (v.dual_vol_phase[0] + v.dual_vol_phase[1]) > 0]

    # Pressure check
    p1_vals = [v.p_phase[1] for v in iface]
    p0_vals = [v.p_phase[0] for v in iface]

    print(f"\n  split_method='{split_method}':")
    print(f"    Interface volume fraction (phase 1): "
          f"mean={np.mean(vol_fracs):.4f}, std={np.std(vol_fracs):.4f}")
    print(f"    p_phase[1] at interface: mean={np.mean(p1_vals):.6f}, "
          f"std={np.std(p1_vals):.2e}")
    print(f"    p_phase[0] at interface: mean={np.mean(p0_vals):.6f}, "
          f"std={np.std(p0_vals):.2e}")
    print(f"    Laplace jump: mean={np.mean(np.array(p1_vals) - np.array(p0_vals)):.6f}")
    print(f"    Max |F| at t=0: {max(forces_t0):.6e}")
    print(f"    Mean |F| at t=0: {np.mean(forces_t0):.6e}")

    # Run 100 steps with dual-only retopo
    meos = MultiphaseEOS(
        [mps.phases[0].eos, mps.phases[1].eos])
    from ddgclib.operators.multiphase_stress import multiphase_dudt_i
    dudt_fn = partial(multiphase_dudt_i, dim=dim, mps=mps, HC=HC,
                      pressure_model=meos)
    from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV)

    c_s = float(np.sqrt(K_d / rho_d))
    dx_min = min(
        float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min ** 3 / gamma))

    KE_list = [0.0]

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if step % 10 == 0:
            KE = sum(0.5 * v.m * np.dot(v.u[:dim], v.u[:dim])
                     for v in HC_cb.V if not v.boundary)
            KE_list.append(KE)

    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=callback,
        retopologize_fn=partial(dual_only_retopo, _mps=mps),
    )

    print(f"    KE after {n_steps} steps: {KE_list[-1]:.6e}")
    print(f"    Max KE: {max(KE_list):.6e}")


def main():
    print("=" * 70)
    print("Split method comparison for equilibrium force balance")
    print("=" * 70)
    run_test('neighbour_count')
    run_test('exact')


if __name__ == '__main__':
    main()
