#!/usr/bin/env python3
"""Debug the _balance_interface_forces function."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet,
)
from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.initial_conditions import ZeroVelocity
from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
from ddgclib.geometry.domains import droplet_in_box_2d
from ddgclib.operators.stress import dual_area_vector
from ddgclib.operators.multiphase_stress import (
    _interface_surface_tension, multiphase_stress_force,
)
from ddgclib.geometry._dual_split_2d import edge_phase_area_fractions
from functools import partial


def main():
    dim = 2
    epsilon = 0.0

    # Replicate setup steps 1-5 manually (without the balance step)
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
    split_method = 'neighbour_count'
    mps.refresh(HC, dim, reset_mass=True, split_method=split_method)

    # Young-Laplace (step 4)
    curvature = (dim - 1) / R0
    gamma_val = mps.get_gamma_pair(0, 1)
    delta_p = gamma_val * curvature
    p_outer = float(eos_outer.pressure(rho_o))
    rho_d_eq = float(eos_drop.density(p_outer + delta_p))
    for v in HC.V:
        vol_d = v.dual_vol_phase[1]
        if vol_d > 1e-30:
            v.m_phase[1] = rho_d_eq * vol_d
            v.m = float(np.sum(v.m_phase))

    # Step 5: refresh
    mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

    # Check forces BEFORE balance
    print("BEFORE _balance_interface_forces:")
    iface_verts = [v for v in HC.V if getattr(v, 'is_interface', False)]
    for v in iface_verts[:3]:
        F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC)
        print(f"  v({v.x_a[0]:+.5f},{v.x_a[1]:+.5f}): |F|={np.linalg.norm(F):.6e}, "
              f"p_phase={v.p_phase}, m_phase={v.m_phase}")

    # Now run the balance step
    print("\nRunning _balance_interface_forces...")
    for v in HC.V:
        if not getattr(v, 'is_interface', False):
            continue

        F_st = _interface_surface_tension(v, dim, mps)
        if np.linalg.norm(F_st) < 1e-30:
            print(f"  SKIP (F_st=0): {v.x_a[:dim]}")
            continue

        S_1 = np.zeros(dim)
        for v_j in v.nn:
            A_ij = dual_area_vector(v, v_j, HC, dim)
            fracs = edge_phase_area_fractions(v, v_j, dim=dim, interface=HC)
            frac_1 = fracs.get(1, 0.0)
            S_1 += frac_1 * A_ij

        S_1_mag = np.linalg.norm(S_1)
        if S_1_mag < 1e-30:
            print(f"  SKIP (S_1=0): {v.x_a[:dim]}")
            continue

        delta_p_eff = -np.dot(F_st, S_1) / (S_1_mag * S_1_mag)

        p_target = p_outer + delta_p_eff
        rho_target = float(eos_drop.density(p_target))
        vol_d = v.dual_vol_phase[1]

        if v in list(iface_verts)[:3]:
            r = np.linalg.norm(v.x_a[:dim])
            theta = np.degrees(np.arctan2(v.x_a[1], v.x_a[0]))
            print(f"  θ={theta:6.1f}: F_st={F_st}, S_1={S_1}, |S_1|={S_1_mag:.4e}")
            print(f"    delta_p_eff={delta_p_eff:.4f} (analytical={delta_p:.4f})")
            print(f"    p_target={p_target:.4f}, rho_target={rho_target:.4f}")
            print(f"    vol_d={vol_d:.4e}, m_old={v.m_phase[1]:.4e}, m_new={rho_target*vol_d:.4e}")

        if delta_p_eff < 0:
            continue
        if vol_d > 1e-30:
            v.m_phase[1] = rho_target * vol_d
            v.m = float(np.sum(v.m_phase))

    # Refresh to recompute pressures
    mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

    # Check forces AFTER balance
    print("\nAFTER _balance_interface_forces + refresh:")
    for v in iface_verts[:3]:
        F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC)
        print(f"  v({v.x_a[0]:+.5f},{v.x_a[1]:+.5f}): |F|={np.linalg.norm(F):.6e}, "
              f"p_phase={v.p_phase}, m_phase={v.m_phase}")


if __name__ == '__main__':
    main()
