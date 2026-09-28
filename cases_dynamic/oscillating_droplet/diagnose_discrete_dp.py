#!/usr/bin/env python3
"""Compute the discrete-consistent ΔP and compare with analytical.

The analytical Young-Laplace gives ΔP = γκ. But the discrete operators
(face-average pressure flux and integrated curvature) may need a
different ΔP to produce zero net force at every interface vertex.

Computes:
  - Per-vertex effective ΔP: F_st_i · S_1_i / |S_1_i|²
  - Least-squares optimal ΔP: Σ(F_st · S_1) / Σ|S_1|²
  - Force residuals with analytical vs optimal ΔP
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet,
)
from ddgclib.eos import TaitMurnaghan
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.initial_conditions import ZeroVelocity
from ddgclib.geometry.domains import droplet_in_box_2d
from ddgclib.operators.stress import dual_area_vector
from ddgclib.operators.multiphase_stress import _interface_surface_tension
from ddgclib.geometry._dual_split_2d import edge_phase_area_fractions


def main():
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
    mps.refresh(HC, dim, reset_mass=True, split_method='neighbour_count')

    kappa = 1.0 / R0
    dp_analytical = gamma * kappa
    p_outer = float(eos_outer.pressure(rho_o))

    iface = [v for v in HC.V if getattr(v, 'is_interface', False)]

    # Compute per-vertex S_1 and F_st
    numerator = 0.0  # Σ(F_st · S_1)
    denominator = 0.0  # Σ|S_1|²
    per_vertex = []
    for v in iface:
        F_st = _interface_surface_tension(v, dim, mps)
        S_1 = np.zeros(dim)
        for v_j in v.nn:
            A_ij = dual_area_vector(v, v_j, HC, dim)
            fracs = edge_phase_area_fractions(v, v_j, dim=dim, interface=HC)
            S_1 += fracs.get(1, 0.0) * A_ij
        S_1_sq = np.dot(S_1, S_1)
        FdotS = np.dot(F_st, S_1)
        numerator += FdotS
        denominator += S_1_sq
        dp_v = FdotS / S_1_sq if S_1_sq > 1e-30 else 0.0
        per_vertex.append({
            'theta': np.degrees(np.arctan2(v.x_a[1], v.x_a[0])),
            'F_st_mag': np.linalg.norm(F_st),
            'S_1_mag': np.linalg.norm(S_1),
            'dp_eff': dp_v,
            'F_st': F_st,
            'S_1': S_1,
        })

    dp_optimal = numerator / denominator

    print(f"Analytical ΔP = γ/R = {dp_analytical:.6f}")
    print(f"Optimal ΔP (least-squares) = {dp_optimal:.6f}")
    print(f"Ratio optimal/analytical = {dp_optimal / dp_analytical:.6f}")

    # Per-vertex details
    print(f"\nPer-vertex effective ΔP:")
    per_vertex.sort(key=lambda x: x['theta'])
    for pv in per_vertex:
        print(f"  θ={pv['theta']:7.1f}° | |F_st|={pv['F_st_mag']:.4e} "
              f"|S_1|={pv['S_1_mag']:.4e} | ΔP_eff={pv['dp_eff']:.4f}")

    dp_effs = [pv['dp_eff'] for pv in per_vertex]
    print(f"\nΔP_eff: mean={np.mean(dp_effs):.4f}, std={np.std(dp_effs):.4f}, "
          f"min={min(dp_effs):.4f}, max={max(dp_effs):.4f}")

    # Compute force residuals with analytical vs optimal ΔP
    def compute_residuals(dp_value, label):
        rho_eq = float(eos_drop.density(p_outer + dp_value))
        # Set mass
        for v in HC.V:
            vol_d = v.dual_vol_phase[1]
            if vol_d > 1e-30:
                v.m_phase[1] = rho_eq * vol_d
                v.m = float(np.sum(v.m_phase))
        mps.refresh(HC, dim, reset_mass=False, split_method='neighbour_count')

        # Check pressure
        p_vals = [v.p_phase[1] for v in iface]

        # Compute forces
        from ddgclib.operators.multiphase_stress import multiphase_stress_force
        forces = [np.linalg.norm(multiphase_stress_force(v, dim=dim, mps=mps, HC=HC))
                  for v in iface]
        print(f"\n  {label}: ΔP = {dp_value:.6f}")
        print(f"    p_phase[1] at interface: mean={np.mean(p_vals):.6f}")
        print(f"    Max |F|: {max(forces):.6e}")
        print(f"    Mean |F|: {np.mean(forces):.6e}")
        return forces

    f_analytical = compute_residuals(dp_analytical, "Analytical ΔP = γ/R")
    f_optimal = compute_residuals(dp_optimal, "Optimal ΔP (least-squares)")

    print(f"\nImprovement: {max(f_analytical)/max(f_optimal):.1f}x "
          f"(max |F| ratio)")


if __name__ == '__main__':
    main()
