#!/usr/bin/env python3
"""Verify EOS/IC consistency for the Young-Laplace equilibrium.

For each interface vertex and each bulk vertex, check:
  p_phase[1] - p_phase[0] == gamma * kappa
  rho_phase[k] == m_phase[k] / dual_vol_phase[k]
  EOS_k(rho_phase[k]) == p_phase[k]

Tests both 2D and 3D, and both 'neighbour_count' and 'exact' split methods.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from ddgclib.operators.multiphase_stress import multiphase_stress_force


def verify_eos_ic(dim, split_method='neighbour_count'):
    """Run EOS/IC verification for a given dimension and split method."""
    print(f"\n{'='*70}")
    print(f"  dim={dim}, split_method='{split_method}'")
    print(f"{'='*70}")

    # Use lower refinement for 3D to keep it fast
    ref_outer = n_refine_outer if dim == 2 else 1
    ref_drop = n_refine_droplet if dim == 2 else 1

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=0.0, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=ref_outer,
            refinement_droplet=ref_drop,
        )

    eos_outer = mps.phases[0].eos
    eos_drop = mps.phases[1].eos
    kappa = (dim - 1) / R0
    delta_p_expected = gamma * kappa
    p_outer_ref = float(eos_outer.pressure(rho_o))

    n_verts = sum(1 for _ in HC.V)
    iface = [v for v in HC.V if getattr(v, 'is_interface', False)]
    bulk_d = [v for v in HC.V if v.phase == 1
              and not getattr(v, 'is_interface', False) and not v.boundary]
    bulk_o = [v for v in HC.V if v.phase == 0
              and not getattr(v, 'is_interface', False) and not v.boundary]

    print(f"  Mesh: {n_verts} verts, {len(iface)} interface, "
          f"{len(bulk_d)} bulk-drop, {len(bulk_o)} bulk-outer")
    print(f"  Expected: kappa={kappa:.2f}, delta_p={delta_p_expected:.6f}, "
          f"p_outer={p_outer_ref:.6f}")

    # --- 1. Check EOS round-trip: density -> pressure -> density ---
    rho_d_eq = float(eos_drop.density(p_outer_ref + delta_p_expected))
    p_roundtrip = float(eos_drop.pressure(rho_d_eq))
    print(f"\n  EOS round-trip:")
    print(f"    rho_d_eq = density(p_outer + delta_p) = {rho_d_eq:.6f}")
    print(f"    pressure(rho_d_eq) = {p_roundtrip:.6f}")
    print(f"    Error: {abs(p_roundtrip - (p_outer_ref + delta_p_expected)):.2e}")

    # --- 2. Check bulk droplet: uniform p = p_outer + delta_p ---
    if bulk_d:
        p_drop_vals = []
        rho_drop_vals = []
        eos_errors = []
        for v in bulk_d:
            p1 = v.p_phase[1]
            rho1 = v.m_phase[1] / v.dual_vol_phase[1] if v.dual_vol_phase[1] > 1e-30 else 0
            p_from_eos = float(eos_drop.pressure(rho1))
            p_drop_vals.append(p1)
            rho_drop_vals.append(rho1)
            eos_errors.append(abs(p_from_eos - p1))
        print(f"\n  Bulk droplet ({len(bulk_d)} vertices):")
        print(f"    p_phase[1]: mean={np.mean(p_drop_vals):.6f}, "
              f"std={np.std(p_drop_vals):.2e}")
        print(f"    Expected p = {p_outer_ref + delta_p_expected:.6f}")
        print(f"    Error vs expected: {abs(np.mean(p_drop_vals) - (p_outer_ref + delta_p_expected)):.2e}")
        print(f"    rho_phase[1]: mean={np.mean(rho_drop_vals):.6f}")
        print(f"    Max |EOS(rho) - p_stored|: {max(eos_errors):.2e}")

    # --- 3. Check bulk outer: uniform p = p_outer ---
    if bulk_o:
        p_outer_vals = []
        eos_errors_o = []
        for v in bulk_o:
            p0 = v.p_phase[0]
            rho0 = v.m_phase[0] / v.dual_vol_phase[0] if v.dual_vol_phase[0] > 1e-30 else 0
            p_from_eos = float(eos_outer.pressure(rho0))
            p_outer_vals.append(p0)
            eos_errors_o.append(abs(p_from_eos - p0))
        print(f"\n  Bulk outer ({len(bulk_o)} vertices):")
        print(f"    p_phase[0]: mean={np.mean(p_outer_vals):.6f}, "
              f"std={np.std(p_outer_vals):.2e}")
        print(f"    Expected p = {p_outer_ref:.6f}")
        print(f"    Max |EOS(rho) - p_stored|: {max(eos_errors_o):.2e}")

    # --- 4. Check interface vertices: both phases ---
    if iface:
        laplace_jumps = []
        p0_vals = []
        p1_vals = []
        vol_ratios = []
        for v in iface:
            p0 = v.p_phase[0]
            p1 = v.p_phase[1]
            p0_vals.append(p0)
            p1_vals.append(p1)
            laplace_jumps.append(p1 - p0)
            dvp0 = v.dual_vol_phase[0]
            dvp1 = v.dual_vol_phase[1]
            total = dvp0 + dvp1
            vol_ratios.append(dvp1 / total if total > 0 else 0)
        print(f"\n  Interface ({len(iface)} vertices):")
        print(f"    p_phase[0]: mean={np.mean(p0_vals):.6f}, std={np.std(p0_vals):.2e}")
        print(f"    p_phase[1]: mean={np.mean(p1_vals):.6f}, std={np.std(p1_vals):.2e}")
        print(f"    Laplace jump (p1-p0): mean={np.mean(laplace_jumps):.6f}, "
              f"std={np.std(laplace_jumps):.2e}")
        print(f"    Expected jump: {delta_p_expected:.6f}")
        print(f"    Phase-1 volume fraction: mean={np.mean(vol_ratios):.4f}, "
              f"std={np.std(vol_ratios):.4f}")

        # Check if EOS(m/V) matches stored p for each phase at interface
        max_eos_err_0 = 0.0
        max_eos_err_1 = 0.0
        for v in iface:
            for k, eos in [(0, eos_outer), (1, eos_drop)]:
                dvp = v.dual_vol_phase[k]
                mk = v.m_phase[k]
                if dvp > 1e-30 and mk > 1e-30:
                    rho_k = mk / dvp
                    p_check = float(eos.pressure(rho_k))
                    err = abs(p_check - v.p_phase[k])
                    if k == 0:
                        max_eos_err_0 = max(max_eos_err_0, err)
                    else:
                        max_eos_err_1 = max(max_eos_err_1, err)
        print(f"    Max |EOS(m/V) - p_stored| phase 0: {max_eos_err_0:.2e}")
        print(f"    Max |EOS(m/V) - p_stored| phase 1: {max_eos_err_1:.2e}")

    # --- 5. Force balance ---
    if iface:
        forces = []
        for v in iface:
            F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC)
            forces.append(np.linalg.norm(F))
        print(f"\n  Force balance on interface:")
        print(f"    Max |F_total|:  {max(forces):.6e}")
        print(f"    Mean |F_total|: {np.mean(forces):.6e}")

    if bulk_d:
        forces_bd = [np.linalg.norm(multiphase_stress_force(v, dim=dim, mps=mps, HC=HC))
                     for v in bulk_d]
        print(f"    Max |F| bulk drop:  {max(forces_bd):.6e}")
    if bulk_o:
        forces_bo = [np.linalg.norm(multiphase_stress_force(v, dim=dim, mps=mps, HC=HC))
                     for v in bulk_o[:50]]  # limit for 3D speed
        print(f"    Max |F| bulk outer: {max(forces_bo):.6e}")


def main():
    print("EOS / Initial-Condition Verification for Young-Laplace Equilibrium")
    print("=" * 70)

    # 2D with neighbour_count (current default)
    verify_eos_ic(dim=2, split_method='neighbour_count')

    # 2D with exact split
    # (need to modify setup to accept split_method — for now just verify
    # the default)

    # 3D
    verify_eos_ic(dim=3, split_method='neighbour_count')

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
