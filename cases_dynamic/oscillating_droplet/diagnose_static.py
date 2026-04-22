#!/usr/bin/env python3
"""Diagnostic: probe force balance on the static circular droplet at t=0.

Investigates why KE grows from ~0 when epsilon=0 (no perturbation).
Checks:
1. Force balance at t=0 (pressure, viscous, surface tension components)
2. Initial condition consistency (Young-Laplace pre-loading)
3. Mesh conformity (interface vertices on circle)
4. Surface tension force direction (should point inward for convex)
5. Phase fraction consistency on interface edges
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
from ddgclib.operators.stress import dual_area_vector, pressure_flux, viscous_flux
from ddgclib.operators.multiphase_stress import (
    multiphase_stress_force, _interface_surface_tension,
    _phases_present, _phase_pressure,
)
from ddgclib.operators.curvature_2d import (
    integrated_curvature_normal_2d, surface_tension_force_2d,
)
from ddgclib.geometry._dual_split_2d import edge_phase_area_fractions


def main():
    dim = 2
    epsilon = 0.0  # STATIC — no perturbation
    print("=" * 70)
    print("DIAGNOSTIC: Static droplet force balance at t=0")
    print("=" * 70)

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
        )

    n_verts = sum(1 for _ in HC.V)
    iface_verts = [v for v in HC.V if getattr(v, 'is_interface', False)]
    bulk_drop = [v for v in HC.V if v.phase == 1 and not getattr(v, 'is_interface', False) and not v.boundary]
    bulk_outer = [v for v in HC.V if v.phase == 0 and not getattr(v, 'is_interface', False) and not v.boundary]
    n_iface = len(iface_verts)
    print(f"Mesh: {n_verts} vertices, {n_iface} interface, "
          f"{len(bulk_drop)} bulk-drop, {len(bulk_outer)} bulk-outer")

    # ==================================================================
    # 1. MESH CONFORMITY: Are interface vertices on the circle?
    # ==================================================================
    print("\n" + "=" * 70)
    print("1. MESH CONFORMITY — interface vertices on circle r == R0?")
    print("=" * 70)
    radii = []
    for v in iface_verts:
        r = np.linalg.norm(v.x_a[:dim])
        radii.append(r)
    radii = np.array(radii)
    print(f"  R0 = {R0}")
    print(f"  Interface vertex radii: min={radii.min():.10f}, max={radii.max():.10f}")
    print(f"  Max deviation from R0: {np.max(np.abs(radii - R0)):.2e}")
    print(f"  All on circle? {np.allclose(radii, R0, atol=1e-8)}")

    # ==================================================================
    # 2. INITIAL CONDITION CONSISTENCY: Young-Laplace pre-loading
    # ==================================================================
    print("\n" + "=" * 70)
    print("2. INITIAL CONDITION — Young-Laplace pressure consistency")
    print("=" * 70)
    kappa = (dim - 1) / R0  # 1/R for 2D
    delta_p_expected = gamma * kappa
    print(f"  Expected Laplace jump: gamma * kappa = {gamma} * {kappa:.2f} = {delta_p_expected:.6f}")

    # Outer-phase pressure at bulk outer vertices
    p_outer_bulk = [v.p_phase[0] for v in bulk_outer if hasattr(v, 'p_phase')]
    if p_outer_bulk:
        p_outer_mean = np.mean(p_outer_bulk)
        p_outer_std = np.std(p_outer_bulk)
        print(f"  Bulk outer p_phase[0]: mean={p_outer_mean:.6f}, std={p_outer_std:.2e}")
    else:
        p_outer_mean = 0.0
        print("  WARNING: No bulk outer vertices with p_phase!")

    # Droplet-phase pressure at bulk droplet vertices
    p_drop_bulk = [v.p_phase[1] for v in bulk_drop if hasattr(v, 'p_phase')]
    if p_drop_bulk:
        p_drop_mean = np.mean(p_drop_bulk)
        p_drop_std = np.std(p_drop_bulk)
        print(f"  Bulk drop  p_phase[1]: mean={p_drop_mean:.6f}, std={p_drop_std:.2e}")
        print(f"  Actual Laplace jump (drop - outer): {p_drop_mean - p_outer_mean:.6f}")
        print(f"  Error vs expected: {abs(p_drop_mean - p_outer_mean - delta_p_expected):.2e}")
    else:
        p_drop_mean = 0.0
        print("  WARNING: No bulk droplet vertices with p_phase!")

    # Interface vertices: check both phases
    print("\n  Interface vertex pressures (sample of 5):")
    for v in iface_verts[:5]:
        r = np.linalg.norm(v.x_a[:dim])
        phases = getattr(v, 'interface_phases', frozenset())
        p0 = v.p_phase[0] if hasattr(v, 'p_phase') else 'N/A'
        p1 = v.p_phase[1] if hasattr(v, 'p_phase') else 'N/A'
        dvp0 = v.dual_vol_phase[0] if hasattr(v, 'dual_vol_phase') else 'N/A'
        dvp1 = v.dual_vol_phase[1] if hasattr(v, 'dual_vol_phase') else 'N/A'
        mp0 = v.m_phase[0] if hasattr(v, 'm_phase') else 'N/A'
        mp1 = v.m_phase[1] if hasattr(v, 'm_phase') else 'N/A'
        print(f"    r={r:.6f} | phases={phases} | p_phase=[{p0:.4f}, {p1:.4f}] "
              f"| dvp=[{dvp0:.2e}, {dvp1:.2e}] | mp=[{mp0:.2e}, {mp1:.2e}]")

    # ==================================================================
    # 3. FORCE BALANCE AT t=0 — decomposed into components
    # ==================================================================
    print("\n" + "=" * 70)
    print("3. FORCE BALANCE AT t=0 — per-component decomposition")
    print("=" * 70)

    def decompose_force(v, dim, mps, HC):
        """Decompose multiphase_stress_force into pressure, viscous, surface tension."""
        n_phases = mps.n_phases
        has_p_phase = hasattr(v, 'p_phase')
        is_interface_i = bool(getattr(v, 'is_interface', False))
        phases_present_list = _phases_present(v, n_phases)

        if has_p_phase:
            p_i_by_phase = {k: float(v.p_phase[k]) for k in phases_present_list}
        else:
            p_i_by_phase = {k: 0.0 for k in phases_present_list}

        u_i = v.u[:dim]
        x_i = v.x_a[:dim]

        F_pressure = np.zeros(dim)
        F_viscous = np.zeros(dim)

        for v_j in v.nn:
            A_ij = dual_area_vector(v, v_j, HC, dim)
            delta_u = v_j.u[:dim] - u_i
            d_ij = v_j.x_a[:dim] - x_i

            fractions = edge_phase_area_fractions(v, v_j, dim=dim, interface=HC)

            for k, frac in fractions.items():
                if k not in phases_present_list:
                    continue
                A_k = frac * A_ij
                p_i_k = p_i_by_phase[k]
                p_j_k = _phase_pressure(v_j, k, fallback=p_i_k)
                mu_k = float(mps.get_mu(k))

                F_pressure += pressure_flux(p_i_k, p_j_k, A_k)
                F_viscous += viscous_flux(mu_k, delta_u, d_ij, A_k)

        F_st = np.zeros(dim)
        if is_interface_i:
            F_st = _interface_surface_tension(v, dim, mps)

        return F_pressure, F_viscous, F_st

    # Sample interface vertices
    print("\n  --- Interface vertices ---")
    max_F_total_iface = 0.0
    force_magnitudes = []
    for i, v in enumerate(iface_verts):
        F_p, F_v, F_st = decompose_force(v, dim, mps, HC)
        F_total = F_p + F_v + F_st
        mag = np.linalg.norm(F_total)
        force_magnitudes.append(mag)
        max_F_total_iface = max(max_F_total_iface, mag)
        if i < 8:
            r = np.linalg.norm(v.x_a[:dim])
            theta = np.arctan2(v.x_a[1], v.x_a[0])
            print(f"    v(r={r:.6f}, θ={np.degrees(theta):6.1f}°):")
            print(f"      F_press  = [{F_p[0]:+.6e}, {F_p[1]:+.6e}]  |F|={np.linalg.norm(F_p):.6e}")
            print(f"      F_visc   = [{F_v[0]:+.6e}, {F_v[1]:+.6e}]  |F|={np.linalg.norm(F_v):.6e}")
            print(f"      F_st     = [{F_st[0]:+.6e}, {F_st[1]:+.6e}]  |F|={np.linalg.norm(F_st):.6e}")
            print(f"      F_TOTAL  = [{F_total[0]:+.6e}, {F_total[1]:+.6e}]  |F|={mag:.6e}")
            # Check direction of surface tension: should point inward (toward origin)
            if np.linalg.norm(F_st) > 1e-15:
                r_hat = v.x_a[:dim] / r  # outward radial
                dot = np.dot(F_st, r_hat)
                print(f"      F_st . r_hat = {dot:+.6e}  ({'INWARD' if dot < 0 else 'OUTWARD !!!'})")

    force_magnitudes = np.array(force_magnitudes)
    print(f"\n  Interface force summary:")
    print(f"    Max |F_total| on interface: {max_F_total_iface:.6e}")
    print(f"    Mean |F_total| on interface: {np.mean(force_magnitudes):.6e}")
    print(f"    Std  |F_total| on interface: {np.std(force_magnitudes):.6e}")

    # Sample bulk droplet vertices
    print("\n  --- Bulk droplet vertices (sample of 5) ---")
    max_F_bulk_drop = 0.0
    for v in bulk_drop[:5]:
        F_p, F_v, F_st = decompose_force(v, dim, mps, HC)
        F_total = F_p + F_v + F_st
        mag = np.linalg.norm(F_total)
        max_F_bulk_drop = max(max_F_bulk_drop, mag)
        print(f"    v({v.x_a[0]:+.5f}, {v.x_a[1]:+.5f}): "
              f"|F_p|={np.linalg.norm(F_p):.2e} |F_v|={np.linalg.norm(F_v):.2e} "
              f"|F_total|={mag:.2e}")

    # Sample bulk outer vertices
    print("\n  --- Bulk outer vertices (sample of 5) ---")
    max_F_bulk_outer = 0.0
    for v in bulk_outer[:5]:
        F_p, F_v, F_st = decompose_force(v, dim, mps, HC)
        F_total = F_p + F_v + F_st
        mag = np.linalg.norm(F_total)
        max_F_bulk_outer = max(max_F_bulk_outer, mag)
        print(f"    v({v.x_a[0]:+.5f}, {v.x_a[1]:+.5f}): "
              f"|F_p|={np.linalg.norm(F_p):.2e} |F_v|={np.linalg.norm(F_v):.2e} "
              f"|F_total|={mag:.2e}")

    print(f"\n  Max |F| across all vertex types:")
    print(f"    Interface:    {max_F_total_iface:.6e}")
    print(f"    Bulk droplet: {max_F_bulk_drop:.6e}")
    print(f"    Bulk outer:   {max_F_bulk_outer:.6e}")

    # ==================================================================
    # 4. SURFACE TENSION FORCE DIRECTION
    # ==================================================================
    print("\n" + "=" * 70)
    print("4. SURFACE TENSION — direction check (should be INWARD)")
    print("=" * 70)
    n_inward = 0
    n_outward = 0
    n_zero = 0
    for v in iface_verts:
        F_st = _interface_surface_tension(v, dim, mps)
        r = np.linalg.norm(v.x_a[:dim])
        if np.linalg.norm(F_st) < 1e-20:
            n_zero += 1
            continue
        r_hat = v.x_a[:dim] / r
        dot = np.dot(F_st, r_hat)
        if dot < 0:
            n_inward += 1
        else:
            n_outward += 1
    print(f"  Inward:  {n_inward} / {n_iface}")
    print(f"  Outward: {n_outward} / {n_iface}")
    print(f"  Zero:    {n_zero} / {n_iface}")
    if n_outward > 0:
        print("  >>> WARNING: Some surface tension forces point OUTWARD!")

    # ==================================================================
    # 5. EDGE PHASE FRACTIONS on interface edges
    # ==================================================================
    print("\n" + "=" * 70)
    print("5. EDGE PHASE FRACTIONS — interface edge consistency")
    print("=" * 70)
    sample_count = 0
    for v in iface_verts[:3]:
        for v_j in v.nn:
            fracs = edge_phase_area_fractions(v, v_j, dim=dim, interface=HC)
            is_j_iface = getattr(v_j, 'is_interface', False)
            if sample_count < 15:
                print(f"    ({v.x_a[0]:+.5f},{v.x_a[1]:+.5f}) -> "
                      f"({v_j.x_a[0]:+.5f},{v_j.x_a[1]:+.5f}) "
                      f"[j_iface={is_j_iface}, j_phase={v_j.phase}] "
                      f"fracs={fracs}")
                sample_count += 1

    # ==================================================================
    # 6. QUANTIFY: predicted KE after 1 step
    # ==================================================================
    print("\n" + "=" * 70)
    print("6. PREDICTED KE GROWTH (1 step)")
    print("=" * 70)
    c_s = float(np.sqrt(K_d / rho_d))
    dx_min = min(
        float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min ** 3 / gamma) if gamma > 0 else 1.0)
    print(f"  dt = {dt:.4e}")
    print(f"  dx_min = {dx_min:.4e}")

    # Compute acceleration on every non-boundary vertex
    KE_after_1 = 0.0
    max_accel = 0.0
    for v in HC.V:
        if v.boundary:
            continue
        F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC)
        a = F / v.m if v.m > 1e-30 else np.zeros(dim)
        u_new = v.u[:dim] + a * dt
        KE_after_1 += 0.5 * v.m * np.dot(u_new, u_new)
        accel_mag = np.linalg.norm(a)
        if accel_mag > max_accel:
            max_accel = accel_mag
            worst_v = v

    print(f"  Predicted KE after 1 step: {KE_after_1:.6e}")
    print(f"  Max |acceleration|: {max_accel:.6e} at ({worst_v.x_a[0]:.5f}, {worst_v.x_a[1]:.5f})")
    print(f"  Worst vertex: phase={worst_v.phase}, is_interface={getattr(worst_v, 'is_interface', False)}")

    # ==================================================================
    # 7. VALIDATE INTERFACE CLOSURE
    # ==================================================================
    print("\n" + "=" * 70)
    print("7. INTERFACE CLOSURE CHECK")
    print("=" * 70)
    interface_edges = getattr(HC, 'interface_edges', set())
    print(f"  Number of interface edges: {len(interface_edges)}")
    print(f"  Number of interface vertices: {n_iface}")

    # Each interface vertex should have exactly 2 interface-edge neighbors
    for v in iface_verts[:5]:
        iface_nbs = {nb for nb in v.nn if getattr(nb, 'is_interface', False)}
        # Count how many of these are actual interface edges
        edge_count = 0
        for nb in iface_nbs:
            ekey = frozenset({v.x, nb.x})
            if ekey in interface_edges:
                edge_count += 1
        print(f"    v(r={np.linalg.norm(v.x_a[:dim]):.6f}): "
              f"{len(iface_nbs)} iface nbs, {edge_count} iface edges")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
