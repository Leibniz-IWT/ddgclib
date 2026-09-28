#!/usr/bin/env python3
"""Diagnose exactly what changes after a SINGLE Delaunay retopo step.

Sets up the static droplet, then applies one retopologization
(without any time step/vertex motion), and reports what changed:
- Dual volumes
- Pressures
- Interface edges
- Force balance
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


def main():
    dim = 2
    epsilon = 0.0
    print("=" * 70)
    print("DIAGNOSTIC: Effect of a single Delaunay retopo on static droplet")
    print("=" * 70)

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
        )

    # ---- BEFORE retopo ----
    print("\n--- BEFORE retopo ---")
    iface_verts_before = set()
    before = {}
    for v in HC.V:
        vid = v.x  # tuple key
        is_if = getattr(v, 'is_interface', False)
        if is_if:
            iface_verts_before.add(vid)
        before[vid] = {
            'is_interface': is_if,
            'phase': v.phase,
            'dual_vol': getattr(v, 'dual_vol', 0.0),
            'dvp': v.dual_vol_phase.copy() if hasattr(v, 'dual_vol_phase') else None,
            'p_phase': v.p_phase.copy() if hasattr(v, 'p_phase') else None,
            'm_phase': v.m_phase.copy() if hasattr(v, 'm_phase') else None,
            'p': float(v.p) if np.ndim(v.p) == 0 else float(v.p[0]),
        }

    # Interface edges before
    ie_before = set(getattr(HC, 'interface_edges', set()))
    print(f"  Interface vertices: {len(iface_verts_before)}")
    print(f"  Interface edges: {len(ie_before)}")

    # Force on interface vertices before
    forces_before = {}
    for v in HC.V:
        if getattr(v, 'is_interface', False):
            F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC)
            forces_before[v.x] = F.copy()

    max_F_before = max(np.linalg.norm(f) for f in forces_before.values())
    print(f"  Max |F| on interface: {max_F_before:.6e}")

    # ---- APPLY RETOPO (no vertex motion) ----
    print("\n--- Applying single retopo (NO vertex motion) ---")
    retopo_fn(HC, bV, dim)

    # ---- AFTER retopo ----
    print("\n--- AFTER retopo ---")
    iface_verts_after = set()
    after = {}
    for v in HC.V:
        vid = v.x
        is_if = getattr(v, 'is_interface', False)
        if is_if:
            iface_verts_after.add(vid)
        after[vid] = {
            'is_interface': is_if,
            'phase': v.phase,
            'dual_vol': getattr(v, 'dual_vol', 0.0),
            'dvp': v.dual_vol_phase.copy() if hasattr(v, 'dual_vol_phase') else None,
            'p_phase': v.p_phase.copy() if hasattr(v, 'p_phase') else None,
            'm_phase': v.m_phase.copy() if hasattr(v, 'm_phase') else None,
            'p': float(v.p) if np.ndim(v.p) == 0 else float(v.p[0]),
        }

    ie_after = set(getattr(HC, 'interface_edges', set()))
    print(f"  Interface vertices: {len(iface_verts_after)}")
    print(f"  Interface edges: {len(ie_after)}")

    # Force on interface vertices after
    forces_after = {}
    for v in HC.V:
        if getattr(v, 'is_interface', False):
            F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC)
            forces_after[v.x] = F.copy()

    max_F_after = max(np.linalg.norm(f) for f in forces_after.values())
    print(f"  Max |F| on interface: {max_F_after:.6e}")

    # ---- COMPARE ----
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Interface vertex changes
    gained = iface_verts_after - iface_verts_before
    lost = iface_verts_before - iface_verts_after
    print(f"  Interface vertices gained: {len(gained)}")
    print(f"  Interface vertices lost: {len(lost)}")
    if gained:
        for vid in list(gained)[:5]:
            print(f"    GAINED: {vid}")
    if lost:
        for vid in list(lost)[:5]:
            print(f"    LOST: {vid}")

    # Interface edge changes
    ie_gained = ie_after - ie_before
    ie_lost = ie_before - ie_after
    print(f"  Interface edges gained: {len(ie_gained)}")
    print(f"  Interface edges lost: {len(ie_lost)}")

    # Dual volume changes on interface vertices
    print("\n  Dual volume changes on interface vertices:")
    common = iface_verts_before & iface_verts_after
    max_dv_change = 0.0
    max_dvp_change = [0.0, 0.0]
    max_p_change = [0.0, 0.0]
    for vid in sorted(common, key=lambda x: np.arctan2(x[1], x[0])):
        if vid not in before or vid not in after:
            continue
        b = before[vid]
        a = after[vid]
        dv_change = a['dual_vol'] - b['dual_vol']
        max_dv_change = max(max_dv_change, abs(dv_change))

        if b['dvp'] is not None and a['dvp'] is not None:
            for k in range(2):
                dvp_ch = a['dvp'][k] - b['dvp'][k]
                max_dvp_change[k] = max(max_dvp_change[k], abs(dvp_ch))
        if b['p_phase'] is not None and a['p_phase'] is not None:
            for k in range(2):
                p_ch = a['p_phase'][k] - b['p_phase'][k]
                max_p_change[k] = max(max_p_change[k], abs(p_ch))

    print(f"    Max |Δdual_vol|: {max_dv_change:.6e}")
    print(f"    Max |Δdual_vol_phase[0]|: {max_dvp_change[0]:.6e}")
    print(f"    Max |Δdual_vol_phase[1]|: {max_dvp_change[1]:.6e}")
    print(f"    Max |Δp_phase[0]|: {max_p_change[0]:.6e}")
    print(f"    Max |Δp_phase[1]|: {max_p_change[1]:.6e}")

    # Detailed per-vertex comparison (sample)
    print("\n  Detailed per-interface-vertex changes (sample):")
    count = 0
    for vid in sorted(common, key=lambda x: np.arctan2(x[1], x[0])):
        if vid not in before or vid not in after:
            continue
        b = before[vid]
        a = after[vid]
        if b['dvp'] is None or a['dvp'] is None:
            continue
        r = np.linalg.norm(np.array(vid[:dim]))
        theta = np.degrees(np.arctan2(vid[1], vid[0]))

        dv0 = a['dvp'][0] - b['dvp'][0]
        dv1 = a['dvp'][1] - b['dvp'][1]
        dp0 = a['p_phase'][0] - b['p_phase'][0] if a['p_phase'] is not None else 0
        dp1 = a['p_phase'][1] - b['p_phase'][1] if a['p_phase'] is not None else 0

        F_before_mag = np.linalg.norm(forces_before.get(vid, np.zeros(2)))
        F_after_mag = np.linalg.norm(forces_after.get(vid, np.zeros(2)))

        if count < 8:
            print(f"    θ={theta:6.1f}° | ΔV0={dv0:+.3e} ΔV1={dv1:+.3e} | "
                  f"Δp0={dp0:+.3e} Δp1={dp1:+.3e} | "
                  f"|F|: {F_before_mag:.3e} → {F_after_mag:.3e}")
        count += 1

    # Force comparison
    print(f"\n  Force magnitude comparison:")
    print(f"    Before retopo: max |F| = {max_F_before:.6e}")
    print(f"    After retopo:  max |F| = {max_F_after:.6e}")
    print(f"    Ratio: {max_F_after / max_F_before:.2f}x")

    # Check bulk vertex pressure changes
    print("\n  Bulk vertex pressure changes:")
    max_p_change_bulk_drop = 0.0
    max_p_change_bulk_outer = 0.0
    for vid in before:
        if vid not in after:
            continue
        b = before[vid]
        a = after[vid]
        if b['is_interface'] or a['is_interface']:
            continue
        dp = abs(a['p'] - b['p'])
        if b['phase'] == 1:
            max_p_change_bulk_drop = max(max_p_change_bulk_drop, dp)
        elif b['phase'] == 0:
            max_p_change_bulk_outer = max(max_p_change_bulk_outer, dp)

    print(f"    Max |Δp| on bulk droplet: {max_p_change_bulk_drop:.6e}")
    print(f"    Max |Δp| on bulk outer:   {max_p_change_bulk_outer:.6e}")


if __name__ == '__main__':
    main()
