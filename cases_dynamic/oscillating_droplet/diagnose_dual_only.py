#!/usr/bin/env python3
"""Diagnostic: static droplet with dual-only retopo (no Delaunay).

Same as static_droplet_2D.py but uses dual_only_retopo like
cube2droplet's diagnostic_no_retopo.py.  Isolates whether the
instability is from retopologization or from the force balance.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hyperct.ddg import compute_vd

from cases_dynamic.oscillating_droplet.src._params import (
    R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    compute_diagnostics,
)
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.dynamic_integrators import symplectic_euler


def dual_only_retopo(HC, bV, dim, _mps=None):
    """Recompute duals on existing connectivity (no Delaunay)."""
    dV = HC.boundary()
    for v in HC.V:
        v.boundary = v in dV
    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)
    if _mps is not None:
        _mps.split_dual_volumes(HC, dim)
    bV.clear()
    bV.update(dV)


def main():
    dim = 2
    epsilon = 0.0
    print("=" * 60)
    print("DIAGNOSTIC: Static Droplet — DUAL-ONLY retopo")
    print("=" * 60)

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
        )
    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"Mesh: {n_verts} vertices, {n_iface} interface")

    c_s = float(np.sqrt(K_d / rho_d))
    dx_min = min(
        float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min ** 3 / gamma) if gamma > 0 else 1.0)
    n_steps = 100
    print(f"dt={dt:.2e}, n_steps={n_steps}")

    from functools import partial
    retopo_dual_only = partial(dual_only_retopo, _mps=mps)

    diag_list = []

    def record(t):
        d = compute_diagnostics(HC, dim=dim)
        d['t'] = float(t)
        diag_list.append(d)

    record(0.0)

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if step % 10 == 0:
            record(t)
            d = diag_list[-1]
            n_if = sum(1 for v in HC_cb.V if getattr(v, 'is_interface', False))
            print(f"  step={step:4d} t={t:.4e} | KE={d['KE']:.6e} | "
                  f"R_max={d['R_max']:.6f} R_min={d['R_min']:.6f} | "
                  f"n_iface={n_if}")

    print("\n--- Running with DUAL-ONLY retopo ---")
    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=callback,
        retopologize_fn=retopo_dual_only,
    )
    record(t_final)

    # Summary
    KE_arr = [d['KE'] for d in diag_list]
    print(f"\nFinal KE = {KE_arr[-1]:.6e}")
    print(f"Max KE   = {max(KE_arr):.6e}")
    print(f"KE[0]    = {KE_arr[0]:.6e}")

    # Also run with FULL Delaunay retopo for comparison
    print("\n\n" + "=" * 60)
    print("COMPARISON: Static Droplet — FULL DELAUNAY retopo")
    print("=" * 60)

    HC2, bV2, mps2, bc_set2, dudt_fn2, retopo_fn2, params2 = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
        )

    diag_list2 = []

    def record2(t):
        d = compute_diagnostics(HC2, dim=dim)
        d['t'] = float(t)
        diag_list2.append(d)

    record2(0.0)

    def callback2(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if step % 10 == 0:
            record2(t)
            d = diag_list2[-1]
            n_if = sum(1 for v in HC_cb.V if getattr(v, 'is_interface', False))
            print(f"  step={step:4d} t={t:.4e} | KE={d['KE']:.6e} | "
                  f"R_max={d['R_max']:.6f} R_min={d['R_min']:.6f} | "
                  f"n_iface={n_if}")

    print("\n--- Running with FULL DELAUNAY retopo ---")
    try:
        t_final2 = symplectic_euler(
            HC2, bV2, dudt_fn2, dt=dt, n_steps=n_steps, dim=dim,
            bc_set=bc_set2, callback=callback2,
            retopologize_fn=retopo_fn2,
            remesh_mode=params2['remesh_mode'],
            remesh_kwargs=params2['remesh_kwargs'],
        )
        record2(t_final2)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    KE_arr2 = [d['KE'] for d in diag_list2]
    print(f"\nFinal KE = {KE_arr2[-1]:.6e}")
    print(f"Max KE   = {max(KE_arr2):.6e}")
    print(f"KE[0]    = {KE_arr2[0]:.6e}")

    # Compare
    print("\n\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Dual-only:  Max KE = {max(KE_arr):.6e}, Final KE = {KE_arr[-1]:.6e}")
    print(f"  Delaunay:   Max KE = {max(KE_arr2):.6e}, Final KE = {KE_arr2[-1]:.6e}")
    ratio = max(KE_arr2) / max(KE_arr) if max(KE_arr) > 0 else float('inf')
    print(f"  Ratio (Delaunay/dual-only): {ratio:.1f}x")


if __name__ == '__main__':
    main()
