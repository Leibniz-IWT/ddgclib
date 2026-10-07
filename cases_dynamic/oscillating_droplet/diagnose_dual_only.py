#!/usr/bin/env python3
"""Diagnostic: static droplet with dual-only retopo (no Delaunay).

Same as static_droplet_2D.py (``PRESETS['static_droplet_2D']``,
connectivity='dual_only_bare': the library ``bare_dual_refresh``, which
is the closure this script used to carry) against the setup's bare
per-step Delaunay (``PRESETS['oscillating_droplet_2D_bare_delaunay']``).
Isolates whether the instability is from retopologization or from the
force balance.
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
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    compute_diagnostics,
)
from ddgclib.methods import PRESETS


def main():
    dim = 2
    epsilon = 0.0
    print("=" * 60)
    print("DIAGNOSTIC: Static Droplet — DUAL-ONLY retopo")
    print("=" * 60)

    methods = PRESETS['static_droplet_2D']
    print(methods.describe())
    HC, bV, mps, bc_set, dudt_fn, _setup_retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
            methods=methods,
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
    t_final = methods.integrate(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
        bc_set=bc_set, callback=callback, mps=mps,
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

    methods2 = PRESETS['oscillating_droplet_2D_bare_delaunay']
    print(methods2.describe())
    HC2, bV2, mps2, bc_set2, dudt_fn2, _setup_retopo_fn2, params2 = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
            methods=methods2,
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
        t_final2 = methods2.integrate(
            HC2, bV2, dudt_fn2, dt=dt, n_steps=n_steps,
            bc_set=bc_set2, callback=callback2, mps=mps2,
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
