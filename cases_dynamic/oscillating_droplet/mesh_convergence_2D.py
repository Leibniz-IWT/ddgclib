#!/usr/bin/env python3
"""Mesh convergence study for the 2D oscillating droplet.

Runs the simulation at multiple refinement levels and measures how
the solution converges as the mesh is refined.  The key metric is
the max-radius envelope error vs the Lamb/Rayleigh analytical solution.

The research goal is mesh independence: finding the same numerical
solution (not necessarily the exact analytical one) at different
refinement levels.

Usage
-----
    python cases_dynamic/oscillating_droplet/mesh_convergence_2D.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, epsilon, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, beta_2d,
)
from cases_dynamic.oscillating_droplet.src._analytical import (
    rayleigh_frequency, lamb_damping_rate, max_radius_envelope,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    compute_diagnostics,
)
from ddgclib.dynamic_integrators import symplectic_euler


def run_single(refinement_outer, refinement_droplet, n_steps_max=500):
    """Run one simulation at a given refinement level."""
    dim = 2
    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o,
            L_domain=L_domain,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
        )

    n_verts = sum(1 for _ in HC.V)
    c_s = np.sqrt(K_d / rho_d)
    dt = 0.1 * R0 / c_s
    t_end = min(2.0 / beta_2d, 0.01)
    n_steps = min(int(t_end / dt) + 1, n_steps_max)
    record_every = max(1, n_steps // 50)

    t_arr = [0.0]
    R_max_arr = [compute_diagnostics(HC, dim=dim)['R_max']]

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if step % record_every != 0:
            return
        diag = compute_diagnostics(HC_cb, dim=dim)
        t_arr.append(t)
        R_max_arr.append(diag['R_max'])

    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=callback,
        retopologize_fn=retopo_fn,
    )

    # Final
    diag_final = compute_diagnostics(HC, dim=dim)
    t_arr.append(t_final)
    R_max_arr.append(diag_final['R_max'])

    return n_verts, np.array(t_arr), np.array(R_max_arr)


def main():
    dim = 2
    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)

    print("=" * 60)
    print("2D Oscillating Droplet — Mesh Convergence Study")
    print("=" * 60)

    # Refinement levels to test
    levels = [
        (1, 2),   # coarse
        (2, 3),   # medium
        (3, 4),   # fine
    ]

    results = []
    for ref_outer, ref_drop in levels:
        print(f"\n--- Refinement: outer={ref_outer}, droplet={ref_drop} ---")
        n_verts, t_arr, R_max_arr = run_single(ref_outer, ref_drop)
        R_max_analytical = max_radius_envelope(t_arr, R0, epsilon, omega, beta)

        # L2 error
        dt_arr = np.diff(np.concatenate([[0], t_arr]))
        err = R_max_arr - R_max_analytical
        L2 = np.sqrt(np.sum(err**2 * dt_arr) / t_arr[-1])
        Linf = np.max(np.abs(err))

        print(f"  Vertices: {n_verts}")
        print(f"  L2 error:   {L2:.6e}")
        print(f"  Linf error: {Linf:.6e}")
        results.append((n_verts, L2, Linf, t_arr, R_max_arr))

    # -- Convergence summary --
    print("\n" + "=" * 60)
    print("Convergence Summary")
    print("=" * 60)
    print(f"{'N_verts':>10} {'L2 error':>12} {'Linf error':>12}")
    for n_verts, L2, Linf, _, _ in results:
        print(f"{n_verts:>10d} {L2:>12.4e} {Linf:>12.4e}")

    if len(results) >= 2:
        # Approximate convergence order
        for i in range(1, len(results)):
            n1, L2_1, _, _, _ = results[i - 1]
            n2, L2_2, _, _, _ = results[i]
            if L2_1 > 0 and L2_2 > 0 and n1 != n2:
                h_ratio = (n1 / n2) ** (1.0 / dim)
                order = np.log(L2_1 / L2_2) / np.log(1 / h_ratio)
                print(f"  Order ({n1}->{n2}): {order:.2f}")

    # -- Mesh independence check --
    # Compare solutions at different refinements
    print("\n--- Mesh Independence ---")
    if len(results) >= 2:
        _, _, _, t1, R1 = results[-2]
        _, _, _, t2, R2 = results[-1]
        # Interpolate to common time grid
        t_common = np.linspace(0, min(t1[-1], t2[-1]), 50)
        R1_interp = np.interp(t_common, t1, R1)
        R2_interp = np.interp(t_common, t2, R2)
        mesh_diff = np.max(np.abs(R1_interp - R2_interp))
        print(f"  Max |R_coarse - R_fine|: {mesh_diff:.6e}")
        print(f"  Relative: {mesh_diff / R0:.6e}")

    # -- Plot --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs('fig', exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: R_max(t) at all refinements
        for n_verts, _, _, t_arr, R_max_arr in results:
            ax1.plot(t_arr, R_max_arr, '-o', markersize=2,
                     label=f'N={n_verts}')
        # Analytical
        t_fine = np.linspace(0, results[-1][3][-1], 200)
        R_anal = max_radius_envelope(t_fine, R0, epsilon, omega, beta)
        ax1.plot(t_fine, R_anal, 'k--', label='Analytical')
        ax1.axhline(R0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("R_max [m]")
        ax1.set_title("R_max(t) at different refinements")
        ax1.legend()

        # Right: Convergence plot
        n_arr = [r[0] for r in results]
        L2_arr = [r[1] for r in results]
        ax2.loglog(n_arr, L2_arr, 'bo-', label='L2 error')
        ax2.set_xlabel("Number of vertices")
        ax2.set_ylabel("L2 error")
        ax2.set_title("Mesh convergence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig('fig/oscillating_droplet_2D_convergence.png', dpi=150)
        plt.close(fig)
        print("\nPlot saved to fig/oscillating_droplet_2D_convergence.png")
    except ImportError:
        print("(matplotlib not available, skipping plots)")

    print("\nDone.")


if __name__ == '__main__':
    main()
