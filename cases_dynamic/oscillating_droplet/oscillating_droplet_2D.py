#!/usr/bin/env python3
"""2D oscillating droplet simulation.

Produces:
  fig/oscillating_droplet_2D_fluid.png    — final pressure + velocity snapshot
  fig/oscillating_droplet_2D_phases.png   — final phase field
  fig/oscillating_droplet_2D_radius.png   — R_max(t) vs analytical
  fig/oscillating_droplet_2D_energy.png   — KE(t)
  fig/oscillating_droplet_2D.mp4          — animation (pressure + velocity + interface)
  results/oscillating_droplet_2D/snapshots/ — JSON snapshots for polyscope

Usage
-----
    python cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, epsilon, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet, beta_2d, t_end_2d,
)
from cases_dynamic.oscillating_droplet.src._analytical import (
    rayleigh_frequency, lamb_damping_rate, damped_frequency,
    max_radius_envelope,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    plot_droplet_fluid, plot_droplet_phases,
    plot_radius_envelope, plot_energy_history, compute_diagnostics,
)
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results')
_SNAPSHOTS = os.path.join(_RESULTS, 'snapshots')


def main():
    dim = 2
    print("=" * 60)
    print("2D Oscillating Droplet — Overdamped Case")
    print("=" * 60)

    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)
    omega_d = damped_frequency(omega, beta)
    regime = "underdamped" if omega_d > 0 else "overdamped"
    print(f"Mode l={l}: omega={omega:.2f}, beta={beta:.2f}, "
          f"omega_d={omega_d:.2f} ({regime})")

    # -- Setup --
    print("\nBuilding mesh...")
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

    # -- CFL timestep --
    c_s = np.sqrt(K_d / rho_d)
    dx_min = min(
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min**3 / gamma) if gamma > 0 else 1.0)
    t_end = min(t_end_2d, 5.0 / beta if beta > 0 else 0.01)
    n_steps = int(t_end / dt) + 1
    record_every = max(1, n_steps // 200)
    print(f"dt={dt:.2e}, n_steps={n_steps}, t_end={t_end:.4f}")

    # -- Recording --
    os.makedirs(_SNAPSHOTS, exist_ok=True)
    os.makedirs(_FIG, exist_ok=True)

    # StateHistory records u, p AND phase/interface for multiphase animation
    history = StateHistory(
        fields=['u', 'p', 'phase', 'is_interface'],
        record_every=record_every,
        save_dir=_SNAPSHOTS,
    )

    t_arr, R_max_arr, KE_arr, mass_arr = [0.0], [], [], []
    diag0 = compute_diagnostics(HC, dim=dim)
    R_max_arr.append(diag0['R_max'])
    KE_arr.append(diag0['KE'])
    mass_arr.append(diag0['total_mass'])

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            diag = compute_diagnostics(HC_cb, dim=dim)
            t_arr.append(t)
            R_max_arr.append(diag['R_max'])
            KE_arr.append(diag['KE'])
            mass_arr.append(diag['total_mass'])
            if step % (record_every * 10) == 0:
                print(f"  t={t:.4e} | R_max={diag['R_max']:.6f} | "
                      f"KE={diag['KE']:.4e} | mass={diag['total_mass']:.6f}")

    # -- Run --
    print("\nRunning simulation...")
    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=callback, retopologize_fn=retopo_fn,
    )

    diag_final = compute_diagnostics(HC, dim=dim)
    t_arr.append(t_final)
    R_max_arr.append(diag_final['R_max'])
    KE_arr.append(diag_final['KE'])
    mass_arr.append(diag_final['total_mass'])

    t_arr = np.array(t_arr)
    R_max_arr = np.array(R_max_arr)
    R_max_analytical = max_radius_envelope(t_arr, R0, epsilon, omega, beta)

    print(f"\nMass conservation: |dM/M0| = "
          f"{abs(mass_arr[-1] - mass_arr[0]) / mass_arr[0]:.4e}")
    print(f"Final R_max: {R_max_arr[-1]:.6f} (R0 = {R0})")
    print(f"StateHistory: {history.n_snapshots} snapshots")

    # -- Static plots --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plot_droplet_fluid(HC, bV=bV, t=t_final, dim=dim,
                           save_path=os.path.join(_FIG, 'oscillating_droplet_2D_fluid.png'))
        plot_droplet_phases(HC, bV=bV, dim=dim,
                            title=f"Phase field (t={t_final:.4f} s)",
                            save_path=os.path.join(_FIG, 'oscillating_droplet_2D_phases.png'))

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_radius_envelope(t_arr, R_max_arr, R_max_analytical, R0=R0, ax=ax)
        fig.savefig(os.path.join(_FIG, 'oscillating_droplet_2D_radius.png'), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_energy_history(t_arr, np.array(KE_arr), ax=ax)
        fig.savefig(os.path.join(_FIG, 'oscillating_droplet_2D_energy.png'), dpi=150)
        plt.close(fig)
        print("Static plots saved to fig/")
    except ImportError:
        pass

    # -- Animation (uses unified dynamic_plot_fluid with phase overlay) --
    if history.n_snapshots > 1:
        try:
            anim = dynamic_plot_fluid(
                history, HC, bV=bV,
                save_path=os.path.join(_FIG, 'oscillating_droplet_2D.mp4'),
                fps=20, dpi=100,
                xlim=(-2.5 * R0, 2.5 * R0),
                ylim=(-2.5 * R0, 2.5 * R0),
                phase_field='phase',
                interface_field='is_interface',
                reference_R=R0,
            )
            print(f"Animation saved to fig/oscillating_droplet_2D.mp4")
        except Exception as e:
            print(f"Animation failed: {e}")

    print(f"\nTo view in polyscope:")
    print(f"  python -m ddgclib.scripts.view_polyscope "
          f"--snapshots {os.path.relpath(_SNAPSHOTS)}")
    print("Done.")


if __name__ == '__main__':
    main()
