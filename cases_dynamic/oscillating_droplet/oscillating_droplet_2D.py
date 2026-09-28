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

from functools import partial

from cases_dynamic.oscillating_droplet.src._params import (
    R0, epsilon, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet, beta_2d, t_end_2d,
    retopo_policy_2d,
)
from cases_dynamic.oscillating_droplet.src._analytical import (
    rayleigh_frequency, lamb_damping_rate, damped_frequency,
    max_radius_envelope, lamb_damping_rate_two_fluid,
    two_fluid_omega_beta_2d,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    plot_droplet_fluid, plot_droplet_phases,
    plot_radius_envelope, plot_energy_history, plot_apex_trajectory,
    compute_diagnostics,
)
from cases_dynamic.oscillating_droplet.src._metrics import (
    oscillation_score, save_score, add_two_fluid_reference,
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

    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim, rho_outer=rho_o)
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

    # -- Retopology policy (lane-5 sweep + lane-E adoption, see
    # src/_params.py) --
    # 'delaunay_remap' (default): per-step FULL Delaunay reconnection
    # kept thermodynamically neutral by the lane-D conservative remap
    # (pressure structure restored bit-exactly across each rebuild,
    # per-phase EOS volume gauge, p_ref-style level anchor).  Scores
    # l2 0.17479 / tail 0.99990 on this run while retopology stays
    # honestly active.
    # 'dual_only' (opt-in): keep the interface-conforming builder
    # connectivity for the whole run; refresh duals / per-phase split /
    # mass redistribution / EOS pressures every step (l2 0.17857 /
    # tail 0.99925).  Bare per-step Delaunay without the remap pumps
    # spurious KE (~5e4x physical; l2 0.490, tail 1.725).
    if retopo_policy_2d == 'dual_only':
        retopo_fn = partial(retopo_fn, skip_triangulation=True)
    elif retopo_policy_2d == 'delaunay_remap':
        retopo_fn = partial(retopo_fn, retopo_remap='conservative')
    print(f"Retopo policy: {retopo_policy_2d}")

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

    diag_list: list[dict] = []

    def record(t):
        d = compute_diagnostics(HC, dim=dim)
        d['t'] = float(t)
        diag_list.append(d)

    record(0.0)

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            record(t)
            if step % (record_every * 10) == 0:
                d = diag_list[-1]
                print(f"  t={t:.4e} | R_max={d['R_max']:.6f} | "
                      f"r_apex={d['r_apex']:.6f} | "
                      f"KE={d['KE']:.4e} | mass={d['total_mass']:.6f}")

    # -- Run --
    print("\nRunning simulation...")
    # remesh_mode / remesh_kwargs must be passed through the integrator:
    # _do_retopologize inspects retopo_fn's signature and, if it accepts
    # `remesh_mode`, forwards the integrator's value — so pre-binding
    # via functools.partial alone does not work.
    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=callback, retopologize_fn=retopo_fn,
        remesh_mode=params['remesh_mode'],
        remesh_kwargs=params['remesh_kwargs'],
    )

    record(t_final)

    t_arr = np.array([d['t'] for d in diag_list])
    R_max_arr = np.array([d['R_max'] for d in diag_list])
    KE_arr = np.array([d['KE'] for d in diag_list])
    mass_arr = np.array([d['total_mass'] for d in diag_list])
    r_apex_arr = np.array([d['r_apex'] for d in diag_list])
    theta_apex_arr = np.array([d['theta_apex'] for d in diag_list])
    R_max_analytical = max_radius_envelope(t_arr, R0, epsilon, omega, beta, l=l)

    print(f"\nMass conservation: |dM/M0| = "
          f"{abs(mass_arr[-1] - mass_arr[0]) / mass_arr[0]:.4e}")
    print(f"Final R_max: {R_max_arr[-1]:.6f} (R0 = {R0})")
    print(f"StateHistory: {history.n_snapshots} snapshots")

    # -- Score --
    score = oscillation_score(
        diag_list, R0=R0, epsilon=epsilon, l=l, omega=omega, beta=beta,
    )
    # Secondary two-fluid reference (lane C): exact 2D two-fluid
    # normal-mode dispersion (outer bath rho_o/mu_o included; no-slip
    # wall at 5*R0 still NOT modelled).  The pinned metric above stays
    # the single-fluid Lamb one for regression continuity.
    omega_tf, beta_tf = two_fluid_omega_beta_2d(
        l, gamma, mu_d, rho_d, R0, mu_outer=mu_o, rho_outer=rho_o,
    )
    beta_tf_energy = lamb_damping_rate_two_fluid(
        l, mu_d, rho_d, R0, mu_outer=mu_o, rho_outer=rho_o,
    )
    score = add_two_fluid_reference(
        score, diag_list, R0=R0, epsilon=epsilon, l=l,
        omega_two_fluid=omega_tf, beta_two_fluid=beta_tf,
        beta_energy=beta_tf_energy,
    )
    score_path = os.path.join(_RESULTS, 'score.json')
    save_score(score_path, score)
    # Raw diagnostic series alongside (scalar fields only — 'com' is an
    # ndarray), so future reference changes can re-score without
    # re-running the case.
    diag_scalars = [
        {k: float(v) for k, v in dd.items() if np.ndim(v) == 0}
        for dd in diag_list
    ]
    save_score(os.path.join(_RESULTS, 'diag_series.json'),
               {'kind': 'diag_series', 'diags': diag_scalars})
    print(f"\nOscillation score saved to {score_path}")
    print(f"  summary                  = {score['summary']:.4e}")
    print(f"  l2_error_normalized      = {score['l2_error_normalized']:.4e}")
    print(f"  linf_error_normalized    = {score['linf_error_normalized']:.4e}")
    print(f"  tail_growth              = {score['tail_growth']:.4e}")
    print(f"  mass_drift               = {score['mass_drift']:.4e}")
    print(f"  -- two-fluid reference (omega={omega_tf:.4f}, "
          f"beta={beta_tf:.4f}; energy-method beta={beta_tf_energy:.4f}) --")
    print(f"  l2_error_norm_two_fluid  = "
          f"{score['l2_error_normalized_two_fluid']:.4e}")
    print(f"  linf_error_norm_two_fluid= "
          f"{score['linf_error_normalized_two_fluid']:.4e}")

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
        plot_energy_history(t_arr, KE_arr, ax=ax)
        fig.savefig(os.path.join(_FIG, 'oscillating_droplet_2D_energy.png'), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_apex_trajectory(
            t_arr, r_apex_arr, theta_apex_arr,
            R0=R0, epsilon=epsilon, l=l, omega=omega, beta=beta, ax=ax,
        )
        fig.savefig(os.path.join(_FIG, 'oscillating_droplet_2D_apex.png'), dpi=150)
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
