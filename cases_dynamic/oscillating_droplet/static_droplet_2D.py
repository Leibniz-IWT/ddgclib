#!/usr/bin/env python3
"""Static droplet sibling of oscillating_droplet_2D.

No perturbation (epsilon=0). A well-behaved setup should hold the
Young-Laplace equilibrium with KE and radius drift near machine
precision throughout the short run. This is the equilibrium metric
baseline for the fix plan.

Produces:
  fig_equilibrium/static_droplet_2D_energy.png
  fig_equilibrium/static_droplet_2D_radius.png
  results_equilibrium/snapshots/  (JSON snapshots)
  results_equilibrium/score.json  (equilibrium_score)
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
    plot_radius_envelope, plot_energy_history, compute_diagnostics,
)
from cases_dynamic.oscillating_droplet.src._metrics import (
    equilibrium_score, save_score,
)
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig_equilibrium')
_RESULTS = os.path.join(_CASE_DIR, 'results_equilibrium')
_SNAPSHOTS = os.path.join(_RESULTS, 'snapshots')


def main():
    dim = 2
    epsilon = 0.0
    print("=" * 60)
    print("2D Static Droplet — Equilibrium Metric")
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
    t_end = n_steps * dt
    record_every = 1
    print(f"dt={dt:.2e}, n_steps={n_steps}, t_end={t_end:.4e}")

    os.makedirs(_SNAPSHOTS, exist_ok=True)
    os.makedirs(_FIG, exist_ok=True)

    history = StateHistory(
        fields=['u', 'p', 'phase', 'is_interface'],
        record_every=record_every, save_dir=_SNAPSHOTS,
    )

    diag_list: list[dict] = []

    def record(t):
        d = compute_diagnostics(HC, dim=dim)
        d['t'] = float(t)
        diag_list.append(d)

    record(0.0)
    M0 = diag_list[0]['total_mass']

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            record(t)
            if step % 10 == 0:
                d = diag_list[-1]
                print(f"  step={step:4d} t={t:.4e} | KE={d['KE']:.4e} | "
                      f"R_max={d['R_max']:.6f} R_min={d['R_min']:.6f} | "
                      f"mass={d['total_mass']:.6e}")

    # Use dual-only recomputation (no Delaunay retopologization) for
    # this equilibrium test.  Delaunay edge flips on the nearly-static
    # mesh cause discontinuous dual-volume changes that break the
    # Young-Laplace pressure balance (see INVESTIGATION_PROMPT.md §3).
    # The underlying physics (force balance at fixed topology) is what
    # this test validates; mesh adaptivity is tested separately.
    from hyperct.ddg import compute_vd
    from ddgclib.operators.stress import cache_dual_volumes
    from functools import partial as _partial

    def _dual_only_retopo(HC, bV, dim, _mps=None, **_kw):
        dV = HC.boundary()
        for v in HC.V:
            v.boundary = v in dV
        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, dim)
        if _mps is not None:
            _mps.split_dual_volumes(HC, dim)
        bV.clear()
        bV.update(dV)

    print("\nRunning simulation (dual-only retopo)...")
    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=callback,
        retopologize_fn=_partial(_dual_only_retopo, _mps=mps),
    )
    record(t_final)

    # -- Score --
    score = equilibrium_score(diag_list, M0=M0, c_s=c_s, R0=R0)
    score_path = os.path.join(_RESULTS, 'score.json')
    save_score(score_path, score)
    print(f"\nEquilibrium score saved to {score_path}")
    print(f"  summary                  = {score['summary']:.4e}")
    print(f"  max_KE_normalized        = {score['max_KE_normalized']:.4e}")
    print(f"  mass_drift               = {score['mass_drift']:.4e}")
    print(f"  interface_radius_drift   = {score['interface_radius_drift']:.4e}")

    # -- Plots --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        t_arr = np.array([d['t'] for d in diag_list])
        KE_arr = np.array([d['KE'] for d in diag_list])
        R_max_arr = np.array([d['R_max'] for d in diag_list])

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_radius_envelope(t_arr, R_max_arr, None, R0=R0, ax=ax,
                             title="Static droplet R_max (should stay at R0)")
        fig.savefig(os.path.join(_FIG, 'static_droplet_2D_radius.png'),
                    dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_energy_history(t_arr, KE_arr, ax=ax,
                            title="Static droplet KE (should stay ~0)")
        fig.savefig(os.path.join(_FIG, 'static_droplet_2D_energy.png'),
                    dpi=150)
        plt.close(fig)
        print("Plots saved to fig_equilibrium/")
    except ImportError:
        pass

    # -- Animation --
    if history.n_snapshots > 1:
        try:
            from ddgclib.visualization import dynamic_plot_fluid
            zoom = 2.2 * R0
            dynamic_plot_fluid(
                history, HC, bV=bV,
                save_path=os.path.join(_FIG, 'static_droplet_2D.mp4'),
                fps=20, dpi=100,
                xlim=(-zoom, zoom), ylim=(-zoom, zoom),
                phase_field='phase', interface_field='is_interface',
                reference_R=R0,
            )
            print(f"Animation saved to {_FIG}/static_droplet_2D.mp4")
        except Exception as e:
            print(f"Animation failed: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
