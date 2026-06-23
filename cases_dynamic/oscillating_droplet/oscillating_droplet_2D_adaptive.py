#!/usr/bin/env python3
"""Adaptive vs Delaunay comparison for the 2D oscillating droplet.

Side-by-side comparison of ``remesh_mode='delaunay'`` (current default)
vs ``remesh_mode='adaptive'`` (interface-preserving local mesh
operations from ``hyperct.remesh``).  Each mode runs on an independent
copy of the same initial mesh and produces a set of diagnostic time
series that are plotted together.

Usage
-----
    # Quick smoke test (~20 seconds: 200 steps per mode, coarse mesh)
    python cases_dynamic/oscillating_droplet/oscillating_droplet_2D_adaptive.py

    # Full comparison (~80 minutes: 5 damping times, production mesh)
    python cases_dynamic/oscillating_droplet/oscillating_droplet_2D_adaptive.py --full

    # Custom: more steps but still manageable (~5-10 min)
    python cases_dynamic/oscillating_droplet/oscillating_droplet_2D_adaptive.py --n-steps 2000

Key metrics compared
--------------------
- **R_max(t)** — maximum interface radius; Rayleigh-Lamb analytical
  envelope is the reference
- **KE(t)** — kinetic energy; should decay monotonically after the
  initial pressure wave
- **mass conservation** — |dM/M0|; should be < 1e-6
- **interface edge count** — how many cross-phase edges at each
  diagnostic frame (Delaunay typically erodes, adaptive preserves)
- **oscillation_score** — L2 error of r_apex vs analytical
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, epsilon, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, beta_2d, t_end_2d,
)
from cases_dynamic.oscillating_droplet.src._analytical import (
    rayleigh_frequency, lamb_damping_rate, damped_frequency,
    max_radius_envelope,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    compute_diagnostics,
)
from cases_dynamic.oscillating_droplet.src._metrics import (
    oscillation_score, save_score,
)
from ddgclib.dynamic_integrators import symplectic_euler
from hyperct.remesh import is_interface_edge

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results')


def _count_interface_edges(HC) -> int:
    seen = set()
    n = 0
    for v in HC.V:
        for nb in v.nn:
            k = frozenset((id(v), id(nb)))
            if k in seen:
                continue
            seen.add(k)
            if is_interface_edge(v, nb):
                n += 1
    return n


def run_one_mode(
    mode_label: str,
    remesh_mode: str,
    remesh_kwargs: dict | None,
    dim: int,
    dt: float,
    n_steps: int,
    record_every: int,
    refine_outer: int,
    refine_droplet: int,
) -> dict:
    """Run the oscillating droplet with a given remesh mode and return
    time-series diagnostics."""
    print(f"\n{'=' * 60}")
    print(f"  {mode_label}")
    print(f"{'=' * 60}")

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=refine_outer,
            refinement_droplet=refine_droplet,
        )

    n_verts = sum(1 for _ in HC.V)
    n_iface = _count_interface_edges(HC)
    print(f"  Mesh: {n_verts} vertices, {n_iface} interface edges")
    print(f"  remesh_mode='{remesh_mode}', dt={dt:.2e}, n_steps={n_steps}")

    diag_list: list[dict] = []

    def record(t):
        d = compute_diagnostics(HC, dim=dim)
        d['t'] = float(t)
        d['n_verts'] = sum(1 for _ in HC.V)
        d['n_iface_edges'] = _count_interface_edges(HC)
        diag_list.append(d)

    record(0.0)

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if step % record_every == 0:
            record(t)
        if step % max(1, n_steps // 5) == 0 and step > 0:
            d = compute_diagnostics(HC_cb, dim=dim)
            print(f"  step {step}/{n_steps}: t={t:.4e} R_max={d['R_max']:.6f} "
                  f"KE={d['KE']:.4e}")

    t0 = time.time()
    try:
        t_final = symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
            bc_set=bc_set, callback=callback, retopologize_fn=retopo_fn,
            remesh_mode=remesh_mode,
            remesh_kwargs=remesh_kwargs,
        )
    except Exception as e:
        print(f"  STOPPED: {e}")
        t_final = diag_list[-1]['t'] if diag_list else 0.0

    wall = time.time() - t0
    record(t_final)

    print(f"  Wall time: {wall:.1f}s ({wall / max(1, n_steps):.3f}s/step)")
    print(f"  Final: {diag_list[-1]['n_verts']} verts, "
          f"{diag_list[-1]['n_iface_edges']} iface edges")

    return {
        'label': mode_label,
        'remesh_mode': remesh_mode,
        'wall_time_s': wall,
        'diags': diag_list,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive vs Delaunay oscillating droplet comparison")
    parser.add_argument('--full', action='store_true',
                        help='Full run: production mesh, 5 damping times '
                             '(~80 min total)')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Override step count')
    parser.add_argument('--refine', type=int, default=None,
                        help='Override refinement level (outer & droplet)')
    args = parser.parse_args()

    dim = 2
    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim, rho_outer=rho_o)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)
    omega_d = damped_frequency(omega, beta)
    regime = "underdamped" if omega_d > 0 else "overdamped"

    if args.full:
        refine_outer = 3
        refine_droplet = 3
    else:
        refine_outer = args.refine or 2
        refine_droplet = args.refine or 2

    # Build a throwaway mesh to compute CFL dt
    HC_tmp, _, _, _, _, _, _ = setup_oscillating_droplet(
        dim=dim, R0=R0, epsilon=epsilon, l=l,
        rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
        gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
        refinement_outer=refine_outer, refinement_droplet=refine_droplet,
    )
    c_s = np.sqrt(K_d / rho_d)
    dx_min = min(
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC_tmp.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min**3 / gamma) if gamma > 0 else 1.0)
    del HC_tmp

    if args.n_steps is not None:
        n_steps = args.n_steps
    elif args.full:
        t_end = min(t_end_2d, 5.0 / beta if beta > 0 else 0.01)
        n_steps = int(t_end / dt) + 1
    else:
        n_steps = 200  # quick smoke test

    record_every = max(1, n_steps // 100)

    print("=" * 60)
    print("Oscillating Droplet: Delaunay vs Adaptive Remesh")
    print("=" * 60)
    print(f"Mode l={l}: omega={omega:.2f}, beta={beta:.2f}, "
          f"omega_d={omega_d:.2f} ({regime})")
    print(f"Mesh: refine_outer={refine_outer}, refine_droplet={refine_droplet}")
    print(f"dt={dt:.2e}, n_steps={n_steps}, "
          f"t_physical={n_steps * dt:.4f}s")
    est_per_step = 0.05 if refine_outer <= 2 else 0.3
    print(f"Estimated wall time: ~{2 * n_steps * est_per_step / 60:.0f} min total")

    # Adaptive remesh kwargs: conservative thresholds sized to the
    # mesh's edge-length distribution.
    edge_lens = [
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=refine_outer, refinement_droplet=refine_droplet,
        )[0].V
        for nb in v.nn
    ]
    h_mean = float(np.mean(edge_lens)) if edge_lens else R0 * 0.1

    adaptive_kwargs = {
        'L_min': 0.3 * h_mean,
        'L_max': 2.5 * h_mean,
        'quality_target_deg': 20.0,
        'max_iterations': 1,
        'smooth_iterations': 1,
        'smooth_relax': 0.2,
    }

    # --- Run both modes ---
    results = []
    for label, mode, kwargs in [
        ("Delaunay (default)", 'delaunay', None),
        ("Adaptive (interface-preserving)", 'adaptive', adaptive_kwargs),
    ]:
        r = run_one_mode(
            mode_label=label,
            remesh_mode=mode,
            remesh_kwargs=kwargs,
            dim=dim, dt=dt, n_steps=n_steps, record_every=record_every,
            refine_outer=refine_outer, refine_droplet=refine_droplet,
        )
        results.append(r)

    # --- Compute oscillation scores ---
    print("\n" + "=" * 60)
    print("Oscillation Scores")
    print("=" * 60)
    for r in results:
        score = oscillation_score(
            r['diags'], R0=R0, epsilon=epsilon, l=l,
            omega=omega, beta=beta,
        )
        r['score'] = score
        print(f"\n  {r['label']}:")
        print(f"    L2 error (normalized):   {score['l2_error_normalized']:.4e}")
        print(f"    L-inf error (normalized): {score['linf_error_normalized']:.4e}")
        print(f"    Mass drift:              {score['mass_drift']:.4e}")
        print(f"    Tail growth:             {score['tail_growth']:.4f}")
        print(f"    Summary:                 {score['summary']:.4e}")

    # --- Interface preservation comparison ---
    print("\n" + "=" * 60)
    print("Interface Preservation")
    print("=" * 60)
    for r in results:
        d0 = r['diags'][0]
        df = r['diags'][-1]
        print(f"\n  {r['label']}:")
        print(f"    Vertices:       {d0['n_verts']} -> {df['n_verts']}")
        print(f"    Interface edges: {d0['n_iface_edges']} -> {df['n_iface_edges']}")
        print(f"    Wall time:       {r['wall_time_s']:.1f}s")

    # --- Save results ---
    os.makedirs(_RESULTS, exist_ok=True)
    out_path = os.path.join(_RESULTS, 'adaptive_comparison.json')
    serializable = []
    for r in results:
        s = {
            'label': r['label'],
            'remesh_mode': r['remesh_mode'],
            'wall_time_s': r['wall_time_s'],
            'n_frames': len(r['diags']),
            'score': r['score'],
            'initial_verts': r['diags'][0]['n_verts'],
            'final_verts': r['diags'][-1]['n_verts'],
            'initial_iface': r['diags'][0]['n_iface_edges'],
            'final_iface': r['diags'][-1]['n_iface_edges'],
        }
        serializable.append(s)
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # --- Plot comparison ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(_FIG, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Oscillating Droplet: Delaunay vs Adaptive "
                     f"(l={l}, {regime}, {n_steps} steps)", fontsize=13)
        colors = ['tab:blue', 'tab:orange']

        for i, r in enumerate(results):
            t = np.array([d['t'] for d in r['diags']])
            R_max = np.array([d['R_max'] for d in r['diags']])
            KE = np.array([d['KE'] for d in r['diags']])
            mass = np.array([d['total_mass'] for d in r['diags']])
            n_ie = np.array([d['n_iface_edges'] for d in r['diags']])

            # R_max vs analytical
            ax = axes[0, 0]
            ax.plot(t * 1000, R_max * 1000, color=colors[i],
                    label=r['label'], lw=1.5)
            if i == 0:
                t_fine = np.linspace(0, t[-1], 500)
                R_env = max_radius_envelope(t_fine, R0, epsilon, omega, beta, l=l)
                ax.plot(t_fine * 1000, R_env * 1000, 'k--', lw=1, alpha=0.5,
                        label='Analytical envelope')

            # KE
            ax = axes[0, 1]
            ax.plot(t * 1000, KE, color=colors[i], label=r['label'], lw=1.5)
            ax.set_yscale('log')

            # Mass conservation
            ax = axes[1, 0]
            dM = np.abs(mass - mass[0]) / mass[0]
            ax.plot(t * 1000, dM, color=colors[i], label=r['label'], lw=1.5)

            # Interface edge count
            ax = axes[1, 1]
            ax.plot(t * 1000, n_ie, color=colors[i], label=r['label'], lw=1.5)

        axes[0, 0].set_xlabel('Time [ms]')
        axes[0, 0].set_ylabel('R_max [mm]')
        axes[0, 0].set_title('Maximum Interface Radius')
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].set_xlabel('Time [ms]')
        axes[0, 1].set_ylabel('KE [J]')
        axes[0, 1].set_title('Kinetic Energy')
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].set_xlabel('Time [ms]')
        axes[1, 0].set_ylabel('|dM/M0|')
        axes[1, 0].set_title('Mass Conservation')
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].set_xlabel('Time [ms]')
        axes[1, 1].set_ylabel('Cross-phase edges')
        axes[1, 1].set_title('Interface Edge Count')
        axes[1, 1].legend(fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = os.path.join(_FIG, 'adaptive_comparison.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved: {fig_path}")
    except ImportError:
        print("matplotlib not available; skipping plot")
    except Exception as e:
        print(f"Plotting error: {e}")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
