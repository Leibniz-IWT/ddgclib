#!/usr/bin/env python3
"""2D oscillating droplet: 3-way comparison of retopologization strategies.

Runs three simulations side by side and produces comparison plots:

    NO_REDIST : Full Delaunay retriangulation every step (baseline)
    REDIST    : Full Delaunay + pressure-preserving mass redistribution
    NO_RETOPO : No retriangulation (dual-only recompute on fixed connectivity)

Produces:
  fig/comparison_radius.png         — R_max(t) for all three cases
  fig/comparison_energy.png         — KE(t)
  fig/comparison_pressure_std.png   — pressure std dev over time
  fig/comparison_mass.png           — mass conservation comparison
  fig/oscillating_droplet_2D_redist.mp4     — animation (REDIST case)
  fig/oscillating_droplet_2D_no_retopo.mp4  — animation (NO_RETOPO case)

Usage
-----
    python cases_dynamic/oscillating_droplet/oscillating_droplet_2D_mass_redist.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from functools import partial

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
    compute_diagnostics,
)
from hyperct.ddg import compute_vd
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _retopologize_multiphase,
)
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results')


def dual_only_retopo_multiphase(HC, bV, dim, _mps=None):
    """Recompute duals on existing connectivity (no Delaunay).

    Keeps edges/triangles intact, just recomputes barycentric duals,
    dual volumes, and per-phase volume splits.  Interface identity
    is kept frozen from the initial mesh.
    """
    dV = HC.boundary()
    for v in HC.V:
        v.boundary = v in dV

    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)

    if _mps is not None:
        # Refresh per-phase volumes (but DO NOT re-identify interface)
        _mps.split_dual_volumes(HC, dim)
        _mps.compute_phase_pressures(HC)

    bV.clear()
    bV.update(dV)


def _run_simulation(label, retopo_mode):
    """Run a single simulation.

    Parameters
    ----------
    label : str
        Short label for console output.
    retopo_mode : {'no_redist', 'redist', 'no_retopo'}
        Retopologization strategy.
    """
    dim = 2

    HC, bV, mps, bc_set, dudt_fn, _, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
        )

    # Build retopo function based on mode
    if retopo_mode == 'no_redist':
        retopo_fn = partial(
            _retopologize_multiphase, mps=mps, redistribute_mass=False,
        )
    elif retopo_mode == 'redist':
        retopo_fn = partial(
            _retopologize_multiphase, mps=mps, redistribute_mass=True,
        )
    elif retopo_mode == 'no_retopo':
        retopo_fn = partial(dual_only_retopo_multiphase, _mps=mps)
    else:
        raise ValueError(f"Unknown retopo_mode: {retopo_mode}")

    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"\n[{label}] Mesh: {n_verts} vertices, {n_iface} interface")

    # CFL timestep
    c_s = np.sqrt(K_d / rho_d)
    dx_min = min(
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min**3 / gamma) if gamma > 0 else 1.0)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)
    t_end = min(t_end_2d, 5.0 / beta if beta > 0 else 0.01)
    n_steps = int(t_end / dt) + 1
    record_every = max(1, n_steps // 200)
    print(f"[{label}] dt={dt:.2e}, n_steps={n_steps}, t_end={t_end:.4f}")

    # Set up history (for animation of REDIST and NO_RETOPO cases)
    history = None
    if retopo_mode in ('redist', 'no_retopo'):
        snapshot_dir = os.path.join(_RESULTS, f'snapshots_{retopo_mode}')
        os.makedirs(snapshot_dir, exist_ok=True)
        history = StateHistory(
            fields=['u', 'p', 'phase', 'is_interface'],
            record_every=record_every,
            save_dir=snapshot_dir,
        )

    # Diagnostics arrays
    t_arr = [0.0]
    R_max_arr, KE_arr, mass_arr, p_std_arr = [], [], [], []
    diag0 = compute_diagnostics(HC, dim=dim)
    R_max_arr.append(diag0['R_max'])
    KE_arr.append(diag0['KE'])
    mass_arr.append(diag0['total_mass'])
    p_std_arr.append(_pressure_std(HC, bV))

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if history is not None:
            history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            diag = compute_diagnostics(HC_cb, dim=dim)
            t_arr.append(t)
            R_max_arr.append(diag['R_max'])
            KE_arr.append(diag['KE'])
            mass_arr.append(diag['total_mass'])
            p_std_arr.append(_pressure_std(HC_cb, bV_cb or set()))
            if step % (record_every * 20) == 0:
                print(f"  [{label}] t={t:.4e} | R_max={diag['R_max']:.6f} | "
                      f"mass={diag['total_mass']:.6f} | "
                      f"p_std={p_std_arr[-1]:.4e}")

    t0 = time.time()
    try:
        t_final = symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
            bc_set=bc_set, callback=callback, retopologize_fn=retopo_fn,
        )
    except Exception as e:
        print(f"[{label}] Simulation stopped: {e}")
        t_final = t_arr[-1] if t_arr else 0.0
    wall_time = time.time() - t0

    diag_final = compute_diagnostics(HC, dim=dim)
    t_arr.append(t_final)
    R_max_arr.append(diag_final['R_max'])
    KE_arr.append(diag_final['KE'])
    mass_arr.append(diag_final['total_mass'])
    p_std_arr.append(_pressure_std(HC, bV))

    print(f"[{label}] Done in {wall_time:.1f}s | "
          f"|dM/M0| = {abs(mass_arr[-1] - mass_arr[0]) / mass_arr[0]:.4e}")

    return {
        't': np.array(t_arr),
        'R_max': np.array(R_max_arr),
        'KE': np.array(KE_arr),
        'mass': np.array(mass_arr),
        'p_std': np.array(p_std_arr),
        'HC': HC,
        'bV': bV,
        'history': history,
        'wall_time': wall_time,
    }


def _pressure_std(HC, bV):
    """Standard deviation of pressure over interior vertices."""
    ps = []
    for v in HC.V:
        if v not in bV and getattr(v, 'dual_vol', 0.0) > 1e-30:
            p = v.p
            if np.ndim(p) > 0:
                p = float(p[0])
            ps.append(float(p))
    if not ps:
        return 0.0
    return float(np.std(ps))


def _style_for(mode):
    """Return (color, linestyle, label) for each mode."""
    return {
        'no_redist': ('#1f77b4', '-', 'Delaunay (no redist)'),
        'redist':    ('#d62728', '-', 'Delaunay + redist'),
        'no_retopo': ('#2ca02c', '-', 'No retopo (dual-only)'),
    }[mode]


def main():
    dim = 2
    print("=" * 60)
    print("2D Oscillating Droplet — 3-Way Retopo Comparison")
    print("=" * 60)

    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)
    omega_d = damped_frequency(omega, beta)
    regime = "underdamped" if omega_d > 0 else "overdamped"
    print(f"Mode l={l}: omega={omega:.2f}, beta={beta:.2f}, "
          f"omega_d={omega_d:.2f} ({regime})")

    results = {}

    print("\n--- Baseline: Full Delaunay, no mass redistribution ---")
    results['no_redist'] = _run_simulation("NO_REDIST", retopo_mode='no_redist')

    print("\n--- Full Delaunay WITH mass redistribution ---")
    results['redist'] = _run_simulation("REDIST", retopo_mode='redist')

    print("\n--- No retopologization (dual-only recompute) ---")
    results['no_retopo'] = _run_simulation("NO_RETOPO", retopo_mode='no_retopo')

    # -- Comparison plots --
    os.makedirs(_FIG, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Analytical reference
        t_max = max(r['t'][-1] for r in results.values())
        t_dense = np.linspace(0, t_max, 500)
        R_analytical = max_radius_envelope(t_dense, R0, epsilon, omega, beta)

        # 1. Radius comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.plot(res['t'], res['R_max'], color=c, linestyle=ls,
                    marker='.', markersize=2, alpha=0.8, label=lbl)
        ax.plot(t_dense, R_analytical, 'k--', linewidth=1.5,
                label='Analytical envelope')
        ax.axhline(R0, color='gray', linestyle=':', alpha=0.5, label=f'R0={R0}')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("R_max [m]")
        ax.set_title("Droplet Radius: Retopologization Comparison")
        ax.legend()
        fig.savefig(os.path.join(_FIG, 'comparison_radius.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # 2. Energy comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.semilogy(res['t'], np.maximum(res['KE'], 1e-30),
                        color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Kinetic Energy [J]")
        ax.set_title("Energy Decay: Retopologization Comparison")
        ax.legend()
        fig.savefig(os.path.join(_FIG, 'comparison_energy.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # 3. Pressure std dev comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.semilogy(res['t'], np.maximum(res['p_std'], 1e-30),
                        color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure Std Dev [Pa]")
        ax.set_title("Pressure Stability: Retopologization Comparison")
        ax.legend()
        fig.savefig(os.path.join(_FIG, 'comparison_pressure_std.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # 4. Mass conservation comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            M0 = res['mass'][0]
            ax.plot(res['t'], np.abs(res['mass'] - M0) / M0,
                    color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("|dM / M0|")
        ax.set_title("Mass Conservation: Retopologization Comparison")
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(os.path.join(_FIG, 'comparison_mass.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        print(f"\nComparison plots saved to {_FIG}/")

    except ImportError:
        print("matplotlib not available — skipping plots")

    # -- Animations for REDIST and NO_RETOPO --
    for mode in ('redist', 'no_retopo'):
        history = results[mode].get('history')
        if history is None or history.n_snapshots <= 1:
            continue
        try:
            HC_r = results[mode]['HC']
            bV_r = results[mode]['bV']
            suffix = 'redist' if mode == 'redist' else 'no_retopo'
            anim = dynamic_plot_fluid(
                history, HC_r, bV=bV_r,
                save_path=os.path.join(
                    _FIG, f'oscillating_droplet_2D_{suffix}.mp4'
                ),
                fps=20, dpi=100,
                xlim=(-2.5 * R0, 2.5 * R0),
                ylim=(-2.5 * R0, 2.5 * R0),
                phase_field='phase',
                interface_field='is_interface',
                reference_R=R0,
            )
            print(f"Animation saved: fig/oscillating_droplet_2D_{suffix}.mp4")
        except Exception as e:
            print(f"Animation failed for {mode}: {e}")

    # -- Summary table --
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Mode':<20} {'Wall [s]':>10} {'Final R_max':>14} "
          f"{'|dM/M0|':>12} {'Final p_std':>14}")
    print("-" * 72)
    for mode, res in results.items():
        _, _, lbl = _style_for(mode)
        M0 = res['mass'][0]
        dM = abs(res['mass'][-1] - M0) / M0
        print(f"{lbl:<20} {res['wall_time']:>10.1f} "
              f"{res['R_max'][-1]:>14.6f} {dM:>12.2e} "
              f"{res['p_std'][-1]:>14.4e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
