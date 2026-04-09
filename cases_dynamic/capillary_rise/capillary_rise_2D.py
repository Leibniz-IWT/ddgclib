#!/usr/bin/env python3
"""2D Capillary Rise — Dynamic simulation with movie output.

Builds a 2D slit channel, sets EOS-consistent compressible hydrostatic
IC, then runs the Lagrangian symplectic Euler integrator with a capillary
body force.  Generates pressure + velocity movie via StateHistory +
dynamic_plot_fluid (same pattern as Cube2droplet cases).

Usage
-----
    python cases_dynamic/capillary_rise/capillary_rise_2D.py
"""
import os
import sys
import pickle

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.capillary_rise.src._params import (
    FLUIDS, g, jurin_height, capillary_pressure,
)
from cases_dynamic.capillary_rise.src._analytical import washburn_solve
from cases_dynamic.capillary_rise.src._setup import setup_capillary_rise
from cases_dynamic.capillary_rise.src._data import load_sample_data
from cases_dynamic.capillary_rise.src._plot_helpers import compute_diagnostics

from ddgclib.data import StateHistory, save_state
from ddgclib.visualization.unified import plot_primal
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _recompute_duals, _interior_verts, _move,
)
from ddgclib.analytical._integrated_comparison import (
    integrated_pressure_error,
    integrated_l2_norm,
    compare_stress_force,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RESULTS, exist_ok=True)


def _savefig(fig, bn):
    fig.savefig(os.path.join(_FIG, f'{bn}.pdf'), dpi=150)
    fig.savefig(os.path.join(_FIG, f'{bn}.png'), dpi=150)
    print(f"  -> fig/{bn}.pdf, fig/{bn}.png")


def main(fluid_name="water", R_mm=0.5, n_refine=3):
    dim = 2
    fp = FLUIDS[fluid_name]
    r = R_mm * 1e-3
    gravity_axis = 1

    h_j = jurin_height(r, fp['gamma'], fp['theta_s_deg'], fp['rho'], g, dim=dim)
    P_cap = capillary_pressure(r, fp['gamma'], fp['theta_s_deg'], dim=dim)

    print("=" * 60)
    print(f"2D Capillary Rise — {fluid_name}, R = {R_mm} mm")
    print("=" * 60)
    print(f"  Jurin height (2D) = {h_j*100:.3f} cm")
    print(f"  P_cap (2D) = {P_cap:.1f} Pa")

    # ── Setup ────────────────────────────────────────────────────────────
    print("\nBuilding mesh...")
    HC, bV, bc_set, dudt_fn, free_surf, params = setup_capillary_rise(
        dim=dim, r=r, gamma=fp['gamma'], theta_deg=fp['theta_s_deg'],
        mu=fp['mu'], rho=fp['rho'], g=g, n_refine=n_refine,
    )

    h_init = params['h_init']
    c0 = params['c0']
    eos = params['eos']
    set_h = params['set_h']

    n_verts = sum(1 for _ in HC.V)
    print(f"  Mesh: {n_verts} vertices")
    print(f"  h_init = {h_init*100:.3f} cm  ({h_init/h_j*100:.0f}% of Jurin)")
    print(f"  c0 = {c0:.1f} m/s, mu_art = {params['mu_art']:.4f}")
    print(f"  v_rise (Washburn) = {params['v_rise']*100:.2f} cm/s")

    # ── Validate initial pressure (integrated comparison) ────────────
    interior = [v for v in HC.V if v not in bV]
    P_atm = params['P_atm']
    K_eos = params['K_eos']
    alpha_eos = fp['rho'] * g / K_eos

    def P_analytical(x):
        """Compressible hydrostatic: P = P_atm + K(exp(alpha*depth) - 1)."""
        depth = h_init - x[gravity_axis]
        return P_atm + K_eos * (np.exp(alpha_eos * max(depth, 0.0)) - 1.0)

    int_errs = integrated_pressure_error(
        HC, interior, P_analytical=P_analytical, dim=dim)
    int_l2 = integrated_l2_norm(
        HC, interior, P_analytical=P_analytical, dim=dim)
    print(f"\n  Initial integrated pressure error:")
    print(f"    max|p*V - ∫P dV| = {max(int_errs):.4e}")
    print(f"    L2 norm = {int_l2:.4e} Pa")

    # ── Initial pressure profile plot (integrated) ───────────────────
    fig_p, ax_p = plt.subplots(figsize=(6, 6))
    y_all = np.array([v.x_a[gravity_axis] for v in HC.V])
    p_all = np.array([v.p for v in HC.V])
    # Analytical reference (for visual guide only; v.p is volume-averaged)
    y_ana = np.linspace(0, h_init, 200)
    P_comp = np.array([P_analytical(np.array([0, y])) for y in y_ana])
    ax_p.plot(P_comp, y_ana * 100, 'g--', lw=1.5, label='Analytical (compressible)')
    ax_p.plot(p_all, y_all * 100, 'bo', ms=4, alpha=0.6,
              label='DDG (vol-averaged $v.p$)')
    ax_p.axhline(h_init * 100, color='orange', ls=':', lw=1,
                 label=f'Free surface h={h_init*100:.2f} cm')
    ax_p.set_xlabel('P [Pa]')
    ax_p.set_ylabel('y [cm]')
    ax_p.set_title(f'Initial Pressure (integrated err L2={int_l2:.2e} Pa)')
    ax_p.legend(fontsize=8)
    ax_p.grid(True, alpha=0.3)
    fig_p.tight_layout()
    _savefig(fig_p, 'caprise_2d_pressure_profile')
    plt.close(fig_p)

    # ── Boundary groups plot ─────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(6, 8))
    for v in HC.V:
        for nb in v.nn:
            ax_b.plot([v.x_a[0]*1e3, nb.x_a[0]*1e3],
                      [v.x_a[1]*100, nb.x_a[1]*100],
                      'k-', lw=0.3, alpha=0.3)
    for verts, color, label in [
        (bV, 'red', 'Frozen (walls)'),
        (free_surf, 'green', 'Free surface'),
    ]:
        if verts:
            coords = np.array([v.x_a[:dim] for v in verts])
            ax_b.scatter(coords[:, 0]*1e3, coords[:, 1]*100, c=color,
                         s=40, zorder=5, label=label)
    others = [v for v in HC.V if v not in bV and v not in free_surf]
    if others:
        coords = np.array([v.x_a[:dim] for v in others])
        ax_b.scatter(coords[:, 0]*1e3, coords[:, 1]*100, c='gray',
                     s=15, zorder=4, label='Interior/inlet')
    ax_b.set_xlabel('x [mm]'); ax_b.set_ylabel('y [cm]')
    ax_b.set_title(f'Mesh & Boundaries — ref={n_refine}')
    ax_b.legend(fontsize=8); ax_b.grid(True, alpha=0.3)
    fig_b.tight_layout()
    _savefig(fig_b, 'caprise_2d_mesh_boundaries')
    plt.close(fig_b)

    # ── Time stepping (manual adaptive CFL, no retopologize for 2D) ──
    CFL = 0.25
    t_ac = h_init / c0
    n_trav = 300
    t_end = n_trav * t_ac
    dx_min_init = params['dx_mean'] * 0.5
    rec = max(1, int(0.5 * t_ac / (CFL * dx_min_init / c0)))

    print(f"\n  CFL={CFL}, t_ac={t_ac:.6f} s, {n_trav} traversals -> t_end={t_end:.4f} s")
    print(f"  Recording every {rec} steps")

    # StateHistory for movie
    history = StateHistory(fields=['u', 'p'], record_every=1)

    # Diagnostics
    t_arr, h_arr, KE_arr, mass_arr = [0.0], [], [], []
    diag0 = compute_diagnostics(HC, dim, gravity_axis, free_surf)
    h_arr.append(diag0['h_mean'])
    KE_arr.append(diag0['KE'])
    mass_arr.append(diag0['total_mass'])
    history.append(0.0, HC)

    # ── Manual CFL loop (following Hydrostatic_2D pattern) ───────────
    print("\nRunning simulation...")
    t, step = 0.0, 0

    while t < t_end:
        _recompute_duals(HC)
        cache_dual_volumes(HC, dim=dim)
        iv = _interior_verts(HC, bV)

        u_max = max((np.linalg.norm(v.u[:dim]) for v in iv), default=0.0)
        dx_min = min(
            (np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
             for v in iv for nb in v.nn
             if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 0),
            default=dx_min_init,
        )
        dt = min(CFL * dx_min / (c0 + u_max), t_end - t)

        acc = {v: dudt_fn(v) for v in iv}
        for v in iv:
            v.u[:dim] += dt * acc[v][:dim]
            _move(v, v.x_a[:dim] + dt * v.u[:dim], HC, bV)

        if bc_set:
            bc_set.apply_all(HC, bV, dt)

        t += dt
        step += 1

        # Record diagnostics + history frames
        if step % rec == 0 or t >= t_end:
            diag = compute_diagnostics(HC, dim, gravity_axis, free_surf)
            t_arr.append(t)
            h_arr.append(diag['h_mean'])
            KE_arr.append(diag['KE'])
            mass_arr.append(diag['total_mass'])
            set_h(diag['h_mean'])
            history.append(t, HC)

        if step % 500 == 0:
            diag = compute_diagnostics(HC, dim, gravity_axis, free_surf)
            print(f"  step {step}, t={t:.6f} s, h={diag['h_mean']*100:.4f} cm, "
                  f"u_max={u_max:.4f}")

        if u_max > 10 * c0:
            print(f"  ABORT: u_max={u_max:.2e} >> c0={c0:.1f}")
            break

    t_final = t
    t_arr = np.array(t_arr)
    h_arr = np.array(h_arr)
    KE_arr = np.array(KE_arr)
    mass_arr = np.array(mass_arr)

    # ── Results summary ──────────────────────────────────────────────
    print(f"\n  Done: {step} steps, t = {t_final:.6f} s")
    print(f"  Final h = {h_arr[-1]*100:.4f} cm  (Jurin = {h_j*100:.3f} cm)")
    if mass_arr[0] > 0:
        print(f"  Mass conservation: |dM/M0| = "
              f"{abs(mass_arr[-1]-mass_arr[0])/mass_arr[0]:.4e}")

    # ── Save state ───────────────────────────────────────────────────
    save_state(HC, bV, t=t_final, fields=['u', 'p', 'm'],
               path=os.path.join(_RESULTS, 'caprise_2d_final.json'),
               extra_meta={'case': 'capillary_rise_2d', 'fluid': fluid_name,
                           'R_mm': R_mm, 'h_jurin_cm': h_j*100})
    history_path = os.path.join(_RESULTS, 'caprise_2d_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"  State saved to {_RESULTS}/")

    # ── Washburn reference ───────────────────────────────────────────
    t_wash, h_wash = washburn_solve(
        (0.0, max(t_arr[-1], 0.01)), 1e-6, r,
        fp['gamma'], fp['theta_s_deg'], fp['mu'], fp['rho'], g, dim=dim,
    )

    # ── PLOTS ────────────────────────────────────────────────────────
    # 1. Height vs time
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t_arr, np.array(h_arr)*100, 'b-', lw=2, label='DDG simulation')
    ax1.plot(t_wash, h_wash*100, 'r--', lw=1.5, label='Washburn ODE')
    ax1.axhline(h_j*100, color='k', ls=':', lw=1, alpha=0.5,
                label=f'Jurin = {h_j*100:.2f} cm')
    ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Meniscus height [cm]')
    ax1.set_title(f'2D Capillary Rise — {fluid_name}, R={R_mm} mm')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    _savefig(fig1, 'caprise_2d_height')
    plt.close(fig1)

    # 2. Diagnostics (KE + mass)
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4))
    if np.any(KE_arr > 0):
        ax2a.semilogy(t_arr, KE_arr, 'b-', lw=1)
    ax2a.set_xlabel('t [s]'); ax2a.set_ylabel('KE [J]')
    ax2a.set_title('Kinetic Energy'); ax2a.grid(True, alpha=0.3)
    ax2b.plot(t_arr, mass_arr, 'k-', lw=1)
    ax2b.set_xlabel('t [s]'); ax2b.set_ylabel('Total mass [kg]')
    ax2b.set_title('Mass Conservation'); ax2b.grid(True, alpha=0.3)
    fig2.tight_layout()
    _savefig(fig2, 'caprise_2d_diagnostics')
    plt.close(fig2)

    # 3. Final integrated pressure error
    interior_final = [v for v in HC.V if v not in bV]
    if interior_final:
        try:
            int_errs_f = integrated_pressure_error(
                HC, interior_final, P_analytical=P_analytical, dim=dim)
            force_diag = compare_stress_force(HC, interior_final, dim=dim,
                                              mu=params['mu_art'])
            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
            y_int = [v.x_a[gravity_axis]*100 for v in interior_final]
            ax3a.scatter(y_int, int_errs_f, s=10, alpha=0.5)
            ax3a.set_xlabel('y [cm]')
            ax3a.set_ylabel('$|p_i V_i - \\int P \\, dV|$')
            ax3a.set_title('Integrated Pressure Error')
            ax3a.grid(True, alpha=0.3)
            ax3b.scatter(y_int, force_diag['F_norms'], s=10, alpha=0.5)
            ax3b.set_xlabel('y [cm]')
            ax3b.set_ylabel('$||F_{stress}||$')
            ax3b.set_title('Force Balance')
            ax3b.grid(True, alpha=0.3)
            fig3.suptitle('Final Integrated Diagnostics')
            fig3.tight_layout()
            _savefig(fig3, 'caprise_2d_integrated')
            plt.close(fig3)
        except Exception as e:
            print(f"  (integrated diagnostics: {e})")

    # 4. Movie via StateHistory + dynamic_plot_fluid
    if history.n_snapshots > 1:
        print(f"\nGenerating animation ({history.n_snapshots} frames)...")
        try:
            from ddgclib.visualization import dynamic_plot_fluid
            anim = dynamic_plot_fluid(
                history, HC, bV,
                scalar_field='p', vector_field='u',
                save_path=os.path.join(_FIG, 'caprise_2d.gif'),
                fps=15, writer='pillow',
            )
            print(f"  -> fig/caprise_2d.gif")
        except Exception as e:
            print(f"  Animation failed: {e}")
    plt.close('all')

    print(f"\nAll output in {_FIG}/ and {_RESULTS}/")
    print("Done.")
    return t_arr, h_arr, params


if __name__ == '__main__':
    main()
