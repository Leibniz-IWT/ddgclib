#!/usr/bin/env python3
"""3D Capillary Rise — Dynamic simulation with data saving.

Builds a 3D cylindrical tube via cylinder_volume(), runs symplectic Euler,
and saves StateHistory + final state for post-processing.

Usage
-----
    python cases_dynamic/capillary_rise/capillary_rise_3D.py
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
from cases_dynamic.capillary_rise.src._plot_helpers import compute_diagnostics

from ddgclib.data import StateHistory, save_state
from ddgclib.visualization.unified import plot_primal
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _recompute_duals, _interior_verts, _move,
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


def main(fluid_name="water", R_mm=0.5, n_refine=2):
    dim = 3
    fp = FLUIDS[fluid_name]
    r = R_mm * 1e-3
    gravity_axis = 2

    h_j = jurin_height(r, fp['gamma'], fp['theta_s_deg'], fp['rho'], g, dim=dim)
    P_cap = capillary_pressure(r, fp['gamma'], fp['theta_s_deg'], dim=dim)

    print("=" * 60)
    print(f"3D Capillary Rise — {fluid_name}, R = {R_mm} mm")
    print("=" * 60)
    print(f"  Jurin height (3D) = {h_j*100:.3f} cm, P_cap = {P_cap:.1f} Pa")

    # ── Setup ────────────────────────────────────────────────────────
    print("\nBuilding mesh...")
    HC, bV, bc_set, dudt_fn, free_surf, params = setup_capillary_rise(
        dim=dim, r=r, gamma=fp['gamma'], theta_deg=fp['theta_s_deg'],
        mu=fp['mu'], rho=fp['rho'], g=g, n_refine=n_refine,
    )

    h_init = params['h_init']
    c0 = params['c0']
    set_h = params['set_h']

    n_verts = sum(1 for _ in HC.V)
    print(f"  Mesh: {n_verts} verts, h_init={h_init*100:.3f} cm")
    print(f"  c0={c0:.1f}, mu_art={params['mu_art']:.4f}, v_rise={params['v_rise']*100:.2f} cm/s")

    # ── Time stepping ────────────────────────────────────────────────
    dt = 0.1 * params['dx_mean'] / c0
    n_steps = 2000  # fewer for 3D
    record_every = max(1, n_steps // 50)

    print(f"\n  dt={dt:.2e}, n_steps={n_steps}, record_every={record_every}")

    history = StateHistory(fields=['u', 'p'], record_every=record_every)
    t_arr, h_arr, KE_arr = [0.0], [], []
    diag0 = compute_diagnostics(HC, dim, gravity_axis, free_surf)
    h_arr.append(diag0['h_mean'])
    KE_arr.append(diag0['KE'])

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            diag = compute_diagnostics(HC_cb, dim, gravity_axis, free_surf)
            t_arr.append(t)
            h_arr.append(diag['h_mean'])
            KE_arr.append(diag['KE'])
            set_h(diag['h_mean'])
        if step % 500 == 0 and step > 0:
            diag = compute_diagnostics(HC_cb, dim, gravity_axis, free_surf)
            u_max = max((np.linalg.norm(v.u[:dim]) for v in HC_cb.V), default=0.0)
            print(f"  step {step}: t={t:.4e}  h={diag['h_mean']*100:.4f} cm  |u|={u_max:.3e}")

    # ── Manual CFL loop (avoids auto-retopologize on curved cylinder) ─
    # Recompute duals only every DUAL_EVERY steps (expensive in 3D).
    print("\nRunning simulation...")
    CFL = 0.25
    DUAL_EVERY = 50  # recompute duals every 50 steps
    dx_min_init = params['dx_mean'] * 0.5
    t, step = 0.0, 0
    t_end = dt * n_steps

    # Initial dual computation
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=dim)

    while t < t_end:
        # Recompute duals periodically (expensive in 3D)
        if step > 0 and step % DUAL_EVERY == 0:
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
        dt_step = min(CFL * dx_min / (c0 + u_max), t_end - t)

        acc = {v: dudt_fn(v) for v in iv}
        for v in iv:
            v.u[:dim] += dt_step * acc[v][:dim]
            _move(v, v.x_a[:dim] + dt_step * v.u[:dim], HC, bV)

        if bc_set:
            bc_set.apply_all(HC, bV, dt_step)

        t += dt_step
        step += 1

        callback(step, t, HC, bV)

        if step % 100 == 0:
            diag = compute_diagnostics(HC, dim, gravity_axis, free_surf)
            print(f"  step {step}, t={t:.4e}, h={diag['h_mean']*100:.4f} cm, u={u_max:.3e}")

        if u_max > 10 * c0:
            print(f"  ABORT: u_max={u_max:.2e} >> c0={c0:.1f}")
            break

    t_final = t
    t_arr = np.array(t_arr)
    h_arr = np.array(h_arr)

    print(f"\n  Done: {step} steps, t={t_final:.4e} s")
    print(f"  Final h = {h_arr[-1]*100:.4f} cm (Jurin = {h_j*100:.3f} cm)")

    # ── Save ─────────────────────────────────────────────────────────
    save_state(HC, bV, t=t_final, fields=['u', 'p', 'm'],
               path=os.path.join(_RESULTS, 'caprise_3d_final.json'),
               extra_meta={'case': 'capillary_rise_3d', 'fluid': fluid_name,
                           'R_mm': R_mm})
    with open(os.path.join(_RESULTS, 'caprise_3d_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    np.savez(os.path.join(_RESULTS, 'caprise_3d_diagnostics.npz'),
             t=t_arr, h=h_arr, KE=np.array(KE_arr))
    print(f"  Data saved to {_RESULTS}/")

    # ── Plots ────────────────────────────────────────────────────────
    # Height vs time
    t_wash, h_wash = washburn_solve(
        (0.0, max(t_arr[-1], 0.01)), 1e-6, r,
        fp['gamma'], fp['theta_s_deg'], fp['mu'], fp['rho'], g, dim=dim)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t_arr, h_arr*100, 'b-', lw=2, label='DDG')
    ax1.plot(t_wash, h_wash*100, 'r--', lw=1.5, label='Washburn')
    ax1.axhline(h_j*100, color='k', ls=':', lw=1, alpha=0.5,
                label=f'Jurin={h_j*100:.2f} cm')
    ax1.set_xlabel('Time [s]'); ax1.set_ylabel('h [cm]')
    ax1.set_title(f'3D Capillary Rise — {fluid_name}, R={R_mm} mm')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    _savefig(fig1, 'caprise_3d_height')
    plt.close(fig1)

    # 3D pressure plot
    fig2, ax2 = plot_primal(HC, bV=bV, scalar_field='p', dim=dim,
                            title=f'3D Final State t={t_final:.4e} s',
                            vertex_size=30, cmap='coolwarm')
    fig2.tight_layout()
    _savefig(fig2, 'caprise_3d_final_pressure')
    plt.close(fig2)

    print("\nDone.")
    return t_arr, h_arr, params


if __name__ == '__main__':
    main()
