#!/usr/bin/env python3
"""3D Dynamic Capillary Rise — data-driven contact-line forcing (dynCA).

3D companion of ``capillary_rise_2D_dynCA.py``: the actual experimental
tube (radius R), driving pressure P_cap(t) = 2 gamma cos(theta_exp(t))/R
back-computed from the measured dynamic contact angle.

DIAGNOSTIC CASE.  3D still runs the legacy dual-volume path (the exact
simplex-based 3D switch was built but backed out in the 2026-07-02
campaign — known 1-4 % interior / ~20 % boundary undercount), and there
is no 3D adaptive remesh, so the reservoir band stretches without
resolution recovery.  Keep the window short (band stretch <~ 2.5x).
Output is intended for solver debugging as much as physics.

Usage
-----
    python cases_dynamic/capillary_rise/capillary_rise_3D_dynCA.py \
        [--fluid water] [--R-mm 0.5] [--t0 0.03] [--t-end 0.06] [--smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.capillary_rise.src._params import g as G
from cases_dynamic.capillary_rise.src._dynamic_ca import (
    ExperimentalDrive, rise_ode_solve,
)
from cases_dynamic.capillary_rise.src._setup_dynca import (
    build_tube_3d, tag_groups_3d, make_eos, band_mass_reset,
    boundary_mass_reset, apply_ics, make_dudt_dynca,
)

from ddgclib.data import StateHistory, save_state
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _recompute_duals, _move,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RESULTS, exist_ok=True)


def _savefig(fig, bn):
    fig.savefig(os.path.join(_FIG, f'{bn}.png'), dpi=150)
    fig.savefig(os.path.join(_FIG, f'{bn}.pdf'))
    print(f"  -> fig/{bn}.png")


def main(fluid='water', R_mm=0.5, t0=0.03, t_end=0.06, n_rings=3, cfl=0.4,
         c0_factor=10.0, band_stretch_max=2.5, record_dt=None, tag='') -> dict:
    dim, gravity_axis = 3, 2
    drive = ExperimentalDrive(fluid, R_mm)
    rho, mu, gamma = drive.rho, drive.mu, drive.gamma
    R = drive.R
    if t_end is None:
        t_end = drive.t_max

    h0 = float(drive.h_exp(t0))
    hdot0 = float(drive.hdot_exp(t0))
    h_jurin = 2.0 * gamma * np.cos(np.deg2rad(drive.theta_s_deg)) / (rho * G * R)

    tq = np.linspace(drive.t_min, drive.t_max, 500)
    u_ref = max(float(np.max(np.abs(drive.hdot_exp(tq)))), np.sqrt(G * h_jurin))
    c0 = c0_factor * u_ref
    eos = make_eos(rho, c0)

    # Band deep enough that the expected elongation keeps stretch bounded
    dh_expected = float(drive.h_exp(t_end)) - h0
    H_res = max(2.0 * R, dh_expected / max(band_stretch_max - 1.0, 0.5))

    label = f"{fluid}_R{R_mm:g}mm{tag}"
    print("=" * 64)
    print(f"3D dynCA capillary rise — {fluid}, R={R_mm} mm (DIAGNOSTIC)")
    print("=" * 64)
    print(f"  window: t_exp in [{t0}, {t_end}] s, h0={h0*100:.3f} cm, "
          f"hdot0={hdot0*100:.2f} cm/s, dh_expected={dh_expected*100:.3f} cm")
    print(f"  h_jurin = {h_jurin*100:.3f} cm, c0 = {c0:.2f} m/s, "
          f"band depth = {H_res*1e3:.2f} mm")

    HC, meta = build_tube_3d(R, -H_res, h0, n_rings)
    dx0 = meta['dx']
    bset = meta['bset']            # topological boundary (Delaunay hull)
    wall_tol = 0.25 * R / n_rings
    n_verts0 = sum(1 for _ in HC.V)
    print(f"  mesh: {n_verts0} verts ({meta['n_layer']}/layer x {meta['nz']+1} "
          f"layers), dx = {dx0*1e6:.0f} um")

    groups = tag_groups_3d(HC, R, -H_res, wall_tol)
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=dim)

    apply_ics(HC, dim, rho, eos, G, R, hdot0, -H_res, wall_tol)
    band_mass_reset(HC, eos, rho, G, gravity_axis)

    state = {'t': t0, 'h': h0, 'p_cap': 0.0, 'F_surf': {}}
    dudt_fn, update_state = make_dudt_dynca(
        dim, mu, HC, eos, rho, G, drive.p_cap_tube, state,
        drive_mode='body')
    update_state(t0, h0)
    print(f"  P_cap(t0) = {state['p_cap']:.1f} Pa (body drive)")

    snap_dir = os.path.join(_RESULTS, 'snapshots', f'dynca_3d_{label}')
    history = StateHistory(fields=['u', 'p'], record_every=1, save_dir=snap_dir)
    if record_dt is None:
        record_dt = (t_end - t0) / 100.0

    t_arr = [t0]
    h_arr = [h0]
    KE_arr = [0.5 * sum(v.m * float(v.u @ v.u) for v in HC.V)]
    mass_arr = [sum(v.m for v in HC.V)]
    injected = 0.0
    inj_arr = [0.0]
    pcap_arr = [state['p_cap']]
    history.append(t0, HC)

    t_sim, step = 0.0, 0
    T_win = t_end - t0
    next_rec = record_dt
    wall_t0 = time.time()
    aborted = None

    from ddgclib.multiphase import mass_conserving_merge
    A_tube = np.pi * R * R

    def min_edge():
        return min((float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
                    for v in HC.V for nb in v.nn
                    if float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])) > 0),
                   default=dx0)

    while t_sim < T_win:
        groups = tag_groups_3d(HC, R, -H_res, wall_tol)
        contact = groups['contact']
        surf = groups['surface']
        frozen = groups['frozen'] - surf
        # dual computation needs the TOPOLOGICAL boundary, not the
        # positional group tags
        for v in HC.V:
            v.boundary = v in bset

        _recompute_duals(HC)
        cache_dual_volumes(HC, dim=dim)

        # no 3D remesh: merge collapsing pairs, then re-Delaunay (the
        # domain is convex, so a global retriangulation is safe)
        dx_min = min_edge()
        if step % 10 == 0 or dx_min < 0.12 * dx0:
            if mass_conserving_merge(HC, cdist=0.15 * dx0):
                from hyperct.ddg import (
                    connect_and_cache_simplices, boundary_from_simplices,
                )
                verts_all = list(HC.V)
                for v in verts_all:
                    for nb in list(v.nn):
                        v.disconnect(nb)
                coords = np.array([v.x_a for v in verts_all])
                connect_and_cache_simplices(HC, verts_all, 3, coords=coords)
                bset = set(boundary_from_simplices(HC, 3))
                for v in HC.V:
                    v.boundary = v in bset
                _recompute_duals(HC)
                cache_dual_volumes(HC, dim=dim)
                groups = tag_groups_3d(HC, R, -H_res, wall_tol)
                contact = groups['contact']
                surf = groups['surface']
                frozen = groups['frozen'] - surf
                for v in HC.V:
                    v.boundary = v in bset
            dx_min = min_edge()
            if dx_min < 0.05 * dx0:
                aborted = f"degenerate edge {dx_min:.3e} m survived merge"
                print(f"  ABORT: {aborted}")
                break

        injected += band_mass_reset(HC, eos, rho, G, gravity_axis)
        injected += boundary_mass_reset(HC, frozen | contact, eos, rho, G,
                                        gravity_axis)
        for v in frozen:                      # strict no-slip on the film
            v.u[:dim] = 0.0
        # one-sided free-surface BC: remove compression excess only
        for v in surf:
            vol = getattr(v, 'dual_vol', 0.0)
            if vol > 0.0 and v not in frozen and v.m > rho * vol:
                m_new = rho * vol
                injected += m_new - v.m
                v.m = m_new

        iv = [v for v in HC.V if v not in frozen]
        u_max = max((float(np.linalg.norm(v.u[:dim])) for v in iv), default=0.0)
        dt = min(cfl * dx_min / (c0 + u_max), T_win - t_sim)

        # column height from mass above datum (robust to surface churn)
        m_above = sum(v.m for v in HC.V if v.x_a[2] > 0.0)
        h_now = m_above / (rho * A_tube)
        update_state(t0 + t_sim, h_now)

        acc = {v: dudt_fn(v) for v in iv}
        for v in iv:
            v.u[:dim] = v.u[:dim] + dt * acc[v][:dim]
            u_norm = float(np.linalg.norm(v.u[:dim]))
            if u_norm > 2.0 * c0:
                v.u[:dim] *= 2.0 * c0 / u_norm
            if v in contact:               # contact ring slides along wall
                v.u[0] = 0.0
                v.u[1] = 0.0
            new_x = v.x_a[:dim] + dt * v.u[:dim]
            rr = float(np.hypot(new_x[0], new_x[1]))
            if rr > R:                     # wall is impenetrable
                new_x[0] *= R / rr
                new_x[1] *= R / rr
                ur = (v.u[0] * new_x[0] + v.u[1] * new_x[1]) / R
                v.u[0] -= ur * new_x[0] / R
                v.u[1] -= ur * new_x[1] / R
            _move(v, new_x, HC, frozen)

        t_sim += dt
        step += 1

        if t_sim >= next_rec or t_sim >= T_win:
            next_rec += record_dt
            t_arr.append(t0 + t_sim)
            h_arr.append(h_now)
            KE_arr.append(0.5 * sum(v.m * float(v.u @ v.u) for v in HC.V))
            mass_arr.append(sum(v.m for v in HC.V))
            inj_arr.append(injected)
            pcap_arr.append(state['p_cap'])
            history.append(t0 + t_sim, HC)

        if step % 200 == 0:
            rate = step / max(time.time() - wall_t0, 1e-9)
            eta = (T_win - t_sim) / max(dt, 1e-12) / max(rate, 1e-9) / 60.0
            print(f"  step {step}: t_exp={t0+t_sim:.4f}s h={h_arr[-1]*100:.3f}cm "
                  f"h_exp={float(drive.h_exp(t0+t_sim))*100:.3f}cm "
                  f"u_max={u_max:.3f} [{rate:.1f} st/s, ETA {eta:.0f} min]")

        if u_max > 5.0 * c0 or not np.isfinite(u_max):
            aborted = f"u_max={u_max:.3e} exceeded 5*c0"
            print(f"  ABORT: {aborted}")
            break

    wall_min = (time.time() - wall_t0) / 60.0
    t_arr, h_arr = np.array(t_arr), np.array(h_arr)
    KE_arr, mass_arr = np.array(KE_arr), np.array(mass_arr)
    inj_arr, pcap_arr = np.array(inj_arr), np.array(pcap_arr)

    h_exp_i = drive.h_exp(t_arr)
    l2 = float(np.sqrt(np.mean((h_arr - h_exp_i) ** 2)) / max(np.ptp(h_exp_i), 1e-12))
    err_final = float((h_arr[-1] - h_exp_i[-1]) / max(h_exp_i[-1], 1e-12))
    mass_err = float(abs(mass_arr[-1] - mass_arr[0] - inj_arr[-1])
                     / max(mass_arr[0], 1e-30))
    print(f"\n  done: {step} steps in {wall_min:.1f} min")
    print(f"  h_final sim/exp = {h_arr[-1]*100:.3f} / {h_exp_i[-1]*100:.3f} cm "
          f"({100*err_final:+.1f} %)")
    print(f"  normalized L2(h_sim - h_exp) = {l2:.4f}")
    print(f"  mass closure err = {mass_err:.3e}")

    t_ode, h_dyn, _ = rise_ode_solve(drive, t0, t_end, theta_mode='dynamic')
    _, h_sta, _ = rise_ode_solve(drive, t0, t_end, theta_mode='static')

    out = {
        'case': 'capillary_rise_3d_dynCA', 'fluid': fluid, 'R_mm': R_mm,
        't0': t0, 't_end': t_end, 'n_rings': n_rings, 'cfl': cfl, 'c0': c0,
        'steps': step, 'wall_minutes': wall_min, 'aborted': aborted,
        'l2_h_norm': l2, 'err_final_rel': err_final, 'mass_err': mass_err,
        'h_jurin_m': h_jurin, 'H_res': H_res,
        't': t_arr.tolist(), 'h_sim': h_arr.tolist(),
        'h_exp': np.asarray(h_exp_i).tolist(),
        'KE': KE_arr.tolist(), 'mass': mass_arr.tolist(),
        'injected': inj_arr.tolist(), 'p_cap': pcap_arr.tolist(),
        't_ode': t_ode.tolist(), 'h_ode_dynCA': h_dyn.tolist(),
        'h_ode_static': h_sta.tolist(),
    }
    res_path = os.path.join(_RESULTS, f'dynca_3d_{label}.json')
    with open(res_path, 'w') as f:
        json.dump(out, f)
    print(f"  results -> {res_path}")
    save_state(HC, groups['frozen'], t=t_arr[-1], fields=['u', 'p', 'm'],
               path=os.path.join(_RESULTS, f'dynca_3d_{label}_final.json'),
               extra_meta={'case': 'capillary_rise_3d_dynCA'})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(drive.t_data, drive.h_data * 100, 'ko', ms=4, mfc='none',
            label='experiment')
    ax.plot(t_arr, h_arr * 100, 'b-', lw=2, label='DDG 3D sim (exp CA(t))')
    ax.plot(t_ode, h_dyn * 100, 'g--', lw=1.5, label='ODE, exp CA(t)')
    ax.plot(t_ode, h_sta * 100, 'r:', lw=1.5, label=r'ODE, static $\theta_s$')
    ax.set_xlim(t0 - 0.2 * T_win, t_end + 0.2 * T_win)
    ax.set_xlabel('t [s]'); ax.set_ylabel('h [cm]')
    ax.set_title(f'3D dynCA capillary rise — {fluid}, R={R_mm} mm (L2={l2:.3f})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, f'dynca_3d_{label}_height'); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogy(t_arr, np.maximum(KE_arr, 1e-30), 'b-', lw=1)
    ax1.set_xlabel('t [s]'); ax1.set_ylabel('KE [J]')
    ax1.set_title('Kinetic energy'); ax1.grid(True, alpha=0.3)
    ax2.plot(t_arr, mass_arr - mass_arr[0], 'k-', lw=1.5, label='dM(t)')
    ax2.plot(t_arr, inj_arr, 'g--', lw=1.5, label='injected (band)')
    ax2.set_xlabel('t [s]'); ax2.set_ylabel('mass [kg]')
    ax2.set_title(f'Mass closure (err {mass_err:.1e})')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, f'dynca_3d_{label}_diag'); plt.close(fig)

    # mid-height velocity profile vs Poiseuille
    try:
        z_mid = 0.5 * float(h_arr[-1])
        band = [v for v in HC.V if abs(v.x_a[2] - z_mid) < 1.2 * dx0]
        if len(band) >= 6:
            rr = np.array([np.hypot(v.x_a[0], v.x_a[1]) for v in band])
            uz = np.array([v.u[2] for v in band])
            hd_now = np.gradient(h_arr, t_arr)[-1]
            ri = np.linspace(0, R, 100)
            up = 2.0 * hd_now * (1 - (ri / R) ** 2)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rr * 1e3, uz * 100, 'bo', label='sim (mid-height)')
            ax.plot(ri * 1e3, up * 100, 'r--', label='Poiseuille @ mean dh/dt')
            ax.set_xlabel('r [mm]'); ax.set_ylabel('u_z [cm/s]')
            ax.set_title('Velocity profile (3D)'); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _savefig(fig, f'dynca_3d_{label}_profile'); plt.close(fig)
    except Exception as e:
        print(f"  (profile plot skipped: {e})")
    plt.close('all')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fluid', default='water')
    ap.add_argument('--R-mm', type=float, default=0.5)
    ap.add_argument('--t0', type=float, default=0.03)
    ap.add_argument('--t-end', type=float, default=0.06)
    ap.add_argument('--n-rings', type=int, default=3)
    ap.add_argument('--cfl', type=float, default=0.4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    kw = dict(fluid=args.fluid, R_mm=args.R_mm, t0=args.t0, t_end=args.t_end,
              n_rings=args.n_rings, cfl=args.cfl, tag=args.tag)
    if args.smoke:
        kw.update(n_rings=2, t_end=args.t0 + 0.005, tag=args.tag + '_smoke')
    main(**kw)
