#!/usr/bin/env python3
"""2D Dynamic Capillary Rise — data-driven contact-line forcing (dynCA).

Modern version of the dynamic capillary rise case using the machinery
validated in the 2026-07 oscillating-droplet campaign (exact barycentric
dual volumes, EOS clip coherence, mass-conserving adaptive remesh, no
global retriangulation).

The three-phase contact force is NOT modelled: the meniscus driving
pressure is back-computed from the *measured* dynamic contact angle
CA(t) of Heshmati & Piri (2014) / Lunowa et al. (2022), and the
simulation predicts the meniscus rise h(t), which is compared against
the measured rise.

Geometry: matched 2D slit (half-width a = R/2, mu_2d = 2/3 mu) whose
reduced-order dynamics are identical to the experimental 3D tube.
Reservoir: Eulerian mass-source band below the datum (y < 0) with
adaptive edge splitting — a constant-pressure reservoir without ghost
mesh injection.

Usage
-----
    python cases_dynamic/capillary_rise/capillary_rise_2D_dynCA.py \
        [--fluid water] [--R-mm 0.5] [--t0 0.01] [--t-end T] [--nx 6] [--smoke]
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

from cases_dynamic.capillary_rise.src._params import FLUIDS, g as G
from cases_dynamic.capillary_rise.src._dynamic_ca import (
    ExperimentalDrive, matched_slit_params, rise_ode_solve,
)
from cases_dynamic.capillary_rise.src._setup_dynca import (
    build_strip_2d, tag_groups_2d, make_eos, band_mass_reset,
    boundary_mass_reset, apply_ics, make_dudt_dynca,
    extract_surface_chain, surface_tension_forces,
)

from hyperct.remesh import adaptive_remesh
from hyperct.ddg import rebuild_simplex_cache_2d
from ddgclib.multiphase import mass_conserving_merge
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


def main(fluid='water', R_mm=0.5, t0=0.01, t_end=None, nx=6, cfl=0.4,
         c0_factor=10.0, H_res_frac=4.0, remesh_every=5, record_dt=None,
         gif=False, tag='', debug=False, drive_mode='surface') -> dict:
    dim, gravity_axis = 2, 1
    drive = ExperimentalDrive(fluid, R_mm)
    rho, mu, gamma = drive.rho, drive.mu, drive.gamma
    a, mu2 = matched_slit_params(drive.R, mu)      # matched slit
    width = 2.0 * a
    if t_end is None:
        t_end = drive.t_max

    h0 = float(drive.h_exp(t0))
    hdot0 = float(drive.hdot_exp(t0))
    h_jurin = 2.0 * gamma * np.cos(np.deg2rad(drive.theta_s_deg)) / (rho * G * drive.R)

    # Sound speed: 10x the fastest expected signal (rise velocity or
    # gravity wave on the final column)
    tq = np.linspace(drive.t_min, drive.t_max, 500)
    u_ref = max(float(np.max(np.abs(drive.hdot_exp(tq)))), np.sqrt(G * h_jurin))
    c0 = c0_factor * u_ref
    eos = make_eos(rho, c0)

    label = f"{fluid}_R{R_mm:g}mm{tag}"
    print("=" * 64)
    print(f"2D dynCA capillary rise — {fluid}, R={R_mm} mm (matched slit)")
    print("=" * 64)
    print(f"  a = {a*1e3:.3f} mm, mu_2d = {mu2:.3e} Pa s, rho = {rho}")
    print(f"  window: t_exp in [{t0}, {t_end}] s, h0 = {h0*100:.3f} cm, "
          f"hdot0 = {hdot0*100:.2f} cm/s")
    print(f"  h_jurin(tube) = {h_jurin*100:.3f} cm, c0 = {c0:.2f} m/s")

    # ── Mesh ────────────────────────────────────────────────────────────
    H_res = H_res_frac * width          # reservoir band depth below datum
    HC, meta = build_strip_2d(width, -H_res, h0, nx)
    dx0 = meta['dx']
    # geometric tolerance: fluid within 5% of a cell of the wall line is
    # absorbed into the wall film (prevents un-tagged stragglers pushing
    # through the wall)
    wall_tol = 0.05 * dx0
    n_verts0 = sum(1 for _ in HC.V)
    print(f"  mesh: {n_verts0} verts, dx = {dx0*1e6:.1f} um, band depth "
          f"{H_res*1e3:.2f} mm")

    groups = tag_groups_2d(HC, width, -H_res, wall_tol)
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=dim)

    # ── ICs + forcing ───────────────────────────────────────────────────
    apply_ics(HC, dim, rho, eos, G, a, hdot0, -H_res, wall_tol)
    band_mass_reset(HC, eos, rho, G, gravity_axis)

    state = {'t': t0, 'h': h0, 'p_cap': 0.0, 'F_surf': {}}
    dudt_fn, update_state = make_dudt_dynca(
        dim, mu2, HC, eos, rho, G, drive.p_cap_slit, state,
        drive_mode=drive_mode)
    update_state(t0, h0)
    print(f"  drive mode: {drive_mode}"
          + (" (resolved meniscus line tension + data contact force)"
             if drive_mode == 'surface' else " (Washburn-equivalent)"))
    print(f"  P_cap(t0) = {state['p_cap']:.1f} Pa "
          f"(static-theta: {gamma*np.cos(np.deg2rad(drive.theta_s_deg))/a:.1f} Pa)")

    # ── History / diagnostics ──────────────────────────────────────────
    snap_dir = os.path.join(_RESULTS, 'snapshots', f'dynca_2d_{label}')
    history = StateHistory(fields=['u', 'p'], record_every=1, save_dir=snap_dir)
    if record_dt is None:
        record_dt = (t_end - t0) / 150.0

    t_arr = [t0]
    h_arr = [h0]
    KE_arr = [0.5 * sum(v.m * float(v.u @ v.u) for v in HC.V)]
    mass_tot0 = sum(v.m for v in HC.V)
    mass_arr = [mass_tot0]
    injected = 0.0
    inj_arr = [0.0]
    pcap_arr = [state['p_cap']]
    history.append(t0, HC)

    # ── Mesh maintenance: merge collapsing pairs + local remesh ────────
    # Lagrangian Poiseuille shear slides the fast center columns past the
    # frozen wall columns, so connectivity must be repaired continuously
    # (flips) and near-coincident pairs merged before the CFL dt
    # collapses.  All ops are the mass-conserving lane4 remesh primitives.
    n_cap_events = 0

    def _cleanup_orphans():
        """Merge under-connected vertices (nn <= 2) into their nearest
        neighbour — mass and momentum conserving.  A free surface with
        no resolved surface tension has no cohesion beyond EOS tension,
        and boundary-adjacent remesh ops can leave torn-off vertices
        whose duals are degenerate."""
        n = 0
        for v in [v for v in HC.V if len(list(v.nn)) <= 2]:
            nbs = list(v.nn)
            if not nbs:
                HC.V.remove(v)
                n += 1
                continue
            w = min(nbs, key=lambda q: float(
                np.linalg.norm(q.x_a[:dim] - v.x_a[:dim])))
            m_v = getattr(v, 'm', 0.0)
            m_w = getattr(w, 'm', 0.0)
            if m_v > 0.0 and m_w + m_v > 0.0:
                w.u = (m_w * w.u + m_v * v.u) / (m_w + m_v)
                w.m = m_w + m_v
            for q in nbs:
                if q is not w:
                    q.connect(w)
            HC.V.remove(v)
            n += 1
        return n

    def maintain_mesh():
        n_merged = mass_conserving_merge(HC, cdist=0.3 * dx0)
        stats = adaptive_remesh(
            HC, dim=2, L_min=0.45 * dx0, L_max=1.6 * dx0,
            quality_target_deg=15.0, max_iterations=1,
            smooth_iterations=0)
        # collapse/flip can themselves leave a near-zero edge — sweep again
        n_merged += mass_conserving_merge(HC, cdist=0.3 * dx0)
        n_orph = _cleanup_orphans()
        if n_merged or n_orph or stats['n_splits'] or stats['n_collapses'] \
                or stats['n_flips']:
            # Local ops bypass connect_and_cache_simplices — rebuild the
            # simplex cache so exact dual volumes stay fresh
            # (lane4-remesh-upstream, 2026-07-02).
            rebuild_simplex_cache_2d(HC)
        stats['n_orphans'] = n_orph
        return n_merged + n_orph, stats

    def min_edge():
        return min((float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
                    for v in HC.V for nb in v.nn
                    if float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])) > 0),
                   default=dx0)

    # ── Manual CFL loop ────────────────────────────────────────────────
    t_sim, step = 0.0, 0
    T_win = t_end - t0
    next_rec = record_dt
    wall_t0 = time.time()
    aborted = None

    while t_sim < T_win:
        groups = tag_groups_2d(HC, width, -H_res, wall_tol)
        _recompute_duals(HC)
        cache_dual_volumes(HC, dim=dim)

        # mesh maintenance on cadence or when an edge is collapsing
        dx_min = min_edge()
        if step % remesh_every == 0 or dx_min < 0.25 * dx0:
            # Density-continuity repair: local remesh ops conserve mass
            # within each op, but ring-neighbour dual shares still shift
            # (e.g. collapse neighbours), stepping densities and firing
            # spurious EOS forces.  Snapshot fresh densities, remesh,
            # then restore each surviving vertex's density at its new
            # dual volume; the difference is booked to the exchange
            # ledger so closure stays exactly auditable.
            dens0 = {id(v): v.m / v.dual_vol for v in HC.V
                     if getattr(v, 'dual_vol', 0.0) > 0.0}
            n_merged, stats = maintain_mesh()
            if n_merged or stats['n_splits'] or stats['n_collapses'] \
                    or stats['n_flips']:
                _recompute_duals(HC)
                cache_dual_volumes(HC, dim=dim)
                for v in HC.V:
                    tgt = dens0.get(id(v))
                    vol = getattr(v, 'dual_vol', 0.0)
                    if tgt is not None and vol > 0.0:
                        m_new = tgt * vol
                        injected += m_new - v.m
                        v.m = m_new
                groups = tag_groups_2d(HC, width, -H_res, wall_tol)
            dx_min = min_edge()
            if dx_min < 0.05 * dx0:
                aborted = f"degenerate edge {dx_min:.3e} m survived merge"
                print(f"  ABORT: {aborted}")
                break

        # resolved free-surface chain (surface verts + contact endpoints)
        chain = extract_surface_chain(HC, width, -H_res, wall_tol)
        if chain is not None:
            contact = {chain[0], chain[-1]}
            surf = set(chain)
            frozen = (groups['frozen'] - surf) | (groups['bottom'] - contact)
        else:
            if step % 500 == 0:
                print(f"  step {step}: WARNING surface chain broken, "
                      f"falling back to positional groups")
            contact = groups['contact']
            surf = groups['surface']
            frozen = groups['frozen']
        injected += band_mass_reset(HC, eos, rho, G, gravity_axis)
        injected += boundary_mass_reset(HC, frozen | contact, eos, rho, G,
                                        gravity_axis)
        for v in frozen:                      # strict no-slip on the film
            v.u[:dim] = 0.0
        # One-sided free-surface BC with a 10% tolerance band: a free
        # surface cannot support COMPRESSION (spurious +p spikes at
        # truncated half-cells stall the rise), but shaving at exactly
        # rho0 bleeds mass on every noise fluctuation (production run:
        # early rise rate dropped to ~25% of experiment), and clamping
        # to p=0 outright inverts the pressure field of a column under
        # capillary suction.  Shave only clear compression spikes.
        for v in surf:
            vol = getattr(v, 'dual_vol', 0.0)
            if vol > 0.0 and v not in frozen and v.m > 1.10 * rho * vol:
                m_new = 1.10 * rho * vol
                injected += m_new - v.m
                v.m = m_new

        if debug:
            for v in HC.V:
                if v in frozen:
                    continue          # reset to target every step
                vol = getattr(v, 'dual_vol', None)
                dens = v.m / vol if vol and vol > 0 else np.nan
                if (vol is None or vol <= 0 or v.m <= 0
                        or not np.isfinite(v.x_a[:dim]).all()
                        or not np.isfinite(v.u[:dim]).all()
                        or not (0.25 * rho < dens < 3.5 * rho)):
                    print(f"  DEBUG step {step}: invariant violation at "
                          f"x={v.x_a[:dim].tolist()} m={v.m:.3e} "
                          f"vol={vol} dens={dens:.1f} u={v.u[:dim].tolist()} "
                          f"nn={len(list(v.nn))} "
                          f"frozen={v in frozen} contact={v in contact}")
                    aborted = 'debug invariant violation'
                    break
            if aborted:
                break

        # Contact vertices are slaved kinematically (see below), not
        # force-integrated: a finite line force on a point vertex of
        # vanishing mass has mesh-dependent acceleration — the discrete
        # face of the missing mesh-independent contact force model.
        iv = [v for v in HC.V if v not in frozen and v not in contact]
        u_max = max((float(np.linalg.norm(v.u[:dim])) for v in iv), default=0.0)
        dt_cap = 0.5 * np.sqrt(rho * dx_min ** 3 / gamma)
        dt = min(cfl * dx_min / (c0 + u_max), dt_cap, T_win - t_sim)

        # meniscus height + time-dependent forcing
        # column height from mass above datum — smooth and immune to
        # surface-chain churn (h = M+ / (rho0 * width) per unit depth)
        m_above = sum(v.m for v in HC.V if v.x_a[1] > 0.0)
        h_now = m_above / (rho * width)
        F_surf = None
        if chain is not None:
            # surface mode: measured-CA contact pull drives the rise;
            # body mode: zero contact pull — tension only regularizes
            cos_drive = (float(drive.cos_theta_exp(t0 + t_sim))
                         if drive_mode == 'surface' else 0.0)
            F_surf = surface_tension_forces(chain, gamma, cos_drive)
        update_state(t0 + t_sim, h_now, F_surf)

        acc = {v: dudt_fn(v) for v in iv}
        for v in iv:
            v.u[:dim] = v.u[:dim] + dt * acc[v][:dim]
            # failsafe: cap runaway vertices at 2*c0 (count occurrences)
            u_norm = float(np.linalg.norm(v.u[:dim]))
            if u_norm > 2.0 * c0:
                v.u[:dim] *= 2.0 * c0 / u_norm
                n_cap_events += 1
            new_x = v.x_a[:dim] + dt * v.u[:dim]
            if new_x[0] < 0.0 or new_x[0] > width:   # wall is impenetrable
                new_x[0] = min(max(new_x[0], 0.0), width)
                v.u[0] = 0.0
            _move(v, new_x, HC, frozen)

        # Kinematic contact-line slaving: place each contact vertex so
        # the first surface segment meets the wall at the MEASURED
        # dynamic contact angle theta_exp(t) (data-driven, no model).
        if chain is not None and len(chain) >= 3:
            # body mode slaves the contact flat (90 deg): the surface is
            # a regularized free surface, the drive is the body force
            th = (np.deg2rad(float(drive.theta_exp_deg(t0 + t_sim)))
                  if drive_mode == 'surface' else 0.5 * np.pi)
            tan_th = max(np.tan(th), 1e-6)
            # contact-line speed limited to the physical scale (a few
            # times the measured rise rate) — the slaving velocity feeds
            # the neighbours' viscous flux, so ramming the target angle
            # at the displacement cap catapults the surface.
            v_cl = max(3.0 * abs(float(drive.hdot_exp(t0 + t_sim))), 0.02)
            for vc, vn, xw in ((chain[0], chain[1], 0.0),
                               (chain[-1], chain[-2], width)):
                dx_n = abs(vn.x_a[0] - xw)
                y_t = vn.x_a[1] + dx_n / tan_th
                dy = float(np.clip(y_t - vc.x_a[1], -v_cl * dt, v_cl * dt))
                vc.u[0] = 0.0
                vc.u[1] = dy / dt
                _move(vc, np.array([xw, vc.x_a[1] + dy]), HC, frozen)

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

        if step % 500 == 0:
            rate = step / max(time.time() - wall_t0, 1e-9)
            eta = (T_win - t_sim) / max(dt, 1e-12) / max(rate, 1e-9) / 60.0
            print(f"  step {step}: t_exp={t0+t_sim:.4f}s h={h_arr[-1]*100:.3f}cm "
                  f"h_exp={float(drive.h_exp(t0+t_sim))*100:.3f}cm "
                  f"u_max={u_max:.3f} dx_min/dx0={dx_min/dx0:.2f} "
                  f"n={sum(1 for _ in HC.V)} caps={n_cap_events} "
                  f"[{rate:.1f} st/s, ETA {eta:.0f} min]")

        if u_max > 5.0 * c0 or not np.isfinite(u_max):
            aborted = f"u_max={u_max:.3e} exceeded 5*c0"
            print(f"  ABORT: {aborted}")
            break

    wall_min = (time.time() - wall_t0) / 60.0
    t_arr, h_arr = np.array(t_arr), np.array(h_arr)
    KE_arr, mass_arr = np.array(KE_arr), np.array(mass_arr)
    inj_arr, pcap_arr = np.array(inj_arr), np.array(pcap_arr)

    # ── Comparison metrics ──────────────────────────────────────────────
    h_exp_i = drive.h_exp(t_arr)
    l2 = float(np.sqrt(np.mean((h_arr - h_exp_i) ** 2)) / max(np.ptp(h_exp_i), 1e-12))
    err_final = float((h_arr[-1] - h_exp_i[-1]) / max(h_exp_i[-1], 1e-12))
    mass_err = float(abs(mass_arr[-1] - mass_arr[0] - inj_arr[-1])
                     / max(mass_arr[0], 1e-30))
    print(f"\n  done: {step} steps in {wall_min:.1f} min "
          f"(t_exp reached {t_arr[-1]:.4f} s)")
    print(f"  h_final sim/exp = {h_arr[-1]*100:.3f} / {h_exp_i[-1]*100:.3f} cm "
          f"({100*err_final:+.1f} %)")
    print(f"  normalized L2(h_sim - h_exp) = {l2:.4f}")
    print(f"  mass closure |dM - injected|/M0 = {mass_err:.3e}")

    # ── ODE references ─────────────────────────────────────────────────
    t_ode, h_dyn, _ = rise_ode_solve(drive, t0, t_end, theta_mode='dynamic')
    _, h_sta, _ = rise_ode_solve(drive, t0, t_end, theta_mode='static')

    # ── Save results ────────────────────────────────────────────────────
    out = {
        'case': 'capillary_rise_2d_dynCA', 'fluid': fluid, 'R_mm': R_mm,
        'matched_a_m': a, 'mu_2d': mu2, 't0': t0, 't_end': t_end,
        'nx': nx, 'cfl': cfl, 'c0': c0, 'steps': step,
        'wall_minutes': wall_min, 'aborted': aborted,
        'n_cap_events': n_cap_events,
        'l2_h_norm': l2, 'err_final_rel': err_final, 'mass_err': mass_err,
        'h_jurin_m': h_jurin,
        't': t_arr.tolist(), 'h_sim': h_arr.tolist(),
        'h_exp': np.asarray(h_exp_i).tolist(),
        'KE': KE_arr.tolist(), 'mass': mass_arr.tolist(),
        'injected': inj_arr.tolist(), 'p_cap': pcap_arr.tolist(),
        't_ode': t_ode.tolist(), 'h_ode_dynCA': h_dyn.tolist(),
        'h_ode_static': h_sta.tolist(),
    }
    res_path = os.path.join(_RESULTS, f'dynca_2d_{label}.json')
    with open(res_path, 'w') as f:
        json.dump(out, f)
    print(f"  results -> {res_path}")
    save_state(HC, groups['frozen'], t=t_arr[-1], fields=['u', 'p', 'm'],
               path=os.path.join(_RESULTS, f'dynca_2d_{label}_final.json'),
               extra_meta={'case': 'capillary_rise_2d_dynCA'})

    # ── Plots ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(drive.t_data, drive.h_data * 100, 'ko', ms=4, mfc='none',
            label='experiment (Lunowa/Heshmati-Piri)')
    ax.plot(t_arr, h_arr * 100, 'b-', lw=2, label='DDG sim (exp CA(t) forcing)')
    ax.plot(t_ode, h_dyn * 100, 'g--', lw=1.5, label='ODE, exp CA(t)')
    ax.plot(t_ode, h_sta * 100, 'r:', lw=1.5, label=r'ODE, static $\theta_s$')
    ax.axhline(h_jurin * 100, color='k', ls=':', lw=1, alpha=0.5,
               label=f'Jurin {h_jurin*100:.2f} cm')
    ax.set_xlabel('t [s]'); ax.set_ylabel('h [cm]')
    ax.set_title(f'Dynamic capillary rise — {fluid}, R={R_mm} mm '
                 f'(L2={l2:.3f})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, f'dynca_2d_{label}_height'); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    tq2 = np.linspace(t0, t_end, 300)
    ax1.plot(drive._t_ca, drive._ca, 'ko', ms=3, label='measured CA')
    ax1.plot(tq2, drive.theta_exp_deg(tq2), 'b-', lw=1, label='interpolant')
    ax1.axhline(drive.theta_s_deg, color='r', ls=':', label=r'$\theta_s$')
    ax1.set_xlabel('t [s]'); ax1.set_ylabel('CA [deg]')
    ax1.set_title('Dynamic contact angle (input)'); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(t_arr, pcap_arr, 'b-', lw=1.5)
    ax2.set_xlabel('t [s]'); ax2.set_ylabel(r'$P_{cap}(t)$ [Pa]')
    ax2.set_title('Back-computed driving pressure'); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, f'dynca_2d_{label}_forcing'); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogy(t_arr, np.maximum(KE_arr, 1e-30), 'b-', lw=1)
    ax1.set_xlabel('t [s]'); ax1.set_ylabel('KE [J/m]')
    ax1.set_title('Kinetic energy'); ax1.grid(True, alpha=0.3)
    ax2.plot(t_arr, mass_arr - mass_arr[0], 'k-', lw=1.5, label='dM(t)')
    ax2.plot(t_arr, inj_arr, 'g--', lw=1.5, label='injected (band)')
    ax2.set_xlabel('t [s]'); ax2.set_ylabel('mass [kg/m]')
    ax2.set_title(f'Mass closure (err {mass_err:.1e})')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, f'dynca_2d_{label}_diag'); plt.close(fig)

    # velocity profile at mid-height vs Poiseuille
    try:
        ys = np.array([v.x_a[1] for v in HC.V])
        y_mid = 0.5 * float(h_arr[-1])
        band = [v for v in HC.V if abs(v.x_a[1] - y_mid) < 1.2 * dx0]
        if len(band) >= 4:
            xs = np.array([v.x_a[0] for v in band])
            us = np.array([v.u[1] for v in band])
            hd_now = np.gradient(h_arr, t_arr)[-1]
            xi = np.linspace(0, width, 100)
            up = 1.5 * hd_now * (1 - ((xi - a) / a) ** 2)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs * 1e3, us * 100, 'bo', label='sim (mid-height)')
            ax.plot(xi * 1e3, up * 100, 'r--', label='Poiseuille @ mean dh/dt')
            ax.set_xlabel('x [mm]'); ax.set_ylabel('u_y [cm/s]')
            ax.set_title('Velocity profile'); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _savefig(fig, f'dynca_2d_{label}_profile'); plt.close(fig)
    except Exception as e:
        print(f"  (profile plot skipped: {e})")

    if gif and history.n_snapshots > 1:
        try:
            from ddgclib.visualization import dynamic_plot_fluid
            dynamic_plot_fluid(
                history, HC, groups['frozen'], scalar_field='p',
                vector_field='u',
                save_path=os.path.join(_FIG, f'dynca_2d_{label}.gif'),
                fps=12, writer='pillow')
            print(f"  -> fig/dynca_2d_{label}.gif")
        except Exception as e:
            print(f"  animation failed: {e}")
    plt.close('all')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fluid', default='water')
    ap.add_argument('--R-mm', type=float, default=0.5)
    ap.add_argument('--t0', type=float, default=0.01)
    ap.add_argument('--t-end', type=float, default=None)
    ap.add_argument('--nx', type=int, default=6)
    ap.add_argument('--cfl', type=float, default=0.4)
    ap.add_argument('--remesh-every', type=int, default=5)
    ap.add_argument('--gif', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--smoke', action='store_true',
                    help='coarse mesh, short window (sanity check)')
    ap.add_argument('--debug', action='store_true',
                    help='per-step invariant checks (slow)')
    ap.add_argument('--drive-mode', default='surface',
                    choices=['surface', 'body'],
                    help='surface: resolved meniscus line tension + data '
                         'contact force; body: Washburn-equivalent body force')
    args = ap.parse_args()
    kw = dict(fluid=args.fluid, R_mm=args.R_mm, t0=args.t0, t_end=args.t_end,
              nx=args.nx, cfl=args.cfl, remesh_every=args.remesh_every,
              gif=args.gif, tag=args.tag, debug=args.debug,
              drive_mode=args.drive_mode)
    if args.smoke:
        kw.update(nx=4, t_end=args.t0 + 0.02, tag=args.tag + '_smoke')
    main(**kw)
