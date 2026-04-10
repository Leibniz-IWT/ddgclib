#!/usr/bin/env python3
"""Diagnostic: cube-to-droplet WITHOUT Delaunay retopologization.

This test keeps the primary connectivity (edges/triangles) fixed at the
initial state and only recomputes barycentric duals after each position
update.  Interface vertices remain exactly those identified at t=0.

Purpose: isolate whether the instability comes from Delaunay reconnection
creating spurious interface vertices at corners vs. the underlying
surface tension / pressure dynamics.

Usage
-----
    python cases_dynamic/Cube2droplet/diagnostic_no_retopo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from functools import partial
from hyperct import Complex
from hyperct.ddg import compute_vd
from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.initial_conditions import ZeroVelocity, PhaseAssignment
from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
from cases_dynamic.Cube2droplet.src._setup import AtmosphericPressureBC
from ddgclib.operators.multiphase_stress import multiphase_dudt_i
from ddgclib.operators.stress import dual_volume, cache_dual_volumes
from ddgclib.dynamic_integrators import symplectic_euler

# =====================================================================
# Parameters (same as main case)
# =====================================================================
R = 0.01
L_DOMAIN = 0.03
R_EQ = R * 2.0 / np.sqrt(np.pi)

DIM = 2
RHO_D, RHO_O = 800.0, 1000.0
MU_D, MU_O = 2.0, 1.0
GAMMA = 0.01
K_D, K_O = 100.0, 125.0
N_REFINE = 4
DT = 2e-4
N_STEPS = 5000        # 1.0 s
RECORD_EVERY = 12
ZOOM = 2.2 * R


def dual_only_retopo(HC, bV, dim, _mps=None):
    """Recompute duals on existing connectivity (no Delaunay).

    Keeps all edges/triangles intact, just recomputes barycentric dual
    vertices and caches dual volumes + per-phase volume splits.
    Interface identity is unchanged (same vertices as t=0).
    """
    # Recompute boundary (positions moved but topology is same)
    dV = HC.boundary()
    for v in HC.V:
        v.boundary = v in dV

    # Recompute barycentric duals on the existing triangulation
    compute_vd(HC, method="barycentric")

    # Recache dual volumes
    cache_dual_volumes(HC, dim)

    # Recompute per-phase volume splits (needed by MultiphaseEOS)
    # but do NOT re-identify interface — keep it frozen from t=0
    if _mps is not None:
        _mps.split_dual_volumes(HC, dim)

    # Update bV
    bV.clear()
    bV.update(dV)


def record_frame(HC, dim):
    xs, ps, us, phases, is_iface = [], [], [], [], []
    for v in HC.V:
        xs.append(v.x_a[:dim].copy())
        ps.append(float(v.p) if np.ndim(v.p) == 0 else float(v.p[0]))
        us.append(v.u[:dim].copy())
        phases.append(int(v.phase))
        is_iface.append(bool(getattr(v, 'is_interface', False)))
    return {
        'x': np.array(xs), 'p': np.array(ps), 'u': np.array(us),
        'phase': np.array(phases), 'is_interface': np.array(is_iface),
    }


def compute_diagnostics(HC, dim):
    R_max, R_min = 0.0, np.inf
    p_drop, p_outer = [], []
    for v in HC.V:
        if getattr(v, 'is_interface', False):
            r = np.linalg.norm(v.x_a[:dim])
            R_max = max(R_max, r)
            if r > 1e-30:
                R_min = min(R_min, r)
        if v.phase == 1 and not getattr(v, 'is_interface', False) \
                and not v.boundary:
            p_drop.append(v.p)
        elif v.phase == 0 and not v.boundary:
            p_outer.append(v.p)
    if R_min == np.inf:
        R_min = 0.0
    circ = R_min / R_max if R_max > 0 else 0
    return {
        'circularity': circ, 'R_max': R_max, 'R_min': R_min,
        'P_drop': float(np.mean(p_drop)) if p_drop else 0.0,
        'P_outer': float(np.mean(p_outer)) if p_outer else 0.0,
    }


def main():
    dim = DIM
    print("=" * 60)
    print("DIAGNOSTIC: No-retopo cube-to-droplet")
    print("  (fixed connectivity, dual-only recomputation)")
    print("=" * 60)

    # -- Build mesh --
    HC = Complex(dim, domain=[(-L_DOMAIN, L_DOMAIN)] * dim)
    HC.triangulate()
    for _ in range(N_REFINE):
        HC.refine_all()
    bV = HC.boundary()
    for v in HC.V:
        v.boundary = v in bV
    compute_vd(HC, method="barycentric")
    for v in HC.V:
        try:
            v.dual_vol = dual_volume(v, HC, dim)
        except (ValueError, IndexError):
            v.dual_vol = 0.0

    PhaseAssignment(
        lambda x: 1 if all(abs(x[i]) <= R for i in range(dim)) else 0
    ).apply(HC, bV)

    eos_o = TaitMurnaghan(rho0=RHO_O, P0=0., K=K_O, n=1.0,
                           rho_clip=(0.1, 10.0))
    eos_d = TaitMurnaghan(rho0=RHO_D, P0=0., K=K_D, n=1.0,
                           rho_clip=(0.1, 10.0))
    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_o, mu=MU_O, rho0=RHO_O, name="outer"),
            PhaseProperties(eos=eos_d, mu=MU_D, rho0=RHO_D, name="droplet"),
        ],
        gamma={(0, 1): GAMMA},
    )
    mps.identify_interface(HC)
    mps.init_phase_fields(HC)
    mps.split_dual_volumes(HC, dim)
    mps.compute_phase_masses(HC)
    ZeroVelocity(dim=dim).apply(HC, bV)
    # Mass already set by compute_phase_masses: m_i = sum_k rho_k * V_k
    for v in HC.V:
        v.p = 0.0

    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV)

    # Atmospheric pressure BC on gas-phase vertices near walls
    atm_verts = set()
    for v_wall in bV:
        for nb in v_wall.nn:
            if nb not in bV and nb.phase == 0:
                atm_verts.add(nb)
    if atm_verts:
        bc_set.add(AtmosphericPressureBC(rho0=RHO_O, gas_phase=0),
                    atm_verts)

    meos = MultiphaseEOS([eos_o, eos_d])
    dudt_fn = partial(multiphase_dudt_i, dim=dim, mps=mps, HC=HC,
                      pressure_model=meos)

    # Snapshot initial interface
    n_iface_init = sum(1 for v in HC.V
                       if getattr(v, 'is_interface', False))
    iface_ids_init = {id(v) for v in HC.V
                      if getattr(v, 'is_interface', False)}
    print(f"Mesh: {sum(1 for _ in HC.V)} vertices, "
          f"{n_iface_init} interface")

    diag0 = compute_diagnostics(HC, dim)
    print(f"Initial circularity: {diag0['circularity']:.4f}")

    # -- Recording --
    frames = [(0.0, record_frame(HC, dim))]
    t_arr, circ_arr = [0.0], [diag0['circularity']]
    R_max_arr = [diag0['R_max']]
    R_min_arr = [diag0['R_min']]
    pd_arr = [diag0['P_drop']]
    po_arr = [diag0['P_outer']]

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if step % RECORD_EVERY == 0:
            frames.append((t, record_frame(HC_cb, dim)))
        if step % 200 == 0:
            diag = compute_diagnostics(HC_cb, dim)
            t_arr.append(t)
            circ_arr.append(diag['circularity'])
            R_max_arr.append(diag['R_max'])
            R_min_arr.append(diag['R_min'])
            pd_arr.append(diag['P_drop'])
            po_arr.append(diag['P_outer'])
        if step % 2000 == 0 and step > 0:
            diag = compute_diagnostics(HC_cb, dim)
            n_if = sum(1 for v in HC_cb.V
                       if getattr(v, 'is_interface', False))
            same = sum(1 for v in HC_cb.V
                       if getattr(v, 'is_interface', False)
                       and id(v) in iface_ids_init)
            max_u = max(np.linalg.norm(v.u[:dim]) for v in HC_cb.V)
            dp = diag['P_drop'] - diag['P_outer']
            print(f"  step {step}: t={t:.4f} circ={diag['circularity']:.4f} "
                  f"dP={dp:+.2f} |u|={max_u:.3e} "
                  f"n_if={n_if} (same={same}/{n_iface_init})")

    # -- Run with dual-only retopo --
    print(f"\nRunning {N_STEPS} steps (no Delaunay, dual-only)...")
    try:
        t_final = symplectic_euler(
            HC, bV, dudt_fn, dt=DT, n_steps=N_STEPS, dim=dim,
            bc_set=bc_set, retopologize_fn=partial(dual_only_retopo, _mps=mps),
            callback=callback,
        )
    except Exception as e:
        print(f"Stopped: {e}")
        import traceback; traceback.print_exc()
        t_final = t_arr[-1] if t_arr else 0.0

    # Final
    diag_final = compute_diagnostics(HC, dim)
    t_arr.append(t_final)
    circ_arr.append(diag_final['circularity'])
    R_max_arr.append(diag_final['R_max'])
    R_min_arr.append(diag_final['R_min'])
    pd_arr.append(diag_final['P_drop'])
    po_arr.append(diag_final['P_outer'])

    t_arr = np.array(t_arr)
    circ_arr = np.array(circ_arr)
    R_max_arr = np.array(R_max_arr)
    R_min_arr = np.array(R_min_arr)
    pd_arr = np.array(pd_arr)
    po_arr = np.array(po_arr)

    n_if_final = sum(1 for v in HC.V
                     if getattr(v, 'is_interface', False))
    same_final = sum(1 for v in HC.V
                     if getattr(v, 'is_interface', False)
                     and id(v) in iface_ids_init)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Initial circularity:  {circ_arr[0]:.4f}")
    print(f"Final circularity:    {circ_arr[-1]:.4f}")
    print(f"Max circularity:      {np.max(circ_arr):.4f}")
    dp = pd_arr[-1] - po_arr[-1]
    print(f"Final dP:             {dp:+.2f} Pa (Laplace={GAMMA/R_EQ:.3f})")
    print(f"Interface vertices:   {n_if_final} "
          f"(same as t=0: {same_final}/{n_iface_init})")
    print(f"Frames recorded:      {len(frames)}")

    # -- Plotting --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig_dir = os.path.join(os.path.dirname(__file__), 'fig')
        os.makedirs(fig_dir, exist_ok=True)

        # Circularity comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_arr * 1000, circ_arr, 'b-', lw=1.5)
        ax.axhline(1.0, color='k', ls=':', alpha=0.4, label='Circle')
        ax.axhline(circ_arr[0], color='grey', ls='--', alpha=0.4,
                    label=f'Initial ({circ_arr[0]:.3f})')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Circularity')
        ax.set_title('No-retopo diagnostic: Shape Relaxation')
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.savefig(os.path.join(fig_dir,
                                 'diagnostic_no_retopo_circ.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Pressure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_arr * 1000, pd_arr, 'r-', label='P_drop')
        ax.plot(t_arr * 1000, po_arr, 'b-', label='P_outer')
        ax.plot(t_arr * 1000, pd_arr - po_arr, 'k--', label='dP')
        ax.axhline(GAMMA / R_EQ, color='g', ls=':', alpha=0.5,
                    label=f'Laplace={GAMMA/R_EQ:.3f}')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Pressure [Pa]')
        ax.set_title('No-retopo diagnostic: Pressure')
        ax.legend()
        fig.savefig(os.path.join(fig_dir,
                                 'diagnostic_no_retopo_pressure.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Animation
        print("\nGenerating animation...")
        all_p = np.concatenate([f['p'] for _, f in frames])
        pvmin, pvmax = np.percentile(all_p, [2, 98])
        if pvmin == pvmax:
            pvmin, pvmax = pvmin - 1, pvmax + 1

        fig_a, axes_a = plt.subplots(1, 2, figsize=(14, 6))
        fig_a.subplots_adjust(top=0.88)
        sup = fig_a.suptitle('t = 0.0000 s', fontsize=13)

        def update(fi):
            t, fd = frames[fi]
            x, p, u = fd['x'], fd['p'], fd['u']
            phase, is_if = fd['phase'], fd['is_interface']
            for ax in axes_a:
                ax.clear()
            sup.set_text(f't = {t:.4f} s')

            # Left: pressure
            ax_p = axes_a[0]
            ax_p.scatter(x[:, 0], x[:, 1], c=p, cmap='viridis',
                         vmin=pvmin, vmax=pvmax, s=8, zorder=2)
            if np.any(is_if):
                ax_p.scatter(x[is_if, 0], x[is_if, 1],
                             facecolors='none', edgecolors='red',
                             s=30, lw=1.0, zorder=4,
                             label='Interface')
            ax_p.set_title('Pressure [Pa]')
            ax_p.set_xlim(-ZOOM, ZOOM)
            ax_p.set_ylim(-ZOOM, ZOOM)
            ax_p.set_aspect('equal')
            ax_p.legend(fontsize=7, loc='upper right')

            # Right: phase + velocity
            ax_v = axes_a[1]
            c_ph = np.where(phase == 1, 'lightsalmon', 'lightblue')
            ax_v.scatter(x[:, 0], x[:, 1], c=c_ph, s=8, zorder=1,
                         alpha=0.6)
            u_mag = np.linalg.norm(u, axis=1)
            mx = u_mag.max()
            if mx > 1e-15:
                ax_v.quiver(x[:, 0], x[:, 1], u[:, 0], u[:, 1],
                            u_mag, cmap='hot', scale=mx * 20,
                            width=0.003, zorder=3)
            if np.any(is_if):
                ax_v.scatter(x[is_if, 0], x[is_if, 1],
                             facecolors='none', edgecolors='red',
                             s=30, lw=1.0, zorder=4,
                             label='Interface')
            ax_v.set_title('Phase + Velocity')
            ax_v.set_xlim(-ZOOM, ZOOM)
            ax_v.set_ylim(-ZOOM, ZOOM)
            ax_v.set_aspect('equal')
            ax_v.legend(fontsize=7, loc='upper right')
            return []

        anim = FuncAnimation(fig_a, update, frames=len(frames),
                             interval=33, blit=False)
        mp4 = os.path.join(fig_dir, 'diagnostic_no_retopo_fluid.mp4')
        anim.save(mp4, writer='ffmpeg', fps=30, dpi=120)
        plt.close(fig_a)
        print(f"Animation: {mp4}")

        print(f"\nAll output in {fig_dir}/")
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback; traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
