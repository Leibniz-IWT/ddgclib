#!/usr/bin/env python3
"""2D cube-to-droplet: 3-way comparison of retopologization strategies.

Runs three simulations side by side and produces comparison plots:

    NO_REDIST : Full Delaunay retriangulation every step (baseline)
    REDIST    : Full Delaunay + pressure-preserving mass redistribution
    NO_RETOPO : No retriangulation (dual-only recompute on fixed connectivity)

Produces:
  fig/comparison_circularity.png    — circularity(t) for all three cases
  fig/comparison_radii.png          — R_max / R_min over time
  fig/comparison_pressure.png       — droplet pressure, Laplace reference
  fig/comparison_pressure_std.png   — pressure std dev over time
  fig/comparison_mass.png           — mass conservation comparison
  fig/comparison_ke.png             — kinetic energy
  fig/cube2droplet_2D_redist.mp4    — animation (REDIST case)
  fig/cube2droplet_2D_no_retopo.mp4 — animation (NO_RETOPO case)

Usage
-----
    python cases_dynamic/Cube2droplet/cube_to_droplet_2D_mass_redist.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from functools import partial

from hyperct.ddg import compute_vd

from cases_dynamic.Cube2droplet.src._setup import setup_cube_to_droplet
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _retopologize_multiphase,
)
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid

# =====================================================================
# Parameters (match cube_to_droplet_2D.py)
# =====================================================================
R = 0.01              # half-side of square droplet [m]
L_DOMAIN = 0.03       # half-side of outer box [m]
R_EQ = R * 2.0 / np.sqrt(np.pi)

RHO_D, RHO_O = 800.0, 1000.0
MU_D, MU_O = 2.0, 1.0
GAMMA = 0.01
K_D, K_O = 100.0, 125.0

N_REFINE = 4
DT = 2e-4
N_STEPS = 5000        # 1.0 s of physical time
RECORD_EVERY = 25     # ~200 snapshots

ZOOM = 2.2 * R

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
        _mps.split_dual_volumes(HC, dim)
        _mps.compute_phase_pressures(HC)

    bV.clear()
    bV.update(dV)


def compute_diagnostics(HC, bV, dim=2):
    """Interface shape + pressure + mass diagnostics."""
    R_max, R_min = 0.0, np.inf
    KE, total_mass = 0.0, 0.0
    p_drop, p_outer = [], []

    for v in HC.V:
        total_mass += v.m
        u = v.u[:dim]
        KE += 0.5 * v.m * np.dot(u, u)

        if getattr(v, 'is_interface', False):
            r = np.linalg.norm(v.x_a[:dim])
            R_max = max(R_max, r)
            if r > 1e-30:
                R_min = min(R_min, r)

        if v in bV:
            continue
        if v.phase == 1 and not getattr(v, 'is_interface', False):
            p_drop.append(float(v.p))
        elif v.phase == 0:
            p_outer.append(float(v.p))

    if R_min == np.inf:
        R_min = 0.0

    return {
        'R_max': R_max,
        'R_min': R_min,
        'circularity': R_min / R_max if R_max > 0 else 0.0,
        'KE': KE,
        'total_mass': total_mass,
        'P_drop': float(np.mean(p_drop)) if p_drop else 0.0,
        'P_outer': float(np.mean(p_outer)) if p_outer else 0.0,
    }


def _pressure_std(HC, bV):
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


def _run_simulation(label, retopo_mode):
    """Run a single simulation with the given retopo strategy."""
    dim = 2

    HC, bV, mps, meos, bc_set, dudt_fn, _, params = \
        setup_cube_to_droplet(
            dim=dim, R=R, L_domain=L_DOMAIN,
            rho_d=RHO_D, rho_o=RHO_O, mu_d=MU_D, mu_o=MU_O,
            gamma=GAMMA, K_d=K_D, K_o=K_O, n_refine=N_REFINE,
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

    # Set up history (for animation)
    history = None
    if retopo_mode in ('redist', 'no_retopo'):
        snapshot_dir = os.path.join(_RESULTS, f'snapshots_{retopo_mode}')
        os.makedirs(snapshot_dir, exist_ok=True)
        history = StateHistory(
            fields=['u', 'p', 'phase', 'is_interface'],
            record_every=RECORD_EVERY,
            save_dir=snapshot_dir,
        )

    # Diagnostics arrays
    t_arr = [0.0]
    circ_arr, R_max_arr, R_min_arr = [], [], []
    pd_arr, po_arr, KE_arr, mass_arr, p_std_arr = [], [], [], [], []

    diag0 = compute_diagnostics(HC, bV, dim)
    circ_arr.append(diag0['circularity'])
    R_max_arr.append(diag0['R_max'])
    R_min_arr.append(diag0['R_min'])
    pd_arr.append(diag0['P_drop'])
    po_arr.append(diag0['P_outer'])
    KE_arr.append(diag0['KE'])
    mass_arr.append(diag0['total_mass'])
    p_std_arr.append(_pressure_std(HC, bV))
    print(f"[{label}] Initial circularity: {diag0['circularity']:.4f}")

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if history is not None:
            history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % 100 == 0:
            diag = compute_diagnostics(HC_cb, bV_cb or set(), dim)
            t_arr.append(t)
            circ_arr.append(diag['circularity'])
            R_max_arr.append(diag['R_max'])
            R_min_arr.append(diag['R_min'])
            pd_arr.append(diag['P_drop'])
            po_arr.append(diag['P_outer'])
            KE_arr.append(diag['KE'])
            mass_arr.append(diag['total_mass'])
            p_std_arr.append(_pressure_std(HC_cb, bV_cb or set()))

        if step % 1000 == 0 and step > 0:
            diag = compute_diagnostics(HC_cb, bV_cb or set(), dim)
            dp = diag['P_drop'] - diag['P_outer']
            max_u = max(np.linalg.norm(v.u[:dim]) for v in HC_cb.V)
            print(f"  [{label}] step {step}: t={t:.4f} "
                  f"circ={diag['circularity']:.4f} "
                  f"dP={dp:+.2f} |u|={max_u:.3e}")

    t0 = time.time()
    try:
        t_final = symplectic_euler(
            HC, bV, dudt_fn, dt=DT, n_steps=N_STEPS, dim=dim,
            bc_set=bc_set, retopologize_fn=retopo_fn,
            callback=callback,
        )
    except Exception as e:
        print(f"[{label}] Simulation stopped: {e}")
        t_final = t_arr[-1] if t_arr else 0.0
    wall_time = time.time() - t0

    diag_final = compute_diagnostics(HC, bV, dim)
    t_arr.append(t_final)
    circ_arr.append(diag_final['circularity'])
    R_max_arr.append(diag_final['R_max'])
    R_min_arr.append(diag_final['R_min'])
    pd_arr.append(diag_final['P_drop'])
    po_arr.append(diag_final['P_outer'])
    KE_arr.append(diag_final['KE'])
    mass_arr.append(diag_final['total_mass'])
    p_std_arr.append(_pressure_std(HC, bV))

    print(f"[{label}] Done in {wall_time:.1f}s | "
          f"final circ={circ_arr[-1]:.4f} | "
          f"|dM/M0|={abs(mass_arr[-1] - mass_arr[0]) / mass_arr[0]:.4e}")

    return {
        't': np.array(t_arr),
        'circ': np.array(circ_arr),
        'R_max': np.array(R_max_arr),
        'R_min': np.array(R_min_arr),
        'P_drop': np.array(pd_arr),
        'P_outer': np.array(po_arr),
        'KE': np.array(KE_arr),
        'mass': np.array(mass_arr),
        'p_std': np.array(p_std_arr),
        'HC': HC,
        'bV': bV,
        'history': history,
        'wall_time': wall_time,
    }


def _style_for(mode):
    return {
        'no_redist': ('#1f77b4', '-', 'Delaunay (no redist)'),
        'redist':    ('#d62728', '-', 'Delaunay + redist'),
        'no_retopo': ('#2ca02c', '-', 'No retopo (dual-only)'),
    }[mode]


def main():
    dim = 2
    print("=" * 60)
    print("2D Cube-to-Droplet — 3-Way Retopo Comparison")
    print("=" * 60)
    print(f"R={R} m, L={L_DOMAIN} m, R_eq={R_EQ:.5f} m")
    print(f"gamma={GAMMA}, mu_d={MU_D}, K_d={K_D}")

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

        laplace_dp = GAMMA / R_EQ

        # 1. Circularity
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.plot(res['t'] * 1000, res['circ'], color=c, linestyle=ls,
                    label=lbl)
        ax.axhline(1.0, color='k', linestyle=':', alpha=0.4, label='Circle')
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Circularity (R_min / R_max)")
        ax.set_title("Shape Relaxation: Retopologization Comparison")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.savefig(os.path.join(_FIG, 'comparison_circularity.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # 2. Radii
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.plot(res['t'] * 1000, res['R_max'] * 1000, color=c,
                    linestyle=ls, label=f'{lbl} R_max')
            ax.plot(res['t'] * 1000, res['R_min'] * 1000, color=c,
                    linestyle='--', alpha=0.6, label=f'{lbl} R_min')
        ax.axhline(R_EQ * 1000, color='k', linestyle=':', alpha=0.5,
                    label=f'R_eq={R_EQ * 1000:.2f} mm')
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Radius [mm]")
        ax.set_title("Interface Radii: Retopologization Comparison")
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(os.path.join(_FIG, 'comparison_radii.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # 3. Pressure evolution (dP = P_drop - P_outer)
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.plot(res['t'] * 1000,
                    res['P_drop'] - res['P_outer'],
                    color=c, linestyle=ls, label=lbl)
        ax.axhline(laplace_dp, color='k', linestyle=':', alpha=0.5,
                    label=f'Laplace dP={laplace_dp:.3f}')
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("P_drop - P_outer [Pa]")
        ax.set_title("Pressure Jump: Retopologization Comparison")
        ax.legend()
        fig.savefig(os.path.join(_FIG, 'comparison_pressure.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # 4. Pressure std dev
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.semilogy(res['t'] * 1000, np.maximum(res['p_std'], 1e-30),
                        color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Pressure Std Dev [Pa]")
        ax.set_title("Pressure Stability: Retopologization Comparison")
        ax.legend()
        fig.savefig(os.path.join(_FIG, 'comparison_pressure_std.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # 5. Mass conservation
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            M0 = res['mass'][0]
            ax.plot(res['t'] * 1000,
                    np.abs(res['mass'] - M0) / M0,
                    color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("|dM / M0|")
        ax.set_title("Mass Conservation: Retopologization Comparison")
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(os.path.join(_FIG, 'comparison_mass.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # 6. Kinetic energy
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.semilogy(res['t'] * 1000, np.maximum(res['KE'], 1e-30),
                        color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Kinetic Energy [J]")
        ax.set_title("Energy: Retopologization Comparison")
        ax.legend()
        fig.savefig(os.path.join(_FIG, 'comparison_ke.png'), dpi=150,
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
            suffix = mode
            anim = dynamic_plot_fluid(
                history, HC_r, bV=bV_r,
                save_path=os.path.join(
                    _FIG, f'cube2droplet_2D_{suffix}.mp4'
                ),
                fps=20, dpi=100,
                xlim=(-ZOOM, ZOOM),
                ylim=(-ZOOM, ZOOM),
                phase_field='phase',
                interface_field='is_interface',
                reference_R=R_EQ,
            )
            print(f"Animation saved: fig/cube2droplet_2D_{suffix}.mp4")
        except Exception as e:
            print(f"Animation failed for {mode}: {e}")

    # -- Summary table --
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Mode':<22} {'Wall [s]':>10} {'Final circ':>12} "
          f"{'|dM/M0|':>12} {'Final dP':>12}")
    print("-" * 72)
    for mode, res in results.items():
        _, _, lbl = _style_for(mode)
        M0 = res['mass'][0]
        dM = abs(res['mass'][-1] - M0) / M0
        dP = res['P_drop'][-1] - res['P_outer'][-1]
        print(f"{lbl:<22} {res['wall_time']:>10.1f} "
              f"{res['circ'][-1]:>12.4f} {dM:>12.2e} "
              f"{dP:>12.3f}")
    print(f"\nLaplace dP reference: {GAMMA / R_EQ:.3f} Pa")

    print("\nDone.")


if __name__ == '__main__':
    main()
