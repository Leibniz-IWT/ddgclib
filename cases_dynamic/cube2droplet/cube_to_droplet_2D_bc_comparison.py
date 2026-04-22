#!/usr/bin/env python3
"""2D cube-to-droplet: comparison of outer-phase boundary condition strategies.

This script investigates how different treatments of the outer-phase mass
affect pressure wave propagation and interface stability.  The underlying
retopologization is kept fixed to NO_RETOPO (dual-only recompute) since
the previous comparison showed that Delaunay retriangulation is a
*separate* source of instability from BC issues.

Compared strategies:

    ATMOS      : Baseline — hard mass reset (AtmosphericPressureBC)
    RESERVOIR  : Smooth relaxation toward fixed rho_target (PressureReservoirBC)
    ABSORB     : Neighbour-averaged absorbing BC (AbsorbingPressureBC)
    EXPANDING  : Floating reservoir that drifts with interior density
                 (ExpandingDomainBC)

Produces:
  fig/bc_comparison_circularity.png    — convergence toward round shape
  fig/bc_comparison_radii.png          — R_max / R_min over time
  fig/bc_comparison_pressure.png       — droplet pressure jump vs Laplace
  fig/bc_comparison_pressure_std.png   — domain-wide pressure variance
  fig/bc_comparison_mass.png           — mass conservation (reservoirs leak!)
  fig/bc_comparison_ke.png             — kinetic energy
  fig/cube2droplet_2D_bc_reservoir.mp4 — animation (PressureReservoirBC)
  fig/cube2droplet_2D_bc_expanding.mp4 — animation (ExpandingDomainBC)

Usage
-----
    python cases_dynamic/Cube2droplet/cube_to_droplet_2D_bc_comparison.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from functools import partial

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.initial_conditions import ZeroVelocity, PhaseAssignment
from ddgclib._boundary_conditions import (
    BoundaryConditionSet, NoSlipWallBC,
    PressureReservoirBC, AbsorbingPressureBC, ExpandingDomainBC,
)
from ddgclib.operators.multiphase_stress import multiphase_dudt_i
from ddgclib.operators.stress import dual_volume, cache_dual_volumes
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid

from cases_dynamic.Cube2droplet.src._setup import AtmosphericPressureBC

# =====================================================================
# Parameters (match main cube2droplet case)
# =====================================================================
R = 0.01
L_DOMAIN = 0.03
R_EQ = R * 2.0 / np.sqrt(np.pi)

RHO_D, RHO_O = 800.0, 1000.0
MU_D, MU_O = 2.0, 1.0
GAMMA = 0.01
K_D, K_O = 100.0, 125.0

# Acoustic timescales
C_S = np.sqrt(K_O / RHO_O)          # outer-phase sound speed ≈ 0.35 m/s
TAU_ACOUSTIC = L_DOMAIN / C_S       # ≈ 0.085 s
TAU_BULK = 10.0 * TAU_ACOUSTIC      # slow drift timescale for ExpandingDomain

N_REFINE = 4
DT = 2e-4
N_STEPS = 5000
RECORD_EVERY = 25

ZOOM = 2.2 * R

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results')


def dual_only_retopo_multiphase(HC, bV, dim, _mps=None):
    """Fixed-topology retopo: recompute duals on existing connectivity."""
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


def build_case():
    """Build a fresh cube2droplet mesh + physics (same as setup but
    returns atm_verts instead of wiring the BC)."""
    dim = 2
    bounds = [(-L_DOMAIN, L_DOMAIN)] * dim
    HC = Complex(dim, domain=bounds)
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

    eos_outer = TaitMurnaghan(rho0=RHO_O, P0=0.0, K=K_O, n=1.0,
                               rho_clip=(0.1, 10.0))
    eos_drop = TaitMurnaghan(rho0=RHO_D, P0=0.0, K=K_D, n=1.0,
                              rho_clip=(0.1, 10.0))

    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_outer, mu=MU_O, rho0=RHO_O, name="outer"),
            PhaseProperties(eos=eos_drop, mu=MU_D, rho0=RHO_D, name="droplet"),
        ],
        gamma={(0, 1): GAMMA},
    )
    mps.refresh(
        HC, dim, reset_mass=True,
        criterion_fn=lambda c: 1 if all(abs(c[i]) <= R for i in range(dim)) else 0,
    )

    ZeroVelocity(dim=dim).apply(HC, bV)
    for v in HC.V:
        v.p = 0.0

    # The gas-phase vertices adjacent to walls that outer-phase BCs target
    atm_verts = set()
    for v_wall in bV:
        for nb in v_wall.nn:
            if nb not in bV and nb.phase == 0:
                atm_verts.add(nb)

    meos = MultiphaseEOS([eos_outer, eos_drop])
    dudt_fn = partial(
        multiphase_dudt_i, dim=dim, mps=mps, HC=HC, pressure_model=meos,
    )
    return HC, bV, mps, meos, dudt_fn, atm_verts


def compute_diagnostics(HC, bV, dim=2):
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


def _make_outer_bc(mode):
    """Return a BoundaryCondition for the outer-phase wall vertices."""
    if mode == 'atmos':
        return AtmosphericPressureBC(rho0=RHO_O, gas_phase=0)
    if mode == 'reservoir':
        return PressureReservoirBC(
            rho_target=RHO_O,
            tau_inv=1.0 / TAU_ACOUSTIC,  # fast: one acoustic transit
            gas_phase=0,
        )
    if mode == 'absorb':
        return AbsorbingPressureBC(
            tau_inv=1.0 / TAU_ACOUSTIC,
            phase=0,
        )
    if mode == 'expanding':
        return ExpandingDomainBC(
            initial_rho_target=RHO_O,
            tau_inv=1.0 / TAU_ACOUSTIC,
            tau_inv_target=1.0 / TAU_BULK,
            gas_phase=0,
        )
    raise ValueError(f"Unknown mode: {mode}")


def _run_simulation(label, bc_mode):
    dim = 2
    HC, bV, mps, meos, dudt_fn, atm_verts = build_case()

    retopo_fn = partial(dual_only_retopo_multiphase, _mps=mps)

    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV)
    bc_set.add(_make_outer_bc(bc_mode), atm_verts)

    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"\n[{label}] Mesh: {n_verts} vertices, {n_iface} interface, "
          f"{len(atm_verts)} wall-adjacent gas vertices")

    # History for animation (only for two key cases)
    history = None
    if bc_mode in ('reservoir', 'expanding'):
        snapshot_dir = os.path.join(_RESULTS, f'snapshots_bc_{bc_mode}')
        os.makedirs(snapshot_dir, exist_ok=True)
        history = StateHistory(
            fields=['u', 'p', 'phase', 'is_interface'],
            record_every=RECORD_EVERY,
            save_dir=snapshot_dir,
        )

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
                  f"circ={diag['circularity']:.4f} dP={dp:+.2f} "
                  f"|u|={max_u:.3e}")

    t0 = time.time()
    try:
        t_final = symplectic_euler(
            HC, bV, dudt_fn, dt=DT, n_steps=N_STEPS, dim=dim,
            bc_set=bc_set, retopologize_fn=retopo_fn, callback=callback,
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
        'atmos':     ('#1f77b4', '-', 'AtmosphericPressureBC (reset)'),
        'reservoir': ('#d62728', '-', 'PressureReservoirBC (sink)'),
        'absorb':    ('#2ca02c', '-', 'AbsorbingPressureBC (neighbour avg)'),
        'expanding': ('#9467bd', '-', 'ExpandingDomainBC (drifting)'),
    }[mode]


def main():
    print("=" * 60)
    print("2D Cube-to-Droplet — Boundary Condition Comparison")
    print("=" * 60)
    print(f"R={R} m, L={L_DOMAIN} m, R_eq={R_EQ:.5f} m")
    print(f"gamma={GAMMA}, K_o={K_O}, c_s={C_S:.3f} m/s, "
          f"tau_ac={TAU_ACOUSTIC:.4f} s")
    print(f"Retopo mode: dual-only (topology frozen)")

    results = {}
    for label, mode in [
        ('ATMOS', 'atmos'),
        ('RESERVOIR', 'reservoir'),
        ('ABSORB', 'absorb'),
        ('EXPANDING', 'expanding'),
    ]:
        print(f"\n--- {label} ---")
        results[mode] = _run_simulation(label, bc_mode=mode)

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
        ax.set_title("Shape Relaxation: Outer-Phase BC Comparison")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        fig.savefig(os.path.join(_FIG, 'bc_comparison_circularity.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 2. Radii
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.plot(res['t'] * 1000, res['R_max'] * 1000, color=c,
                    linestyle=ls, label=f'{lbl} R_max')
            ax.plot(res['t'] * 1000, res['R_min'] * 1000, color=c,
                    linestyle='--', alpha=0.5)
        ax.axhline(R_EQ * 1000, color='k', linestyle=':', alpha=0.5,
                    label=f'R_eq={R_EQ * 1000:.2f} mm')
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Radius [mm]")
        ax.set_title("Interface Radii: Outer-Phase BC Comparison")
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(os.path.join(_FIG, 'bc_comparison_radii.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 3. Pressure jump (dP - Laplace)
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.plot(res['t'] * 1000, res['P_drop'] - res['P_outer'],
                    color=c, linestyle=ls, label=lbl)
        ax.axhline(laplace_dp, color='k', linestyle=':', alpha=0.6,
                    label=f'Laplace dP={laplace_dp:.3f}')
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("P_drop - P_outer [Pa]")
        ax.set_title("Pressure Jump: Outer-Phase BC Comparison")
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(_FIG, 'bc_comparison_pressure.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 4. Pressure std dev
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.semilogy(res['t'] * 1000, np.maximum(res['p_std'], 1e-30),
                        color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Pressure Std Dev [Pa]")
        ax.set_title("Pressure Stability (Wave Damping): BC Comparison")
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(_FIG, 'bc_comparison_pressure_std.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 5. Mass conservation
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            M0 = res['mass'][0]
            ax.plot(res['t'] * 1000,
                    (res['mass'] - M0) / M0,
                    color=c, linestyle=ls, label=lbl)
        ax.axhline(0, color='k', linestyle=':', alpha=0.4)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Relative mass change (dM / M0)")
        ax.set_title("Mass Flux at Reservoir: BC Comparison")
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(_FIG, 'bc_comparison_mass.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 6. Kinetic energy
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode, res in results.items():
            c, ls, lbl = _style_for(mode)
            ax.semilogy(res['t'] * 1000, np.maximum(res['KE'], 1e-30),
                        color=c, linestyle=ls, label=lbl)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Kinetic Energy [J]")
        ax.set_title("Energy: Outer-Phase BC Comparison")
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(_FIG, 'bc_comparison_ke.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\nBC comparison plots saved to {_FIG}/")

    except ImportError:
        print("matplotlib not available — skipping plots")

    # -- Animations for RESERVOIR and EXPANDING --
    for mode in ('reservoir', 'expanding'):
        history = results[mode].get('history')
        if history is None or history.n_snapshots <= 1:
            continue
        try:
            HC_r = results[mode]['HC']
            bV_r = results[mode]['bV']
            anim = dynamic_plot_fluid(
                history, HC_r, bV=bV_r,
                save_path=os.path.join(
                    _FIG, f'cube2droplet_2D_bc_{mode}.mp4'
                ),
                fps=20, dpi=100,
                xlim=(-ZOOM, ZOOM),
                ylim=(-ZOOM, ZOOM),
                phase_field='phase',
                interface_field='is_interface',
                reference_R=R_EQ,
            )
            print(f"Animation saved: fig/cube2droplet_2D_bc_{mode}.mp4")
        except Exception as e:
            print(f"Animation failed for {mode}: {e}")

    # -- Summary table --
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Mode':<32} {'Wall [s]':>10} {'Final circ':>12} "
          f"{'|dM/M0|':>12} {'Final dP':>12}")
    print("-" * 82)
    for mode, res in results.items():
        _, _, lbl = _style_for(mode)
        M0 = res['mass'][0]
        dM = abs(res['mass'][-1] - M0) / M0
        dP = res['P_drop'][-1] - res['P_outer'][-1]
        print(f"{lbl:<32} {res['wall_time']:>10.1f} "
              f"{res['circ'][-1]:>12.4f} {dM:>12.2e} "
              f"{dP:>12.3f}")
    print(f"\nLaplace dP reference: {laplace_dp:.3f} Pa")
    print(f"Acoustic timescale tau_ac = L/c_s = {TAU_ACOUSTIC:.4f} s "
          f"= {TAU_ACOUSTIC * 1000:.1f} ms")

    print("\nDone.")


if __name__ == '__main__':
    main()
