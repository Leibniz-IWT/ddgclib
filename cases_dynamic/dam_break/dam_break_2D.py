#!/usr/bin/env python3
"""2D dam break — multiphase (liquid + air) with surface tension.

A rectangular tank of size ``L x H`` is initialised with a liquid
column of size ``col_w x col_h`` in the lower-left corner; the rest
of the tank is filled with air.  Both fluids start in a hydrostatic
pressure state and are released at t=0.  Gravity drives the collapse;
the liquid–air interface carries a surface tension ``gamma``.

Produces:
  fig/dam_break_2D_fluid.png        — final pressure + velocity
  fig/dam_break_2D_phases.png       — final phase field
  fig/dam_break_2D.mp4              — animation with interface overlay
  results/snapshots/                — JSON snapshots for polyscope

Usage
-----
    python cases_dynamic/dam_break/dam_break_2D.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(line_buffering=True)

from cases_dynamic.dam_break.src._params import (
    a, L, H, W, col_w, col_h, col_d,
    rho_l, rho_g, mu_l, mu_g, gamma, K_l, K_g,
    g, gravity_axis, P_atm, t_end, cfl, n_refine_2d, alpha_art,
)
from cases_dynamic.dam_break.src._setup import (
    setup_dam_break_multiphase, cfl_timestep,
)
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid
from ddgclib.visualization.unified import plot_fluid

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results')
_SNAPSHOTS = os.path.join(_RESULTS, 'snapshots_2D')


def main():
    dim = 2
    print("=" * 60)
    print("2D Dam Break — Multiphase (liquid + air)")
    print("=" * 60)

    print("\nBuilding mesh...")
    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_dam_break_multiphase(
            dim=dim, a=a, L=L, H=H, W=W,
            col_w=col_w, col_h=col_h, col_d=col_d,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
            gamma=gamma, K_l=K_l, K_g=K_g,
            g=g, gravity_axis=gravity_axis, P_atm=P_atm,
            n_refine=n_refine_2d, alpha_art=alpha_art,
        )
    n_verts = sum(1 for _ in HC.V)
    n_liq = sum(1 for v in HC.V if v.phase == 1)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"Mesh: {n_verts} vertices ({n_liq} liquid, {n_iface} interface)")
    print(f"Tank: L={L:.3f} m x H={H:.3f} m, column: "
          f"{col_w:.3f} x {col_h:.3f} m")

    # -- CFL timestep --
    c_s = np.sqrt(K_l / rho_l)
    dt = cfl_timestep(HC, dim, c_s, cfl=cfl)
    n_steps = int(t_end / dt) + 1
    record_every = max(1, n_steps // 150)
    print(f"c_s={c_s:.1f} m/s, dt={dt:.2e} s, n_steps={n_steps}, "
          f"t_end={t_end:.3f} s")

    os.makedirs(_SNAPSHOTS, exist_ok=True)
    os.makedirs(_FIG, exist_ok=True)

    history = StateHistory(
        fields=['u', 'p', 'phase', 'is_interface'],
        record_every=record_every,
        save_dir=_SNAPSHOTS,
    )

    KE_hist, t_hist = [], []

    def _diag(HC_cb):
        ke = 0.0
        for v in HC_cb.V:
            if v.phase == 1:
                ke += 0.5 * v.m * float(np.dot(v.u[:dim], v.u[:dim]))
        return ke

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            ke = _diag(HC_cb)
            KE_hist.append(ke)
            t_hist.append(t)
            if step % (record_every * 10) == 0:
                u_max = max(
                    (float(np.linalg.norm(v.u[:dim])) for v in HC_cb.V),
                    default=0.0,
                )
                print(f"  step {step:>6d}  t={t:.4f} s  "
                      f"KE_liq={ke:.4e}  |u|_max={u_max:.3f}")

    print("\nRunning simulation...")
    # ``skip_triangulation=True`` keeps the initial Delaunay connectivity
    # frozen and only recomputes duals as vertices move.  Without this,
    # full Delaunay retopologisation every step creates cross-phase edges
    # that destabilise the interface (see FEATURES.md / AMR remeshing).
    try:
        t_final = symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
            bc_set=bc_set, callback=callback, retopologize_fn=retopo_fn,
            skip_triangulation=True,
        )
    except Exception as e:
        t_final = 0.0
        print(f"  integrator aborted: {e}")
        print(f"  recorded {history.n_snapshots} snapshots before abort")
    print(f"Simulation finished at t={t_final:.4f} s, "
          f"snapshots recorded: {history.n_snapshots}")

    # -- Static plots --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        try:
            plot_fluid(
                HC, bV=bV,
                save_path=os.path.join(_FIG, 'dam_break_2D_fluid.png'),
            )
            plt.close('all')
        except Exception as e:
            print(f"plot_fluid failed: {e}")

        # Phase field snapshot
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = np.array([v.x_a[0] for v in HC.V])
        ys = np.array([v.x_a[1] for v in HC.V])
        phs = np.array([int(v.phase) for v in HC.V])
        ax.scatter(xs[phs == 0], ys[phs == 0], s=8, c='lightblue',
                   label='air')
        ax.scatter(xs[phs == 1], ys[phs == 1], s=10, c='navy',
                   label='water')
        iface = [v for v in HC.V if getattr(v, 'is_interface', False)]
        if iface:
            ax.scatter([v.x_a[0] for v in iface],
                       [v.x_a[1] for v in iface],
                       s=20, c='red', marker='o', label='interface')
        ax.set_xlim(0, L)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.set_title(f'2D Dam Break phases (t={t_final:.3f} s)')
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(_FIG, 'dam_break_2D_phases.png'), dpi=150)
        plt.close(fig)

        if len(t_hist) > 1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(np.array(t_hist), np.array(KE_hist), 'b-')
            ax.set_xlabel('t [s]')
            ax.set_ylabel('KE (liquid) [J]')
            ax.set_title('2D Dam Break — liquid kinetic energy')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(_FIG, 'dam_break_2D_energy.png'),
                        dpi=150)
            plt.close(fig)
        print("Static plots saved to fig/")
    except ImportError:
        pass

    # -- Animation --
    if history.n_snapshots > 1:
        try:
            dynamic_plot_fluid(
                history, HC, bV=bV,
                save_path=os.path.join(_FIG, 'dam_break_2D.mp4'),
                fps=20, dpi=100,
                xlim=(0.0, L), ylim=(0.0, H),
                phase_field='phase',
                interface_field='is_interface',
            )
            print(f"Animation saved to fig/dam_break_2D.mp4")
        except Exception as e:
            print(f"Animation failed: {e}")

    print(f"\nTo view in polyscope:")
    print(f"  python -m ddgclib.scripts.view_polyscope "
          f"--snapshots {os.path.relpath(_SNAPSHOTS)}")
    print("Done.")


if __name__ == '__main__':
    main()
