#!/usr/bin/env python3
"""2D dam break — liquid only with an implicit atmospheric free surface.

The mesh covers ONLY the water column (``col_w x col_h``).  The
bottom and left faces are frozen no-slip walls; the top and right
faces are free surfaces whose vertices advect freely under gravity.

There is no explicit air phase; the absolute pressure is tracked via
the Tait–Murnaghan EOS and initialised from the hydrostatic profile
``P(y) = P_atm + rho_l * g * (col_h - y)``.  At a free surface this
collapses to ``P_atm`` and the compressibility of the liquid carries
the surface inwards — if surface tension were active it would add
``gamma * kappa`` to the boundary pressure.

Produces:
  fig/dam_break_2D_no_air_fluid.png
  fig/dam_break_2D_no_air.mp4
  results/snapshots_2D_no_air/

Usage
-----
    python cases_dynamic/dam_break/dam_break_2D_no_air.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(line_buffering=True)

from cases_dynamic.dam_break.src._params import (
    a, col_w, col_h, col_d,
    rho_l, mu_l, K_l,
    g, gravity_axis, P_atm, t_end, cfl, n_refine_2d, alpha_art,
)
from cases_dynamic.dam_break.src._setup import (
    setup_dam_break_single_phase, cfl_timestep,
)
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid
from ddgclib.visualization.unified import plot_fluid

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results')
_SNAPSHOTS = os.path.join(_RESULTS, 'snapshots_2D_no_air')


def main():
    dim = 2
    print("=" * 60)
    print("2D Dam Break — Single phase (liquid only, free surface)")
    print("=" * 60)

    print("\nBuilding mesh...")
    HC, bV, bc_set, dudt_fn, params = setup_dam_break_single_phase(
        dim=dim, a=a,
        col_w=col_w, col_h=col_h, col_d=col_d,
        rho_l=rho_l, mu_l=mu_l, K_l=K_l,
        g=g, gravity_axis=gravity_axis, P_atm=P_atm,
        n_refine=n_refine_2d, alpha_art=alpha_art,
    )
    n_verts = sum(1 for _ in HC.V)
    n_free = len(params['free_face'])
    print(f"Mesh: {n_verts} vertices "
          f"({len(bV)} frozen walls, {n_free} free-surface)")
    print(f"Column: {col_w:.3f} x {col_h:.3f} m")

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
        fields=['u', 'p'],
        record_every=record_every,
        save_dir=_SNAPSHOTS,
    )

    KE_hist, t_hist = [], []

    def _diag(HC_cb):
        ke = 0.0
        for v in HC_cb.V:
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
                      f"KE={ke:.4e}  |u|_max={u_max:.3f}")

    print("\nRunning simulation...")
    # boundary_filter picks wall vertices to freeze; topological
    # boundary vertices without ``is_wall`` (free-surface) are still
    # tagged by _retopologize for compute_vd but are NOT frozen, so
    # they advect under gravity.
    def _wall_filter(v):
        return bool(getattr(v, 'is_wall', False))

    try:
        t_final = symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
            bc_set=bc_set, callback=callback,
            boundary_filter=_wall_filter,
        )
    except Exception as e:
        t_final = 0.0
        print(f"  integrator aborted: {e}")
        print(f"  recorded {history.n_snapshots} snapshots before abort")
    print(f"Simulation finished at t={t_final:.4f} s, "
          f"snapshots recorded: {history.n_snapshots}")

    # -- Static plot --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        try:
            plot_fluid(
                HC, bV=bV,
                save_path=os.path.join(_FIG, 'dam_break_2D_no_air_fluid.png'),
            )
            plt.close('all')
        except Exception as e:
            print(f"plot_fluid failed: {e}")

        if len(t_hist) > 1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(np.array(t_hist), np.array(KE_hist), 'b-')
            ax.set_xlabel('t [s]')
            ax.set_ylabel('KE [J]')
            ax.set_title('2D Dam Break (no air) — kinetic energy')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(_FIG, 'dam_break_2D_no_air_energy.png'),
                dpi=150,
            )
            plt.close(fig)
        print("Static plots saved to fig/")
    except ImportError:
        pass

    # -- Animation --
    if history.n_snapshots > 1:
        try:
            dynamic_plot_fluid(
                history, HC, bV=bV,
                save_path=os.path.join(_FIG, 'dam_break_2D_no_air.mp4'),
                fps=20, dpi=100,
                xlim=(0.0, 1.1 * col_w),
                ylim=(0.0, 1.1 * col_h),
            )
            print(f"Animation saved to fig/dam_break_2D_no_air.mp4")
        except Exception as e:
            print(f"Animation failed: {e}")

    print(f"\nTo view in polyscope:")
    print(f"  python -m ddgclib.scripts.view_polyscope "
          f"--snapshots {os.path.relpath(_SNAPSHOTS)}")
    print("Done.")


if __name__ == '__main__':
    main()
