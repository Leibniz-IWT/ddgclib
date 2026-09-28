#!/usr/bin/env python3
"""2D shearing-plate droplet simulation.

Oil droplet suspended in water between two counter-moving plates.
Top plate moves at +U_wall, bottom plate at -U_wall, driving a
far-field simple-shear flow about ``y = 0``.  Streamwise boundaries
are periodic.

Outputs (in this directory):
  fig/shearing_plate_droplet_2D_fluid.png       — final pressure + velocity
  fig/shearing_plate_droplet_2D_phases.png      — final phase field
  fig/shearing_plate_droplet_2D_profile.png     — mean u_x(y) vs Couette
  fig/shearing_plate_droplet_2D_deformation.png — D(t), tilt(t)
  fig/shearing_plate_droplet_2D.mp4             — animation
  results/snapshots/                             — JSON StateHistory dumps

Usage
-----
    python cases_dynamic/shearing_plate_droplet/shearing_plate_droplet_2D.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.shearing_plate_droplet.src._params import (
    R0, L_x, L_y, U_wall, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    n_refine_outer, n_refine_droplet, t_end_default, Ca, visc_ratio, Re,
    shear_rate,
)
from cases_dynamic.shearing_plate_droplet.src._setup import (
    setup_shearing_plate_droplet,
)
from cases_dynamic.shearing_plate_droplet.src._analytical import (
    taylor_deformation,
)
from cases_dynamic.shearing_plate_droplet.src._plot_helpers import (
    compute_diagnostics, plot_velocity_profile, plot_deformation_history,
)
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid, plot_fluid

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results')
_SNAPSHOTS = os.path.join(_RESULTS, 'snapshots')


def main():
    dim = 2
    print("=" * 64)
    print("2D Shearing-Plate Droplet")
    print("=" * 64)
    print(f"Ca   = {Ca:.3f}  (mu_o gamma_dot R0 / gamma)")
    print(f"Re   = {Re:.3f}  (rho_o gamma_dot R0^2 / mu_o)")
    print(f"lambda = mu_d/mu_o = {visc_ratio:.3f}")
    print(f"U_wall = {U_wall} m/s,  shear_rate = {shear_rate:.2f} 1/s")
    D_taylor = taylor_deformation(Ca, visc_ratio)
    print(f"Taylor small-D prediction: D = {D_taylor:.4f}")

    print("\nBuilding mesh...")
    HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params = \
        setup_shearing_plate_droplet(
            dim=dim, R0=R0, L_x=L_x, L_y=L_y,
            U_wall=U_wall, rho_d=rho_d, rho_o=rho_o,
            mu_d=mu_d, mu_o=mu_o, gamma=gamma,
            K_d=K_d, K_o=K_o,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
        )
    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"Mesh: {n_verts} vertices, {n_iface} interface, "
          f"{len(groups['top_wall'])} top plate, "
          f"{len(groups['bottom_wall'])} bottom plate")

    # -- CFL timestep --
    c_s = float(np.sqrt(K_o / rho_o))
    dx_min = min(
        float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt_acoustic = 0.25 * dx_min / c_s
    dt_cap = 0.5 * np.sqrt(rho_o * dx_min ** 3 / gamma) if gamma > 0 else np.inf
    dt_visc = 0.25 * rho_o * dx_min ** 2 / max(mu_o, mu_d) if max(mu_o, mu_d) > 0 else np.inf
    dt = min(dt_acoustic, dt_cap, dt_visc)
    t_end = t_end_default
    n_steps = int(t_end / dt) + 1
    record_every = max(1, n_steps // 200)
    print(f"dt = {dt:.3e} s, n_steps = {n_steps}, t_end = {t_end:.4f} s")
    print(f"  dt bounds: acoustic={dt_acoustic:.2e}, "
          f"capillary={dt_cap:.2e}, viscous={dt_visc:.2e}")

    # -- Output directories --
    os.makedirs(_SNAPSHOTS, exist_ok=True)
    os.makedirs(_FIG, exist_ok=True)

    # -- StateHistory (multiphase snapshots) --
    history = StateHistory(
        fields=['u', 'p', 'phase', 'is_interface'],
        record_every=record_every,
        save_dir=_SNAPSHOTS,
    )

    diag_list: list[dict] = []

    def record(t):
        d = compute_diagnostics(HC, dim=dim)
        d['t'] = float(t)
        diag_list.append(d)

    record(0.0)

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            record(t)
            if step % (record_every * 10) == 0:
                d = diag_list[-1]
                print(f"  t={t:.4e} | D={d['D']:.4f} | "
                      f"tilt={np.degrees(d['tilt']):.1f}° | "
                      f"KE={d['KE']:.3e} | u_max={d['u_max']:.3e} | "
                      f"mass={d['total_mass']:.6e}")

    print("\nRunning simulation...")
    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=callback, retopologize_fn=retopo_fn,
        remesh_mode=params['remesh_mode'],
        remesh_kwargs=params['remesh_kwargs'],
    )
    record(t_final)

    t_arr = np.array([d['t'] for d in diag_list])
    D_arr = np.array([d['D'] for d in diag_list])
    tilt_arr = np.array([d['tilt'] for d in diag_list])
    mass_arr = np.array([d['total_mass'] for d in diag_list])

    print(f"\nFinal deformation:  D = {D_arr[-1]:.4f}")
    print(f"Taylor prediction:  D = {D_taylor:.4f}")
    print(f"Final tilt angle:   {np.degrees(tilt_arr[-1]):.2f}°")
    print(f"Mass conservation:  |dM/M0| = "
          f"{abs(mass_arr[-1] - mass_arr[0]) / mass_arr[0]:.3e}")
    print(f"Snapshots recorded: {history.n_snapshots}")

    # -- Static plots --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plot_fluid(
            HC, bV=bV, t=t_final,
            save_path=os.path.join(_FIG, 'shearing_plate_droplet_2D_fluid.png'),
        )

        fig, ax = plot_velocity_profile(
            HC, U_wall=U_wall, L_y=L_y, dim=dim, n_bins=24,
            title=f'Mean $u_x(y)$ at t = {t_final:.3f} s',
            save_path=os.path.join(_FIG,
                                    'shearing_plate_droplet_2D_profile.png'),
        )
        plt.close(fig)

        fig, _ = plot_deformation_history(
            t_arr, D_arr, tilt_arr, Ca=Ca, D_taylor=D_taylor,
            save_path=os.path.join(
                _FIG, 'shearing_plate_droplet_2D_deformation.png'),
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
                save_path=os.path.join(_FIG,
                                        'shearing_plate_droplet_2D.mp4'),
                fps=20, dpi=100,
                xlim=(-L_x, L_x),
                ylim=(-L_y, L_y),
                phase_field='phase',
                interface_field='is_interface',
                reference_R=R0,
            )
            print(f"Animation saved to fig/shearing_plate_droplet_2D.mp4")
        except Exception as e:
            print(f"Animation failed: {e}")

    print(f"\nTo view in polyscope:")
    print(f"  python -m ddgclib.scripts.view_polyscope "
          f"--snapshots {os.path.relpath(_SNAPSHOTS)}")
    print("Done.")


if __name__ == '__main__':
    main()
