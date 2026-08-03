"""Short-window 2D driver for producing a demo video.

Runs the 2D setup with a stricter CFL and a ~0.05 s physical window
(roughly 1/4 of a shear period) so the droplet visibly tilts without
the full simulation time.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.shearing_plate_droplet.src._params import (
    R0, L_x, L_y, U_wall, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    Ca, visc_ratio, Re, shear_rate,
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
    print("2D Shearing-Plate Droplet (SHORT WINDOW)")
    print("=" * 64)
    print(f"Ca = {Ca:.3f}, Re = {Re:.3f}, lambda = {visc_ratio:.3f}")
    D_taylor = taylor_deformation(Ca, visc_ratio)
    print(f"Taylor small-D prediction: D = {D_taylor:.4f}")

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params = \
        setup_shearing_plate_droplet(
            dim=dim, R0=R0, L_x=L_x, L_y=L_y,
            U_wall=U_wall, rho_d=rho_d, rho_o=rho_o,
            mu_d=mu_d, mu_o=mu_o, gamma=gamma,
            K_d=K_d, K_o=K_o,
            refinement_outer=3,
            refinement_droplet=3,
        )
    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"Mesh: {n_verts} vertices, {n_iface} interface, bV={len(bV)}")

    c_s = float(np.sqrt(K_o / rho_o))
    dx_min = min(float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15)
    dt = 0.1 * dx_min / c_s
    t_end = 0.05   # ~1/4 shear period
    n_steps = int(t_end / dt) + 1
    record_every = max(1, n_steps // 80)
    print(f"dt={dt:.2e} s, n_steps={n_steps}, t_end={t_end} s, "
          f"record_every={record_every}")

    os.makedirs(_SNAPSHOTS, exist_ok=True)
    os.makedirs(_FIG, exist_ok=True)

    history = StateHistory(
        fields=['u', 'p', 'phase', 'is_interface'],
        record_every=record_every, save_dir=_SNAPSHOTS,
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
            if step % (record_every * 5) == 0:
                d = diag_list[-1]
                print(f"  step={step} t={t:.3e} D={d['D']:.3f} "
                      f"tilt={np.degrees(d['tilt']):.0f}° "
                      f"KE={d['KE']:.2e} mass={d['total_mass']:.4e}")

    print("Running...")
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

    print(f"\nFinal D={D_arr[-1]:.3f} (Taylor={D_taylor:.3f})")
    print(f"mass drift = {abs(mass_arr[-1]-mass_arr[0])/mass_arr[0]:.3e}")
    print(f"snapshots recorded = {history.n_snapshots}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_fluid(HC, bV=bV, t=t_final,
        save_path=os.path.join(_FIG, 'shearing_plate_droplet_2D_fluid.png'))

    fig, _ = plot_velocity_profile(HC, U_wall=U_wall, L_y=L_y, dim=dim,
        n_bins=24, title=f'Mean $u_x(y)$ at t={t_final:.3f} s',
        save_path=os.path.join(_FIG, 'shearing_plate_droplet_2D_profile.png'))
    plt.close(fig)

    fig, _ = plot_deformation_history(t_arr, D_arr, tilt_arr, Ca=Ca,
        D_taylor=D_taylor,
        save_path=os.path.join(_FIG,
            'shearing_plate_droplet_2D_deformation.png'))
    plt.close(fig)

    if history.n_snapshots > 1:
        dynamic_plot_fluid(history, HC, bV=bV,
            save_path=os.path.join(_FIG,
                'shearing_plate_droplet_2D.mp4'),
            fps=20, dpi=100,
            xlim=(-L_x, L_x), ylim=(-L_y, L_y),
            phase_field='phase', interface_field='is_interface',
            reference_R=R0)
        print(f"Animation saved -> fig/shearing_plate_droplet_2D.mp4")


if __name__ == '__main__':
    main()
