#!/usr/bin/env python3
"""2D electrolysis hydrogen-bubble dynamic simulation.

Proof-of-concept case: a gas bubble grows on a flat electrode (bottom
wall) under a constant linear mass-generation rate (placeholder for a
real electrochemical reaction).  Gravity acts on all masses; the
compressible EOS carries the hydrostatic pressure field.  The bubble
grows until buoyancy overcomes the surface-tension / wall attraction
and the bubble detaches.

Produces:
  fig/electrolysis_bubble_2D.mp4       -- animation (p + u + interface)
  fig/electrolysis_bubble_2D_R.png     -- R_eq(t) vs Fritz detachment
  fig/electrolysis_bubble_2D_z.png     -- bubble centroid height vs t
  results/snapshots/                   -- JSON snapshots for polyscope

Usage
-----
    python cases_dynamic/electrolysis_bubble/electrolysis_bubble_2D.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.electrolysis_bubble.src._params import (
    R0, L_domain, nucleation_frac, rho_liq, rho_gas, mu_liq, mu_gas,
    gamma, K_liq, K_gas, g, P0,
    n_refine_outer_2d, n_refine_drop_2d, t_end_2d, dm_dt_2d,
    cfl_safety,
)
from cases_dynamic.electrolysis_bubble.src._setup import (
    setup_electrolysis_bubble,
)
from cases_dynamic.electrolysis_bubble.src._reaction import inject_gas_mass
from cases_dynamic.electrolysis_bubble.src._analytical import (
    capillary_length, fritz_detachment_radius, bond_number,
    young_laplace_jump,
)
from cases_dynamic.electrolysis_bubble.src._plot_helpers import (
    compute_diagnostics, plot_radius_history, plot_centroid_history,
)

from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid


_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results')
_SNAPSHOTS = os.path.join(_RESULTS, 'snapshots')


def main():
    dim = 2
    os.makedirs(_FIG, exist_ok=True)
    os.makedirs(_SNAPSHOTS, exist_ok=True)

    print("=" * 60)
    print("2D Electrolysis Hydrogen Bubble -- proof of concept")
    print("=" * 60)

    rho_diff = rho_liq - rho_gas
    lam = capillary_length(gamma, rho_diff, g)
    R_fritz = fritz_detachment_radius(R0, gamma, rho_diff, g)
    Bo0 = bond_number(R0, gamma, rho_diff, g)
    dP_lap = young_laplace_jump(gamma, R0, dim=dim)
    print(f"Capillary length  lambda = {lam*1e3:.3f} mm")
    print(f"Fritz detachment  R_det  = {R_fritz*1e3:.3f} mm "
          f"(initial R0 = {R0*1e3:.3f} mm)")
    print(f"Initial Bond num  Bo(R0) = {Bo0:.3e}")
    print(f"Initial Laplace jump     = {dP_lap:.2f} Pa")

    # -- Setup -------------------------------------------------------------
    print("\nBuilding mesh...")
    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_electrolysis_bubble(
            dim=dim, R0=R0, L_domain=L_domain,
            nucleation_frac=nucleation_frac,
            rho_liq=rho_liq, rho_gas=rho_gas,
            mu_liq=mu_liq, mu_gas=mu_gas,
            gamma=gamma, K_liq=K_liq, K_gas=K_gas,
            g=g, P0=P0,
            refinement_outer=n_refine_outer_2d,
            refinement_droplet=n_refine_drop_2d,
        )
    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"Mesh: {n_verts} vertices, {n_iface} interface")

    # -- CFL time step ----------------------------------------------------
    c_s_liq = np.sqrt(K_liq / rho_liq)
    c_s_gas = np.sqrt(K_gas / rho_gas)
    c_s = max(c_s_liq, c_s_gas)
    dx_min = min(
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt_cfl = cfl_safety * dx_min / c_s
    dt_st = 0.4 * np.sqrt(rho_liq * dx_min**3 / gamma)
    dt = min(dt_cfl, dt_st)
    n_steps = int(t_end_2d / dt) + 1
    record_every = max(1, n_steps // 200)
    print(f"c_s = max({c_s_liq:.2f}, {c_s_gas:.2f}) = {c_s:.2f} m/s")
    print(f"dx_min = {dx_min:.3e}, dt_cfl = {dt_cfl:.2e}, "
          f"dt_st = {dt_st:.2e}")
    print(f"dt = {dt:.2e}, n_steps = {n_steps}, t_end = {t_end_2d}")

    # -- Recording --------------------------------------------------------
    history = StateHistory(
        fields=['u', 'p', 'phase', 'is_interface'],
        record_every=record_every,
        save_dir=_SNAPSHOTS,
    )
    diag_list: list[dict] = []
    electrode_level = -L_domain

    def record(t):
        d = compute_diagnostics(HC, dim=dim,
                                 electrode_level=electrode_level)
        d['t'] = float(t)
        diag_list.append(d)

    record(0.0)
    z_com_0 = float(diag_list[0]['gas_com'][dim - 1])
    print(f"Initial gas volume: {diag_list[-1]['gas_volume']:.3e} m^2, "
          f"R_eq={diag_list[-1]['R_eq']*1e3:.3f} mm, z0={z_com_0*1e3:.3f} mm")

    # -- Callback: log + inject gas mass ---------------------------------
    detached_step = [None]

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)

        # Gas generation placeholder: add mass proportional to dt.
        inject_gas_mass(HC_cb, mps, dm_dt=dm_dt_2d, dt=dt, gas_phase=1)

        if step % record_every == 0:
            record(t)
            d = diag_list[-1]
            # Stop printing once the bubble has fully dissolved (the
            # interface is gone; every field is trivially zero).
            if step % (record_every * 5) == 0 and d['gas_mass'] > 1e-12:
                print(
                    f"  step={step:5d} t={t:.3e} | "
                    f"R_eq={d['R_eq']*1e3:.3f} mm | "
                    f"z_com={d['gas_com'][dim-1]*1e3:+.3f} mm | "
                    f"min_iface_z={d['min_iface_z']*1e3:+.3f} mm | "
                    f"KE={d['KE']:.2e} | M_gas={d['gas_mass']:.3e}"
                )
            # Detachment: bubble centroid has risen significantly
            # from the nucleation point due to buoyancy exceeding the
            # resistance of surrounding liquid + surface tension.
            dz = float(d['gas_com'][dim - 1]) - z_com_0
            if dz > 0.5 * R0 and detached_step[0] is None:
                detached_step[0] = step
                print(f"  ** RISING (nascent detachment) at step {step}, "
                      f"t={t:.4e}s, R_eq={d['R_eq']*1e3:.3f} mm, "
                      f"dz={dz*1e3:+.3f} mm **")

    # -- Run --------------------------------------------------------------
    print("\nRunning simulation...")
    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=callback, retopologize_fn=retopo_fn,
        remesh_mode=params['remesh_mode'],
        remesh_kwargs=params['remesh_kwargs'],
    )
    record(t_final)
    print(f"\nSimulation finished at t = {t_final:.4f} s "
          f"(n_snapshots = {history.n_snapshots})")
    if detached_step[0] is None:
        print("  Bubble did not detach within the simulation window.")
    else:
        print(f"  Bubble detached at step {detached_step[0]}.")

    t_arr = np.array([d['t'] for d in diag_list])
    R_arr = np.array([d['R_eq'] for d in diag_list])
    z_arr = np.array([d['gas_com'][dim - 1] for d in diag_list])

    # -- Static plots -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_radius_history(t_arr, R_arr, R0, R_detach=R_fritz, ax=ax,
                            title="2D H2 bubble radius (electrolysis)")
        fig.savefig(os.path.join(_FIG, 'electrolysis_bubble_2D_R.png'),
                    dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_centroid_history(t_arr, z_arr, electrode_level,
                              R_eq_arr=R_arr, ax=ax,
                              title="2D bubble centroid rise")
        fig.savefig(os.path.join(_FIG, 'electrolysis_bubble_2D_z.png'),
                    dpi=150)
        plt.close(fig)
        print("Static plots saved under fig/")
    except ImportError:
        pass

    # -- Animation --------------------------------------------------------
    if history.n_snapshots > 1:
        try:
            dynamic_plot_fluid(
                history, HC, bV=bV,
                save_path=os.path.join(_FIG, 'electrolysis_bubble_2D.mp4'),
                fps=20, dpi=100,
                xlim=(-L_domain * 1.05, L_domain * 1.05),
                ylim=(-L_domain * 1.05, L_domain * 1.05),
                phase_field='phase',
                interface_field='is_interface',
            )
            print(f"Animation saved to "
                  f"fig/electrolysis_bubble_2D.mp4")
        except Exception as e:
            print(f"Animation failed: {e}")

    print("\nTo view in polyscope:")
    print(f"  python cases_dynamic/electrolysis_bubble/view_polyscope.py "
          f"--dim 2")
    print("Done.")


if __name__ == '__main__':
    main()
