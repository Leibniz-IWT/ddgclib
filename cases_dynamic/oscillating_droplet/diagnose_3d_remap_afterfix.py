#!/usr/bin/env python3
"""laneI: 3D oscillating droplet, per-step Delaunay + conservative remap.

Mirrors ``oscillating_droplet_3D.py`` (same setup call, refine 2/2, CFL dt
formula, record cadence and ``oscillation_score_3d``) but runs the solver
configuration

    PRESETS['oscillating_droplet_3D_delaunay'].replace(remap='conservative')

which the 3D runner has no CLI for.  Written after the interface apex-cache
invalidation fix (audit 2026-09-25 F1) to re-measure laneE's pre-fix 3D
delaunay+remap A/B (l2 1.87348 / tail 0.25327).  No figures / animation.

Writes
  results_3d/score_delaunay_remap_afterfix.json
  results_3d/methods_delaunay_remap_afterfix.json
  results_3d/snapshots_delaunay_remap_afterfix/

Usage
-----
    python cases_dynamic/oscillating_droplet/diagnose_3d_remap_afterfix.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, epsilon, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, t_end_3d,
)
from cases_dynamic.oscillating_droplet.src._analytical import (
    rayleigh_frequency, lamb_damping_rate,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    compute_diagnostics,
)
from cases_dynamic.oscillating_droplet.src._metrics import (
    oscillation_score_3d, save_score,
)
from ddgclib.data import StateHistory
from ddgclib.methods import PRESETS, record_methods

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_CASE_DIR, 'results_3d')
_TAG = 'delaunay_remap_afterfix'


def main():
    dim = 3
    methods = PRESETS['oscillating_droplet_3D_delaunay'].replace(
        remap='conservative',
        label='cases_dynamic/oscillating_droplet/diagnose_3d_remap_afterfix.py',
        notes='laneI: 3D delaunay + conservative remap after the '
              'interface apex-cache invalidation fix (audit F1)',
    )
    print(methods.describe())

    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim, rho_outer=rho_o)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)

    HC, bV, mps, bc_set, dudt_fn, _setup_retopo_fn, _params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=2,
            refinement_droplet=2,
            methods=methods,
        )
    print(f"Mesh: {sum(1 for _ in HC.V)} vertices, "
          f"{sum(1 for v in HC.V if getattr(v, 'is_interface', False))} "
          f"interface")

    c_s = np.sqrt(K_d / rho_d)
    dx_min = min(
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min**3 / gamma) if gamma > 0 else 1.0)
    t_end = min(t_end_3d, 5.0 / beta if beta > 0 else 0.01)
    n_steps = int(t_end / dt) + 1
    record_every = max(1, n_steps // 100)
    print(f"dt={dt:.2e}, n_steps={n_steps}, t_end={t_end:.4f}")

    snapshots_dir = os.path.join(_RESULTS, f'snapshots_{_TAG}')
    os.makedirs(snapshots_dir, exist_ok=True)
    history = StateHistory(
        fields=['u', 'p', 'phase', 'is_interface'],
        record_every=record_every,
        save_dir=snapshots_dir,
    )

    diag_list: list[dict] = []

    def record(t):
        d = compute_diagnostics(HC, dim=dim, polar_axis='z')
        d['t'] = float(t)
        d['total_dual_vol'] = float(sum(
            float(getattr(v, 'dual_vol', 0.0) or 0.0) for v in HC.V
        ))
        diag_list.append(d)

    record(0.0)

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            record(t)
            if step % (record_every * 10) == 0:
                d = diag_list[-1]
                print(f"  step={step} t={t:.4e} | R_max={d['R_max']:.6f} | "
                      f"KE={d['KE']:.4e} | mass={d['total_mass']:.6e}",
                      flush=True)

    t0 = time.perf_counter()
    t_final = methods.integrate(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
        bc_set=bc_set, callback=callback, mps=mps,
    )
    wall = time.perf_counter() - t0
    record(t_final)

    score = oscillation_score_3d(
        diag_list, R0=R0, epsilon=epsilon, l=l, omega=omega, beta=beta,
        r_boundary=L_domain,
    )
    score['refinement_outer'] = 2
    score['refinement_droplet'] = 2
    score['retopo_policy'] = 'delaunay_remap'
    score['wall_time_s'] = wall
    score_path = os.path.join(_RESULTS, f'score_{_TAG}.json')
    save_score(score_path, score)
    record_methods(
        os.path.join(_RESULTS, f'methods_{_TAG}.json'), methods, HC,
        extra={'retopo_policy': 'delaunay_remap', 'dt': dt,
               'n_steps': n_steps, 't_end': t_end,
               'refinement_outer': 2, 'refinement_droplet': 2,
               'K_d': K_d, 'K_o': K_o, 'wall_time_s': wall},
    )
    print(f"\nscore saved to {score_path}")
    for k in ('l2_error_normalized', 'tail_growth', 'mass_drift',
              'R_max_peak', 'boundary_saturation', 'n_interface_min',
              'n_interface_max'):
        print(f"  {k:24s} = {score[k]}")
    print(f"  wall_time_s              = {wall:.1f}")


if __name__ == '__main__':
    main()
