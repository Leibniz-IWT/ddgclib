#!/usr/bin/env python3
"""3D oscillating droplet simulation.

Produces:
  fig/oscillating_droplet_3D_radius.png   — R_max(t)
  fig/oscillating_droplet_3D_energy.png   — KE(t)
  fig/oscillating_droplet_3D.mp4          — 3D animation (pressure + velocity)
  results_3d/snapshots/                   — JSON snapshots for polyscope
  results_3d/score.json                   — Tier 3B oscillation score

Usage
-----
    python cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py
    python cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py \
        --retopo delaunay    # A/B: per-step Delaunay retriangulation
                             # (non-default policies write suffixed
                             # artifacts, e.g. score_delaunay.json)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, epsilon, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, beta_3d, t_end_3d, retopo_policy_3d,
)
from cases_dynamic.oscillating_droplet.src._analytical import (
    rayleigh_frequency, lamb_damping_rate, damped_frequency,
    max_radius_envelope,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    plot_radius_envelope, plot_energy_history, compute_diagnostics,
)
from cases_dynamic.oscillating_droplet.src._metrics import (
    oscillation_score_3d, save_score,
)
from ddgclib.data import StateHistory
from ddgclib.methods import PRESETS, record_methods
from ddgclib.visualization import dynamic_plot_fluid

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')
_RESULTS = os.path.join(_CASE_DIR, 'results_3d')
_SNAPSHOTS = os.path.join(_RESULTS, 'snapshots')

# Retopology policy string -> solver-method preset (ddgclib.methods,
# documented in METHODS.md).
_POLICY_PRESETS = {
    'dual_only': 'oscillating_droplet_3D',
    'delaunay': 'oscillating_droplet_3D_delaunay',
}


def main(retopo_policy: str | None = None, box_shift: str = 'move_all'):
    dim = 3
    if retopo_policy is None:
        retopo_policy = retopo_policy_3d
    methods = PRESETS[_POLICY_PRESETS[retopo_policy]]
    print(methods.describe())
    print("=" * 60)
    print("3D Oscillating Droplet — Overdamped Case")
    print("=" * 60)

    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim, rho_outer=rho_o)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)
    omega_d = damped_frequency(omega, beta)
    regime = "underdamped" if omega_d > 0 else "overdamped"
    print(f"Mode l={l}: omega={omega:.2f}, beta={beta:.2f}, "
          f"omega_d={omega_d:.2f} ({regime})")

    # -- Setup (lower refinement for 3D) --
    # NOTE(laneG 2026-07-30, docs_temp/debug_session/
    # laneG-3d-inflation-gap.md): keep refine 2/2.  The droplet
    # "inflation" behind summary 0.24811 is the relaxation of the
    # coarse polyhedral interface toward its DISCRETE equilibrium
    # shape (cube-symmetry mode: the 6 valence-8 face-center vertices
    # ARE R_max), and it does NOT converge under droplet refinement
    # on this cube-sphere mesh family (eps=0 bump at matched t=0.056:
    # +9.1% of R0 at droplet refine 1, +2.0% at 2, +4.1% at 3 with a
    # monotone energy-pumping face-scale mode; outer refinement is a
    # no-op at the interface).  Do NOT flip refinement_droplet to 3
    # expecting an O(h^2) shrink: measured worse (see the lane log
    # scored table).  Also measured no-ops for this symptom: a
    # discrete-consistent scalar YL preload (see _setup.py step 4
    # note) and stencil variants (the Stokes conormal form was the
    # cotangent form to round-off on the moving mesh as well and was
    # removed in laneM, 2026-10-05; csf_dual is measured worse there).
    print("\nBuilding mesh...")
    HC, bV, mps, bc_set, dudt_fn, _setup_retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=epsilon, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=2,
            refinement_droplet=2,
            split_method=methods.split_method,
            redistribute_mass=methods.redistribute_mass,
            box_shift=box_shift,
            methods=methods,   # force axes (curvature_path, ...) bound
        )
    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"Mesh: {n_verts} vertices, {n_iface} interface "
          f"(box_shift={box_shift})")

    # -- Retopology policy (A/B via --retopo; default from _params) --
    # 'dual_only' freezes the builder connectivity (skip_triangulation)
    # while refreshing duals / per-phase splits / mass redistribution /
    # EOS pressures every step.  NOTE lane-5 caveat: the 3D
    # skip-triangulation path has different boundary-volume bookkeeping
    # (batch_e_star zeroing) — trust its score only after checking
    # mass_drift, n_interface_* and the dual_vol_* fields in the score.
    # The preset builds the partial the old dispatch built by hand.
    print(f"Retopo policy: {retopo_policy} -> preset "
          f"{_POLICY_PRESETS[retopo_policy]}")

    # Non-default policies write suffixed artifacts (snapshots, figures,
    # score) so an A/B run never clobbers the default baseline outputs.
    # laneB: the same for the lossy pre-laneB outer mesh (--box-shift evict).
    suffix = '' if retopo_policy == retopo_policy_3d else f'_{retopo_policy}'
    if box_shift != 'move_all':
        suffix += f'_{box_shift}'
    snapshots_dir = _SNAPSHOTS + suffix

    # -- CFL timestep --
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

    # -- Recording --
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(_FIG, exist_ok=True)

    history = StateHistory(
        fields=['u', 'p', 'phase', 'is_interface'],
        record_every=record_every,
        save_dir=snapshots_dir,
    )

    diag_list: list[dict] = []

    def record(t):
        # polar_axis='z': theta_apex measured from the perturbation
        # axis so the apex score in oscillation_score_3d is consistent
        # with _setup._apply_perturbation.
        d = compute_diagnostics(HC, dim=dim, polar_axis='z')
        d['t'] = float(t)
        # Total dual volume for the dual_only A/B bookkeeping check
        # (boundary cells are zeroed by _retopologize step 5b — the
        # frame-0 -> 1 jump is the known transition artefact).
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
                print(f"  t={t:.4e} | R_max={d['R_max']:.6f} | "
                      f"KE={d['KE']:.4e} | mass={d['total_mass']:.6e}")

    # -- Run --
    print("\nRunning simulation...")
    t_final = methods.integrate(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
        bc_set=bc_set, callback=callback, mps=mps,
    )

    record(t_final)

    t_arr = np.array([d['t'] for d in diag_list])
    R_max_arr = np.array([d['R_max'] for d in diag_list])
    KE_arr = np.array([d['KE'] for d in diag_list])
    mass_arr = np.array([d['total_mass'] for d in diag_list])
    R_max_analytical = max_radius_envelope(t_arr, R0, epsilon, omega, beta, l=l)

    print(f"\nMass conservation: |dM/M0| = "
          f"{abs(mass_arr[-1] - mass_arr[0]) / mass_arr[0]:.4e}")
    print(f"Final R_max: {R_max_arr[-1]:.6f} (R0 = {R0})")
    print(f"StateHistory: {history.n_snapshots} snapshots")

    # -- Score (Tier 3B harness) --
    score = oscillation_score_3d(
        diag_list, R0=R0, epsilon=epsilon, l=l, omega=omega, beta=beta,
        r_boundary=L_domain,
    )
    # Record the runner configuration in the score so score.json is
    # self-describing (laneG; baseline diffs ignore extra keys).
    score['refinement_outer'] = 2
    score['refinement_droplet'] = 2
    score['retopo_policy'] = retopo_policy
    score['box_shift'] = box_shift
    score_path = os.path.join(_RESULTS, f'score{suffix}.json')
    save_score(score_path, score, methods=methods)   # self-describing score
    record_methods(
        os.path.join(_RESULTS, f'methods{suffix}.json'), methods, HC,
        extra={'retopo_policy': retopo_policy, 'dt': dt,
               'n_steps': n_steps, 't_end': t_end,
               'refinement_outer': 2, 'refinement_droplet': 2,
               'box_shift': params['box_shift'],
               'K_d': K_d, 'K_o': K_o},
    )
    print(f"\n3D oscillation score saved to {score_path}")
    print(f"  summary                  = {score['summary']:.4e}")
    print(f"  l2_error_normalized      = {score['l2_error_normalized']:.4e}")
    print(f"  linf_error_normalized    = {score['linf_error_normalized']:.4e}")
    if score['apex_l2_error_normalized'] is not None:
        print(f"  apex_l2_error_normalized = "
              f"{score['apex_l2_error_normalized']:.4e}")
    print(f"  tail_growth              = {score['tail_growth']:.4e}")
    print(f"  mass_drift               = {score['mass_drift']:.4e}")
    print(f"  R_max_peak               = {score['R_max_peak']:.6f}")
    print(f"  boundary_saturation      = {score['boundary_saturation']}")
    if score['boundary_saturation']:
        print("  WARNING: R_max reached the outer mesh boundary — the "
              "trajectory is non-physical; do NOT read the errors above "
              "as a physics comparison.")

    # -- Static plots --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_radius_envelope(t_arr, R_max_arr, R_max_analytical, R0=R0, ax=ax,
                              title="3D Oscillating Droplet: R_max(t)")
        fig.savefig(os.path.join(
            _FIG, f'oscillating_droplet_3D_radius{suffix}.png'), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_energy_history(t_arr, np.array(KE_arr), ax=ax,
                             title="3D Oscillating Droplet: KE(t)")
        fig.savefig(os.path.join(
            _FIG, f'oscillating_droplet_3D_energy{suffix}.png'), dpi=150)
        plt.close(fig)
        print("Static plots saved to fig/")
    except ImportError:
        pass

    # -- 3D Animation --
    if history.n_snapshots > 1:
        try:
            anim = dynamic_plot_fluid(
                history, HC, bV=bV,
                save_path=os.path.join(
                    _FIG, f'oscillating_droplet_3D{suffix}.mp4'),
                fps=15, dpi=100,
                phase_field='phase',
                interface_field='is_interface',
            )
            print(f"Animation saved to fig/oscillating_droplet_3D{suffix}.mp4")
        except Exception as e:
            print(f"Animation failed: {e}")

    print(f"\nTo view in polyscope:")
    print(f"  python -m ddgclib.scripts.view_polyscope "
          f"--snapshots {os.path.relpath(snapshots_dir)}")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--retopo', choices=('delaunay', 'dual_only'),
        default=retopo_policy_3d,
        help="Retopology policy (default: retopo_policy_3d from "
             "src/_params.py). 'dual_only' skips per-step Delaunay "
             "retriangulation (frozen connectivity, duals refreshed).",
    )
    parser.add_argument(
        '--box-shift', choices=('move_all', 'evict'), default='move_all',
        dest='box_shift',
        help="Outer box shift of droplet_in_box_3d (laneB): 'move_all' "
             "keeps every outer vertex (default); 'evict' is the lossy "
             "pre-laneB mesh (3 of 189 outer vertices missing), written "
             "to suffixed artifacts",
    )
    args = parser.parse_args()
    main(retopo_policy=args.retopo, box_shift=args.box_shift)
