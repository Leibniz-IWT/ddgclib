#!/usr/bin/env python3
"""3D Cube-to-droplet relaxation simulation.

A cubic fluid droplet relaxes toward a spherical equilibrium shape
under surface tension.  Uses polyscope for 3D interactive visualization.

Produces:
  fig/cube2droplet_3D_sphericity.png — shape evolution plot
  fig/cube2droplet_3D_fluid.mp4      — matplotlib animation
  (optional) polyscope interactive viewer

Usage
-----
    python cases_dynamic/Cube2droplet/cube_to_droplet_3D.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.cube2droplet.src._setup import setup_cube_to_droplet
from ddgclib.analytical import integrated_phase_pressure_jump
from ddgclib.data import StateHistory
from ddgclib.methods import PRESETS, record_methods

# =====================================================================
# Parameters
# =====================================================================
R = 0.01
L_DOMAIN = 0.03
R_EQ = R * (6.0 / np.pi) ** (1 / 3)

RHO_D, RHO_O = 800.0, 1000.0
MU_D, MU_O = 2.0, 1.0
GAMMA = 0.01
K_D, K_O = 100.0, 125.0

N_REFINE = 2          # 3D: fewer refinements (many more vertices)
DT = 5e-5
N_STEPS = 2000
RECORD_EVERY = 20


def compute_diagnostics(HC, dim: int = 3):
    """Interface shape + pressure diagnostics."""
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

    return {
        'R_max': R_max, 'R_min': R_min,
        'sphericity': R_min / R_max if R_max > 0 else 0.0,
        'P_drop': float(np.mean(p_drop)) if p_drop else 0.0,
        'P_outer': float(np.mean(p_outer)) if p_outer else 0.0,
    }


# A/B arms of the shipped preset (laneX): 'bare' is the historic setup
# configuration (per-step Delaunay + redistribution, no remap), which
# loses the droplet's bulk by step 1000; 'dual_only' the fixed-connectivity
# refresh, on which the cube does not relax (laneX log 4.1).
ARMS = {
    'base': PRESETS['cube_to_droplet_3D'],
    'bare': PRESETS['cube_to_droplet_3D'].replace(remap=None),
    'dual_only': PRESETS['cube_to_droplet_3D_dual_only'],
}


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='3D cube-to-droplet relaxation')
    ap.add_argument('--n-steps', type=int, default=N_STEPS)
    ap.add_argument('--n-refine', type=int, default=N_REFINE)
    ap.add_argument('--no-anim', action='store_true',
                    help='skip the animation and the viewer (smoke runs)')
    ap.add_argument('--arm', choices=sorted(ARMS), default='base',
                    help='configuration arm (A/B through the preset)')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    n_steps, n_refine = args.n_steps, args.n_refine
    dim = 3
    print("=" * 60)
    print("3D Cube-to-Droplet Relaxation")
    print("=" * 60)
    print(f"R={R} m, R_eq={R_EQ:.5f} m, gamma={GAMMA}")

    # -- Setup (the preset is the single source of truth for the run) --
    print("\nBuilding mesh...")
    methods = ARMS[args.arm]
    print(f"arm={args.arm}: {methods.label}")
    HC, bV, mps, meos, bc_set, dudt_fn, _, params = \
        setup_cube_to_droplet(
            dim=dim, R=R, L_domain=L_DOMAIN,
            rho_d=RHO_D, rho_o=RHO_O, mu_d=MU_D, mu_o=MU_O,
            gamma=GAMMA, K_d=K_D, K_o=K_O, n_refine=n_refine,
            methods=methods,
        )

    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"Mesh: {n_verts} vertices, {n_iface} interface")

    # -- Recording --
    history = StateHistory(fields=('u', 'p'), record_every=RECORD_EVERY)
    history.append(0.0, HC)

    diag0 = compute_diagnostics(HC, dim)
    t_arr = [0.0]
    sph_arr = [diag0['sphericity']]
    pd_arr = [diag0['P_drop']]
    po_arr = [diag0['P_outer']]
    print(f"Initial sphericity: {diag0['sphericity']:.4f}")

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % 100 == 0:
            diag = compute_diagnostics(HC_cb, dim)
            t_arr.append(t)
            sph_arr.append(diag['sphericity'])
            pd_arr.append(diag['P_drop'])
            po_arr.append(diag['P_outer'])
        if step % 500 == 0 and step > 0:
            diag = compute_diagnostics(HC_cb, dim)
            dp = diag['P_drop'] - diag['P_outer']
            max_u = max(np.linalg.norm(v.u[:dim]) for v in HC_cb.V)
            print(f"  step {step}: t={t:.4f} sph={diag['sphericity']:.4f} "
                  f"dP={dp:+.2f} |u|={max_u:.3e}")

    # -- Run simulation --
    print(f"\nRunning {n_steps} steps...")
    try:
        t_final = methods.integrate(
            HC, bV, dudt_fn, dt=DT, n_steps=n_steps,
            bc_set=bc_set, callback=callback, mps=mps,
        )
    except Exception as e:
        print(f"Simulation stopped: {e}")
        t_final = t_arr[-1] if t_arr else 0.0

    diag_final = compute_diagnostics(HC, dim)
    t_arr.append(t_final)
    sph_arr.append(diag_final['sphericity'])
    t_arr = np.array(t_arr)
    sph_arr = np.array(sph_arr)

    # -- Results --
    print("\n" + "=" * 60)
    print(f"Initial sphericity: {sph_arr[0]:.4f}")
    print(f"Final sphericity:   {sph_arr[-1]:.4f}")
    dp_laplace = 2.0 * GAMMA / R_EQ
    try:
        dp_int = integrated_phase_pressure_jump(HC, 1, 0)
        print(f"Integrated bulk jump: {dp_int:+.4f} Pa "
              f"(error {dp_int / dp_laplace - 1:+.3%} of 2 gamma / R_eq = "
              f"{dp_laplace:.4f} Pa)")
    except ValueError as e:      # a refinement without droplet bulk
        dp_int = float('nan')
        print(f"Integrated bulk jump: n/a ({e})")
    print(f"Recorded {history.n_snapshots} frames")
    record_methods(os.path.join(os.path.dirname(__file__), 'results',
                                'methods.json'), methods, HC,
                   extra={'arm': args.arm,
                          'dt': DT, 'n_steps': n_steps, 'n_refine': n_refine,
                          't_final': t_final,
                          'sphericity_final': float(sph_arr[-1]),
                          'dp_integrated_final': float(dp_int),
                          'dp_laplace': dp_laplace})
    if args.no_anim:
        print("\nDone (animation skipped).")
        return history

    # -- Plotting --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig_dir = os.path.join(os.path.dirname(__file__), 'fig')
        os.makedirs(fig_dir, exist_ok=True)

        # Sphericity evolution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_arr * 1000, sph_arr, 'b-', lw=1.5)
        ax.axhline(1.0, color='k', ls=':', alpha=0.4, label='Sphere')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Sphericity')
        ax.set_title('3D Cube-to-Droplet: Shape Relaxation')
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.savefig(os.path.join(fig_dir,
                                 'cube2droplet_3D_sphericity.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Matplotlib animation
        print("\nGenerating animation...")
        from ddgclib.visualization import dynamic_plot_fluid
        dynamic_plot_fluid(
            history, HC, bV=bV,
            scalar_field='p', vector_field='u',
            save_path=os.path.join(fig_dir,
                                   'cube2droplet_3D_fluid.mp4'),
            fps=30, dpi=100,
        )
        print("Animation saved")

        print(f"\nPlots saved to {fig_dir}/")
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback; traceback.print_exc()

    # -- Optional polyscope interactive viewer --
    try:
        from ddgclib.visualization.polyscope_3d import (
            interactive_history_viewer,
        )
        print("\nLaunching polyscope viewer...")
        interactive_history_viewer(
            history, HC,
            scalar_fields=['p'],
            vector_fields=['u'],
        )
    except ImportError:
        print("(polyscope not available for interactive 3D viewer)")
    except Exception as e:
        print(f"Polyscope error: {e}")

    print("\nDone.")
    return history


if __name__ == '__main__':
    main()
