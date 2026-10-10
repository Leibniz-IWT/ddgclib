#!/usr/bin/env python3
"""2D Cube-to-droplet relaxation simulation.

A square fluid droplet in an outer fluid relaxes toward a circular
equilibrium shape under surface tension.  Both phases respond:
the droplet compresses (interior pressure rises), the outer fluid
adjusts, and viscous damping damps oscillations.

Produces:
  fig/cube2droplet_2D_final.png       — phase field snapshot
  fig/cube2droplet_2D_circularity.png — shape evolution
  fig/cube2droplet_2D_radii.png       — R_max / R_min convergence
  fig/cube2droplet_2D_pressure.png    — pressure evolution
  fig/cube2droplet_2D_fluid.mp4       — animation with interface markers

Usage
-----
    python cases_dynamic/cube2droplet/cube_to_droplet_2D.py [--n-steps N]
        [--n-refine R] [--no-anim] [--arm {base,bare,dual_only}]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.cube2droplet.src._setup import setup_cube_to_droplet
from ddgclib.analytical import integrated_phase_pressure_jump
from ddgclib.methods import PRESETS, record_methods

# =====================================================================
# Parameters
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
RECORD_EVERY = 12     # ~400 movie frames

# Zoom: show droplet + immediate surroundings
ZOOM = 2.2 * R        # ±0.022 m


def compute_diagnostics(HC, dim: int = 2):
    """Interface shape + pressure diagnostics."""
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

        if v.phase == 1 and not getattr(v, 'is_interface', False) \
                and not v.boundary:
            p_drop.append(v.p)
        elif v.phase == 0 and not v.boundary:
            p_outer.append(v.p)

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


def record_frame(HC, dim):
    """Capture per-vertex data for one animation frame.

    Returns a dict with arrays for positions, fields, phase, and
    interface membership — everything needed to render a single frame.
    """
    xs, ps, us, phases, is_iface = [], [], [], [], []
    for v in HC.V:
        xs.append(v.x_a[:dim].copy())
        ps.append(float(v.p) if np.ndim(v.p) == 0 else float(v.p[0]))
        us.append(v.u[:dim].copy())
        phases.append(int(v.phase))
        is_iface.append(bool(getattr(v, 'is_interface', False)))
    return {
        'x': np.array(xs),
        'p': np.array(ps),
        'u': np.array(us),
        'phase': np.array(phases),
        'is_interface': np.array(is_iface),
    }


# A/B arms of the shipped preset (laneX): 'bare' is the historic setup
# configuration (per-step Delaunay + redistribution, no remap), which loses
# the droplet by t = 0.4 s; 'dual_only' the fixed-connectivity refresh.
ARMS = {
    'base': PRESETS['cube_to_droplet_2D'],
    'bare': PRESETS['cube_to_droplet_2D'].replace(remap=None),
    'dual_only': PRESETS['cube_to_droplet_2D_dual_only'],
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1]
                                 if __doc__ else None)
    ap.add_argument('--n-steps', type=int, default=N_STEPS)
    ap.add_argument('--n-refine', type=int, default=N_REFINE)
    ap.add_argument('--no-anim', action='store_true',
                    help='skip the ffmpeg animation (smoke runs)')
    ap.add_argument('--arm', choices=sorted(ARMS), default='base',
                    help='configuration arm (A/B through the preset)')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    n_steps, n_refine = args.n_steps, args.n_refine
    dim = 2
    print("=" * 60)
    print("2D Cube-to-Droplet Relaxation")
    print("=" * 60)
    print(f"R={R} m, L={L_DOMAIN} m, R_eq={R_EQ:.5f} m")
    print(f"gamma={GAMMA}, mu_d={MU_D}, K_d={K_D}")

    # -- Setup (the preset is the single source of truth for the run) --
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
    frames = []         # list of (t, frame_dict)
    diag0 = compute_diagnostics(HC, dim)
    t_arr = [0.0]
    circ_arr = [diag0['circularity']]
    R_max_arr = [diag0['R_max']]
    R_min_arr = [diag0['R_min']]
    pd_arr = [diag0['P_drop']]
    po_arr = [diag0['P_outer']]
    print(f"Initial circularity: {diag0['circularity']:.4f}")

    frames.append((0.0, record_frame(HC, dim)))

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if step % RECORD_EVERY == 0:
            frames.append((t, record_frame(HC_cb, dim)))

        if step % 200 == 0:
            diag = compute_diagnostics(HC_cb, dim)
            t_arr.append(t)
            circ_arr.append(diag['circularity'])
            R_max_arr.append(diag['R_max'])
            R_min_arr.append(diag['R_min'])
            pd_arr.append(diag['P_drop'])
            po_arr.append(diag['P_outer'])

        if step % 2000 == 0 and step > 0:
            diag = compute_diagnostics(HC_cb, dim)
            dp = diag['P_drop'] - diag['P_outer']
            max_u = max(np.linalg.norm(v.u[:dim]) for v in HC_cb.V)
            print(f"  step {step}: t={t:.4f} circ={diag['circularity']:.4f} "
                  f"dP={dp:+.2f} |u|={max_u:.3e}")

    # -- Run simulation --
    print(f"\nRunning {n_steps} steps (dt={DT:.1e}, "
          f"recording every {RECORD_EVERY})...")

    try:
        t_final = methods.integrate(
            HC, bV, dudt_fn, dt=DT, n_steps=n_steps,
            bc_set=bc_set, callback=callback, mps=mps,
        )
    except Exception as e:
        print(f"Simulation stopped: {e}")
        t_final = t_arr[-1] if t_arr else 0.0

    # Final
    diag_final = compute_diagnostics(HC, dim)
    t_arr.append(t_final)
    circ_arr.append(diag_final['circularity'])
    R_max_arr.append(diag_final['R_max'])
    R_min_arr.append(diag_final['R_min'])
    pd_arr.append(diag_final['P_drop'])
    po_arr.append(diag_final['P_outer'])

    t_arr = np.array(t_arr)
    circ_arr = np.array(circ_arr)
    R_max_arr = np.array(R_max_arr)
    R_min_arr = np.array(R_min_arr)
    pd_arr = np.array(pd_arr)
    po_arr = np.array(po_arr)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Initial circularity:  {circ_arr[0]:.4f}")
    print(f"Final circularity:    {circ_arr[-1]:.4f}")
    print(f"Max circularity:      {np.max(circ_arr):.4f}")
    dp_final = pd_arr[-1] - po_arr[-1]
    print(f"Final P_drop - P_out: {dp_final:+.2f} Pa "
          f"(Laplace = {GAMMA / R_EQ:.3f} Pa)")
    # Integrated (volume-weighted bulk) Laplace jump, the comparison the
    # campaign uses (ddgclib.analytical); the nodal mean above is kept
    # for the historic plots.
    try:
        dp_int = integrated_phase_pressure_jump(HC, 1, 0)
        print(f"Integrated bulk jump:  {dp_int:+.4f} Pa "
              f"(error {dp_int / (GAMMA / R_EQ) - 1:+.3%} of gamma / R_eq)")
    except ValueError as e:      # the droplet lost its bulk (laneX: bare arm)
        dp_int = float('nan')
        print(f"Integrated bulk jump:  n/a ({e})")
    print(f"Recorded {len(frames)} movie frames")

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    record_methods(os.path.join(results_dir, 'methods.json'), methods, HC,
                   extra={'arm': args.arm,
                          'dt': DT, 'n_steps': n_steps, 'n_refine': n_refine,
                          't_final': t_final,
                          'circularity_final': float(circ_arr[-1]),
                          'dp_nodal_final': float(dp_final),
                          'dp_integrated_final': float(dp_int),
                          'dp_laplace': GAMMA / R_EQ})

    # ---- Plotting ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.animation import FuncAnimation

        fig_dir = os.path.join(os.path.dirname(__file__), 'fig')
        os.makedirs(fig_dir, exist_ok=True)

        # -- 1. Phase field snapshot with interface labels --
        fig, ax = plt.subplots(figsize=(8, 8))
        triangles, colors = [], []
        for s in HC.V.cache:
            if len(s) == dim + 1:
                verts = [HC.V[v] for v in s]
                coords = [v.x_a[:dim] for v in verts]
                triangles.append(coords)
                colors.append(np.mean([v.phase for v in verts]))
        if triangles:
            pc = PolyCollection(triangles, array=np.array(colors),
                                cmap='coolwarm', edgecolors='grey',
                                linewidths=0.2, alpha=0.7)
            ax.add_collection(pc)
            plt.colorbar(pc, ax=ax, label='Phase', shrink=0.8)

        # Interface vertices (red) with label explaining what they are
        xs_if = [v.x_a[0] for v in HC.V
                 if getattr(v, 'is_interface', False)]
        ys_if = [v.x_a[1] for v in HC.V
                 if getattr(v, 'is_interface', False)]
        if xs_if:
            ax.scatter(xs_if, ys_if, c='red', s=20, zorder=5,
                       label='Interface (phase=1, has phase=0 neighbour)',
                       edgecolors='darkred', linewidths=0.5)

        sq = np.array([[-R, -R], [R, -R], [R, R], [-R, R], [-R, -R]])
        ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=1.5, alpha=0.4,
                label='Initial square')
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R_EQ * np.cos(th), R_EQ * np.sin(th), 'g--', lw=1.5,
                alpha=0.5, label=f'R_eq={R_EQ:.4f}')
        ax.set_xlim(-ZOOM, ZOOM)
        ax.set_ylim(-ZOOM, ZOOM)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Phase field (t={t_final:.4f} s)')
        ax.legend(fontsize=7, loc='upper right')
        fig.savefig(os.path.join(fig_dir, 'cube2droplet_2D_final.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # -- 2. Circularity --
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_arr * 1000, circ_arr, 'b-', lw=1.5)
        ax.axhline(1.0, color='k', ls=':', alpha=0.4, label='Circle')
        ax.axhline(circ_arr[0], color='grey', ls='--', alpha=0.4,
                    label=f'Initial ({circ_arr[0]:.3f})')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Circularity')
        ax.set_title('Shape Relaxation')
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.savefig(os.path.join(fig_dir,
                                 'cube2droplet_2D_circularity.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # -- 3. Radii --
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_arr * 1000, R_max_arr * 1000, 'r-', label='R_max')
        ax.plot(t_arr * 1000, R_min_arr * 1000, 'b-', label='R_min')
        ax.axhline(R_EQ * 1000, color='g', ls=':', alpha=0.5,
                    label=f'R_eq={R_EQ * 1000:.2f} mm')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Radius [mm]')
        ax.set_title('Interface Radii')
        ax.legend()
        fig.savefig(os.path.join(fig_dir, 'cube2droplet_2D_radii.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # -- 4. Pressure --
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_arr * 1000, pd_arr, 'r-', label='P_droplet')
        ax.plot(t_arr * 1000, po_arr, 'b-', label='P_outer')
        ax.plot(t_arr * 1000, pd_arr - po_arr, 'k--', label='dP')
        ax.axhline(GAMMA / R_EQ, color='g', ls=':', alpha=0.5,
                    label=f'Laplace dP={GAMMA / R_EQ:.3f}')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Pressure [Pa]')
        ax.set_title('Pressure Evolution')
        ax.legend()
        fig.savefig(os.path.join(fig_dir,
                                 'cube2droplet_2D_pressure.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # -- 5. Custom animation with interface markers --
        if args.no_anim:
            print(f"\nPlots saved to {fig_dir}/ (animation skipped)")
            print("\nDone.")
            return
        print("\nGenerating animation with interface markers...")

        # Pre-compute global pressure range for stable colorbar
        all_p = np.concatenate([f['p'] for _, f in frames])
        pvmin, pvmax = np.percentile(all_p, [2, 98])
        if pvmin == pvmax:
            pvmin, pvmax = pvmin - 1, pvmax + 1

        fig_anim, axes_anim = plt.subplots(1, 2, figsize=(14, 6))
        fig_anim.subplots_adjust(top=0.88)
        suptitle = fig_anim.suptitle('t = 0.0000 s', fontsize=13)

        def update(frame_idx):
            t, fd = frames[frame_idx]
            x = fd['x']
            p = fd['p']
            u = fd['u']
            phase = fd['phase']
            is_if = fd['is_interface']

            for ax in axes_anim:
                ax.clear()

            suptitle.set_text(f't = {t:.4f} s')

            # -- Left panel: pressure + interface --
            ax_p = axes_anim[0]
            sc = ax_p.scatter(x[:, 0], x[:, 1], c=p, cmap='viridis',
                              vmin=pvmin, vmax=pvmax, s=8, zorder=2)
            # Interface vertices: red rings
            if np.any(is_if):
                ax_p.scatter(x[is_if, 0], x[is_if, 1],
                             facecolors='none', edgecolors='red',
                             s=30, linewidths=1.0, zorder=4,
                             label='Interface')
            ax_p.set_title('Pressure [Pa]')
            ax_p.set_xlabel('x [m]')
            ax_p.set_ylabel('y [m]')
            ax_p.set_xlim(-ZOOM, ZOOM)
            ax_p.set_ylim(-ZOOM, ZOOM)
            ax_p.set_aspect('equal')
            ax_p.legend(fontsize=7, loc='upper right')

            # -- Right panel: velocity + phase coloring --
            ax_v = axes_anim[1]
            # Color background by phase
            c_phase = np.where(phase == 1, 'lightsalmon', 'lightblue')
            ax_v.scatter(x[:, 0], x[:, 1], c=c_phase, s=8, zorder=1,
                         alpha=0.6)
            # Velocity quiver
            u_mag = np.linalg.norm(u, axis=1)
            max_u = u_mag.max()
            if max_u > 1e-15:
                ax_v.quiver(x[:, 0], x[:, 1], u[:, 0], u[:, 1],
                            u_mag, cmap='hot', scale=max_u * 20,
                            width=0.003, zorder=3)
            # Interface vertices: red rings
            if np.any(is_if):
                ax_v.scatter(x[is_if, 0], x[is_if, 1],
                             facecolors='none', edgecolors='red',
                             s=30, linewidths=1.0, zorder=4,
                             label='Interface')
            ax_v.set_title('Velocity + Phase')
            ax_v.set_xlabel('x [m]')
            ax_v.set_ylabel('y [m]')
            ax_v.set_xlim(-ZOOM, ZOOM)
            ax_v.set_ylim(-ZOOM, ZOOM)
            ax_v.set_aspect('equal')
            ax_v.legend(fontsize=7, loc='upper right')

            return []

        anim = FuncAnimation(fig_anim, update, frames=len(frames),
                             interval=33, blit=False)
        mp4_path = os.path.join(fig_dir, 'cube2droplet_2D_fluid.mp4')
        anim.save(mp4_path, writer='ffmpeg', fps=30, dpi=120)
        plt.close(fig_anim)
        print(f"Animation saved: {mp4_path} ({len(frames)} frames)")

        print(f"\nAll plots saved to {fig_dir}/")
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
