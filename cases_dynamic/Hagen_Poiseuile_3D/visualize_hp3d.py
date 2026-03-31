"""
Visualize 3D Hagen-Poiseuille simulation results.

Loads saved history and final state, generates:
  1. Radial velocity profile vs analytical solution (matplotlib)
  2. Velocity development along the pipe (matplotlib)
  3. Cross-section velocity contour at midpoint (matplotlib)
  4. Interactive polyscope viewer with timeline slider

Usage::

    # After running the simulation:
    python visualize_hp3d.py

    # Polyscope interactive viewer only:
    python visualize_hp3d.py --polyscope

    # Skip polyscope, only generate matplotlib plots:
    python visualize_hp3d.py --no-polyscope

    # Custom results directory:
    python visualize_hp3d.py --results-dir path/to/results
"""

import argparse
import math
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ddgclib.data import load_state
from ddgclib.initial_conditions import HagenPoiseuille3D

# Import parameters from the case script
sys.path.insert(0, _HERE)
from Hagen_Poiseuile_3D import (
    R, D, L, G, mu, rho, Re_D, U_avg, U_max, flow_axis, n_refine,
)


_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')


# ============================================================
# Helpers
# ============================================================

def _radial_distance(x_a):
    """Radial distance from pipe centerline."""
    radial_axes = [i for i in range(3) if i != flow_axis]
    return np.linalg.norm([x_a[ax] for ax in radial_axes])


def _load_history():
    """Load the simulation history from pickle."""
    path = os.path.join(_RESULTS, 'hp3d_history.pkl')
    if not os.path.exists(path):
        print(f"History file not found: {path}")
        print("Run Hagen_Poiseuile_3D.py first.")
        sys.exit(1)
    with open(path, 'rb') as f:
        history = pickle.load(f)
    print(f"Loaded history: {history.n_snapshots} snapshots")
    return history


def _load_final_state():
    """Load the final simulation state."""
    path = os.path.join(_RESULTS, 'hp3d_final_state.json')
    if not os.path.exists(path):
        print(f"Final state not found: {path}")
        print("Run Hagen_Poiseuile_3D.py first.")
        sys.exit(1)
    HC, bV, meta = load_state(path)
    print(f"Loaded final state: t={meta['time']:.4f}, "
          f"{sum(1 for _ in HC.V)} vertices")
    return HC, bV, meta


def _extract_profile_from_snapshot(snapshot, z_target, tol):
    """Extract radial velocity profile from a snapshot dict at axial position z_target."""
    radii = []
    velocities = []
    for key, fields in snapshot.items():
        x_a = np.array(key)
        if abs(x_a[flow_axis] - z_target) < tol:
            r = _radial_distance(x_a)
            if r < R - 1e-8:  # interior only
                u_vec = fields.get('u', np.zeros(3))
                u_axial = u_vec[flow_axis] if hasattr(u_vec, '__len__') else 0.0
                radii.append(r)
                velocities.append(u_axial)
    return np.array(radii), np.array(velocities)


# ============================================================
# Plot 1: Radial velocity profile vs analytical
# ============================================================

def plot_radial_profile(HC, bV):
    """Radial velocity profile at z=L/2 vs analytical Hagen-Poiseuille."""
    hp_ic = HagenPoiseuille3D(U_max=U_max, R=R, flow_axis=flow_axis, dim=3)

    z_mid = L / 2.0
    tol = L / (2**n_refine) * 0.6

    mid_verts = [v for v in HC.V
                 if v not in bV and abs(v.x_a[flow_axis] - z_mid) < tol]

    r_vals = np.array([_radial_distance(v.x_a) for v in mid_verts])
    u_num = np.array([v.u[flow_axis] for v in mid_verts])

    # Analytical curve
    r_anal = np.linspace(0, R, 100)
    u_anal = U_max * (1 - (r_anal / R)**2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_anal, u_anal, 'k-', lw=2, label='Analytical (Hagen-Poiseuille)')
    ax.scatter(r_vals, u_num, s=30, c='tab:blue', alpha=0.7, zorder=5,
               label=f'Numerical ({len(mid_verts)} vertices)')
    ax.set_xlabel('Radial distance r [m]')
    ax.set_ylabel(f'u_{"xyz"[flow_axis]} [m/s]')
    ax.set_title(f'Velocity profile at z = L/2  (Re={Re_D}, n_refine={n_refine})')
    ax.legend()
    ax.set_xlim(0, R * 1.05)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(_FIG, 'hp3d_velocity_profile.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Radial profile -> {path}")


# ============================================================
# Plot 2: Velocity development along the pipe (centerline + average)
# ============================================================

def plot_axial_development(HC, bV):
    """Plot centerline and cross-section average velocity along the pipe axis."""
    # Bin vertices by axial position
    n_bins = 30
    z_edges = np.linspace(0, L, n_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    u_max_bins = np.full(n_bins, np.nan)
    u_avg_bins = np.full(n_bins, np.nan)

    for i in range(n_bins):
        z_lo, z_hi = z_edges[i], z_edges[i + 1]
        verts_in_bin = [v for v in HC.V
                        if v not in bV
                        and z_lo <= v.x_a[flow_axis] < z_hi]
        if verts_in_bin:
            u_axial = np.array([v.u[flow_axis] for v in verts_in_bin])
            u_max_bins[i] = np.max(u_axial)
            u_avg_bins[i] = np.mean(u_axial)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_centers, u_max_bins, 'o-', ms=4, label='U_max (per bin)')
    ax.plot(z_centers, u_avg_bins, 's-', ms=4, label='U_avg (per bin)')
    ax.axhline(U_max, color='k', ls='--', lw=1, label=f'Analytical U_max = {U_max:.3f}')
    ax.axhline(U_avg, color='gray', ls=':', lw=1, label=f'U_avg = {U_avg:.3f}')
    ax.axvline(0.06 * Re_D * D, color='tab:red', ls='--', lw=1, alpha=0.5,
               label=f'L_e ~ {0.06 * Re_D * D:.1f} m')
    ax.set_xlabel(f'{"xyz"[flow_axis]} [m]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title(f'Axial velocity development  (Re={Re_D}, n_refine={n_refine})')
    ax.legend(fontsize=9)
    ax.set_xlim(0, L)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(_FIG, 'hp3d_axial_development.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Axial development -> {path}")


# ============================================================
# Plot 3: Cross-section velocity contour at z=L/2
# ============================================================

def plot_cross_section(HC, bV):
    """Scatter plot of axial velocity in the cross-section at z=L/2."""
    radial_axes = [i for i in range(3) if i != flow_axis]
    ax0, ax1 = radial_axes

    z_mid = L / 2.0
    tol = L / (2**n_refine) * 0.6

    verts = [v for v in HC.V if abs(v.x_a[flow_axis] - z_mid) < tol]

    x = np.array([v.x_a[ax0] for v in verts])
    y = np.array([v.x_a[ax1] for v in verts])
    u = np.array([v.u[flow_axis] for v in verts])

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=u, s=40, cmap='coolwarm', edgecolors='k', lw=0.3)
    fig.colorbar(sc, ax=ax, label=f'u_{"xyz"[flow_axis]} [m/s]')

    # Draw cylinder boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(R * np.cos(theta), R * np.sin(theta), 'k-', lw=1.5)

    ax.set_xlabel(f'{"xyz"[ax0]} [m]')
    ax.set_ylabel(f'{"xyz"[ax1]} [m]')
    ax.set_title(f'Cross-section velocity at z=L/2  (Re={Re_D}, n_refine={n_refine})')
    ax.set_aspect('equal')
    ax.set_xlim(-R * 1.1, R * 1.1)
    ax.set_ylim(-R * 1.1, R * 1.1)

    fig.tight_layout()
    path = os.path.join(_FIG, 'hp3d_cross_section.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Cross-section -> {path}")


# ============================================================
# Plot 4: Profile evolution over time
# ============================================================

def plot_profile_evolution(history):
    """Overlay radial velocity profiles at different times."""
    hp_ic = HagenPoiseuille3D(U_max=U_max, R=R, flow_axis=flow_axis, dim=3)

    z_mid = L / 2.0
    tol = L / (2**n_refine) * 0.6

    fig, ax = plt.subplots(figsize=(8, 6))

    # Analytical
    r_anal = np.linspace(0, R, 100)
    u_anal = U_max * (1 - (r_anal / R)**2)
    ax.plot(r_anal, u_anal, 'k-', lw=2, label='Analytical')

    # Sample snapshots evenly across the history
    n_snap = history.n_snapshots
    indices = np.linspace(0, n_snap - 1, min(n_snap, 8), dtype=int)
    cmap = plt.colormaps.get_cmap('viridis')

    for k, idx in enumerate(indices):
        t, snapshot, _ = history._snapshots[idx]
        r_vals, u_vals = _extract_profile_from_snapshot(snapshot, z_mid, tol)
        if len(r_vals) > 0:
            color = cmap(k / max(len(indices) - 1, 1))
            order = np.argsort(r_vals)
            ax.plot(r_vals[order], u_vals[order], 'o-', ms=4, color=color,
                    alpha=0.7, label=f't = {t:.2f} s')

    ax.set_xlabel('Radial distance r [m]')
    ax.set_ylabel(f'u_{"xyz"[flow_axis]} [m/s]')
    ax.set_title(f'Profile evolution at z=L/2  (Re={Re_D}, n_refine={n_refine})')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, R * 1.05)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(_FIG, 'hp3d_profile_evolution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Profile evolution -> {path}")


# ============================================================
# Plot 5: Convergence / error metrics over time
# ============================================================

def plot_error_evolution(history):
    """Plot L2 error of velocity profile vs analytical over time."""
    hp_ic = HagenPoiseuille3D(U_max=U_max, R=R, flow_axis=flow_axis, dim=3)

    z_mid = L / 2.0
    tol = L / (2**n_refine) * 0.6

    times = []
    l2_errors = []
    max_u_z = []

    for t, snapshot, _ in history._snapshots:
        r_vals, u_vals = _extract_profile_from_snapshot(snapshot, z_mid, tol)
        if len(r_vals) == 0:
            continue

        # Analytical at these radii
        u_anal = np.array([hp_ic.analytical_velocity(
            _pos_from_r(r, z_mid)) for r in r_vals])
        err = np.sqrt(np.mean((u_vals - u_anal)**2))

        times.append(t)
        l2_errors.append(err)
        max_u_z.append(np.max(u_vals))

    if not times:
        print("  No profile data for error evolution — skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(times, l2_errors, 'o-', ms=4)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('L2 error vs analytical')
    ax1.set_title('Profile error convergence')
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, max_u_z, 'o-', ms=4, color='tab:orange')
    ax2.axhline(U_max, color='k', ls='--', label=f'Analytical U_max = {U_max:.3f}')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel(f'Max u_{"xyz"[flow_axis]} at z=L/2 [m/s]')
    ax2.set_title('Centerline velocity convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(_FIG, 'hp3d_error_evolution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Error evolution -> {path}")


def _pos_from_r(r, z):
    """Construct a 3D position from radial distance r and axial position z."""
    pos = np.zeros(3)
    pos[flow_axis] = z
    radial_axes = [i for i in range(3) if i != flow_axis]
    pos[radial_axes[0]] = r  # put radial distance along first radial axis
    return pos


# ============================================================
# Polyscope interactive viewer
# ============================================================

def launch_polyscope(history):
    """Launch interactive polyscope viewer with timeline slider."""
    try:
        from ddgclib.visualization.polyscope_3d import interactive_history_viewer
        from ddgclib.geometry.domains import cylinder_volume
    except ImportError:
        print("polyscope not available — skipping interactive viewer")
        return

    # Build a dummy HC for dim info
    result = cylinder_volume(R=R, L=1.0, refinement=1, flow_axis=flow_axis)

    print("\nLaunching polyscope viewer ...")
    interactive_history_viewer(
        history, result.HC,
        scalar_fields=['p'],
        vector_fields=['u'],
        name='hp3d',
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D Hagen-Poiseuille results"
    )
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory (default: results/)')
    parser.add_argument('--polyscope', action='store_true',
                        help='Launch polyscope viewer only')
    parser.add_argument('--no-polyscope', action='store_true',
                        help='Skip polyscope, generate matplotlib plots only')
    args = parser.parse_args()

    global _RESULTS, _FIG
    if args.results_dir:
        _RESULTS = args.results_dir

    os.makedirs(_FIG, exist_ok=True)

    # Load data
    history = _load_history()

    if args.polyscope:
        launch_polyscope(history)
        return

    HC, bV, meta = _load_final_state()

    print(f"\nGenerating plots (Re={Re_D}, n_refine={n_refine}) ...")

    # Matplotlib plots
    plot_radial_profile(HC, bV)
    plot_axial_development(HC, bV)
    plot_cross_section(HC, bV)
    plot_profile_evolution(history)
    plot_error_evolution(history)

    print(f"\nAll plots saved to {_FIG}/")

    # Polyscope
    if not args.no_polyscope:
        launch_polyscope(history)


if __name__ == "__main__":
    main()
