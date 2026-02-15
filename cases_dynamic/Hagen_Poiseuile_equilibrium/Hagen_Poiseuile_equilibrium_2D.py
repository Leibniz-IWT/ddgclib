"""
2D Hagen-Poiseuille Equilibrium Validation
===========================================

Validates the stress tensor operators in ``ddgclib.operators.stress`` by
initialising a 2D channel flow at the **analytical equilibrium** and
checking that the discrete ``stress_acceleration`` (dudt_i) is near zero.

Analytical solution (fully-developed planar Poiseuille):
    u_x(y) = (G / 2mu) * y * (h - y),   u_y = 0
    P(x)   = -G * x

At equilibrium the pressure gradient exactly balances viscous diffusion,
so the net force on every fluid parcel should vanish.  On a discrete mesh
there is truncation error; this script measures that error and prints a
convergence table across mesh refinements.

Usage::

    python Hagen_Poiseuile_equilibrium_2D.py
"""

import os
import sys

import numpy as np

# Ensure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_FIG = os.path.join(_HERE, 'fig')

import matplotlib
matplotlib.use('Agg')

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.initial_conditions import (
    CompositeIC,
    LinearPressureGradient,
    PoiseuillePlanar,
    UniformMass,
)
from ddgclib.operators.stress import (
    stress_acceleration,
    stress_force,
    velocity_difference_tensor,
)


# ============================================================
# Configuration
# ============================================================
G = 1.0        # pressure gradient  [Pa/m]
mu = 1.0       # dynamic viscosity  [Pa.s]
L = 1.0        # channel length     [m]
h = 1.0        # channel height     [m]
rho = 1.0      # density            [kg/m^3]
dim = 2

refinement_levels = [1, 2, 3, 4, 5]

U_max = G * h**2 / (8 * mu)


# ============================================================
# Helpers
# ============================================================

def build_equilibrium(n_refine: int):
    """Build mesh at analytical Poiseuille equilibrium.

    Returns (HC, bV, bV_wall).
    """
    HC = Complex(dim, domain=[(0.0, L), (0.0, h)])
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = set()
    for v in HC.V:
        on_boundary = (
            abs(v.x_a[0]) < 1e-14
            or abs(v.x_a[0] - L) < 1e-14
            or abs(v.x_a[1]) < 1e-14
            or abs(v.x_a[1] - h) < 1e-14
        )
        if on_boundary:
            bV.add(v)
            v.boundary = True
        else:
            v.boundary = False

    bV_wall = {
        v for v in bV
        if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14
    }

    ic = CompositeIC(
        PoiseuillePlanar(G=G, mu=mu, y_lb=0.0, y_ub=h,
                         flow_axis=0, normal_axis=1, dim=dim),
        LinearPressureGradient(G=G, axis=0, P_ref=0.0),
        UniformMass(total_volume=L * h, rho=rho),
    )
    ic.apply(HC, bV)
    compute_vd(HC, cdist=1e-10)

    return HC, bV, bV_wall


def analyse_equilibrium(HC, bV):
    """Compute per-vertex residual diagnostics for interior vertices.

    Returns dict with arrays/scalars.
    """
    accel_norms = []        # ||stress_acceleration||
    F_full_norms = []       # ||F_stress||
    F_pressure_x = []       # pressure-only F_x
    F_viscous_x = []        # viscous-only F_x
    du_norms = []           # ||velocity_difference_tensor||
    positions_y = []

    for v in HC.V:
        if v in bV:
            continue
        a = stress_acceleration(v, dim=dim, mu=mu, HC=HC)
        F_full = stress_force(v, dim=dim, mu=mu, HC=HC)
        F_pres = stress_force(v, dim=dim, mu=0.0, HC=HC)
        F_visc = F_full - F_pres
        du = velocity_difference_tensor(v, HC, dim=dim)

        accel_norms.append(np.linalg.norm(a))
        F_full_norms.append(np.linalg.norm(F_full))
        F_pressure_x.append(F_pres[0])
        F_viscous_x.append(F_visc[0])
        du_norms.append(np.linalg.norm(du))
        positions_y.append(v.x_a[1])

    accel_norms = np.array(accel_norms)
    F_full_norms = np.array(F_full_norms)
    return {
        "accel_norms": accel_norms,
        "F_full_norms": F_full_norms,
        "F_pressure_x": np.array(F_pressure_x),
        "F_viscous_x": np.array(F_viscous_x),
        "du_norms": np.array(du_norms),
        "positions_y": np.array(positions_y),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Hagen-Poiseuille Equilibrium Validation  (2D)")
    print("=" * 72)
    print(f"  G  = {G}  Pa/m      mu = {mu}  Pa.s")
    print(f"  h  = {h}  m         L  = {L}  m")
    print(f"  U_max (analytical) = {U_max}")
    print()

    # -- Convergence table --------------------------------------------------
    header = (
        f"{'refine':>6}  {'n_vert':>7}  {'n_int':>6}  "
        f"{'||a|| max':>12}  {'||a|| med':>12}  {'||a|| mean':>12}  "
        f"{'||F|| max':>12}  {'||F|| med':>12}  "
        f"{'oppos%':>7}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for nr in refinement_levels:
        HC, bV, _ = build_equilibrium(nr)
        #HC.plot_complex()
        n_vert = sum(1 for _ in HC.V)
        diag = analyse_equilibrium(HC, bV)
        an = diag["accel_norms"]
        fn = diag["F_full_norms"]
        n_int = len(an)

        # Fraction of interior verts where pressure and viscous x-forces oppose
        fp = diag["F_pressure_x"]
        fv = diag["F_viscous_x"]
        opposing = np.sum((fp * fv) < 0)
        oppos_pct = 100.0 * opposing / n_int if n_int > 0 else 0.0

        print(
            f"{nr:>6d}  {n_vert:>7d}  {n_int:>6d}  "
            f"{np.max(an):>12.4e}  {np.median(an):>12.4e}  {np.mean(an):>12.4e}  "
            f"{np.max(fn):>12.4e}  {np.median(fn):>12.4e}  "
            f"{oppos_pct:>6.1f}%"
        )
        results.append({
            "n_refine": nr, "n_vert": n_vert, "n_int": n_int,
            "accel_max": np.max(an), "accel_med": np.median(an),
            "accel_mean": np.mean(an),
            "F_max": np.max(fn), "F_med": np.median(fn),
            "opposing_pct": oppos_pct,
        })

    # -- Convergence rates --------------------------------------------------
    print()
    print("Convergence of median ||stress_acceleration||:")
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        if prev["accel_med"] > 0 and curr["accel_med"] > 0:
            ratio = curr["accel_med"] / prev["accel_med"]
            print(
                f"  refine {prev['n_refine']} -> {curr['n_refine']}:  "
                f"ratio = {ratio:.4f}  "
                f"({'converging' if ratio < 1.0 else 'not converging'})"
            )

    # -- Per-vertex detail at finest mesh -----------------------------------
    print()
    finest = refinement_levels[-1]
    HC, bV, _ = build_equilibrium(finest)
    diag = analyse_equilibrium(HC, bV)

    print(f"Per-vertex detail at n_refine={finest}  "
          f"(sorted by y-position, interior only):")
    print(f"  {'y':>8}  {'||a||':>12}  {'F_pres_x':>12}  "
          f"{'F_visc_x':>12}  {'||du||':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    idx = np.argsort(diag["positions_y"])
    for i in idx:
        print(
            f"  {diag['positions_y'][i]:>8.4f}  "
            f"{diag['accel_norms'][i]:>12.4e}  "
            f"{diag['F_pressure_x'][i]:>12.4e}  "
            f"{diag['F_viscous_x'][i]:>12.4e}  "
            f"{diag['du_norms'][i]:>12.4e}"
        )

    # -- Visualization -------------------------------------------------------
    print()
    print("Generating plots ...")
    os.makedirs(_FIG, exist_ok=True)
    plot_n_refine = refinement_levels[-1]
    HC_vis, bV_vis, _ = build_equilibrium(plot_n_refine)
    _annotate_vertices(HC_vis, bV_vis)
    _plot_equilibrium_fields(HC_vis, bV_vis, plot_n_refine)
    print(f"Figures saved to {_FIG}/")

    print()
    print("Done.")


# ============================================================
# Visualization
# ============================================================

def _annotate_vertices(HC, bV):
    """Compute diagnostics and store as vertex attributes for plotting."""
    for v in HC.V:
        if v in bV:
            # Boundary: set neutral values so colour maps don't break
            v.a_mag = 0.0
            v.F_pres_x = 0.0
            v.F_visc_x = 0.0
            v.F_net_x = 0.0
            v.du_norm = 0.0
            continue
        a = stress_acceleration(v, dim=dim, mu=mu, HC=HC)
        F_full = stress_force(v, dim=dim, mu=mu, HC=HC)
        F_pres = stress_force(v, dim=dim, mu=0.0, HC=HC)
        F_visc = F_full - F_pres
        du = velocity_difference_tensor(v, HC, dim=dim)

        v.a_mag = float(np.linalg.norm(a))
        v.F_pres_x = float(F_pres[0])
        v.F_visc_x = float(F_visc[0])
        v.F_net_x = float(F_full[0])
        v.du_norm = float(np.linalg.norm(du))


def _plot_equilibrium_fields(HC, bV, n_refine):
    """Create a 6-panel figure of the equilibrium fields."""
    import matplotlib.pyplot as plt
    from ddgclib.visualization.matplotlib_2d import (
        plot_mesh_2d,
        plot_scalar_field_2d,
        plot_vector_field_2d,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Hagen-Poiseuille Equilibrium  (n_refine={n_refine}, '
        f'G={G}, mu={mu})',
        fontsize=14,
    )

    # (0,0) — Mesh with velocity quiver overlay
    plot_mesh_2d(HC, ax=axes[0, 0], bV=bV, edge_color='lightgray',
                 vertex_size=5, vertex_color='steelblue',
                 boundary_color='tomato',
                 title='Mesh + velocity')
    plot_vector_field_2d(HC, ax=axes[0, 0], scale=3.0,
                         title='Mesh + velocity')

    # (0,1) — u_x  (flow-direction velocity, the parabolic profile)
    plot_scalar_field_2d(HC, field='u', ax=axes[0, 1], cmap='coolwarm',
                         s=15, title='$u_x$ (analytical IC)')
    # Note: plot_scalar_field_2d reads v.u and takes float(v.u[0])

    # (0,2) — Pressure P(x) = -G*x
    plot_scalar_field_2d(HC, field='p', ax=axes[0, 2], cmap='viridis',
                         s=15, title='Pressure $P$')

    # (1,0) — ||stress_acceleration||  (equilibrium residual)
    plot_scalar_field_2d(HC, field='a_mag', ax=axes[1, 0], cmap='hot',
                         s=15, title='$\\|\\mathbf{a}\\|$ (residual)')

    # (1,1) — Pressure force F_x  vs  Viscous force F_x
    ax_force = axes[1, 1]
    x_all = np.array([v.x_a[0] for v in HC.V if v not in bV])
    y_all = np.array([v.x_a[1] for v in HC.V if v not in bV])
    fp = np.array([v.F_pres_x for v in HC.V if v not in bV])
    fv = np.array([v.F_visc_x for v in HC.V if v not in bV])

    ax_force.scatter(y_all, fp, s=8, alpha=0.6, label='$F^{pres}_x$',
                     color='tab:blue')
    ax_force.scatter(y_all, fv, s=8, alpha=0.6, label='$F^{visc}_x$',
                     color='tab:orange')
    ax_force.axhline(0, color='gray', lw=0.5, ls='--')
    ax_force.set_xlabel('y')
    ax_force.set_ylabel('Force (x-component)')
    ax_force.set_title('Pressure vs viscous force')
    ax_force.legend(fontsize=9)

    # (1,2) — ||du|| (velocity difference tensor norm)
    plot_scalar_field_2d(HC, field='du_norm', ax=axes[1, 2], cmap='plasma',
                         s=15, title='$\\|\\nabla u\\|$ (vel. diff. tensor)')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(_FIG, 'equilibrium_fields.png'), dpi=150)
    plt.close(fig)

    # -- Residual profile: ||a|| vs y  (1D slice) --------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    y_int = np.array([v.x_a[1] for v in HC.V if v not in bV])
    a_int = np.array([v.a_mag for v in HC.V if v not in bV])
    ax2.scatter(y_int, a_int, s=10, alpha=0.5, color='tab:red')
    ax2.set_xlabel('y')
    ax2.set_ylabel('$\\|\\mathbf{a}\\|$  (stress acceleration)')
    ax2.set_title(
        f'Equilibrium residual vs y-position  (n_refine={n_refine})'
    )
    ax2.axhline(0, color='gray', lw=0.5, ls='--')
    fig2.tight_layout()
    fig2.savefig(os.path.join(_FIG, 'residual_vs_y.png'), dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    main()
