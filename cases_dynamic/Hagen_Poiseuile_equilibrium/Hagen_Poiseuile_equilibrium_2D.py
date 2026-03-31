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
    velocity_difference_tensor_pointwise,
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
    from ddgclib.analytical._integrated_comparison import (
        integrated_pressure_error,
        integrated_l2_norm,
    )
    from hyperct.ddg import dual_cell_polygon_2d
    from ddgclib.analytical._divergence_theorem import (
        integrated_gradient_2d_vector,
    )

    accel_norms = []        # ||stress_acceleration||
    F_full_norms = []       # ||F_stress||
    F_pressure_x = []       # pressure-only F_x
    F_viscous_x = []        # viscous-only F_x
    du_norms = []           # ||velocity_difference_tensor||
    du_errors = []          # ||Du_DDG - Du_analytical|| (integrated)
    positions_y = []

    interior = [v for v in HC.V if v not in bV]

    for v in interior:
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

        # Integrated comparison: DDG Du vs analytical ∫ ∇u dV
        try:
            polygon = dual_cell_polygon_2d(v)
            u_callable = lambda x: np.array([
                G / (2.0 * mu) * x[1] * (h - x[1]), 0.0
            ])
            du_ana = integrated_gradient_2d_vector(u_callable, polygon)
            du_errors.append(np.linalg.norm(du - du_ana))
        except Exception:
            du_errors.append(float('nan'))

    accel_norms = np.array(accel_norms)
    F_full_norms = np.array(F_full_norms)

    # Integrated pressure comparison
    P_analytical = lambda x: -G * x[0]
    int_p_errs = integrated_pressure_error(
        HC, interior, P_analytical, dim=dim,
    )
    int_p_l2 = integrated_l2_norm(
        HC, interior, P_analytical, dim=dim,
    )

    return {
        "accel_norms": accel_norms,
        "F_full_norms": F_full_norms,
        "F_pressure_x": np.array(F_pressure_x),
        "F_viscous_x": np.array(F_viscous_x),
        "du_norms": np.array(du_norms),
        "du_errors": np.array(du_errors),
        "positions_y": np.array(positions_y),
        "int_pressure_errs": int_p_errs,
        "int_pressure_l2": int_p_l2,
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
    # NOTE: ||F|| (integrated force) is the meaningful convergence metric.
    # ||a|| = ||F||/m stays O(1) because both F and m scale as O(h^dim),
    # so their ratio does not converge.  The force residual shows clean
    # O(h^2) convergence for the face-centred FVM on barycentric duals.
    header = (
        f"{'refine':>6}  {'n_vert':>7}  {'n_int':>6}  "
        f"{'||F|| max':>12}  {'||F|| med':>12}  "
        f"{'||a|| max':>12}  {'||a|| med':>12}  "
        f"{'Du_err max':>12}  {'P_L2':>12}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for nr in refinement_levels:
        HC, bV, _ = build_equilibrium(nr)
        n_vert = sum(1 for _ in HC.V)
        diag = analyse_equilibrium(HC, bV)
        an = diag["accel_norms"]
        fn = diag["F_full_norms"]
        du_err = diag["du_errors"]
        n_int = len(an)

        print(
            f"{nr:>6d}  {n_vert:>7d}  {n_int:>6d}  "
            f"{np.max(fn):>12.4e}  {np.median(fn):>12.4e}  "
            f"{np.max(an):>12.4e}  {np.median(an):>12.4e}  "
            f"{np.nanmax(du_err):>12.4e}  {diag['int_pressure_l2']:>12.4e}"
        )
        results.append({
            "n_refine": nr, "n_vert": n_vert, "n_int": n_int,
            "accel_max": np.max(an), "accel_med": np.median(an),
            "accel_mean": np.mean(an),
            "F_max": np.max(fn), "F_med": np.median(fn),
            "du_err_max": float(np.nanmax(du_err)),
            "int_pressure_l2": diag["int_pressure_l2"],
        })

    # -- Convergence rates --------------------------------------------------
    print()
    print("Convergence of median ||F|| (integrated force):")
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        if prev["F_med"] > 0 and curr["F_med"] > 0:
            ratio = curr["F_med"] / prev["F_med"]
            rate = np.log2(1.0 / ratio) if ratio > 0 else float('inf')
            print(
                f"  refine {prev['n_refine']} -> {curr['n_refine']}:  "
                f"ratio = {ratio:.4f}  "
                f"(order ~ {rate:.2f})"
            )
    a_med = results[-1]["accel_med"]
    F_med = results[-1]["F_med"]
    if F_med < 1e-10:
        print()
        print(f"Force residual is machine-precision zero ({F_med:.2e}).")
        print("Pressure and viscous fluxes cancel exactly for Poiseuille flow.")
    else:
        print()
        print("Note: ||a|| = ||F||/m stays ~{:.3f} because both F and m".format(a_med))
        print("  scale as O(h^2).  This is expected for the integrated FVM.")

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

    # -- Residual profile: ||F|| and ||a|| vs y  (1D slice) ----------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    y_int = np.array([v.x_a[1] for v in HC.V if v not in bV])
    a_int = np.array([v.a_mag for v in HC.V if v not in bV])
    F_int = np.array([np.linalg.norm(
        stress_force(v, dim=dim, mu=mu, HC=HC)
    ) for v in HC.V if v not in bV])

    ax2a.scatter(y_int, F_int, s=10, alpha=0.5, color='tab:blue')
    ax2a.set_xlabel('y')
    ax2a.set_ylabel('$\\|\\mathbf{F}\\|$  (integrated force)')
    ax2a.set_title(
        f'Force residual vs y  (n_refine={n_refine})'
    )
    ax2a.axhline(0, color='gray', lw=0.5, ls='--')
    ax2a.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    ax2b.scatter(y_int, a_int, s=10, alpha=0.5, color='tab:red')
    ax2b.set_xlabel('y')
    ax2b.set_ylabel('$\\|\\mathbf{a}\\|$ = $\\|\\mathbf{F}\\|$ / m')
    ax2b.set_title(
        f'Acceleration residual vs y  (n_refine={n_refine})'
    )
    ax2b.axhline(0, color='gray', lw=0.5, ls='--')

    fig2.tight_layout()
    fig2.savefig(os.path.join(_FIG, 'residual_vs_y.png'), dpi=150)
    plt.close(fig2)

    # -- Integrated comparison: Du_DDG vs Du_analytical -----------------------
    from hyperct.ddg import dual_cell_polygon_2d
    from ddgclib.analytical._divergence_theorem import integrated_gradient_2d_vector

    interior_list = [v for v in HC.V if v not in bV]
    du_err_list = []
    y_du = []
    u_callable = lambda x: np.array([
        G / (2.0 * mu) * x[1] * (h - x[1]), 0.0
    ])
    for v in interior_list:
        try:
            Du_num = velocity_difference_tensor(v, HC, dim=dim)
            polygon = dual_cell_polygon_2d(v)
            Du_ana = integrated_gradient_2d_vector(u_callable, polygon)
            du_err_list.append(np.linalg.norm(Du_num - Du_ana))
            y_du.append(v.x_a[1])
        except Exception:
            pass

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    if du_err_list:
        ax3a.scatter(y_du, du_err_list, s=10, alpha=0.5, color='tab:green')
        ax3a.set_xlabel('y')
        ax3a.set_ylabel('$\\|Du_{DDG} - Du_{analytical}\\|$')
        ax3a.set_title(f'Velocity gradient error (n_refine={n_refine})')
        ax3a.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax3a.grid(True, alpha=0.3)

    # Pressure + viscous force balance
    fp = np.array([v.F_pres_x for v in HC.V if v not in bV])
    fv = np.array([v.F_visc_x for v in HC.V if v not in bV])
    f_net = fp + fv
    ax3b.scatter(y_int, f_net, s=10, alpha=0.5, color='tab:purple')
    ax3b.axhline(0, color='gray', lw=0.5, ls='--')
    ax3b.set_xlabel('y')
    ax3b.set_ylabel('$F^{pres}_x + F^{visc}_x$ (net)')
    ax3b.set_title(f'Net x-force (should be ~0)')
    ax3b.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    ax3b.grid(True, alpha=0.3)

    fig3.suptitle('Integrated Analytical Comparison')
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.savefig(os.path.join(_FIG, 'integrated_comparison.png'), dpi=150)
    plt.close(fig3)


if __name__ == "__main__":
    main()
