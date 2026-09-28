"""1D Hydrostatic column — pressure operator validation.

Tests the face-averaged pressure force in the DDG stress operator for a
1D simplicial chain under gravity.

Sections
--------
1. Domain setup & prescribed pressure profile
2. Static equilibrium residual (DDG operator check, no EOS)
2b. EOS equilibrium residual (compressible balance)
3. Dynamic settling from far-from-equilibrium
   - Free surface at top, gauge pressure (P0=0)
   - ARTIFICIAL VISCOSITY for acoustic damping
   - Adaptive CFL, _recompute_duals + cache_dual_volumes each step
   - Tracks: KE, max|u|, pressure error, force residual over time
   - Compares against COMPRESSIBLE analytical P(x) = K*(exp(rho0*g*(H-x)/K)-1)

Outputs: fig/ (PDF + PNG) + mp4 animation.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
H = 10.0; rho = 1000.0; g = 9.81; mu_phys = 1e-3
n_refine = 4; gravity_axis = 0
P_atm = 101325.0  # for sections 1-2 (absolute pressure)

_FIG_DIR = os.path.join(os.path.dirname(__file__), 'fig')
os.makedirs(_FIG_DIR, exist_ok=True)


def _savefig(fig, bn):
    fig.savefig(os.path.join(_FIG_DIR, f'{bn}.pdf'), dpi=150)
    fig.savefig(os.path.join(_FIG_DIR, f'{bn}.png'), dpi=150)
    print(f"  -> fig/{bn}.pdf, fig/{bn}.png")


# Analytical solutions
def P_abs(x):
    """Incompressible hydrostatic (absolute)."""
    return P_atm + rho * g * (H - x)


# EOS parameters (used in sections 2b and 3)
c0_min = 10.0 * np.sqrt(g * H)
K_eos = rho * c0_min**2  # linear EOS stiffness


def P_compressible(x):
    """Compressible hydrostatic (gauge, linear EOS n=1).

    dP/dx = -rho(x)*g, rho = rho0*(1+P/K)
    => P(x) = K*(exp(rho0*g*(H-x)/K) - 1)
    """
    return K_eos * (np.exp(rho * g * (H - x) / K_eos) - 1.0)


def P_incompressible(x):
    """Incompressible hydrostatic (gauge)."""
    return rho * g * (H - x)


# ===========================================================================
# Section 1: Domain setup & pressure profile
# ===========================================================================
print("=" * 60)
print("1D Hydrostatic Column — Pressure Operator Validation")
print("=" * 60)

from cases_dynamic.Hydrostatic_column.src._setup import (
    setup_hydrostatic_column, make_gravity_dudt,
)

HC, bV, bc_set, params, dudt_fn = setup_hydrostatic_column(
    dim=1, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=P_atm, mu=mu_phys,
    gravity_axis=gravity_axis,
)

n_verts = sum(1 for _ in HC.V)
interior = [v for v in HC.V if v not in bV]
print(f"Mesh: {n_verts} vertices, {len(interior)} interior, {len(bV)} boundary")

from ddgclib.visualization import plot_scalar_field_1d

fig1, ax1 = plt.subplots(figsize=(8, 4))
plot_scalar_field_1d(
    HC, field='p', ax=ax1, label='DDG',
    analytical_fn=P_abs, analytical_label='Analytical',
    xlabel='x [m]', ylabel='P [Pa]',
    title='1D Prescribed Pressure Profile (absolute)',
)
fig1.tight_layout()
_savefig(fig1, 'hydrostatic_1d_pressure')

# ===========================================================================
# Section 2: Static equilibrium residual
# ===========================================================================
print(f"\nSection 2: Static equilibrium residual (prescribed P)")
residuals = np.array([dudt_fn(v)[0] for v in interior])
print(f"  Max |a_total| = {np.max(np.abs(residuals)):.6e} m/s^2 (exact for 1D)")

# ===========================================================================
# Section 2b: EOS equilibrium residual
# ===========================================================================
print(f"\nSection 2b: EOS equilibrium residual")

from ddgclib.eos import TaitMurnaghan
from ddgclib.initial_conditions import HydrostaticEOSMass

eos_abs = TaitMurnaghan(rho0=rho, P0=P_atm, K=K_eos, n=1.0, rho_clip=(0.5, 2.0))
c0 = float(eos_abs.sound_speed(rho))
print(f"  EOS: P0={P_atm:.0f}, K={K_eos:.2e}, c0={c0:.1f} m/s")

HydrostaticEOSMass(
    eos=eos_abs, rho0=rho, g=g, gravity_axis=gravity_axis,
    h_ref=H, P_ref=P_atm,
).apply(HC, bV)

dudt_eos = make_gravity_dudt(
    dim=1, mu=mu_phys, HC=HC, g=g, gravity_axis=0, pressure_model=eos_abs,
)
res_eos = np.array([dudt_eos(v)[0] for v in interior])
print(f"  Max |a_total| = {np.max(np.abs(res_eos)):.4f} m/s^2")
print(f"  (Expected: ~g*delta_rho/rho0 = {g*rho*g*H/K_eos:.4f} m/s^2 from compressibility)")

# ===========================================================================
# Section 3: Dynamic settling (gauge, free surface, art. viscosity)
# ===========================================================================
print(f"\nSection 3: Dynamic settling")
print(f"  Gauge pressure (P0=0). Free surface at top.")
print(f"  Compressible analytical: P(x) = K*(exp(rho0*g*(H-x)/K) - 1)")

from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.initial_conditions import DualVolumeMass, ZeroVelocity
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _recompute_duals, _interior_verts, _move,
)

eos_gauge = TaitMurnaghan(rho0=rho, P0=0.0, K=K_eos, n=1.0, rho_clip=(0.5, 2.0))

HC, bV, bc_set, _, _ = setup_hydrostatic_column(
    dim=1, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=0.0, mu=mu_phys,
    gravity_axis=gravity_axis, free_surface=True,
)
ZeroVelocity(dim=1).apply(HC, bV)
DualVolumeMass(rho=rho).apply(HC, bV)
interior = [v for v in HC.V if v not in bV]
interior_ordered = sorted(interior, key=lambda v: v.x_a[0])

print(f"  {n_verts} vertices, {len(bV)} frozen (bottom), {len(interior)} free")

# ARTIFICIAL VISCOSITY
dx0 = H / (n_verts - 1)
alpha_visc = 0.5
mu_art = alpha_visc * rho * c0 * dx0
print(f"  ARTIFICIAL VISCOSITY: mu_art = {mu_art:.0f} Pa.s (alpha={alpha_visc})")

dudt_dyn = make_gravity_dudt(
    dim=1, mu=mu_art, HC=HC, g=g, gravity_axis=0, pressure_model=eos_gauge,
)

CFL = 0.25
t_acoustic = H / c0
n_traversals = 200
t_end = n_traversals * t_acoustic
print(f"  CFL={CFL}, c0={c0:.1f}, {n_traversals} traversals → t_end={t_end:.2f} s")

# Diagnostics storage
KE_hist, time_hist, maxu_hist = [], [], []
int_P_err_comp_hist, int_P_err_incomp_hist = [], []
a_max_hist = []

from ddgclib.analytical._integrated_comparison import (
    integrated_pressure_error, integrated_l2_norm,
)
frames = []  # (t, x_arr, P_arr) for movie

t, step = 0.0, 0
rec = max(1, int(0.2 * t_acoustic / (CFL * dx0 / c0)))

while t < t_end:
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=1)

    int_verts = _interior_verts(HC, bV)
    u_max = max((abs(v.u[0]) for v in int_verts), default=0.0)

    positions = sorted(v.x_a[0] for v in HC.V)
    dx_min = max(min(positions[i+1]-positions[i] for i in range(len(positions)-1)), 1e-15)

    dt = min(CFL * dx_min / (c0 + u_max), t_end - t)

    accel = {v: dudt_dyn(v) for v in int_verts}
    for v in int_verts:
        v.u[:1] += dt * accel[v][:1]
        _move(v, v.x_a[:1] + dt * v.u[:1], HC, bV)

    if bc_set:
        bc_set.apply_all(HC, bV, dt)
    t += dt
    step += 1

    if step % rec == 0 or t >= t_end:
        ke = sum(0.5 * v.m * v.u[0]**2 for v in HC.V if v not in bV)
        KE_hist.append(ke)
        maxu_hist.append(max((abs(v.u[0]) for v in HC.V if v not in bV), default=0.0))
        time_hist.append(t)

        # Integrated pressure error vs both analyticals
        ie_c = integrated_pressure_error(
            HC, interior_ordered,
            P_analytical=lambda x: P_compressible(x[0]), dim=1,
        )
        ie_i = integrated_pressure_error(
            HC, interior_ordered,
            P_analytical=lambda x: P_incompressible(x[0]), dim=1,
        )
        int_P_err_comp_hist.append(max(ie_c) if ie_c else 0.0)
        int_P_err_incomp_hist.append(max(ie_i) if ie_i else 0.0)

        # Force residual
        a_norms = [abs(accel[v][0]) if v in accel else 0.0 for v in interior_ordered]
        a_max_hist.append(max(a_norms))

        # Frame for movie
        x_f = np.array([v.x_a[0] for v in HC.V])
        P_f = np.array([v.p for v in HC.V])
        frames.append((t, x_f.copy(), P_f.copy()))

    if u_max > 10 * c0:
        print(f"  WARNING: u_max > 10*c0 at step {step} — aborting")
        break

print(f"  Completed {step} steps in {t:.2f} s (~{t/t_acoustic:.0f} traversals)")

KE_arr = np.array(KE_hist)
time_arr = np.array(time_hist)
maxu_arr = np.array(maxu_hist)
int_P_err_comp = np.array(int_P_err_comp_hist)
int_P_err_incomp = np.array(int_P_err_incomp_hist)
a_max_arr = np.array(a_max_hist)

int_l2_comp = integrated_l2_norm(
    HC, interior_ordered,
    P_analytical=lambda x: P_compressible(x[0]), dim=1,
)
int_l2_incomp = integrated_l2_norm(
    HC, interior_ordered,
    P_analytical=lambda x: P_incompressible(x[0]), dim=1,
)

print(f"  Final KE = {KE_arr[-1]:.6e}, Max |u| = {maxu_arr[-1]:.6e}")
print(f"  Integrated P error (vs compressible):   max|p*V - ∫P dV| = {int_P_err_comp[-1]:.4e}, L2 = {int_l2_comp:.4e}")
print(f"  Integrated P error (vs incompressible): max|p*V - ∫P dV| = {int_P_err_incomp[-1]:.4e}, L2 = {int_l2_incomp:.4e}")
print(f"  Max |a_total| = {a_max_arr[-1]:.6e} m/s^2")

# ===== PLOTS =====

# 1. KE + max velocity
fig_ke, (ax_ke, ax_u) = plt.subplots(1, 2, figsize=(12, 4))
if np.any(KE_arr > 0):
    ax_ke.semilogy(time_arr / t_acoustic, KE_arr, 'b-', lw=1)
ax_ke.set_xlabel('$t / t_{ac}$'); ax_ke.set_ylabel('KE [J/m²]')
ax_ke.set_title('Kinetic Energy'); ax_ke.grid(True, alpha=0.3)

ax_u.plot(time_arr / t_acoustic, maxu_arr, 'r-', lw=1)
ax_u.set_xlabel('$t / t_{ac}$'); ax_u.set_ylabel('Max |u| [m/s]')
ax_u.set_title('Max Velocity'); ax_u.grid(True, alpha=0.3)
fig_ke.suptitle(f'1D Settling (c0={c0:.0f}, $\\mu_{{art}}$={mu_art:.0f})')
fig_ke.tight_layout()
_savefig(fig_ke, 'hydrostatic_1d_dynamic_settling')

# 2. Integrated pressure error + force residual over time
fig_diag, (ax_pe, ax_fr) = plt.subplots(1, 2, figsize=(12, 4))
ax_pe.semilogy(time_arr / t_acoustic, int_P_err_comp, 'r-', lw=1, label='vs compressible')
ax_pe.semilogy(time_arr / t_acoustic, int_P_err_incomp, 'b-', lw=1, label='vs incompressible')
ax_pe.set_xlabel('$t / t_{ac}$'); ax_pe.set_ylabel('Max |$p V - \\int P dV$|')
ax_pe.set_title('Integrated Pressure Error Over Time'); ax_pe.legend()
ax_pe.grid(True, alpha=0.3)

ax_fr.semilogy(time_arr / t_acoustic, a_max_arr, 'k-', lw=1)
ax_fr.set_xlabel('$t / t_{ac}$'); ax_fr.set_ylabel('Max |$a_{total}$| [m/s²]')
ax_fr.set_title('Force Residual Over Time'); ax_fr.grid(True, alpha=0.3)
fig_diag.suptitle('1D Convergence Diagnostics')
fig_diag.tight_layout()
_savefig(fig_diag, 'hydrostatic_1d_diagnostics')

# 3. Final pressure profile (vs BOTH analyticals)
fig_prof, ax_prof = plt.subplots(figsize=(8, 4))
x_final = np.array([v.x_a[0] for v in HC.V])
P_final = np.array([v.p for v in HC.V])
idx = np.argsort(x_final)
x_fine = np.linspace(0, H, 300)
ax_prof.plot(x_fine, [P_incompressible(x) for x in x_fine], 'k--', lw=1.5,
             label='Analytical (incompressible)')
ax_prof.plot(x_fine, [P_compressible(x) for x in x_fine], 'g--', lw=1.5,
             label='Analytical (compressible)')
ax_prof.plot(x_final[idx], P_final[idx], 'o-', ms=3, color='C0', label='DDG (settled)')
ax_prof.set_xlabel('x [m]'); ax_prof.set_ylabel('P [Pa]')
ax_prof.set_title(f'1D Final Pressure Profile (t = {t:.2f} s, gauge)')
ax_prof.legend(); ax_prof.grid(True, alpha=0.3)
fig_prof.tight_layout()
_savefig(fig_prof, 'hydrostatic_1d_final_profile')

# 4. Integrated pressure error (spatial distribution)
fig_dev, ax_dev = plt.subplots(figsize=(8, 4))
x_dev = np.array([v.x_a[0] for v in interior_ordered])
int_errs_final = integrated_pressure_error(
    HC, interior_ordered,
    P_analytical=lambda x: P_compressible(x[0]), dim=1,
)
ax_dev.plot(x_dev, int_errs_final, 'ro-', ms=3)
ax_dev.axhline(0, color='k', ls='--', lw=0.5)
ax_dev.set_xlabel('x [m]'); ax_dev.set_ylabel('$|p_i V_i - \\int P \\, dV|$')
ax_dev.set_title('Integrated Pressure Error (vs compressible)')
ax_dev.grid(True, alpha=0.3)
fig_dev.tight_layout()
_savefig(fig_dev, 'hydrostatic_1d_integrated_error')

# 5. MP4 animation
print(f"\n  Generating mp4 animation ({len(frames)} frames)...")

from matplotlib.animation import FuncAnimation

fig_anim, ax_anim = plt.subplots(figsize=(8, 4))
x_fine = np.linspace(0, H, 300)
ax_anim.plot(x_fine, [P_compressible(x) for x in x_fine], 'k--', lw=1.5,
             label='Compressible analytical')
line, = ax_anim.plot([], [], 'o-', ms=3, color='C0', label='DDG')
time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes,
                         fontsize=10, verticalalignment='top')
ax_anim.set_xlim(-0.5, H + 0.5)
ax_anim.set_ylim(-0.15 * rho * g * H, 1.15 * rho * g * H)
ax_anim.set_xlabel('x [m]'); ax_anim.set_ylabel('P [Pa]')
ax_anim.set_title('1D Hydrostatic Settling (gauge)')
ax_anim.legend(loc='upper right'); ax_anim.grid(True, alpha=0.3)


def _init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def _update(i):
    t_f, x_f, P_f = frames[i]
    idx_f = np.argsort(x_f)
    line.set_data(x_f[idx_f], P_f[idx_f])
    time_text.set_text(f't = {t_f:.4f} s  ({t_f/t_acoustic:.0f} $t_{{ac}}$)')
    return line, time_text


anim = FuncAnimation(fig_anim, _update, init_func=_init,
                     frames=len(frames), interval=50, blit=True)
mp4_path = os.path.join(_FIG_DIR, 'hydrostatic_1d_settling.mp4')
try:
    anim.save(mp4_path, fps=20, dpi=100,
              writer='ffmpeg', extra_args=['-pix_fmt', 'yuv420p'])
    print(f"  -> fig/hydrostatic_1d_settling.mp4")
except Exception as e:
    # Fallback to gif if ffmpeg unavailable
    gif_path = os.path.join(_FIG_DIR, 'hydrostatic_1d_settling.gif')
    from matplotlib.animation import PillowWriter
    anim.save(gif_path, writer=PillowWriter(fps=12))
    print(f"  -> fig/hydrostatic_1d_settling.gif (ffmpeg unavailable: {e})")
plt.close(fig_anim)

print("\nDone.")
