"""2D Hydrostatic column — pressure operator validation.

1 m x H m rectangle, gravity along y-axis. No-slip walls + free surface at top.
Gauge pressure, artificial viscosity, adaptive CFL.
Compares against compressible analytical: P(y) = K*(exp(rho0*g*(H-y)/K) - 1).
"""

import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

H = 1.0; rho = 1000.0; g = 9.81; mu_phys = 1e-3; n_refine = 3
gravity_axis = 1; P_atm = 101325.0

_FIG_DIR = os.path.join(os.path.dirname(__file__), 'fig')
os.makedirs(_FIG_DIR, exist_ok=True)

def _savefig(fig, bn):
    fig.savefig(os.path.join(_FIG_DIR, f'{bn}.pdf'), dpi=150)
    fig.savefig(os.path.join(_FIG_DIR, f'{bn}.png'), dpi=150)
    print(f"  -> fig/{bn}.pdf, fig/{bn}.png")

K_eos = rho * (10 * np.sqrt(g * H))**2
def P_comp(y): return K_eos * (np.exp(rho*g*(H-y)/K_eos) - 1.0)
def P_incomp(y): return rho * g * (H - y)
def P_abs(y): return P_atm + rho * g * (H - y)

print("=" * 60)
print("2D Hydrostatic Column"); print("=" * 60)

from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic_column, make_gravity_dudt
from ddgclib.eos import TaitMurnaghan
from ddgclib.initial_conditions import DualVolumeMass, ZeroVelocity, HydrostaticEOSMass
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.dynamic_integrators._integrators_dynamic import _recompute_duals, _interior_verts, _move
from ddgclib.visualization.unified import plot_primal

# === Section 1-2: Static checks (absolute pressure) ===
HC, bV, bc_set, _, dudt_fn = setup_hydrostatic_column(
    dim=2, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=P_atm, mu=mu_phys,
    gravity_axis=gravity_axis,
)
n_verts = sum(1 for _ in HC.V)
interior = [v for v in HC.V if v not in bV]
print(f"Mesh: {n_verts} vertices, {len(interior)} interior")

fig1, ax1 = plot_primal(HC, bV=bV, scalar_field='p', title='2D Pressure (abs)',
                        save_path=None, vertex_size=30, cmap='coolwarm')
fig1.tight_layout(); _savefig(fig1, 'hydrostatic_2d_mesh_pressure')

# Section 2: static residual
res = np.array([np.linalg.norm(dudt_fn(v)) for v in interior])
print(f"\nSection 2: Static residual — max|a|={np.max(res):.4f}, mean={np.mean(res):.4f} m/s^2")

# Section 2b: EOS residual
eos_abs = TaitMurnaghan(rho0=rho, P0=P_atm, K=K_eos, n=1.0, rho_clip=(0.5,2.0))
c0 = float(eos_abs.sound_speed(rho))
HydrostaticEOSMass(eos=eos_abs, rho0=rho, g=g, gravity_axis=gravity_axis,
                   h_ref=H, P_ref=P_atm).apply(HC, bV)
dudt_eos = make_gravity_dudt(dim=2, mu=mu_phys, HC=HC, g=g, gravity_axis=gravity_axis,
                             pressure_model=eos_abs)
res_eos = np.array([np.linalg.norm(dudt_eos(v)) for v in interior])
print(f"Section 2b: EOS residual — max|a|={np.max(res_eos):.4f} m/s^2")

# === Section 3: Dynamic settling (gauge, free surface) ===
print(f"\nSection 3: Dynamic settling (gauge, free surface, no-slip walls)")
eos = TaitMurnaghan(rho0=rho, P0=0.0, K=K_eos, n=1.0, rho_clip=(0.5,2.0))

HC, bV, bc_set, _, _ = setup_hydrostatic_column(
    dim=2, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=0.0, mu=mu_phys,
    gravity_axis=gravity_axis, free_surface=True,
)
ZeroVelocity(dim=2).apply(HC, bV)
DualVolumeMass(rho=rho).apply(HC, bV)
interior = [v for v in HC.V if v not in bV]

edges = [np.linalg.norm(v.x_a[:2]-nb.x_a[:2]) for v in HC.V for nb in v.nn]
dx_mean, dx_min_init = np.mean(edges), min(edges)

alpha = 0.5; mu_art = alpha * rho * c0 * dx_mean
print(f"  ARTIFICIAL VISCOSITY: mu_art={mu_art:.0f} (alpha={alpha})")
print(f"  c0={c0:.1f}, K={K_eos:.2e}")

dudt_dyn = make_gravity_dudt(dim=2, mu=mu_art, HC=HC, g=g,
                             gravity_axis=gravity_axis, pressure_model=eos)

CFL = 0.25; t_ac = H/c0; n_trav = 100; t_end = n_trav * t_ac
print(f"  {n_trav} traversals → t_end={t_end:.4f} s")

KE_h, t_h, u_h, Pe_i_h, Pe_c_h, a_h = [], [], [], [], [], []
t, step = 0.0, 0
rec = max(1, int(0.3 * t_ac / (CFL * dx_min_init / c0)))

while t < t_end:
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=2)
    iv = _interior_verts(HC, bV)
    u_max = max((np.linalg.norm(v.u[:2]) for v in iv), default=0.0)
    dx_min = min((np.linalg.norm(v.x_a[:2]-nb.x_a[:2]) for v in iv for nb in v.nn
                  if np.linalg.norm(v.x_a[:2]-nb.x_a[:2])>0), default=dx_min_init)
    dt = min(CFL*dx_min/(c0+u_max), t_end-t)

    acc = {v: dudt_dyn(v) for v in iv}
    for v in iv:
        v.u[:2] += dt*acc[v][:2]
        _move(v, v.x_a[:2]+dt*v.u[:2], HC, bV)
    if bc_set: bc_set.apply_all(HC, bV, dt)
    t += dt; step += 1

    if step % rec == 0 or t >= t_end:
        KE_h.append(sum(0.5*v.m*np.dot(v.u[:2],v.u[:2]) for v in HC.V if v not in bV))
        u_h.append(max((np.linalg.norm(v.u[:2]) for v in HC.V if v not in bV), default=0.0))
        t_h.append(t)
        from ddgclib.analytical._integrated_comparison import (
            integrated_pressure_error as _ipe,
        )
        _ie_c = _ipe(HC, list(acc.keys()),
                      P_analytical=lambda x: P_comp(x[1]), dim=2)
        _ie_i = _ipe(HC, list(acc.keys()),
                      P_analytical=lambda x: P_incomp(x[1]), dim=2)
        Pe_i_h.append(max(_ie_i) if _ie_i else 0.0)
        Pe_c_h.append(max(_ie_c) if _ie_c else 0.0)
        a_h.append(max(np.linalg.norm(a) for a in acc.values()))

    if step % 500 == 0:
        print(f"  step {step}, t/t_ac={t/t_ac:.1f}, u_max={u_max:.4f}")
    if u_max > 10*c0: print(f"  ABORT"); break

KE_a, t_a, u_a = np.array(KE_h), np.array(t_h), np.array(u_h)
Pe_i, Pe_c, a_a = np.array(Pe_i_h), np.array(Pe_c_h), np.array(a_h)

print(f"  Done: {step} steps, t={t:.4f} s (~{t/t_ac:.0f} t_ac)")
print(f"  KE={KE_a[-1]:.6e}, |u|={u_a[-1]:.6e}")
print(f"  Integrated P err: max|p*V-∫P dV| (incomp)={Pe_i[-1]:.4e}, (comp)={Pe_c[-1]:.4e}")
print(f"  Max |a|={a_a[-1]:.6e}")

# Check final force balance
_recompute_duals(HC); cache_dual_volumes(HC, dim=2)
interior = list(_interior_verts(HC, bV))
a_settled = [np.linalg.norm(dudt_dyn(v)) for v in interior]
print(f"  Settled force balance: max|a|={max(a_settled):.6e}")

# === PLOTS ===
fig2, (a1,a2) = plt.subplots(1,2,figsize=(12,4))
if np.any(KE_a>0): a1.semilogy(t_a/t_ac, KE_a, 'b-', lw=1)
a1.set_xlabel('$t/t_{ac}$'); a1.set_ylabel('KE'); a1.set_title('KE'); a1.grid(True, alpha=0.3)
a2.plot(t_a/t_ac, u_a, 'r-', lw=1)
a2.set_xlabel('$t/t_{ac}$'); a2.set_ylabel('Max |u|'); a2.set_title('Velocity'); a2.grid(True, alpha=0.3)
fig2.suptitle(f'2D Settling ($\\mu_{{art}}$={mu_art:.0f})'); fig2.tight_layout()
_savefig(fig2, 'hydrostatic_2d_dynamic_settling')

fig3, (a3,a4) = plt.subplots(1,2,figsize=(12,4))
a3.semilogy(t_a/t_ac, Pe_c, 'r-', lw=1, label='vs compressible')
a3.semilogy(t_a/t_ac, Pe_i, 'b-', lw=1, label='vs incompressible')
a3.set_xlabel('$t/t_{ac}$'); a3.set_ylabel('Max $|pV - \\int P dV|$')
a3.set_title('Integrated Pressure Error'); a3.legend(); a3.grid(True, alpha=0.3)
a4.semilogy(t_a/t_ac, a_a, 'k-', lw=1)
a4.set_xlabel('$t/t_{ac}$'); a4.set_ylabel('Max |a| [m/s²]')
a4.set_title('Force Residual'); a4.grid(True, alpha=0.3)
fig3.suptitle('2D Diagnostics'); fig3.tight_layout()
_savefig(fig3, 'hydrostatic_2d_diagnostics')

fig4, ax4 = plt.subplots(figsize=(6,6))
y_all = np.array([v.x_a[1] for v in HC.V])
P_all = np.array([v.p for v in HC.V])
yf = np.linspace(0,H,200)
ax4.plot([P_comp(y) for y in yf], yf, 'g--', lw=1.5, label='Compressible')
ax4.plot([P_incomp(y) for y in yf], yf, 'k--', lw=1.5, label='Incompressible')
ax4.plot(P_all, y_all, 'o', ms=3, alpha=0.5, label='DDG')
ax4.set_xlabel('P [Pa]'); ax4.set_ylabel('y [m]')
ax4.set_title(f'2D Final Profile (t={t:.2f} s)'); ax4.legend(); ax4.grid(True, alpha=0.3)
fig4.tight_layout(); _savefig(fig4, 'hydrostatic_2d_final_profile')

from ddgclib.analytical._integrated_comparison import (
    integrated_pressure_error, integrated_l2_norm, compare_stress_force,
)
cache_dual_volumes(HC, dim=2)

int_errs = integrated_pressure_error(
    HC, interior, P_analytical=lambda x: P_comp(x[1]), dim=2,
)
int_l2 = integrated_l2_norm(
    HC, interior, P_analytical=lambda x: P_comp(x[1]), dim=2,
)
force_diag = compare_stress_force(HC, interior, dim=2, mu=mu_art)

print(f"\n  Final integrated error:  max|p*V - ∫P dV| = {max(int_errs):.4e}")
print(f"  Integrated L2 norm = {int_l2:.4e} Pa")
print(f"  Force balance: max|F| = {force_diag['max_F']:.4e}, "
      f"median|F| = {force_diag['median_F']:.4e}")

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
y_int = [v.x_a[1] for v in interior]
ax5a.scatter(y_int, int_errs, s=10, alpha=0.5)
ax5a.set_xlabel('y [m]'); ax5a.set_ylabel('$|p_i V_i - \\int P \\, dV|$')
ax5a.set_title('Integrated Pressure Error (vs compressible)')
ax5a.grid(True, alpha=0.3)

ax5b.scatter(y_int, force_diag['F_norms'], s=10, alpha=0.5)
ax5b.axhline(0, color='k', ls='--', lw=0.5)
ax5b.set_xlabel('y [m]'); ax5b.set_ylabel('$||F_{stress}||$')
ax5b.set_title('Force Balance on Final Mesh')
ax5b.grid(True, alpha=0.3)
ax5b.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
fig5.suptitle('2D Integrated Analytical Comparison')
fig5.tight_layout(); _savefig(fig5, 'hydrostatic_2d_integrated')

print("\nDone.")
