"""3D Hydrostatic column — pressure operator validation.

1 m x 1 m x H m box, gravity along z-axis. No-slip walls + free surface at top.
Gauge pressure, artificial viscosity, adaptive CFL.
Compares against compressible analytical: P(z) = K*(exp(rho0*g*(H-z)/K) - 1).
"""

import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

H = 1.0; rho = 1000.0; g = 9.81; mu_phys = 1e-3; n_refine = 2
gravity_axis = 2; P_atm = 101325.0

_FIG_DIR = os.path.join(os.path.dirname(__file__), 'fig')
os.makedirs(_FIG_DIR, exist_ok=True)

def _savefig(fig, bn):
    fig.savefig(os.path.join(_FIG_DIR, f'{bn}.pdf'), dpi=150)
    fig.savefig(os.path.join(_FIG_DIR, f'{bn}.png'), dpi=150)
    print(f"  -> fig/{bn}.pdf, fig/{bn}.png")

K_eos = rho * (10 * np.sqrt(g * H))**2
def P_comp(z): return K_eos * (np.exp(rho*g*(H-z)/K_eos) - 1.0)
def P_incomp(z): return rho * g * (H - z)
def P_abs(z): return P_atm + rho * g * (H - z)

print("=" * 60)
print("3D Hydrostatic Column"); print("=" * 60)

from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic_column, make_gravity_dudt
from ddgclib.eos import TaitMurnaghan
from ddgclib.initial_conditions import DualVolumeMass, ZeroVelocity, HydrostaticEOSMass
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.dynamic_integrators._integrators_dynamic import _recompute_duals, _interior_verts, _move
from ddgclib.visualization.unified import plot_primal

# === Section 1-2: Static checks (absolute pressure) ===
HC, bV, bc_set, _, dudt_fn = setup_hydrostatic_column(
    dim=3, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=P_atm, mu=mu_phys,
    gravity_axis=gravity_axis,
)
n_verts = sum(1 for _ in HC.V)
interior = [v for v in HC.V if v not in bV]
print(f"Mesh: {n_verts} vertices, {len(interior)} interior, {len(bV)} boundary")

fig1, ax1 = plot_primal(HC, bV=bV, scalar_field='p', title='3D Pressure (abs)',
                        save_path=None, vertex_size=40, cmap='coolwarm',
                        face_alpha=0.08, pointsize=2)
fig1.tight_layout(); _savefig(fig1, 'hydrostatic_3d_mesh_pressure')

# Section 2: static residual
res = np.array([np.linalg.norm(dudt_fn(v)) for v in interior])
print(f"\nSection 2: Static residual — max|a|={np.max(res):.4f}, mean={np.mean(res):.4f} m/s^2")

# Pressure profile at center
tol_xy = 0.15
col = [(v.x_a[2], v.p) for v in HC.V
       if abs(v.x_a[0]-0.5) < tol_xy and abs(v.x_a[1]-0.5) < tol_xy]
if not col: col = [(v.x_a[2], v.p) for v in HC.V]
col.sort()
z_col, P_col = np.array([c[0] for c in col]), np.array([c[1] for c in col])
print(f"  Central column: {len(col)} vertices, "
      f"max|P-P_anal| = {np.max(np.abs(P_col - np.array([P_abs(z) for z in z_col]))):.6e} Pa")

fig_prof, ax_prof = plt.subplots(figsize=(6, 6))
z_fine = np.linspace(0, H, 200)
ax_prof.plot([P_abs(z) for z in z_fine], z_fine, 'k--', lw=1.5, label='Analytical')
ax_prof.plot(P_col, z_col, 'o', ms=5, label='DDG')
ax_prof.set_xlabel('P [Pa]'); ax_prof.set_ylabel('z [m]')
ax_prof.set_title('3D Vertical Pressure Profile'); ax_prof.legend(); ax_prof.grid(True, alpha=0.3)
fig_prof.tight_layout(); _savefig(fig_prof, 'hydrostatic_3d_profile')

# Section 2b: EOS residual (now using COMPRESSIBLE HydrostaticEOSMass)
eos_abs = TaitMurnaghan(rho0=rho, P0=P_atm, K=K_eos, n=1.0, rho_clip=(0.5,2.0))
c0 = float(eos_abs.sound_speed(rho))
print(f"\nSection 2b: EOS residual (compressible HydrostaticEOSMass)")
print(f"  EOS: P0={P_atm:.0f}, K={K_eos:.2e}, c0={c0:.1f} m/s")

HydrostaticEOSMass(eos=eos_abs, rho0=rho, g=g, gravity_axis=gravity_axis,
                   h_ref=H, P_ref=P_atm).apply(HC, bV)
dudt_eos = make_gravity_dudt(dim=3, mu=mu_phys, HC=HC, g=g, gravity_axis=gravity_axis,
                             pressure_model=eos_abs)
res_eos = np.array([np.linalg.norm(dudt_eos(v)) for v in interior])
print(f"  Max |a_total| = {np.max(res_eos):.6e} m/s^2 (should match Section 2's ~{np.max(res):.2f})")

# === Section 3: Dynamic settling (gauge, free surface) ===
print(f"\nSection 3: Dynamic settling (gauge, free surface, no-slip walls)")
eos = TaitMurnaghan(rho0=rho, P0=0.0, K=K_eos, n=1.0, rho_clip=(0.5,2.0))

HC, bV, bc_set, _, _ = setup_hydrostatic_column(
    dim=3, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=0.0, mu=mu_phys,
    gravity_axis=gravity_axis, free_surface=True,
)
ZeroVelocity(dim=3).apply(HC, bV)
DualVolumeMass(rho=rho).apply(HC, bV)

edges = [np.linalg.norm(v.x_a[:3]-nb.x_a[:3]) for v in HC.V for nb in v.nn]
dx_mean, dx_min_init = np.mean(edges), min(edges)

alpha = 0.5; mu_art = alpha * rho * c0 * dx_mean
print(f"  ARTIFICIAL VISCOSITY: mu_art={mu_art:.0f} (alpha={alpha})")
print(f"  c0={c0:.1f}, K={K_eos:.2e}")

dudt_dyn = make_gravity_dudt(dim=3, mu=mu_art, HC=HC, g=g,
                             gravity_axis=gravity_axis, pressure_model=eos)

CFL = 0.25; t_ac = H/c0; n_trav = 15; t_end = n_trav * t_ac
print(f"  {n_trav} traversals → t_end={t_end:.4f} s")

KE_h, t_h, u_h, Pe_i_h, Pe_c_h, a_h = [], [], [], [], [], []
t, step = 0.0, 0
rec = max(1, int(0.3 * t_ac / (CFL * dx_min_init / c0)))

while t < t_end:
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=3)
    iv = _interior_verts(HC, bV)
    u_max = max((np.linalg.norm(v.u[:3]) for v in iv), default=0.0)
    dx_min = min((np.linalg.norm(v.x_a[:3]-nb.x_a[:3]) for v in iv for nb in v.nn
                  if np.linalg.norm(v.x_a[:3]-nb.x_a[:3])>0), default=dx_min_init)
    dt = min(CFL*dx_min/(c0+u_max), t_end-t)

    acc = {v: dudt_dyn(v) for v in iv}
    for v in iv:
        v.u[:3] += dt*acc[v][:3]
        _move(v, v.x_a[:3]+dt*v.u[:3], HC, bV)
    if bc_set: bc_set.apply_all(HC, bV, dt)
    t += dt; step += 1

    if step % rec == 0 or t >= t_end:
        KE_h.append(sum(0.5*v.m*np.dot(v.u[:3],v.u[:3]) for v in HC.V if v not in bV))
        u_h.append(max((np.linalg.norm(v.u[:3]) for v in HC.V if v not in bV), default=0.0))
        t_h.append(t)
        Pe_i_h.append(max(abs(v.p-P_incomp(v.x_a[2])) for v in acc.keys())/(rho*g*H)*100)
        Pe_c_h.append(max(abs(v.p-P_comp(v.x_a[2])) for v in acc.keys())/(rho*g*H)*100)
        a_h.append(max(np.linalg.norm(a) for a in acc.values()))

    if step % 100 == 0:
        print(f"  step {step}, t/t_ac={t/t_ac:.1f}, u_max={u_max:.4f}, dt={dt:.2e}")
    if u_max > 10*c0: print(f"  ABORT"); break

KE_a, t_a, u_a = np.array(KE_h), np.array(t_h), np.array(u_h)
Pe_i, Pe_c, a_a = np.array(Pe_i_h), np.array(Pe_c_h), np.array(a_h)

print(f"  Done: {step} steps, t={t:.4f} s (~{t/t_ac:.0f} t_ac)")
print(f"  KE={KE_a[-1]:.6e}, |u|={u_a[-1]:.6e}")
print(f"  P err (incomp)={Pe_i[-1]:.2f}%, (comp)={Pe_c[-1]:.2f}%")
print(f"  Max |a|={a_a[-1]:.6e}")

# Check final force balance
_recompute_duals(HC); cache_dual_volumes(HC, dim=3)
interior = list(_interior_verts(HC, bV))
a_settled = [np.linalg.norm(dudt_dyn(v)) for v in interior]
print(f"  Settled force balance: max|a|={max(a_settled):.6e}")

# === PLOTS ===
fig2, (a1,a2) = plt.subplots(1,2,figsize=(12,4))
if np.any(KE_a>0): a1.semilogy(t_a/t_ac, KE_a, 'b-', lw=1)
a1.set_xlabel('$t/t_{ac}$'); a1.set_ylabel('KE'); a1.set_title('KE'); a1.grid(True, alpha=0.3)
a2.plot(t_a/t_ac, u_a, 'r-', lw=1)
a2.set_xlabel('$t/t_{ac}$'); a2.set_ylabel('Max |u|'); a2.set_title('Velocity'); a2.grid(True, alpha=0.3)
fig2.suptitle(f'3D Settling ($\\mu_{{art}}$={mu_art:.0f})'); fig2.tight_layout()
_savefig(fig2, 'hydrostatic_3d_dynamic_settling')

fig3, (a3,a4) = plt.subplots(1,2,figsize=(12,4))
a3.plot(t_a/t_ac, Pe_i, 'b-', lw=1, label='vs incompressible')
a3.plot(t_a/t_ac, Pe_c, 'r-', lw=1, label='vs compressible')
a3.set_xlabel('$t/t_{ac}$'); a3.set_ylabel('Max $|\\Delta P|$ [%]')
a3.set_title('Pressure Error'); a3.legend(); a3.grid(True, alpha=0.3)
a4.semilogy(t_a/t_ac, a_a, 'k-', lw=1)
a4.set_xlabel('$t/t_{ac}$'); a4.set_ylabel('Max |a| [m/s²]')
a4.set_title('Force Residual'); a4.grid(True, alpha=0.3)
fig3.suptitle('3D Diagnostics'); fig3.tight_layout()
_savefig(fig3, 'hydrostatic_3d_diagnostics')

fig4, ax4 = plt.subplots(figsize=(6,6))
z_all = np.array([v.x_a[gravity_axis] for v in HC.V])
P_all = np.array([v.p for v in HC.V])
zf = np.linspace(0,H,200)
ax4.plot([P_comp(z) for z in zf], zf, 'g--', lw=1.5, label='Compressible')
ax4.plot([P_incomp(z) for z in zf], zf, 'k--', lw=1.5, label='Incompressible')
ax4.plot(P_all, z_all, 'o', ms=3, alpha=0.5, label='DDG')
ax4.set_xlabel('P [Pa]'); ax4.set_ylabel('z [m]')
ax4.set_title(f'3D Final Profile (t={t:.2f} s)'); ax4.legend(); ax4.grid(True, alpha=0.3)
fig4.tight_layout(); _savefig(fig4, 'hydrostatic_3d_final_profile')

fig5, ax5 = plt.subplots(figsize=(8,4))
P_dev = [v.p - P_comp(v.x_a[gravity_axis]) for v in interior]
z_dev = [v.x_a[gravity_axis] for v in interior]
ax5.scatter(z_dev, P_dev, s=10, alpha=0.5)
ax5.axhline(0, color='k', ls='--', lw=0.5)
ax5.set_xlabel('z [m]'); ax5.set_ylabel('$P_{DDG} - P_{comp}$ [Pa]')
ax5.set_title('3D Pressure Deviation from Compressible Analytical')
ax5.grid(True, alpha=0.3); fig5.tight_layout()
_savefig(fig5, 'hydrostatic_3d_pressure_deviation')

print("\nDone.")
