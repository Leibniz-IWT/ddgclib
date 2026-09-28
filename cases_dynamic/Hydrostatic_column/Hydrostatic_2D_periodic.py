"""2D Hydrostatic — free-slip sides (eliminates wall meniscus).

Side walls: ALL vertices are clamped horizontally to [0, L] (reflecting
boundary). Only bottom wall is no-slip frozen. Top is free surface.

Uses gauge pressure (P0=0), artificial viscosity, adaptive CFL.
Compares against compressible analytical: P(y) = K*(exp(rho0*g*(H-y)/K) - 1).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

H = 1.0; L = 1.0; rho = 1000.0; g = 9.81; mu_phys = 1e-3; n_refine = 3
gravity_axis = 1

_FIG_DIR = os.path.join(os.path.dirname(__file__), 'fig')
os.makedirs(_FIG_DIR, exist_ok=True)


def _savefig(fig, bn):
    fig.savefig(os.path.join(_FIG_DIR, f'{bn}.pdf'), dpi=150)
    fig.savefig(os.path.join(_FIG_DIR, f'{bn}.png'), dpi=150)
    print(f"  -> fig/{bn}.pdf, fig/{bn}.png")


K_eos = rho * (10 * np.sqrt(g * H))**2


def P_comp(y):
    return K_eos * (np.exp(rho * g * (H - y) / K_eos) - 1.0)


def P_incomp(y):
    return rho * g * (H - y)


print("=" * 60)
print("2D Hydrostatic — Free-slip (reflecting) sides")
print("=" * 60)

from cases_dynamic.Hydrostatic_column.src._setup import (
    setup_hydrostatic_column, make_gravity_dudt,
)
from ddgclib.geometry.domains._boundary_groups import identify_face_groups
from ddgclib.eos import TaitMurnaghan
from ddgclib.initial_conditions import DualVolumeMass, ZeroVelocity
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _recompute_duals, _interior_verts, _move,
)

eos = TaitMurnaghan(rho0=rho, P0=0.0, K=K_eos, n=1.0, rho_clip=(0.5, 2.0))
c0 = float(eos.sound_speed(rho))

HC, bV, bc_set, _, _ = setup_hydrostatic_column(
    dim=2, n_refine=n_refine, H=H, rho=rho, g=g, P_ref=0.0, mu=mu_phys,
    gravity_axis=gravity_axis, free_surface=True,
)
ZeroVelocity(dim=2).apply(HC, bV)
DualVolumeMass(rho=rho).apply(HC, bV)

groups = identify_face_groups(HC, {
    'bottom': (gravity_axis, 0.0),
    'left': (0, 0.0), 'right': (0, L),
})
bottom_verts = groups['bottom']

# Only freeze bottom wall; all other vertices are free
bV.clear()
bV.update(bottom_verts)

# Build set of vertices that started on a side wall (for initial free-slip)
tol = 1e-10
side_init = {v for v in HC.V if v not in bV
             and (v.x_a[0] <= 0.0 + tol or v.x_a[0] >= L - tol)}

n_verts = sum(1 for _ in HC.V)
interior = [v for v in HC.V if v not in bV]
print(f"Mesh: {n_verts} vertices, {len(bV)} frozen (bottom), "
      f"{len(side_init)} initial side verts, {len(interior)} free total")

edges = [np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) for v in HC.V for nb in v.nn]
dx_mean, dx_min_init = np.mean(edges), min(edges)

alpha_visc = 0.5
mu_art = alpha_visc * rho * c0 * dx_mean
print(f"ARTIFICIAL VISCOSITY: mu_art = {mu_art:.0f} Pa.s (alpha={alpha_visc})")
print(f"EOS: P0=0, K={K_eos:.2e}, c0={c0:.1f} m/s")

dudt_dyn = make_gravity_dudt(
    dim=2, mu=mu_art, HC=HC, g=g, gravity_axis=gravity_axis,
    pressure_model=eos,
)

CFL = 0.25
t_acoustic = H / c0
n_traversals = 100
t_end = n_traversals * t_acoustic
print(f"CFL={CFL}, {n_traversals} traversals -> t_end={t_end:.4f} s")

# Storage
KE_h, t_h, u_h, Pe_i_h, Pe_c_h, a_h = [], [], [], [], [], []
frames_mesh = []

t, step = 0.0, 0
rec = max(1, int(0.2 * t_acoustic / (CFL * dx_min_init / c0)))
aborted = False

while t < t_end:
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=2)

    int_verts = _interior_verts(HC, bV)
    u_max = max((np.linalg.norm(v.u[:2]) for v in int_verts), default=0.0)

    dx_min = float('inf')
    for v in int_verts:
        for nb in v.nn:
            d = np.linalg.norm(v.x_a[:2] - nb.x_a[:2])
            if d > 0:
                dx_min = min(dx_min, d)
    if dx_min == float('inf'):
        dx_min = dx_min_init

    dt = min(CFL * dx_min / (c0 + u_max), t_end - t)

    accel = {v: dudt_dyn(v) for v in int_verts}
    for v in int_verts:
        v.u[:2] += dt * accel[v][:2]
        x_new = v.x_a[:2] + dt * v.u[:2]

        # REFLECTING BOUNDARY: clamp x to [0, L] and reflect velocity
        if x_new[0] < 0.0:
            x_new[0] = -x_new[0]      # reflect position
            v.u[0] = abs(v.u[0])      # reflect velocity
        elif x_new[0] > L:
            x_new[0] = 2 * L - x_new[0]
            v.u[0] = -abs(v.u[0])

        # Vertices that started on a side wall: enforce free-slip (u_x = 0)
        if v in side_init:
            v.u[0] = 0.0

        _move(v, x_new, HC, bV)

    if bc_set:
        bc_set.apply_all(HC, bV, dt)
    t += dt
    step += 1

    if step % rec == 0 or t >= t_end or u_max > 2 * c0:
        ke = sum(0.5 * v.m * np.dot(v.u[:2], v.u[:2])
                 for v in HC.V if v not in bV)
        KE_h.append(ke)
        u_h.append(max((np.linalg.norm(v.u[:2]) for v in HC.V if v not in bV),
                       default=0.0))
        t_h.append(t)

        P_errs_i = [abs(v.p - P_incomp(v.x_a[gravity_axis]))
                    for v in int_verts if not np.isnan(v.p)]
        P_errs_c = [abs(v.p - P_comp(v.x_a[gravity_axis]))
                    for v in int_verts if not np.isnan(v.p)]
        Pe_i_h.append(max(P_errs_i) / (rho * g * H) * 100 if P_errs_i else 0)
        Pe_c_h.append(max(P_errs_c) / (rho * g * H) * 100 if P_errs_c else 0)

        a_norms = [np.linalg.norm(accel[v]) for v in int_verts if v in accel]
        a_h.append(max(a_norms) if a_norms else 0)

        xs = np.array([v.x_a[0] for v in HC.V])
        ys = np.array([v.x_a[gravity_axis] for v in HC.V])
        Ps = np.array([v.p for v in HC.V])
        frames_mesh.append((t, xs.copy(), ys.copy(), Ps.copy()))

    if step % 200 == 0:
        print(f"  step {step}, t/t_ac={t / t_acoustic:.1f}, "
              f"u_max={u_max:.4f}, dt={dt:.2e}")

    if u_max > 2 * c0:
        print(f"  INSTABILITY: u_max={u_max:.1f} > 2*c0={2*c0:.1f} at step "
              f"{step}, t/t_ac={t / t_acoustic:.1f} -- recording and aborting")
        aborted = True
        break

KE_a, t_a, u_a = np.array(KE_h), np.array(t_h), np.array(u_h)
Pe_i, Pe_c, a_a = np.array(Pe_i_h), np.array(Pe_c_h), np.array(a_h)

print(f"  {'ABORTED' if aborted else 'Completed'}: {step} steps in {t:.4f} s "
      f"(~{t / t_acoustic:.0f} traversals)")
print(f"  Final KE = {KE_a[-1]:.6e}, Max |u| = {u_a[-1]:.6e}")
print(f"  P err (incomp)={Pe_i[-1]:.2f}%, (comp)={Pe_c[-1]:.2f}%")

# Check final force balance (if not aborted)
if not aborted:
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=2)
    iv_final = list(_interior_verts(HC, bV))
    a_settled = [np.linalg.norm(dudt_dyn(v)) for v in iv_final]
    print(f"  Settled force balance: max|a|={max(a_settled):.6e}")

# ===== PLOTS =====
status = 'ABORTED' if aborted else 'Settled'

# 1. KE + max velocity
fig1, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
if np.any(KE_a > 0):
    a1.semilogy(t_a / t_acoustic, KE_a, 'b-', lw=1)
a1.set_xlabel('$t/t_{ac}$'); a1.set_ylabel('KE')
a1.set_title('KE'); a1.grid(True, alpha=0.3)
a2.plot(t_a / t_acoustic, u_a, 'r-', lw=1)
a2.axhline(c0, color='k', ls=':', lw=0.5, label=f'$c_0$={c0:.0f}')
a2.set_xlabel('$t/t_{ac}$'); a2.set_ylabel('Max |u| [m/s]')
a2.set_title('Max Velocity'); a2.legend(); a2.grid(True, alpha=0.3)
fig1.suptitle(f'2D Free-slip Settling ({status}, $\\mu_{{art}}$={mu_art:.0f})')
fig1.tight_layout()
_savefig(fig1, 'hydrostatic_2d_periodic_settling')

# 2. Pressure error + force residual
fig2, (a3, a4) = plt.subplots(1, 2, figsize=(12, 4))
a3.plot(t_a / t_acoustic, Pe_i, 'b-', lw=1, label='vs incompressible')
a3.plot(t_a / t_acoustic, Pe_c, 'r-', lw=1, label='vs compressible')
a3.set_xlabel('$t/t_{ac}$'); a3.set_ylabel('Max $|\\Delta P|$ [%]')
a3.set_title('Pressure Error'); a3.legend(); a3.grid(True, alpha=0.3)
a4.semilogy(t_a / t_acoustic, a_a, 'k-', lw=1)
a4.set_xlabel('$t/t_{ac}$'); a4.set_ylabel('Max |a| [m/s^2]')
a4.set_title('Force Residual'); a4.grid(True, alpha=0.3)
fig2.suptitle(f'2D Free-slip Diagnostics')
fig2.tight_layout()
_savefig(fig2, 'hydrostatic_2d_periodic_diagnostics')

# 3. Mesh snapshots at selected times
n_frames = len(frames_mesh)
n_snap = min(6, n_frames)
snap_idx = np.linspace(0, n_frames - 1, n_snap, dtype=int)

fig3, axes = plt.subplots(1, n_snap, figsize=(4 * n_snap, 5))
if n_snap == 1:
    axes = [axes]
for ax, idx in zip(axes, snap_idx):
    t_f, xs, ys, Ps = frames_mesh[idx]
    ax.scatter(xs, ys, c=Ps, s=15, cmap='coolwarm', vmin=0, vmax=rho * g * H)
    ax.set_title(f't={t_f:.3f}s\n({t_f / t_acoustic:.1f} $t_{{ac}}$)', fontsize=9)
    ax.set_xlim(-0.15, L + 0.15)
    ax.set_ylim(-0.15, H + 0.15)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
fig3.suptitle(f'2D Free-slip -- Mesh Evolution ({status})')
fig3.tight_layout()
_savefig(fig3, 'hydrostatic_2d_periodic_mesh_evolution')

# 4. Final pressure profile
fig4, ax4 = plt.subplots(figsize=(6, 6))
y_all = np.array([v.x_a[gravity_axis] for v in HC.V])
P_all = np.array([v.p for v in HC.V])
yf = np.linspace(0, H, 200)
ax4.plot([P_comp(y) for y in yf], yf, 'g--', lw=1.5, label='Compressible')
ax4.plot([P_incomp(y) for y in yf], yf, 'k--', lw=1.5, label='Incompressible')
ax4.plot(P_all, y_all, 'o', ms=3, alpha=0.5, label='DDG (free-slip)')
ax4.set_xlabel('P [Pa]'); ax4.set_ylabel('y [m]')
ax4.set_title(f'2D Free-slip -- Profile (t={t:.2f} s)')
ax4.legend(); ax4.grid(True, alpha=0.3)
fig4.tight_layout()
_savefig(fig4, 'hydrostatic_2d_periodic_profile')

# 5. Pressure deviation from compressible analytical
fig5, ax5 = plt.subplots(figsize=(8, 4))
P_dev = [v.p - P_comp(v.x_a[gravity_axis]) for v in HC.V if v not in bV]
y_dev = [v.x_a[gravity_axis] for v in HC.V if v not in bV]
ax5.scatter(y_dev, P_dev, s=10, alpha=0.5)
ax5.axhline(0, color='k', ls='--', lw=0.5)
ax5.set_xlabel('y [m]')
ax5.set_ylabel('$P_{DDG} - P_{comp}$ [Pa]')
ax5.set_title('2D Free-slip Pressure Deviation')
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
_savefig(fig5, 'hydrostatic_2d_periodic_pressure_deviation')

print("\nDone.")
