"""
2D Planar Poiseuille (Hagen-Poiseuille) Developing Flow — Eulerian

Simulates a 2D channel flow between parallel plates that develops from
rest toward the analytical parabolic Poiseuille profile under a constant
pressure gradient, on a **fixed mesh** (Eulerian frame).

Physics:
    - Channel: [0, L] x [0, D]  (flow in x, walls at y=0 and y=D)
    - No-slip walls at y=0 and y=D
    - Constant pressure gradient G = -dP/dx driving the flow
    - Initial condition: zero velocity, linear pressure P(x) = -G*x
    - Analytical steady-state: u_x(y) = (G / 2mu) * y * (D - y)

Uses the Cauchy stress tensor pipeline (ddgclib.operators.stress.dudt_i)
with the euler_velocity_only integrator (explicit Euler, fixed mesh).
"""

import os
import pickle
import numpy as np
from functools import partial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperct import Complex
from hyperct.ddg import compute_vd

# ddgclib modules
from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    NoSlipWallBC,
    identify_boundary_vertices,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    ZeroVelocity,
    LinearPressureGradient,
    UniformMass,
    PoiseuillePlanar,
)
from ddgclib.operators.stress import dudt_i
from ddgclib.dynamic_integrators import euler_velocity_only
from ddgclib.data import StateHistory, save_state
from ddgclib.visualization import plot_fluid

# ============================================================
# Parameters
# ============================================================
# Geometry
D = 1.0       # channel height [m]
L = 2.0       # channel length [m]

# Flow conditions
G = 1.0       # pressure gradient magnitude -dP/dx [Pa/m]
mu = 0.1      # dynamic viscosity [Pa.s]
rho = 1.0     # density [kg/m^3]

# Derived
U_max = G * D**2 / (8 * mu)         # centerline velocity at steady state
U_avg = U_max * 2 / 3               # mean velocity (2D planar: U_avg = 2/3 U_max)
Re_D = rho * U_avg * D / mu         # Reynolds number

# Output directories
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RESULTS, exist_ok=True)

print("=== Hagen-Poiseuille 2D (Eulerian, fixed mesh) ===")
print(f"Channel: {L} x {D} m")
print(f"G = {G:.4f} Pa/m,  mu = {mu:.4f} Pa.s,  rho = {rho}")
print(f"U_max (analytical) = {U_max:.6f} m/s")
print(f"U_avg (analytical) = {U_avg:.6f} m/s")
print(f"Re_D = {Re_D:.2f}")

# ============================================================
# Step 1: Domain Setup
# ============================================================
d = 2  # spatial dimension
n_refine = 3

HC = Complex(d, domain=[(0.0, L), (0.0, D)])
HC.triangulate()
for _ in range(n_refine):
    HC.refine_all()

n_verts = sum(1 for _ in HC.V)
print(f"\nMesh: {n_verts} vertices, {n_refine} refinements")

# Tag boundary vertices (required before compute_vd)
bV = identify_boundary_vertices(HC, lambda v: (
    abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L) < 1e-14 or
    abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - D) < 1e-14
))
for v in HC.V:
    v.boundary = v in bV

# Compute barycentric dual mesh (needed by stress tensor operators)
compute_vd(HC, method="barycentric")

# Wall vertices
bV_wall = identify_boundary_vertices(HC, lambda v: (
    abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - D) < 1e-14
))
print(f"Boundary vertices: {len(bV)},  Wall vertices: {len(bV_wall)}")

# ============================================================
# Step 2: Boundary Conditions
# ============================================================
# No-slip walls at y=0 and y=D (zero velocity on wall vertices)
bc_set = BoundaryConditionSet()
bc_set.add(NoSlipWallBC(dim=d), bV_wall)

# ============================================================
# Step 3: Initial Conditions
# ============================================================
# Start from rest: zero velocity, linear pressure drop, uniform mass.
# The pressure gradient G drives the flow; viscous forces + no-slip
# walls shape the velocity field toward the parabolic profile.
ic = CompositeIC(
    ZeroVelocity(dim=d),
    LinearPressureGradient(G=G, axis=0, P_ref=0.0),
    UniformMass(total_volume=L * D, rho=rho),
)
ic.apply(HC, bV)

# Enforce BCs on initial state
bc_set.apply_all(HC, bV, dt=0.0)

# Build the analytical profile object for comparison
poiseuille_ic = PoiseuillePlanar(
    G=G, mu=mu, y_lb=0.0, y_ub=D,
    flow_axis=0, normal_axis=1, dim=d,
)

print(f"ICs applied: zero velocity, P gradient G={G:.4f} Pa/m")

# ============================================================
# Step 4: Dynamic Integration (Eulerian — fixed mesh)
# ============================================================
dudt_fn = partial(dudt_i, dim=d, mu=mu, HC=HC)

dt = 0.005
n_steps = 2000
record_every = 10

history = StateHistory(fields=['u', 'p'], record_every=record_every)

print(f"\nRunning: dt={dt}, n_steps={n_steps}, t_final={dt * n_steps:.2f}")
print(f"Recording every {record_every} steps ({n_steps // record_every} snapshots)")

t_final = euler_velocity_only(
    HC, bV, dudt_fn,
    dt=dt, n_steps=n_steps, dim=d,
    bc_set=bc_set,
    callback=history.callback,
)

print(f"Simulation complete: t = {t_final:.4f}")

# ============================================================
# Step 5: Save Results and Analyze
# ============================================================

# 5a: Save final state
save_state(HC, bV, t=t_final, fields=['u', 'p', 'm'],
           path=os.path.join(_RESULTS, 'hp2d_eulerian_final.json'),
           extra_meta={'case': 'hagen_poiseuille_2d_eulerian',
                       'mu': mu, 'G': G, 'Re_D': Re_D})
print(f"Final state saved to {_RESULTS}/hp2d_eulerian_final.json")

# 5b: Save history
history_path = os.path.join(_RESULTS, 'hp2d_eulerian_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history, f)
print(f"History saved to {history_path} ({history.n_snapshots} snapshots)")

# 5c: Error metrics at channel midpoint (x = L/2)
x_mid = L / 2.0
tol = L / (2**n_refine) * 0.6
mid_verts = sorted(
    [v for v in HC.V if abs(v.x_a[0] - x_mid) < tol],
    key=lambda v: v.x_a[1]
)

errors = []
for v in mid_verts:
    u_anal = poiseuille_ic.analytical_velocity(v.x_a)
    errors.append(abs(v.u[0] - u_anal))

max_err = max(errors) if errors else float('nan')
l2_err = np.sqrt(np.mean(np.array(errors)**2)) if errors else float('nan')
ux_num = np.array([v.u[0] for v in mid_verts])
print(f"\nError at x=L/2 vs analytical: max={max_err:.6e}, L2={l2_err:.6e}")
print(f"U_max analytical = {U_max:.6f}, U_max numerical = "
      f"{max(ux_num) if len(ux_num) > 0 else float('nan'):.6f}")

# 5d: Quick profile comparison
print("\n--- Velocity profile at x = L/2 ---")
print(f"{'y':>8s}  {'u_x (num)':>12s}  {'u_x (anal)':>12s}  {'error':>12s}")
for v in mid_verts:
    u_anal = poiseuille_ic.analytical_velocity(v.x_a)
    err = abs(v.u[0] - u_anal)
    print(f"{v.x_a[1]:8.4f}  {v.u[0]:12.6f}  {u_anal:12.6f}  {err:12.6e}")

print(f"\nVertex count unchanged: {sum(1 for _ in HC.V)} (Eulerian, fixed mesh)")

# ============================================================
# Step 6: Visualization
# ============================================================

# 6a: Final state — pressure + velocity (ddgclib unified)
fig, axes = plot_fluid(
    HC, bV, t=t_final,
    scalar_field='p', vector_field='u',
    scalar_label='Pressure [Pa]', vector_label='Velocity [m/s]',
    save_path=os.path.join(_FIG, 'hp2d_eulerian_final.png'),
)
plt.close(fig)
print(f"Final state plot saved to {_FIG}/hp2d_eulerian_final.png")

# 6b: Velocity profile at x = L/2  vs analytical
y_anal = np.linspace(0, D, 200)
u_anal = (G / (2 * mu)) * y_anal * (D - y_anal)

y_num = np.array([v.x_a[1] for v in mid_verts])
u_num = np.array([v.u[0] for v in mid_verts])

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(u_anal, y_anal, 'k-', linewidth=2, label='Analytical (steady)')
ax.plot(u_num, y_num, 'ro', markersize=6, label=f'Numerical (t={t_final:.1f}s)')
ax.set_xlabel('$u_x$ [m/s]')
ax.set_ylabel('$y$ [m]')
ax.set_title(f'Velocity profile at x = L/2\n'
             f'G={G}, $\\mu$={mu}, $\\tau_{{visc}}$={rho*D**2/mu:.1f}s')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(0, D)
fig.tight_layout()
profile_path = os.path.join(_FIG, 'hp2d_eulerian_profile.png')
fig.savefig(profile_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Profile plot saved to {profile_path}")

# 6c: Time evolution of centerline velocity
# Snapshots are (time, {vertex_key: {field: value}}, diagnostics)
center_y = D / 2.0
t_hist = []
u_center_hist = []
for t_snap, snapshot, _diag in history._snapshots:
    t_hist.append(t_snap)
    best_u = 0.0
    best_dist = float('inf')
    for vkey, fields in snapshot.items():
        dist = abs(vkey[0] - x_mid) + abs(vkey[1] - center_y)
        if dist < best_dist:
            best_dist = dist
            best_u = fields['u'][0] if isinstance(fields['u'], np.ndarray) else fields['u']
    u_center_hist.append(best_u)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_hist, u_center_hist, 'b-', linewidth=1.5, label='$u_x$ at centerline')
ax.axhline(y=U_max, color='k', linestyle='--', linewidth=1, label=f'Analytical $U_{{max}}$={U_max:.3f}')
tau_visc = rho * D**2 / mu
ax.axvline(x=tau_visc, color='gray', linestyle=':', alpha=0.5, label=f'$\\tau_{{visc}}$={tau_visc:.1f}s')
ax.set_xlabel('Time [s]')
ax.set_ylabel('$u_x$ [m/s]')
ax.set_title('Centerline velocity development')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, t_final)
ax.set_ylim(bottom=0)
fig.tight_layout()
evolution_path = os.path.join(_FIG, 'hp2d_eulerian_evolution.png')
fig.savefig(evolution_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Evolution plot saved to {evolution_path}")

# 6d: Error convergence over time
err_hist = []
for t_snap, snapshot, _diag in history._snapshots:
    snap_err = []
    for vkey, fields in snapshot.items():
        if abs(vkey[0] - x_mid) < tol:
            u_val = fields['u'][0] if isinstance(fields['u'], np.ndarray) else fields['u']
            u_a = (G / (2 * mu)) * vkey[1] * (D - vkey[1])
            snap_err.append(abs(u_val - u_a))
    if snap_err:
        err_hist.append(np.sqrt(np.mean(np.array(snap_err)**2)))
    else:
        err_hist.append(float('nan'))

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(t_hist, err_hist, 'r-', linewidth=1.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('L2 error vs analytical')
ax.set_title('Error convergence at x = L/2')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0, t_final)
fig.tight_layout()
error_path = os.path.join(_FIG, 'hp2d_eulerian_error.png')
fig.savefig(error_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Error convergence plot saved to {error_path}")

# 6e: Summary figure (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Top-left: velocity magnitude on mesh
coords = np.array([v.x_a[:2] for v in HC.V])
u_mag = np.array([np.linalg.norm(v.u[:2]) for v in HC.V])
sc = axes[0, 0].tripcolor(
    coords[:, 0], coords[:, 1], u_mag,
    shading='gouraud', cmap='coolwarm',
)
fig.colorbar(sc, ax=axes[0, 0], label='|u| [m/s]')
axes[0, 0].set_aspect('equal')
axes[0, 0].set_title(f'Velocity magnitude (t={t_final:.1f}s)')
axes[0, 0].set_xlabel('x [m]')
axes[0, 0].set_ylabel('y [m]')

# Top-right: velocity profile
axes[0, 1].plot(u_anal, y_anal, 'k-', linewidth=2, label='Analytical')
axes[0, 1].plot(u_num, y_num, 'ro', markersize=5, label=f'Numerical')
axes[0, 1].set_xlabel('$u_x$ [m/s]')
axes[0, 1].set_ylabel('y [m]')
axes[0, 1].set_title('Velocity profile at x = L/2')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(left=0)

# Bottom-left: centerline evolution
axes[1, 0].plot(t_hist, u_center_hist, 'b-', linewidth=1.5)
axes[1, 0].axhline(y=U_max, color='k', linestyle='--', linewidth=1,
                    label=f'$U_{{max}}$={U_max:.3f}')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('$u_x$ [m/s]')
axes[1, 0].set_title('Centerline velocity')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, t_final)
axes[1, 0].set_ylim(bottom=0)

# Bottom-right: error convergence
axes[1, 1].semilogy(t_hist, err_hist, 'r-', linewidth=1.5)
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_ylabel('L2 error')
axes[1, 1].set_title('Error vs analytical')
axes[1, 1].grid(True, alpha=0.3, which='both')
axes[1, 1].set_xlim(0, t_final)

fig.suptitle(f'Hagen-Poiseuille 2D Eulerian  |  G={G}, $\\mu$={mu}, '
             f'Re={Re_D:.1f}, {n_verts} vertices', fontsize=13, y=1.01)
fig.tight_layout()
summary_path = os.path.join(_FIG, 'hp2d_eulerian_summary.png')
fig.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Summary plot saved to {summary_path}")
