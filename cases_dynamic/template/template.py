"""
Template for dynamic (Lagrangian) continuum simulations with ddgclib.

Copy this file and modify it for each new case.  The workflow is:

    1. Domain      - build the mesh with a domain builder, compute duals
    2. Boundary    - collect boundary conditions in a BoundaryConditionSet
    3. Initial     - set the vertex fields (volume-averaged where it matters)
    4. Methods     - name EVERY solver method choice in one SolverMethods
    5. Integrate   - build dudt_fn and run through methods.integrate()
    6. Record      - write methods.json next to the results (what actually ran)
    7. Postprocess - save state, plot, animate

The concrete example is a weakly compressible single-phase fluid in a
closed box: a smooth swirl decays viscously on a Lagrangian mesh.
Vertices advect, connectivity is rebuilt by Delaunay every step, and the
pressure comes from an equation of state, p = eos(m / dual volume).  It
runs in a few seconds.

The one method choice that is not optional here is
``remap='conservative'``.  A Delaunay flip changes a vertex's dual
volume by 33-100 % at fixed positions; without the remap the EOS reads
that as 3e4-5e4 Pa of compression and the run blows up at any time step
or sound speed (lane K,
docs_temp/debug_session/laneK-single-phase-eos-instability.md).  The
remap keeps the pressure field invariant across the rebuild.
``SolverMethods.dudt_fn`` warns if you bind an EOS without it.  The
alternative that needs no remap is ``connectivity='dual_only'`` (fixed
connectivity, fine for small deformation).  For two fluids start from
``cases_dynamic/oscillating_droplet/`` and
``ddgclib.methods.PRESETS['oscillating_droplet_2D']``.

Every method switch (time integrator, connectivity policy, mass
redistribution, ...) is a field of ``SolverMethods``; the allowed values,
their measured status and the evidence behind them are tabulated in
``METHODS.md`` (repo root).  Do not build ``functools.partial`` chains or
integrator kwargs by hand in a case: let the config build them, so the
recorded ``methods.json`` is the truth.
"""
import os
import sys

import numpy as np

# Make the repo root importable when this file is run by path
# (python cases_dynamic/template/template.py); every case runner does this.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Output directories next to this script (project convention)
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RESULTS, exist_ok=True)


# ---------------------------------------------------------------------
# Step 1: Domain
# ---------------------------------------------------------------------
# Domain builders return the mesh, the topological boundary set and named
# boundary groups (walls, inlet, outlet, ...).  Other options: build a
# hyperct.Complex by hand (Complex.triangulate + refine_all), import an
# external mesh, or discretise a level set.
from ddgclib.geometry.domains import rectangle
from hyperct.ddg import compute_vd, rebuild_simplex_cache_2d
from ddgclib.operators.stress import cache_dual_volumes

d = 2
L, h = 1.0, 1.0
result = rectangle(L=L, h=h, refinement=2, flow_axis=0)
HC, bV = result.HC, result.bV
# Closed box: every boundary face is a wall.  (For a channel use
# result.boundary_groups['walls'] / ['inlet'] / ['outlet'] instead.)
walls = bV

# Duals are needed BEFORE any volume-averaged initial condition and
# before the first force evaluation.  The integrator rebuilds them every
# step according to the connectivity policy chosen in step 4.
# The simplex cache makes the setup dual volumes exact (they tile the
# domain, corners included), i.e. the same measure every later
# retopology uses; without it the 2D fallback undercounts corner cells
# 4x and the EOS starts from a wrong density there (lane K).
rebuild_simplex_cache_2d(HC)
compute_vd(HC, method='barycentric')
cache_dual_volumes(HC, d)

print(f"Mesh: {sum(1 for _ in HC.V)} vertices, {len(bV)} boundary, "
      f"{len(walls)} wall")


# ---------------------------------------------------------------------
# Step 2: Boundary conditions
# ---------------------------------------------------------------------
# BCs are applied by the integrator after every step.  Wall vertices are
# also FROZEN because they sit in bV (the topological boundary set the
# integrator rebuilds each step); pass boundary_filter=... to integrate()
# when only part of the hull should be frozen (e.g. walls but not an
# inlet/outlet).
from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC

bc_set = BoundaryConditionSet()
bc_set.add(NoSlipWallBC(dim=d), walls)


# ---------------------------------------------------------------------
# Step 3: Initial conditions
# ---------------------------------------------------------------------
# Physics: water-like density and a large viscosity so the decay is
# visible within a few hundred steps (nu = mu/rho0 = 0.05 m^2/s).  The
# sound speed is numerical: Mach 0.01 keeps density within 0.1 % of rho0.
from ddgclib.eos import TaitMurnaghan
from ddgclib.initial_conditions import CompositeIC, DualVolumeMass, CustomFieldIC, UniformPressure

rho0 = 1000.0        # kg/m^3
mu = 50.0            # Pa s
u0 = 0.1             # m/s   perturbation amplitude
c_s = 100.0 * u0     # m/s   numerical sound speed (Mach 0.01)
eos = TaitMurnaghan(rho0=rho0, P0=0.0, K=rho0 * c_s**2, n=1.0)


def perturbation(x):
    """Divergence-free swirl from the stream function
    psi = sin^2(pi x) sin^2(pi y); u = (dpsi/dy, -dpsi/dx) vanishes on
    all four walls."""
    sx, cx = np.sin(np.pi * x[0]), np.cos(np.pi * x[0])
    sy, cy = np.sin(np.pi * x[1]), np.cos(np.pi * x[1])
    return u0 * np.array([sx * sx * sy * cy, -sx * cx * sy * sy])


ic = CompositeIC(
    CustomFieldIC(perturbation, field_name='u'),   # v.u = f(x)
    UniformPressure(P0=0.0),                        # = eos(rho0); the EOS takes over
    DualVolumeMass(rho=rho0),                       # v.m = rho0 * dual_vol
)
ic.apply(HC, bV)


# ---------------------------------------------------------------------
# Step 4: Solver methods (the single source of truth for the run)
# ---------------------------------------------------------------------
# Every field is a registered axis (ddgclib.methods.AXES / METHODS.md).
# Invalid or silently-ignored combinations raise ValueError here, not
# three functions deeper.  For a shipped case use a PRESET instead:
#     from ddgclib.methods import PRESETS
#     methods = PRESETS['oscillating_droplet_2D']
from ddgclib.methods import SolverMethods, record_methods, effective_methods

methods = SolverMethods(
    dim=d,
    phases='single',
    integrator='symplectic_euler',   # Lagrangian: u += dt a, x += dt u
    connectivity='delaunay',         # per-step global Delaunay + dual rebuild
    remap='conservative',            # pressure invariant across the rebuild
    redistribute_mass=True,          # (the remap re-targets the masses)
    label='template: weakly compressible viscous decay in a closed box',
)
print(methods.describe())


# ---------------------------------------------------------------------
# Step 5: Integrate
# ---------------------------------------------------------------------
# dudt_fn is the integrated Cauchy-stress acceleration (pressure flux +
# viscous flux over the dual cell) bound the canonical way.  The EOS
# goes in twice: into the force (pressure from density) and into
# integrate() (the remap needs its inverse).  With pressure_model=None
# the pressure field stays at its IC instead.  Add
# body_force=[0, -9.81] for gravity.
dudt_fn = methods.dudt_fn(HC, mu=mu, pressure_model=eos)

dx_min = min(float(np.linalg.norm(v.x_a[:d] - nb.x_a[:d]))
             for v in HC.V for nb in v.nn)
nu = mu / rho0
dt = min(0.25 * dx_min / c_s,        # acoustic CFL (stable up to 1.5)
         0.1 * dx_min**2 / nu)       # explicit viscous limit
n_steps = 400                        # ~1.8 s: KE decays by ~3 orders
record_every = 20

from ddgclib.data import StateHistory
history = StateHistory(fields=['u', 'p'], record_every=record_every,
                       save_dir=os.path.join(_RESULTS, 'snapshots'))

KE = []


def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
    history.callback(step, t, HC_cb, bV_cb, diagnostics)
    if step % record_every == 0:
        ke = sum(0.5 * v.m * float(np.dot(v.u[:d], v.u[:d])) for v in HC_cb.V)
        KE.append((t, ke))


print(f"\nRunning: dt={dt:.3e}, n_steps={n_steps}, t_end={dt * n_steps:.3f}")
t_final = methods.integrate(
    HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
    bc_set=bc_set, callback=callback, pressure_model=eos,
)
rho = np.array([v.m / v.dual_vol for v in HC.V])
print(f"Done: t = {t_final:.4f}, KE {KE[0][1]:.3e} -> {KE[-1][1]:.3e} J, "
      f"density within {np.max(np.abs(rho / rho0 - 1)):.1e} of rho0, "
      f"{history.n_snapshots} snapshots")


# ---------------------------------------------------------------------
# Step 6: Record what actually ran
# ---------------------------------------------------------------------
# methods.json = requested config + the implicit (dimension / cache
# gated) choices resolved on the final mesh + git SHAs of both repos.
# Quote it (or the preset name) whenever you discuss a result.
record_methods(os.path.join(_RESULTS, 'methods.json'), methods, HC,
               extra={'dt': dt, 'n_steps': n_steps, 'mu': mu, 'u0': u0,
                      'c_s': c_s, 'eos': 'TaitMurnaghan(n=1, P0=0)',
                      'refinement': 2})
print("Effective (implicit) choices:",
      {k: v for k, v in effective_methods(HC, d, methods).items()
       if k in ('dual_volume', 'edge_area_source', 'boundary_dual_vol')})


# ---------------------------------------------------------------------
# Step 7: Post-processing
# ---------------------------------------------------------------------
from ddgclib.data import save_state, load_state

save_state(HC, bV, t=t_final, fields=['u', 'p', 'm'],
           path=os.path.join(_RESULTS, 'final_state.json'),
           extra_meta={'case': 'template_box_decay', 'mu': mu})
HC_loaded, bV_loaded, meta = load_state(os.path.join(_RESULTS, 'final_state.json'))
print(f"State round-trip: t={meta['time']}, case={meta.get('case')}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ddgclib.visualization import plot_fluid

plot_fluid(HC, bV=bV, t=t_final,
           save_path=os.path.join(_FIG, 'template_fluid.png'))

fig, ax = plt.subplots(figsize=(6, 4))
t_arr = np.array([k[0] for k in KE])
ke_arr = np.array([k[1] for k in KE])
ax.semilogy(t_arr, ke_arr, 'o-', label='simulation')
ax.semilogy(t_arr, ke_arr[0] * np.exp(-2 * nu * 2 * np.pi**2 * t_arr), '--',
            label=r'$e^{-2\nu k^2 t}$ (unbounded Stokes reference)')
ax.set_xlabel('t [s]')
ax.set_ylabel('kinetic energy [J]')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(_FIG, 'template_ke.png'), dpi=150)
plt.close('all')
print(f"Figures saved to {_FIG}/")

# Animation from the recorded history (mp4 needs ffmpeg):
#   from ddgclib.visualization import dynamic_plot_fluid
#   dynamic_plot_fluid(history, HC, bV=bV,
#                      save_path=os.path.join(_FIG, 'template.mp4'))
# Interactive 3D replay of the snapshots:
#   python -m ddgclib.scripts.view_polyscope --snapshots results/snapshots
