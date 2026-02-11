"""
Template for dynamic continuum simulations using ddgclib.

Copy this file and modify it for each new case. The 5-step workflow is:

    1. Domain      — Build mesh (hyperct.Complex) and identify boundary vertices
    2. Boundary    — Define boundary conditions (no-slip, Dirichlet, Neumann, ...)
    3. Initial     — Set initial fields (velocity, pressure, mass)
    4. Integrate   — Choose a time integrator and run the simulation
    5. Postprocess — Save results, visualize, analyze

This template demonstrates a 2D channel flow (Poiseuille) as a concrete example.
Replace/modify each section for your own problem.

Step 1 in more detail:

1.a The geometry can be created using the starting cube (Complex.triangulate),
    refining (Complex.refine_all or similar) and then modifying it (e.g. by
    removing vertices, edges, faces) or by using the built-in geometry generators
    (e.g. for a cylinder using the functions in ddgclib.geometry).
1.b Alternatively, the geometry can be imported from an external mesh file (e.g.
    a tet mesh) and converted to a Complex using the hyperct functions.
1.c Finally a domain can be defined by abstract mathematical functions (e.g. level
    sets) and then discretized into a Complex using the hyperct functions.
"""

import os
import numpy as np
from hyperct import Complex

# Output directory: fig/ next to this script
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
os.makedirs(_FIG, exist_ok=True)

# ============================================================================
# Step 1: Define the domain and boundary set
# ============================================================================
# Create a simplicial complex on [0, L] x [0, h]

d = 2                  # dimension (1, 2, or 3)
L, h = 2.0, 1.0       # channel length and height

HC = Complex(d, domain=[(0.0, L), (0.0, h)])
HC.triangulate()       # initial triangulation of the cube/rectangle
HC.refine_all()        # refine once (increase for finer mesh)
HC.refine_all()        # refine again

# Identify boundary vertices — any vertex on the domain faces
from ddgclib._boundary_conditions import (
    identify_boundary_vertices,
    identify_cube_boundaries,
    BoundaryConditionSet,
    NoSlipWallBC,
    DirichletPressureBC,
)

# For axis-aligned cube/rectangle domains, use the convenience helper:
#   bV = identify_cube_boundaries(HC, lb=0.0, ub=1.0, dim=d)
#
# For non-cube domains or when faces have different bounds, use the general helper:
bV = identify_boundary_vertices(HC, lambda v: (
    abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L) < 1e-14 or
    abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14
))

# Identify specific boundary regions for different BCs
bV_walls = identify_boundary_vertices(
    HC, lambda v: abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14
)
bV_inlet = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0]) < 1e-14)
bV_outlet = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0] - L) < 1e-14)

print(f"Mesh: {sum(1 for _ in HC.V)} vertices, {len(bV)} boundary, "
      f"{len(bV_walls)} wall, {len(bV_inlet)} inlet, {len(bV_outlet)} outlet")


# ============================================================================
# Step 2: Define the boundary conditions
# ============================================================================
# BCs are collected in a BoundaryConditionSet and applied each time step
# automatically by the integrator (when bc_set is passed).

bc_set = BoundaryConditionSet()

# No-slip walls at y=0 and y=h (velocity = 0)
bc_set.add(NoSlipWallBC(dim=d), bV_walls)

# Fixed pressure at outlet (atmospheric, P = 0)
bc_set.add(DirichletPressureBC(value=0.0), bV_outlet)

# You can also use callable BCs for spatially varying conditions:
#   from ddgclib._boundary_conditions import DirichletVelocityBC
#   bc_set.add(DirichletVelocityBC(lambda v: np.array([v.x_a[1], 0.0]), dim=2), bV_inlet)

print(f"Boundary conditions: {len(bc_set._bcs)} BCs registered")


# ============================================================================
# Step 3: Define the initial conditions
# ============================================================================
# ICs set vertex fields (v.u, v.P, v.m) on the entire mesh.
# Use CompositeIC to combine multiple ICs — they are applied in order.

from ddgclib.initial_conditions import (
    CompositeIC,
    ZeroVelocity,
    LinearPressureGradient,
    UniformMass,
    PoiseuillePlanar,
)

# Physical parameters
G = 1.0     # pressure gradient magnitude
mu = 0.1    # dynamic viscosity
rho = 1.0   # density

# Option A: Start from rest (zero velocity + linear pressure + uniform mass)
ic_from_rest = CompositeIC(
    ZeroVelocity(dim=d),
    LinearPressureGradient(G=G, axis=0),
    UniformMass(total_volume=L * h, rho=rho),
)

# Option B: Start from analytical Poiseuille profile (for equilibrium testing)
ic_equilibrium = CompositeIC(
    PoiseuillePlanar(G=G, mu=mu, y_lb=0.0, y_ub=h, flow_axis=0, normal_axis=1, dim=d),
    LinearPressureGradient(G=G, axis=0),
    UniformMass(total_volume=L * h, rho=rho),
)

# Choose which IC to apply:
ic = ic_from_rest
ic.apply(HC, bV)

print(f"Initial conditions applied. Sample vertex: "
      f"u={list(next(iter(HC.V)).u)}, P={next(iter(HC.V)).P:.4f}")


# ============================================================================
# Step 4: Specify the integrator and run the simulation
# ============================================================================
# Choose an acceleration function (du/dt) and a time integrator.
#
# The acceleration function computes forces on each vertex:
#   acceleration(v) = (-grad_P + mu * laplacian_u) / m
#
# For problems requiring the full DDG operators with barycentric duals:
#   from ddgclib.barycentric._duals import compute_vd
#   compute_vd(HC, cdist=1e-10)
#   from ddgclib.operators.gradient import acceleration as dudt_fn
#
# For this template, we use a simple mock acceleration for demonstration:

def dudt_fn(v, dim=2, mu=0.1, **kwargs):
    """Mock acceleration: diffusion toward neighbor average (Laplacian-like)."""
    if not v.nn:
        return np.zeros(dim)
    avg_u = np.mean([nb.u[:dim] for nb in v.nn], axis=0)
    return mu * (avg_u - v.u[:dim])


# --- Option A: Direct integrator call ---
from ddgclib.dynamic_integrators import euler_velocity_only

# Record history for post-processing
from ddgclib.data import StateHistory
history = StateHistory(fields=['u', 'P'], record_every=50)

dt = 0.001
n_steps = 500

t_final = euler_velocity_only(
    HC, bV, dudt_fn,
    dt=dt, n_steps=n_steps, dim=d,
    bc_set=bc_set,                  # BCs enforced after each step
    callback=history.callback,      # record snapshots
    mu=mu,                          # forwarded to dudt_fn as **dudt_kwargs
)

print(f"Simulation complete: t = {t_final:.4f}, {history.n_snapshots} snapshots recorded")


# --- Option B: DynamicSimulation runner (convenience wrapper) ---
# Bundles everything into a single object. Uncomment to use instead of Option A:
#
# from ddgclib.dynamic_integrators import DynamicSimulation, SimulationParams
#
# sim = (DynamicSimulation(HC, bV, SimulationParams(dt=0.001, n_steps=500, dim=2, mu=0.1))
#        .set_initial_conditions(ic)
#        .set_boundary_conditions(bc_set)
#        .set_integrator(euler_velocity_only)
#        .set_acceleration_fn(dudt_fn))
# t_final = sim.run(callback=history.callback)


# ============================================================================
# Step 5: Post-processing — save, visualize, analyze
# ============================================================================

# --- 5a: Save state to disk (JSON format) ---
from ddgclib.data import save_state

save_state(HC, bV, t=t_final, fields=['u', 'P', 'm'],
           path=os.path.join(_FIG, 'final_state.json'),
           extra_meta={'case': 'template_channel', 'mu': mu, 'G': G})
print(f"State saved to {_FIG}/final_state.json")


# --- 5b: Load a saved state (round-trip) ---
from ddgclib.data import load_state
HC_loaded, bV_loaded, meta = load_state(os.path.join(_FIG, 'final_state.json'))
print(f"Loaded state at t={meta['time']}, case={meta.get('case')}")


# --- 5c: Query history ---
some_vertex = next(iter(HC.V))
vertex_key = tuple(float(x) for x in some_vertex.x_a)
times, pressures = history.query_vertex(vertex_key, 'P')
if times:
    print(f"Vertex at {vertex_key}: P went from {pressures[0]:.4f} to {pressures[-1]:.4f}")


# --- 5d: Visualization ---
import matplotlib
matplotlib.use('Agg')  # use 'TkAgg' for interactive display

from ddgclib.visualization import (
    plot_scalar_field_2d,
    plot_vector_field_2d,
    plot_mesh_2d,
)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_mesh_2d(HC, bV=bV, ax=axes[0], title='Mesh')
plot_scalar_field_2d(HC, field='P', ax=axes[1], title='Pressure')
plot_vector_field_2d(HC, bV=bV, ax=axes[2], title='Velocity')

plt.tight_layout()
plt.savefig(os.path.join(_FIG, 'template_results.png'), dpi=150)
print(f"Plot saved to {_FIG}/template_results.png")
# plt.show()  # uncomment for interactive display

# Animation from history:
from ddgclib.visualization.animation import animate_scalar_2d
anim = animate_scalar_2d(history, field='P')
anim.save(os.path.join(_FIG, 'pressure_evolution.gif'), writer='pillow', fps=10)
print(f"Animation saved to {_FIG}/pressure_evolution.gif")
