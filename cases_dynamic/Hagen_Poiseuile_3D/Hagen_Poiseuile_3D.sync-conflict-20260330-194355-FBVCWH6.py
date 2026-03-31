"""
3D Hagen-Poiseuille Developing Pipe Flow (Lagrangian)
=====================================================

Simulates a 3D cylindrical pipe flow that develops from an initial uniform
plug flow toward the analytical parabolic Hagen-Poiseuille profile under
a constant pressure gradient.

Physics:
    - Cylindrical tube of radius R, length L
    - Flow along ``flow_axis`` (default z=2)
    - No-slip wall on the cylindrical surface (r = R)
    - Open outlet: vertices past L + buffer are deleted
    - Periodic inlet: ghost mesh injects new vertices at inlet
    - Constant pressure gradient G = -dP/dz driving the flow
    - Analytical steady-state: u_z(r) = U_max * (1 - (r/R)^2)
      where U_max = G*R^2 / (4*mu)

Uses the Cauchy stress tensor pipeline (ddgclib.operators.stress.dudt_i)
with the symplectic_euler integrator (semi-implicit, Lagrangian mesh).

Usage::

    python Hagen_Poiseuile_3D.py

    # Then visualize interactively:
    python Hagen_Poiseuile_3D.py --visualize

    # Low-refinement quick test:
    python Hagen_Poiseuile_3D.py --n-refine 1 --n-steps 100
"""

import argparse
import math
import os
import pickle
import sys

import warnings
import numpy as np
from functools import partial

# Suppress RuntimeWarnings from degenerate simplices in compute_vd
# (expected near cylindrical surface after Delaunay retriangulation)
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='hyperct.ddg._geometry')

# Ensure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    PositionalNoSlipWallBC,
    OutletBufferedDeleteBC,
    PeriodicInletBC,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    UniformVelocity,
    LinearPressureGradient,
    UniformMass,
    HagenPoiseuille3D,
)
from ddgclib.geometry.domains import cylinder_volume
from ddgclib.operators.stress import dudt_i
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.data import StateHistory, save_state


# ============================================================
# Configuration  (edit these to change the problem)
# ============================================================

# --- Flow axis (change to 0 or 1 to reorient the pipe) ---
flow_axis = 2

# --- Geometry ---
R = 0.5          # tube radius [m]
D = 2 * R        # tube diameter [m]
L = 15.0         # tube length [m]

# --- Flow conditions ---
# Lower Re gives faster development: L_e ~ 0.06 * Re * D
# At Re=10, L_e ~ 0.6 m  (well within L=15 m)
Re_D = 10        # Reynolds number based on diameter and average velocity
U_avg = 0.1      # average inlet velocity (uniform plug) [m/s]
U_max = 2 * U_avg  # fully developed centerline velocity [m/s]

# --- Fluid properties ---
rho = 1.0        # fluid density [kg/m^3]
mu = (rho * U_avg * D) / Re_D  # dynamic viscosity [Pa.s]

# --- Derived quantities ---
L_e_approx = 0.06 * Re_D * D   # theoretical entrance length [m]
G = 8 * mu * U_avg / R**2      # pressure gradient for fully developed flow [Pa/m]

# --- Mesh parameters ---
n_refine = 2     # refinement levels (keep low for initial experiments)
cdist = 1e-10    # vertex merging tolerance [m]

# --- Outlet buffer ---
outlet_buffer = 2.0  # buffer width beyond L for OutletBufferedDeleteBC

# --- Time stepping ---
dt = 0.01
n_steps = 360
record_every = 1
save_every = 500
n_workers = 8    # threads for parallel stress computation (GIL-free numpy)

# Backend for batch geometry operations (e_star, compute_vd).
# Set to None for default numpy, or use get_backend for GPU/multiprocessing.
_retop_backend = None  # initialized in main() based on available hardware

# Output directories
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')


def print_params():
    print("=" * 72)
    print("3D Hagen-Poiseuille Developing Pipe Flow (Lagrangian)")
    print("=" * 72)
    print(f"  Flow axis       = {flow_axis} ({'xyz'[flow_axis]})")
    print(f"  Radius R        = {R} m")
    print(f"  Diameter D      = {D} m")
    print(f"  Tube length L   = {L} m  (L/D = {L/D:.1f})")
    print(f"  Re_D            = {Re_D}")
    print(f"  Entrance length ~ {L_e_approx:.2f} m  (L_e/D ~ {L_e_approx/D:.1f})")
    print(f"  Inlet U_avg     = {U_avg} m/s")
    print(f"  Developed U_max = {U_max} m/s")
    print(f"  Density rho     = {rho} kg/m^3")
    print(f"  Viscosity mu    = {mu:.6f} Pa.s")
    print(f"  Pressure grad G = {G:.6f} Pa/m")
    print(f"  n_refine        = {n_refine}")
    print(f"  dt              = {dt},  n_steps = {n_steps}")
    print(f"  workers         = {n_workers}")
    print()


def _radial_distance_fast(v, _flow_axis=flow_axis):
    """Fast radial distance (avoids sqrt for comparison)."""
    radial_axes = [i for i in range(3) if i != _flow_axis]
    return math.sqrt(sum(v.x_a[ax]**2 for ax in radial_axes))


def wall_criterion(v, _flow_axis=flow_axis, _R=R, _tol=1e-8):
    """True if vertex lies on the cylindrical wall (r >= R - tol)."""
    radial_axes = [i for i in range(3) if i != _flow_axis]
    r = math.sqrt(sum(v.x_a[ax]**2 for ax in radial_axes))
    return r >= _R - _tol


def boundary_criterion(v, _flow_axis=flow_axis, _R=R, _L=L, _tol=1e-8):
    """True if vertex lies on ANY domain boundary (wall, inlet, or outlet)."""
    # Wall: cylindrical surface
    radial_axes = [i for i in range(3) if i != _flow_axis]
    r = math.sqrt(sum(v.x_a[ax]**2 for ax in radial_axes))
    if r >= _R - _tol:
        return True
    # Inlet: z ~ 0 (or below, for buffer vertices entering from behind)
    if v.x_a[_flow_axis] <= _tol:
        return True
    # Outlet: z ~ L (or beyond, for buffer vertices)
    if v.x_a[_flow_axis] >= _L - _tol:
        return True
    return False


def retopologize_cylinder(HC, bV, dim):
    """Retopologize with degenerate tetrahedra filtering.

    The standard ``_retopologize`` uses Delaunay + ``HC.boundary()``.
    The 3D Delaunay creates zero-volume (coplanar) tetrahedra near
    curved surfaces, which break ``compute_vd`` and ``e_star``.
    Filtering these by quality ratio (vol / max_edge^3) before building
    connectivity restores correct ``HC.boundary()`` results.

    This function is domain-agnostic — it works for any 3D geometry
    where Delaunay produces degenerate simplices (cylinders, spheres,
    curved surfaces).

    Steps:
        1. Merge near-duplicate vertices (cdist tolerance)
        2. Disconnect all edges, Delaunay retriangulation
        3. Filter degenerate tetrahedra (quality ratio threshold)
        4. Build connectivity from good tetrahedra only
        5. Boundary detection via HC.boundary() (topological)
        6. Recompute barycentric dual mesh
        7. Cache dual volumes via batch_e_star
    """
    from hyperct.ddg import compute_vd
    from scipy.spatial import Delaunay as _Delaunay

    verts = list(HC.V)
    if len(verts) < dim + 1:
        return

    # 1. Merge near-duplicate vertices (prevents accumulation from inlet)
    HC.V.merge_all(cdist=1e-9)
    bV.intersection_update(set(HC.V))
    verts = list(HC.V)
    if len(verts) < dim + 1:
        return

    # 2. Disconnect ALL existing edges
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)

    # 3. Delaunay retriangulation
    coords = np.array([v.x_a[:dim] for v in verts])
    tri = _Delaunay(coords)

    # 4. Filter degenerate tetrahedra (zero-volume = coplanar vertices).
    #    These occur near the cylindrical surface where wall vertices
    #    lie in a plane.  Removing them prevents broken dual connectivity.
    #    Use a relative threshold: volume must be > fraction of cube of
    #    the longest edge.  This catches near-degenerate slivers that have
    #    tiny but nonzero absolute volume.
    quality_threshold = 1e-4  # min vol / max_edge^3 ratio
    for simplex in tri.simplices:
        pts = coords[simplex]
        edges = pts[1:] - pts[0]
        vol = abs(np.linalg.det(edges)) / 6.0
        # Compute max edge length for quality ratio
        max_edge3 = 0.0
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                e_len = np.linalg.norm(pts[i] - pts[j])
                e3 = e_len ** 3
                if e3 > max_edge3:
                    max_edge3 = e3
        if max_edge3 > 0 and vol / max_edge3 > quality_threshold:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    verts[simplex[i]].connect(verts[simplex[j]])

    # 5. Boundary detection via HC.boundary() (topological).
    #    This works correctly AFTER degenerate tet filtering — the
    #    filtered Delaunay gives the same boundary as the original mesh.
    dV = HC.boundary()
    bV.clear()
    bV.update(dV)
    for v in HC.V:
        v.boundary = v in bV

    # 6. Recompute duals
    compute_vd(HC, method="barycentric", backend=_retop_backend)

    # 7. Cache dual volumes.
    #    Try batch_e_star (vectorized, ~6x faster) if available in
    #    the installed hyperct; fall back to per-vertex dual_volume.
    from ddgclib.operators.stress import dual_volume
    try:
        from hyperct.ddg import batch_e_star
        interior = [v for v in HC.V if v not in bV]
        _, failed, vols = batch_e_star(
            interior, HC, dim=3, backend=_retop_backend,
            compute_volumes=True,
        )
        for v in failed:
            v.boundary = True
            bV.add(v)
        for v in HC.V:
            if v in bV:
                v.dual_vol = 0.0
            else:
                v.dual_vol = vols.get(id(v), 0.0)
    except ImportError:
        # Older hyperct without batch_e_star — sequential fallback
        for v in HC.V:
            if v in bV:
                v.dual_vol = 0.0
            else:
                v.dual_vol = dual_volume(v, HC, dim)


# ============================================================
# Step 1: Domain Setup  (cylindrical tube)
# ============================================================

def build_domain(n_ref):
    """Build the 3D cylindrical pipe domain.

    Returns (HC, bV, domain_result).
    """
    print(f"Building cylinder: R={R}, L={L}, refinement={n_ref}, "
          f"flow_axis={flow_axis}")
    result = cylinder_volume(R=R, L=L, refinement=n_ref, flow_axis=flow_axis)
    HC = result.HC
    bV = result.bV

    n_verts = sum(1 for _ in HC.V)
    n_wall = len(result.boundary_groups['walls'])
    n_inlet = len(result.boundary_groups['inlet'])
    n_outlet = len(result.boundary_groups['outlet'])
    print(f"  Mesh: {n_verts} vertices")
    print(f"  Boundary: {len(bV)} total  "
          f"({n_wall} wall, {n_inlet} inlet, {n_outlet} outlet)")

    return HC, bV, result


# ============================================================
# Step 2: Build the inlet ghost mesh (unit cylinder slice)
# ============================================================

def build_inlet_mesh(n_ref):
    """Build a unit-length cylinder slice for the periodic inlet.

    The ghost mesh is a short cylinder segment with the same cross-section
    as the main domain.  Its axial length (period) is 1.0 so that the
    vertex density along the flow axis matches the main mesh.
    """
    result = cylinder_volume(R=R, L=1.0, refinement=n_ref, flow_axis=flow_axis)
    unit_mesh = result.HC
    unit_bV = result.bV

    # Apply ICs on the unit mesh (plug flow + pressure + mass)
    unit_ic = CompositeIC(
        UniformVelocity(u_vec=_make_velocity_vec(U_avg)),
        LinearPressureGradient(G=G, axis=flow_axis, P_ref=0.0),
        UniformMass(total_volume=math.pi * R**2 * 1.0, rho=rho),
    )
    unit_ic.apply(unit_mesh, unit_bV)

    return unit_mesh


def _make_velocity_vec(u_axial):
    """Create a 3D velocity vector with u_axial along the flow axis."""
    u = np.zeros(3)
    u[flow_axis] = u_axial
    return u


# ============================================================
# Step 3: Boundary Conditions
# ============================================================

def build_bc_set(HC, bV, unit_mesh):
    """Assemble the boundary condition set for Lagrangian pipe flow."""
    bc_set = BoundaryConditionSet()

    # 1. Positional no-slip wall (cylindrical surface r = R)
    bc_set.add(
        PositionalNoSlipWallBC(criterion_fn=wall_criterion, dim=3, bV=bV),
        None,
    )

    # 2. Outlet: buffered delete beyond L
    bc_set.add(
        OutletBufferedDeleteBC(
            outlet_pos=L, buffer_width=outlet_buffer, axis=flow_axis, bV=bV,
        ),
        None,
    )

    # 3. Periodic inlet: ghost mesh injects new vertices at z=0
    bc_set.add(
        PeriodicInletBC(
            unit_mesh=unit_mesh,
            velocity=U_avg,
            axis=flow_axis,
            inlet_pos=0.0,
            cdist=cdist,
            fields=['u', 'p', 'm'],
            period=1.0,
        ),
        None,
    )

    return bc_set


# ============================================================
# Step 4: Initial Conditions
# ============================================================

def apply_initial_conditions(HC, bV):
    """Apply plug flow ICs to the main domain."""
    ic = CompositeIC(
        UniformVelocity(u_vec=_make_velocity_vec(U_avg)),
        LinearPressureGradient(G=G, axis=flow_axis, P_ref=0.0),
        UniformMass(total_volume=math.pi * R**2 * L, rho=rho),
    )
    ic.apply(HC, bV)
    return ic


# ============================================================
# Step 5: Run Simulation
# ============================================================

def run_simulation(HC, bV, bc_set, n_steps_override=None, dt_override=None):
    """Run the Lagrangian symplectic Euler integration."""
    import time as _time

    _dt = dt_override if dt_override is not None else dt
    _n_steps = n_steps_override if n_steps_override is not None else n_steps

    # Bind physics parameters via partial
    dudt_fn = partial(dudt_i, dim=3, mu=mu, HC=HC)

    history = StateHistory(fields=['u', 'p'], record_every=record_every)

    print(f"\nRunning: dt={_dt}, n_steps={_n_steps}, t_final={_dt*_n_steps:.2f}")
    print(f"Recording every {record_every} steps "
          f"({_n_steps // record_every} snapshots)")

    # Progress callback: prints step timing every 10 steps
    _wall_t0 = _time.perf_counter()
    _last_report = [0, _wall_t0]  # [step, wall_time]

    def _progress_callback(step, t, HC, bV=None, diagnostics=None):
        history.callback(step, t, HC, bV, diagnostics)
        now = _time.perf_counter()
        if step % 10 == 0 and step > 0:
            elapsed = now - _last_report[1]
            steps_done = step - _last_report[0]
            rate = steps_done / elapsed if elapsed > 0 else 0
            total_elapsed = now - _wall_t0
            n_verts = sum(1 for _ in HC.V)
            eta = (_n_steps - step) / rate if rate > 0 else float('inf')
            print(f"  step {step:>6d}/{_n_steps}  t={t:.4f}  "
                  f"{rate:.2f} step/s  verts={n_verts}  "
                  f"elapsed={total_elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)
            _last_report[0] = step
            _last_report[1] = now

    # Use custom retopologize that detects boundaries geometrically
    # (cylindrical wall + inlet/outlet end caps) instead of the default
    # HC.boundary() which fails to identify all surface vertices after
    # 3D Delaunay retriangulation.
    t_final = symplectic_euler(
        HC, bV, dudt_fn,
        dt=_dt, n_steps=_n_steps, dim=3,
        bc_set=bc_set,
        retopologize_fn=retopologize_cylinder,
        callback=_progress_callback,
        save_every=save_every,
        save_dir=_RESULTS,
        workers=n_workers,
    )

    print(f"Simulation complete: t = {t_final:.4f}")
    return t_final, history


# ============================================================
# Step 6: Post-processing
# ============================================================

def save_results(HC, bV, t_final, history):
    """Save final state and history to disk."""
    os.makedirs(_RESULTS, exist_ok=True)

    # Final state
    save_state(HC, bV, t=t_final, fields=['u', 'p', 'm'],
               path=os.path.join(_RESULTS, 'hp3d_final_state.json'),
               extra_meta={'case': 'hagen_poiseuille_3d', 'mu': mu, 'G': G,
                           'Re_D': Re_D, 'R': R, 'L': L,
                           'flow_axis': flow_axis})
    print(f"Final state saved to {_RESULTS}/hp3d_final_state.json")

    # History
    history_path = os.path.join(_RESULTS, 'hp3d_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"History saved to {history_path} ({history.n_snapshots} snapshots)")


def analyse_profile(HC, bV):
    """Compare the velocity profile at the tube midpoint to the analytical solution."""
    # Analytical profile
    hp_ic = HagenPoiseuille3D(U_max=U_max, R=R, flow_axis=flow_axis, dim=3)

    # Collect interior vertices near the midpoint
    z_mid = L / 2.0
    # Tolerance: use a fraction of the tube length scaled by refinement
    tol = L / (2**n_refine) * 0.6

    mid_verts = sorted(
        [v for v in HC.V
         if v not in bV and abs(v.x_a[flow_axis] - z_mid) < tol],
        key=lambda v: _radial_distance(v),
    )

    if not mid_verts:
        print("\nNo interior vertices near midpoint for profile comparison.")
        return

    errors = []
    for v in mid_verts:
        u_anal = hp_ic.analytical_velocity(v.x_a)
        u_num = v.u[flow_axis]
        errors.append(abs(u_num - u_anal))

    max_err = max(errors)
    l2_err = np.sqrt(np.mean(np.array(errors)**2))
    ux_num = np.array([v.u[flow_axis] for v in mid_verts])

    print(f"\nProfile comparison at z=L/2 ({len(mid_verts)} vertices):")
    print(f"  Error vs analytical: max={max_err:.6e}, L2={l2_err:.6e}")
    print(f"  U_max analytical = {U_max:.6f}")
    print(f"  U_max numerical  = {max(ux_num):.6f}")

    # Radial profile plot (matplotlib, non-interactive)
    _plot_radial_profile(mid_verts, hp_ic)


def _radial_distance(v):
    """Compute radial distance from pipe centerline."""
    radial_axes = [i for i in range(3) if i != flow_axis]
    return np.linalg.norm([v.x_a[ax] for ax in radial_axes])


def _plot_radial_profile(mid_verts, hp_ic):
    """Plot radial velocity profile vs analytical at pipe midpoint."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(_FIG, exist_ok=True)

    r_vals = np.array([_radial_distance(v) for v in mid_verts])
    u_num = np.array([v.u[flow_axis] for v in mid_verts])

    # Analytical curve
    r_anal = np.linspace(0, R, 100)
    u_anal = U_max * (1 - (r_anal / R)**2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_anal, u_anal, 'k-', lw=2, label='Analytical (Hagen-Poiseuille)')
    ax.scatter(r_vals, u_num, s=20, c='tab:blue', alpha=0.7, label='Numerical')
    ax.set_xlabel('Radial distance r [m]')
    ax.set_ylabel(f'u_{"xyz"[flow_axis]} [m/s]')
    ax.set_title(f'Velocity profile at z = L/2  (n_refine={n_refine})')
    ax.legend()
    ax.set_xlim(0, R * 1.05)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_FIG, 'hp3d_velocity_profile.png'), dpi=150)
    plt.close(fig)
    print(f"  Profile plot saved to {_FIG}/hp3d_velocity_profile.png")


# ============================================================
# Step 7: Polyscope Visualization (interactive timeline)
# ============================================================

def visualize_polyscope(history, HC):
    """Launch interactive polyscope viewer with timeline slider."""
    from ddgclib.visualization.polyscope_3d import interactive_history_viewer

    print("\nLaunching polyscope viewer with timeline slider...")
    interactive_history_viewer(
        history, HC,
        scalar_fields=['p'],
        vector_fields=['u'],
        name='hp3d',
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Hagen-Poiseuille developing pipe flow (Lagrangian)"
    )
    parser.add_argument('--n-refine', type=int, default=None,
                        help='Mesh refinement level (overrides config)')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Number of time steps (overrides config)')
    parser.add_argument('--dt', type=float, default=None,
                        help='Time step size (overrides config)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Thread count for parallel stress computation')
    parser.add_argument('--visualize', action='store_true',
                        help='Launch polyscope viewer after simulation')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Skip simulation, load history and visualize')
    args = parser.parse_args()

    global n_refine, n_workers
    if args.n_refine is not None:
        n_refine = args.n_refine
    if args.workers is not None:
        n_workers = args.workers

    if args.visualize_only:
        # Load saved history and visualize
        history_path = os.path.join(_RESULTS, 'hp3d_history.pkl')
        if not os.path.exists(history_path):
            print(f"No history file found at {history_path}. "
                  "Run simulation first.")
            sys.exit(1)
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        print(f"Loaded {history.n_snapshots} snapshots from {history_path}")
        # Build a dummy HC for dim info
        result = cylinder_volume(R=R, L=1.0, refinement=1, flow_axis=flow_axis)
        visualize_polyscope(history, result.HC)
        return

    # Initialize computation backend for batch geometry ops
    global _retop_backend
    try:
        from hyperct._backend import get_backend
        _retop_backend = get_backend("torch")
        print(f"Backend: torch (device={_retop_backend.device})")
    except Exception:
        try:
            from hyperct._backend import get_backend
            _retop_backend = get_backend("numpy")
            print("Backend: numpy")
        except Exception:
            _retop_backend = None
            print("Backend: numpy (fallback)")

    print_params()

    # Step 1: Build domain
    HC, bV, domain_result = build_domain(n_refine)

    # Step 2: Build inlet ghost mesh
    unit_mesh = build_inlet_mesh(n_refine)

    # Step 3: Boundary conditions
    bc_set = build_bc_set(HC, bV, unit_mesh)

    # Step 4: Initial conditions
    apply_initial_conditions(HC, bV)

    # Apply BCs on initial state
    bc_set.apply_all(HC, bV, dt=0.0)

    # Compute initial dual mesh
    compute_vd(HC, method="barycentric")

    # Step 5: Run simulation
    t_final, history = run_simulation(
        HC, bV, bc_set,
        n_steps_override=args.n_steps,
        dt_override=args.dt,
    )

    # Step 6: Save and analyse
    save_results(HC, bV, t_final, history)
    analyse_profile(HC, bV)

    # Step 7: Visualize (if requested)
    if args.visualize:
        visualize_polyscope(history, HC)

    print("\nDone.")
    print("Run with --visualize to launch the interactive polyscope viewer.")
    print("Run with --visualize-only to view saved results without re-running.")


if __name__ == "__main__":
    main()
