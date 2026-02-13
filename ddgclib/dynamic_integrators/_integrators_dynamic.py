"""
Time integration routines for dynamic (velocity-based) simulations.

These integrators advance both velocity u and position x of vertices in a
simplicial Complex using the momentum equation:

    du/dt = dudt_fn(v)    (acceleration from pressure gradient + viscous terms)
    dx/dt = u             (kinematic relation, Lagrangian frame)

Each vertex v must have:
    v.x_a  - position as numpy array
    v.u    - velocity as numpy array
    v.m    - mass (scalar)

Position is updated via HC.V.move(v, tuple(x_new)) which maintains the cache.

Usage
-----
    from ddgclib.operators.stress import dudt_i
    from ddgclib.dynamic_integrators import euler_velocity_only

    # dudt_i is the Cauchy stress acceleration: a_i = F_stress_i / m_i
    # Pass dim, mu, HC as keyword args (forwarded by integrator):
    t = euler_velocity_only(HC, bV, dudt_i, dt=1e-4, n_steps=100,
                            dim=2, mu=0.1, HC=HC)

    # Or bind parameters with functools.partial:
    from functools import partial
    dudt_fn = partial(dudt_i, dim=3, mu=8.9e-4, HC=HC)
    t = euler_velocity_only(HC, bV, dudt_fn, dt=1e-4, n_steps=100)

    # With boundary conditions:
    from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
    bc_set = BoundaryConditionSet().add(NoSlipWallBC(dim=3), bV_wall)
    t = euler(HC, bV, dudt_i, dt=1e-4, n_steps=100, dim=3,
              bc_set=bc_set, mu=8.9e-4, HC=HC)
"""

import inspect
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay


# Helpers

def _retopologize(HC, bV, dim):
    """Retriangulate, recompute boundaries, and rebuild duals.

    Called at the start of every integrator time step to ensure that:
    1. Delaunay connectivity is correct after vertex movement
    2. Newly injected (inlet) and removed (outlet) vertices are handled
    3. All vertices have valid dual cells (``v.vd``) for stress operators

    Steps:
        1. Retriangulation — Delaunay for dim >= 2, sorted chain for dim == 1
        2. Boundary recomputation via ``HC.boundary()`` — update *bV* in-place
        3. Tag ``v.boundary`` on all vertices
        4. Recompute barycentric dual mesh via ``compute_vd``
    """
    from hyperct.ddg import compute_vd

    verts = list(HC.V)
    if len(verts) < dim + 1:
        return  # not enough vertices for a simplex

    # 1. Disconnect ALL existing edges
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)

    # 2. Retriangulate
    if dim == 1:
        # 1D: sort by coordinate and connect as a chain
        sorted_verts = sorted(verts, key=lambda v: v.x_a[0])
        for i in range(len(sorted_verts) - 1):
            sorted_verts[i].connect(sorted_verts[i + 1])
    else:
        # 2D/3D: Delaunay triangulation
        coords = np.array([v.x_a[:dim] for v in verts])
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    verts[simplex[i]].connect(verts[simplex[j]])

    # 3. Recompute boundary via HC.boundary()
    dV = HC.boundary()
    bV.clear()
    bV.update(dV)

    # 4. Tag v.boundary on all vertices
    for v in HC.V:
        v.boundary = v in bV

    # 5. Recompute barycentric duals
    compute_vd(HC, method="barycentric")


def _recompute_duals(HC):
    """Lightweight dual mesh recomputation after vertex position changes.

    Only recomputes barycentric dual cells (``v.vd``) without
    retriangulation or boundary detection.  Use this when vertex
    positions have changed but topology (vertex count, connectivity)
    has not.
    """
    from hyperct.ddg import compute_vd
    compute_vd(HC, method="barycentric")


def _maybe_save_state(save_every, save_dir, step, t, HC, bV,
                      fields=('u', 'p', 'm')):
    """Save simulation state to disk if save_every and save_dir are set.

    Parameters
    ----------
    save_every : int or None
        Save every N steps.  ``None`` disables saving.
    save_dir : str or None
        Directory for state files.  ``None`` disables saving.
    step : int
        Current step number.
    t : float
        Current simulation time.
    HC : Complex
        Simplicial complex.
    bV : set
        Boundary vertex set.
    fields : sequence of str
        Vertex attributes to save (default ``('u', 'p', 'm')``).
    """
    if save_every is None or save_dir is None:
        return
    if step % save_every != 0:
        return
    from ddgclib.data._io import save_state
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'state_{step:06d}_t{t:.6f}.json')
    save_state(HC, bV, t=t, fields=list(fields), path=path)


def _move(v, pos, HC, bV):
    """Move vertex, preserving boundary set membership."""
    if v in bV:
        bV.remove(v)
        HC.V.move(v, tuple(pos))
        bV.add(v)
    else:
        HC.V.move(v, tuple(pos))


def _interior_verts(HC, bV):
    """Return list of non-boundary vertices (stable ordering for one step)."""
    return [v for v in HC.V if v not in bV]


def _apply_bc_set(bc_set, HC, bV, dt):
    """Apply boundary condition set if provided."""
    if bc_set is not None:
        return bc_set.apply_all(HC, bV, dt)
    return {}


def _compute_accel(dudt_fn, verts, workers=None, **dudt_kwargs):
    """Compute acceleration for all interior vertices.

    When *workers* > 1, evaluations are distributed across threads
    using :class:`concurrent.futures.ThreadPoolExecutor`.  This is
    effective because the heavy computation (numpy C routines in
    ``dual_area_vector``, ``d_area``, etc.) releases the GIL.

    Parameters
    ----------
    dudt_fn : callable
        ``dudt_fn(v, **kwargs) -> ndarray``.
    verts : list
        Interior vertices to evaluate.
    workers : int or None
        Thread count.  ``None`` or 1 means sequential (default).
    **dudt_kwargs
        Extra keyword arguments forwarded to *dudt_fn*.

    Returns
    -------
    dict
        ``{v: accel_array}`` for every vertex in *verts*.
    """
    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        def _eval(v):
            return dudt_fn(v, **dudt_kwargs)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_eval, verts))
        return dict(zip(verts, results))
    return {v: dudt_fn(v, **dudt_kwargs) for v in verts}


def _invoke_callback(callback, step, t, HC, bV=None, diagnostics=None):
    """Call user callback, auto-detecting old (3-arg) vs new (5-arg) signature.

    Old signature: callback(step, t, HC)
    New signature: callback(step, t, HC, bV, diagnostics)
    """
    if callback is None:
        return
    try:
        sig = inspect.signature(callback)
        n_params = len(sig.parameters)
    except (ValueError, TypeError):
        n_params = 3  # fallback to old signature

    if n_params >= 5:
        callback(step, t, HC, bV, diagnostics)
    else:
        callback(step, t, HC)


def _pack_state(verts, dim):
    """Pack vertex positions and velocities into a flat state vector.

    Returns y = [x_0, ..., x_{N-1}, u_0, ..., u_{N-1}], length 2*N*dim.
    """
    n = len(verts)
    y = np.empty(2 * n * dim)
    for i, v in enumerate(verts):
        y[i * dim:(i + 1) * dim] = v.x_a[:dim]
        y[(n + i) * dim:(n + i + 1) * dim] = v.u[:dim]
    return y


def _unpack_state(y, n, dim):
    """Unpack flat state vector into position and velocity arrays."""
    x_flat = y[:n * dim]
    u_flat = y[n * dim:]
    return x_flat, u_flat


def _sync_mesh(verts, x_flat, u_flat, dim, HC, bV):
    """Push positions and velocities from flat arrays back onto the mesh."""
    for i, v in enumerate(verts):
        v.u[:dim] = u_flat[i * dim:(i + 1) * dim]
        x_new = x_flat[i * dim:(i + 1) * dim]
        _move(v, x_new, HC, bV)


# Euler (explicit, forward)

def euler(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None, bc_set=None,
          save_every=None, save_dir=None, workers=None, **dudt_kwargs):
    """Explicit (forward) Euler integration.

    Update rule per step::

        a^n     = dudt_fn(v)          for all interior vertices
        u^{n+1} = u^n  + dt * a^n
        x^{n+1} = x^n  + dt * u^n    (old velocity)
        apply bc_set (if provided)

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    bV : set
        Boundary vertex objects (skipped during integration).
    dudt_fn : callable
        ``dudt_fn(v, **dudt_kwargs) -> ndarray`` returning the acceleration.
    dt : float
        Time step.
    n_steps : int
        Number of steps.
    dim : int
        Spatial dimension (default 3).
    callback : callable or None
        ``callback(step, t, HC)`` (old) or
        ``callback(step, t, HC, bV, diagnostics)`` (new).
    bc_set : BoundaryConditionSet or None
        Applied after each step.
    save_every : int or None
        Save state to disk every N steps.  Requires *save_dir*.
    save_dir : str or None
        Directory for periodic state dumps (JSON via ``save_state``).
    **dudt_kwargs
        Forwarded to *dudt_fn* (e.g. ``dim=3, mu=8.9e-4``).

    Returns
    -------
    float
        Final time ``n_steps * dt``.
    """
    t = 0.0
    for step in range(n_steps):
        _retopologize(HC, bV, dim)
        verts = _interior_verts(HC, bV)

        accel = _compute_accel(dudt_fn, verts, workers, **dudt_kwargs)

        # Update: position uses OLD velocity, then velocity is advanced
        updates = {}
        for v in verts:
            x_new = v.x_a[:dim] + dt * v.u[:dim]
            u_new = v.u[:dim] + dt * accel[v][:dim]
            updates[v] = (x_new, u_new)

        for v, (x_new, u_new) in updates.items():
            v.u[:dim] = u_new
            _move(v, x_new, HC, bV)

        diagnostics = _apply_bc_set(bc_set, HC, bV, dt)
        t += dt
        _invoke_callback(callback, step, t, HC, bV, diagnostics)
        _maybe_save_state(save_every, save_dir, step, t, HC, bV)

    return t


# Symplectic (semi-implicit) Euler

def symplectic_euler(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None,
                     bc_set=None, save_every=None, save_dir=None,
                     workers=None, **dudt_kwargs):
    """Symplectic (semi-implicit) Euler integration.

    Update rule per step::

        a^n     = dudt_fn(v)
        u^{n+1} = u^n  + dt * a^n        (velocity first)
        x^{n+1} = x^n  + dt * u^{n+1}    (NEW velocity for position)
        apply bc_set (if provided)

    This is a symplectic integrator: it conserves a modified Hamiltonian
    and gives much better long-time energy behaviour than forward Euler.

    Parameters
    ----------
    HC, bV, dudt_fn, dt, n_steps, dim, callback, bc_set, **dudt_kwargs
        Same as :func:`euler`.
    save_every : int or None
        Save state to disk every N steps.  Requires *save_dir*.
    save_dir : str or None
        Directory for periodic state dumps (JSON via ``save_state``).

    Returns
    -------
    float
        Final time.
    """
    t = 0.0
    for step in range(n_steps):
        _retopologize(HC, bV, dim)
        verts = _interior_verts(HC, bV)

        accel = _compute_accel(dudt_fn, verts, workers, **dudt_kwargs)

        # Velocity first, then position with updated velocity
        updates = {}
        for v in verts:
            u_new = v.u[:dim] + dt * accel[v][:dim]
            x_new = v.x_a[:dim] + dt * u_new
            updates[v] = (x_new, u_new)

        for v, (x_new, u_new) in updates.items():
            v.u[:dim] = u_new
            _move(v, x_new, HC, bV)

        diagnostics = _apply_bc_set(bc_set, HC, bV, dt)
        t += dt
        _invoke_callback(callback, step, t, HC, bV, diagnostics)
        _maybe_save_state(save_every, save_dir, step, t, HC, bV)

    return t


# RK45 via scipy.integrate.solve_ivp

def rk45(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None, bc_set=None,
          rtol=1e-6, atol=1e-9, save_every=None, save_dir=None,
          workers=None, **dudt_kwargs):
    """Runge-Kutta 4(5) integration via :func:`scipy.integrate.solve_ivp`.

    The full coupled ODE system is solved::

        dy/dt = [ u_0, ..., u_{N-1},  dudt(v_0), ..., dudt(v_{N-1}) ]

    where ``y = [x_0, ..., x_{N-1}, u_0, ..., u_{N-1}]``.

    At each RHS evaluation the mesh is synchronised so that ``dudt_fn``
    sees the correct intermediate positions and velocities on all vertices.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    bV : set
        Boundary vertex objects (held fixed).
    dudt_fn : callable
        ``dudt_fn(v, **dudt_kwargs) -> ndarray``.
    dt : float
        Macro time step — the interval advanced per call to *solve_ivp*.
    n_steps : int
        Number of macro steps.  Total time = ``n_steps * dt``.
    dim : int
        Spatial dimension (default 3).
    callback : callable or None
        ``callback(step, t, HC)`` (old) or
        ``callback(step, t, HC, bV, diagnostics)`` (new).
    bc_set : BoundaryConditionSet or None
        Applied after each macro step.
    rtol, atol : float
        Relative / absolute tolerances forwarded to *solve_ivp*.
    save_every : int or None
        Save state to disk every N steps.  Requires *save_dir*.
    save_dir : str or None
        Directory for periodic state dumps (JSON via ``save_state``).
    **dudt_kwargs
        Forwarded to *dudt_fn*.

    Returns
    -------
    float
        Final time.

    Notes
    -----
    Each RHS evaluation moves all interior vertices via ``HC.V.move()``
    so that discrete operators (pressure gradient, viscous Laplacian, …)
    are evaluated at the correct intermediate geometry.  This is necessary
    because the RK stages sample the ODE at intermediate states.

    For large meshes the overhead of repeated ``HC.V.move()`` calls can be
    significant.  In such cases consider using :func:`symplectic_euler` with
    a smaller *dt*.
    """
    t = 0.0
    for step in range(n_steps):
        _retopologize(HC, bV, dim)
        verts = _interior_verts(HC, bV)
        n = len(verts)
        if n == 0:
            t += dt
            continue

        y0 = _pack_state(verts, dim)

        def rhs(_t, y):
            x_flat, u_flat = _unpack_state(y, n, dim)
            # Sync mesh to this intermediate state
            _sync_mesh(verts, x_flat, u_flat, dim, HC, bV)

            dydt = np.empty_like(y)
            # dx/dt = u
            dydt[:n * dim] = u_flat
            # du/dt = dudt_fn(v)  — parallel when workers > 1
            accel = _compute_accel(dudt_fn, verts, workers, **dudt_kwargs)
            for i, v in enumerate(verts):
                dydt[(n + i) * dim:(n + i + 1) * dim] = accel[v][:dim]
            return dydt

        sol = solve_ivp(
            rhs,
            t_span=(t, t + dt),
            y0=y0,
            method='RK45',
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"solve_ivp failed at step {step}: {sol.message}"
            )

        # Apply final state to mesh
        y_final = sol.y[:, -1]
        x_final, u_final = _unpack_state(y_final, n, dim)
        _sync_mesh(verts, x_final, u_final, dim, HC, bV)

        diagnostics = _apply_bc_set(bc_set, HC, bV, dt)
        t += dt
        _invoke_callback(callback, step, t, HC, bV, diagnostics)
        _maybe_save_state(save_every, save_dir, step, t, HC, bV)

    return t


# Velocity-only Euler (no position update, for fixed-mesh CFD)

def euler_velocity_only(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None,
                        bc_set=None, save_every=None, save_dir=None,
                        workers=None, **dudt_kwargs):
    """Explicit Euler that only advances velocity (mesh stays fixed).

    Useful for Eulerian CFD (e.g. Poiseuille flow on a static mesh) where
    only the velocity field evolves::

        u^{n+1} = u^n + dt * dudt_fn(v)
        apply bc_set (if provided)

    Parameters
    ----------
    HC, bV, dudt_fn, dt, n_steps, dim, callback, bc_set, **dudt_kwargs
        Same as :func:`euler`.
    save_every : int or None
        Save state to disk every N steps.  Requires *save_dir*.
    save_dir : str or None
        Directory for periodic state dumps (JSON via ``save_state``).

    Returns
    -------
    float
        Final time.
    """
    t = 0.0
    for step in range(n_steps):
        _retopologize(HC, bV, dim)
        verts = _interior_verts(HC, bV)
        accel = _compute_accel(dudt_fn, verts, workers, **dudt_kwargs)

        for v, a in accel.items():
            v.u[:dim] += dt * a[:dim]

        diagnostics = _apply_bc_set(bc_set, HC, bV, dt)
        t += dt
        _invoke_callback(callback, step, t, HC, bV, diagnostics)
        _maybe_save_state(save_every, save_dir, step, t, HC, bV)

    return t


# Adaptive Euler with CFL-based time stepping

def euler_adaptive(HC, bV, dudt_fn, dt_initial, t_end, dim=3, callback=None,
                   bc_set=None, cfl_target=0.5, dt_min=1e-12, dt_max=None,
                   velocity_only=True, save_every=None, save_dir=None,
                   workers=None, **dudt_kwargs):
    """Explicit Euler with CFL-based adaptive time stepping.

    The time step is adjusted each step based on the CFL condition::

        dt = cfl_target * h_min / max(|u|)

    where ``h_min`` is the minimum edge length and ``max(|u|)`` is the
    maximum velocity magnitude over all interior vertices.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    bV : set
        Boundary vertex objects.
    dudt_fn : callable
        ``dudt_fn(v, **dudt_kwargs) -> ndarray``.
    dt_initial : float
        Initial time step.
    t_end : float
        Final simulation time.
    dim : int
        Spatial dimension (default 3).
    callback : callable or None
        ``callback(step, t, HC)`` (old) or
        ``callback(step, t, HC, bV, diagnostics)`` (new).
    bc_set : BoundaryConditionSet or None
        Applied after each step.
    cfl_target : float
        Target CFL number (default 0.5).
    dt_min : float
        Minimum allowed time step.
    dt_max : float or None
        Maximum allowed time step. Defaults to dt_initial.
    velocity_only : bool
        If True, only update velocity (no position update). Default True.
    save_every : int or None
        Save state to disk every N steps.  Requires *save_dir*.
    save_dir : str or None
        Directory for periodic state dumps (JSON via ``save_state``).
    **dudt_kwargs
        Forwarded to *dudt_fn*.

    Returns
    -------
    float
        Final time reached.
    """
    if dt_max is None:
        dt_max = dt_initial

    t = 0.0
    dt = dt_initial
    step = 0

    while t < t_end - 1e-15:
        # Don't overshoot t_end
        dt = min(dt, t_end - t)

        _retopologize(HC, bV, dim)
        verts = _interior_verts(HC, bV)
        accel = _compute_accel(dudt_fn, verts, workers, **dudt_kwargs)

        if velocity_only:
            for v, a in accel.items():
                v.u[:dim] += dt * a[:dim]
        else:
            updates = {}
            for v in verts:
                x_new = v.x_a[:dim] + dt * v.u[:dim]
                u_new = v.u[:dim] + dt * accel[v][:dim]
                updates[v] = (x_new, u_new)
            for v, (x_new, u_new) in updates.items():
                v.u[:dim] = u_new
                _move(v, x_new, HC, bV)

        diagnostics = _apply_bc_set(bc_set, HC, bV, dt)
        diagnostics['dt'] = dt

        t += dt
        _invoke_callback(callback, step, t, HC, bV, diagnostics)
        _maybe_save_state(save_every, save_dir, step, t, HC, bV)
        step += 1

        # Adaptive CFL: compute new dt
        u_max = 0.0
        for v in verts:
            u_mag = np.linalg.norm(v.u[:dim])
            if u_mag > u_max:
                u_max = u_mag

        if u_max > 1e-30:
            # Estimate minimum edge length from any interior vertex
            h_min = float('inf')
            for v in verts:
                for nb in v.nn:
                    h = np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
                    if h < h_min:
                        h_min = h
            if h_min < float('inf'):
                dt_cfl = cfl_target * h_min / u_max
                dt = np.clip(dt_cfl, dt_min, dt_max)
            else:
                dt = dt_max
        else:
            dt = dt_max

    return t
