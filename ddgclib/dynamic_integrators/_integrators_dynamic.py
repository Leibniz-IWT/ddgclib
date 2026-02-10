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
    from ddgclib.barycentric._duals import dudt
    from ddgclib.dynamic_integrators._integrators_dynamic import euler

    t = euler(HC, bV, dudt, dt=1e-4, n_steps=100, dim=3, mu=8.9e-4)
"""

import numpy as np
from scipy.integrate import solve_ivp


# Helpers
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
def euler(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None, **dudt_kwargs):
    """Explicit (forward) Euler integration.

    Update rule per step::

        a^n     = dudt_fn(v)          for all interior vertices
        u^{n+1} = u^n  + dt * a^n
        x^{n+1} = x^n  + dt * u^n    (old velocity)

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
        ``callback(step, t, HC)`` called after each step.
    **dudt_kwargs
        Forwarded to *dudt_fn* (e.g. ``dim=3, mu=8.9e-4``).

    Returns
    -------
    float
        Final time ``n_steps * dt``.
    """
    t = 0.0
    for step in range(n_steps):
        verts = _interior_verts(HC, bV)

        # Evaluate acceleration at current state
        accel = {v: dudt_fn(v, **dudt_kwargs) for v in verts}

        # Update: position uses OLD velocity, then velocity is advanced
        updates = {}
        for v in verts:
            x_new = v.x_a[:dim] + dt * v.u[:dim]
            u_new = v.u[:dim] + dt * accel[v][:dim]
            updates[v] = (x_new, u_new)

        for v, (x_new, u_new) in updates.items():
            v.u[:dim] = u_new
            _move(v, x_new, HC, bV)

        t += dt
        if callback is not None:
            callback(step, t, HC)

    return t


# Symplectic (semi-implicit) Euler
def symplectic_euler(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None,
                     **dudt_kwargs):
    """Symplectic (semi-implicit) Euler integration.

    Update rule per step::

        a^n     = dudt_fn(v)
        u^{n+1} = u^n  + dt * a^n        (velocity first)
        x^{n+1} = x^n  + dt * u^{n+1}    (NEW velocity for position)

    This is a symplectic integrator: it conserves a modified Hamiltonian
    and gives much better long-time energy behaviour than forward Euler.

    Parameters
    ----------
    HC, bV, dudt_fn, dt, n_steps, dim, callback, **dudt_kwargs
        Same as :func:`euler`.

    Returns
    -------
    float
        Final time.
    """
    t = 0.0
    for step in range(n_steps):
        verts = _interior_verts(HC, bV)

        # Evaluate acceleration at current state
        accel = {v: dudt_fn(v, **dudt_kwargs) for v in verts}

        # Velocity first, then position with updated velocity
        updates = {}
        for v in verts:
            u_new = v.u[:dim] + dt * accel[v][:dim]
            x_new = v.x_a[:dim] + dt * u_new
            updates[v] = (x_new, u_new)

        for v, (x_new, u_new) in updates.items():
            v.u[:dim] = u_new
            _move(v, x_new, HC, bV)

        t += dt
        if callback is not None:
            callback(step, t, HC)

    return t


# RK45 via scipy.integrate.solve_ivp
def rk45(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None,
          rtol=1e-6, atol=1e-9, **dudt_kwargs):
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
        ``callback(step, t, HC)`` called after each macro step.
    rtol, atol : float
        Relative / absolute tolerances forwarded to *solve_ivp*.
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
            # du/dt = dudt_fn(v)
            for i, v in enumerate(verts):
                a = dudt_fn(v, **dudt_kwargs)
                dydt[(n + i) * dim:(n + i + 1) * dim] = a[:dim]
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

        t += dt
        if callback is not None:
            callback(step, t, HC)

    return t


# Velocity-only Euler (no position update, for fixed-mesh CFD)
def euler_velocity_only(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None,
                        **dudt_kwargs):
    """Explicit Euler that only advances velocity (mesh stays fixed).

    Useful for Eulerian CFD (e.g. Poiseuille flow on a static mesh) where
    only the velocity field evolves::

        u^{n+1} = u^n + dt * dudt_fn(v)

    Parameters
    ----------
    HC, bV, dudt_fn, dt, n_steps, dim, callback, **dudt_kwargs
        Same as :func:`euler`.

    Returns
    -------
    float
        Final time.
    """
    t = 0.0
    for step in range(n_steps):
        verts = _interior_verts(HC, bV)
        accel = {v: dudt_fn(v, **dudt_kwargs) for v in verts}

        for v, a in accel.items():
            v.u[:dim] += dt * a[:dim]

        t += dt
        if callback is not None:
            callback(step, t, HC)

    return t
