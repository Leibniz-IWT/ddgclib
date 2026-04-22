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


# Helpers

def _retopologize(HC, bV, dim, boundary_filter=None, merge_cdist=None,
                  periodic_axes=None, domain_bounds=None, backend=None,
                  skip_triangulation=False,
                  pressure_model=None, redistribute_mass=False,
                  remesh_mode='delaunay', remesh_kwargs=None):
    """Retriangulate, recompute boundaries, and rebuild duals.

    Called at the start of every integrator time step to ensure that:
    1. Delaunay connectivity is correct after vertex movement
    2. Newly injected (inlet) and removed (outlet) vertices are handled
    3. All vertices have valid dual cells (``v.vd``) for stress operators

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    bV : set
        Boundary vertex set (modified in-place).
    dim : int
        Spatial dimension.
    boundary_filter : callable or None
        If provided, ``boundary_filter(v) -> bool`` selects which
        topological boundary vertices are actually frozen (added to
        *bV*).  Vertices for which the filter returns ``False`` remain
        interior and participate in integration.  Typical usage: pass
        the wall criterion so that only wall vertices are frozen while
        inlet/outlet boundary vertices advect freely.
    merge_cdist : float or None
        If provided, merge vertices closer than this distance before
        retriangulation.  Prevents accumulation of near-duplicate
        vertices (e.g. from periodic inlet injection at wall positions).
        A good default is ``0.5 * min_edge_length``.
    periodic_axes : list[int] or None
        Axes along which the domain is periodic (e.g. ``[0]``).
        When set, delegates to :func:`retopologize_periodic`.
    domain_bounds : list[tuple[float, float]] or None
        Domain extent per axis.  Required when *periodic_axes* is set.
    skip_triangulation : bool
        If True, skip the disconnect/retriangulate steps (1-2) and keep
        the existing connectivity.  Boundary tagging, dual mesh
        recomputation (``compute_vd``), and dual volume caching are still
        performed.  Useful when the topology has not changed (e.g.
        Eulerian fixed-mesh or velocity-only integrators) but vertex
        positions have moved and duals need refreshing.
    pressure_model : EquationOfState or None
        EOS instance for mass redistribution.  Required when
        *redistribute_mass* is True.
    redistribute_mass : bool
        If True, redistribute vertex masses after retriangulation to
        preserve the pre-retriangulation pressure field.  Requires
        *pressure_model* with a ``.density(P)`` inverse method.
    remesh_mode : {'delaunay', 'adaptive'}
        Connectivity update strategy.

        - ``'delaunay'`` (default): disconnect all edges and run a
          global scipy Delaunay retriangulation.  Fast and robust but
          creates cross-phase edges at sharp interfaces.
        - ``'adaptive'``: use ``hyperct.remesh.adaptive_remesh`` to apply
          local mesh operations (edge split, edge collapse, edge flip)
          that preserve a sharp ``v.phase`` interface.  Requires 2D.
          Only applies when *skip_triangulation* is False.
    remesh_kwargs : dict or None
        Extra keyword arguments forwarded to
        :func:`hyperct.remesh.adaptive_remesh` (e.g. ``L_min``,
        ``L_max``, ``alpha_max``, ``quality_target_deg``).

    Steps:
        0. (Optional) Merge close vertices via ``HC.V.merge_all``
        1. Retriangulation — Delaunay for dim >= 2, sorted chain for dim == 1
           (skipped when *skip_triangulation* is True)
        2. Boundary recomputation via ``HC.boundary()`` — update *bV* in-place
           (skipped when *skip_triangulation* is True; uses existing *bV*)
        3. Tag ``v.boundary`` on all vertices
        4. Recompute barycentric dual mesh via ``compute_vd``
    """
    # Dispatch to periodic path if periodic_axes is set
    if periodic_axes:
        from ddgclib.geometry.periodic import retopologize_periodic
        retopologize_periodic(
            HC, bV, dim, periodic_axes, domain_bounds,
            boundary_filter=boundary_filter,
            merge_cdist=merge_cdist,
        )
        return

    from hyperct.ddg import compute_vd

    verts = list(HC.V)
    if len(verts) < dim + 1:
        return  # not enough vertices for a simplex

    # Snapshot pressure field before topology change (for mass redistribution)
    _p_snap = None
    if redistribute_mass and pressure_model is not None:
        from ddgclib.operators.mass_redistribution import snapshot_pressure
        _p_snap = snapshot_pressure(HC)

    if not skip_triangulation:
        # 0. Merge close vertices before retriangulation
        if merge_cdist is not None and merge_cdist > 0:
            HC.V.merge_all(cdist=merge_cdist)
            # Refresh vertex list and clean up stale bV references
            bV.intersection_update(set(HC.V))
            verts = list(HC.V)
            if len(verts) < dim + 1:
                return

        if remesh_mode == 'adaptive':
            # Local mesh operations preserving v.phase interfaces.
            # The existing connectivity is the starting point — no
            # global disconnect — so interior structure and interface
            # edges are retained.  For dim != 2 the adaptive driver
            # raises NotImplementedError, which we intentionally let
            # propagate so the user sees a clear error rather than a
            # silent fallback to Delaunay.
            from hyperct.remesh import adaptive_remesh
            adaptive_remesh(HC, dim=dim, **(remesh_kwargs or {}))
            # Recompute boundary from the updated connectivity.
            dV = HC.boundary()
        else:
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
                # 2D/3D: Delaunay triangulation with simplex caching for
                # the simplex-aware 3D dual path.
                from ddgclib.geometry import connect_and_cache_simplices
                coords = np.array([v.x_a[:dim] for v in verts])
                connect_and_cache_simplices(HC, verts, dim, coords=coords)

            # 3. Recompute boundary via HC.boundary()
            dV = HC.boundary()
    else:
        # skip_triangulation: keep existing connectivity,
        # use current bV as the boundary set
        dV = set(bV)

    # 4. Tag v.boundary on ALL topological boundary vertices.
    #    compute_vd needs the full boundary to build correct half-cells.
    for v in HC.V:
        v.boundary = v in dV

    # 5. Recompute barycentric duals (uses v.boundary)
    compute_vd(HC, method="barycentric")

    # 5b. Cache dual volumes and oriented edge areas for FVM operators.
    #     Use batch_e_star when available (vectorized, supports GPU backend).
    try:
        from hyperct.ddg import batch_e_star
        interior = [v for v in HC.V if v not in dV]
        edge_areas, failed, vols = batch_e_star(
            interior, HC, dim=dim, backend=backend,
            orient=True, compute_volumes=True,
        )
        for v in failed:
            v.boundary = True
            dV.add(v)
        for v in HC.V:
            v.dual_vol = vols.get(id(v), 0.0) if v not in dV else 0.0
        HC._edge_area_cache = edge_areas
    except (ImportError, NotImplementedError):
        from ddgclib.operators.stress import cache_dual_volumes
        cache_dual_volumes(HC, dim)
        HC._edge_area_cache = None

    # 6. Populate bV — controls which vertices are frozen (excluded
    #    from integration).  When boundary_filter is set, only matching
    #    vertices (e.g. walls) are frozen; the rest remain interior.
    if boundary_filter is not None:
        dV = {v for v in dV if boundary_filter(v)}
    bV.clear()
    bV.update(dV)

    # 7. Mass redistribution (pressure-preserving)
    if redistribute_mass and pressure_model is not None and _p_snap is not None:
        from ddgclib.operators.mass_redistribution import (
            redistribute_mass_single_phase,
        )
        redistribute_mass_single_phase(
            HC, dim, pressure_model, bV=bV, pressure_snapshot=_p_snap,
        )


def _do_retopologize(HC, bV, dim, boundary_filter=None, retopologize_fn=None,
                     merge_cdist=None, periodic_axes=None,
                     domain_bounds=None, backend=None,
                     skip_triangulation=False,
                     pressure_model=None, redistribute_mass=False,
                     remesh_mode='delaunay', remesh_kwargs=None):
    """Dispatch topology management to custom or default function.

    Parameters
    ----------
    retopologize_fn : callable, False, or None
        - ``None`` (default): use :func:`_retopologize` (Delaunay + compute_vd).
        - ``False``: skip topology management entirely.
        - callable: call ``retopologize_fn(HC, bV, dim)`` instead of the
          default.  ``remesh_mode`` and ``remesh_kwargs`` are forwarded
          as keyword arguments when the callable accepts them (detected
          via :mod:`inspect`), so existing 3-arg closures remain
          backward-compatible.  Useful for surface meshes where
          Delaunay/compute_vd don't apply, or for
          :func:`_retopologize_multiphase` wrappers.
    merge_cdist : float or None
        Forwarded to :func:`_retopologize`.  See its docstring.
    periodic_axes : list[int] or None
        Forwarded to :func:`_retopologize`.
    domain_bounds : list[tuple[float, float]] or None
        Forwarded to :func:`_retopologize`.
    skip_triangulation : bool
        If True, skip Delaunay retriangulation but still recompute duals
        and dual volumes.  Forwarded to :func:`_retopologize`.
        Ignored when *retopologize_fn* is a callable or False.
    pressure_model : EquationOfState or None
        Forwarded to :func:`_retopologize` for mass redistribution.
    redistribute_mass : bool
        Forwarded to :func:`_retopologize`.
    remesh_mode : {'delaunay', 'adaptive'}
        Forwarded to :func:`_retopologize` (and to custom
        ``retopologize_fn`` when it accepts the parameter).
    remesh_kwargs : dict or None
        Forwarded alongside *remesh_mode*.
    """
    if retopologize_fn is False:
        return
    if retopologize_fn is not None:
        # Forward remesh_mode/kwargs only if the callable declares them
        # (either by name or via **kwargs).  This preserves the legacy
        # 3-arg signature while allowing multiphase / adaptive wrappers
        # to opt in.
        extra = {}
        try:
            sig = inspect.signature(retopologize_fn)
            params = sig.parameters
            accepts_var_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            if accepts_var_kw or 'remesh_mode' in params:
                extra['remesh_mode'] = remesh_mode
            if accepts_var_kw or 'remesh_kwargs' in params:
                extra['remesh_kwargs'] = remesh_kwargs
        except (ValueError, TypeError):
            pass  # C-builtins, partials without __signature__, etc.
        retopologize_fn(HC, bV, dim, **extra)
    else:
        _retopologize(HC, bV, dim, boundary_filter=boundary_filter,
                      merge_cdist=merge_cdist,
                      periodic_axes=periodic_axes,
                      domain_bounds=domain_bounds,
                      backend=backend,
                      skip_triangulation=skip_triangulation,
                      pressure_model=pressure_model,
                      redistribute_mass=redistribute_mass,
                      remesh_mode=remesh_mode,
                      remesh_kwargs=remesh_kwargs)


def _retopologize_multiphase(HC, bV, dim, mps=None, boundary_filter=None,
                             merge_cdist=None, backend=None,
                             skip_triangulation=False,
                             redistribute_mass=False,
                             remesh_mode='delaunay', remesh_kwargs=None,
                             split_method='neighbour_count'):
    """Retriangulate with multiphase interface tracking.

    Performs standard Delaunay retopologization (or adaptive local
    mesh operations when ``remesh_mode='adaptive'``), then refreshes
    multiphase state: interface identification and mass fractions.

    Phase labels (``v.phase``) are vertex attributes and survive
    reconnection.

    Parameters
    ----------
    HC : Complex
    bV : set
    dim : int
    mps : MultiphaseSystem
        Multiphase system for interface identification.
    boundary_filter : callable or None
    merge_cdist : float or None
        Merge tolerance.  ``None`` disables merging (default).
        If merging is needed, use :func:`mass_conserving_merge`
        beforehand to preserve total mass.
    backend : str or None
    skip_triangulation : bool
        If True, skip Delaunay retriangulation but still recompute duals.
        Forwarded to :func:`_retopologize`.
    redistribute_mass : bool
        If True, redistribute per-phase masses after retriangulation to
        preserve the pre-retriangulation pressure field per phase.
    remesh_mode : {'delaunay', 'adaptive'}
        Connectivity update strategy forwarded to :func:`_retopologize`.
        ``'adaptive'`` uses ``hyperct.remesh.adaptive_remesh`` to apply
        local mesh operations (edge split, collapse, flip) that
        preserve the sharp ``v.phase`` interface — the primary reason
        this wrapper exists.  Currently only 2D is supported by the
        adaptive driver.
    remesh_kwargs : dict or None
        Extra keyword arguments forwarded to the adaptive driver.
    """
    # Snapshot per-phase pressure before topology change
    _p_snap = None
    if redistribute_mass and mps is not None:
        from ddgclib.operators.mass_redistribution import (
            snapshot_pressure_multiphase,
        )
        _p_snap = snapshot_pressure_multiphase(HC, mps.n_phases)

    # Retopologization (Delaunay or adaptive + duals, no single-phase redistrib)
    _retopologize(HC, bV, dim, boundary_filter=boundary_filter,
                  merge_cdist=merge_cdist, backend=backend,
                  skip_triangulation=skip_triangulation,
                  remesh_mode=remesh_mode,
                  remesh_kwargs=remesh_kwargs)

    # Refresh multiphase state
    if mps is not None:
        # reset_mass=False preserves Lagrangian mass (v.m, v.m_phase)
        # Only geometry (dual_vol_phase) and pressure are recomputed.
        # ``split_method='exact'`` uses the geometric 2D dual split
        # aligned with the per-edge phase split used by the per-phase
        # stress force; default 'neighbour_count' preserves the legacy
        # behaviour for callers that have not opted in.
        mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

        # Per-phase mass redistribution (after dual_vol_phase is available)
        if redistribute_mass and _p_snap is not None:
            from ddgclib.operators.mass_redistribution import (
                redistribute_mass_multiphase,
            )
            redistribute_mass_multiphase(
                HC, dim, mps, bV=bV, pressure_snapshot=_p_snap,
            )
            # Recompute pressures from the redistributed masses so that
            # v.p_phase (read by multiphase_stress_force) reflects the
            # adjusted densities, not the stale pre-redistribution values.
            mps.compute_phase_pressures(HC)


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

    When *workers* > 1, evaluations are distributed across processes
    using :mod:`multiprocessing` with the ``fork`` start method
    (Linux only).  Fork shares the parent's memory space copy-on-write,
    so the Complex and all vertex objects are accessible without
    pickling.  Vertex indices are passed instead of objects.

    Falls back to sequential evaluation on non-Linux platforms or
    when *workers* <= 1.

    Parameters
    ----------
    dudt_fn : callable
        ``dudt_fn(v, **kwargs) -> ndarray``.
    verts : list
        Interior vertices to evaluate.
    workers : int or None
        Process count.  ``None`` or 1 means sequential (default).
    **dudt_kwargs
        Extra keyword arguments forwarded to *dudt_fn*.

    Returns
    -------
    dict
        ``{v: accel_array}`` for every vertex in *verts*.
    """
    if workers and workers > 1:
        import sys
        if sys.platform != 'linux':
            # fork not available; fall back to sequential
            return {v: dudt_fn(v, **dudt_kwargs) for v in verts}

        import multiprocessing as mp
        # Store state in module globals so forked children can access it
        global _mp_dudt_fn, _mp_dudt_kwargs, _mp_verts
        _mp_dudt_fn = dudt_fn
        _mp_dudt_kwargs = dudt_kwargs
        _mp_verts = verts

        ctx = mp.get_context('fork')
        with ctx.Pool(workers) as pool:
            results = pool.map(_mp_eval_vertex, range(len(verts)))
        return dict(zip(verts, results))
    return {v: dudt_fn(v, **dudt_kwargs) for v in verts}


# Module-level globals for fork-based multiprocessing
_mp_dudt_fn = None
_mp_dudt_kwargs = None
_mp_verts = None


def _mp_eval_vertex(idx):
    """Evaluate dudt_fn on a vertex by index (for multiprocessing)."""
    return _mp_dudt_fn(_mp_verts[idx], **_mp_dudt_kwargs)


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
          save_every=None, save_dir=None, workers=None,
          boundary_filter=None, retopologize_fn=None, merge_cdist=None,
          backend=None, periodic_axes=None, domain_bounds=None,
          skip_triangulation=False,
          pressure_model=None, redistribute_mass=False,
          remesh_mode='delaunay', remesh_kwargs=None,
          **dudt_kwargs):
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
    boundary_filter : callable or None
        See :func:`_retopologize`.
    retopologize_fn : callable, False, or None
        See :func:`_do_retopologize`.  Use a custom callable for surface
        meshes where Delaunay/compute_vd don't apply.
    skip_triangulation : bool
        If True, skip Delaunay retriangulation but still recompute duals.
        See :func:`_retopologize`.
    remesh_mode : {'delaunay', 'adaptive'}
        Connectivity update strategy forwarded to :func:`_retopologize`.
        ``'adaptive'`` (2D only) replaces the global Delaunay step
        with local interface-preserving mesh operations from
        :mod:`hyperct.remesh`, which is the only mode that keeps a
        sharp ``v.phase`` interface intact across remeshes.
    remesh_kwargs : dict or None
        Extra kwargs forwarded to ``adaptive_remesh`` (e.g.
        ``L_min``, ``L_max``, ``quality_target_deg``,
        ``max_iterations``, ``smooth_iterations``).  Ignored when
        ``remesh_mode='delaunay'``.
    **dudt_kwargs
        Forwarded to *dudt_fn* (e.g. ``dim=3, mu=8.9e-4``).

    Returns
    -------
    float
        Final time ``n_steps * dt``.
    """
    t = 0.0
    for step in range(n_steps):
        _do_retopologize(HC, bV, dim, boundary_filter, retopologize_fn,
                             merge_cdist, periodic_axes=periodic_axes,
                             domain_bounds=domain_bounds, backend=backend,
                             skip_triangulation=skip_triangulation,
                             pressure_model=pressure_model,
                             redistribute_mass=redistribute_mass,
                             remesh_mode=remesh_mode,
                             remesh_kwargs=remesh_kwargs)
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
                     workers=None, boundary_filter=None,
                     retopologize_fn=None, merge_cdist=None,
                     backend=None, periodic_axes=None, domain_bounds=None,
                     skip_triangulation=False,
                     pressure_model=None, redistribute_mass=False,
                     remesh_mode='delaunay', remesh_kwargs=None,
                     **dudt_kwargs):
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
    boundary_filter : callable or None
        See :func:`_retopologize`.
    retopologize_fn : callable, False, or None
        See :func:`_do_retopologize`.
    skip_triangulation : bool
        If True, skip Delaunay retriangulation but still recompute duals.
        See :func:`_retopologize`.
    remesh_mode : {'delaunay', 'adaptive'}
        Connectivity update strategy forwarded to :func:`_retopologize`.
        ``'adaptive'`` (2D only) replaces the global Delaunay step
        with local interface-preserving mesh operations from
        :mod:`hyperct.remesh`, which is the only mode that keeps a
        sharp ``v.phase`` interface intact across remeshes.
    remesh_kwargs : dict or None
        Extra kwargs forwarded to ``adaptive_remesh`` (e.g.
        ``L_min``, ``L_max``, ``quality_target_deg``,
        ``max_iterations``, ``smooth_iterations``).  Ignored when
        ``remesh_mode='delaunay'``.

    Returns
    -------
    float
        Final time.
    """
    t = 0.0
    for step in range(n_steps):
        _do_retopologize(HC, bV, dim, boundary_filter, retopologize_fn,
                             merge_cdist, periodic_axes=periodic_axes,
                             domain_bounds=domain_bounds, backend=backend,
                             skip_triangulation=skip_triangulation,
                             pressure_model=pressure_model,
                             redistribute_mass=redistribute_mass,
                             remesh_mode=remesh_mode,
                             remesh_kwargs=remesh_kwargs)
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
          workers=None, boundary_filter=None, retopologize_fn=None,
          merge_cdist=None, backend=None, periodic_axes=None,
          domain_bounds=None, skip_triangulation=False,
          pressure_model=None, redistribute_mass=False,
          remesh_mode='delaunay', remesh_kwargs=None,
          **dudt_kwargs):
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
    boundary_filter : callable or None
        See :func:`_retopologize`.
    retopologize_fn : callable, False, or None
        See :func:`_do_retopologize`.
    skip_triangulation : bool
        If True, skip Delaunay retriangulation but still recompute duals.
        See :func:`_retopologize`.
    remesh_mode : {'delaunay', 'adaptive'}
        Connectivity update strategy forwarded to :func:`_retopologize`.
        ``'adaptive'`` (2D only) replaces the global Delaunay step
        with local interface-preserving mesh operations from
        :mod:`hyperct.remesh`, which is the only mode that keeps a
        sharp ``v.phase`` interface intact across remeshes.
    remesh_kwargs : dict or None
        Extra kwargs forwarded to ``adaptive_remesh`` (e.g.
        ``L_min``, ``L_max``, ``quality_target_deg``,
        ``max_iterations``, ``smooth_iterations``).  Ignored when
        ``remesh_mode='delaunay'``.
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
        _do_retopologize(HC, bV, dim, boundary_filter, retopologize_fn,
                             merge_cdist, periodic_axes=periodic_axes,
                             domain_bounds=domain_bounds, backend=backend,
                             skip_triangulation=skip_triangulation,
                             pressure_model=pressure_model,
                             redistribute_mass=redistribute_mass,
                             remesh_mode=remesh_mode,
                             remesh_kwargs=remesh_kwargs)
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
                        workers=None, boundary_filter=None,
                        retopologize_fn=None, merge_cdist=None,
                        backend=None, periodic_axes=None,
                        domain_bounds=None, skip_triangulation=False,
                        pressure_model=None, redistribute_mass=False,
                        remesh_mode='delaunay', remesh_kwargs=None,
                        **dudt_kwargs):
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
    boundary_filter : callable or None
        See :func:`_retopologize`.
    retopologize_fn : callable, False, or None
        See :func:`_do_retopologize`.
    skip_triangulation : bool
        If True, skip Delaunay retriangulation but still recompute duals.
        See :func:`_retopologize`.
    remesh_mode : {'delaunay', 'adaptive'}
        Connectivity update strategy forwarded to :func:`_retopologize`.
        ``'adaptive'`` (2D only) replaces the global Delaunay step
        with local interface-preserving mesh operations from
        :mod:`hyperct.remesh`, which is the only mode that keeps a
        sharp ``v.phase`` interface intact across remeshes.
    remesh_kwargs : dict or None
        Extra kwargs forwarded to ``adaptive_remesh`` (e.g.
        ``L_min``, ``L_max``, ``quality_target_deg``,
        ``max_iterations``, ``smooth_iterations``).  Ignored when
        ``remesh_mode='delaunay'``.

    Returns
    -------
    float
        Final time.
    """
    t = 0.0
    for step in range(n_steps):
        _do_retopologize(HC, bV, dim, boundary_filter, retopologize_fn,
                             merge_cdist, periodic_axes=periodic_axes,
                             domain_bounds=domain_bounds, backend=backend,
                             skip_triangulation=skip_triangulation,
                             pressure_model=pressure_model,
                             redistribute_mass=redistribute_mass,
                             remesh_mode=remesh_mode,
                             remesh_kwargs=remesh_kwargs)
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
                   workers=None, boundary_filter=None,
                   retopologize_fn=None, merge_cdist=None,
                   backend=None, periodic_axes=None,
                   domain_bounds=None, skip_triangulation=False,
                   pressure_model=None, redistribute_mass=False,
                   remesh_mode='delaunay', remesh_kwargs=None,
                   **dudt_kwargs):
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
    boundary_filter : callable or None
        See :func:`_retopologize`.
    retopologize_fn : callable, False, or None
        See :func:`_do_retopologize`.
    skip_triangulation : bool
        If True, skip Delaunay retriangulation but still recompute duals.
        See :func:`_retopologize`.
    remesh_mode : {'delaunay', 'adaptive'}
        Connectivity update strategy forwarded to :func:`_retopologize`.
        ``'adaptive'`` (2D only) replaces the global Delaunay step
        with local interface-preserving mesh operations from
        :mod:`hyperct.remesh`, which is the only mode that keeps a
        sharp ``v.phase`` interface intact across remeshes.
    remesh_kwargs : dict or None
        Extra kwargs forwarded to ``adaptive_remesh`` (e.g.
        ``L_min``, ``L_max``, ``quality_target_deg``,
        ``max_iterations``, ``smooth_iterations``).  Ignored when
        ``remesh_mode='delaunay'``.
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

        _do_retopologize(HC, bV, dim, boundary_filter, retopologize_fn,
                             merge_cdist, periodic_axes=periodic_axes,
                             domain_bounds=domain_bounds, backend=backend,
                             skip_triangulation=skip_triangulation,
                             pressure_model=pressure_model,
                             redistribute_mass=redistribute_mass,
                             remesh_mode=remesh_mode,
                             remesh_kwargs=remesh_kwargs)
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
