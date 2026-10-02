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

import functools
import inspect
import os

import numpy as np
from scipy.integrate import solve_ivp


# Helpers

# NOTE(laneL): measured on a channel (rectangle(L=2, h=1), walls y = 0, 1).
# One adaptive retopology splits wall edges and the 8 new wall-line
# vertices are not members (10 of 18 frozen against 18 of 18 under 'hull');
# with 7 members off the hull, adaptive_remesh moved up to 7 of them
# (Laplacian smoothing, up to 1.7e-01) and removed up to 5 (edge collapse).
_MEMBERSHIP_ADAPTIVE_MSG = (
    "frozen_set='membership' is not implemented with remesh_mode='adaptive': "
    "hyperct.remesh protects vertices by the topological tag v.boundary, "
    "not by bV, so a vertex created by splitting a wall edge is not a "
    "member (it is integrated and leaves the wall) and a member that is "
    "off the hull is collapsed and smoothed like an interior vertex"
)


def _retopologize(HC, bV, dim, boundary_filter=None, merge_cdist=None,
                  periodic_axes=None, domain_bounds=None, backend=None,
                  skip_triangulation=False,
                  pressure_model=None, redistribute_mass=False,
                  remesh_mode='delaunay', remesh_kwargs=None,
                  retopo_remap=None, frozen_set='hull'):
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
    retopo_remap : {None, 'conservative'}
        Single-phase conservative remap of the thermodynamic state across
        the connectivity rebuild (default ``None``: previous behaviour,
        bit-identical).  A reconnection changes the barycentric dual
        volume of a vertex by 33-100 % at fixed positions; with masses
        held the EOS reads that as compression and the run blows up at
        any time step (laneK).  ``'conservative'`` makes the pressure
        field invariant across the rebuild:

        1. snapshot ``p_i = eos(m_i / Vol_i)`` with ``Vol_i`` re-measured
           on the OLD connectivity at the current positions
           (:func:`~ddgclib.operators.mass_redistribution.snapshot_pressure_fresh`),
           so the step's physical compression is kept;
        2. rebuild connectivity and duals;
        3. re-target the mass of EVERY vertex with a dual volume,
           frozen boundary vertices included, to that pressure and
           rescale once so total mass is conserved exactly.

        Requires *pressure_model* (an EOS with ``.density``) and
        ``redistribute_mass=True``; no-op when *skip_triangulation* is
        True; not available on the periodic path.  The multiphase
        counterpart is ``_retopologize_multiphase(retopo_remap=...)``.
        See docs_temp/debug_session/laneK-single-phase-eos-instability.md.
    frozen_set : {'hull', 'membership'}
        Which vertices end up in *bV*, the set the integrators do not
        move (method axis ``frozen_set``).

        - ``'hull'`` (default: previous behaviour, bit-identical): *bV*
          is rebuilt from the topological boundary of this call's
          connectivity, narrowed by *boundary_filter*.  Nothing keeps a
          vertex behind a wall, and one vertex that steps past a straight
          wall takes the wall vertices next to it off the hull, i.e. out
          of *bV*: the wall is integrated and collapses (audit
          2026-09-25, F10 C1).
        - ``'membership'``: *bV* is persistent.  This call keeps the
          members that are still in the complex and, if *boundary_filter*
          is given, pass it (that is how a runner's initial "whole hull"
          set is narrowed to the walls).  It never adds a vertex because
          it is on the hull and never drops one because it is not.
          ``v.boundary``, which ``compute_vd`` needs for the half cells,
          still follows the topological boundary, so the two sets are
          no longer the same:

          * a hull vertex that is not a member (inlet, outlet, free
            surface, a vertex that left through a wall) is tagged, gets
            a half cell and is integrated;
          * a member that the hull no longer contains stays frozen and
            gets a closed dual cell;
          * a vertex that reaches a wall is not captured here.  That is
            the decision of a BC that holds *bV*
            (``PositionalNoSlipWallBC(bV=bV)`` adds what meets its
            criterion, and the addition now persists).

          With *skip_triangulation* the boundary tag is read from the
          kept connectivity instead of being carried in *bV*.  In 3D a
          vertex whose dual fan fails is tagged and zero-volumed as
          before but not frozen.  Not available on the periodic path
          and not with ``remesh_mode='adaptive'`` (both raise):
          ``hyperct.remesh`` knows the topological tag ``v.boundary``
          only, so a vertex it creates by splitting a wall edge is not
          a member, and a member off the hull is collapsed and
          smoothed like an interior vertex.
          See docs_temp/debug_session/laneL-frozen-set-membership.md.

    Steps:
        0. (Optional) Merge close vertices via ``HC.V.merge_all``
        1. Retriangulation — Delaunay for dim >= 2, sorted chain for dim == 1
           (skipped when *skip_triangulation* is True)
        2. Boundary recomputation via ``HC.boundary()`` — update *bV* in-place
           (skipped when *skip_triangulation* is True; uses existing *bV*)
        3. Tag ``v.boundary`` on all vertices
        4. Recompute barycentric dual mesh via ``compute_vd``
    """
    if retopo_remap not in (None, 'conservative'):
        raise ValueError(
            f"retopo_remap must be None or 'conservative', "
            f"got {retopo_remap!r}"
        )
    if frozen_set not in ('hull', 'membership'):
        raise ValueError(
            f"frozen_set must be 'hull' or 'membership', got {frozen_set!r}"
        )
    membership = frozen_set == 'membership'
    if membership and periodic_axes:
        raise ValueError(
            "frozen_set='membership' is not implemented on the periodic "
            "retopology path"
        )
    if membership and remesh_mode == 'adaptive' and not skip_triangulation:
        raise ValueError(_MEMBERSHIP_ADAPTIVE_MSG)
    remap_active = retopo_remap == 'conservative' and not skip_triangulation
    if remap_active:
        if periodic_axes:
            raise ValueError(
                "retopo_remap='conservative' is not implemented on the "
                "periodic retopology path"
            )
        if not (redistribute_mass and hasattr(pressure_model, 'density')):
            raise ValueError(
                "retopo_remap='conservative' requires redistribute_mass=True "
                "and an EquationOfState pressure_model"
            )

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
    if remap_active:
        # NOTE(laneR): fresh snapshot on the OLD connectivity at the
        # current positions, not the stale v.p of the last force
        # evaluation (which would erase this step's compression).
        from ddgclib.operators.mass_redistribution import (
            snapshot_pressure_fresh,
        )
        _p_snap = snapshot_pressure_fresh(HC, dim, pressure_model)
    elif redistribute_mass and pressure_model is not None:
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
            from hyperct.ddg import (
                invalidate_simplex_cache,
                rebuild_simplex_cache_2d,
            )
            adaptive_remesh(HC, dim=dim, **(remesh_kwargs or {}))
            # Adaptive remesh mutates connectivity locally without going
            # through connect_and_cache_simplices, so the simplex cache
            # (if any) is stale.  In 2D, REBUILD it from the updated
            # 1-skeleton (ghost-K3 filtered) instead of dropping it:
            # losing the cache silently downgrades compute_vd, boundary
            # tagging and the exact dual volumes to the 1-skeleton
            # fallbacks, which pumps kinetic energy into dynamic runs
            # (adaptive KE tail 4.4x -> ~1x on the oscillating droplet,
            # lane4-remesh-upstream 2026-07-02).
            if dim == 2:
                rebuild_simplex_cache_2d(HC)
            else:
                invalidate_simplex_cache(HC)
            # Recompute boundary — prefer the exact simplex-aware path
            # (parity with the Delaunay branch below).
            if getattr(HC, '_simplices', None) is not None:
                from hyperct.ddg import boundary_from_simplices
                dV = boundary_from_simplices(HC, dim)
            else:
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
                # 2D/3D: Delaunay triangulation with simplex caching
                # (used by simplex-aware boundary + compute_vd paths).
                from hyperct.ddg import connect_and_cache_simplices
                coords = np.array([v.x_a[:dim] for v in verts])
                connect_and_cache_simplices(HC, verts, dim, coords=coords)

            # 3. Recompute boundary — prefer the exact simplex-aware
            #    path when the cache was just populated (2D / 3D).
            if getattr(HC, '_simplices', None) is not None:
                from hyperct.ddg import boundary_from_simplices
                dV = boundary_from_simplices(HC, dim)
            else:
                dV = HC.boundary()
    elif membership:
        # NOTE(laneL): bV is the frozen set here, not the boundary, so
        # the boundary of the kept connectivity is read, not carried.
        if getattr(HC, '_simplices', None) is not None:
            from hyperct.ddg import boundary_from_simplices
            dV = boundary_from_simplices(HC, dim)
        else:
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
        # NOTE(lane3-dual-volume): in 3D, prefer the exact
        # simplex-container volumes (hyperct.ddg.simplex_dual_volumes,
        # Vol_i = (1/(dim+1)) * sum_{T ∋ i} |T|) over batch_e_star's
        # fan-walk volumes, which undercount 1-4% interior on
        # unstructured 3D meshes (docs_temp/audit/dual-volume-3d.md).
        # Enabled 2026-07-29 together with the matching
        # cache_dual_volumes/dual_volume dim==3 branches in stress.py
        # (mixed volume sources across setup/retopo create a
        # first-retopo pressure jump) and the canonical 3D qhull input
        # order in hyperct connect_and_cache_simplices
        # (NOTE(laneA-canonical-order), kills the order-dependent
        # settle-step artifact); the pinned 3D static-droplet
        # retopology floor was re-pinned 7.3768e-5 -> 7.274172e-5
        # accordingly (the switch alone measures 7.616854e-5: the
        # exact measure honestly reports the larger settle-step volume
        # jump that the redistribution rescale converts into a uniform
        # pressure offset).  2D keeps
        # batch_e_star's volumes here (bit-identical to the validated
        # 2D baseline; the 2D fan walk is exact on interior vertices).
        # Boundary zeroing convention preserved in both paths.  See
        # docs_temp/debug_session/lane3-exact-dual-volumes.md.
        exact_vols = None
        if dim == 3:
            from ddgclib.operators.stress import (
                _use_exact_barycentric_volume,
            )
            if _use_exact_barycentric_volume(HC):
                from hyperct.ddg import simplex_dual_volumes
                exact_vols = simplex_dual_volumes(HC, dim)
        if exact_vols is not None:
            for v in HC.V:
                v.dual_vol = exact_vols.get(v, 0.0) if v not in dV else 0.0
        else:
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
    if membership:
        # NOTE(laneL): persistent wall membership.  The hull of this
        # rebuild decides v.boundary (above) but not who is frozen: a
        # member stays frozen when it is off the hull, and a hull vertex
        # that is not a member is integrated.
        dV = {v for v in bV if HC.V.cache.get(v.x) is v
              and (boundary_filter is None or boundary_filter(v))}
    elif boundary_filter is not None:
        dV = {v for v in dV if boundary_filter(v)}
    bV.clear()
    bV.update(dV)

    # 7. Mass redistribution (pressure-preserving)
    if redistribute_mass and pressure_model is not None and _p_snap is not None:
        from ddgclib.operators.mass_redistribution import (
            redistribute_mass_single_phase,
        )
        if remap_active:
            # NOTE(laneR): frozen (bV) vertices are re-targeted too; the
            # interior-only remap leaves the wall-cell flip jumps in.
            redistribute_mass_single_phase(
                HC, dim, pressure_model, bV=bV, pressure_snapshot=_p_snap,
                include_frozen=True,
            )
        else:
            redistribute_mass_single_phase(
                HC, dim, pressure_model, bV=bV, pressure_snapshot=_p_snap,
            )


def _displacement_gate_should_skip(HC, displacement_eps):
    """Skip-retopology gate based on max vertex displacement.

    Returns ``True`` when retopology may be skipped because every vertex
    has moved less than *displacement_eps* since the last call (and the
    vertex set is unchanged).  Otherwise returns ``False`` and the
    caller should run a full retopology, then call
    :func:`_snapshot_retopo_positions` to refresh the cache.

    On the first call (no snapshot yet), takes a snapshot from the
    current positions and returns ``True`` — the rationale is that
    callers opt into the gate by passing *displacement_eps*, which
    implies trust that the existing duals (built by setup) are valid
    for the current geometry; running an immediate full retopo on the
    very first integrator step would defeat the purpose of the gate
    in the static-equilibrium case.
    """
    prev = getattr(HC, '_retopo_prev_positions', None)
    if prev is None:
        _snapshot_retopo_positions(HC)
        return True
    current_ids = {id(v) for v in HC.V}
    if current_ids != set(prev.keys()):
        return False
    eps = float(displacement_eps)
    for v in HC.V:
        dx = np.linalg.norm(np.asarray(v.x_a) - prev[id(v)])
        if dx >= eps:
            return False
    return True


def _snapshot_retopo_positions(HC):
    """Cache current vertex positions for the displacement gate."""
    HC._retopo_prev_positions = {
        id(v): np.asarray(v.x_a).copy() for v in HC.V
    }


def _do_retopologize(HC, bV, dim, boundary_filter=None, retopologize_fn=None,
                     merge_cdist=None, periodic_axes=None,
                     domain_bounds=None, backend=None,
                     skip_triangulation=False,
                     pressure_model=None, redistribute_mass=False,
                     remesh_mode='delaunay', remesh_kwargs=None,
                     displacement_eps=None):
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
          backward-compatible.  The remaining retopology kwargs of this
          function (``skip_triangulation``, ``boundary_filter``,
          ``merge_cdist``, ``backend``, ``periodic_axes``,
          ``domain_bounds``, ``pressure_model``, ``redistribute_mass``)
          are forwarded only when the callable declares them by name
          and a :func:`functools.partial` chain does not already bind
          them (explicit partial bindings win).  Useful for surface
          meshes where Delaunay/compute_vd don't apply, or for
          :func:`_retopologize_multiphase` wrappers.
    merge_cdist : float or None
        Forwarded to :func:`_retopologize`.  See its docstring.
    periodic_axes : list[int] or None
        Forwarded to :func:`_retopologize`.
    domain_bounds : list[tuple[float, float]] or None
        Forwarded to :func:`_retopologize`.
    skip_triangulation : bool
        If True, skip Delaunay retriangulation but still recompute duals
        and dual volumes.  Forwarded to :func:`_retopologize`, and to a
        callable *retopologize_fn* that declares the parameter by name
        (unless the callable is a :func:`functools.partial` that already
        binds it — explicit partial bindings always win).  Ignored when
        *retopologize_fn* is False.
    pressure_model : EquationOfState or None
        Forwarded to :func:`_retopologize` for mass redistribution.
    redistribute_mass : bool
        Forwarded to :func:`_retopologize`.
    remesh_mode : {'delaunay', 'adaptive'}
        Forwarded to :func:`_retopologize` (and to custom
        ``retopologize_fn`` when it accepts the parameter).
    remesh_kwargs : dict or None
        Forwarded alongside *remesh_mode*.
    displacement_eps : float or None
        Skip-retopology gate threshold.  When set to a positive number,
        retopology is short-circuited when every vertex has moved less
        than *displacement_eps* since the last call (and the vertex set
        has not changed).  This avoids 3D Delaunay non-uniqueness
        churning ~48 cross-phase edges per step on a near-cospherical
        static interface cloud (Phase 2 finding 2026-04-29 — the
        dominant residual driver for the static-droplet 3D residual
        once the redistribute_mass guard fix is in place).

        On the first call no snapshot exists; positions are snapshotted
        and the call is short-circuited too (the gate assumes the
        existing duals built by setup are valid).  Pass ``None`` (the
        default) to disable the gate entirely and preserve the previous
        every-step behaviour.

        Suggested first cut for dynamic runs: ``1e-4 * h_min`` where
        ``h_min`` is the minimum edge length.
    """
    if retopologize_fn is False:
        return

    if displacement_eps is not None and displacement_eps > 0:
        if _displacement_gate_should_skip(HC, displacement_eps):
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
            # NOTE(laneF-forward): keywords already bound in a
            # functools.partial chain are the case's explicit retopo
            # configuration (e.g. dual_only wrappers bind
            # skip_triangulation=True in the partial) and must never be
            # overridden by the integrator-level values.
            bound_kw = set()
            fn = retopologize_fn
            while isinstance(fn, functools.partial):
                bound_kw.update((fn.keywords or {}).keys())
                fn = fn.func
            # NOTE(laneF-forward): these integrator kwargs used to be
            # silently DROPPED for callable retopo_fns — the shipped dam
            # break ran per-step full Delaunay despite passing
            # skip_triangulation=True (laneD §2.3).  Forward them when
            # the callable declares them BY NAME (not into **kwargs
            # sinks, which legacy dual-only closures use to ignore
            # unknown keys) and the name is not partial-bound.
            for name, value in (
                ('skip_triangulation', skip_triangulation),
                ('boundary_filter', boundary_filter),
                ('merge_cdist', merge_cdist),
                ('backend', backend),
                ('periodic_axes', periodic_axes),
                ('domain_bounds', domain_bounds),
                ('pressure_model', pressure_model),
                ('redistribute_mass', redistribute_mass),
            ):
                if (name in params and name not in bound_kw
                        and params[name].kind is not
                        inspect.Parameter.VAR_KEYWORD):
                    extra[name] = value
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

    if displacement_eps is not None and displacement_eps > 0:
        _snapshot_retopo_positions(HC)


def _retopologize_multiphase(HC, bV, dim, mps=None, boundary_filter=None,
                             merge_cdist=None, backend=None,
                             skip_triangulation=False,
                             redistribute_mass=False,
                             remesh_mode='delaunay', remesh_kwargs=None,
                             split_method='neighbour_count',
                             retopo_remap=None,
                             projection_every=1,
                             frozen_set='hull'):
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
    retopo_remap : {None, 'conservative'}
        Opt-in conservative remap of the thermodynamic state across the
        connectivity rebuild (default ``None`` — previous behaviour,
        bit-identical).  ``'conservative'`` makes the pressure field
        exactly invariant across the rebuild: since vertex positions
        are frozen inside this call, any change the reconnection makes
        to measured dual volumes is a measurement artifact, not a
        physical compression, and must not enter the EOS.  Two stages:

        1. Physical update on the OLD connectivity — the validated
           dual-only per-step sequence (dual refresh at the current
           positions, per-phase split, pressure-preserving mass
           redistribution, EOS pressures).
        2. Connectivity rebuild forced pressure-neutral — full
           retriangulation + refresh + redistribution against the
           stage-1 snapshot (per-vertex inertia consistent with the
           new dual cells, exact per-phase mass conservation), then
           :func:`restore_pressure_multiphase` cancels the residual
           uniform per-phase offset ``~K_k*(scale_k - 1)`` that the
           global mass-conservation rescale would otherwise inject as
           a per-step interface jolt (see
           docs_temp/debug_session/laneD-conservative-retopo-remap.md).

        Requires *mps* and ``redistribute_mass=True``; no-op when
        *skip_triangulation* is True (nothing to remap).
    projection_every : int
        Cadence of the pressure-structure PROJECTION — the step where
        per-phase masses are re-targeted to reproduce the *pre-call*
        pressure snapshot (``redistribute_mass_multiphase`` against the
        field of the previous step).  Default 1 preserves the previous
        every-call behaviour bit-exactly.

        NOTE(laneH 2026-07-30): applied every call, the projection
        erases each step's local EOS compression response, so the
        pressure STRUCTURE can never evolve and the interface relaxes
        far too fast (2D droplet l=2 amplitude decay 3-4x the
        analytical rate; l2 0.17479 on the pinned benchmark).  With
        ``projection_every=N`` (N in 5..20, saturated dose-response)
        the local compression response accumulates between projection
        calls while the occasional projection still damps the spurious
        acoustic launch transient (the ``redistribute_mass=False``
        end-member rings at KE ~1000x the mode level): benchmark l2
        drops to 0.0363 and the trajectory lands on the exact two-fluid
        reference to ~2% (l2_two_fluid 0.0203, KE-shape correlation
        0.997).  See docs_temp/debug_session/laneH-2d-over-decay.md.

        Cadence semantics per path:

        - ``skip_triangulation=True`` (dual_only): the redistribution
          block simply does not run on off-cadence calls (mass stays
          Lagrangian; duals/splits/EOS pressures still refresh).
        - ``retopo_remap='conservative'``: the remap machinery runs on
          EVERY call (reconnection neutrality is not optional), but on
          off-cadence calls the snapshot that redistribution/restore
          reproduce is taken AFTER the stage-1 dual refresh at the new
          positions — the physically EVOLVED field — instead of before
          the call, so only the connectivity-rebuild artifact is
          projected out, not the step's compression response.
        - plain Delaunay without the remap: ``projection_every > 1``
          raises — skipping redistribution while reconnection fires
          re-opens the KE pump (lane-5 measured mechanism).

        Requires *mps* and ``redistribute_mass=True`` when > 1.  The
        call counter lives on ``mps._projection_call_idx`` (the first
        call always projects).
    frozen_set : {'hull', 'membership'}
        Forwarded to every :func:`_retopologize` call of this function
        (default ``'hull'``: previous behaviour).  See its docstring.
        ``'membership'`` with ``remesh_mode='adaptive'`` raises.
    """
    if retopo_remap not in (None, 'conservative'):
        raise ValueError(
            f"retopo_remap must be None or 'conservative', "
            f"got {retopo_remap!r}"
        )
    if (frozen_set == 'membership' and remesh_mode == 'adaptive'
            and not skip_triangulation):
        # Refuse before the remap's first stage touches anything.
        raise ValueError(_MEMBERSHIP_ADAPTIVE_MSG)
    remap_active = (retopo_remap == 'conservative'
                    and not skip_triangulation and mps is not None)
    if remap_active and not redistribute_mass:
        raise ValueError(
            "retopo_remap='conservative' requires redistribute_mass=True"
        )
    if not (isinstance(projection_every, int) and projection_every >= 1):
        raise ValueError(
            f"projection_every must be an int >= 1, got {projection_every!r}"
        )
    project_now = True
    if projection_every > 1:
        if mps is None or not redistribute_mass:
            raise ValueError(
                "projection_every > 1 requires mps and "
                "redistribute_mass=True"
            )
        if not (skip_triangulation or remap_active):
            raise ValueError(
                "projection_every > 1 under active Delaunay reconnection "
                "requires retopo_remap='conservative': skipping the "
                "redistribution while connectivity reconnects re-opens "
                "the per-rewire KE pump (lane-5 measured mechanism)"
            )
        _idx = getattr(mps, '_projection_call_idx', 0)
        project_now = (_idx % projection_every == 0)
        mps._projection_call_idx = _idx + 1

    # Snapshot per-phase pressure AND sub-volume before topology change.
    # The pre-retopo dual_vol_phase is needed by
    # redistribute_mass_multiphase to gate phase-presence at *v* —
    # otherwise a phase at reference pressure P0=0 looks identical to
    # an absent phase and is silently skipped.
    _p_snap = None
    if redistribute_mass and mps is not None and (project_now
                                                  or remap_active):
        from ddgclib.operators.mass_redistribution import (
            snapshot_geometry_multiphase,
        )
        _p_snap = snapshot_geometry_multiphase(HC, mps.n_phases)

    _vol_mid = None
    if remap_active:
        from ddgclib.operators.mass_redistribution import (
            evolve_snapshot_local_strain,
            phase_volume_totals,
        )
        # Stage 1 — measurement pass on the OLD connectivity at the
        # CURRENT (frozen) positions: refresh duals + per-phase split
        # without touching masses or pressures, and record the total
        # per-phase volumes.  Together with the same totals measured
        # after the rebuild, this isolates the pure connectivity
        # measurement artifact ratio (no physics can hide in it —
        # positions do not move inside this call), which the level
        # anchor below folds into the per-phase volume targets.
        _retopologize(HC, bV, dim, boundary_filter=boundary_filter,
                      backend=backend, skip_triangulation=True,
                      frozen_set=frozen_set)
        mps.refresh(HC, dim, reset_mass=False, split_method=split_method)
        _vol_mid = phase_volume_totals(HC, mps.n_phases)
        if not project_now:
            # NOTE(laneH): off-cadence remap call — advance the
            # pre-call snapshot by this step's LOCAL Lagrangian strain
            # (mass-conserving compression of each parcel from its
            # pre-call sub-volume to the refreshed one).
            # Redistribution/restore below then reproduce THIS field
            # across the rebuild: the connectivity artifact is still
            # cancelled exactly, but the step's local EOS compression
            # response survives instead of being erased.  Do NOT use
            # the raw eos(m/dual_vol) recompute here — it loses the
            # restore/anchor level corrections that live in p_phase
            # but not in the mass ledger (see
            # evolve_snapshot_local_strain).
            _p_snap = evolve_snapshot_local_strain(HC, mps, _p_snap)

    # Retopologization (Delaunay or adaptive + duals, no single-phase redistrib)
    _retopologize(HC, bV, dim, boundary_filter=boundary_filter,
                  merge_cdist=merge_cdist, backend=backend,
                  skip_triangulation=skip_triangulation,
                  remesh_mode=remesh_mode,
                  remesh_kwargs=remesh_kwargs,
                  frozen_set=frozen_set)

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
            _redist_diag = redistribute_mass_multiphase(
                HC, dim, mps, bV=bV, pressure_snapshot=_p_snap,
            )
            if remap_active:
                # Stage 2 closure, part 1 — volume-gauge update: the
                # redistribution scale factor is exactly the ratio of
                # conserved phase mass to the phase volume measured on
                # the NEW connectivity at the (frozen) stage-1
                # positions and pressures, i.e. the pure connectivity
                # measurement artifact of this rebuild (times the
                # previous gauge).  Storing it as the per-phase EOS
                # volume gauge makes the mass ledger and the pressure
                # field self-consistent, so the NEXT redistribution
                # does not bounce the artifact back in as a uniform
                # pressure offset (K*(scale-1) jolt).
                for _rec in _redist_diag['per_phase_diagnostics']:
                    mps.vol_corr[_rec['phase']] = _rec['scale_factor']
            # Recompute pressures from the redistributed masses so that
            # v.p_phase (read by multiphase_stress_force) reflects the
            # adjusted densities, not the stale pre-redistribution values.
            mps.compute_phase_pressures(HC)
            if remap_active:
                # Stage 2 closure, part 2 — structure restore: the
                # pre-call pressure STRUCTURE must survive the rebuild
                # bit-exactly wherever phase presence persists (the
                # global rescale reproduces it only up to a uniform
                # per-phase offset).
                from ddgclib.operators.mass_redistribution import (
                    anchor_phase_pressure_levels,
                    phase_volume_totals,
                    restore_pressure_multiphase,
                )
                restore_pressure_multiphase(HC, mps, _p_snap)
                # Stage 2 closure, part 3 — level anchor: pin each
                # phase's pressure LEVEL to the volume strain relative
                # to the artifact-corrected per-phase volume targets
                # (p_ref pattern: rebuild targets after every retopo so
                # connectivity changes are not read as compression).
                # Without this the level is an integral of noisy
                # per-step scale factors and reconnection noise
                # rectifies into a runaway phase tension.
                _vol_new = phase_volume_totals(HC, mps.n_phases)
                anchor_phase_pressure_levels(
                    HC, mps, _vol_mid, _vol_new,
                )


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


def _density_diffusion(HC, verts, delta, pressure_model, dt, dim):
    """Gradient-corrected density diffusion step on the interior vertices
    (method axis ``density_diffusion``; needs an EOS for the sound speed)."""
    from ddgclib.operators.stabilisation import density_diffusion_step
    from ddgclib.operators.stress import _get_dual_vol
    for v in HC.V:                      # make sure every cell has a volume
        _get_dual_vol(v, HC, dim)
    c0 = float(pressure_model.sound_speed(pressure_model.rho0))
    return density_diffusion_step(HC, verts, delta, c0, dt, dim=dim)


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
          displacement_eps=None, density_diffusion=None,
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
                             remesh_kwargs=remesh_kwargs,
                             displacement_eps=displacement_eps)
        verts = _interior_verts(HC, bV)
        if density_diffusion:
            if not hasattr(pressure_model, 'sound_speed'):
                raise ValueError("density_diffusion needs an EquationOfState "
                                 "pressure_model (sound speed)")
            _density_diffusion(HC, verts, density_diffusion, pressure_model, dt, dim)

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
                     displacement_eps=None, density_diffusion=None,
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
    density_diffusion : float or None
        Coefficient ``delta`` of the gradient-corrected density diffusion
        (:func:`ddgclib.operators.stabilisation.density_diffusion_step`)
        applied to the interior vertices before every force evaluation.
        Requires an EOS *pressure_model* (sound speed).  Damps the
        checkerboard density mode of the centred pressure flux; method
        axis ``density_diffusion``.

    Returns
    -------
    float
        Final time.
    """
    if density_diffusion and not hasattr(pressure_model, 'sound_speed'):
        raise ValueError("density_diffusion needs an EquationOfState "
                         "pressure_model (sound speed)")
    t = 0.0
    for step in range(n_steps):
        _do_retopologize(HC, bV, dim, boundary_filter, retopologize_fn,
                             merge_cdist, periodic_axes=periodic_axes,
                             domain_bounds=domain_bounds, backend=backend,
                             skip_triangulation=skip_triangulation,
                             pressure_model=pressure_model,
                             redistribute_mass=redistribute_mass,
                             remesh_mode=remesh_mode,
                             remesh_kwargs=remesh_kwargs,
                             displacement_eps=displacement_eps)
        verts = _interior_verts(HC, bV)
        if density_diffusion:
            _density_diffusion(HC, verts, density_diffusion, pressure_model, dt, dim)

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
          displacement_eps=None,
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
                             remesh_kwargs=remesh_kwargs,
                             displacement_eps=displacement_eps)
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
                        displacement_eps=None,
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
                             remesh_kwargs=remesh_kwargs,
                             displacement_eps=displacement_eps)
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
                   displacement_eps=None,
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
                             remesh_kwargs=remesh_kwargs,
                             displacement_eps=displacement_eps)
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
