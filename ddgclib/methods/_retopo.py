"""Library versions of the retopology variants that used to live only in
case scripts.

Each function reproduces a case-local closure operation for operation, so
the pinned numbers of the case that introduced it are unchanged (proofs in
``ddgclib/tests/test_methods.py``).  They are selected through
``SolverMethods.connectivity`` and are kept here, outside
``dynamic_integrators/_integrators_dynamic.py``, so the core retopology
code is not touched.

``bare_dual_refresh``
    ``connectivity='dual_only_bare'``.  What ``static_droplet_2D.py``
    called ``_dual_only_retopo``: keep connectivity, retag the boundary
    from ``HC.boundary()``, recompute duals and dual volumes
    (``cache_dual_volumes``, boundary half-cells kept) and re-split the
    per-phase volumes.  It does NOT call ``mps.refresh``, does not
    redistribute mass and does not update EOS pressures, so ``p_phase``
    stays at its setup value (audit 2026-09-25 F11).

``retopologize_multiphase_periodic``
    ``connectivity='periodic'`` with ``phases='multi'``.  What
    ``shearing_plate_droplet/src/_setup.py`` built as
    ``_make_periodic_multiphase_retopo``: geometry snapshot, periodic ghost
    Delaunay, ``mps.refresh``, per-phase mass redistribution against the
    snapshot, EOS pressures.  No conservative remap and no projection
    cadence exist on this path.  ``remesh_mode`` / ``remesh_kwargs`` are
    accepted for the integrator's forwarding and ignored, exactly like the
    case closure.
"""
from __future__ import annotations

__all__ = ['bare_dual_refresh', 'retopologize_multiphase_periodic']


def bare_dual_refresh(HC, bV, dim, mps=None, **_kw):
    """Boundary retag + dual rebuild on frozen connectivity; nothing else.

    ``_kw`` swallows the ``remesh_mode``/``remesh_kwargs`` the integrator
    forwards to every callable retopology function.
    """
    from hyperct.ddg import compute_vd
    from ddgclib.operators.stress import cache_dual_volumes

    dV = HC.boundary()
    for v in HC.V:
        v.boundary = v in dV
    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)
    if mps is not None:
        mps.split_dual_volumes(HC, dim)
    bV.clear()
    bV.update(dV)


def retopologize_multiphase_periodic(HC, bV, dim, mps=None, periodic_axes=None,
                                     domain_bounds=None,
                                     split_method='neighbour_count',
                                     redistribute_mass=False,
                                     remesh_mode='delaunay',
                                     remesh_kwargs=None):
    """Periodic ghost-cell Delaunay + multiphase refresh (+ redistribution).

    Mirrors ``_retopologize_multiphase`` with :func:`retopologize_periodic`
    in place of the plain Delaunay step.  When *redistribute_mass* is True
    the pre-call per-phase ``dual_vol_phase`` is snapshotted and used as the
    gating mask in ``redistribute_mass_multiphase`` so per-phase pressure is
    preserved across reconnection.  *remesh_mode*/*remesh_kwargs* are
    ignored (periodic adaptive remesh is not implemented).
    """
    if periodic_axes is None or domain_bounds is None:
        raise ValueError("retopologize_multiphase_periodic needs periodic_axes "
                         "and domain_bounds")
    from ddgclib.geometry.periodic import retopologize_periodic

    _p_snap = None
    if redistribute_mass and mps is not None:
        from ddgclib.operators.mass_redistribution import (
            snapshot_geometry_multiphase,
        )
        _p_snap = snapshot_geometry_multiphase(HC, mps.n_phases)

    retopologize_periodic(
        HC, bV, dim,
        periodic_axes=list(periodic_axes),
        domain_bounds=[tuple(b) for b in domain_bounds],
    )
    if mps is not None:
        mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

        if redistribute_mass and _p_snap is not None:
            from ddgclib.operators.mass_redistribution import (
                redistribute_mass_multiphase,
            )
            redistribute_mass_multiphase(
                HC, dim, mps, bV=bV, pressure_snapshot=_p_snap,
            )
            mps.compute_phase_pressures(HC)
