"""Resolve the IMPLICIT method choices on a concrete mesh, and record a
run's full method description next to its results.

The operator layer still decides several things by dimension or by
whether a cache exists (see the ``reported`` axes in ``_axes.py``).
:func:`effective_methods` reads the attributes those decisions key on
and reports what will actually run, so a ``methods.json`` next to a
``score.json`` says both what was ASKED (``SolverMethods``) and what
the code DID.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

__all__ = ['effective_methods', 'record_methods', 'git_state']


def effective_methods(HC, dim: int, methods=None) -> dict[str, Any]:
    """Report the implicit (dimension / cache gated) choices for *HC*.

    Reads attributes only; never computes duals.  Call it AFTER the first
    retopology (or after setup + ``compute_vd``) to see the runtime state.
    Keys match the ``reported`` axes of :data:`ddgclib.methods.AXES` plus
    a few bookkeeping counts.
    """
    simplices = getattr(HC, '_simplices', None)
    has_simplices = bool(simplices)
    cache = getattr(HC, '_edge_area_cache', None)
    vd_method = getattr(HC, '_vd_method', None)
    periodic = getattr(HC, '_periodic_axes', None)
    barycentric = vd_method in (None, 'barycentric')

    if dim == 1:
        dual_volume = 'interval_1d'
    elif has_simplices and barycentric:
        dual_volume = 'simplex_exact'
    elif dim == 2:
        dual_volume = 'dual_cell_area_2d'
    else:
        dual_volume = 'fan_walk_3d'

    if dim == 3:
        edge_area = 'batch_e_star_cache' if cache is not None else 'p_ij_ring_3d'
    elif dim == 2:
        edge_area = 'min_image_2d' if periodic else 'shared_vd_2d'
    else:
        edge_area = 'interval_1d'

    connectivity = getattr(methods, 'connectivity', None)
    if connectivity == 'dual_only':
        boundary_rule = 'carried_bV'
    elif connectivity == 'dual_only_bare':
        boundary_rule = 'HC.boundary'
    elif connectivity == 'custom':
        boundary_rule = 'custom (not resolvable from the mesh)'
    elif has_simplices:
        boundary_rule = 'boundary_from_simplices'
    else:
        boundary_rule = 'HC.boundary'

    n_v = n_b = n_i = 0
    b_vols: list[float] = []
    for v in HC.V:
        n_v += 1
        is_b = bool(getattr(v, 'boundary', False))
        n_b += is_b
        n_i += bool(getattr(v, 'is_interface', False))
        if is_b:
            b_vols.append(float(getattr(v, 'dual_vol', 0.0) or 0.0))
    # Boundary dual-volume convention is read from the DATA, not inferred
    # from the cache: a custom retopology can zero the boundary cells and
    # then drop the edge-area cache (laneJ's p_ij arm).
    if not b_vols:
        boundary_dual_vol = 'n/a (no boundary-tagged vertices)'
    elif all(x == 0.0 for x in b_vols):
        boundary_dual_vol = 'zeroed'
    else:
        boundary_dual_vol = 'half_cell'

    return {
        'mesh': 'simplicial' if getattr(HC, '_SC', None) is not None
                else 'complex',
        'mesh_class': f"{type(HC).__module__}.{type(HC).__name__}",
        'dual_method': vd_method if vd_method is not None
                       else 'barycentric (compute_vd not yet called)',
        'dual_path': 'simplex_aware' if has_simplices else 'nn_walk',
        'dual_volume': dual_volume,
        'boundary_dual_vol': boundary_dual_vol,
        'edge_area_source': edge_area,
        'boundary_rule': boundary_rule,
        'edge_area_cache_present': cache is not None,
        'simplex_cache_size': len(simplices) if has_simplices else 0,
        'interface_apex_cache_present':
            getattr(HC, '_interface_edge_to_apex', None) is not None,
        'n_vertices': n_v,
        'n_boundary_tagged': n_b,
        'n_interface': n_i,
    }


def git_state(path: str | Path) -> dict[str, Any]:
    """Short SHA + dirty flag of the git repo containing *path* (best effort)."""
    try:
        p = Path(path).resolve()
        root = subprocess.run(
            ['git', '-C', str(p if p.is_dir() else p.parent),
             'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        sha = subprocess.run(
            ['git', '-C', root, 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', '-C', root, 'status', '--porcelain',
             '--untracked-files=no'],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip() != ''
        return {'root': root, 'sha': sha, 'dirty': dirty}
    except Exception as e:  # noqa: BLE001 - reporting only
        return {'error': f"{type(e).__name__}: {e}"}


def record_methods(path: str | Path, methods, HC=None, dim: int | None = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write ``methods.json``: the requested config, the effective
    (implicit) choices on *HC*, git state of ddgclib and hyperct, and any
    *extra* run metadata (dt, n_steps, refinement, ...).  Returns the dict.
    """
    import ddgclib
    doc: dict[str, Any] = {
        'schema': 'ddgclib.methods/1',
        'config': methods.to_dict(),
        'status': {name: methods.status_of(name)
                   for name, _ in methods.explicit_items()},
    }
    if HC is not None:
        doc['effective'] = effective_methods(
            HC, dim if dim is not None else methods.dim, methods)
    doc['git'] = {'ddgclib': git_state(Path(ddgclib.__file__).parent)}
    try:
        import hyperct
        doc['git']['hyperct'] = git_state(Path(hyperct.__file__).resolve().parent)
    except Exception as e:  # noqa: BLE001
        doc['git']['hyperct'] = {'error': str(e)}
    if extra:
        doc['extra'] = {k: _jsonable(v) for k, v in extra.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=_jsonable) + '\n')
    return doc


def _jsonable(v: Any) -> Any:
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, (set, frozenset, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)
