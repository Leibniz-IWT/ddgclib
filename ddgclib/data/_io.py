"""Save and load simulation state to/from JSON files.

The format stores vertex coordinates, connectivity, field values (u, P, m),
boundary membership, and simulation time. It is designed to be human-readable
and compatible with hyperct's ``Complex`` reconstruction.

Usage
-----
    from ddgclib.data import save_state, load_state

    save_state(HC, bV, t=0.5, fields=['u', 'P', 'm'], path='state.json')
    HC2, bV2, meta = load_state('state.json')
"""

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from hyperct import Complex


def _vertex_key(v) -> str:
    """Canonical string key for a vertex (its coordinate tuple)."""
    return str(tuple(float(x) for x in v.x_a))


def save_state(
    HC,
    bV: set,
    t: float = 0.0,
    fields: Sequence[str] = ('u', 'P', 'm'),
    path: str = 'state.json',
    extra_meta: Optional[dict] = None,
) -> str:
    """Serialize simulation state to a JSON file.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    bV : set
        Boundary vertex set.
    t : float
        Current simulation time.
    fields : sequence of str
        Vertex attribute names to save. Supports scalars and arrays.
    path : str or Path
        Output file path.
    extra_meta : dict or None
        Additional metadata to store.

    Returns
    -------
    str
        The path written to (for chaining).
    """
    vertices = []
    bV_coords = set()

    for v in HC.V:
        coords = [float(x) for x in v.x_a]
        vdata = {'coords': coords}

        for f in fields:
            val = getattr(v, f, None)
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                vdata[f] = val.tolist()
            elif isinstance(val, (int, float, np.integer, np.floating)):
                vdata[f] = float(val)
            else:
                vdata[f] = val

        vertices.append(vdata)

        if v in bV:
            bV_coords.add(tuple(coords))

    # Connectivity: edges as pairs of coordinate tuples
    edges = []
    seen = set()
    for v in HC.V:
        for nb in v.nn:
            key = (tuple(v.x_a), tuple(nb.x_a))
            rkey = (tuple(nb.x_a), tuple(v.x_a))
            if key not in seen and rkey not in seen:
                edges.append([
                    [float(x) for x in v.x_a],
                    [float(x) for x in nb.x_a],
                ])
                seen.add(key)

    state = {
        'format': 'ddgclib_state_v1',
        'time': float(t),
        'dim': len(vertices[0]['coords']) if vertices else 0,
        'n_vertices': len(vertices),
        'fields': list(fields),
        'vertices': vertices,
        'edges': edges,
        'boundary_coords': [list(c) for c in bV_coords],
    }

    if extra_meta:
        state['meta'] = extra_meta

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)

    return str(path)


def load_state(
    path: str,
) -> tuple:
    """Load simulation state from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to state JSON file.

    Returns
    -------
    HC : Complex
        Reconstructed simplicial complex.
    bV : set
        Reconstructed boundary vertex set.
    meta : dict
        Metadata including 'time', 'dim', 'fields', and any 'meta' extras.
    """
    with open(path) as f:
        state = json.load(f)

    dim = state['dim']
    fields = state.get('fields', [])

    # Reconstruct domain bounds from vertex coords
    all_coords = [v['coords'] for v in state['vertices']]
    if not all_coords:
        HC = Complex(dim)
        return HC, set(), {'time': state.get('time', 0.0), 'dim': dim}

    coords_arr = np.array(all_coords)
    lb = coords_arr.min(axis=0)
    ub = coords_arr.max(axis=0)
    domain = [(float(lb[i]), float(ub[i])) for i in range(dim)]

    # Create Complex and add vertices
    HC = Complex(dim, domain=domain)

    # Add all vertices by accessing via their coordinate tuples
    vertex_map = {}
    for vdata in state['vertices']:
        coord_tuple = tuple(vdata['coords'])
        v = HC.V[coord_tuple]
        vertex_map[str(coord_tuple)] = v

        # Restore fields
        for f in fields:
            if f in vdata:
                val = vdata[f]
                if isinstance(val, list):
                    setattr(v, f, np.array(val))
                else:
                    setattr(v, f, val)

    # Restore connectivity
    for e in state.get('edges', []):
        c1, c2 = tuple(e[0]), tuple(e[1])
        v1 = HC.V[c1]
        v2 = HC.V[c2]
        v1.connect(v2)

    # Restore boundary set
    bV_coords = {tuple(c) for c in state.get('boundary_coords', [])}
    bV = set()
    for v in HC.V:
        if tuple(float(x) for x in v.x_a) in bV_coords:
            bV.add(v)

    meta = {
        'time': state.get('time', 0.0),
        'dim': dim,
        'fields': fields,
    }
    if 'meta' in state:
        meta.update(state['meta'])

    return HC, bV, meta
