"""
Geometric operations on a hyperct.Complex object

These operation can be used to build more complex test case domains and boundary conditions.

This folder is intended for methods that share a lot of generality while case specific
"""
# std library imports
import copy
import numpy as np
# Dependency imports
from hyperct import Complex

## Simpler operations
# Translation
def translate(HC, axis=0, d=0.0, copy_complex=True, jitter=0.0):
    """
    Translate all vertices of a Complex along one coordinate axis.

    IMPORTANT WARNING
    -----------------
    The vertex cache uses exact tuple coordinates as keys.
    If a translated vertex lands exactly on an existing vertex (common when
    translating by integer or 0.5 multiples on a cube-derived mesh), the cache
    will merge the two vertices, collapsing parts of the mesh.

    Recommended practices:
      - Use small incremental translations (e.g. d=0.1 repeated 10 times)
      - OR set jitter > 0 (e.g. jitter=1e-12) to break exact equality
      - Avoid exact multiples of the original grid spacing (0.5, 1.0, etc.)

    Parameters
    ----------
    HC : Complex
        The simplicial complex to translate.
    axis : int, default=0
        Axis to translate along (0=x, 1=y, 2=z).
    d : float, default=0.0
        Translation distance.
    copy : bool, default=True
        If True, return a deep copy; if False, modify in-place.
    jitter : float, default=0.0
        If > 0, add uniform random noise ∈ [-jitter/2, jitter/2] to each
        coordinate after translation. Recommended value: 1e-12 when large
        translations are needed.


    Examples
    --------
    HC_unit = unit_cylinder(r, refinements=1, height=1, up='z', distr_law='sinusoidal')
    for i in range(5):
        HC_unit_trans = translate_complex(HC_unit, axis=2, d=0.1, copy_complex=False)

    Returns
    -------
    Complex
        The translated complex.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    if copy_complex:
        HC = copy.deepcopy(HC)

    rng = np.random.default_rng()  # reproducible if seeded later if needed

    for v in list(HC.V):
        new_pos = v.x_a.copy()
        new_pos[axis] += d

        # Optional jitter to prevent exact cache collisions
        if jitter > 0:
            new_pos += rng.uniform(-jitter/2, jitter/2, size=3)

        HC.V.move(v, tuple(new_pos))

    return HC

# Extrusion
def extrude(HC_unit, L, axis=2, cdist=1e-10):
    """
    Extrude a unit-length simplicial complex (e.g. unit_cylinder) to total length L
    along the specified axis.

    - Uses ceil(L) segments → minimum one unit per integer length
    - Each segment is scaled to length L/n_segments
    - Manual vertex replication + topology copy (no deepcopy)
    - Final merge_all glues adjacent segments together

    Parameters
    ----------
    HC_unit : Complex
        Unit-length mesh (height ≈1 along the extrusion axis, centered at 0)
    L : float > 0
        Total extrusion length
    axis : int, default=2
        Extrusion axis (0=x, 1=y, 2=z)
    cdist : float, default=1e-10
        Tolerance for merge_all at segment interfaces

    Returns
    -------
    Complex
        Extruded mesh of exact length L
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    n_segments = max(1, int(np.ceil(L)))
    seg_len = L / n_segments

    extruded = Complex(3, domain=None)   # fresh target complex

    for k in range(n_segments):
        offset = k * seg_len
        vertex_map = {}  # old_v → new_v for this segment

        # 1. Create transformed vertices
        for old_v in HC_unit.V:
            pos = np.array(old_v.x_a, dtype=float)
            pos[axis] = pos[axis] * seg_len + offset   # scale + shift
            new_v = extruded.V[tuple(pos)]             # insert into cache
            vertex_map[old_v] = new_v

        # 2. Copy connectivity
        for old_v, new_v in vertex_map.items():
            for old_nb in old_v.nn:
                new_v.connect(vertex_map[old_nb])

    # 3. Glue segment interfaces
    extruded.V.merge_all(cdist=cdist)

    return extruded


## Initiation
