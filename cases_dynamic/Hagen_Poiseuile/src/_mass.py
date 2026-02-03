# src/_mass.py
import numpy as np


def set_mass_3d(HC, r, L, rho=1.0, approximate=True):
    """
    Assign mass to all vertices in 3D tube.
    - approximate=True: simple uniform volume share (fast, good enough for start)
    - approximate=False: future extension for proper dual volume (requires v_star)
    """
    if len(HC.V.cache) == 0:
        return

    if approximate:
        # Simple uniform volume distribution
        tube_volume = np.pi * r ** 2 * L  # total physical volume
        vol_per_vertex = tube_volume / len(HC.V.cache)
        for v in HC.V:
            v.m = rho * vol_per_vertex
    else:
        # TODO: proper dual volume using v_star or compute_vd volumes
        # For now fallback to uniform
        tube_volume = np.pi * r ** 2 * L
        vol_per_vertex = tube_volume / len(HC.V.cache)
        for v in HC.V:
            v.m = rho * vol_per_vertex

    print(f"Mass assigned: {len(HC.V.cache)} vertices, "
          f"total mass = {sum(v.m for v in HC.V):.6f} kg")


def update_mass_after_topology_change(HC, rho=1.0):
    """Recompute mass after add/remove vertices (call after merge_all)"""
    set_mass_3d(HC, rho=rho, approximate=True)  # uniform re-distribution