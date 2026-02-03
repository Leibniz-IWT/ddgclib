# src/_boundary_conditions.py
import numpy as np
from ddgclib.barycentric._duals import compute_vd#, triang_dual
from ._mass import set_mass_3d


def add_inlet_layer(HC, U_avg, rho, r, L, layer_thickness=0.15, cdist=1e-10):
    """Add a new layer of vertices upstream of the inlet with uniform velocity."""
    z_min = min(v.x_a[2] for v in HC.V)
    inlet_vertices = [v for v in HC.V if abs(v.x_a[2] - z_min) < 0.05]

    if not inlet_vertices:
        return 0

    # Create new layer slightly upstream
    new_vertices = []
    added_count = 0

    for v in inlet_vertices:
        x_new = v.x_a.copy()
        x_new[2] = z_min - layer_thickness

        # Create new vertex
        new_v = HC.V[tuple(x_new)]
        new_v.u = np.array([0.0, 0.0, U_avg])

        # Estimate mass from local dual area (approximate)
        area = 1.0  # placeholder; improve with v_star or d_area if available
        volume = area * layer_thickness
        new_v.m = rho * volume

        new_vertices.append(new_v)
        added_count += 1

    # Connect new layer to old inlet layer
    for new_v, old_v in zip(new_vertices, inlet_vertices):
        new_v.connect(old_v)

    # Merge and rebuild dual
    HC.V.merge_all(cdist)
    compute_vd(HC, cdist=cdist)

    set_mass_3d(HC, r, L, rho=rho)  # rebalance mass

    return added_count


def remove_outlet_vertices(HC, z_max, outlet_buffer=0.5, cdist=1e-10):
    """Remove vertices that have exited the domain."""
    z_exit = z_max + outlet_buffer
    to_remove = [v for v in HC.V if v.x_a[2] > z_exit]

    removed_count = 0
    for v in to_remove:
        HC.V.remove(v)
        removed_count += 1

    if removed_count > 0:
        HC.V.merge_all(cdist)
        compute_vd(HC, cdist=cdist)

    return removed_count


def enforce_no_slip_walls(HC):
    """Set velocity = 0 on all side_boundary vertices."""
    for v in HC.V:
        if getattr(v, 'side_boundary', False):
            v.u = np.zeros(3)


def compute_inlet_flow_rate(HC, inlet_z, dr_tol=0.08):
    """Approximate volumetric flow rate at inlet slice."""
    Q = 0.0
    for v in HC.V:
        if abs(v.x_a[2] - inlet_z) < dr_tol:
            area_contrib = 1.0  # TODO: replace with proper dual area projection
            Q += v.u[2] * area_contrib
    return Q