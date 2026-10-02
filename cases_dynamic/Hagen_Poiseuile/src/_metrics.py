"""Measurements for the developing Lagrangian Poiseuille runs (laneH).

Every comparison is weighted by the dual volume: a cell contributes with
the volume it stands for, so a cluster of small cells cannot dominate.
"""
from __future__ import annotations

import numpy as np


def _slab(HC, axis: int, x0: float, x1: float) -> list:
    return [v for v in HC.V if x0 <= v.x_a[axis] <= x1]


def profile_error(HC, bV, params: dict, x0: float, x1: float) -> dict:
    """Velocity against the developed profile on ``x0 <= x <= x1``.

    ``l2`` is ``sqrt(sum_i V_i |u_i - u_ana(x_i)|^2 / sum_i V_i
    |u_ana(x_i)|^2)`` over the vertices that are not frozen, all velocity
    components included; ``u_max`` is the largest axial velocity,
    ``u_cross`` the largest transverse component.
    """
    axis = params['flow_axis']
    dim = params['dim']
    analytical = params['poiseuille_ic'].analytical_velocity
    num = den = 0.0
    u_max = u_cross = 0.0
    n = 0
    for v in _slab(HC, axis, x0, x1):
        vol = getattr(v, 'dual_vol', 0.0)
        if v in bV or not vol > 0.0:
            continue
        u_ana = np.zeros(dim)
        u_ana[axis] = analytical(v.x_a)
        du = v.u[:dim] - u_ana
        num += vol * float(du @ du)
        den += vol * u_ana[axis] ** 2
        u_max = max(u_max, float(v.u[axis]))
        u_cross = max(u_cross, float(np.linalg.norm(np.delete(du, axis))))
        n += 1
    return {'l2': float(np.sqrt(num / den)) if den > 0 else float('nan'),
            'u_max': u_max, 'u_cross': u_cross, 'n': n}


def fluxes(HC, params: dict, x0: float, x1: float) -> dict:
    """Volume and mass flux through the slab ``x0 <= x <= x1``: the
    integrals of ``u`` and of ``rho u`` over the slab divided by its
    length, relative to the inlet values ``U_avg A`` and ``rho U_avg A``
    (``A`` the cross-section of the mesh)."""
    axis = params['flow_axis']
    q = mdot = 0.0
    for v in _slab(HC, axis, x0, x1):
        q += getattr(v, 'dual_vol', 0.0) * float(v.u[axis])
        mdot += v.m * float(v.u[axis])
    ref = params['U_avg'] * params['area'] * (x1 - x0)
    return {'volume': q / ref, 'mass': mdot / (params['rho'] * ref)}


def census(HC, bV, params: dict) -> dict:
    """Where the vertices are: buffers, channel, walls, and how many left
    the cross-section."""
    axis = params['flow_axis']
    L = params['L']
    dim = params['dim']
    n = {'total': 0, 'inlet_buffer': 0, 'channel': 0, 'outlet_buffer': 0,
         'frozen': len(bV), 'outside': 0}
    for v in HC.V:
        n['total'] += 1
        if v in bV:
            continue
        x = v.x_a[axis]
        key = ('inlet_buffer' if x <= 0.0 else
               'channel' if x <= L else 'outlet_buffer')
        n[key] += 1
        if dim == 2:
            out = v.x_a[1] < -1e-9 or v.x_a[1] > params['D'] + 1e-9
        else:
            out = np.hypot(v.x_a[0], v.x_a[1]) > params['R'] + 1e-9
        n['outside'] += bool(out)
    return n


class PlaneFlux:
    """Mass carried through planes ``x = c`` of the flow axis by the
    vertices that cross them (call once per step, after the BCs).

    The Lagrangian mass flux: no cell volumes and no sampling of a slab
    are involved, so it is exact for the markers.  A vertex that is
    created or deleted between two calls is not counted.
    """

    def __init__(self, axis: int, planes: list[float]):
        self.axis = axis
        self.mass = {float(c): 0.0 for c in planes}
        self.count = {float(c): 0 for c in planes}
        self._last: dict = {}

    def update(self, HC) -> None:
        now = {id(v): (v, float(v.x_a[self.axis])) for v in HC.V}
        for vid, (v, x) in now.items():
            before = self._last.get(vid)
            if before is None or before[1] == x:
                continue
            x0 = before[1]
            for c in self.mass:
                if x0 <= c < x:
                    self.mass[c] += v.m
                    self.count[c] += 1
                elif x <= c < x0:
                    self.mass[c] -= v.m
                    self.count[c] -= 1
        self._last = now

    def reset(self) -> None:
        for c in self.mass:
            self.mass[c] = 0.0
            self.count[c] = 0
