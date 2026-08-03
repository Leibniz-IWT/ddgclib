"""Dissect F_p and F_st on a 2D static droplet interface vertex."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from ddgclib.operators.multiphase_stress import (
    multiphase_stress_force, _interface_surface_tension, _phases_present,
    _phase_pressure,
)
from ddgclib.operators.stress import (
    dual_area_vector, pressure_flux,
)
from ddgclib.operators.curvature_2d import (
    _select_curve_neighbours, _interface_neighbours,
)
from ddgclib.geometry._dual_split_2d import edge_phase_area_fractions


def main():
    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=2, R0=R0, epsilon=0.0, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=n_refine_outer,
            refinement_droplet=n_refine_droplet,
        )
    iface = sorted(
        [v for v in HC.V if getattr(v, 'is_interface', False)],
        key=lambda v: float(np.arctan2(v.x_a[1], v.x_a[0])),
    )
    print(f"Mesh: {sum(1 for _ in HC.V)} verts, {len(iface)} interface")

    delta_p = gamma / R0
    print(f"  R0={R0}, gamma={gamma}, gamma/R0={delta_p}")

    # Inspect first 3 interface vertices in polar order
    for v in iface[:3]:
        x = v.x_a[:2]
        theta = float(np.arctan2(x[1], x[0]))
        r = float(np.linalg.norm(x))
        print(f"\n--- Vertex theta={theta:+.4f} ({np.degrees(theta):+.2f} deg), r={r:.6f}")

        iface_nbs = _interface_neighbours(v)
        v_prev, v_next = _select_curve_neighbours(v, iface_nbs)
        print(f"  Interface 1-ring count: {len(iface_nbs)}")
        for nb in iface_nbs:
            d = np.linalg.norm(nb.x_a[:2] - x)
            ang = float(np.arctan2(nb.x_a[1], nb.x_a[0]))
            tag = ''
            if nb is v_prev: tag += ' [PREV]'
            if nb is v_next: tag += ' [NEXT]'
            print(f"    nb at theta={ang:+.4f} (chord={d:.4e}){tag}")

        # Total 1-ring breakdown
        n_iface_nb = 0
        n_inner_bulk = 0
        n_outer_bulk = 0
        for v_j in v.nn:
            if getattr(v_j, 'is_interface', False):
                n_iface_nb += 1
            elif v_j.phase == 1:
                n_inner_bulk += 1
            elif v_j.phase == 0:
                n_outer_bulk += 1
        print(f"  1-ring: {n_iface_nb} iface, {n_inner_bulk} inner-bulk, {n_outer_bulk} outer-bulk")

        # Compute v.dual_vol_phase
        dvp = getattr(v, 'dual_vol_phase', None)
        print(f"  dual_vol_phase = {tuple(float(x) for x in dvp) if dvp is not None else None}")
        print(f"  m_phase = {tuple(float(x) for x in v.m_phase)}")
        print(f"  p_phase = {tuple(float(x) for x in v.p_phase)}")

        # Stress integrals
        F_total = multiphase_stress_force(v, dim=2, mps=mps, HC=HC)
        F_st = _interface_surface_tension(v, 2, mps, HC=HC)
        F_p = F_total - F_st
        print(f"  |F_st|     = {np.linalg.norm(F_st):.6e}")
        print(f"  |F_p|      = {np.linalg.norm(F_p):.6e}")
        print(f"  |F_total|  = {np.linalg.norm(F_total):.6e}")

        # If F_st were exactly canceling F_p: ratio = -F_p/F_st
        if np.linalg.norm(F_st) > 0:
            ratio = -F_p / F_st  # element-wise; should be all 1's at equilibrium
            print(f"  -F_p/F_st (element)  = {ratio}")

        # Compute S_inner via integration; expected curvature = γ*S_inner/R should = -F_p/γ
        S_inner = np.zeros(2)
        for v_j in v.nn:
            A_ij = dual_area_vector(v, v_j, HC, 2)
            fractions = edge_phase_area_fractions(v, v_j, dim=2, interface=HC)
            for k, frac in fractions.items():
                if k == 1:
                    S_inner += frac * A_ij
        # Predicted F_p assuming uniform p_in inside: F_p = -p_in * S_inner = -delta_p * S_inner
        F_p_predicted = -delta_p * S_inner
        print(f"  F_p predicted (-delta_p * S_inner) = {F_p_predicted}")
        print(f"  F_p actual                          = {F_p}")
        print(f"  S_inner = {S_inner}, |S|={np.linalg.norm(S_inner):.4e}")

        # Predicted F_st_required = -F_p_predicted = delta_p * S_inner
        F_st_required = delta_p * S_inner
        print(f"  F_st REQUIRED (= delta_p * S_inner) = {F_st_required}")
        print(f"  F_st actual                          = {F_st}")
        print(f"  ratio |F_st|/|F_st_required|        = {np.linalg.norm(F_st)/np.linalg.norm(F_st_required):.4f}")


if __name__ == '__main__':
    main()
