"""
decomposition.py
================

Answers the critical scientific question:

    Is bubble pull-off driven by an electrostatic FORCE, or by the bubble
    geometry set by the two INTERFACIAL quantities -- the liquid-vapour surface
    tension sigma_lv and the three-phase contact angle theta?

Corrected picture (see manuscript/CORRECTIONS.md)
-------------------------------------------------
Detachment volume V_d is governed by the Young-Laplace shape, which depends on
BOTH interfacial quantities:

    * sigma_lv  -- set by COMPOSITION (mixture thermodynamics; e-NRTL Butler).
                   Lever: V_d(sigma) master curve.
    * theta     -- set by the applied POTENTIAL via ELECTROWETTING
                   (Young-Lippmann on the contact angle, NOT on sigma_lv).
                   Lever: V_d(theta) master curve.

A third, genuinely-electrostatic channel (bulk DEP FORCE added to the balance)
is NOT an interfacial coordinate and does not move V_d at all.

So the field enters through theta (electrowetting), composition through sigma;
they are TWO SEPARATE interfacial levers -- not the same curve.  Both are large;
the direct bulk electrostatic force is negligible.  => pull-off is
interfacial-geometry-governed, not bulk-force-governed, which is exactly why the
mixture-thermodynamics model (setting sigma) and the electrowetting model
(setting theta) are both required.

Run:  python decomposition.py
Out:  fig/detachment_decomposition.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import (
    EnvState, Buoyancy, CapillaryPinning, BulkDEP, Electrowetting,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _Vd_sigma(sigma, theta=np.deg2rad(30.0)):
    r = toy.detachment_volume(sigma, theta)
    return r['V'] if r['converged'] else np.nan


def _Vd_field(sig0, dE, p, theta0=np.deg2rad(30.0)):
    """Electrowetting (THETA) field lever at fixed sigma0."""
    env = EnvState(C_dl=p['C_dl'], E_pzc=p['E_pzc'], E_cell=p['E_pzc'] - dE)
    r = toy.detachment_volume(sig0, theta0, models=[
        Buoyancy(), CapillaryPinning(), BulkDEP(), Electrowetting()], env=env)
    return (r['V'], r['theta_eff']) if r['converged'] else (np.nan, np.nan)


def main() -> None:
    salt = toy.KOH
    theta0 = np.deg2rad(30.0)
    p = toy.ELECTROCHEM_PARAMS['KOH']
    sig0 = toy.butler_surface_tension(1.0, salt)[0]
    Vref = _Vd_sigma(sig0, theta0)

    # --- composition lever: e-NRTL sigma(m) ---
    m_grid = np.linspace(0.01, 10.0, 25)
    sig_comp = np.array([toy.butler_surface_tension(m, salt)[0] for m in m_grid])
    Vd_comp = np.array([_Vd_sigma(s, theta0) for s in sig_comp])

    # --- field lever: electrowetting sigma fixed, theta(V) ---
    dE_grid = np.linspace(0.0, 0.40, 25)
    Vd_field, th_field = np.array([_Vd_field(sig0, d, p, theta0) for d in dE_grid]).T

    # --- direct bulk DEP force lever (fixed sigma, theta), swept over j ---
    j_grid = np.logspace(0, 4, 20)
    Vd_dep = []
    for j in j_grid:
        r = toy.detachment_volume(sig0, theta0,
                                  models=[Buoyancy(), CapillaryPinning(), BulkDEP()],
                                  env=EnvState(j=j, kappa_e=p['kappa_e']))
        Vd_dep.append(r['V'] if r['converged'] else np.nan)
    Vd_dep = np.array(Vd_dep)

    # --- theta master curve for panel (b) ---
    th_grid = np.deg2rad(np.linspace(10, 90, 40))
    Vd_th = np.array([_Vd_sigma(sig0, th) for th in th_grid])

    from fig_style import apply_style
    apply_style()
    plt.rcParams.update({'figure.dpi': 110})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # -- Panel (a): which driver moves V_d? --
    axA.plot(m_grid, 100 * (Vd_comp - Vref) / Vref, 'o-', color='C2',
             label='Composition: e-NRTL $\\sigma(m)$')
    axA.axhline(0, color='k', lw=1)
    axA.set_xlabel(r'Composition $m$ (mol kg$^{-1}$)', color='C2')
    axA.tick_params(axis='x', labelcolor='C2')
    axA.set_ylabel(r'$\Delta V_d / V_d^{\,ref}$ (%)')
    axT = axA.twiny()
    axT.plot(dE_grid, 100 * (Vd_field - Vref) / Vref, 's-', color='C0',
             label='Field: electrowetting $\\theta(V)$')
    axT.plot(np.linspace(0, 0.40, len(Vd_dep)),
             100 * (Vd_dep - Vref) / Vref, '--', color='C3', lw=2,
             label='Direct DEP force ($j=1..10^4$): $\\equiv$ 0')
    axT.set_xlabel(r'Field drive $|E_{cell}-E_{pzc}|$ (V)', color='C0')
    axT.tick_params(axis='x', labelcolor='C0')
    lines = axA.get_lines()[:1] + axT.get_lines()
    axA.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='lower left')
    axA.set_title('(a) Two large interfacial levers, one negligible force:\n'
                  'composition (+29%), field ($-$99%), DEP ($\\equiv$0)')

    # -- Panel (b): detachment is set by the interfacial PAIR (sigma, theta) --
    axB.plot(1e3 * sig_comp, 1e9 * Vd_comp, 'o-', color='C2',
             label='$V_d(\\sigma)$: composition lever')
    axB.set_xlabel(r'Surface tension $\sigma_{lv}$ (mN m$^{-1}$)', color='C2')
    axB.tick_params(axis='x', labelcolor='C2')
    axB.set_ylabel(r'Detachment volume $V_d$ (mm$^3$)')
    axB2 = axB.twiny()
    axB2.plot(np.rad2deg(th_grid), 1e9 * Vd_th, 's-', color='C0', ms=3,
              label='$V_d(\\theta)$: field / electrowetting lever')
    axB2.plot(np.rad2deg(th_field), 1e9 * Vd_field, 'x', color='C0', ms=7,
              label='Electrowetting operating points')
    axB2.set_xlabel(r'Contact angle $\theta_{gas}$ (deg)', color='C0')
    axB2.tick_params(axis='x', labelcolor='C0')
    lb = axB.get_lines() + axB2.get_lines()[:1]
    axB.legend(lb, [l.get_label() for l in lb], fontsize=8, loc='upper left')
    axB.set_title('(b) $V_d$ is governed by the interfacial pair $(\\sigma,\\theta)$;\n'
                  'the bulk force is not an interfacial coordinate')

    fig.suptitle('Pull-off is interfacial-geometry-governed, not force-governed '
                 '(KOH, $m=1$, $\\theta_0=30^\\circ$)', y=1.02, fontsize=12)
    fig.tight_layout()
    out = os.path.join(_HERE, 'fig', 'detachment_decomposition.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
    fig.savefig(out[:-4] + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print('reference (m=1): V_d = %.3e m^3' % Vref)
    print('composition sigma-lever  m=0.01..10 : dV_d = %+.1f%% .. %+.1f%%'
          % (100 * (Vd_comp[0] - Vref) / Vref, 100 * (Vd_comp[-1] - Vref) / Vref))
    print('field theta-lever (electrowetting) dE=0..0.4V : dV_d = %+.1f%% .. %+.1f%%'
          % (100 * (Vd_field[0] - Vref) / Vref, 100 * (Vd_field[-1] - Vref) / Vref))
    print('direct DEP force j=1..1e4 : dV_d span = %.2e%%'
          % (100 * (np.nanmax(Vd_dep) - np.nanmin(Vd_dep)) / Vref))
    print('Wrote %s' % out)


if __name__ == '__main__':
    main()
