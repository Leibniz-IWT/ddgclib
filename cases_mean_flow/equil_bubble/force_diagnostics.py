"""
force_diagnostics.py
====================

Diagnostic: why does the bulk DEP / Maxwell-stress electrostatic force have
essentially no effect on the H2 bubble detachment volume in
``bubble_enrtl_electrostatic_toy.py``?

This script does NOT re-run the full e-NRTL pipeline.  It isolates the
*force balance* at detachment and plots every term on the same axes so the
orders of magnitude are unambiguous:

    F_buoy   = (rho_l - rho_g) g V          buoyancy           (drives pull-off)
    F_pin    = 2 pi r_cl sigma sin(theta)   capillary pinning  (resists)
    F_DEP    = pi eps_0 CM R_d^2 E_z^2       bulk DEP/Maxwell   (toy electro term)
    dF_lipp  = 2 pi r_cl sin(theta) * dsigma electrocapillary (Lippmann, via sigma)

Two electrostatic *mechanisms* are contrasted:
  (1) bulk DEP using the ohmic field  E_z = j / kappa_e   (tens-thousands V/m)
  (2) Lippmann electrocapillarity using the double-layer  ~ 1/2 C_dl dE^2

Conclusion the figure makes visible:
  * F_DEP with the bulk ohmic field is ~8-10 orders of magnitude below
    buoyancy across the whole industrial current-density range.
  * It would take E ~ 1e6 V/m for F_DEP ~ F_buoy; the ohmic bulk field only
    reaches ~1e2-1e4 V/m.  (The ~1e9 V/m double-layer field lives in a few nm
    at the contact line -- captured by Lippmann, NOT by a bulk DEP term.)
  * The Lippmann electrocapillary term reaches F_buoy-scale at ~0.5 V, i.e.
    it is the first-order electrostatic effect for detachment.

Run:  python force_diagnostics.py
Out:  fig/force_diagnostics.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

import bubble_enrtl_electrostatic_toy as toy
from electrocapillarity_lippmann import lippmann_sigma_elec, LippmannParams

_HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    theta = np.deg2rad(30.0)
    salt = toy.KOH
    m = 1.0
    sigma, _ = toy.butler_surface_tension(m, salt)
    kappa_e = 21.5          # S/m, 1 M KOH

    # --- baseline detachment geometry (j = 0) ---
    res = toy.detachment_volume(sigma, theta, j=0.0, kappa_e=kappa_e)
    V = res['V']
    r_cl = res['r_cl']
    R_d = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    F_buoy = res['F_buoy']
    F_pin = res['F_pin']
    CM = toy.clausius_mossotti(78.4)

    # field that would make F_DEP == F_buoy
    E_parity = np.sqrt(F_buoy / (np.pi * toy.EPS_0 * CM * R_d ** 2))

    # --- sweep current density (bulk ohmic field) ---
    j = np.logspace(0, 4, 200)                 # 1 .. 1e4 A/m^2
    E_ohmic = j / kappa_e
    F_DEP = toy.dep_force(R_d, 0.0) * 0.0 + \
        np.pi * toy.EPS_0 * CM * R_d ** 2 * E_ohmic ** 2

    # --- Lippmann electrocapillary term vs applied cell voltage ---
    p = LippmannParams()
    E_cell = np.linspace(0.0, -1.0, 200) + p.E_pzc   # so dE spans 0..~1 V
    dsigma = np.array([sigma - lippmann_sigma_elec(sigma, ec, p) for ec in E_cell])
    dF_lipp = 2.0 * np.pi * r_cl * np.sin(theta) * dsigma
    dE = np.abs(E_cell - p.E_pzc)

    # ------------------------------------------------------------------ plot
    from fig_style import apply_style
    apply_style()
    plt.rcParams.update({'figure.dpi': 110})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # --- Panel A: all forces vs current density (bulk DEP mechanism) ---
    axA.loglog(j, 1e9 * np.full_like(j, F_buoy), 'k-', label=f'$F_{{buoy}}$ = {1e9*F_buoy:.0f} nN')
    axA.loglog(j, 1e9 * np.full_like(j, F_pin), color='0.5', ls='-',
               label=f'$F_{{pin}}$ = {1e9*F_pin:.0f} nN')
    axA.loglog(j, 1e9 * F_DEP, 'C3-', label=r'$F_{DEP}=\pi\varepsilon_0\,CM\,R_d^2 E_z^2$')
    axA.axvline(1.0, color='green', ls=':', alpha=0.6, label='Raman $j$ (1-5)')
    axA.axvline(5.0, color='green', ls=':', alpha=0.6)
    axA.axvline(1000.0, color='orange', ls='--', alpha=0.6, label='Industrial $j=1000$')
    axA.set_xlabel(r'Current density $j$ (A m$^{-2}$)')
    axA.set_ylabel('Force (nN)')
    axA.set_title('(a) Bulk-DEP mechanism: $E_z=j/\\kappa_e$\n'
                  f'$F_{{DEP}}/F_{{buoy}}\\sim 10^{{{np.log10(F_DEP[-1]/F_buoy):.0f}}}$ even at $j=10^4$')
    axA.legend(fontsize=8, loc='lower right')

    # --- Panel B: F_DEP vs field, showing the parity field and the two field regimes ---
    E = np.logspace(1, 9, 300)
    F_DEP_E = np.pi * toy.EPS_0 * CM * R_d ** 2 * E ** 2
    axB.loglog(E, 1e9 * F_DEP_E, 'C3-', label='$F_{DEP}(E)$')
    axB.axhline(1e9 * F_buoy, color='k', ls='-', label=f'$F_{{buoy}}$')
    axB.axvline(E_parity, color='C3', ls='--', alpha=0.7,
                label=f'Parity $E\\approx${E_parity:.1e} V m$^{{-1}}$')
    axB.axvspan(1e1, 1e4, color='orange', alpha=0.15, label='Bulk ohmic $E=j/\\kappa$ ($10^1$-$10^4$)')
    axB.axvspan(1e8, 1e9, color='blue', alpha=0.12, label='Double-layer $E$ ($10^8$-$10^9$)')
    axB.set_xlabel(r'Electric field $E$ (V m$^{-1}$)')
    axB.set_ylabel('Force (nN)')
    axB.set_title('(b) Field needed for parity is $\\sim 10^6$ V m$^{-1}$\n'
                  'bulk ohmic field falls 4-5 decades short')
    axB.legend(fontsize=8, loc='upper left')

    # --- Panel C: Lippmann electrocapillary force vs voltage (the real first-order term) ---
    axC.semilogy(dE, 1e9 * dF_lipp, 'C0-', label=r'$\Delta F_{Lippmann}=2\pi r_{cl}\sin\theta\,\Delta\sigma$')
    axC.axhline(1e9 * F_buoy, color='k', ls='-', label='$F_{buoy}$')
    # mark where Lippmann reaches buoyancy
    axC.axhline(1e9 * F_DEP[np.argmin(np.abs(j - 1000.0))], color='C3', ls='--',
                label='$F_{DEP}$ at $j=1000$ (for scale)')
    axC.set_xlabel(r'$|E_{cell}-E_{pzc}|$ (V)')
    axC.set_ylabel('Force (nN)')
    axC.set_title('(c) Lippmann electrocapillarity is first-order:\n'
                  '$F_{buoy}$-scale by $\\sim$0.5 V')
    axC.legend(fontsize=8, loc='lower right')

    fig.suptitle('H$_2$ bubble detachment: electrostatic force budget '
                 f'(KOH, $m={m}$ mol kg$^{{-1}}$, $\\theta_{{gas}}=30^\\circ$)', y=1.02)
    fig.tight_layout()
    out = os.path.join(_HERE, 'fig', 'force_diagnostics.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    fig.savefig(out[:-4] + '.pdf', bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ console
    print('=== Detachment force budget (KOH, m=1, theta=30 deg) ===')
    print(f'  V_d        = {V:.3e} m^3   (R_d = {R_d*1e3:.3f} mm)')
    print(f'  F_buoy     = {F_buoy:.3e} N')
    print(f'  F_pin      = {F_pin:.3e} N')
    print(f'  F_DEP(j=1000)   = {F_DEP[np.argmin(np.abs(j-1000))]:.3e} N'
          f'  -> F_DEP/F_buoy = {F_DEP[np.argmin(np.abs(j-1000))]/F_buoy:.2e}')
    print(f'  F_DEP(j=1e4)    = {F_DEP[-1]:.3e} N'
          f'  -> F_DEP/F_buoy = {F_DEP[-1]/F_buoy:.2e}')
    print(f'  E for parity    = {E_parity:.3e} V/m'
          f'  (bulk ohmic only reaches ~{1e4/kappa_e:.0f} V/m at j=1e4)')
    print(f'  dF_Lippmann(0.5V) = {dF_lipp[np.argmin(np.abs(dE-0.5))]:.3e} N'
          f'  -> {dF_lipp[np.argmin(np.abs(dE-0.5))]/F_buoy:.2f} x F_buoy')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
