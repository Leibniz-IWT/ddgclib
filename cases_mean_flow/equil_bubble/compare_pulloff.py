"""
compare_pulloff.py
=================

Capstone comparison of the pluggable pull-off mechanisms in
``pulloff_models.py``, using the detachment solver in
``bubble_enrtl_electrostatic_toy.py``.

Demonstrates the central distinction of the abstraction:

  * FORCE-coupled electrostatics (bulk DEP, E_z = j/kappa_e) do NOT move the
    detachment volume -- the max-closable-volume criterion is geometric and the
    bulk field is ~9 orders too weak anyway.
  * TENSION-coupled electrostatics (Lippmann electrocapillarity) DO move it,
    because lowering sigma reshapes the whole bubble.

Two panels:
  (left)  detachment diameter D_d vs current density j  -> flat (DEP)
  (right) detachment diameter D_d vs |E_cell - E_pzc|    -> falls (Lippmann)

Run:  python compare_pulloff.py
Out:  fig/pulloff_comparison.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import (
    EnvState, Buoyancy, CapillaryPinning, BulkDEP, Lippmann,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    theta = np.deg2rad(30.0)
    salt = toy.KOH
    m = 1.0
    sigma0, _ = toy.butler_surface_tension(m, salt)
    kappa_e = 21.5

    # ---- FORCE mechanism: sweep current density (bulk DEP) ----
    j_sweep = np.logspace(0, 4, 40)
    D_dep, sig_dep = [], []
    for j in j_sweep:
        env = EnvState(j=j, kappa_e=kappa_e)
        r = toy.detachment_volume(sigma0, theta, models=[
            Buoyancy(), CapillaryPinning(), BulkDEP()], env=env)
        D_dep.append(r['D'] * 1e3 if r['converged'] else np.nan)
    D_dep = np.array(D_dep)

    # ---- TENSION mechanism: sweep applied voltage (Lippmann) ----
    dE = np.linspace(0.0, 0.6, 40)          # |E_cell - E_pzc|
    D_lip, sig_lip = [], []
    for d in dE:
        env = EnvState(E_cell=EnvState().E_pzc - d)   # cathodic
        r = toy.detachment_volume(sigma0, theta, models=[
            Buoyancy(), CapillaryPinning(), BulkDEP(), Lippmann()], env=env)
        D_lip.append(r['D'] * 1e3 if r['converged'] else np.nan)
        sig_lip.append(r['sigma_eff'] * 1e3 if r['converged'] else np.nan)
    D_lip = np.array(D_lip)
    sig_lip = np.array(sig_lip)

    # baseline
    D0 = toy.detachment_volume(sigma0, theta)['D'] * 1e3

    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3,
                         'figure.dpi': 110, 'lines.linewidth': 2.2})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    axL.semilogx(j_sweep, D_dep, 'C3-o', ms=3, label='FORCE: bulk DEP')
    axL.axhline(D0, color='k', ls='--', label=f'baseline  $D_d$={D0:.3f} mm')
    axL.axvline(1000.0, color='orange', ls=':', alpha=0.7, label='industrial $j$=1000')
    axL.set_xlabel(r'current density $j$  (A/m$^2$)')
    axL.set_ylabel(r'detachment diameter $D_d$  (mm)')
    axL.set_title('(a) FORCE-coupled (bulk DEP): $D_d$ unchanged')
    axL.set_ylim(D0 * 0.5, D0 * 1.1)
    axL.legend(fontsize=9, loc='lower left')

    axR.plot(dE, D_lip, 'C0-o', ms=3, label='TENSION: Lippmann')
    axR.axhline(D0, color='k', ls='--', label=f'baseline  $D_d$={D0:.3f} mm')
    axR.set_xlabel(r'$|E_{cell}-E_{pzc}|$  (V)')
    axR.set_ylabel(r'detachment diameter $D_d$  (mm)')
    axR.set_title('(b) TENSION-coupled (Lippmann): $D_d$ falls with voltage')
    axR.legend(fontsize=9, loc='lower left')
    # twin axis: show sigma_eff dropping
    axR2 = axR.twinx()
    axR2.plot(dE, sig_lip, 'C2:', lw=1.8, label=r'$\sigma_{eff}$')
    axR2.set_ylabel(r'$\sigma_{eff}$  (mN/m)', color='C2')
    axR2.tick_params(axis='y', labelcolor='C2')
    axR2.grid(False)

    fig.suptitle('Pull-off mechanism comparison  (KOH, m=1 mol/kg, '
                 r'$\theta_{gas}$=30$^\circ$):  FORCE vs TENSION coupling',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = os.path.join(_HERE, 'fig', 'pulloff_comparison.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'baseline D_d          = {D0:.4f} mm')
    print(f'D_d @ j=1e4 (DEP)     = {D_dep[-1]:.4f} mm  (change {100*(D_dep[-1]-D0)/D0:+.3e} %)')
    print(f'D_d @ dE=0.5V (Lipp)  = {D_lip[np.argmin(np.abs(dE-0.5))]:.4f} mm'
          f'  (change {100*(D_lip[np.argmin(np.abs(dE-0.5))]-D0)/D0:+.1f} %)')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
