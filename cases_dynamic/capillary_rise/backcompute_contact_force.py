#!/usr/bin/env python3
"""Back-compute the three-phase contact force from experimental rise data.

For every fluid/radius pair in the Lunowa et al. (2022) dataset this
script runs both directions of the data-driven closure (no contact-angle
model anywhere):

1. FORWARD:  integrate the reduced tube momentum balance driven by the
   *measured* dynamic contact angle CA(t) and compare the predicted
   rise h(t) with the measured rise (plus the static-angle Washburn
   solution, to show what the dynamic contact angle buys).

2. INVERSE:  from the measured h(t), back-compute the driving pressure
   the momentum balance requires,

       P_cap_impl = rho (h h'' + h'^2) + rho g h + 8 mu h h' / R^2,

   convert to an implied contact angle, and compare against the
   measured CA(t).  Agreement means the measured rise and measured
   contact angle are mutually consistent under Poiseuille drag —
   i.e. the "rest of the experiment" is captured without any contact
   force model.

Usage
-----
    python cases_dynamic/capillary_rise/backcompute_contact_force.py
"""
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.capillary_rise.src._data import _FLUID_RADII
from cases_dynamic.capillary_rise.src._dynamic_ca import (
    ExperimentalDrive, rise_ode_solve, backcompute_pcap_from_h,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
_RESULTS = os.path.join(_HERE, 'results')
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RESULTS, exist_ok=True)


def _l2(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)) / max(np.nanmax(b) - np.nanmin(b), 1e-12))


def main():
    summary = {}
    for fluid, radii in _FLUID_RADII.items():
        fig_h, axes_h = plt.subplots(1, len(radii), figsize=(4.2 * len(radii), 3.6),
                                     sharey=False)
        fig_ca, axes_ca = plt.subplots(1, len(radii), figsize=(4.2 * len(radii), 3.6))
        for k, R_mm in enumerate(radii):
            drive = ExperimentalDrive(fluid, R_mm)
            # start slightly inside the data range (h=0 is singular)
            t0 = drive.t_data[np.searchsorted(drive.h_data, 0.05 * drive.h_data[-1])]
            t0 = float(max(t0, drive.t_min + 1e-6))
            t_end = drive.t_max

            t_dyn, h_dyn, _ = rise_ode_solve(drive, t0, t_end, theta_mode='dynamic')
            t_sta, h_sta, _ = rise_ode_solve(drive, t0, t_end, theta_mode='static')
            inv = backcompute_pcap_from_h(drive)

            h_exp_dyn = drive.h_exp(t_dyn)
            l2_dyn = _l2(h_dyn, h_exp_dyn)
            l2_sta = _l2(h_sta, drive.h_exp(t_sta))
            ca_l2 = _l2(inv['theta_impl_deg'], inv['theta_meas_deg'])
            summary[f'{fluid}_R{R_mm:g}mm'] = {
                't0': t0, 'l2_h_dynCA': l2_dyn, 'l2_h_staticCA': l2_sta,
                'l2_theta_impl_vs_meas': ca_l2,
            }

            ax = axes_h[k] if len(radii) > 1 else axes_h
            ax.plot(drive.t_data, drive.h_data * 100, 'ko', ms=3, mfc='none',
                    label='experiment')
            ax.plot(t_dyn, h_dyn * 100, 'g-', lw=1.6,
                    label=f'ODE, exp CA(t)  (L2={l2_dyn:.3f})')
            ax.plot(t_sta, h_sta * 100, 'r--', lw=1.3,
                    label=f'ODE, static $\\theta_s$  (L2={l2_sta:.3f})')
            ax.set_xscale('log')
            ax.set_xlabel('t [s]'); ax.set_ylabel('h [cm]')
            ax.set_title(f'{fluid}, R={R_mm} mm')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

            ax = axes_ca[k] if len(radii) > 1 else axes_ca
            ax.plot(inv['t'], inv['theta_meas_deg'], 'k-', lw=1.4,
                    label='measured CA(t)')
            ax.plot(inv['t'], inv['theta_impl_deg'], 'b--', lw=1.4,
                    label='implied by h(t) (back-computed)')
            ax.axhline(drive.theta_s_deg, color='r', ls=':', lw=1,
                       label=r'$\theta_s$')
            ax.set_xscale('log')
            ax.set_xlabel('t [s]'); ax.set_ylabel('contact angle [deg]')
            ax.set_title(f'{fluid}, R={R_mm} mm (L2={ca_l2:.3f})')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        for fig, bn in ((fig_h, f'backcompute_h_{fluid}'),
                        (fig_ca, f'backcompute_theta_{fluid}')):
            fig.tight_layout()
            fig.savefig(os.path.join(_FIG, f'{bn}.png'), dpi=150)
            fig.savefig(os.path.join(_FIG, f'{bn}.pdf'))
            plt.close(fig)
            print(f"  -> fig/{bn}.png")

    out = os.path.join(_RESULTS, 'backcompute_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {out}")

    print("\n%-22s %10s %10s %12s" % ("case", "L2 h dyn", "L2 h stat", "L2 theta"))
    for k, v in summary.items():
        print("%-22s %10.4f %10.4f %12.4f"
              % (k, v['l2_h_dynCA'], v['l2_h_staticCA'], v['l2_theta_impl_vs_meas']))


if __name__ == '__main__':
    main()
