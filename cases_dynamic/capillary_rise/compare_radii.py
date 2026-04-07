#!/usr/bin/env python3
"""Compare capillary rise across fluids and tube radii.

Generates reference plots overlaying:
  - Washburn ODE analytical solutions
  - Lunowa et al. (2022) fitted data from CSV

Usage
-----
    python cases_dynamic/capillary_rise/compare_radii.py
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.capillary_rise.src._params import FLUIDS, g, jurin_height, capillary_pressure
from cases_dynamic.capillary_rise.src._analytical import washburn_solve
from cases_dynamic.capillary_rise.src._data import load_sample_data, load_all_sample_data

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig')
os.makedirs(_FIG, exist_ok=True)

def _savefig(fig, bn):
    fig.savefig(os.path.join(_FIG, f'{bn}.pdf'), dpi=150)
    fig.savefig(os.path.join(_FIG, f'{bn}.png'), dpi=150)
    print(f"  -> fig/{bn}.pdf, fig/{bn}.png")


def plot_fluid_comparison(fluid_name: str, dim: int = 3):
    """Generate a multi-panel figure for one fluid, all radii.

    Each panel shows h(t) for one radius with:
    - Washburn ODE solution (red dashed)
    - Lunowa fitted data from CSV (green dots)
    - Jurin height (black dotted horizontal line)
    """
    fp = FLUIDS[fluid_name]
    radii = fp['radii_mm']
    n = len(radii)

    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), squeeze=False)
    axes = axes[0]

    for i, R_mm in enumerate(radii):
        ax = axes[i]
        r = R_mm * 1e-3

        h_j = jurin_height(r, fp['gamma'], fp['theta_s_deg'], fp['rho'], g, dim=dim)
        P_cap = capillary_pressure(r, fp['gamma'], fp['theta_s_deg'], dim=dim)

        # Determine time span from data or estimate
        try:
            ref = load_sample_data(fluid_name, R_mm)
            t_max = ref['t_s'][-1] * 1.2  # 20% beyond data
            ax.plot(ref['t_s'], ref['h_cm'], 'go', ms=4, alpha=0.6,
                    label='Lunowa data')
        except FileNotFoundError:
            t_max = 2.0  # default
            ref = None

        # Washburn analytical (3D by default, or 2D)
        t_wash, h_wash = washburn_solve(
            (0.0, t_max), 1e-6, r, fp['gamma'], fp['theta_s_deg'],
            fp['mu'], fp['rho'], g, dim=dim,
        )
        ax.plot(t_wash, h_wash * 100, 'r--', lw=1.5, label='Washburn')
        ax.axhline(h_j * 100, color='k', ls=':', lw=1, alpha=0.5,
                    label=f'Jurin = {h_j*100:.2f} cm')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Height [cm]')
        ax.set_title(f'R = {R_mm} mm\nh_j = {h_j*100:.2f} cm, ΔP = {P_cap:.0f} Pa')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Capillary Rise — {fluid_name.capitalize()} ({dim}D)', fontsize=14)
    fig.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Capillary Rise — Multi-Fluid Reference Comparison")
    print("=" * 60)

    # For each fluid, generate 3D comparison (cylindrical tube)
    for fluid_name in FLUIDS:
        print(f"\n--- {fluid_name.capitalize()} ---")
        fp = FLUIDS[fluid_name]

        for R_mm in fp['radii_mm']:
            r = R_mm * 1e-3
            h_j = jurin_height(r, fp['gamma'], fp['theta_s_deg'], fp['rho'], g, dim=dim)
            print(f"  R={R_mm} mm: h_jurin = {h_j*100:.3f} cm")

        # 3D reference plot
        fig3d = plot_fluid_comparison(fluid_name, dim=3)
        _savefig(fig3d, f'caprise_reference_{fluid_name}_3d')
        plt.close(fig3d)

        # 2D reference plot
        fig2d = plot_fluid_comparison(fluid_name, dim=2)
        _savefig(fig2d, f'caprise_reference_{fluid_name}_2d')
        plt.close(fig2d)

    # Summary table
    print("\n" + "=" * 60)
    print("Summary: Jurin Heights [cm]")
    print(f"{'Fluid':<12} {'R [mm]':<10} {'h_jurin 3D [cm]':<18} {'h_jurin 2D [cm]':<18}")
    print("-" * 58)
    for fluid_name, fp in FLUIDS.items():
        for R_mm in fp['radii_mm']:
            r = R_mm * 1e-3
            h3 = jurin_height(r, fp['gamma'], fp['theta_s_deg'], fp['rho'], g, dim=3)
            h2 = jurin_height(r, fp['gamma'], fp['theta_s_deg'], fp['rho'], g, dim=2)
            print(f"{fluid_name:<12} {R_mm:<10.3f} {h3*100:<18.3f} {h2*100:<18.3f}")

    print(f"\nPlots saved to {_FIG}/")
    print("Done.")


if __name__ == '__main__':
    main()
