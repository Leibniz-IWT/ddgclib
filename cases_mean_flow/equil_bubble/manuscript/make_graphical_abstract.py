"""
make_graphical_abstract.py
==========================

Graphical abstract for the water-electrolysis single-bubble manuscript.

Wide landscape banner (>= 531 x 1328 px, height x width; readable at 5 x 13 cm)
composed of two panels:

  LEFT   surface tension sigma_lv(m) for KOH from the Butler equation with
         e-NRTL activities (data-fitted tau), showing the systematic rise
         above the constant sigma = 72 mN/m assumption.
  RIGHT  axisymmetric single-bubble detachment shapes (Young-Laplace) at
         low / medium / high KOH molality, so a reader instantly sees that
         this is a single-bubble study and that composition changes the shape.

Reuses:  butler_surface_tension, young_laplace_shape (via detachment_volume)
         from bubble_enrtl_electrostatic_toy, and fitted_salts() from
         make_water_figure_sets (data-fitted e-NRTL tau).

Run (from cases_mean_flow/equil_bubble):
    /home/endres/anaconda3/envs/ddg/bin/python manuscript/make_graphical_abstract.py

Outputs:
    manuscript/tex/figures/graphical_abstract.pdf   (vector, for the manuscript)
    manuscript/tex/figures/graphical_abstract.png   (>= 1328 x 531 px, preview)
"""

from __future__ import annotations
import os
import sys

# The reusable toy modules live in the case root (parent of manuscript/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CASE_ROOT = os.path.dirname(_HERE)
if _CASE_ROOT not in sys.path:
    sys.path.insert(0, _CASE_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import bubble_enrtl_electrostatic_toy as toy
from bubble_enrtl_electrostatic_toy import (
    butler_surface_tension,
    detachment_volume,
    SIGMA_W,
)
from make_water_figure_sets import fitted_salts

_FIG_DIR = os.path.join(_HERE, "tex", "figures")

# Contact angle through the gas (matches Figure 4 detachment-shape logic).
THETA_GAS = np.deg2rad(30.0)
# KOH molality grid (mol per kg water); alkaline electrolysers run up to ~10-12 M.
M_GRID = np.linspace(0.01, 10.0, 40)


def _koh_salt():
    """Data-fitted KOH Salt object (Butler-eNRTL tau fitted to Hamer & Wu 1972)."""
    salts, _info = fitted_salts()
    for s in salts:
        if s.name == "KOH":
            return s
    raise RuntimeError("KOH not returned by fitted_salts()")


def _sigma_curve(salt, m_grid):
    """sigma(m) in N/m, with the same monotone salting-out guard used in demo()."""
    sigma = np.array([butler_surface_tension(mi, salt)[0] for mi in m_grid])
    if m_grid.size > 1 and np.all(np.diff(m_grid) > 0):
        sigma = np.maximum.accumulate(sigma)
    return sigma


def _style():
    from fig_style import apply_style
    apply_style()
    # Small-banner overrides (the graphical abstract is a compact 13 x 5.8 cm
    # figure, so it runs at reduced font weights on top of the shared style).
    plt.rcParams.update({
        "axes.linewidth": 0.8,
        "font.size": 9,
    })


def make(save_dir: str = _FIG_DIR) -> tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    _style()

    salt = _koh_salt()
    sigma = _sigma_curve(salt, M_GRID)

    # Three representative molalities: low / medium / high.
    idx_show = [0, len(M_GRID) // 2, len(M_GRID) - 1]
    m_show = [M_GRID[i] for i in idx_show]
    sig_show = [sigma[i] for i in idx_show]

    # 13 cm x 5.8 cm banner at 300 dpi -> ~1535 x 685 px (>= 1328 x 531).
    fig = plt.figure(figsize=(13.0 / 2.54, 5.8 / 2.54))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0],
                          left=0.085, right=0.985, bottom=0.175, top=0.755,
                          wspace=0.30)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    labels = ["Low", "Medium", "High"]
    shade = [cm.viridis(t) for t in (0.15, 0.5, 0.85)]

    # ---- LEFT: sigma_lv(m) trend --------------------------------------------
    axL.plot(M_GRID, 1e3 * sigma, color="#1f4e8c", lw=2.0, zorder=3,
             label=r"KOH (Butler eq., e-NRTL)")
    axL.axhline(1e3 * SIGMA_W, color="0.35", ls=":", lw=1.4, zorder=2,
                label=r"Constant $\sigma_{lv}=72$ mN m$^{-1}$")
    # Mark the three molalities used on the right panel.
    for mi, si, c in zip(m_show, sig_show, shade):
        axL.plot(mi, 1e3 * si, "o", color=c, ms=5.5, zorder=4,
                 markeredgecolor="k", markeredgewidth=0.5)
    axL.set_ylim(70.5, 91)
    axL.set_xlim(0, 10)
    # Annotate the rise above the constant assumption (in the empty lower-right).
    axL.annotate("Composition raises\n" r"$\sigma_{lv}$ above 72 mN m$^{-1}$",
                 xy=(8.0, 1e3 * np.interp(8.0, M_GRID, sigma)),
                 xytext=(4.3, 73.2),
                 fontsize=6.8, color="#1f4e8c", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color="#1f4e8c", lw=1.0))
    axL.set_xlabel(r"Bulk molality $m$ (mol kg$^{-1}$)", fontsize=8.5)
    axL.set_ylabel(r"Surface tension $\sigma_{lv}$ (mN m$^{-1}$)", fontsize=8.5)
    axL.set_title("Composition-dependent surface tension", fontsize=8.3, pad=4)
    axL.legend(loc="upper left", fontsize=6.6, frameon=False,
               handlelength=1.6, borderpad=0.2)
    axL.tick_params(labelsize=7.5)

    # ---- RIGHT: single-bubble detachment shapes -----------------------------
    for mi, si, c, lab in zip(m_show, sig_show, shade, labels):
        res = detachment_volume(si, THETA_GAS, criterion="max_volume")
        if not res["converged"]:
            continue
        shape = res["shape"]
        r = 1e3 * shape["r"]
        z = 1e3 * shape["z"]
        axR.plot(r, z, color=c, lw=1.8, label=f"{lab}: {mi:.1f}")
        axR.plot(-r, z, color=c, lw=1.8)
    axR.axhline(0.0, color="0.5", lw=1.0)
    axR.set_aspect("equal")
    axR.set_ylim(-0.1, 2.65)
    axR.set_xlim(-1.75, 1.75)
    axR.set_xlabel(r"$r$ (mm)", fontsize=8.5)
    axR.set_ylabel(r"$z$ (mm)", fontsize=8.5)
    axR.set_title(r"Detachment shapes ($\theta_{\mathrm{gas}}=30^\circ$)",
                  fontsize=8.3, pad=4)
    axR.legend(title=r"$m$ (mol kg$^{-1}$)", ncol=3, loc="upper center",
               fontsize=6.2, title_fontsize=6.6, frameon=False,
               handlelength=1.2, columnspacing=1.0, borderpad=0.2,
               handletextpad=0.4)
    axR.tick_params(labelsize=7.5)

    fig.suptitle("Single-bubble water electrolysis with "
                 r"composition-dependent surface tension",
                 fontsize=8.5, y=0.965)

    pdf_path = os.path.join(save_dir, "graphical_abstract.pdf")
    png_path = os.path.join(save_dir, "graphical_abstract.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


if __name__ == "__main__":
    pdf_path, png_path = make()
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
