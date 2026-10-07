"""Regenerate the manuscript VALIDATION figures as vector PDF.

Three figures, each written to ``fig/validation/`` (relative to the
``equil_bubble`` case root) as both a vector ``.pdf`` and a raster ``.png``:

  * ``mars_scaling``                      -- D vs g (Earth vs Mars) and D vs
                                             molality, computed live from the
                                             detachment model.
  * ``nucleation_german2018_amplification`` -- sigma^3 CNT barrier / rate
                                             amplification, from the numbers in
                                             ``nucleation_german2018.md``.
  * ``uncertainty_tornado``               -- one-at-a-time D_d sensitivity
                                             (two panels), from the numbers in
                                             ``uncertainty_tornado.md``.

Run:
    cd cases_mean_flow/equil_bubble
    /home/endres/anaconda3/envs/ddg/bin/python \
        manuscript/validation/make_validation_figures.py

The three source markdown notes carry the provenance for every literature
number reproduced here.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The model lives in the equil_bubble case root; make it importable regardless
# of the invocation directory.
_CASE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _CASE_ROOT not in sys.path:
    sys.path.insert(0, _CASE_ROOT)

import bubble_enrtl_electrostatic_toy as toy  # noqa: E402
from make_water_figure_sets import fitted_salts  # noqa: E402
from fig_style import apply_style  # noqa: E402

_FIG = os.path.join(_CASE_ROOT, "fig", "validation")

# Physical constants (mirror dilute_and_mars.md).
G_EARTH = 9.80665      # m/s^2
G_MARS = 3.72          # m/s^2
THETA_GAS = np.deg2rad(30.0)


def _save(fig, name: str) -> None:
    os.makedirs(_FIG, exist_ok=True)
    png = os.path.join(_FIG, name + ".png")
    fig.savefig(png, dpi=140, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(_FIG, name + ".pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {os.path.join(_FIG, name)}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 1 -- Mars detachment scaling (computed live from the model)
# ---------------------------------------------------------------------------
def make_mars_scaling(salts) -> None:
    koh = next(s for s in salts if s.name == "KOH")
    h2so4 = next(s for s in salts if s.name == "H2SO4")

    def D_of(sigma: float, g: float) -> float:
        return toy.detachment_volume(sigma, THETA_GAS, g=g,
                                     criterion="max_volume")["D"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # Panel (a): D vs g at fixed composition (KOH m = 6), model vs analytic
    # capillary-length scaling D ~ (sigma / (g * Delta_rho))^{1/2}.
    sigma_koh6, _ = toy.butler_surface_tension(6.0, koh)
    g_grid = np.linspace(2.0, 12.0, 40)
    D_model = np.array([D_of(sigma_koh6, g) for g in g_grid])
    D_ref = D_of(sigma_koh6, G_EARTH)
    D_analytic = D_ref * np.sqrt(G_EARTH / g_grid)

    axa.plot(g_grid, D_analytic * 1e3, "-", color="0.4", lw=2.0,
             label=r"$D = D_\mathrm{Earth}\sqrt{g_\oplus/g}$")
    axa.plot(g_grid, D_model * 1e3, "o", ms=4, color="tab:blue",
             label="Model (max-volume closure)")
    for g_pt, lab, col in ((G_EARTH, "Earth", "tab:green"),
                           (G_MARS, "Mars", "tab:red")):
        Dp = D_of(sigma_koh6, g_pt) * 1e3
        axa.axvline(g_pt, color=col, ls=":", lw=1.2)
        axa.plot([g_pt], [Dp], "s", color=col, ms=7)
        axa.annotate(f"{lab}\n{Dp:.2f} mm", (g_pt, Dp),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=8, color=col)
    axa.set_xlabel(r"Gravity $g$ (m s$^{-2}$)")
    axa.set_ylabel(r"Detachment diameter $D$ (mm)")
    axa.set_title("(a) $D$ vs gravity (KOH, $m=6$)")
    axa.legend(fontsize=8, loc="upper right")
    axa.grid(alpha=0.3)

    rel_dev = np.max(np.abs(D_model - D_analytic) / D_analytic)
    axa.text(0.03, 0.05,
             f"Max rel. dev. from $g^{{-1/2}}$: {rel_dev:.1e}",
             transform=axa.transAxes, fontsize=7.5, color="0.3")

    # Panel (b): D vs molality for KOH and H2SO4, Earth vs Mars.
    m_koh = np.linspace(0.1, 10.0, 30)
    m_h2so4 = np.linspace(0.1, 6.0, 30)
    for salt, m_grid, col in ((koh, m_koh, "tab:blue"),
                              (h2so4, m_h2so4, "tab:orange")):
        sig = np.array([toy.butler_surface_tension(m, salt)[0] for m in m_grid])
        D_e = np.array([D_of(s, G_EARTH) for s in sig]) * 1e3
        D_m = np.array([D_of(s, G_MARS) for s in sig]) * 1e3
        axb.plot(m_grid, D_e, "-", color=col, lw=1.8,
                 label=f"{salt.name}  Earth")
        axb.plot(m_grid, D_m, "--", color=col, lw=1.8,
                 label=f"{salt.name}  Mars")
    axb.set_xlabel(r"Bulk molality $m$ (mol kg$^{-1}$)")
    axb.set_ylabel(r"Detachment diameter $D$ (mm)")
    axb.set_title("(b) $D$ vs composition (Earth vs Mars)")
    axb.legend(fontsize=8, ncol=2)
    axb.grid(alpha=0.3)
    axb.text(0.03, 0.92,
             r"$D_\mathrm{Mars}/D_\mathrm{Earth}=\sqrt{g_\oplus/g_M}=1.624$"
             " (+62.4%)",
             transform=axb.transAxes, fontsize=8, color="0.3")

    fig.suptitle("Mars detachment scaling: capillary length "
                 r"$a=\sqrt{\sigma/(g\,\Delta\rho)}\sim g^{-1/2}$",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "mars_scaling")


# ---------------------------------------------------------------------------
# Figure 2 -- German 2018 CNT sigma^3 amplification
# (numbers transcribed from nucleation_german2018.md)
# ---------------------------------------------------------------------------
def make_nucleation() -> None:
    sigma_theirs = 72.000    # mN/m (German assumed pure water)
    sigma_ours = 72.699      # mN/m (our Butler, 0.5 M H2SO4)
    ratio_headline = sigma_ours / sigma_theirs        # 1.00971
    Ea_kT = np.array([14.0, 20.0, 26.0])              # German barrier range

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # Panel (a): barrier factor = (sigma_ours/sigma_theirs)^3 across a sigma band.
    r = np.linspace(1.00, 1.15, 100)
    axa.plot((r - 1) * 100, (r ** 3 - 1) * 100, "-", color="tab:purple",
             lw=2.0, label=r"Barrier $\propto\sigma^3$")
    axa.plot([(ratio_headline - 1) * 100], [(ratio_headline ** 3 - 1) * 100],
             "o", ms=8, color="tab:red",
             label=f"0.5 M H$_2$SO$_4$: +{(ratio_headline-1)*100:.2f}% "
                   rf"$\sigma$ $\to$ +{(ratio_headline**3-1)*100:.2f}% barrier")
    axa.set_xlabel(r"Surface-tension increment $\sigma_\mathrm{ours}/"
                   r"\sigma_\mathrm{ref}-1$ (%)")
    axa.set_ylabel(r"CNT barrier increment $\Delta G^*/\Delta G^*_\mathrm{ref}"
                   r"-1$ (%)")
    axa.set_title(r"(a) $\sigma^3$ barrier amplification")
    axa.legend(fontsize=8, loc="upper left")
    axa.grid(alpha=0.3)

    # Panel (b): rate slowdown J_theirs/J_ours vs E_a for the headline sigma.
    delta_barrier = ratio_headline ** 3 - 1.0        # fractional
    Ea_line = np.linspace(5, 30, 100)
    slowdown = np.exp(Ea_line * delta_barrier)       # J_theirs / J_ours
    axb.plot(Ea_line, slowdown, "-", color="tab:blue", lw=2.0)
    for Ea in Ea_kT:
        s = np.exp(Ea * delta_barrier)
        axb.plot([Ea], [s], "o", ms=7, color="tab:red")
        axb.annotate(rf"$E_a$={Ea:.0f} kT: ${s:.2f}\times$",
                     (Ea, s), textcoords="offset points", xytext=(6, -2),
                     fontsize=8, color="tab:red")
    axb.axhline(1.0, color="0.6", ls=":", lw=1.0)
    axb.set_xlabel(r"Activation energy $E_a$ ($k_BT$)")
    axb.set_ylabel(r"Rate slowdown $J_\mathrm{ref}/J_\mathrm{ours}$")
    axb.set_title(r"(b) Rate amplification (exp of barrier)")
    axb.grid(alpha=0.3)
    axb.text(0.03, 0.90,
             f"+{(ratio_headline-1)*100:.2f}% $\\sigma$ "
             f"$\\to$ +{delta_barrier*100:.2f}% barrier\n"
             r"$\to$ $\times$1.5--2.2 slower (German $E_a$=14--26 kT)",
             transform=axb.transAxes, fontsize=8, color="0.3")

    fig.suptitle("German et al. 2018 H$_2$ nucleation recomputed with "
                 "composition-dependent $\\sigma$ (0.5 M H$_2$SO$_4$)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "nucleation_german2018_amplification")


# ---------------------------------------------------------------------------
# Figure 3 -- Detachment-diameter uncertainty tornado
# (numbers transcribed from uncertainty_tornado.md)
# ---------------------------------------------------------------------------
def make_tornado() -> None:
    # (label, D_low mm, D_high mm, |dD| mm, % of base)
    panel_a = {
        "base": 1.848,
        "rows": [
            (r"$\theta_0$  (30$\pm$5$^\circ$)", 1.545, 2.153, 0.608, 32.9),
            (r"$\sigma$ level ($\pm$5%)",        1.801, 1.894, 0.092, 5.0),
            (r"$\delta$ Nernst ($\pm$40%)",      1.846, 1.851, 0.005, 0.3),
            (r"$C_{dl}$ ($\pm$25%)",             1.848, 1.848, 0.000, 0.0),
            (r"$E_{pzc}$ ($\pm$0.05 V)",         1.848, 1.848, 0.000, 0.0),
        ],
    }
    panel_b = {
        "base": 0.804,
        "rows": [
            (r"$\theta_0$  (30$\pm$5$^\circ$)", 0.311, 1.363, 1.052, 130.8),
            (r"$E_{pzc}$ ($\pm$0.05 V)",         1.227, 0.311, 0.916, 113.9),
            (r"$C_{dl}$ ($\pm$25%)",             1.163, 0.311, 0.852, 106.0),
            (r"$\sigma$ level ($\pm$5%)",        0.681, 0.908, 0.227, 28.3),
            (r"$\delta$ Nernst ($\pm$40%)",      0.797, 0.811, 0.013, 1.6),
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    titles = ("(a) No applied voltage ($D_d=1.848$ mm)",
              r"(b) Electrowetting active, $|E_{cell}-E_{pzc}|=0.30$ V"
              " ($D_d=0.804$ mm)")

    for ax, panel, title in zip(axes, (panel_a, panel_b), titles):
        base = panel["base"]
        rows = panel["rows"]
        y = np.arange(len(rows))[::-1]      # most influential on top
        for yi, (lab, lo, hi, dD, pct) in zip(y, rows):
            left = min(lo, hi)
            width = abs(hi - lo)
            ax.barh(yi, width, left=left, height=0.6,
                    color="tab:blue", alpha=0.75, edgecolor="0.3")
            if width > 0:
                ax.text(max(lo, hi) + 0.02, yi, f"{pct:.1f}%",
                        va="center", fontsize=8, color="0.25")
            else:
                ax.text(base + 0.02, yi, "inactive",
                        va="center", fontsize=7.5, color="0.55")
        ax.axvline(base, color="tab:red", ls="--", lw=1.4,
                   label=f"Base $D_d$ = {base:.3f} mm")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlabel(r"Detachment diameter $D_d$ (mm)")
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Detachment-diameter uncertainty apportionment "
                 "(one-at-a-time tornado, KOH $m=6$, $j=1000$ A m$^{-2}$)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "uncertainty_tornado")


def main() -> None:
    apply_style()
    salts, _ = fitted_salts()
    make_mars_scaling(salts)
    make_nucleation()
    make_tornado()


if __name__ == "__main__":
    main()
