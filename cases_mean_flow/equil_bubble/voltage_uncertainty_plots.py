"""
voltage_uncertainty_plots.py
============================

Two plots requested after review:

 (A) ELECTROWETTING sweep -- the three-phase contact-angle channel that is held
     CONSTANT in the other rate figures (they use E_cell=None). Here we vary the
     applied potential and show its (large) effect on detachment size D_d and
     detachment frequency f, at fixed molality, for the DATA-FITTED systems.
     -> fig/detachment_rates/electrowetting_sweep.png

 (B) UNCERTAINTY vs CURRENT DENSITY -- the fig3 histogram is at a SINGLE fixed j,
     so it says nothing about how uncertainty scales with current. Here we
     propagate the j-dependent uncertainty (sigma scale + Nernst-layer delta,
     which sets the concentration-polarization shift m_surf -> sigma(m_surf))
     and plot the relative 90% band of D_d and f vs j. Tests the expectation
     that higher current -> more polarization/non-linearity -> more uncertainty.
     -> fig/detachment_rates/uncertainty_vs_current.png

Run:  python voltage_uncertainty_plots.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import EnvState, Buoyancy, CapillaryPinning, BulkDEP, Electrowetting
from bubble_kinetics import (NernstDiffusionLayer, monte_carlo, UncertainParam,
                             detachment_rate, BubbleGrowth)
from detachment_fitted import fitted_system

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig", "detachment_rates")
_COL = {"KOH": "C0", "H2SO4": "C3"}


def _style():
    from fig_style import apply_style
    apply_style()
    plt.rcParams.update({"figure.dpi": 110})


# ---------------------------------------------------------------------------
# (A) electrowetting sweep: theta -> D_d -> f vs applied |E_cell - E_pzc|
#
# The tail of D_d(dE) FLATTENS because the Young-Lippmann law drives the
# contact angle theta down until it hits the saturation floor theta_min (5 deg),
# after which further voltage does nothing (contact-angle saturation; Mugele &
# Baret 2005).  Panel (a) exposes this mechanism explicitly.  Analytic onset:
#     eta_sat = cos(theta_min) - cos(theta_0),
#     |dE|_sat = sqrt( eta_sat * sigma / (0.5 * C_dl) ).
# Beyond |dE|_sat the plateau height is the Fritz diameter at theta_min,
# D_floor ~ theta_min * sqrt(sigma / (g*drho)), i.e. it scales as sqrt(sigma).
# ---------------------------------------------------------------------------
def fig_electrowetting(systems, m_fix, dE_max=0.50):
    _style()
    ew = Electrowetting()
    theta_min = ew.theta_min
    fig, (axT, axD, axF) = plt.subplots(1, 3, figsize=(16.5, 4.7))
    dE = np.linspace(0.0, dE_max, 60)
    onset = {}
    for name, sysf in systems.items():
        c = _COL.get(name, "C2")
        p = toy.ELECTROCHEM_PARAMS.get(name, toy.ELECTROCHEM_PARAMS["KOH"])
        sig = sysf.sigma_lv(m_fix)
        theta0 = sysf.theta_0
        # analytic saturation onset for the annotation
        eta_sat = np.cos(theta_min) - np.cos(theta0)
        dE_sat = np.sqrt(eta_sat * sig / (0.5 * p["C_dl"]))
        onset[name] = (dE_sat, c)
        Th, Dd, freq = [], [], []
        for d in dE:
            env = EnvState(rho_l=sysf.rho_l, C_dl=p["C_dl"], E_pzc=p["E_pzc"],
                           E_cell=p["E_pzc"] - d)
            th_eff = Electrowetting().theta_shift(theta0, sig, env)  # clamped theta(dE)
            Th.append(np.rad2deg(th_eff))
            r = toy.detachment_volume(sig, theta0, models=[
                Buoyancy(), CapillaryPinning(), BulkDEP(), Electrowetting()], env=env)
            if r["converged"]:
                res = detachment_rate(r["V"], r["sigma_eff"], r["theta_eff"],
                                      m_fix, 1000.0, sysf.reaction(), sysf.growth())
                Dd.append(r["D"] * 1e3); freq.append(res.f_detach)
            else:
                Dd.append(np.nan); freq.append(np.nan)
        lab = f"{name} ($m$={m_fix}, $\\sigma_{{lv}}$={sig*1e3:.1f} mN m$^{{-1}}$)"
        axT.plot(dE, Th, color=c, label=lab)
        axD.plot(dE, Dd, color=c, label=f"{name} ($m$={m_fix})")
        axF.semilogy(dE, freq, color=c, label=f"{name} ($m$={m_fix})")
        # per-system saturation-onset marker in every panel
        for ax in (axT, axD, axF):
            ax.axvline(dE_sat, color=c, ls=":", lw=1.3, alpha=0.7)

    # shade the saturated (theta = theta_min) region from the earliest onset
    dE0 = min(v[0] for v in onset.values())
    for ax in (axT, axD, axF):
        ax.axvspan(dE0, dE_max, color="0.85", alpha=0.5, zorder=0)

    # panel (a): the mechanism -- theta(dE) and the saturation floor
    axT.axhline(np.rad2deg(theta_min), color="k", ls="--", lw=1.2)
    axT.annotate(f"$\\theta_{{min}}$ = {np.rad2deg(theta_min):.0f}$^\\circ$ "
                 "(contact-angle\nsaturation floor)",
                 xy=(dE_max*0.62, np.rad2deg(theta_min)),
                 xytext=(dE_max*0.30, np.rad2deg(theta_min)+8), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="k", lw=1))
    axT.set_xlabel(r"Applied $|E_{cell}-E_{pzc}|$ (V)")
    axT.set_ylabel(r"Contact angle $\theta_{gas}(\Delta E)$ (deg)")
    axT.set_title("(a) Mechanism: Young-Lippmann drives $\\theta$ down,\n"
                  "then it saturates at $\\theta_{min}$ "
                  "($\\cos\\theta=\\cos\\theta_0+\\eta_{\\mathrm{EW}}$, "
                  "$\\eta_{\\mathrm{EW}}\\propto\\Delta E^2$)")
    axT.set_ylim(0, 34)
    axT.legend(fontsize=8, loc="upper right")

    # panel (b): the observable -- D_d flattens at the saturation onset
    axD.set_xlabel(r"Applied $|E_{cell}-E_{pzc}|$ (V)")
    axD.set_ylabel(r"Detachment diameter $D_d$ (mm)")
    axD.set_title("(b) $D_d$ shrinks, then flattens past "
                  "$|\\Delta E|_{sat}$\n(plateau = Fritz $D_d$ at "
                  "$\\theta_{min}$, $\\propto\\sqrt{\\sigma}$)")
    # annotate the onset value (use the mean; the two systems are within ~0.01 V)
    dE_sat_txt = ", ".join(f"{k}={v[0]:.2f} V" for k, v in onset.items())
    axD.text(0.5*(dE0+dE_max), axD.get_ylim()[1]*0.9, "Saturated:\n$\\theta=\\theta_{min}$",
             ha="center", va="top", fontsize=8.5, color="0.35")
    axD.legend(fontsize=8, loc="lower left",
               title=f"$|\\Delta E|_{{sat}}$: {dE_sat_txt}", title_fontsize=8)

    # panel (c): the rate consequence -- f rises then plateaus too
    axF.set_xlabel(r"Applied $|E_{cell}-E_{pzc}|$ (V)")
    axF.set_ylabel(r"Detachment frequency $f$ (Hz)")
    axF.set_title("(c) Frequency rises with voltage, then plateaus\n"
                  "(electrowetting-assisted removal; $j=1000$ A m$^{-2}$)")
    axF.legend(fontsize=8, loc="lower right")

    fig.suptitle("Electrowetting (three-phase contact-angle) channel: the "
                 "$D_d$ plateau is contact-angle saturation ($\\theta\\!\\to\\!"
                 "\\theta_{min}$), not a numerical artefact; held constant in "
                 "the other rate figures ($E_{cell}=$ none)", y=1.03, fontsize=11)
    fig.tight_layout()
    _save(fig, "electrowetting_sweep.png")
    for name, (dE_sat, _) in onset.items():
        print(f"  {name}: theta saturates (theta->{np.rad2deg(theta_min):.0f} deg) "
              f"at |dE|_sat = {dE_sat:.3f} V  -> D_d plateau beyond this")


# ---------------------------------------------------------------------------
# (B) uncertainty vs current density (j-dependent sources only)
# ---------------------------------------------------------------------------
def fig_uncertainty_vs_current(name, sysf, m_bulk, n_mc=90, n_j=11):
    _style()
    c_bulk = m_bulk * 997.0 / (1.0 + m_bulk * sysf.M_salt)
    jlim = sysf.nernst().limiting_current(c_bulk)
    js = np.linspace(0.02 * jlim, 0.9 * jlim, n_j)

    def band(j):
        def ev(d):
            ms = NernstDiffusionLayer(delta=d["delta"], D=sysf.D, n=sysf.polar_n,
                                      sign=sysf.polar_sign, M_solute=sysf.M_salt
                                      ).surface_molality(m_bulk, j)
            sig = sysf.sigma_lv(ms) * d["sigma_scale"]
            r = toy.detachment_volume(sig, sysf.theta_0,
                                      models=sysf.pulloff_models(), env=sysf.env(j=j))
            if not r["converged"]:
                return {}
            res = detachment_rate(r["V"], r["sigma_eff"], r["theta_eff"], ms, j,
                                  sysf.reaction(), BubbleGrowth(site_density=sysf.site_density))
            return {"Dd": res.D_d * 1e3, "f": res.f_detach, "ms": ms}
        mc = monte_carlo(ev, [UncertainParam("sigma_scale", 1.0, rel_sigma=0.05),
                              UncertainParam("delta", sysf.delta, dist="lognormal",
                                             rel_sigma=0.4)], n=n_mc, seed=1)
        return mc

    Dd_med, Dd_lo, Dd_hi, msur, rel = [], [], [], [], []
    for j in js:
        mc = band(j)
        d = mc["Dd"]
        Dd_med.append(d["p50"]); Dd_lo.append(d["p05"]); Dd_hi.append(d["p95"])
        msur.append(mc["ms"]["p50"])
        rel.append(100 * (d["p95"] - d["p05"]) / d["p50"])
    Dd_med, Dd_lo, Dd_hi = map(np.array, (Dd_med, Dd_lo, Dd_hi))

    fig, (axB, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axB.fill_between(js, Dd_lo, Dd_hi, color="C0", alpha=0.25, label="90% band")
    axB.plot(js, Dd_med, "C0-", label="Median $D_d$")
    axB.axvline(jlim, color="grey", ls=":", label=f"$j_{{lim}}\\approx${jlim:.0f}")
    axB.set_xlabel(r"Current density $j$ (A m$^{-2}$)")
    axB.set_ylabel(r"Detachment diameter $D_d$ (mm)")
    axB.set_title(f"(a) {name}, $m$={m_bulk}: $D_d$ 90% band widens toward $j_{{lim}}$")
    axB.legend(fontsize=8)
    axR.plot(js, rel, "C3-o", ms=4)
    axR.set_xlabel(r"Current density $j$ (A m$^{-2}$)")
    axR.set_ylabel(r"$D_d$ relative 90% spread (%)")
    axR.set_title("(b) Uncertainty grows with current\n"
                  "(polarization pushes $m_{surf}$ up the non-linear $\\sigma(m)$)")
    ax2 = axR.twinx()
    ax2.plot(js, msur, "k--", alpha=0.5)
    ax2.set_ylabel(r"Median $m_{surf}$ (mol kg$^{-1}$)", color="k")
    fig.suptitle("Uncertainty vs current density (thermodynamic + polarization "
                 "sources; fig. 3 was at a single fixed $j$)", y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, "uncertainty_vs_current.png")
    print(f"  {name}: Dd rel-spread {rel[0]:.1f}% (j={js[0]:.0f}) -> "
          f"{rel[-1]:.1f}% (j={js[-1]:.0f}); m_surf {msur[0]:.2f}->{msur[-1]:.2f}")


def _save(fig, name):
    os.makedirs(_FIG, exist_ok=True)
    out_path = os.path.join(_FIG, name)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    if out_path.lower().endswith(".png"):
        fig.savefig(out_path[:-4] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {os.path.join(_FIG, name)}")


def main():
    koh, _, _ = fitted_system("KOH", os.path.join(os.path.dirname(__file__),
                              "enrtl_fit", "data", "koh_hamer_wu.csv"))
    h2so4, _, _ = fitted_system("H2SO4", os.path.join(os.path.dirname(__file__),
                                "enrtl_fit", "data", "h2so4_que2011_fig10.csv"))
    print("(A) electrowetting sweep")
    fig_electrowetting({"KOH": koh, "H2SO4": h2so4}, m_fix=6.0)
    print("(B) uncertainty vs current (KOH)")
    fig_uncertainty_vs_current("KOH", koh, m_bulk=6.0)


if __name__ == "__main__":
    main()
