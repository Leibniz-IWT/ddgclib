"""
detachment_rates_regolith.py
============================

Stage-1 Mars molten-regolith-electrolysis test case (mirrors
../detachment_rates_case.py).  Produces three figures in ../fig/regolith/:

  fig1_sigma_bias.png   sigma(xi, model) + detachment-size bias of the
                        constant-sigma assumption, Earth vs Mars
  fig2_rates.png        local composition (Nernst) -> sigma -> D_d and
                        f = Q/V_d vs current density, Earth vs Mars, with the
                        basicity-capped sustainable-j window
  fig3_uncertainty.png  Monte-Carlo propagation of the verified melt
                        uncertainty set to V_d, D_d, f (vs the aqueous case's
                        ~8-11% spread)

Run:
    /home/endres/anaconda3/envs/ddg/bin/python regolith/detachment_rates_regolith.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASE = os.path.dirname(_HERE)
for p in (_HERE, _CASE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import (solve_detachment, Buoyancy, CapillaryPinning,
                            EnvState, ConstantSigma, G_EARTH, G_MARS)
from bubble_kinetics import UncertainParam, monte_carlo, FARADAY, R_GAS

from melt_nrtl import OXIDES, IDX, interim_melt, stage1_composition
from melt_butler_sigma import (MeltHandle, MeltButlerSigma, XinLinearSigma,
                               sigma0_nakamoto)
from regolith_system import (RegolithMeltSystem, MeltNernstLayer,
                             kappa_khetpal, kappa_haskin, t_electronic,
                             current_efficiency, rho_melt, rho_o2,
                             T_OP, P_HEAD, J_NOMINAL)

FIG_DIR = os.path.join(_CASE, "fig", "regolith")
os.makedirs(FIG_DIR, exist_ok=True)

X_FEO_0 = stage1_composition(0.0)[IDX["FeO"]]     # 0.188


# ---------------------------------------------------------------------------
# Optical basicity of the melt (Duffy scale; approximate Lambda_i values) and
# the Sibille 2009 sustainable-current-density law (KINETICS.md sec.2).
# ---------------------------------------------------------------------------
LAMBDA_I = {"SiO2": 0.48, "Al2O3": 0.60, "CaO": 1.00, "MgO": 0.78, "FeO": 1.00}
N_OXY = {"SiO2": 2, "Al2O3": 3, "CaO": 1, "MgO": 1, "FeO": 1}
_SIBILLE = (np.array([0.555, 0.605, 0.632, 0.661, 0.693]),
            np.array([0.02, 0.28, 0.31, 0.54, 0.67]))     # Lambda -> A/cm^2


def optical_basicity(x: np.ndarray) -> float:
    num = sum(x[IDX[ox]] * N_OXY[ox] * LAMBDA_I[ox] for ox in OXIDES)
    den = sum(x[IDX[ox]] * N_OXY[ox] for ox in OXIDES)
    return float(num / den)


def j_max_sustainable(x: np.ndarray) -> float:
    """Sustainable anodic j [A/m^2] from the Sibille Fig.4 basicity data."""
    lam = optical_basicity(x)
    j_cm2 = np.interp(lam, *_SIBILLE)
    return float(j_cm2) * 1e4


def xi_from_x_feo(x_feo: float, a: float = X_FEO_0) -> float:
    """Invert stage1_composition: x_FeO -> conversion xi."""
    x_feo = float(np.clip(x_feo, 0.0, a))
    return float(np.clip((a - x_feo) / (a * (1.0 - x_feo)), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Validation gates (INTEGRATION_PLAN.md sec.6) -- run before the figures
# ---------------------------------------------------------------------------
def validation_gates() -> None:
    print("== validation gates ==")
    # 1. aqueous regression (untouched code path)
    sig_koh = float(np.ravel(toy.butler_surface_tension(1.0, toy.KOH)[0])[0])
    env_aq = EnvState(g=G_EARTH)
    r = solve_detachment(sig_koh, np.deg2rad(30.0),
                         models=[Buoyancy(), CapillaryPinning()], env=env_aq,
                         shape_fn=toy.young_laplace_shape)
    print(f" 1. aqueous KOH m=1 theta=30: D_d = {r['D']*1e3:.3f} mm "
          f"(canonical 1.733) {'OK' if abs(r['D']*1e3-1.733)<0.02 else 'FAIL'}")
    sysm = RegolithMeltSystem()
    # 2+3. similarity + Mars/Earth ratio
    rE, rM = sysm.detach(0.0, g=G_EARTH), sysm.detach(0.0, g=G_MARS)
    ratio = rM["D"] / rE["D"]
    print(f" 2. Mars/Earth D ratio = {ratio:.4f} (exact 1.6236) "
          f"{'OK' if abs(ratio-1.6236)<2e-3 else 'FAIL'}")
    # 4. gamma_FeO anchor
    gf = interim_melt(1.70).gamma_of("FeO", stage1_composition(0.0))
    print(f" 3. gamma_FeO(baseline) = {gf:.3f} (target 1.70, envelope 1.2-2.1) "
          f"{'OK' if 1.2 <= gf <= 2.1 else 'FAIL'}")
    # 5. sigma band
    s0 = sysm.sigma(0.0)
    print(f" 4. sigma(xi=0, 1850K) = {s0*1e3:.1f} mN/m (band 300-500) "
          f"{'OK' if 0.30 <= s0 <= 0.50 else 'FAIL'}")
    # 6. cryolite analogue (indicative)
    env_cr = EnvState(g=G_EARTH, rho_l=2050.0, rho_g=1.0)
    r_cr = solve_detachment(0.13, np.deg2rad(112.0),
                            models=[Buoyancy(), CapillaryPinning()], env=env_cr,
                            shape_fn=toy.young_laplace_shape)
    print(f" 5. cryolite theta=112: D_d = {r_cr['D']*1e3:.2f} mm "
          f"(Stanic measured 5.7-7.2; indicative)")


# ---------------------------------------------------------------------------
# Figure 1: sigma(xi) per model + constant-sigma detachment bias
# ---------------------------------------------------------------------------
def fig1_sigma_bias() -> None:
    xis = np.linspace(0.0, 1.0, 25)
    handle17 = MeltHandle(melt=interim_melt(1.70), T=T_OP)
    models = {
        r"Butler-NRTL, $\gamma_{FeO}$=1.7": (MeltButlerSigma(), handle17, "C0", "-"),
        r"Butler-NRTL, $\gamma_{FeO}$=1.2": (MeltButlerSigma(),
                                             MeltHandle(interim_melt(1.2), T_OP), "C0", ":"),
        r"Butler-NRTL, $\gamma_{FeO}$=2.1": (MeltButlerSigma(),
                                             MeltHandle(interim_melt(2.1), T_OP), "C0", "--"),
        r"Butler, $\sigma^0_{FeO}$ low (Xin)": (MeltButlerSigma(sigma_feo0_1773=0.571),
                                                handle17, "C2", "-."),
        "Xin 2020 linear (upper bound)": (XinLinearSigma(), handle17, "C3", "-"),
    }
    SIG_CONST = 0.35      # literature-constant twin (natural basalt value)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    curves = {}
    for label, (m, h, c, ls) in models.items():
        s = np.array([m.sigma_lv(x, h, T_OP) for x in xis])
        curves[label] = s
        ax.plot(xis, s * 1e3, color=c, ls=ls, label=label, lw=1.6)
    ax.axhline(SIG_CONST * 1e3, color="k", lw=1.2, alpha=0.7,
               label=r"constant $\sigma$ = 350 mN/m")
    ax.axhspan(350, 370, color="0.85", zorder=0,
               label="natural basalts (Walker & Mullins)")
    ax.set_xlabel(r"FeO conversion $\xi$ (Stage-1 progress)")
    ax.set_ylabel(r"$\sigma_{lv}$  [mN/m]")
    ax.set_title(f"Melt surface tension, T = {T_OP:.0f} K")
    ax.legend(fontsize=6.5, loc="center left")

    sysm = RegolithMeltSystem()
    ax = axes[1]
    for g, tag, ls in ((G_EARTH, "Earth", "--"), (G_MARS, "Mars", "-")):
        for label, sig_curve, c in (
                (r"Butler-NRTL $\gamma_{FeO}$=1.7",
                 curves[r"Butler-NRTL, $\gamma_{FeO}$=1.7"], "C0"),
                (r"constant $\sigma$=350", np.full_like(xis, SIG_CONST), "k")):
            D = np.array([sysm.detach(x, g=g, sigma=s)["D"]
                          for x, s in zip(xis, sig_curve)])
            ax.plot(xis, D * 1e3, color=c, ls=ls, lw=1.6,
                    label=f"{label}, {tag}")
    ax.set_xlabel(r"FeO conversion $\xi$")
    ax.set_ylabel(r"$D_d$  [mm]")
    ax.set_title(r"Detachment diameter ($\theta_{gas}$ = 25$^\circ$)")
    ax.legend(fontsize=7)

    ax = axes[2]
    for label, c, ls in ((r"Butler-NRTL, $\gamma_{FeO}$=1.7", "C0", "-"),
                         (r"Butler-NRTL, $\gamma_{FeO}$=1.2", "C0", ":"),
                         (r"Butler-NRTL, $\gamma_{FeO}$=2.1", "C0", "--"),
                         ("Xin 2020 linear (upper bound)", "C3", "-")):
        s = curves[label]
        bias_f = (s / SIG_CONST) ** (-1.5) - 1.0     # f ~ sigma^{-3/2}
        ax.plot(xis, 100 * bias_f, color=c, ls=ls, lw=1.6, label=label)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"FeO conversion $\xi$")
    ax.set_ylabel(r"bias in $f$ vs constant-$\sigma$  [%]")
    ax.set_title(r"Constant-$\sigma$ rate bias  ($f \propto \sigma^{-3/2}$)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_sigma_bias.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: current density -> local composition -> sigma -> D_d, f
# ---------------------------------------------------------------------------
def fig2_rates() -> None:
    sysm = RegolithMeltSystem()
    nernst = MeltNernstLayer()
    x0 = stage1_composition(0.0)
    lam = optical_basicity(x0)
    j_cap = j_max_sustainable(x0)
    j_lim = nernst.limiting_current_feo(X_FEO_0)
    js = np.logspace(np.log10(0.02e4), np.log10(2.0e4), 24)   # 0.02-2 A/cm^2

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # local composition + sigma vs j
    x_surf = np.array([nernst.surface_x_feo(X_FEO_0, j) for j in js])
    xi_eff = np.array([xi_from_x_feo(xs) for xs in x_surf])
    sig_loc = np.array([sysm.sigma(x) for x in xi_eff])
    ax = axes[0]
    ax.semilogx(js / 1e4, x_surf, "C0-", lw=1.6, label=r"$x_{FeO}$ at anode")
    ax.axhline(X_FEO_0, color="C0", ls=":", lw=1, label=r"$x_{FeO}$ bulk")
    ax2 = ax.twinx()
    ax2.semilogx(js / 1e4, sig_loc * 1e3, "C3-", lw=1.6)
    ax2.set_ylabel(r"local $\sigma$  [mN/m]", color="C3")
    ax.axvline(j_lim / 1e4, color="0.4", ls="--", lw=1)
    ax.text(j_lim / 1e4, 0.02, r" $j_{lim}$(Fe$^{2+}$)", fontsize=7, color="0.3")
    ax.set_xlabel(r"$j$  [A/cm$^2$]")
    ax.set_ylabel(r"anode-local $x_{FeO}$", color="C0")
    ax.set_title("Nernst polarisation of the anode melt")
    ax.legend(fontsize=7, loc="center left")

    # D_d vs j (through local sigma), Earth vs Mars
    ax = axes[1]
    for g, tag, ls in ((G_EARTH, "Earth", "--"), (G_MARS, "Mars", "-")):
        D = np.array([sysm.detach(x, g=g, sigma=s)["D"]
                      for x, s in zip(xi_eff, sig_loc)])
        ax.semilogx(js / 1e4, D * 1e3, "C0", ls=ls, lw=1.6, label=tag)
    ax.axvspan(j_cap / 1e4, js[-1] / 1e4, color="0.9", zorder=0)
    ax.axvline(j_cap / 1e4, color="0.5", ls="-.", lw=1)
    ax.text(j_cap / 1e4, ax.get_ylim()[0], r" $j_{max}(\Lambda$="
            f"{lam:.3f})", fontsize=7, color="0.3")
    ax.set_xlabel(r"$j$  [A/cm$^2$]")
    ax.set_ylabel(r"$D_d$  [mm]")
    ax.set_title("Detachment size vs current density")
    ax.legend(fontsize=8)

    # f vs j
    ax = axes[2]
    for g, tag, ls in ((G_EARTH, "Earth", "--"), (G_MARS, "Mars", "-")):
        f_arr = []
        for j, x, s in zip(js, xi_eff, sig_loc):
            V_d = sysm.detach(x, g=g, sigma=s)["V"]
            res = sysm.rate(V_d, x, j=j, sigma_eff=s)
            f_arr.append(res.f_detach)
        ax.loglog(js / 1e4, f_arr, "C0", ls=ls, lw=1.6, label=tag)
    ax.axvspan(j_cap / 1e4, js[-1] / 1e4, color="0.9", zorder=0)
    ax.set_xlabel(r"$j$  [A/cm$^2$]")
    ax.set_ylabel(r"$f = Q_{site}/V_d$  [Hz]")
    ax.set_title(f"Detachment frequency (CE = 1 - 1.99$x_{{FeO}}$;\n"
                 f"site density {sysm.site_density:.0e} m$^{{-2}}$, "
                 f"p = {sysm.P/1e3:.0f} kPa)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_rates.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}  [Lambda = {lam:.3f}, j_max = {j_cap/1e4:.2f} A/cm2, "
          f"j_lim(Fe2+) = {j_lim/1e4:.2f} A/cm2]")


# ---------------------------------------------------------------------------
# Figure 3: Monte-Carlo uncertainty (verified melt priors)
# ---------------------------------------------------------------------------
def fig3_uncertainty(n: int = 250, seed: int = 7) -> None:
    sysm = RegolithMeltSystem()

    # Thermodynamic/mixture priors (the paper's subject) ...
    thermo_params = [
        UncertainParam("gamma_feo", 1.70, dist="uniform", lo=1.2, hi=2.1),
        UncertainParam("sigma_feo0", 0.625, dist="uniform", lo=0.571, hi=0.679),
        UncertainParam("sigma_form", 0.5, dist="uniform", lo=0.0, hi=1.0),
        UncertainParam("rho_l", 2800.0, dist="uniform", lo=2700.0, hi=2900.0),
        UncertainParam("T", 1850.0, dist="uniform", lo=1825.0, hi=1950.0),
        UncertainParam("xi", 0.25, dist="uniform", lo=0.0, hi=0.5),
    ]
    # ... plus geometry/site priors for the full budget.
    full_params = thermo_params + [
        UncertainParam("theta_deg", 25.0, dist="uniform", lo=10.0, hi=40.0),
        UncertainParam("site_density", 1e4, dist="lognormal", rel_sigma=0.8),
    ]

    def evaluate(draw: dict) -> dict:
        gf = round(draw["gamma_feo"], 2)
        handle = MeltHandle(melt=interim_melt(gf), T=draw["T"])
        butler = MeltButlerSigma(sigma_feo0_1773=draw["sigma_feo0"])
        s_b = butler.sigma_lv(draw["xi"], handle, draw["T"])
        s_x = XinLinearSigma().sigma_lv(draw["xi"], handle, draw["T"])
        sigma = s_b + draw["sigma_form"] * (s_x - s_b)   # model-form blend
        env = EnvState(g=G_MARS, rho_l=draw["rho_l"],
                       rho_g=rho_o2(draw["T"], sysm.P))
        theta = draw.get("theta_deg", sysm.theta_gas_deg)
        r = solve_detachment(sigma, np.deg2rad(theta),
                             models=[Buoyancy(), CapillaryPinning()], env=env,
                             shape_fn=toy.young_laplace_shape)
        if not r.get("converged"):
            return {}
        x = stage1_composition(draw["xi"])
        ce = current_efficiency(x)
        A_site = 1.0 / draw.get("site_density", sysm.site_density)
        Q = ce * (J_NOMINAL * A_site / (4 * FARADAY)) * R_GAS * draw["T"] / sysm.P
        return dict(V_d=r["V"], D_d=r["D"], sigma=sigma,
                    f=Q / r["V"], gas_rate=Q)

    stats_thermo = monte_carlo(evaluate, thermo_params, n=n, seed=seed)
    stats = monte_carlo(evaluate, full_params, n=n, seed=seed)
    if not stats:
        print("MC produced no converged samples!")
        return

    def half_spread(s):
        return (s["p95"] - s["p05"]) / (2 * s["p50"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, key, unit, scale in ((axes[0], "D_d", "mm", 1e3),
                                 (axes[1], "sigma", "mN/m", 1e3),
                                 (axes[2], "f", "Hz", 1.0)):
        s, st = stats[key], stats_thermo[key]
        all_v = np.concatenate([s["samples"], st["samples"]]) * scale
        if key == "f":
            bins = np.logspace(np.log10(all_v.min()), np.log10(all_v.max()), 28)
        else:
            bins = np.linspace(all_v.min(), all_v.max(), 28)
        ax.hist(st["samples"] * scale, bins=bins, color="C2", alpha=0.65,
                label=f"mixture-thermo only ({100*half_spread(st):.0f}%)")
        ax.hist(s["samples"] * scale, bins=bins, color="C0", alpha=0.55,
                label=f"+ wetting & site priors ({100*half_spread(s):.0f}%)")
        for q, ls in (("p05", ":"), ("p50", "-"), ("p95", ":")):
            ax.axvline(s[q] * scale, color="k", ls=ls, lw=1.0)
        ax.set_xlabel(f"{key}  [{unit}]")
        ax.set_title(f"{key}: p50 = {s['p50']*scale:.3g} {unit}")
        ax.legend(fontsize=7)
        if key == "f":
            ax.set_xscale("log")
    axes[0].set_ylabel(f"count (n = {len(stats['D_d']['samples'])})")
    fig.suptitle("Mars Stage-1 O$_2$ detachment under verified melt-model "
                 "uncertainty (aqueous-case $D_d$ spread: ~8-11%)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_uncertainty.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")
    for tag, st in (("thermo-only", stats_thermo), ("full", stats)):
        for key in ("D_d", "sigma", "f"):
            s = st[key]
            print(f"   [{tag:11s}] {key:6s}: p50 = {s['p50']:.4g}, "
                  f"p05 = {s['p05']:.4g}, p95 = {s['p95']:.4g}, "
                  f"half-spread = {100*half_spread(s):.0f}%")


if __name__ == "__main__":
    validation_gates()
    fig1_sigma_bias()
    fig2_rates()
    fig3_uncertainty()
