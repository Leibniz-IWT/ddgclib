"""Finalize surface-tension validation against Weissenborn & Pugh 1996.

Recompute predicted d(sigma)/dc for KOH and H2SO4 with the FIXED Butler
surface-tension code (multi-root branch selection; kink removed), compare to the
REAL W&P 1996 measured increments (Table 2), and regenerate the comparison
figure as vector PDF + PNG.

Run:  /home/endres/anaconda3/envs/ddg/bin/python \
        manuscript/validation/sigma_finalize.py
from cases_mean_flow/equil_bubble.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/home/endres/projects/ddgclib/cases_mean_flow/equil_bubble")
os.chdir("/home/endres/projects/ddgclib/cases_mean_flow/equil_bubble")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bubble_enrtl_electrostatic_toy as toy
from make_water_figure_sets import fitted_salts
from fig_style import apply_style

RHO_W = toy.RHO_W          # kg/m^3
SIGMA_W_mN = 1e3 * toy.SIGMA_W

# --- REAL Weissenborn & Pugh 1996 measured increments (Table 2, p.554) -------
# units: mN m^-1 per mol/L (mN m^-1 M^-1)
WP = {
    "KOH":   {"static": 1.98, "static_sd": 0.04,
              "dynamic": 2.06, "dynamic_sd": 0.03, "lit": 1.77},
    "H2SO4": {"static": 0.44, "static_sd": 0.06,
              "dynamic": 0.59, "dynamic_sd": 0.08, "lit": 0.64},
}
WP_EXP_ERR = 0.1   # overall experimental error, Table 2 footnote a

salts, info = fitted_salts()
salt_by_name = {s.name: s for s in salts}


def m_to_c(m: float, V_app: float) -> float:
    """molality (mol/kg water) -> molarity (mol/L) using apparent molar volume."""
    V_solution = 1.0 / RHO_W + m * V_app    # m^3 per kg water
    return m / (1000.0 * V_solution)


results = {}
for name in ("KOH", "H2SO4"):
    salt = salt_by_name[name]
    p_fit, m_max = info[name]
    V_app = salt.V_app

    # molality grid up to fit range
    m_grid = np.linspace(0.0, m_max, 201)
    sig = np.array([1e3 * toy.butler_surface_tension(max(m, 1e-9), salt)[0]
                    for m in m_grid])
    c_grid = np.array([m_to_c(m, V_app) for m in m_grid])

    # dilute-limit slope d(sigma)/dc via a small finite difference at m->0
    m_lo = 0.02
    sig_lo = 1e3 * toy.butler_surface_tension(m_lo, salt)[0]
    dsig_dm_dilute = (sig_lo - SIGMA_W_mN) / m_lo
    dcdm0 = RHO_W / 1000.0            # (mol/L)/(mol/kg) at m->0
    dsig_dc_dilute = dsig_dm_dilute / dcdm0

    # W&P-range least-squares slope over c = 0.05 - 1.0 M (their fit range,
    # p.551/553): the apples-to-apples comparison to their reported gradient.
    mask = (c_grid >= 0.05) & (c_grid <= 1.0)
    A = np.vstack([c_grid[mask], np.ones(mask.sum())]).T
    slope_wprange, _ = np.linalg.lstsq(A, sig[mask], rcond=None)[0]

    results[name] = {
        "tau_sw": float(p_fit.tau_cw), "tau_ws": float(p_fit.tau_wc),
        "fit_mmax": float(m_max), "V_app": float(V_app),
        "sigma0_mN": float(sig[0]),
        "sigma_at_mmax_mN": float(sig[-1]),
        "dsig_dm_dilute": float(dsig_dm_dilute),
        "dsig_dc_dilute": float(dsig_dc_dilute),
        "dsig_dc_wprange_fit": float(slope_wprange),
        "dcdm0": float(dcdm0),
        "m_grid": m_grid.tolist(),
        "c_grid": c_grid.tolist(),
        "sig_mN": sig.tolist(),
    }
    print(f"{name}: sigma0={sig[0]:.3f} mN/m  d(sig)/dc dilute={dsig_dc_dilute:+.3f}"
          f"  W&P-range(0.05-1M) fit={slope_wprange:+.3f}  (mN/m per mol/L)")

json.dump(results, open("manuscript/validation/sigma_results.json", "w"), indent=2)
print("Wrote manuscript/validation/sigma_results.json")

# --- comparison figure -------------------------------------------------------
apply_style()
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
for ax, name in zip(axes, ("KOH", "H2SO4")):
    r = results[name]
    w = WP[name]
    c = np.array(r["c_grid"])
    s = np.array(r["sig_mN"])
    s0 = SIGMA_W_mN
    cmax = c.max()

    # our full model curve
    ax.plot(c, s, color="C0", lw=2.2, label="e-NRTL$\\rightarrow$Butler (this work)")
    # our least-squares slope over the W&P fit range (0.05-1 M)
    ax.plot(c, s0 + r["dsig_dc_wprange_fit"] * c, color="C0", ls="--", lw=1.2,
            label=f"Our slope (0.05-1 M) {r['dsig_dc_wprange_fit']:+.2f}")

    # W&P measured static-slope band (+/- overall experimental error 0.1)
    lo = s0 + (w["static"] - WP_EXP_ERR) * c
    hi = s0 + (w["static"] + WP_EXP_ERR) * c
    ax.fill_between(c, lo, hi, color="C3", alpha=0.18,
                    label="W&P 1996 static $\\pm$0.1")
    ax.plot(c, s0 + w["static"] * c, color="C3", lw=1.8,
            label=f"W&P static {w['static']:.2f}$\\pm${w['static_sd']:.2f}")
    # W&P dynamic slope
    ax.plot(c, s0 + w["dynamic"] * c, color="C3", ls=":", lw=1.8,
            label=f"W&P dynamic {w['dynamic']:.2f}$\\pm${w['dynamic_sd']:.2f}")

    ax.set_xlim(0, cmax)
    ax.set_xlabel(r"Electrolyte concentration $c$ (mol L$^{-1}$)")
    ax.set_ylabel(r"Surface tension $\sigma_{lv}$ (mN m$^{-1}$)")
    ax.set_title(f"({'a' if name=='KOH' else 'b'}) {name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle("Predicted vs measured (Weissenborn & Pugh 1996) surface-tension "
             "increment $d\\sigma/dc$", y=1.02, fontsize=11)
fig.tight_layout()
out_png = "manuscript/tex/figures/sigma_vs_measured.png"
out_pdf = "manuscript/tex/figures/sigma_vs_measured.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight")            # vector PDF
plt.close(fig)
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")

# --- comparison summary ------------------------------------------------------
print("\n=== d(sigma)/dc comparison (mN/m per mol/L) ===")
print(f"{'salt':6s} {'ours':>7s} {'W&P stat':>9s} {'W&P dyn':>8s} {'lit':>5s}"
      f" {'ours/stat':>9s}")
for name in ("KOH", "H2SO4"):
    o = results[name]["dsig_dc_wprange_fit"]
    w = WP[name]
    print(f"{name:6s} {o:+7.2f} {w['static']:9.2f} {w['dynamic']:8.2f}"
          f" {w['lit']:5.2f} {o/w['static']:9.2f}")
print(f"ordering: ours KOH {results['KOH']['dsig_dc_wprange_fit']:+.2f} vs "
      f"H2SO4 {results['H2SO4']['dsig_dc_wprange_fit']:+.2f}  |  "
      f"W&P KOH {WP['KOH']['static']:.2f} vs H2SO4 {WP['H2SO4']['static']:.2f}")
