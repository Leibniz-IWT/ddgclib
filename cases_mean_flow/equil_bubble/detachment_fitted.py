"""
detachment_fitted.py
====================

Produce the generalised non-ideality / detachment figures using the REAL
FITTED e-NRTL parameters (from enrtl_fit/) instead of the hand-tuned
"representative" ones, for every electrolyte system that has a dataset on disk.

This closes the loop: enrtl_fit fits tau to data -> to_toy_salt -> a fitted
ElectrochemicalSystem -> the same detachment pipeline / figures as
detachment_rates_case.py, but data-backed.

Run:  python detachment_fitted.py
Out:  fig/detachment_rates/fitted_nonideality.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from dataclasses import replace

import bubble_enrtl_electrostatic_toy as toy
import enrtl_fit.enrtl_binary as eb
from enrtl_fit.datasets import load_csv, available_datasets
from pulloff_models import ConstantSigma
from systems import KOH_WATER, H2SO4_WATER

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, "fig", "detachment_rates")

# base system + fit template + max fit molality per salt (electrolysis regime)
BASE = {"KOH": (KOH_WATER, eb.KOH_TEMPLATE, 10.0),
        "H2SO4": (H2SO4_WATER, eb.H2SO4_TEMPLATE, 6.0)}
# extra toy.Salt fields needed by to_toy_salt (mirror the toy's Salt defs)
SALT_EXTRA = {"KOH": dict(sigma0=toy.SIGMA0_KOH_PURE, A_s=toy.A_KOH, V_app=27e-6, k_sech=0.134),
              "H2SO4": dict(sigma0=toy.SIGMA0_H2SO4_PURE, A_s=toy.A_H2SO4, V_app=53e-6, k_sech=0.099)}


def fit_salt(system_name, csv_path):
    """Fit tau to the dataset and return (fitted toy.Salt, p_fit, m_max)."""
    base, template, m_max = BASE[system_name]
    data = load_csv(csv_path, system=system_name)
    if m_max is not None:
        keep = data["m"] <= m_max
        data = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    use = tuple(o for o in ("gamma_pm", "phi")
                if o in data and np.any(np.isfinite(data[o])))
    sol = least_squares(eb.residuals, eb.pack(template),
                        args=(data, template), kwargs=dict(use=use), method="lm")
    p_fit = eb.unpack(sol.x, template)
    salt = eb.to_toy_salt(p_fit, system_name + "_fit", **SALT_EXTRA[system_name])
    return salt, p_fit, m_max


def fitted_system(system_name, csv_path):
    """A copy of the base ElectrochemicalSystem with the FITTED salt."""
    base = BASE[system_name][0]
    salt, p_fit, m_max = fit_salt(system_name, csv_path)
    return replace(base, salt=salt, name=f"{system_name} (fitted e-NRTL)"), p_fit, m_max


def _Dd(system, m):
    sig = system.sigma_lv(m)
    r = toy.detachment_volume(sig, system.theta_0)
    return r["D"] if r["converged"] else np.nan


def main():
    ds = {s: p for s, p in available_datasets().items() if s in BASE}
    if not ds:
        print("No electrolysis-system datasets found; nothing to plot.")
        return
    print(f"Building fitted systems for: {list(ds)}")

    from fig_style import apply_style
    apply_style()
    plt.rcParams.update({"figure.dpi": 110})
    fig, (axS, axB) = plt.subplots(1, 2, figsize=(12.8, 4.9))
    colors = {"KOH": "C0", "H2SO4": "C3"}

    for system_name, path in ds.items():
        sysf, p_fit, m_max = fitted_system(system_name, path)
        c = colors.get(system_name, "C2")
        mmax = 10.0 if system_name == "KOH" else 6.0
        m = np.linspace(0.05, mmax, 40)
        sig = np.array([sysf.sigma_lv(mi) for mi in m])
        sig0 = sysf.sigma_lv(0.05)                      # dilute-limit (literature const-sigma)
        Dd = np.array([_Dd(sysf, mi) for mi in m])
        Dd_const = np.array([toy.detachment_volume(sig0, sysf.theta_0)["D"]] * len(m))

        axS.plot(m, 1e3 * sig, color=c,
                 label=f"{system_name}: fitted e-NRTL "
                       f"($\\tau_{{cw}}$={p_fit.tau_cw:.2f},$\\tau_{{wc}}$={p_fit.tau_wc:.2f})")
        axS.axhline(1e3 * sig0, color=c, ls=":", alpha=0.7)
        axB.plot(m, 100 * (Dd - Dd_const) / Dd_const, color=c, label=system_name)

    axS.set_xlabel(r"Bulk molality $m$ (mol kg$^{-1}$)")
    axS.set_ylabel(r"Surface tension $\sigma_{lv}$ (mN m$^{-1}$)")
    axS.set_title("(a) Fitted e-NRTL $\\sigma_{lv}(m)$ (solid) vs fixed-$\\sigma$ (dotted)")
    axS.legend(fontsize=8)
    axB.axhline(0, color="k", lw=1)
    axB.set_xlabel(r"Bulk molality $m$ (mol kg$^{-1}$)")
    axB.set_ylabel(r"Detachment-diameter bias $\Delta D_d/D_d$ (%)")
    axB.set_title("(b) Model-form bias from neglecting $\\sigma(m)$ (data-fitted)")
    axB.legend(fontsize=8)
    fig.suptitle("Detachment non-ideality with data-fitted e-NRTL "
                 "(Hamer & Wu KOH / Que 2011 H$_2$SO$_4$)", y=1.02, fontsize=12)
    fig.tight_layout()
    os.makedirs(_FIG, exist_ok=True)
    out = os.path.join(_FIG, "fitted_nonideality.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
