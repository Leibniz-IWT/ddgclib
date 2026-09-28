"""
fit_example.py
=============

End-to-end example: fit the lumped e-NRTL (`enrtl_binary`) to binary
electrolyte-water activity data (`datasets.load_csv`) with
`scipy.optimize.least_squares`, and report fit quality.

This is a reference driver -- swap in your own fitting library / data by
replacing the loader and the optimiser call; the model, `pack/unpack`, and
`residuals` are library-agnostic.

Run:  python fit_example.py [path/to/data.csv] [system]
Default: data/kcl_example.csv, system KCl.
"""
from __future__ import annotations
import os
import sys

import numpy as np
from scipy.optimize import least_squares

import enrtl_binary as eb
from datasets import load_csv

_HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATES = {
    "KCl": eb.KCL_TEMPLATE,
    "KOH": eb.KOH_TEMPLATE,
    "H2SO4": eb.H2SO4_TEMPLATE,
}


def fit(csv_path: str, system: str, fit_keys=("tau_cw", "tau_wc"),
        use=("gamma_pm", "phi"), also_fit_pdh: bool = False):
    data = load_csv(csv_path, system=system)
    p0 = TEMPLATES.get(system, eb.ENRTLParams())
    keys = tuple(fit_keys) + (("A_phi", "rho") if also_fit_pdh else ())

    theta0 = eb.pack(p0, fit=keys)
    sol = least_squares(eb.residuals, theta0,
                        args=(data, p0), kwargs=dict(fit=keys, use=use),
                        method="lm", max_nfev=2000)
    p_fit = eb.unpack(sol.x, p0, keys)

    # fit quality on the observables actually present
    m = data["m"]
    pred = eb.predict(m, p_fit)
    print(f"\n=== fit: {system}  ({csv_path}) ===")
    print(f"  fitted {keys} = {np.round(sol.x, 4)}")
    print(f"  cost = {sol.cost:.3e}   n_points = {len(m)}")
    for obs in use:
        if obs in data and np.any(np.isfinite(data[obs])):
            d = data[obs]
            mask = np.isfinite(d)
            rms = float(np.sqrt(np.mean((pred[obs][mask] - d[mask]) ** 2)))
            rel = float(np.sqrt(np.mean(((pred[obs][mask] - d[mask])
                                         / d[mask]) ** 2)))
            print(f"  {obs:9s}: RMS={rms:.4f}  RMS-rel={100*rel:.1f}%  "
                  f"(model {np.round(pred[obs][mask][:4],3)} vs "
                  f"data {np.round(d[mask][:4],3)} ...)")
    return p_fit, sol


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "data", "kcl_example.csv")
    system = sys.argv[2] if len(sys.argv) > 2 else "KCl"
    print("Fitting lumped e-NRTL (tau_cw, tau_wc).")
    fit(path, system)
    print("\nRetry also fitting the PDH constants (A_phi, rho):")
    fit(path, system, also_fit_pdh=True)
    print("\nNOTE: large RMS-rel => the lumped 2-parameter model cannot match "
          "the data across the full range; see data/SOURCES.md 'Model-form caveat'.")
