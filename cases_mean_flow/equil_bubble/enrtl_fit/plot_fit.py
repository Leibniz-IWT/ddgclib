"""
plot_fit.py
==========

Plot each available binary-electrolyte dataset (the CSVs under ``data/``)
together with the fitted lumped e-NRTL model, so the extracted/digitised data
can be visually compared against its source figure.

For every system with a dataset on disk (see ``datasets.available_datasets``):
  * scatter the experimental gamma_pm(m) and/or phi(m),
  * overlay the model curve for the FITTED tau (and, dashed, for the toy's
    current REPRESENTATIVE tau, for reference),
  * annotate the fit RMS.

Run:  python plot_fit.py
Out:  fig/enrtl_fit/<system>_fit.png  (+ a combined overview)
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import enrtl_binary as eb
from datasets import load_csv, available_datasets

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, "..", "fig", "enrtl_fit")

TEMPLATES = {"KCl": eb.KCL_TEMPLATE, "KOH": eb.KOH_TEMPLATE,
             "H2SO4": eb.H2SO4_TEMPLATE}

# the PRIOR (unfitted) lumped tau for reference: hand-tuned "representative" for
# KOH/H2SO4 (toy tau_sw/tau_ws slots -> tau_cw/tau_wc); Valverde LITERATURE for
# KCl (which was fit in a different, full e-NRTL, not this lumped one).
REPRESENTATIVE = {
    "KOH":   dict(tau_cw=-4.5,  tau_wc=11.5),    # toy representative (hand-tuned)
    "H2SO4": dict(tau_cw=-5.2,  tau_wc=13.0),    # toy representative (hand-tuned)
    "KCl":   dict(tau_cw=-4.117, tau_wc=8.085),  # Valverde2023 literature
}
REP_LABEL = {"KCl": "Valverde lit. $\\tau$ (full-model)",
             "KOH": "representative $\\tau$ (hand-tuned)",
             "H2SO4": "representative $\\tau$ (hand-tuned)"}


def _fit(system, data, m_max=None):
    p0 = TEMPLATES.get(system, eb.ENRTLParams())
    if m_max is not None:
        keep = data["m"] <= m_max
        data = {k: (v[keep] if isinstance(v, np.ndarray) else v)
                for k, v in data.items()}
    use = tuple(o for o in ("gamma_pm", "phi")
                if o in data and np.any(np.isfinite(data[o])))
    sol = least_squares(eb.residuals, eb.pack(p0),
                        args=(data, p0), kwargs=dict(use=use), method="lm")
    return eb.unpack(sol.x, p0), use, data


def plot_system(system, path, m_max_fit=None):
    data = load_csv(path, system=system)
    p_fit, use, dfit = _fit(system, data, m_max=m_max_fit)

    m_line = np.linspace(max(1e-3, data["m"].min() * 0.5),
                         data["m"].max() * 1.02, 200)
    pred = eb.predict(m_line, p_fit)
    rep = REPRESENTATIVE.get(system)
    pred_rep = eb.predict(m_line, eb.replace(TEMPLATES.get(system, eb.ENRTLParams()),
                                             **rep)) if rep else None

    obs_list = [o for o in ("gamma_pm", "phi")
                if o in data and np.any(np.isfinite(data[o]))]
    fig, axes = plt.subplots(1, len(obs_list), figsize=(6.2 * len(obs_list), 4.6),
                             squeeze=False)
    labels = {"gamma_pm": r"mean ionic activity coeff  $\gamma_\pm$",
              "phi": r"osmotic coefficient  $\phi$"}
    for ax, obs in zip(axes[0], obs_list):
        d = data[obs]
        mask = np.isfinite(d)
        # fit-quality RMS on the fitted range
        dmask = dfit[obs]
        fmask = np.isfinite(dmask)
        rms = 100 * np.sqrt(np.mean(((eb.predict(dfit["m"][fmask], p_fit)[obs]
                                      - dmask[fmask]) / dmask[fmask]) ** 2))
        ax.plot(data["m"][mask], d[mask], "o", color="k", ms=6, zorder=5,
                label=f"data ({data['source'][0]})")
        ax.plot(m_line, pred[obs], "-", color="C0", lw=2.2,
                label=f"fitted e-NRTL ($\\tau_{{cw}}$={p_fit.tau_cw:.2f}, "
                      f"$\\tau_{{wc}}$={p_fit.tau_wc:.2f})")
        if pred_rep is not None:
            ax.plot(m_line, pred_rep[obs], "--", color="C3", lw=1.6, alpha=0.8,
                    label=REP_LABEL.get(system, "prior $\\tau$ (unfitted)"))
        if obs == "gamma_pm":
            ax.set_yscale("log")
        ax.set_xlabel("molality  m  (mol/kg)")
        ax.set_ylabel(labels[obs])
        ax.set_title(f"{obs}: fit RMS {rms:.1f}%"
                     + (f" (m$\\leq${m_max_fit})" if m_max_fit else ""))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{system}(aq), 25 $^\\circ$C — data vs fitted lumped e-NRTL",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    os.makedirs(_FIG, exist_ok=True)
    out = os.path.join(_FIG, f"{system}_fit.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {system}: fitted tau_cw={p_fit.tau_cw:.3f} tau_wc={p_fit.tau_wc:.3f}"
          + (f"  (fit m<={m_max_fit})" if m_max_fit else "") + f"  -> {out}")
    return p_fit


# molality range to fit (None = full). The lumped 2-param model cannot follow
# the extreme high-m rise of KOH (gamma_pm -> 46 at 20 m) or H2SO4's bisulfate
# speciation; restrict to the electrolysis-relevant regime.
FIT_RANGE = {"H2SO4": 6.0, "KOH": 10.0}


def main():
    ds = available_datasets()
    if not ds:
        print("No datasets found in data/.")
        return
    print(f"Plotting {len(ds)} dataset(s): {list(ds)}")
    for system, path in ds.items():
        plot_system(system, path, m_max_fit=FIT_RANGE.get(system))


if __name__ == "__main__":
    main()
