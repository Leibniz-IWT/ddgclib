import os, sys, json
sys.path.insert(0, "/home/endres/projects/ddgclib/cases_mean_flow/equil_bubble")
os.chdir("/home/endres/projects/ddgclib/cases_mean_flow/equil_bubble")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import EnvState, Buoyancy, CapillaryPinning, BulkDEP, Electrowetting
from bubble_kinetics import (NernstDiffusionLayer, monte_carlo, UncertainParam,
                             detachment_rate, BubbleGrowth)
from detachment_fitted import fitted_system

FIG = "fig/validation"
os.makedirs(FIG, exist_ok=True)

koh, _, _ = fitted_system("KOH", os.path.join("enrtl_fit", "data", "koh_hamer_wu.csv"))

# ---- base case -----------------------------------------------------------
M_BULK = 6.0
J_BASE = 1000.0          # A/m^2  (representative HER current density)
DE_BASE = 0.30           # V  applied double-layer bias |E_cell - E_pzc|
BASE = dict(sigma_scale=1.0, C_dl=koh.C_dl, E_pzc=koh.E_pzc,
            theta_0_deg=koh.theta_0_deg, delta=koh.delta)
E_CELL = koh.E_pzc - DE_BASE      # fixed potentiostat setpoint

MODELS = [Buoyancy(), CapillaryPinning(), BulkDEP(), Electrowetting()]


def eval_Dd(sigma_scale, C_dl, E_pzc, theta_0_deg, delta, j=J_BASE, e_cell=E_CELL):
    """Detachment diameter D_d [mm] and volume V_d [m^3] for one input set."""
    ms = NernstDiffusionLayer(delta=delta, D=koh.D, n=koh.polar_n,
                              sign=koh.polar_sign,
                              M_solute=koh.M_salt).surface_molality(M_BULK, j)
    sig = koh.sigma_lv(ms) * sigma_scale
    env = EnvState(rho_l=koh.rho_l, j=j, kappa_e=koh.kappa_e, eps_r=koh.eps_r,
                   C_dl=C_dl, E_pzc=E_pzc, E_cell=e_cell)
    r = toy.detachment_volume(sig, np.deg2rad(theta_0_deg), models=MODELS, env=env)
    if not r["converged"]:
        return None, None, ms
    return r["D"] * 1e3, r["V"], ms


def tornado_rows(e_cell):
    D0 = eval_Dd(**BASE, e_cell=e_cell)[0]
    rows = []
    for label, key, lo, hi, note in PERTS:
        plo = dict(BASE); plo[key] = lo
        phi = dict(BASE); phi[key] = hi
        Dlo = eval_Dd(**plo, e_cell=e_cell)[0]
        Dhi = eval_Dd(**phi, e_cell=e_cell)[0]
        dlo, dhi = Dlo - D0, Dhi - D0
        rows.append(dict(label=label, key=key, note=note, Dlo=Dlo, Dhi=Dhi,
                         dlo=dlo, dhi=dhi, span=abs(dhi - dlo),
                         rel_span_pct=100*abs(dhi-dlo)/D0))
    rows.sort(key=lambda r: r["span"])
    return D0, rows


D0, V0, ms0 = eval_Dd(**BASE)
print(f"BASE: m_surf={ms0:.4f}  D_d={D0:.5f} mm  V_d={V0:.4e} m^3  (dE={DE_BASE} V, j={J_BASE})")

# ---- one-at-a-time perturbations (tornado) -------------------------------
# (label, key, low_value, high_value, human-readable range note)
PERTS = [
    (r"$\sigma$ level ($\tau$ proxy, $\pm$5%)", "sigma_scale", 0.95, 1.05, "+/-5%"),
    (r"$\theta_0$ (30$\pm$5$^\circ$)",          "theta_0_deg", 25.0, 35.0, "25-35 deg"),
    (r"$\delta$ Nernst layer ($\pm$40%)",       "delta", 0.6*koh.delta, 1.4*koh.delta, "+/-40%"),
    (r"$C_{dl}$ ($\pm$25%)",                    "C_dl", 0.75*koh.C_dl, 1.25*koh.C_dl, "+/-25%"),
    (r"$E_{pzc}$ ($\pm$0.05 V)",                "E_pzc", koh.E_pzc-0.05, koh.E_pzc+0.05, "+/-0.05 V"),
]

D0_bias, rows_bias = tornado_rows(E_CELL)     # electrowetting active (|dE|=0.3 V)
D0_nov, rows_nov = tornado_rows(None)         # no applied voltage (manuscript default)

for tag, D0b, rws in (("BIAS |dE|=0.3V", D0_bias, rows_bias),
                      ("NO-VOLTAGE E_cell=None", D0_nov, rows_nov)):
    print(f"\n=== TORNADO [{tag}] base D_d = {D0b:.5f} mm ===")
    print(f"{'input':12s} {'range':12s} {'D_lo':>8s} {'D_hi':>8s} {'|dD_d| mm':>10s} {'%':>7s}")
    for r in reversed(rws):
        print(f"{r['key']:12s} {r['note']:12s} {r['Dlo']:8.4f} {r['Dhi']:8.4f} "
              f"{r['span']:10.5f} {r['rel_span_pct']:7.2f}")

# ---- tornado figure (two panels) -----------------------------------------
plt.rcParams.update({"figure.dpi": 120})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.0, 4.8))

def draw(ax, D0b, rws, title):
    for i, r in enumerate(rws):
        left, width = min(r["dlo"], r["dhi"]), abs(r["dhi"] - r["dlo"])
        ax.barh(i, width, left=left, height=0.62, color="#4C72B0",
                edgecolor="k", linewidth=0.6)
        ax.text(r["dlo"], i, f"{r['dlo']:+.3f} ", va="center",
                ha="right" if r["dlo"] < 0 else "left", fontsize=7, color="#333")
        ax.text(r["dhi"], i, f" {r['dhi']:+.3f}", va="center",
                ha="left" if r["dhi"] >= 0 else "right", fontsize=7, color="#333")
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(np.arange(len(rws)))
    ax.set_yticklabels([r["label"] for r in rws], fontsize=9)
    ax.set_xlabel(r"$\Delta D_d$  (mm)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)

draw(axA, D0_nov, rows_nov,
     f"(a) NO applied voltage (manuscript default)\nbase $D_d$={D0_nov:.3f} mm; "
     r"$C_{dl}$,$E_{pzc}$ inactive")
draw(axB, D0_bias, rows_bias,
     f"(b) electrowetting active, $|\\Delta E|$={DE_BASE} V\nbase $D_d$={D0_bias:.3f} mm; "
     "electrocapillary channel dominates")
fig.suptitle(f"Local sensitivity (tornado) of detachment diameter $D_d$  "
             f"(KOH, m={M_BULK:.0f}, j={J_BASE:.0f} A/m$^2$); bars = one-at-a-time "
             r"$\pm$realistic perturbation", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "uncertainty_tornado.png"), dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("\nWrote", os.path.join(FIG, "uncertainty_tornado.png"))
rows = rows_bias; D0 = D0_bias

# ---- current-density trend (short MC) ------------------------------------
c_bulk = M_BULK * 997.0 / (1.0 + M_BULK * koh.M_salt)
jlim = koh.nernst().limiting_current(c_bulk)
js = np.array([0.05, 0.25, 0.5, 0.75, 0.9]) * jlim

def band(j, n_mc=200):
    def ev(d):
        Dd, V, ms = eval_Dd(d["sigma_scale"], BASE["C_dl"], BASE["E_pzc"],
                            BASE["theta_0_deg"], d["delta"], j=j)
        if Dd is None:
            return {}
        return {"Dd": Dd, "ms": ms}
    mc = monte_carlo(ev, [UncertainParam("sigma_scale", 1.0, rel_sigma=0.05),
                          UncertainParam("delta", koh.delta, dist="lognormal",
                                         rel_sigma=0.4)], n=n_mc, seed=1)
    return mc

print("\n=== CURRENT-DENSITY TREND (KOH m=6; MC over sigma +/-5% and delta lognormal 40%) ===")
print(f"jlim = {jlim:.0f} A/m^2")
print(f"{'j/jlim':>7s} {'j':>9s} {'m_surf':>8s} {'D_d p50':>9s} {'p05':>8s} {'p95':>8s} {'rel90%':>8s}")
trend = []
for j in js:
    mc = band(j)
    d = mc["Dd"]; msur = mc["ms"]["p50"]
    rel = 100*(d["p95"]-d["p05"])/d["p50"]
    trend.append((j/jlim, j, msur, d["p50"], d["p05"], d["p95"], rel))
    print(f"{j/jlim:7.2f} {j:9.0f} {msur:8.3f} {d['p50']:9.4f} {d['p05']:8.4f} "
          f"{d['p95']:8.4f} {rel:7.2f}%")

# save raw numbers
raw = dict(base=dict(m_bulk=M_BULK, j=J_BASE, dE=DE_BASE, E_cell=E_CELL,
                     D0_mm=D0, V0_m3=V0, m_surf0=ms0,
                     C_dl=koh.C_dl, E_pzc=koh.E_pzc, theta_0_deg=koh.theta_0_deg,
                     delta=koh.delta, D_diff=koh.D, M_salt=koh.M_salt),
           tornado_bias=[{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                     for k, v in r.items()} for r in reversed(rows_bias)],
           tornado_novoltage=[{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                     for k, v in r.items()} for r in reversed(rows_nov)],
           D0_novoltage_mm=float(D0_nov),
           jlim=float(jlim),
           trend=[dict(j_over_jlim=float(a), j=float(b), m_surf=float(c),
                       Dd_p50=float(dd), Dd_p05=float(e), Dd_p95=float(f),
                       rel90_pct=float(g)) for a,b,c,dd,e,f,g in trend])
with open("/tmp/claude-1000/-home-endres-projects-ddgclib/40088a3e-090d-49c0-a0c1-8d711a972cc8/scratchpad/raw.json","w") as fh:
    json.dump(raw, fh, indent=2)
print("\nSaved raw.json")
