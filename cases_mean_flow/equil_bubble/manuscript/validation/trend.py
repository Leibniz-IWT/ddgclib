import os, sys, json
sys.path.insert(0, "/home/endres/projects/ddgclib/cases_mean_flow/equil_bubble")
os.chdir("/home/endres/projects/ddgclib/cases_mean_flow/equil_bubble")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import EnvState, Buoyancy, CapillaryPinning, BulkDEP, Electrowetting
from bubble_kinetics import NernstDiffusionLayer, monte_carlo, UncertainParam
from detachment_fitted import fitted_system

koh, _, _ = fitted_system("KOH", os.path.join("enrtl_fit","data","koh_hamer_wu.csv"))
M_BULK = 6.0
MODELS = [Buoyancy(), CapillaryPinning(), BulkDEP(), Electrowetting()]

def eval_Dd(sigma_scale, delta, j):
    ms = NernstDiffusionLayer(delta=delta, D=koh.D, n=koh.polar_n, sign=koh.polar_sign,
                              M_solute=koh.M_salt).surface_molality(M_BULK, j)
    sig = koh.sigma_lv(ms) * sigma_scale
    env = EnvState(rho_l=koh.rho_l, j=j, kappa_e=koh.kappa_e, eps_r=koh.eps_r,
                   C_dl=koh.C_dl, E_pzc=koh.E_pzc, E_cell=None)   # no voltage (clean trend)
    r = toy.detachment_volume(sig, koh.theta_0, models=MODELS, env=env)
    return (r["D"]*1e3, ms) if r["converged"] else (None, ms)

c_bulk = M_BULK*997.0/(1.0+M_BULK*koh.M_salt)
jlim = koh.nernst().limiting_current(c_bulk)
fracs = np.array([0.05,0.15,0.25,0.4,0.55,0.7,0.85,0.95])
js = fracs*jlim

def band(j, n_mc=300):
    def ev(d):
        Dd, ms = eval_Dd(d["sigma_scale"], d["delta"], j)
        return {"Dd": Dd, "ms": ms} if Dd is not None else {}
    return monte_carlo(ev, [UncertainParam("sigma_scale",1.0,rel_sigma=0.05),
                            UncertainParam("delta",koh.delta,dist="lognormal",rel_sigma=0.4)],
                       n=n_mc, seed=1)

print(f"jlim={jlim:.0f} A/m^2  (KOH m=6, no applied voltage)")
print(f"{'j/jlim':>7s} {'j':>9s} {'m_surf':>8s} {'D_d p50':>9s} {'p05':>8s} {'p95':>8s} {'rel90%':>8s}")
rows=[]; med=[]; lo=[]; hi=[]; rel=[]; msur=[]
for f,j in zip(fracs,js):
    mc=band(j); d=mc["Dd"]; ms=mc["ms"]["p50"]
    r=100*(d["p95"]-d["p05"])/d["p50"]
    print(f"{f:7.2f} {j:9.0f} {ms:8.3f} {d['p50']:9.4f} {d['p05']:8.4f} {d['p95']:8.4f} {r:7.2f}%")
    rows.append(dict(j_over_jlim=float(f), j=float(j), m_surf=float(ms),
                     Dd_p50=float(d["p50"]), Dd_p05=float(d["p05"]),
                     Dd_p95=float(d["p95"]), rel90_pct=float(r)))
    med.append(d["p50"]); lo.append(d["p05"]); hi.append(d["p95"]); rel.append(r); msur.append(ms)

med,lo,hi=map(np.array,(med,lo,hi))
fig,(axB,axR)=plt.subplots(1,2,figsize=(12.5,4.7))
axB.fill_between(js,lo,hi,color="C0",alpha=0.25,label="90% band (p05-p95)")
axB.plot(js,med,"C0-o",ms=4,label="median $D_d$")
axB.axvline(jlim,color="grey",ls=":",label=f"$j_{{lim}}\\approx${jlim:.0f}")
axB.set_xlabel("current density  j  (A/m$^2$)"); axB.set_ylabel(r"detachment diameter $D_d$ (mm)")
axB.set_title("(a) $D_d$ 90% band widens toward $j_{lim}$"); axB.legend(fontsize=8); axB.grid(alpha=0.3)
axR.plot(js,rel,"C3-o",ms=5)
axR.set_xlabel("current density  j  (A/m$^2$)"); axR.set_ylabel(r"$D_d$ relative 90% spread (%)")
axR.set_title("(b) relative uncertainty GROWS toward $j_{lim}$\n(polarization pushes $m_{surf}$ up the non-linear $\\sigma(m)$)")
axR.grid(alpha=0.3)
ax2=axR.twinx(); ax2.plot(js,msur,"k--",alpha=0.5); ax2.set_ylabel(r"median $m_{surf}$ (mol/kg)")
fig.suptitle("KOH, m=6: detachment-diameter uncertainty vs current density "
             "(MC: $\\sigma$ $\\pm$5%, $\\delta$ lognormal 40%; no applied voltage)",y=1.02,fontsize=11)
fig.tight_layout()
fig.savefig("fig/validation/uncertainty_vs_current.png",dpi=150,bbox_inches="tight",facecolor="white")
plt.close(fig)
print("Wrote fig/validation/uncertainty_vs_current.png")
json.dump(dict(jlim=float(jlim),trend=rows),
          open("/tmp/claude-1000/-home-endres-projects-ddgclib/40088a3e-090d-49c0-a0c1-8d711a972cc8/scratchpad/trend.json","w"),indent=2)
