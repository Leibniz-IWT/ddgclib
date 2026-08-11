"""Diagnostic: why does D_d(dE) flatten at high voltage in the electrowetting sweep?

Traces the causal chain dE -> eta -> theta(dE) -> D_d(dE) and marks the dE at
which theta hits the saturation floor theta_min = 5 deg.
"""
import numpy as np
import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import (EnvState, Buoyancy, CapillaryPinning, BulkDEP,
                            Electrowetting)
from detachment_fitted import fitted_system
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ew = Electrowetting()
theta_min = ew.theta_min
print(f"theta_min (saturation floor) = {np.rad2deg(theta_min):.2f} deg\n")

koh, _, _ = fitted_system("KOH", os.path.join(_HERE, "enrtl_fit", "data", "koh_hamer_wu.csv"))
h2so4, _, _ = fitted_system("H2SO4", os.path.join(_HERE, "enrtl_fit", "data", "h2so4_que2011_fig10.csv"))

m_fix = 6.0
dE = np.linspace(0.0, 0.45, 40)

for name, sysf in [("KOH", koh), ("H2SO4", h2so4)]:
    p = toy.ELECTROCHEM_PARAMS.get(name, toy.ELECTROCHEM_PARAMS["KOH"])
    sig = sysf.sigma_lv(m_fix)
    theta0 = sysf.theta_0
    # analytic saturation onset: eta_sat = cos(theta_min) - cos(theta0)
    eta_sat = np.cos(theta_min) - np.cos(theta0)
    dE_sat = np.sqrt(eta_sat * sig / (0.5 * p["C_dl"]))
    print(f"=== {name}: sigma(m={m_fix})={sig*1e3:.2f} mN/m, theta0={np.rad2deg(theta0):.1f} deg, "
          f"C_dl={p['C_dl']} ===")
    print(f"  eta needed to reach theta_min: {eta_sat:.4f}")
    print(f"  --> analytic saturation onset dE_sat = {dE_sat:.3f} V (sweep ends at {dE[-1]:.2f} V)")
    print(f"  {'dE':>6} {'eta':>8} {'theta(deg)':>11} {'clamped?':>9} {'D_d(mm)':>9}")
    prev_D = None
    for d in dE[::4]:
        eta = 0.5 * p["C_dl"] * d**2 / sig
        cos_th = np.clip(np.cos(theta0) + eta, -1.0, 1.0)
        th_raw = np.arccos(cos_th)
        th = max(th_raw, theta_min)
        clamped = "YES" if th_raw < theta_min else ""
        env = EnvState(rho_l=sysf.rho_l, C_dl=p["C_dl"], E_pzc=p["E_pzc"],
                       E_cell=p["E_pzc"] - d)
        r = toy.detachment_volume(sig, theta0, models=[
            Buoyancy(), CapillaryPinning(), BulkDEP(), Electrowetting()], env=env)
        D = r["D"] * 1e3 if r["converged"] else np.nan
        dD = "" if prev_D is None else f"(d={D-prev_D:+.3f})"
        print(f"  {d:6.3f} {eta:8.4f} {np.rad2deg(th):11.2f} {clamped:>9} {D:9.3f} {dD}")
        prev_D = D
    print()
