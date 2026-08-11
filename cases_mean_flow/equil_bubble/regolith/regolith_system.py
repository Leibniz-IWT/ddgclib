"""
regolith_system.py
==================

The Stage-1 Mars molten-regolith-electrolysis system bundle: melt properties,
transport/efficiency laws, the melt Nernst layer, and the O2 rate chain, wired
to the EXISTING detachment solver (../pulloff_models.solve_detachment) and
rate chain (../bubble_kinetics) with zero changes to aqueous code paths
(INTEGRATION_PLAN.md).

Every number carries provenance; see data/*.csv and the topic docs.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import partial

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASE = os.path.dirname(_HERE)
for p in (_HERE, _CASE):
    if p not in sys.path:
        sys.path.insert(0, p)

import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import (EnvState, Buoyancy, CapillaryPinning,
                            solve_detachment, G_EARTH, G_MARS)
from bubble_kinetics import (ReactionRate, NernstDiffusionLayer, BubbleGrowth,
                             detachment_rate, FARADAY, R_GAS)

from melt_nrtl import OXIDES, IDX, M_OXIDE, MeltNRTL, interim_melt, stage1_composition
from melt_butler_sigma import MeltHandle, MeltButlerSigma, XinLinearSigma

# ---------------------------------------------------------------------------
# Operating point (COMPOSITION.md sec.5, GRAVITY_CONTEXT.md sec.5)
# ---------------------------------------------------------------------------
T_OP = 1850.0            # K   (Guerrero-Gonzalez & Zabel 2023 system optimum)
P_HEAD = 50e3            # Pa  (adopted; low end of MOXIE-demonstrated 53-101 kPa)
P_SWEEP = (0.7e3, 2.8e3, 50e3, 101.325e3)   # Pa, sensitivity sweep
M_O2 = 31.998e-3         # kg/mol
THETA_GAS_DEG = 25.0     # deg, O2 on Ir-class anode (sweep 10-40; SURFACE_PROPERTIES.md)
J_NOMINAL = 0.5e4        # A/m^2 = 0.5 A/cm^2 (MIT operating datum)


def rho_o2(T: float = T_OP, P: float = P_HEAD) -> float:
    """Ideal-gas O2 density [kg/m^3] (verified: 0.107 at 1873 K, 52 kPa)."""
    return P * M_O2 / (R_GAS * T)


def rho_melt(x: np.ndarray, T: float = T_OP) -> float:
    """
    Melt density [kg/m^3]: Keene quick model rho_1673 = 2490 + 12*wt%(FeO+
    Fe2O3+MnO), +-5%, with d(rho)/dT = -0.01%/K (SURFACE_PROPERTIES.md sec.3).
    Sanity: baseline melt -> ~2760 at 1850 K (band 2700-2900).
    """
    w = x * np.array([M_OXIDE[ox] for ox in OXIDES])
    wt_feo = 100.0 * w[IDX["FeO"]] / w.sum()
    rho_1673 = 2490.0 + 12.0 * wt_feo
    return rho_1673 * (1.0 - 1e-4 * (T - 1673.0))


# ---------------------------------------------------------------------------
# Transport & efficiency laws (KINETICS.md sec.2; all 1425 C single-lab,
# +-40% inter-lab band; Arrhenius extrapolation to T_OP is OURS)
# ---------------------------------------------------------------------------
def kappa_khetpal(x: np.ndarray) -> float:
    """kappa [S/m] = 100*(0.27104 - 0.192 X_SiO2 + 2.326 X_FeO + 0.879 X_MgO
    + 1.072 X_CaO) [S/cm at 1425 C]. Al2O3 absent from the regression."""
    s_cm = (0.27104 - 0.192 * x[IDX["SiO2"]] + 2.326 * x[IDX["FeO"]]
            + 0.879 * x[IDX["MgO"]] + 1.072 * x[IDX["CaO"]])
    return max(s_cm, 1e-3) * 100.0


def kappa_haskin(x: np.ndarray) -> float:
    """ln kappa[S/cm] = 5.738 - 12.6 X_SiO2 - 10 X_AlO1.5 - 3.7 X_TiO2
    + 1.89 X_FeO + 0.07 X_MgO - 1.25 X_CaO (1425 C). AlO1.5 basis: X_AlO1.5
    computed by splitting Al2O3 into 2 AlO1.5 and renormalising."""
    n = x.copy()
    n_al15 = 2.0 * n[IDX["Al2O3"]]
    tot = n.sum() + n_al15 - n[IDX["Al2O3"]]
    xs = {ox: n[IDX[ox]] / tot for ox in OXIDES}
    x_al = n_al15 / tot
    ln_s = (5.738 - 12.6 * xs["SiO2"] - 10.0 * x_al
            + 1.89 * xs["FeO"] + 0.07 * xs["MgO"] - 1.25 * xs["CaO"])
    return float(np.exp(ln_s)) * 100.0     # S/m


def t_electronic(x: np.ndarray) -> float:
    """t_e = 1.99 X_FeO (fitted X_FeO 0.08-0.16; beyond = extrapolation)."""
    return float(np.clip(1.99 * x[IDX["FeO"]], 0.0, 0.95))


def current_efficiency(x: np.ndarray, floor: float = 0.30) -> float:
    """
    CE = 1 - t_e(X_FeO), floored at Schreiner's Fe-bearing-melt band low end
    (faradaic efficiency 30-60% for Fe-bearing melts; ~95% once Fe-free).
    """
    return float(np.clip(1.0 - t_electronic(x), floor, 0.98))


# ---------------------------------------------------------------------------
# Melt Nernst layer: the ONLY aqueous-convention override needed
# ---------------------------------------------------------------------------
@dataclass
class MeltNernstLayer(NernstDiffusionLayer):
    """
    Composition polarisation in MOLE FRACTION.  Physical picture: near the
    anode, network-modifier cations (Fe2+, Ca2+, Mg2+) migrate away and O2-
    is consumed, so the local melt becomes SiO2-enriched/acidic; we proxy this
    as local FeO depletion along the Stage-1 trajectory (feeds sigma(x) and
    basicity).  Converts x <-> c via c = x * rho / M_bar.

    Defaults: D_Fe2+ = 1.2e-10 m^2/s at 1698 K (Haskin 1992), Arrhenius-scaled
    to T with Ea = 335 kJ/mol (measured for O2 diffusion, Semkow & Haskin 1985
    -- reusing it for Fe2+ is OUR first-estimate extrapolation, flagged in
    KINETICS.md); delta = 3e-4 m (bubble-stirred melt; broad prior 1e-4..1e-3);
    n = 2 (Fe2+ + 2e), sign = -1 (depleted).
    """
    delta: float = 3e-4
    D: float = 1.2e-10           # at T_D_REF
    T_D_REF: float = 1698.0      # K (1425 C)
    EA_D: float = 335e3          # J/mol
    n: int = 2
    sign: int = -1
    x_ref: np.ndarray = field(default_factory=lambda: stage1_composition(0.0))
    T: float = T_OP

    def D_at_T(self, T: float | None = None) -> float:
        T = self.T if T is None else T
        from bubble_kinetics import R_GAS as _R
        return self.D * np.exp(-self.EA_D / _R * (1.0 / T - 1.0 / self.T_D_REF))

    def _M_bar(self) -> float:
        return float(sum(self.x_ref[IDX[ox]] * M_OXIDE[ox] for ox in OXIDES))

    def surface_x_feo(self, x_feo_bulk: float, j: float) -> float:
        rho = rho_melt(self.x_ref, self.T)
        M_bar = self._M_bar()
        c_bulk = x_feo_bulk * rho / M_bar                  # mol/m^3
        dc = self.sign * j * self.delta / (self.n * FARADAY * self.D_at_T())
        c_surf = max(c_bulk + dc, 0.0)
        return float(np.clip(c_surf * M_bar / rho, 0.0, 1.0))

    def limiting_current_feo(self, x_feo_bulk: float) -> float:
        rho = rho_melt(self.x_ref, self.T)
        c_bulk = x_feo_bulk * rho / self._M_bar()
        return self.n * FARADAY * self.D_at_T() * c_bulk / self.delta


# ---------------------------------------------------------------------------
# The system bundle
# ---------------------------------------------------------------------------
@dataclass
class RegolithMeltSystem:
    """Stage-1 FeO/O2 molten regolith electrolysis on an Ir-class anode."""
    name: str = "Mars regolith melt, Stage-1 FeO/O2 (OER on Ir)"
    T: float = T_OP
    P: float = P_HEAD                    # headspace ~ bubble pressure (v1)
    gamma_feo_anchor: float = 1.70       # Holzheid; envelope [1.2, 2.1]
    sigma_model: object = None           # default MeltButlerSigma()
    theta_gas_deg: float = THETA_GAS_DEG
    n_electrons: int = 4                 # O2
    site_density: float = 1e4            # 1/m^2 -- slit/site density, broad prior
    capture_eff: float = 1.0
    ce_floor: float = 0.30

    def __post_init__(self):
        if self.sigma_model is None:
            self.sigma_model = MeltButlerSigma()
        self.handle = MeltHandle(melt=interim_melt(self.gamma_feo_anchor),
                                 T=self.T)

    # -- thermo/props at conversion xi ---------------------------------------
    def composition(self, xi: float) -> np.ndarray:
        return stage1_composition(xi)

    def sigma(self, xi: float) -> float:
        return self.sigma_model.sigma_lv(xi, self.handle, self.T)

    def rho_l(self, xi: float) -> float:
        return rho_melt(self.composition(xi), self.T)

    def rho_g(self) -> float:
        return rho_o2(self.T, self.P)

    def env(self, xi: float, g: float = G_MARS, j: float = 0.0) -> EnvState:
        x = self.composition(xi)
        return EnvState(g=g, rho_l=self.rho_l(xi), rho_g=self.rho_g(),
                        j=j, kappa_e=kappa_khetpal(x), eps_r=1.0,
                        E_cell=None)     # field channels OFF for melt v1

    # -- detachment (Layer E, existing solver) -------------------------------
    def detach(self, xi: float, g: float = G_MARS,
               theta_deg: float | None = None, sigma: float | None = None) -> dict:
        th = np.deg2rad(theta_deg if theta_deg is not None else self.theta_gas_deg)
        sig = self.sigma(xi) if sigma is None else sigma
        env = self.env(xi, g=g)
        return solve_detachment(
            sig, th, models=[Buoyancy(), CapillaryPinning()], env=env,
            shape_fn=toy.young_laplace_shape, criterion="max_volume")

    # -- rate chain (Layer F, existing pieces at melt T, P) -------------------
    def rate(self, V_d: float, xi: float, j: float = J_NOMINAL,
             sigma_eff: float = np.nan):
        x = self.composition(xi)
        ce = current_efficiency(x, floor=self.ce_floor)
        reaction = ReactionRate(n_electrons=self.n_electrons)
        growth = BubbleGrowth(site_density=self.site_density,
                              capture_eff=self.capture_eff * ce,
                              T=self.T, P=self.P)
        return detachment_rate(V_d, sigma_eff, np.deg2rad(self.theta_gas_deg),
                               xi, j, reaction, growth)


if __name__ == "__main__":
    sysm = RegolithMeltSystem()
    x0 = sysm.composition(0.0)
    print(f"{sysm.name}\nT = {sysm.T} K, P = {sysm.P/1e3:.1f} kPa")
    print(f"rho_melt(xi=0) = {sysm.rho_l(0.0):.0f} kg/m3 (band 2700-2900)")
    print(f"rho_O2 = {sysm.rho_g():.3f} kg/m3")
    print(f"kappa Khetpal/Haskin (xi=0): {kappa_khetpal(x0):.1f} / "
          f"{kappa_haskin(x0):.1f} S/m (inter-lab ~40%)")
    print(f"t_e(xi=0) = {t_electronic(x0):.2f} -> CE = {current_efficiency(x0):.2f}")
    for g, tag in ((G_EARTH, "Earth"), (G_MARS, "Mars ")):
        r = sysm.detach(0.0, g=g)
        lc = np.sqrt(sysm.sigma(0.0) / ((sysm.rho_l(0.0) - sysm.rho_g()) * g))
        print(f"{tag}: sigma = {r['sigma_eff']*1e3:.1f} mN/m, l_c = {lc*1e3:.2f} mm, "
              f"D_d = {r['D']*1e3:.3f} mm, V_d = {r['V']*1e9:.3f} mm^3, "
              f"D/l_c = {r['D']/lc:.4f}")
    rE = sysm.detach(0.0, g=G_EARTH); rM = sysm.detach(0.0, g=G_MARS)
    print(f"Mars/Earth D_d ratio = {rM['D']/rE['D']:.4f} (exact sqrt(gE/gM) = "
          f"{np.sqrt(G_EARTH/G_MARS):.4f})")
    res = sysm.rate(rM["V"], 0.0, j=J_NOMINAL, sigma_eff=sysm.sigma(0.0))
    print(f"Rate chain (Mars, j=0.5 A/cm2, site density {sysm.site_density:.0e}/m2): "
          f"Q = {res.Q_bubble:.3e} m3/s, t_d = {res.t_detach:.2f} s, "
          f"f = {res.f_detach:.3f} Hz")
