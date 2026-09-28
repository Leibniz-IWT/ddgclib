"""
systems.py
=========

``ElectrochemicalSystem`` -- a single object bundling everything a
gas-evolving-electrode case needs, so the pipeline generalises from aqueous
water-electrolysis salts (KOH, H2SO4, ...) toward, eventually, multi-oxide
Mars-regolith electrorefining.

A System ties together:
    * the salt / mixture-thermodynamics handle (e-NRTL ``Salt``)
    * the swappable liquid-vapour surface-tension model (Layer B)
    * electrode electrochemistry (C_dl, E_pzc, kappa_e) for electrowetting/DEP
    * transport (diffusivity D, Nernst layer thickness delta) for polarization
    * the gas reaction (electrons n, gas molar mass) for the rate chain
    * base wetting angle theta_0 and liquid density rho_l

The registry ``SYSTEMS`` holds ready-made instances; ``regolith_stub`` is a
clearly-flagged placeholder for the future molten/multicomponent case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import (
    SurfaceTensionModel, ConstantSigma, CallableSigma, EnvState,
    Buoyancy, CapillaryPinning, BulkDEP, Electrowetting,
)
from bubble_kinetics import ReactionRate, NernstDiffusionLayer, BubbleGrowth


@dataclass
class ElectrochemicalSystem:
    """One gas-evolving-electrode system; see module docstring."""
    name: str
    salt: object                       # toy.Salt (e-NRTL handle) or None
    # Layer B -- liquid-vapour surface tension
    sigma_model: SurfaceTensionModel
    # electrode electrochemistry
    C_dl: float = 0.20                 # F/m^2
    E_pzc: float = -0.10               # V vs RHE
    kappa_e: float = 21.5              # S/m
    eps_r: float = 78.4
    # transport (concentration polarization)
    D: float = 1e-9                    # m^2/s
    delta: float = 1e-4                # m (Nernst layer)
    polar_sign: int = +1               # +1 enrich / -1 deplete near electrode
    polar_n: int = 1                   # electrons per SALT formula event (coupling j)
    M_salt: float = 0.05846            # solute molar mass [kg/mol] (Nernst conversion)
    # gas reaction
    n_electrons: int = 2               # H2: 2, O2: 4
    gas_name: str = "H2"
    # geometry / fluid
    theta_0_deg: float = 30.0
    rho_l: float = toy.RHO_W
    rho_g: float = toy.RHO_H2
    # growth / rate chain
    site_density: float = 1e6          # 1/m^2
    capture_eff: float = 1.0

    # ---- convenience builders ----------------------------------------------
    @property
    def theta_0(self) -> float:
        return np.deg2rad(self.theta_0_deg)

    def sigma_lv(self, m: float, T: float = toy.T_REF) -> float:
        return self.sigma_model.sigma_lv(m, self.salt, T)

    def env(self, j: float = 0.0, E_cell: Optional[float] = None,
            g: float = toy.G_GRAV) -> EnvState:
        return EnvState(g=g, rho_l=self.rho_l, rho_g=self.rho_g,
                        j=j, kappa_e=self.kappa_e, eps_r=self.eps_r,
                        C_dl=self.C_dl, E_pzc=self.E_pzc, E_cell=E_cell)

    def pulloff_models(self, electrowetting: bool = True):
        """FORCE budget + optional electrowetting (THETA) coupling."""
        mods = [Buoyancy(), CapillaryPinning(), BulkDEP()]
        if electrowetting:
            mods.append(Electrowetting())
        return mods

    def reaction(self) -> ReactionRate:
        return ReactionRate(n_electrons=self.n_electrons)

    def nernst(self) -> NernstDiffusionLayer:
        return NernstDiffusionLayer(delta=self.delta, D=self.D,
                                    n=self.polar_n, sign=self.polar_sign,
                                    M_solute=self.M_salt)

    def growth(self) -> BubbleGrowth:
        return BubbleGrowth(site_density=self.site_density,
                            capture_eff=self.capture_eff)


# ---------------------------------------------------------------------------
# Ready-made systems
# ---------------------------------------------------------------------------
_ENRTL = CallableSigma(toy.butler_surface_tension, name="enrtl_butler")

KOH_WATER = ElectrochemicalSystem(
    name="KOH/H2O (HER)", salt=toy.KOH, sigma_model=_ENRTL,
    C_dl=0.20, E_pzc=-0.10, kappa_e=21.5, M_salt=0.05611,   # KOH 56.11 g/mol
    D=5.3e-9, delta=1e-4, polar_sign=+1, polar_n=1,   # OH- enriched, 1 e- per KOH
    n_electrons=2, gas_name="H2", theta_0_deg=30.0,
    site_density=1e6, capture_eff=1.0,
)

H2SO4_WATER = ElectrochemicalSystem(
    name="H2SO4/H2O (HER)", salt=toy.H2SO4, sigma_model=_ENRTL,
    C_dl=0.20, E_pzc=+0.26, kappa_e=40.0, M_salt=0.09808,   # H2SO4 98.08 g/mol
    D=9.3e-9, delta=1e-4, polar_sign=-1, polar_n=2,   # 2 H+ per H2SO4 depleted
    n_electrons=2, gas_name="H2", theta_0_deg=30.0,
    site_density=1e6, capture_eff=1.0,
)

# Literature-assumption twin: identical system but with a CONSTANT sigma_lv,
# so a case can plot "e-NRTL vs fixed-sigma" for the same operating conditions.
KOH_WATER_CONSTSIGMA = ElectrochemicalSystem(
    name="KOH/H2O (fixed sigma=72 mN/m)", salt=toy.KOH,
    sigma_model=ConstantSigma(0.072),
    C_dl=0.20, E_pzc=-0.10, kappa_e=21.5, D=5.3e-9, delta=1e-4,
    polar_sign=+1, polar_n=1, n_electrons=2, gas_name="H2", theta_0_deg=30.0,
)


def regolith_stub() -> ElectrochemicalSystem:
    """
    STUB -- molten/aqueous multi-oxide Mars-regolith electrorefining.

    The forward-looking extreme case for the hypothesis: mixture non-ideality is
    far stronger and more non-linear than for a single aqueous salt, so
    neglecting composition-dependent surface tension is expected to bias
    detachment size / gas rate much more.  Requires (FUTURE WORK):
      * a molten-oxide activity model (e.g. MQM / associate-species / e-NRTL
        generalisation) in place of the aqueous e-NRTL Salt handle;
      * high-T surface-tension data and a Butler analogue;
      * O2/metal co-evolution stoichiometry and transport at ~1000+ K.
    Returned here only as a placeholder with flagged, non-physical numbers.
    """
    return ElectrochemicalSystem(
        name="Regolith/molten-oxide (STUB)", salt=None,
        sigma_model=ConstantSigma(0.40),   # placeholder high-T oxide sigma
        C_dl=0.30, E_pzc=0.0, kappa_e=100.0, D=1e-9, delta=1e-4,
        polar_sign=+1, polar_n=2, n_electrons=4, gas_name="O2",
        theta_0_deg=45.0, rho_l=2700.0, rho_g=1.3,
    )


SYSTEMS = {
    "KOH": KOH_WATER,
    "H2SO4": H2SO4_WATER,
    "KOH_const": KOH_WATER_CONSTSIGMA,
}
